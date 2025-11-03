#!/usr/bin/env python3
"""
Reorganize CodeGemma-2B matrices by (layer, module) across all checkpoints
Each unique (layer, module) combination gets an index with results from all available checkpoints
"""

import os
import torch
from safetensors import safe_open
import safetensors.torch
import json
import argparse
import glob
from collections import defaultdict

def discover_checkpoint_files(input_dir):
    """Automatically discover checkpoint files in the input directory"""
    
    print(f"🔍 Discovering checkpoint files in: {input_dir}")
    
    # Find all layer matrix files 
    layer_pattern = os.path.join(input_dir, "layer_*_matrices.safetensors")
    layer_files = glob.glob(layer_pattern)
    
    if not layer_files:
        raise FileNotFoundError(f"No layer_*_matrices.safetensors files found in {input_dir}")
    
    print(f"📦 Found {len(layer_files)} layer files")
    
    # Load first file to discover checkpoint names
    sample_file = layer_files[0]
    checkpoint_names = []
    
    with safe_open(sample_file, framework="pt", device="cpu") as f:
        checkpoint_names = list(f.keys())
    
    print(f"🏷️  Discovered checkpoint names: {checkpoint_names}")
    
    return layer_files, checkpoint_names

def load_all_layer_matrices(layer_files, checkpoint_names):
    """Load all matrices organized by checkpoint and layer"""
    
    print(f"📂 Loading matrices from {len(layer_files)} layer files...")
    
    # Structure: checkpoint_name -> layer_module -> matrix
    all_matrices = defaultdict(dict)
    
    for i, layer_file in enumerate(layer_files):
        # Extract layer module name from filename
        # Example: layer_00_self_attn_q_proj_matrices.safetensors -> layer_00_self_attn_q_proj
        filename = os.path.basename(layer_file)
        layer_module = filename.replace("_matrices.safetensors", "")
        
        # Load matrices for this layer from all checkpoints
        with safe_open(layer_file, framework="pt", device="cpu") as f:
            for checkpoint_name in checkpoint_names:
                if checkpoint_name in f.keys():
                    matrix = f.get_tensor(checkpoint_name)
                    all_matrices[checkpoint_name][layer_module] = matrix
        
        if (i + 1) % 20 == 0:
            print(f"  Loaded {i + 1}/{len(layer_files)} layer files...")
    
    print(f"✅ Loaded matrices for {len(all_matrices)} checkpoints")
    
    return all_matrices

def reorganize_by_layer_module(all_matrices, output_dir):
    """Reorganize matrices by (layer, module) combinations"""
    
    print(f"🔄 Reorganizing matrices by (layer, module)...")
    
    # Get all unique layer modules across all checkpoints
    all_layer_modules = set()
    for checkpoint_matrices in all_matrices.values():
        all_layer_modules.update(checkpoint_matrices.keys())
    
    all_layer_modules = sorted(list(all_layer_modules))
    print(f"📋 Found {len(all_layer_modules)} unique (layer, module) combinations")
    
    # Create mapping: layer_module -> index
    index_mapping = {}
    for i, layer_module in enumerate(all_layer_modules):
        index_mapping[layer_module] = i
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each (layer, module) combination
    saved_count = 0
    
    for layer_module in all_layer_modules:
        index = index_mapping[layer_module]
        
        # Collect matrices for this layer_module from all checkpoints
        layer_module_matrices = {}
        
        for checkpoint_name, checkpoint_matrices in all_matrices.items():
            if layer_module in checkpoint_matrices:
                layer_module_matrices[checkpoint_name] = checkpoint_matrices[layer_module]
        
        if layer_module_matrices:
            # Save to indexed file
            filename = f"index_{index:03d}_{layer_module}_matrices.safetensors"
            filepath = os.path.join(output_dir, filename)
            
            safetensors.torch.save_file(layer_module_matrices, filepath)
            saved_count += 1
            
            if saved_count % 20 == 0:
                print(f"  Saved {saved_count}/{len(all_layer_modules)} combinations...")
    
    print(f"✅ Saved {saved_count} (layer, module) combinations")
    
    # Create master index
    master_index = {
        'total_combinations': len(all_layer_modules),
        'index_mapping': index_mapping,
        'checkpoint_names': list(all_matrices.keys()),
        'organization_type': 'layer_module_indexed',
        'description': 'CodeGemma-2B matrices organized by (layer, module) with indexed access'
    }
    
    master_index_path = os.path.join(output_dir, "master_index.json")
    with open(master_index_path, 'w') as f:
        json.dump(master_index, f, indent=2)
    
    print(f"📋 Master index saved to: {master_index_path}")
    
    return master_index

def create_summary_statistics(all_matrices, output_dir):
    """Create summary statistics about the matrices"""
    
    print(f"📊 Creating summary statistics...")
    
    stats = {
        'total_checkpoints': len(all_matrices),
        'checkpoint_names': list(all_matrices.keys()),
        'layer_module_stats': {},
        'matrix_shape_analysis': {},
        'memory_usage': {}
    }
    
    # Analyze each checkpoint
    for checkpoint_name, checkpoint_matrices in all_matrices.items():
        total_params = 0
        total_memory_mb = 0
        shape_counts = defaultdict(int)
        
        for layer_module, matrix in checkpoint_matrices.items():
            shape = tuple(matrix.shape)
            params = matrix.numel()
            memory_mb = matrix.numel() * 4 / (1024 * 1024)  # Assuming float32
            
            total_params += params
            total_memory_mb += memory_mb
            shape_counts[shape] += 1
        
        stats['layer_module_stats'][checkpoint_name] = {
            'total_layer_modules': len(checkpoint_matrices),
            'total_parameters': total_params,
            'total_memory_mb': round(total_memory_mb, 2)
        }
        
        stats['matrix_shape_analysis'][checkpoint_name] = dict(shape_counts)
        stats['memory_usage'][checkpoint_name] = round(total_memory_mb, 2)
    
    # Overall statistics
    all_layer_modules = set()
    for checkpoint_matrices in all_matrices.values():
        all_layer_modules.update(checkpoint_matrices.keys())
    
    stats['overall'] = {
        'unique_layer_modules': len(all_layer_modules),
        'total_memory_all_checkpoints_mb': sum(stats['memory_usage'].values())
    }
    
    # Save statistics
    stats_path = os.path.join(output_dir, "organization_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📈 Statistics saved to: {stats_path}")
    
    return stats

def main():
    """Main function with command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Reorganize CodeGemma-2B matrices by (layer, module) combinations"
    )
    parser.add_argument(
        "--input", 
        default="extracted_codegemma2b_matrices",
        help="Input directory containing extracted matrices"
    )
    parser.add_argument(
        "--output", 
        default="extracted_codegemma2b_matrices/organized_by_layer_module",
        help="Output directory for reorganized matrices"
    )
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    
    print("🚀 CODEGEMMA-2B MATRIX REORGANIZATION")
    print("=" * 45)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        # Discover checkpoint files
        layer_files, checkpoint_names = discover_checkpoint_files(input_dir)
        
        # Load all matrices
        all_matrices = load_all_layer_matrices(layer_files, checkpoint_names)
        
        # Reorganize by layer module
        master_index = reorganize_by_layer_module(all_matrices, output_dir)
        
        # Create summary statistics
        stats = create_summary_statistics(all_matrices, output_dir)
        
        print(f"\n✅ REORGANIZATION COMPLETE!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🔢 Total combinations: {master_index['total_combinations']}")
        print(f"💾 Total memory: {stats['overall']['total_memory_all_checkpoints_mb']:.1f} MB")
        
    except Exception as e:
        print(f"\n❌ Reorganization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
