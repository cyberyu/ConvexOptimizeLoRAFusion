#!/usr/bin/env python3
"""
Reorganize StableCode matrices by (layer, module) across all checkpoints
Each unique (layer, module) combination gets an index with results from all available checkpoints

This script processes the enhanced extraction output with A, B, and AB matrices
and organizes them by layer-module combinations for optimization workflows.
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
    
    # Check for enhanced extraction format first
    ab_dir = os.path.join(input_dir, "ab_products")
    a_dir = os.path.join(input_dir, "a_matrices")
    b_dir = os.path.join(input_dir, "b_matrices")
    
    enhanced_format = all(os.path.exists(d) for d in [ab_dir, a_dir, b_dir])
    
    if enhanced_format:
        print("📊 Enhanced extraction format detected")
        
        # Find all layer matrix files in ab_products (we'll use this as reference)
        layer_pattern = os.path.join(ab_dir, "layer_*_ab_matrices.safetensors")
        layer_files = glob.glob(layer_pattern)
        
        if not layer_files:
            # Fallback to legacy format
            print("⚠️  No enhanced format files found, checking legacy format...")
            enhanced_format = False
        else:
            print(f"📦 Found {len(layer_files)} enhanced layer files")
            
            # Load first file to discover checkpoint names
            sample_file = layer_files[0]
            checkpoint_names = []
            
            with safe_open(sample_file, framework="pt", device="cpu") as f:
                checkpoint_names = list(f.keys())
            
            print(f"🏷️  Discovered checkpoint names: {checkpoint_names}")
            
            return layer_files, checkpoint_names, enhanced_format
    
    if not enhanced_format:
        print("📦 Legacy extraction format detected")
        
        # Find all layer matrix files (legacy format)
        layer_pattern = os.path.join(input_dir, "layer_*_matrices.safetensors")
        layer_files = glob.glob(layer_pattern)
        
        if not layer_files:
            raise FileNotFoundError(f"No layer_*_matrices.safetensors files found in {input_dir}")
        
        print(f"📦 Found {len(layer_files)} legacy layer files")
        
        # Load first file to discover checkpoint names
        sample_file = layer_files[0]
        checkpoint_names = []
        
        with safe_open(sample_file, framework="pt", device="cpu") as f:
            checkpoint_names = list(f.keys())
        
        print(f"🏷️  Discovered checkpoint names: {checkpoint_names}")
        
        return layer_files, checkpoint_names, enhanced_format

def load_all_layer_matrices_enhanced(input_dir, checkpoint_names):
    """Load all matrices from enhanced extraction format (A, B, AB)"""
    
    print(f"📂 Loading matrices from enhanced extraction format...")
    
    # Structure: checkpoint_name -> layer_module -> {'A': matrix, 'B': matrix, 'AB': matrix}
    all_matrices = defaultdict(lambda: defaultdict(dict))
    
    # Load from each directory type
    matrix_types = {
        'AB': 'ab_products',
        'A': 'a_matrices', 
        'B': 'b_matrices'
    }
    
    for matrix_type, dir_name in matrix_types.items():
        type_dir = os.path.join(input_dir, dir_name)
        if not os.path.exists(type_dir):
            print(f"⚠️  Directory not found: {type_dir}")
            continue
            
        # Find layer files for this matrix type
        if matrix_type == 'AB':
            pattern = os.path.join(type_dir, "layer_*_ab_matrices.safetensors")
        elif matrix_type == 'A':
            pattern = os.path.join(type_dir, "layer_*_a_matrices.safetensors")
        elif matrix_type == 'B':
            pattern = os.path.join(type_dir, "layer_*_b_matrices.safetensors")
        
        layer_files = glob.glob(pattern)
        print(f"  📁 {matrix_type} matrices: {len(layer_files)} files")
        
        for i, layer_file in enumerate(layer_files):
            # Extract layer module name from filename
            filename = os.path.basename(layer_file)
            if matrix_type == 'AB':
                layer_module = filename.replace("_ab_matrices.safetensors", "")
            elif matrix_type == 'A':
                layer_module = filename.replace("_a_matrices.safetensors", "")
            elif matrix_type == 'B':
                layer_module = filename.replace("_b_matrices.safetensors", "")
            
            # Load matrices for this layer from all checkpoints
            with safe_open(layer_file, framework="pt", device="cpu") as f:
                for checkpoint_name in checkpoint_names:
                    if checkpoint_name in f.keys():
                        matrix = f.get_tensor(checkpoint_name)
                        all_matrices[checkpoint_name][layer_module][matrix_type] = matrix
            
            if (i + 1) % 20 == 0:
                print(f"    Loaded {i + 1}/{len(layer_files)} {matrix_type} files...")
    
    print(f"✅ Loaded enhanced matrices for {len(all_matrices)} checkpoints")
    
    return all_matrices

def load_all_layer_matrices_legacy(layer_files, checkpoint_names):
    """Load all matrices from legacy extraction format (AB only)"""
    
    print(f"📂 Loading matrices from legacy extraction format...")
    
    # Structure: checkpoint_name -> layer_module -> {'AB': matrix}
    all_matrices = defaultdict(lambda: defaultdict(dict))
    
    for i, layer_file in enumerate(layer_files):
        # Extract layer module name from filename
        # Example: layer_00_attn_q_proj_matrices.safetensors -> layer_00_attn_q_proj
        filename = os.path.basename(layer_file)
        layer_module = filename.replace("_matrices.safetensors", "")
        
        # Load matrices for this layer from all checkpoints
        with safe_open(layer_file, framework="pt", device="cpu") as f:
            for checkpoint_name in checkpoint_names:
                if checkpoint_name in f.keys():
                    matrix = f.get_tensor(checkpoint_name)
                    all_matrices[checkpoint_name][layer_module]['AB'] = matrix
        
        if (i + 1) % 20 == 0:
            print(f"  Loaded {i + 1}/{len(layer_files)} layer files...")
    
    print(f"✅ Loaded legacy matrices for {len(all_matrices)} checkpoints")
    
    return all_matrices

def reorganize_by_layer_module(all_matrices, output_dir, enhanced_format=False):
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
    
    # Create subdirectories for different matrix types if enhanced format
    if enhanced_format:
        ab_out_dir = os.path.join(output_dir, "ab_products")
        a_out_dir = os.path.join(output_dir, "a_matrices")
        b_out_dir = os.path.join(output_dir, "b_matrices")
        combined_out_dir = os.path.join(output_dir, "combined_abc")
        
        os.makedirs(ab_out_dir, exist_ok=True)
        os.makedirs(a_out_dir, exist_ok=True)
        os.makedirs(b_out_dir, exist_ok=True)
        os.makedirs(combined_out_dir, exist_ok=True)
    
    # Save each (layer, module) combination
    saved_count = 0
    
    for layer_module in all_layer_modules:
        index = index_mapping[layer_module]
        
        if enhanced_format:
            # Collect matrices by type
            ab_matrices = {}
            a_matrices = {}
            b_matrices = {}
            combined_matrices = {}
            
            for checkpoint_name, checkpoint_matrices in all_matrices.items():
                if layer_module in checkpoint_matrices:
                    layer_data = checkpoint_matrices[layer_module]
                    
                    if 'AB' in layer_data:
                        ab_matrices[checkpoint_name] = layer_data['AB']
                        combined_matrices[f"{checkpoint_name}_AB"] = layer_data['AB']
                    if 'A' in layer_data:
                        a_matrices[checkpoint_name] = layer_data['A']
                        combined_matrices[f"{checkpoint_name}_A"] = layer_data['A']
                    if 'B' in layer_data:
                        b_matrices[checkpoint_name] = layer_data['B']
                        combined_matrices[f"{checkpoint_name}_B"] = layer_data['B']
            
            # Save different formats
            base_filename = f"index_{index:03d}_{layer_module}"
            
            if ab_matrices:
                ab_filepath = os.path.join(ab_out_dir, f"{base_filename}_ab_matrices.safetensors")
                safetensors.torch.save_file(ab_matrices, ab_filepath)
            
            if a_matrices:
                a_filepath = os.path.join(a_out_dir, f"{base_filename}_a_matrices.safetensors")
                safetensors.torch.save_file(a_matrices, a_filepath)
            
            if b_matrices:
                b_filepath = os.path.join(b_out_dir, f"{base_filename}_b_matrices.safetensors")
                safetensors.torch.save_file(b_matrices, b_filepath)
            
            if combined_matrices:
                combined_filepath = os.path.join(combined_out_dir, f"{base_filename}_all_matrices.safetensors")
                safetensors.torch.save_file(combined_matrices, combined_filepath)
            
            # Legacy format for backward compatibility
            if ab_matrices:
                legacy_filepath = os.path.join(output_dir, f"{base_filename}_matrices.safetensors")
                safetensors.torch.save_file(ab_matrices, legacy_filepath)
        
        else:
            # Legacy format: only AB matrices
            layer_module_matrices = {}
            
            for checkpoint_name, checkpoint_matrices in all_matrices.items():
                if layer_module in checkpoint_matrices and 'AB' in checkpoint_matrices[layer_module]:
                    layer_module_matrices[checkpoint_name] = checkpoint_matrices[layer_module]['AB']
            
            if layer_module_matrices:
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
        'enhanced_format': enhanced_format,
        'matrix_types': ['A', 'B', 'AB'] if enhanced_format else ['AB'],
        'description': 'StableCode matrices organized by (layer, module) with indexed access'
    }
    
    if enhanced_format:
        master_index['directory_structure'] = {
            'ab_products': 'AB matrices for L1/L2 optimization',
            'a_matrices': 'A matrices for SVD optimization',
            'b_matrices': 'B matrices for SVD optimization',
            'combined_abc': 'All matrices in single files',
            'legacy': 'AB matrices in main directory (backward compatible)'
        }
    
    master_index_path = os.path.join(output_dir, "master_index.json")
    with open(master_index_path, 'w') as f:
        json.dump(master_index, f, indent=2)
    
    print(f"📋 Master index saved to: {master_index_path}")
    
    return master_index

def create_summary_statistics(all_matrices, output_dir, enhanced_format=False):
    """Create summary statistics about the matrices"""
    
    print(f"📊 Creating summary statistics...")
    
    stats = {
        'total_checkpoints': len(all_matrices),
        'checkpoint_names': list(all_matrices.keys()),
        'enhanced_format': enhanced_format,
        'layer_module_stats': {},
        'matrix_shape_analysis': {},
        'memory_usage': {},
        'matrix_type_counts': defaultdict(int)
    }
    
    # Analyze each checkpoint
    for checkpoint_name, checkpoint_matrices in all_matrices.items():
        total_params = 0
        total_memory_mb = 0
        shape_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for layer_module, layer_data in checkpoint_matrices.items():
            for matrix_type, matrix in layer_data.items():
                shape = tuple(matrix.shape)
                shape_str = str(shape)  # Convert tuple to string for JSON serialization
                params = matrix.numel()
                memory_mb = matrix.numel() * 4 / (1024 * 1024)  # Assuming float32
                
                total_params += params
                total_memory_mb += memory_mb
                shape_counts[shape_str] += 1  # Use string key instead of tuple
                type_counts[matrix_type] += 1
                stats['matrix_type_counts'][matrix_type] += 1
        
        stats['layer_module_stats'][checkpoint_name] = {
            'total_layer_modules': len(checkpoint_matrices),
            'total_parameters': total_params,
            'total_memory_mb': round(total_memory_mb, 2),
            'matrix_type_counts': dict(type_counts)
        }
        
        stats['matrix_shape_analysis'][checkpoint_name] = dict(shape_counts)
        stats['memory_usage'][checkpoint_name] = round(total_memory_mb, 2)
    
    # Overall statistics
    all_layer_modules = set()
    for checkpoint_matrices in all_matrices.values():
        all_layer_modules.update(checkpoint_matrices.keys())
    
    stats['overall'] = {
        'unique_layer_modules': len(all_layer_modules),
        'total_memory_all_checkpoints_mb': sum(stats['memory_usage'].values()),
        'matrix_type_distribution': dict(stats['matrix_type_counts'])
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
        description="Reorganize StableCode matrices by (layer, module) combinations"
    )
    parser.add_argument(
        "--input", 
        default="enhanced_extracted_stablecode_matrices",
        help="Input directory containing extracted matrices"
    )
    parser.add_argument(
        "--output", 
        default="enhanced_extracted_stablecode_matrices/organized_by_layer_module",
        help="Output directory for reorganized matrices"
    )
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    
    print("🚀 STABLECODE MATRIX REORGANIZATION")
    print("=" * 45)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        # Discover checkpoint files and format
        layer_files, checkpoint_names, enhanced_format = discover_checkpoint_files(input_dir)
        
        # Load all matrices based on format
        if enhanced_format:
            all_matrices = load_all_layer_matrices_enhanced(input_dir, checkpoint_names)
        else:
            all_matrices = load_all_layer_matrices_legacy(layer_files, checkpoint_names)
        
        # Reorganize by layer module
        master_index = reorganize_by_layer_module(all_matrices, output_dir, enhanced_format)
        
        # Create summary statistics
        stats = create_summary_statistics(all_matrices, output_dir, enhanced_format)
        
        print(f"\n✅ REORGANIZATION COMPLETE!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🔢 Total combinations: {master_index['total_combinations']}")
        print(f"📊 Enhanced format: {enhanced_format}")
        print(f"🧮 Matrix types: {master_index['matrix_types']}")
        print(f"💾 Total memory: {stats['overall']['total_memory_all_checkpoints_mb']:.1f} MB")
        
        if enhanced_format:
            print(f"📁 Directory structure:")
            for dir_name, description in master_index['directory_structure'].items():
                print(f"  📂 {dir_name}: {description}")
        
    except Exception as e:
        print(f"\n❌ Reorganization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())