#!/usr/bin/env python3
"""
Reorganize StarCoder2-7B matrices by (layer, module) across all checkpoints
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
    
    # Find all AB_products.safetensors files
    ab_pattern = os.path.join(input_dir, "*_AB_products.safetensors")
    ab_files = glob.glob(ab_pattern)
    
    if not ab_files:
        raise FileNotFoundError(f"No *_AB_products.safetensors files found in {input_dir}")
    
    # Extract checkpoint names from filenames
    checkpoints = {}
    for ab_file in ab_files:
        filename = os.path.basename(ab_file)
        # Remove the _AB_products.safetensors suffix
        checkpoint_name = filename.replace("_AB_products.safetensors", "")
        # Remove the starcoder27b_ prefix if present
        if checkpoint_name.startswith("starcoder27b_"):
            checkpoint_name = checkpoint_name[len("starcoder27b_"):]
        
        checkpoints[checkpoint_name] = {
            'AB_file': ab_file,
            'base_name': filename.replace("_AB_products.safetensors", "")
        }
    
    print(f"📦 Found {len(checkpoints)} checkpoints:")
    for name, info in checkpoints.items():
        print(f"  - {name}: {os.path.basename(info['AB_file'])}")
    
    return checkpoints

def load_checkpoint_matrices(checkpoint_info):
    """Load AB products from a specific checkpoint"""
    
    AB_path = checkpoint_info['AB_file']
    matrices = {}
    
    with safe_open(AB_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            matrices[key] = f.get_tensor(key)
    
    return matrices

def organize_by_layer_module(input_dir, output_dir):
    """Organize matrices by (layer, module) across all checkpoints"""
    
    print("🔄 REORGANIZING MATRICES BY (LAYER, MODULE)")
    print("=" * 60)
    
    # Discover available checkpoints
    checkpoints = discover_checkpoint_files(input_dir)
    checkpoint_names = list(checkpoints.keys())
    
    # Load all checkpoint data
    all_checkpoint_data = {}
    for checkpoint_name, checkpoint_info in checkpoints.items():
        print(f"📦 Loading {checkpoint_name} matrices...")
        all_checkpoint_data[checkpoint_name] = load_checkpoint_matrices(checkpoint_info)
    
    # Organize by (layer, module)
    layer_module_data = defaultdict(dict)
    
    # Process first checkpoint to get structure
    first_checkpoint = checkpoint_names[0]
    for matrix_key in all_checkpoint_data[first_checkpoint].keys():
        # Parse key: model.layers.0.self_attn.q_proj
        parts = matrix_key.split('.')
        
        if 'layers' in parts:
            layer_idx = None
            module_type = None
            
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                elif part.endswith('_proj'):
                    module_type = part
                    break
            
            # Create layer_module_key if we found both layer and module
            if layer_idx is not None and module_type is not None:
                layer_module_key = f"layer_{layer_idx:02d}_{module_type}"
                
                # Collect matrices from all checkpoints for this (layer, module)
                for checkpoint_name in checkpoint_names:
                    if matrix_key in all_checkpoint_data[checkpoint_name]:
                        layer_module_data[layer_module_key][checkpoint_name] = all_checkpoint_data[checkpoint_name][matrix_key]
    
    print(f"🧩 Found {len(layer_module_data)} unique (layer, module) combinations")
    
    # Create indexed organization
    indexed_data = {}
    layer_module_index = {}
    
    for idx, (layer_module_key, checkpoint_matrices) in enumerate(sorted(layer_module_data.items())):
        indexed_data[idx] = {
            'layer_module': layer_module_key,
            'matrices': checkpoint_matrices,
            'checkpoints': list(checkpoint_matrices.keys())
        }
        layer_module_index[layer_module_key] = idx
    
    print(f"📊 Created indexed structure with {len(indexed_data)} entries")
    
    # Save organized data
    organized_dir = output_dir
    os.makedirs(organized_dir, exist_ok=True)
    
    # Save each (layer, module) combination as separate files
    print(f"\n💾 Saving organized matrices...")
    
    for idx, data in indexed_data.items():
        layer_module = data['layer_module']
        matrices = data['matrices']
        
        # Save matrices for this (layer, module) combination
        matrices_file = os.path.join(organized_dir, f"index_{idx:03d}_{layer_module}_matrices.safetensors")
        safetensors.torch.save_file(matrices, matrices_file)
        
        # Create metadata for this combination
        metadata = {
            'index': idx,
            'layer_module': layer_module,
            'checkpoints': data['checkpoints'],
            'matrix_shapes': {checkpoint: list(matrix.shape) for checkpoint, matrix in matrices.items()},
            'available_checkpoints': len(matrices)
        }
        
        metadata_file = os.path.join(organized_dir, f"index_{idx:03d}_{layer_module}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Create master index file
    master_index = {
        'total_combinations': len(indexed_data),
        'checkpoints': checkpoint_names,
        'index_mapping': layer_module_index,
        'structure_info': {
            'detected_layers': len(set(k.split('_')[1] for k in layer_module_index.keys())),
            'detected_modules': len(set(k.split('_')[2] for k in layer_module_index.keys())),
            'module_types': sorted(list(set(k.split('_')[2] for k in layer_module_index.keys())))
        }
    }
    
    master_index_file = os.path.join(organized_dir, "master_index.json")
    with open(master_index_file, 'w') as f:
        json.dump(master_index, f, indent=2)
    
    print(f"💾 Saved master index: {master_index_file}")
    
    # Print summary
    print(f"\n📋 ORGANIZATION SUMMARY")
    print("=" * 40)
    
    # Sample a few entries to show structure
    sample_indices = list(range(min(5, len(indexed_data))))
    
    for idx in sample_indices:
        data = indexed_data[idx]
        layer_module = data['layer_module']
        checkpoint_count = len(data['matrices'])
        
        print(f"Index {idx:3d}: {layer_module}")
        print(f"  📦 Checkpoints: {checkpoint_count}/{len(checkpoint_names)} {list(data['matrices'].keys())}")
        
        # Show matrix shapes
        sample_checkpoint = list(data['matrices'].keys())[0]
        sample_matrix = data['matrices'][sample_checkpoint]
        print(f"  📐 Matrix shape: {list(sample_matrix.shape)}")
        print()
    
    if len(indexed_data) > 5:
        print(f"... and {len(indexed_data) - 5} more combinations")
    
    print(f"\n✅ Organization complete!")
    print(f"  📁 Organized data saved to: {organized_dir}")
    print(f"  📊 Total combinations: {len(indexed_data)}")
    print(f"  🔢 Each combination has matrices from up to {len(checkpoint_names)} checkpoints")
    
    return organized_dir, indexed_data

def verify_organization(organized_dir, indexed_data, expected_checkpoints):
    """Verify the organization is correct"""
    
    print(f"\n🔍 VERIFYING ORGANIZATION")
    print("=" * 40)
    
    actual_combinations = len(indexed_data)
    print(f"Total combinations found: {actual_combinations}")
    
    # Verify each combination has expected checkpoints
    complete_combinations = 0
    for idx, data in indexed_data.items():
        if len(data['matrices']) == len(expected_checkpoints):
            complete_combinations += 1
    
    print(f"Complete combinations ({len(expected_checkpoints)} checkpoints): {complete_combinations}/{actual_combinations}")
    
    if complete_combinations == actual_combinations:
        print(f"✅ All combinations have all checkpoints!")
    else:
        missing_count = actual_combinations - complete_combinations
        print(f"⚠️  {missing_count} combinations missing some checkpoints")
    
    # Verify file structure
    expected_files = actual_combinations * 2 + 1  # 2 files per combination + master index
    actual_files = len([f for f in os.listdir(organized_dir) if os.path.isfile(os.path.join(organized_dir, f))])
    
    print(f"Expected files: {expected_files}")
    print(f"Actual files: {actual_files}")
    
    if actual_files >= expected_files:
        print(f"✅ All expected files created!")
    else:
        print(f"❌ Some files missing!")

def main():
    """Main organization function"""
    
    parser = argparse.ArgumentParser(description="Organize LoRA matrices by (layer, module) combinations")
    parser.add_argument("--input_dir", 
                       required=True,
                       help="Directory containing extracted matrix files")
    parser.add_argument("--output_dir", 
                       required=True,
                       help="Output directory for organized matrices")
    
    args = parser.parse_args()
    
    print("🚀 STARCODER2-7B MATRIX ORGANIZATION BY (LAYER, MODULE)")
    print("=" * 70)
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        return
    
    organized_dir, indexed_data = organize_by_layer_module(args.input_dir, args.output_dir)
    
    # Get expected checkpoints for verification
    checkpoints = discover_checkpoint_files(args.input_dir)
    verify_organization(organized_dir, indexed_data, list(checkpoints.keys()))
    
    print(f"\n💡 USAGE:")
    print(f"  - Each index (0-{len(indexed_data)-1}) represents one (layer, module) combination")
    print(f"  - Each index contains AB matrices from available checkpoints")
    print(f"  - Detected checkpoints: {list(checkpoints.keys())}")
    print(f"  - Access pattern: index_XXX_layer_YY_module_matrices.safetensors")
    print(f"  - Master index maps layer_module names to indices")

if __name__ == "__main__":
    main()
