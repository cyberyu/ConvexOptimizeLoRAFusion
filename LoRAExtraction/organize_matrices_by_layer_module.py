#!/usr/bin/env python3
"""
Reorganize StarCoder2-7B matrices by (layer, module) across all 4 checkpoints
Each unique (layer, module) combination gets an index with 4 checkpoint results
"""

import os
import torch
from safetensors import safe_open
import safetensors.torch
import json
from collections import defaultdict

def load_checkpoint_matrices(output_dir, checkpoint_name):
    """Load AB products from a specific checkpoint"""
    
    AB_file = f"starcoder27b_{checkpoint_name}_checkpoint-40000_AB_products.safetensors"
    AB_path = os.path.join(output_dir, AB_file)
    
    matrices = {}
    with safe_open(AB_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            matrices[key] = f.get_tensor(key)
    
    return matrices

def organize_by_layer_module():
    """Organize matrices by (layer, module) across all checkpoints"""
    
    output_dir = "extracted_starcoder27b_matrices"
    checkpoints = ["annotated", "concatenationTrained", "multiline", "singleline"]
    
    print("🔄 REORGANIZING MATRICES BY (LAYER, MODULE)")
    print("=" * 60)
    
    # Load all checkpoint data
    all_checkpoint_data = {}
    for checkpoint in checkpoints:
        print(f"📦 Loading {checkpoint} matrices...")
        all_checkpoint_data[checkpoint] = load_checkpoint_matrices(output_dir, checkpoint)
    
    # Organize by (layer, module)
    layer_module_data = defaultdict(dict)
    
    # Process first checkpoint to get structure
    first_checkpoint = checkpoints[0]
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
            
            if layer_idx is not None and module_type:
                layer_module_key = f"layer_{layer_idx:02d}_{module_type}"
                
                # Collect matrices from all checkpoints for this (layer, module)
                for checkpoint in checkpoints:
                    if matrix_key in all_checkpoint_data[checkpoint]:
                        layer_module_data[layer_module_key][checkpoint] = all_checkpoint_data[checkpoint][matrix_key]
    
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
    organized_dir = os.path.join(output_dir, "organized_by_layer_module")
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
        'checkpoints': checkpoints,
        'index_mapping': layer_module_index,
        'structure_info': {
            'layers': 32,
            'modules_per_layer': 4,
            'module_types': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
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
        print(f"  📦 Checkpoints: {checkpoint_count}/4 {list(data['matrices'].keys())}")
        
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
    print(f"  🔢 Each combination has matrices from {len(checkpoints)} checkpoints")
    
    return organized_dir, indexed_data

def verify_organization(organized_dir, indexed_data):
    """Verify the organization is correct"""
    
    print(f"\n🔍 VERIFYING ORGANIZATION")
    print("=" * 40)
    
    # Check that we have expected number of combinations
    expected_combinations = 32 * 4  # 32 layers × 4 modules
    actual_combinations = len(indexed_data)
    
    print(f"Expected combinations: {expected_combinations}")
    print(f"Actual combinations: {actual_combinations}")
    
    if actual_combinations == expected_combinations:
        print(f"✅ Combination count correct!")
    else:
        print(f"❌ Combination count mismatch!")
    
    # Verify each combination has all 4 checkpoints
    complete_combinations = 0
    for idx, data in indexed_data.items():
        if len(data['matrices']) == 4:
            complete_combinations += 1
    
    print(f"Complete combinations (4 checkpoints): {complete_combinations}/{actual_combinations}")
    
    if complete_combinations == actual_combinations:
        print(f"✅ All combinations have 4 checkpoints!")
    else:
        print(f"❌ Some combinations missing checkpoints!")
    
    # Verify file structure
    expected_files = actual_combinations * 2 + 1  # 2 files per combination + master index
    actual_files = len([f for f in os.listdir(organized_dir) if os.path.isfile(os.path.join(organized_dir, f))])
    
    print(f"Expected files: {expected_files}")
    print(f"Actual files: {actual_files}")
    
    if actual_files == expected_files:
        print(f"✅ All files created!")
    else:
        print(f"❌ File count mismatch!")

def main():
    """Main organization function"""
    
    print("🚀 STARCODER2-7B MATRIX ORGANIZATION BY (LAYER, MODULE)")
    print("=" * 70)
    
    organized_dir, indexed_data = organize_by_layer_module()
    verify_organization(organized_dir, indexed_data)
    
    print(f"\n💡 USAGE:")
    print(f"  - Each index (0-127) represents one (layer, module) combination")
    print(f"  - Each index contains AB matrices from all 4 checkpoints:")
    print(f"    * annotated")
    print(f"    * concatenationTrained") 
    print(f"    * multiline")
    print(f"    * singleline")
    print(f"  - Access pattern: index_XXX_layer_YY_module_matrices.safetensors")
    print(f"  - Master index maps layer_module names to indices")

if __name__ == "__main__":
    main()
