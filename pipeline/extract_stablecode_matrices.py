#!/usr/bin/env python3
"""
Enhanced StableCode LoRA A/B Matrix Extractor

This script extracts LoRA A and B matrices from StableCode checkpoint files and also computes
their products (A*B) to get the LoRA change matrices. It processes all four
StableCode checkpoints: annotated, concatenationTrained, multiline, and singleline.

ENHANCED FEATURES:
- Extracts separate A and B matrices (for SVD optimization)
- Computes AB products (for existing L1/L2 optimization)
- Saves all matrices in organized format
- Provides comprehensive verification

The LoRA decomposition works as: ΔW = B @ A * (alpha/r)
Where:
- A: down-projection matrix (rank × input_dim)
- B: up-projection matrix (output_dim × rank) 
- alpha: scaling factor from config
- r: rank from config
"""

import os
import torch
import safetensors.torch
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import argparse
import gc


def load_lora_config(checkpoint_path: str) -> Dict:
    """Load LoRA configuration from adapter_config.json."""
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_checkpoints_config(config_path: str) -> Dict:
    """Load checkpoint configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_lora_matrices_from_checkpoint(checkpoint_path: str, checkpoint_name: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract A, B, and AB matrices from a single checkpoint.
    
    Returns:
        Dictionary mapping layer names to {'A': tensor, 'B': tensor, 'AB': tensor}
    """
    print(f"  📁 Processing checkpoint: {checkpoint_name}")
    print(f"     Path: {checkpoint_path}")
    
    # Load adapter model
    adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter model not found: {adapter_path}")
    
    # Load LoRA configuration
    lora_config = load_lora_config(checkpoint_path)
    alpha = lora_config.get('lora_alpha', 16)
    rank = lora_config.get('r', 16)
    scaling_factor = alpha / rank
    
    print(f"     LoRA config: alpha={alpha}, rank={rank}, scaling={scaling_factor:.4f}")
    
    # Load adapter tensors
    adapter_state = safetensors.torch.load_file(adapter_path, device="cpu")
    
    # Group LoRA A and B matrices by layer
    lora_pairs = {}
    
    for key in adapter_state.keys():
        if '.lora_A.weight' in key:
            # Extract base key (everything before .lora_A.weight)
            base_key = key.replace('.lora_A.weight', '')
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]['A'] = adapter_state[key]
        elif '.lora_B.weight' in key:
            # Extract base key (everything before .lora_B.weight)
            base_key = key.replace('.lora_B.weight', '')
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]['B'] = adapter_state[key]
    
    # Process A, B, and compute AB products
    layer_matrices = {}
    processed_layers = 0
    
    for base_key, matrices in lora_pairs.items():
        if 'A' in matrices and 'B' in matrices:
            A = matrices['A']  # Shape: (rank, input_dim)
            B = matrices['B']  # Shape: (output_dim, rank)
            
            # Compute AB product with scaling
            AB = (B @ A) * scaling_factor  # Shape: (output_dim, input_dim)
            
            # Create a simplified key for storage
            simplified_key = simplify_layer_key(base_key)
            
            # Store all three matrices: A, B, and AB
            layer_matrices[simplified_key] = {
                'A': A.clone(),          # Separate A matrix for SVD
                'B': B.clone(),          # Separate B matrix for SVD  
                'AB': AB.clone(),        # AB product for L1/L2 optimization
                'scaling_factor': scaling_factor,
                'alpha': alpha,
                'rank': rank
            }
            
            processed_layers += 1
            
            # Memory cleanup
            del A, B, AB
    
    print(f"     ✅ Processed {processed_layers} LoRA layer pairs")
    print(f"     📊 Extracted A, B, and AB matrices for each layer")
    
    # Clear memory
    del adapter_state
    gc.collect()
    
    return layer_matrices


def simplify_layer_key(full_key: str) -> str:
    """
    Convert full LoRA key to simplified format.
    
    Example:
        base_model.model.transformer.h.0.attn.q_proj -> layer_00_attn_q_proj
        base_model.model.transformer.h.15.attn.k_proj -> layer_15_attn_k_proj
    """
    # Remove base_model.model.transformer.h. prefix for StableCode
    if full_key.startswith('base_model.model.transformer.h.'):
        key = full_key[len('base_model.model.transformer.h.'):]
    elif full_key.startswith('base_model.model.model.layers.'):
        # Alternative format that might be used
        key = full_key[len('base_model.model.model.layers.'):]
    else:
        key = full_key
    
    # Parse h.X.component format for StableCode (transformer.h.layer_num.component)
    parts = key.split('.')
    if len(parts) >= 2 and parts[0].isdigit():
        layer_num = int(parts[0])
        component_parts = parts[1:]  # attn.q_proj, mlp.gate_proj, etc.
        
        # Format: layer_XX_component
        simplified = f"layer_{layer_num:02d}_" + "_".join(component_parts)
        return simplified
    else:
        # Fallback: just replace dots with underscores
        return key.replace('.', '_')


def save_matrices_to_safetensors(matrices_dict: Dict[str, Dict[str, Dict[str, torch.Tensor]]], output_dir: str) -> None:
    """Save all matrices to individual safetensors files organized by layer and matrix type."""
    
    print(f"\n💾 Saving matrices to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different matrix types
    ab_dir = os.path.join(output_dir, "ab_products")
    a_dir = os.path.join(output_dir, "a_matrices") 
    b_dir = os.path.join(output_dir, "b_matrices")
    combined_dir = os.path.join(output_dir, "combined_abc")
    
    os.makedirs(ab_dir, exist_ok=True)
    os.makedirs(a_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    
    # Get all unique layer names across all checkpoints
    all_layers = set()
    for checkpoint_matrices in matrices_dict.values():
        all_layers.update(checkpoint_matrices.keys())
    
    all_layers = sorted(list(all_layers))
    print(f"   Found {len(all_layers)} unique layers")
    
    # Save matrices grouped by layer and type
    for i, layer_name in enumerate(all_layers):
        # Collect matrices for this layer from all checkpoints
        ab_matrices = {}
        a_matrices = {}
        b_matrices = {}
        combined_matrices = {}
        
        for checkpoint_name, matrices in matrices_dict.items():
            if layer_name in matrices:
                layer_data = matrices[layer_name]
                ab_matrices[checkpoint_name] = layer_data['AB']
                a_matrices[checkpoint_name] = layer_data['A'] 
                b_matrices[checkpoint_name] = layer_data['B']
                
                # Combined format with all data
                combined_matrices[f"{checkpoint_name}_A"] = layer_data['A']
                combined_matrices[f"{checkpoint_name}_B"] = layer_data['B'] 
                combined_matrices[f"{checkpoint_name}_AB"] = layer_data['AB']
        
        if ab_matrices:
            # Save AB products (for existing L1/L2 optimization)
            ab_filename = f"{layer_name}_ab_matrices.safetensors"
            ab_filepath = os.path.join(ab_dir, ab_filename)
            safetensors.torch.save_file(ab_matrices, ab_filepath)
            
            # Save A matrices (for SVD optimization)
            a_filename = f"{layer_name}_a_matrices.safetensors"
            a_filepath = os.path.join(a_dir, a_filename)
            safetensors.torch.save_file(a_matrices, a_filepath)
            
            # Save B matrices (for SVD optimization)
            b_filename = f"{layer_name}_b_matrices.safetensors"
            b_filepath = os.path.join(b_dir, b_filename)
            safetensors.torch.save_file(b_matrices, b_filepath)
            
            # Save combined format (A, B, AB all in one file)
            combined_filename = f"{layer_name}_all_matrices.safetensors"
            combined_filepath = os.path.join(combined_dir, combined_filename)
            safetensors.torch.save_file(combined_matrices, combined_filepath)
            
            # Also save in the main directory for backward compatibility
            legacy_filename = f"{layer_name}_matrices.safetensors"
            legacy_filepath = os.path.join(output_dir, legacy_filename)
            safetensors.torch.save_file(ab_matrices, legacy_filepath)
            
            if (i + 1) % 20 == 0:
                print(f"   Saved {i + 1}/{len(all_layers)} layers...")
    
    print(f"   ✅ Saved all {len(all_layers)} layers in multiple formats:")
    print(f"      📁 AB products: {ab_dir}/")
    print(f"      📁 A matrices: {a_dir}/")
    print(f"      📁 B matrices: {b_dir}/")
    print(f"      📁 Combined: {combined_dir}/")
    print(f"      📁 Legacy: {output_dir}/ (AB only for compatibility)")


def create_layer_index(matrices_dict: Dict[str, Dict[str, Dict[str, torch.Tensor]]], output_dir: str) -> None:
    """Create an index file mapping layers to matrix information."""
    
    print(f"\n📋 Creating comprehensive layer index...")
    
    # Get all unique layers
    all_layers = set()
    for checkpoint_matrices in matrices_dict.values():
        all_layers.update(checkpoint_matrices.keys())
    
    all_layers = sorted(list(all_layers))
    
    # Create index with layer information
    index_data = {
        'total_layers': len(all_layers),
        'checkpoint_names': list(matrices_dict.keys()),
        'matrix_types': ['A', 'B', 'AB'],
        'storage_formats': {
            'ab_products': 'AB matrices only (for L1/L2 optimization)',
            'a_matrices': 'A matrices only (for SVD optimization)',
            'b_matrices': 'B matrices only (for SVD optimization)', 
            'combined_abc': 'A, B, AB all in one file (comprehensive)',
            'legacy': 'AB matrices in main directory (backward compatibility)'
        },
        'layers': {}
    }
    
    for layer_name in all_layers:
        layer_info = {
            'layer_name': layer_name,
            'available_checkpoints': [],
            'matrix_shapes': {},
            'lora_config': {},
            'file_locations': {
                'ab_products': f"ab_products/{layer_name}_ab_matrices.safetensors",
                'a_matrices': f"a_matrices/{layer_name}_a_matrices.safetensors",
                'b_matrices': f"b_matrices/{layer_name}_b_matrices.safetensors",
                'combined': f"combined_abc/{layer_name}_all_matrices.safetensors",
                'legacy': f"{layer_name}_matrices.safetensors"
            }
        }
        
        # Check which checkpoints have this layer
        for checkpoint_name, matrices in matrices_dict.items():
            if layer_name in matrices:
                layer_data = matrices[layer_name]
                layer_info['available_checkpoints'].append(checkpoint_name)
                
                # Store shapes for A, B, AB
                layer_info['matrix_shapes'][checkpoint_name] = {
                    'A': list(layer_data['A'].shape),
                    'B': list(layer_data['B'].shape), 
                    'AB': list(layer_data['AB'].shape)
                }
                
                # Store LoRA config (should be same across checkpoints)
                if 'lora_config' not in layer_info or not layer_info['lora_config']:
                    layer_info['lora_config'] = {
                        'rank': layer_data['rank'],
                        'alpha': layer_data['alpha'],
                        'scaling_factor': layer_data['scaling_factor']
                    }
        
        index_data['layers'][layer_name] = layer_info
    
    # Save index
    index_path = os.path.join(output_dir, "enhanced_layer_index.json")
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"   ✅ Enhanced index saved to: {index_path}")
    
    # Also create backward-compatible index 
    legacy_index = {
        'total_layers': len(all_layers),
        'checkpoint_names': list(matrices_dict.keys()),
        'layers': {}
    }
    
    for layer_name in all_layers:
        layer_info = {
            'layer_name': layer_name,
            'available_checkpoints': index_data['layers'][layer_name]['available_checkpoints'],
            'matrix_shapes': {
                checkpoint: shapes['AB'] 
                for checkpoint, shapes in index_data['layers'][layer_name]['matrix_shapes'].items()
            },
            'filename': f"{layer_name}_matrices.safetensors"
        }
        legacy_index['layers'][layer_name] = layer_info
    
    legacy_index_path = os.path.join(output_dir, "layer_index.json")
    with open(legacy_index_path, 'w') as f:
        json.dump(legacy_index, f, indent=2)
    
    print(f"   ✅ Legacy index saved to: {legacy_index_path}")


def extract_all_stablecode_matrices(config_path: str, output_dir: str) -> None:
    """
    Main function to extract LoRA A, B, and AB matrices from all StableCode checkpoints.
    """
    print("🚀 ENHANCED STABLECODE LORA MATRIX EXTRACTION")
    print("=" * 60)
    print("Features:")
    print("  📊 Extracts separate A and B matrices (for SVD optimization)")
    print("  📊 Computes AB products (for L1/L2 optimization)")
    print("  📁 Saves in multiple formats for different use cases")
    print("  🔧 Backward compatible with existing pipeline")
    print()
    
    # Load configuration
    print(f"📋 Loading configuration from: {config_path}")
    config = load_checkpoints_config(config_path)
    checkpoints = config['checkpoints']
    
    print(f"✅ Found {len(checkpoints)} checkpoints to process")
    
    # Verify all paths exist
    if config.get('settings', {}).get('verify_paths', True):
        print(f"\n🔍 Verifying checkpoint paths...")
        for name, info in checkpoints.items():
            path = info['path']
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint path not found: {path} ({name})")
            print(f"   ✅ {name}: {path}")
    
    # Extract matrices from each checkpoint
    all_matrices = {}
    
    print(f"\n🔧 Extracting LoRA A, B, and AB matrices...")
    for checkpoint_name, checkpoint_info in checkpoints.items():
        checkpoint_path = checkpoint_info['path']
        
        try:
            matrices = extract_lora_matrices_from_checkpoint(checkpoint_path, checkpoint_name)
            all_matrices[checkpoint_name] = matrices
            print(f"     Extracted A, B, AB for {len(matrices)} layers from {checkpoint_name}")
        except Exception as e:
            print(f"     ❌ Error processing {checkpoint_name}: {e}")
            continue
    
    if not all_matrices:
        raise RuntimeError("No matrices were successfully extracted!")
    
    # Save matrices in multiple formats
    save_matrices_to_safetensors(all_matrices, output_dir)
    
    # Create comprehensive index
    create_layer_index(all_matrices, output_dir)
    
    # Create verification script
    create_verification_script(output_dir)
    
    # Save extraction metadata
    sample_layer_data = next(iter(next(iter(all_matrices.values())).values()))
    lora_config_sample = {
        'rank': sample_layer_data['rank'],
        'alpha': sample_layer_data['alpha'], 
        'scaling_factor': sample_layer_data['scaling_factor']
    }
    
    metadata = {
        'extraction_type': 'enhanced_stablecode_lora_matrices',
        'extraction_date': str(Path().absolute()),
        'config_file': config_path,
        'output_directory': output_dir,
        'checkpoints_processed': list(all_matrices.keys()),
        'total_checkpoints': len(checkpoints),
        'total_layers_extracted': len(set().union(*[matrices.keys() for matrices in all_matrices.values()])),
        'matrix_types_extracted': ['A', 'B', 'AB'],
        'storage_formats': [
            'ab_products/ (AB matrices for L1/L2 optimization)',
            'a_matrices/ (A matrices for SVD optimization)',
            'b_matrices/ (B matrices for SVD optimization)',
            'combined_abc/ (All matrices in single files)',
            'legacy format (AB only, backward compatible)'
        ],
        'extraction_summary': {
            checkpoint: len(matrices) for checkpoint, matrices in all_matrices.items()
        },
        'lora_config_sample': lora_config_sample
    }
    
    metadata_path = os.path.join(output_dir, "enhanced_extraction_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ ENHANCED EXTRACTION COMPLETE!")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   📊 Processed {len(all_matrices)} checkpoints")
    print(f"   🔢 Extracted {metadata['total_layers_extracted']} unique layers")
    print(f"   📋 Enhanced metadata: {metadata_path}")
    print(f"\n📁 Directory Structure:")
    print(f"   {output_dir}/")
    print(f"   ├── ab_products/        # AB matrices (for L1/L2 optimization)")
    print(f"   ├── a_matrices/         # A matrices (for SVD optimization)")
    print(f"   ├── b_matrices/         # B matrices (for SVD optimization)")
    print(f"   ├── combined_abc/       # All matrices per layer")
    print(f"   ├── *_matrices.safetensors  # Legacy AB format")
    print(f"   ├── enhanced_layer_index.json")
    print(f"   └── layer_index.json    # Backward compatible")


def create_verification_script(output_dir: str) -> None:
    """Create a verification script to validate the extracted matrices."""
    
    verification_script = '''#!/usr/bin/env python3
"""
Verification script for enhanced LoRA matrix extraction
"""

from safetensors import safe_open
import torch
import json
import os

def verify_extraction(base_dir):
    print("🔍 VERIFYING ENHANCED LORA EXTRACTION")
    print("=" * 50)
    
    # Check directory structure
    required_dirs = ['ab_products', 'a_matrices', 'b_matrices', 'combined_abc']
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"✅ {dir_name}/: {len(files)} files")
        else:
            print(f"❌ {dir_name}/: Missing!")
    
    # Load and verify index
    index_path = os.path.join(base_dir, "enhanced_layer_index.json")
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
        print(f"✅ Enhanced index: {index['total_layers']} layers")
        
        # Check a sample layer
        sample_layer = list(index['layers'].keys())[0]
        layer_info = index['layers'][sample_layer]
        
        print(f"\\n🔍 Verifying sample layer: {sample_layer}")
        
        # Check if all matrix files exist
        for format_name, file_path in layer_info['file_locations'].items():
            full_path = os.path.join(base_dir, file_path)
            if os.path.exists(full_path):
                print(f"  ✅ {format_name}: {file_path}")
                
                # Verify matrix contents
                try:
                    with safe_open(full_path, framework="pt", device="cpu") as f:
                        keys = list(f.keys())
                        print(f"     Keys: {keys}")
                        for key in keys[:2]:  # Check first 2
                            tensor = f.get_tensor(key)
                            print(f"     {key}: {tensor.shape}")
                except Exception as e:
                    print(f"     Error reading: {e}")
            else:
                print(f"  ❌ {format_name}: {file_path} (missing)")
    else:
        print(f"❌ Enhanced index missing: {index_path}")
    
    print(f"\\n✅ Verification complete!")

if __name__ == "__main__":
    verify_extraction(".")
'''
    
    script_path = os.path.join(output_dir, "verify_extraction.py")
    with open(script_path, 'w') as f:
        f.write(verification_script)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print(f"   📝 Verification script created: {script_path}")


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced LoRA matrix extraction from StableCode checkpoints",
        epilog="Extracts A, B, and AB matrices in multiple formats for different optimization approaches"
    )
    parser.add_argument(
        "--config", 
        default="checkpoints_config_stablecode.yml",
        help="Path to checkpoints configuration YAML file"
    )
    parser.add_argument(
        "--output", 
        default="enhanced_extracted_stablecode_matrices",
        help="Output directory for extracted matrices"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    config_path = os.path.abspath(args.config)
    output_dir = os.path.abspath(args.output)
    
    print(f"Enhanced LoRA Matrix Extraction for StableCode")
    print(f"Configuration: {config_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Run extraction
    try:
        extract_all_stablecode_matrices(config_path, output_dir)
        print(f"\n🎉 SUCCESS! Enhanced extraction completed.")
        print(f"   Use the verification script to validate: {output_dir}/verify_extraction.py")
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())