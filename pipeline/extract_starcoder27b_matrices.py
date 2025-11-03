#!/usr/bin/env python3
"""
StarCoder2-7B LoRA A/B Matrix Extractor

This script extracts LoRA A and B matrices from StarCoder2-7B checkpoint files and computes
their products (A*B) to get the LoRA change matrices. It processes all four
StarCoder2-7B checkpoints: annotated, concatenationTrained, multiline, and singleline.

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


def create_default_config(config_path: str) -> None:
    """Create a default YAML configuration file."""
    default_config = {
        'checkpoints': {
            'annotated': {
                'path': '../starcoder27b/annotated/checkpoint-40000',
                'description': 'Annotated training data checkpoint'
            },
            'multiline': {
                'path': '../starcoder27b/multiline/checkpoint-40000',
                'description': 'Multiline training data checkpoint'
            },
            'singleline': {
                'path': '../starcoder27b/singleline/checkpoint-40000',
                'description': 'Singleline training data checkpoint'
            },
            'concatenationTrained': {
                'path': '../starcoder27b/concatenationTrained/checkpoint-40000',
                'description': 'Concatenation trained checkpoint (target for optimization)'
            }
        },
        'settings': {
            'verify_paths': True,
            'verbose_paths': False,
            'expand_env_vars': True,
            'validation': {
                'required_files': [
                    'adapter_model.safetensors',
                    'adapter_config.json'
                ],
                'optional_files': [
                    'tokenizer.json',
                    'tokenizer_config.json',
                    'training_args.bin'
                ]
            }
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created default configuration file: {config_path}")
    print("📝 Please edit this file to match your checkpoint locations")


def validate_checkpoint_paths(config: Dict, verbose: bool = False) -> Dict[str, str]:
    """
    Validate that all checkpoint paths exist and return a mapping of name to path.
    
    Args:
        config: YAML configuration dictionary
        verbose: Whether to show detailed path information
        
    Returns:
        Dictionary mapping checkpoint names to their validated paths
    """
    checkpoints = config.get('checkpoints', {})
    settings = config.get('settings', {})
    verify_paths = settings.get('verify_paths', True)
    verbose_paths = settings.get('verbose_paths', False) or verbose
    expand_env_vars = settings.get('expand_env_vars', True)
    validation_settings = settings.get('validation', {})
    
    # Default required files
    required_files = validation_settings.get('required_files', [
        "adapter_model.safetensors", 
        "adapter_config.json"
    ])
    optional_files = validation_settings.get('optional_files', [])
    
    validated_checkpoints = {}
    
    print(f"\n📋 Validating checkpoint paths:")
    print("-" * 50)
    
    for name, checkpoint_info in checkpoints.items():
        path = checkpoint_info.get('path')
        description = checkpoint_info.get('description', 'No description')
        
        if not path:
            print(f"❌ {name}: No path specified")
            continue
        
        # Expand environment variables if enabled
        if expand_env_vars:
            path = os.path.expandvars(path)
            
        # Expand user home directory (~)
        expanded_path = os.path.expanduser(path)
        absolute_path = os.path.abspath(expanded_path)
        
        if verbose_paths:
            print(f"🔍 {name}:")
            print(f"   Description: {description}")
            print(f"   Original path: {checkpoint_info.get('path')}")
            if expand_env_vars and path != checkpoint_info.get('path'):
                print(f"   Expanded path: {path}")
            print(f"   Absolute path: {absolute_path}")
        
        if verify_paths:
            if os.path.exists(absolute_path):
                # Check required files
                missing_required = []
                for req_file in required_files:
                    file_path = os.path.join(absolute_path, req_file)
                    if not os.path.exists(file_path):
                        missing_required.append(req_file)
                
                if not missing_required:
                    validated_checkpoints[name] = absolute_path
                    
                    # Check optional files for info
                    found_optional = []
                    for opt_file in optional_files:
                        file_path = os.path.join(absolute_path, opt_file)
                        if os.path.exists(file_path):
                            found_optional.append(opt_file)
                    
                    print(f"✅ {name}: Valid checkpoint found")
                    if verbose_paths and found_optional:
                        print(f"   Optional files found: {found_optional}")
                else:
                    print(f"❌ {name}: Missing required files: {missing_required}")
                    if verbose_paths:
                        print(f"   Path: {absolute_path}")
            else:
                print(f"❌ {name}: Path does not exist: {absolute_path}")
        else:
            validated_checkpoints[name] = absolute_path
            print(f"📝 {name}: Added (not verified)")
    
    if not validated_checkpoints:
        raise ValueError("No valid checkpoints found in configuration")
    
    print(f"\n✅ Found {len(validated_checkpoints)} valid checkpoints")
    return validated_checkpoints


def extract_lora_matrices(checkpoint_path: str, output_dir: str) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA A and B matrices from a StarCoder2-7B checkpoint and compute their products.
    
    Args:
        checkpoint_path: Path to the LoRA checkpoint directory
        output_dir: Directory to save extracted matrices
        
    Returns:
        Dictionary containing A matrices, B matrices, and AB products
    """
    print(f"\n🔍 Processing StarCoder2-7B checkpoint: {checkpoint_path}")
    
    # Load configuration
    config = load_lora_config(checkpoint_path)
    lora_alpha = config.get('lora_alpha', 8)
    lora_r = config.get('r', 8) 
    scaling_factor = lora_alpha / lora_r
    target_modules = config.get('target_modules', [])
    
    print(f"  📊 LoRA config: alpha={lora_alpha}, r={lora_r}, scaling={scaling_factor:.3f}")
    print(f"  🎯 Target modules: {target_modules}")
    
    # Load adapter weights
    adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter model not found: {adapter_path}")
    
    weights = safetensors.torch.load_file(adapter_path)
    print(f"  📦 Loaded {len(weights)} weight tensors")
    
    # Organize weights by module
    modules = {}
    attention_modules = set()
    mlp_modules = set()
    
    for key, tensor in weights.items():
        # Parse key: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
        # or: base_model.model.layers.0.mlp.gate_proj.lora_A.weight
        parts = key.split('.')
        if 'lora_A' in key or 'lora_B' in key:
            # Extract module path and matrix type
            matrix_type = parts[-2]  # lora_A or lora_B
            module_path = '.'.join(parts[:-2])  # Everything before lora_A/lora_B
            
            if module_path not in modules:
                modules[module_path] = {}
            
            modules[module_path][matrix_type] = tensor
            
            # Track module types for analysis
            if 'self_attn' in module_path:
                attention_modules.add(module_path)
            elif 'mlp' in module_path:
                mlp_modules.add(module_path)
    
    print(f"  🧩 Found {len(modules)} LoRA modules")
    print(f"    - Attention modules: {len(attention_modules)}")
    print(f"    - MLP modules: {len(mlp_modules)}")
    
    # Verify if this matches expected configuration
    if len(mlp_modules) == 0 and target_modules:
        mlp_target_modules = [m for m in target_modules if m in ['gate_proj', 'up_proj', 'down_proj']]
        if mlp_target_modules:
            print(f"    ⚠️  Note: Config lists MLP targets {mlp_target_modules} but no MLP LoRA found")
            print(f"    ℹ️  StarCoder2-7B uses ATTENTION-ONLY LoRA despite config")
    
    # Extract A, B matrices and compute products
    extracted_data = {
        'A_matrices': {},
        'B_matrices': {},
        'AB_products': {},
        'metadata': {
            'checkpoint_path': checkpoint_path,
            'lora_alpha': lora_alpha,
            'lora_r': lora_r,
            'scaling_factor': scaling_factor,
            'num_modules': len(modules),
            'target_modules': target_modules,
            'actual_attention_modules': len(attention_modules),
            'actual_mlp_modules': len(mlp_modules),
            'attention_only_lora': len(mlp_modules) == 0
        }
    }
    
    processed_count = 0
    for module_path, matrices in modules.items():
        if 'lora_A' in matrices and 'lora_B' in matrices:
            A_matrix = matrices['lora_A']  # Shape: [r, input_dim]
            B_matrix = matrices['lora_B']  # Shape: [output_dim, r]
            
            # Compute LoRA change matrix: ΔW = B @ A * scaling
            AB_product = (B_matrix @ A_matrix) * scaling_factor
            
            # Clean module name for saving
            clean_name = module_path.replace('base_model.model.', '')
            
            extracted_data['A_matrices'][clean_name] = A_matrix
            extracted_data['B_matrices'][clean_name] = B_matrix  
            extracted_data['AB_products'][clean_name] = AB_product
            
            processed_count += 1
            if processed_count <= 10:  # Show first 10 modules
                print(f"    ✓ {clean_name}: A{A_matrix.shape} @ B{B_matrix.shape} -> ΔW{AB_product.shape}")
            elif processed_count == 11:
                print(f"    ... (showing first 10 modules)")
    
    print(f"  ✅ Processed {processed_count} modules successfully")
    
    # Save extracted matrices with memory management
    checkpoint_name = os.path.basename(checkpoint_path)
    # Get parent directory name for unique identification
    parent_dir = os.path.basename(os.path.dirname(checkpoint_path))
    unique_name = f"{parent_dir}_{checkpoint_name}"
    
    # Save A matrices
    A_path = os.path.join(output_dir, f"starcoder27b_{unique_name}_A_matrices.safetensors")
    safetensors.torch.save_file(extracted_data['A_matrices'], A_path)
    print(f"  💾 Saved A matrices: {A_path}")
    
    # Clear A matrices from memory and collect garbage
    del extracted_data['A_matrices']
    gc.collect()
    
    # Save B matrices  
    B_path = os.path.join(output_dir, f"starcoder27b_{unique_name}_B_matrices.safetensors")
    safetensors.torch.save_file(extracted_data['B_matrices'], B_path)
    print(f"  💾 Saved B matrices: {B_path}")
    
    # Clear B matrices from memory
    del extracted_data['B_matrices']
    gc.collect()
    
    # Save AB products (LoRA change matrices)
    AB_path = os.path.join(output_dir, f"starcoder27b_{unique_name}_AB_products.safetensors")
    safetensors.torch.save_file(extracted_data['AB_products'], AB_path)
    print(f"  💾 Saved AB products: {AB_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"starcoder27b_{unique_name}_metadata.json")
    metadata_to_save = extracted_data['metadata'].copy()
    
    # Add shape information (sample a few modules to avoid memory issues)
    sample_modules = list(extracted_data['AB_products'].keys())[:5]
    metadata_to_save['sample_module_shapes'] = {}
    
    for name in sample_modules:
        # We need to reload A and B matrices for shape info since we deleted them
        if name in extracted_data['AB_products']:
            AB_shape = list(extracted_data['AB_products'][name].shape)
            metadata_to_save['sample_module_shapes'][name] = {
                'AB_shape': AB_shape,
                'rank': lora_r
            }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_to_save, f, indent=2)
    
    print(f"  💾 Saved metadata: {metadata_path}")
    
    return extracted_data


def analyze_matrix_statistics(extracted_data: Dict[str, Dict[str, torch.Tensor]]) -> None:
    """Analyze and print statistics about the extracted matrices."""
    print(f"\n📊 MATRIX STATISTICS")
    print("-" * 50)
    
    for checkpoint_name, data in extracted_data.items():
        print(f"\n{checkpoint_name.upper()}:")
        
        AB_products = data['AB_products']
        metadata = data['metadata']
        
        print(f"  📦 Total modules: {metadata['num_modules']}")
        print(f"  🎯 Actual attention modules: {metadata['actual_attention_modules']}")
        print(f"  🧠 Actual MLP modules: {metadata['actual_mlp_modules']}")
        print(f"  ✅ Attention-only LoRA: {metadata['attention_only_lora']}")
        print(f"  🔧 Configured target modules: {metadata['target_modules']}")
        
        # Analyze AB products
        AB_norms = [torch.norm(AB).item() for AB in AB_products.values()]
        print(f"  🔄 AB products:")
        print(f"    - Count: {len(AB_products)}")
        print(f"    - Mean norm: {np.mean(AB_norms):.4f}")
        print(f"    - Std norm: {np.std(AB_norms):.4f}")
        print(f"    - Min norm: {np.min(AB_norms):.4f}")
        print(f"    - Max norm: {np.max(AB_norms):.4f}")
        
        # Categorize modules by type
        attention_count = 0
        mlp_count = 0
        module_types = {}
        
        for name in AB_products.keys():
            if 'self_attn' in name:
                attention_count += 1
                # Extract attention module type (q_proj, k_proj, v_proj, o_proj)
                if 'q_proj' in name:
                    module_types['q_proj'] = module_types.get('q_proj', 0) + 1
                elif 'k_proj' in name:
                    module_types['k_proj'] = module_types.get('k_proj', 0) + 1
                elif 'v_proj' in name:
                    module_types['v_proj'] = module_types.get('v_proj', 0) + 1
                elif 'o_proj' in name:
                    module_types['o_proj'] = module_types.get('o_proj', 0) + 1
            elif 'mlp' in name:
                mlp_count += 1
                # Extract MLP module type (gate_proj, up_proj, down_proj)
                if 'gate_proj' in name:
                    module_types['gate_proj'] = module_types.get('gate_proj', 0) + 1
                elif 'up_proj' in name:
                    module_types['up_proj'] = module_types.get('up_proj', 0) + 1
                elif 'down_proj' in name:
                    module_types['down_proj'] = module_types.get('down_proj', 0) + 1
        
        print(f"  🏗️  Module breakdown: {dict(module_types)}")
        print(f"  📊 Attention vs MLP: {attention_count} vs {mlp_count}")


def compare_across_checkpoints(extracted_data: Dict[str, Dict]) -> None:
    """Compare matrices across different StarCoder2-7B checkpoints."""
    print(f"\n🔍 CROSS-CHECKPOINT COMPARISON")
    print("-" * 50)
    
    checkpoint_names = list(extracted_data.keys())
    if len(checkpoint_names) < 2:
        print("Need at least 2 checkpoints for comparison")
        return
    
    # Find common modules across all checkpoints
    common_modules = set(extracted_data[checkpoint_names[0]]['AB_products'].keys())
    for checkpoint_name in checkpoint_names[1:]:
        common_modules &= set(extracted_data[checkpoint_name]['AB_products'].keys())
    
    print(f"📦 Common modules across all checkpoints: {len(common_modules)}")
    
    if not common_modules:
        print("No common modules found across checkpoints")
        return
    
    # Compare AB products for common modules
    print(f"\n🔄 AB Product Similarities (Cosine):")
    
    # Sample modules for comparison to avoid memory issues
    sample_modules = sorted(list(common_modules))[:20]  # First 20 modules
    print(f"  📊 Comparing {len(sample_modules)} sample modules")
    
    for i, checkpoint1 in enumerate(checkpoint_names):
        for j, checkpoint2 in enumerate(checkpoint_names[i+1:], i+1):
            print(f"\n  📋 {checkpoint1} vs {checkpoint2}:")
            
            similarities = []
            for module_name in sample_modules:
                AB1 = extracted_data[checkpoint1]['AB_products'][module_name]
                AB2 = extracted_data[checkpoint2]['AB_products'][module_name]
                
                # Compute cosine similarity
                AB1_flat = AB1.flatten()
                AB2_flat = AB2.flatten()
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    AB1_flat.unsqueeze(0), AB2_flat.unsqueeze(0)
                ).item()
                
                similarities.append(cos_sim)
                
                if len(similarities) <= 3:  # Show first 3 modules
                    print(f"    {module_name}: {cos_sim:.6f}")
            
            if len(similarities) > 3:
                print(f"    ... ({len(similarities) - 3} more modules)")
            
            print(f"    📊 Mean similarity: {np.mean(similarities):.6f}")
            print(f"    📈 Std similarity: {np.std(similarities):.6f}")
            print(f"    📉 Min similarity: {np.min(similarities):.6f}")
            print(f"    📈 Max similarity: {np.max(similarities):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Extract LoRA A and B matrices from StarCoder2-7B checkpoints")
    parser.add_argument("--config", 
                       default="checkpoints_config.yml",
                       help="YAML configuration file containing checkpoint paths")
    parser.add_argument("--output_dir", 
                       default="extracted_starcoder27b_matrices",
                       help="Output directory for extracted matrices")
    parser.add_argument("--checkpoints",
                       nargs="*",
                       help="Specific checkpoint names to process (from config). If not specified, processes all checkpoints in config.")
    parser.add_argument("--verbose",
                       action="store_true",
                       help="Show detailed path information during validation")
    # Backward compatibility - deprecated
    parser.add_argument("--starcoder_dir", 
                       help="[DEPRECATED] Use --config instead. Directory containing StarCoder2-7B checkpoints")
    parser.add_argument("--create_default_config",
                       action="store_true",
                       help="Create a default configuration file and exit")
    
    args = parser.parse_args()
    
    # Handle default config creation
    if args.create_default_config:
        create_default_config(args.config)
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 StarCoder2-7B LoRA A/B Matrix Extractor")
    print("=" * 60)
    
    # Handle backward compatibility
    if args.starcoder_dir:
        print("⚠️  WARNING: --starcoder_dir is deprecated. Please use --config with a YAML file.")
        print(f"📁 Using legacy StarCoder2-7B directory: {args.starcoder_dir}")
        
        # Legacy mode: use old behavior
        default_checkpoints = ["annotated/checkpoint-40000", 
                              "concatenationTrained/checkpoint-40000",
                              "multiline/checkpoint-40000",
                              "singleline/checkpoint-40000"]
        
        checkpoints_to_process = args.checkpoints if args.checkpoints else default_checkpoints
        validated_checkpoints = {}
        
        for checkpoint_name in checkpoints_to_process:
            checkpoint_path = os.path.join(args.starcoder_dir, checkpoint_name)
            if os.path.exists(checkpoint_path):
                # Get parent directory name for unique identification
                parent_dir = os.path.basename(os.path.dirname(checkpoint_path))
                validated_checkpoints[parent_dir] = checkpoint_path
            else:
                print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    else:
        # New YAML-based mode
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
        print(f"📁 Configuration file: {args.config}")
        
        # Load and validate configuration
        config = load_checkpoints_config(args.config)
        validated_checkpoints = validate_checkpoint_paths(config, verbose=args.verbose)
        
        # Filter checkpoints if specific ones were requested
        if args.checkpoints:
            filtered_checkpoints = {}
            for name in args.checkpoints:
                if name in validated_checkpoints:
                    filtered_checkpoints[name] = validated_checkpoints[name]
                else:
                    print(f"⚠️  Requested checkpoint '{name}' not found in configuration")
            validated_checkpoints = filtered_checkpoints
    
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📦 Checkpoints to process: {list(validated_checkpoints.keys())}")
    
    if not validated_checkpoints:
        print("❌ No valid checkpoints to process")
        return
    
    # Process each checkpoint
    all_extracted_data = {}
    
    for checkpoint_name, checkpoint_path in validated_checkpoints.items():
        print(f"\n🔄 Processing: {checkpoint_name}")
        print(f"📁 Path: {checkpoint_path}")
        
        try:
            extracted_data = extract_lora_matrices(checkpoint_path, args.output_dir)
            # Store only AB products and metadata to save memory
            all_extracted_data[checkpoint_name] = {
                'AB_products': extracted_data['AB_products'],
                'metadata': extracted_data['metadata']
            }
            
            # Clear memory after each checkpoint
            del extracted_data
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error processing {checkpoint_name}: {e}")
            continue
    
    # Analyze statistics
    if all_extracted_data:
        analyze_matrix_statistics(all_extracted_data)
        compare_across_checkpoints(all_extracted_data)
        
        print(f"\n✅ Successfully processed {len(all_extracted_data)} checkpoints")
        print(f"📁 All results saved to: {args.output_dir}")
        
        # Generate summary report
        summary_path = os.path.join(args.output_dir, "STARCODER27B_EXTRACTION_SUMMARY.md")
        with open(summary_path, 'w') as f:
            f.write("# StarCoder2-7B LoRA Matrix Extraction Summary\n\n")
            f.write(f"**Processed Checkpoints:** {len(all_extracted_data)}\n\n")
            
            for checkpoint_name, data in all_extracted_data.items():
                metadata = data['metadata']
                f.write(f"## {checkpoint_name}\n")
                f.write(f"- **Path:** {metadata['checkpoint_path']}\n")
                f.write(f"- **Total modules:** {metadata['num_modules']}\n")
                f.write(f"- **Attention modules:** {metadata['actual_attention_modules']}\n")
                f.write(f"- **MLP modules:** {metadata['actual_mlp_modules']}\n")
                f.write(f"- **LoRA rank:** {metadata['lora_r']}\n")
                f.write(f"- **LoRA alpha:** {metadata['lora_alpha']}\n")
                f.write(f"- **Scaling factor:** {metadata['scaling_factor']:.3f}\n")
                f.write(f"- **Target modules:** {metadata['target_modules']}\n\n")
        
        print(f"📄 Summary report saved to: {summary_path}")
        
    else:
        print("❌ No checkpoints were successfully processed")


if __name__ == "__main__":
    main()
