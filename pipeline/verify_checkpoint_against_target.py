#!/usr/bin/env python3
"""
Verify the adaptive combined checkpoint by calculating L2 norm error against target checkpoint.

This script:
1. Loads the optimized AB products from our combined checkpoint
2. Loads the target AB products from the concatenated checkpoint
3. Calculates L2 norm error for each attention layer
4. Compares with the optimization results reported error
"""

import torch
import json
import os
import numpy as np
from safetensors import safe_open
import safetensors.torch
from typing import Dict, Tuple


def load_optimized_ab_products(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load the optimized AB products from our combined checkpoint"""
    ab_products_path = os.path.join(checkpoint_dir, "combined_ab_products.safetensors")
    
    print(f"📊 Loading optimized AB products from: {ab_products_path}")
    
    ab_products = {}
    with safe_open(ab_products_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            ab_products[key] = f.get_tensor(key)
    
    print(f"  ✅ Loaded {len(ab_products)} optimized AB products")
    
    return ab_products


def load_target_checkpoint_ab_products(target_checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load AB products from the target concatenated checkpoint"""
    
    print(f"🎯 Loading target checkpoint AB products from: {target_checkpoint_path}")
    
    # Load adapter model
    adapter_path = os.path.join(target_checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter model not found: {adapter_path}")
    
    adapter_tensors = safetensors.torch.load_file(adapter_path)
    
    # Load adapter config to get scaling factor
    config_path = os.path.join(target_checkpoint_path, "adapter_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    lora_alpha = config.get('lora_alpha', 8)
    lora_r = config.get('r', 8)
    scaling_factor = lora_alpha / lora_r
    
    print(f"  📊 LoRA config: alpha={lora_alpha}, r={lora_r}, scaling={scaling_factor:.3f}")
    
    # Extract A and B matrices and compute AB products
    ab_products = {}
    a_matrices = {}
    b_matrices = {}
    
    # First, collect A and B matrices
    for key, tensor in adapter_tensors.items():
        if 'lora_A' in key:
            # Extract module path: base_model.model.model.layers.X.self_attn.Y_proj.lora_A.weight
            module_key = key.replace('base_model.model.', '').replace('.lora_A.weight', '')
            a_matrices[module_key] = tensor
        elif 'lora_B' in key:
            module_key = key.replace('base_model.model.', '').replace('.lora_B.weight', '')
            b_matrices[module_key] = tensor
    
    print(f"  📦 Found {len(a_matrices)} A matrices and {len(b_matrices)} B matrices")
    
    # Compute AB products for attention layers only
    attention_count = 0
    for module_key in a_matrices.keys():
        if 'self_attn' in module_key:  # Only attention layers
            if module_key in b_matrices:
                A_matrix = a_matrices[module_key]
                B_matrix = b_matrices[module_key]
                
                # Compute AB product with scaling
                AB_product = (B_matrix @ A_matrix) * scaling_factor
                
                # Convert to our naming convention
                # From: model.layers.0.self_attn.q_proj  
                # To: layer_00_self_attn_q_proj
                parts = module_key.split('.')
                layer_num = int(parts[2])  # layers.X
                module_type = '_'.join(parts[3:])  # self_attn.q_proj -> self_attn_q_proj
                layer_module_key = f"layer_{layer_num:02d}_{module_type}"
                
                ab_products[layer_module_key] = AB_product
                attention_count += 1
    
    print(f"  ✅ Computed {attention_count} attention layer AB products")
    
    return ab_products


def calculate_l2_errors(optimized_ab: Dict[str, torch.Tensor], 
                       target_ab: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], float]:
    """Calculate L2 norm errors between optimized and target AB products"""
    
    print(f"📐 Calculating L2 norm errors...")
    
    layer_errors = {}
    total_error = 0.0
    
    # Check which layers are in both sets
    common_layers = set(optimized_ab.keys()) & set(target_ab.keys())
    missing_optimized = set(target_ab.keys()) - set(optimized_ab.keys())
    missing_target = set(optimized_ab.keys()) - set(target_ab.keys())
    
    print(f"  📊 Common layers: {len(common_layers)}")
    if missing_optimized:
        print(f"  ⚠️  Missing in optimized: {len(missing_optimized)} layers")
        print(f"      {list(missing_optimized)[:5]}{'...' if len(missing_optimized) > 5 else ''}")
    if missing_target:
        print(f"  ⚠️  Missing in target: {len(missing_target)} layers")
        print(f"      {list(missing_target)[:5]}{'...' if len(missing_target) > 5 else ''}")
    
    # Calculate errors for common layers
    for layer_module in sorted(common_layers):
        optimized_matrix = optimized_ab[layer_module]
        target_matrix = target_ab[layer_module]
        
        # Verify shapes match
        if optimized_matrix.shape != target_matrix.shape:
            print(f"  ❌ Shape mismatch for {layer_module}: {optimized_matrix.shape} vs {target_matrix.shape}")
            continue
        
        # Calculate element-wise squared error (same as optimization)
        diff = optimized_matrix - target_matrix
        element_wise_squared_error = torch.sum(diff * diff).item()
        layer_errors[layer_module] = element_wise_squared_error
        total_error += element_wise_squared_error  # Sum of element-wise squared errors
        
        if len(layer_errors) <= 5:  # Show first 5 errors
            print(f"    ✓ {layer_module}: Element-wise squared error = {element_wise_squared_error:.6f}")
    
    print(f"  ✅ Calculated errors for {len(layer_errors)} layers")
    print(f"  📊 Total element-wise squared error: {total_error:.6f}")
    
    return layer_errors, total_error


def load_optimization_results(results_path: str) -> Dict:
    """Load the optimization results for comparison"""
    
    print(f"📋 Loading optimization results from: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    reported_error = results['global_statistics']['total_adaptive_error']
    print(f"  📈 Reported total adaptive error: {reported_error:.6f}")
    
    return results


def compare_errors(calculated_error: float, reported_error: float) -> None:
    """Compare calculated error with reported optimization error"""
    
    print(f"\n🔍 ERROR COMPARISON:")
    print(f"  📐 Calculated L2 error: {calculated_error:.6f}")
    print(f"  📋 Reported optimization error: {reported_error:.6f}")
    
    absolute_diff = abs(calculated_error - reported_error)
    relative_diff = absolute_diff / reported_error if reported_error > 0 else float('inf')
    
    print(f"  📊 Absolute difference: {absolute_diff:.6f}")
    print(f"  📊 Relative difference: {relative_diff:.6%}")
    
    # Determine if they match (within numerical tolerance)
    tolerance = 1e-6
    if absolute_diff < tolerance:
        print(f"  ✅ PERFECT MATCH! (within tolerance {tolerance})")
    elif relative_diff < 0.001:  # 0.1% tolerance
        print(f"  ✅ EXCELLENT MATCH! (within 0.1%)")
    elif relative_diff < 0.01:   # 1% tolerance
        print(f"  ⚠️  GOOD MATCH (within 1%)")
    else:
        print(f"  ❌ SIGNIFICANT DIFFERENCE! Check calculation or optimization")


def save_verification_results(layer_errors: Dict[str, float], 
                            calculated_error: float,
                            reported_error: float,
                            output_path: str) -> None:
    """Save detailed verification results"""
    
    verification_results = {
        'verification_method': 'l2_norm_error_comparison',
        'calculated_total_error': float(calculated_error),
        'reported_optimization_error': float(reported_error),
        'absolute_difference': float(abs(calculated_error - reported_error)),
        'relative_difference': float(abs(calculated_error - reported_error) / reported_error if reported_error > 0 else 0),
        'verification_status': 'VERIFIED' if abs(calculated_error - reported_error) < 1e-6 else 'MISMATCH',
        'layer_wise_errors': layer_errors,
        'error_statistics': {
            'total_layers_verified': len(layer_errors),
            'mean_layer_error': float(np.mean(list(layer_errors.values()))),
            'std_layer_error': float(np.std(list(layer_errors.values()))),
            'min_layer_error': float(np.min(list(layer_errors.values()))),
            'max_layer_error': float(np.max(list(layer_errors.values())))
        }
    }
    
    verification_file = os.path.join(output_path, "checkpoint_verification_results.json")
    with open(verification_file, 'w') as f:
        json.dump(verification_results, f, indent=2)
    
    print(f"💾 Saved verification results to: {verification_file}")


def main():
    """Main verification function"""
    
    print("🔍 ADAPTIVE COMBINED CHECKPOINT VERIFICATION")
    print("=" * 55)
    print("Verifying L2 norm error against target concatenated checkpoint")
    print()
    
    # Configuration
    optimized_checkpoint_dir = "adaptive_combined_ab_products_codegemma2b_attention_only"
    target_checkpoint_path = "/mnt/teamssd/compressed_LLM_tbricks/finetune_gemma_2b_triple_merged_attentionlayeronly/checkpoint-40000"
    optimization_results_path = "adaptive_results_codegemma2b_attention_only/adaptive_optimization_results.json"
    
    print(f"🎯 Optimized checkpoint: {optimized_checkpoint_dir}")
    print(f"📋 Target checkpoint: {target_checkpoint_path}")
    print(f"📊 Optimization results: {optimization_results_path}")
    print()
    
    try:
        # Step 1: Load optimized AB products
        optimized_ab = load_optimized_ab_products(optimized_checkpoint_dir)
        
        # Step 2: Load target checkpoint AB products
        target_ab = load_target_checkpoint_ab_products(target_checkpoint_path)
        
        # Step 3: Calculate L2 errors
        layer_errors, calculated_error = calculate_l2_errors(optimized_ab, target_ab)
        
        # Step 4: Load reported optimization error
        optimization_results = load_optimization_results(optimization_results_path)
        reported_error = optimization_results['global_statistics']['total_adaptive_error']
        
        # Step 5: Compare errors
        compare_errors(calculated_error, reported_error)
        
        # Step 6: Save verification results
        save_verification_results(layer_errors, calculated_error, reported_error, optimized_checkpoint_dir)
        
        print(f"\n✅ CHECKPOINT VERIFICATION COMPLETE!")
        print(f"📁 Verification results saved to: {optimized_checkpoint_dir}")
        
        if abs(calculated_error - reported_error) < 1e-6:
            print(f"🎯 VERIFICATION PASSED: Errors match perfectly!")
        else:
            print(f"⚠️  VERIFICATION WARNING: Errors differ by {abs(calculated_error - reported_error):.6f}")
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
