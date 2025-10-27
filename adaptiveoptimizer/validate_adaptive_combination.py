#!/usr/bin/env python3

import json
import numpy as np
import torch
import safetensors.torch
import os
from typing import Dict

def validate_adaptive_combination():
    """
    Validate that the adaptively combined checkpoint matches the optimization results.
    Compare error norms between combined matrices and target (concatenationTrained).
    """
    
    print("🔍 ADAPTIVE COMBINATION VALIDATION")
    print("=" * 50)
    
    # Load optimization results for comparison
    print("📊 Loading optimization results...")
    with open('adaptive_optimization_results/adaptive_optimization_results.json', 'r') as f:
        data = json.load(f)
    
    optimization_results = data['optimization_results']
    global_stats = data['global_statistics']
    
    expected_total_error = global_stats['total_adaptive_error']
    print(f"  Expected total error from optimization: {expected_total_error:.2e}")
    
    # Load combined matrices
    print("📦 Loading combined matrices...")
    combined_matrices = safetensors.torch.load_file(
        'adaptive_combined_checkpoint_v2/adaptive_combined_matrices.safetensors'
    )
    print(f"  Loaded {len(combined_matrices)} combined matrices")
    
    # Load master index for organized matrices access
    with open('extracted_matrices/organized_by_layer_module/master_index.json', 'r') as f:
        master_index = json.load(f)
    
    print("\n🧮 Calculating reconstruction errors...")
    
    total_error = 0.0
    modules_processed = 0
    error_breakdown = []
    
    for layer_module, opt_result in optimization_results.items():
        try:
            # Load target matrix (concatenationTrained)
            layer_index = master_index['index_mapping'][layer_module]
            safetensors_file = f"index_{layer_index:03d}_{layer_module}_matrices.safetensors"
            safetensors_path = f"extracted_matrices/organized_by_layer_module/{safetensors_file}"
            
            target_matrices = safetensors.torch.load_file(safetensors_path)
            target_matrix = target_matrices['finetune_starcoder2_combinedThree_checkpoint-40000']
            
            # Get combined matrix
            if layer_module in combined_matrices:
                combined_matrix = combined_matrices[layer_module]
            else:
                print(f"  ⚠️  Missing combined matrix for {layer_module}")
                continue
            
            # Calculate error: ||combined - target||_F^2
            diff = combined_matrix - target_matrix
            error = torch.sum(diff * diff).item()
            
            # Compare with optimization result
            expected_error = opt_result['error']
            error_diff = abs(error - expected_error)
            
            error_breakdown.append({
                'layer_module': layer_module,
                'calculated_error': error,
                'expected_error': expected_error,
                'difference': error_diff,
                'relative_diff': error_diff / expected_error if expected_error > 0 else 0
            })
            
            total_error += error
            modules_processed += 1
            
            # Show progress
            if modules_processed % 32 == 0:
                print(f"  Progress: {modules_processed}/{len(optimization_results)} modules")
                
        except Exception as e:
            print(f"  ❌ Error processing {layer_module}: {e}")
            continue
    
    print(f"\n📈 VALIDATION RESULTS:")
    print(f"  Modules processed: {modules_processed}")
    print(f"  Calculated total error: {total_error:.2e}")
    print(f"  Expected total error:   {expected_total_error:.2e}")
    print(f"  Absolute difference:    {abs(total_error - expected_total_error):.2e}")
    print(f"  Relative difference:    {abs(total_error - expected_total_error) / expected_total_error * 100:.6f}%")
    
    # Analyze individual errors
    if error_breakdown:
        max_relative_diff = max(err['relative_diff'] for err in error_breakdown)
        avg_relative_diff = np.mean([err['relative_diff'] for err in error_breakdown])
        
        print(f"\n📊 PER-MODULE ERROR ANALYSIS:")
        print(f"  Maximum relative difference: {max_relative_diff * 100:.6f}%")
        print(f"  Average relative difference: {avg_relative_diff * 100:.6f}%")
        
        # Show worst mismatches
        error_breakdown.sort(key=lambda x: x['relative_diff'], reverse=True)
        
        print(f"\n🔍 TOP 5 LARGEST MISMATCHES:")
        for i, err in enumerate(error_breakdown[:5]):
            print(f"  {i+1}. {err['layer_module']}")
            print(f"     Calculated: {err['calculated_error']:.6e}")
            print(f"     Expected:   {err['expected_error']:.6e}")
            print(f"     Rel. Diff:  {err['relative_diff'] * 100:.6f}%")
    else:
        print(f"\n❌ No modules were successfully processed!")
        max_relative_diff = 0
        avg_relative_diff = 0
    
    # Validation verdict
    print(f"\n🎯 VALIDATION VERDICT:")
    if abs(total_error - expected_total_error) / expected_total_error < 1e-10:
        print("  ✅ PERFECT MATCH! Combined matrices exactly reproduce optimization results.")
    elif abs(total_error - expected_total_error) / expected_total_error < 1e-6:
        print("  ✅ EXCELLENT MATCH! Differences are within numerical precision.")
    elif abs(total_error - expected_total_error) / expected_total_error < 1e-3:
        print("  ⚠️  GOOD MATCH with minor numerical differences.")
    else:
        print("  ❌ SIGNIFICANT MISMATCH! There may be an error in the combination logic.")
    
    # Save detailed validation results
    validation_results = {
        'validation_summary': {
            'modules_processed': modules_processed,
            'calculated_total_error': total_error,
            'expected_total_error': expected_total_error,
            'absolute_difference': abs(total_error - expected_total_error),
            'relative_difference_percent': abs(total_error - expected_total_error) / expected_total_error * 100,
            'max_relative_diff_percent': max_relative_diff * 100,
            'avg_relative_diff_percent': avg_relative_diff * 100
        },
        'per_module_errors': error_breakdown
    }
    
    with open('adaptive_combination_validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n💾 Detailed validation results saved to: adaptive_combination_validation.json")

if __name__ == "__main__":
    validate_adaptive_combination()