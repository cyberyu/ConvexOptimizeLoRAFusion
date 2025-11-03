#!/usr/bin/env python3
"""
Convex Hull Analysis for AB Product Combination

This script analyzes how many rows fall outside the convex hull and 
demonstrates the relationship between constraint methods and convex combinations.
"""

import numpy as np
import torch
from safetensors import safe_open
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

def analyze_convex_hull_violations(ab1: torch.Tensor, ab2: torch.Tensor, ab3: torch.Tensor, 
                                  ab_target: torch.Tensor) -> Dict:
    """
    Analyze how many rows of the target matrix fall outside the convex hull
    formed by the corresponding rows of the source matrices.
    """
    
    rows, cols = ab1.shape
    print(f"🔍 Analyzing convex hull violations for {rows}×{cols} matrix")
    
    # Convert to numpy
    ab1_np = ab1.numpy()
    ab2_np = ab2.numpy()
    ab3_np = ab3.numpy()
    ab_target_np = ab_target.numpy()
    
    violations = {
        'inside_convex_hull': 0,
        'outside_convex_hull': 0,
        'violation_rows': [],
        'violation_percentages': [],
        'optimal_weights': []
    }
    
    for row_idx in range(rows):
        ab1_row = ab1_np[row_idx, :]
        ab2_row = ab2_np[row_idx, :]
        ab3_row = ab3_np[row_idx, :]
        target_row = ab_target_np[row_idx, :]
        
        # Try to solve: α1*AB1 + α2*AB2 + α3*AB3 = target with convex constraints
        # This is a linear system: [AB1 AB2 AB3] * [α1; α2; α3] = target
        AB_matrix = np.column_stack([ab1_row, ab2_row, ab3_row])
        
        try:
            # Solve least squares without constraints
            unconstrained_weights = np.linalg.lstsq(AB_matrix, target_row, rcond=None)[0]
            
            # Check if solution satisfies convex hull constraints
            is_non_negative = np.all(unconstrained_weights >= -1e-6)  # Small tolerance
            sums_to_one = abs(np.sum(unconstrained_weights) - 1.0) < 1e-6
            
            if is_non_negative and sums_to_one:
                violations['inside_convex_hull'] += 1
                violations['optimal_weights'].append(unconstrained_weights)
            else:
                violations['outside_convex_hull'] += 1
                violations['violation_rows'].append(row_idx)
                
                # Calculate how much we're violating
                negative_sum = abs(np.sum(np.minimum(unconstrained_weights, 0)))
                sum_violation = abs(np.sum(unconstrained_weights) - 1.0)
                violation_score = negative_sum + sum_violation
                violations['violation_percentages'].append(violation_score)
                violations['optimal_weights'].append(unconstrained_weights)
                
        except np.linalg.LinAlgError:
            # Singular matrix - count as violation
            violations['outside_convex_hull'] += 1
            violations['violation_rows'].append(row_idx)
            violations['violation_percentages'].append(float('inf'))
            violations['optimal_weights'].append([1/3, 1/3, 1/3])
    
    # Calculate statistics
    total_rows = rows
    inside_percentage = (violations['inside_convex_hull'] / total_rows) * 100
    outside_percentage = (violations['outside_convex_hull'] / total_rows) * 100
    
    print(f"📊 Convex Hull Analysis Results:")
    print(f"   🟢 Inside convex hull: {violations['inside_convex_hull']}/{total_rows} ({inside_percentage:.1f}%)")
    print(f"   🔴 Outside convex hull: {violations['outside_convex_hull']}/{total_rows} ({outside_percentage:.1f}%)")
    
    if violations['violation_percentages']:
        avg_violation = np.mean([v for v in violations['violation_percentages'] if v != float('inf')])
        max_violation = np.max([v for v in violations['violation_percentages'] if v != float('inf')])
        print(f"   📈 Average violation magnitude: {avg_violation:.4f}")
        print(f"   📈 Maximum violation magnitude: {max_violation:.4f}")
    
    return violations

def demonstrate_constraint_methods(ab1: torch.Tensor, ab2: torch.Tensor, ab3: torch.Tensor,
                                 ab_target: torch.Tensor, sample_rows: int = 5):
    """
    Demonstrate different constraint methods on sample rows
    """
    
    print(f"\n🔬 Demonstrating constraint methods on {sample_rows} sample rows:")
    
    # Convert to numpy
    ab1_np = ab1.numpy()
    ab2_np = ab2.numpy()
    ab3_np = ab3.numpy()
    ab_target_np = ab_target.numpy()
    
    for i in range(min(sample_rows, ab1.shape[0])):
        row_idx = i * (ab1.shape[0] // sample_rows)  # Spread samples across matrix
        
        ab1_row = ab1_np[row_idx, :]
        ab2_row = ab2_np[row_idx, :]
        ab3_row = ab3_np[row_idx, :]
        target_row = ab_target_np[row_idx, :]
        
        print(f"\n  📍 Row {row_idx}:")
        
        # Unconstrained least squares
        AB_matrix = np.column_stack([ab1_row, ab2_row, ab3_row])
        try:
            unconstrained = np.linalg.lstsq(AB_matrix, target_row, rcond=None)[0]
            error_unconstrained = np.sum((AB_matrix @ unconstrained - target_row) ** 2)
            
            print(f"    🔓 Unconstrained: α=[{unconstrained[0]:.3f}, {unconstrained[1]:.3f}, {unconstrained[2]:.3f}]")
            print(f"       Sum: {np.sum(unconstrained):.3f}, Min: {np.min(unconstrained):.3f}, Error: {error_unconstrained:.6f}")
            
            # Check convex hull
            in_convex_hull = np.all(unconstrained >= 0) and abs(np.sum(unconstrained) - 1.0) < 1e-6
            print(f"       🎯 In convex hull: {'✅ YES' if in_convex_hull else '❌ NO'}")
            
        except:
            print(f"    🔓 Unconstrained: FAILED (singular matrix)")
        
        # Probability simplex (forced convex hull)
        from scipy.optimize import minimize
        
        def objective(weights):
            predicted = weights[0] * ab1_row + weights[1] * ab2_row + weights[2] * ab3_row
            return np.sum((predicted - target_row) ** 2)
        
        # Constrained optimization
        result = minimize(objective, [1/3, 1/3, 1/3], 
                         method='SLSQP',
                         bounds=[(0, None), (0, None), (0, None)],
                         constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}])
        
        if result.success:
            constrained = result.x
            error_constrained = result.fun
            print(f"    🔒 Probability Simplex: α=[{constrained[0]:.3f}, {constrained[1]:.3f}, {constrained[2]:.3f}]")
            print(f"       Sum: {np.sum(constrained):.3f}, Min: {np.min(constrained):.3f}, Error: {error_constrained:.6f}")
            print(f"       🎯 In convex hull: ✅ YES (by construction)")
        else:
            print(f"    🔒 Probability Simplex: FAILED (optimization error)")

def analyze_layer_module(matrices_dir: str, layer_module: str):
    """Analyze a specific layer module"""
    
    print(f"\n{'='*60}")
    print(f"🔍 ANALYZING LAYER MODULE: {layer_module}")
    print(f"{'='*60}")
    
    # Load AB matrices (using organized format)
    ab_products_dir = os.path.join(matrices_dir, "ab_products")
    
    # Find the file for this layer module
    for filename in os.listdir(ab_products_dir):
        if layer_module in filename and filename.endswith("_ab_matrices.safetensors"):
            matrix_path = os.path.join(ab_products_dir, filename)
            break
    else:
        print(f"❌ File not found for {layer_module}")
        return
    
    # Load matrices
    with safe_open(matrix_path, framework="pt", device="cpu") as f:
        ab1 = f.get_tensor('annotated')
        ab2 = f.get_tensor('multiline') 
        ab3 = f.get_tensor('singleline')
        ab_target = f.get_tensor('concatenationTrained')
    
    print(f"📊 Matrix shape: {ab1.shape}")
    
    # Analyze convex hull violations
    violations = analyze_convex_hull_violations(ab1, ab2, ab3, ab_target)
    
    # Demonstrate constraint methods
    demonstrate_constraint_methods(ab1, ab2, ab3, ab_target, sample_rows=3)
    
    return violations

def main():
    """Main analysis function"""
    
    matrices_dir = "enhanced_extracted_stablecode_matrices/organized_by_layer_module"
    
    # Analyze a few representative layer modules
    layer_modules = [
        "layer_00_self_attn_q_proj",  # Early layer
        "layer_15_self_attn_q_proj",  # Middle layer
        "layer_31_self_attn_q_proj"   # Late layer
    ]
    
    print("🚀 CONVEX HULL ANALYSIS FOR STABLECODE AB PRODUCTS")
    print("=" * 60)
    print("This analysis shows:")
    print("  🎯 How many rows fall outside the convex hull")
    print("  🔄 Difference between constraint methods")
    print("  📊 Impact on optimization error")
    
    all_violations = {}
    
    for layer_module in layer_modules:
        try:
            violations = analyze_layer_module(matrices_dir, layer_module)
            all_violations[layer_module] = violations
        except Exception as e:
            print(f"❌ Error analyzing {layer_module}: {e}")
    
    # Summary across all analyzed layers
    print(f"\n{'='*60}")
    print(f"📈 SUMMARY ACROSS ALL ANALYZED LAYERS")
    print(f"{'='*60}")
    
    total_inside = sum(v['inside_convex_hull'] for v in all_violations.values())
    total_outside = sum(v['outside_convex_hull'] for v in all_violations.values())
    total_rows = total_inside + total_outside
    
    if total_rows > 0:
        print(f"🎯 Overall convex hull statistics:")
        print(f"   🟢 Inside convex hull: {total_inside}/{total_rows} ({100*total_inside/total_rows:.1f}%)")
        print(f"   🔴 Outside convex hull: {total_outside}/{total_rows} ({100*total_outside/total_rows:.1f}%)")
        
        print(f"\n💡 Interpretation:")
        if total_outside / total_rows > 0.1:  # More than 10% outside
            print(f"   📊 High extrapolation needed - probability_simplex will have significant approximation error")
            print(f"   📊 sum_to_one method may provide better target approximation")
        else:
            print(f"   📊 Low extrapolation needed - probability_simplex should work well")
            print(f"   📊 Target is mostly within convex hull of source checkpoints")

if __name__ == "__main__":
    import os
    import sys
    
    # Add current directory to path for imports
    sys.path.append(os.getcwd())
    
    main()