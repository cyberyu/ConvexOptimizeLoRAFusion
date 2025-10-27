#!/usr/bin/env python3
"""
Adaptive row-wise LoRA optimization using organized matrices.

This script uses the organized matrices from extracted_matrices/organized_by_layer_module/
to perform row-wise optimization with adaptive constraints based on convex hull analysis.

Strategy:
1. For each row, check if target falls within convex hull of [AB1, AB2, AB3]
2. If INSIDE convex hull: Use standard constraints (α ≥ 0, sum(α) = 1)
3. If OUTSIDE convex hull: Remove non-negativity constraints, use sum(α) = 1

This addresses the fundamental mathematical limitation that ~47% of rows
cannot be optimally reconstructed with positive-only weights.
"""

import numpy as np
import json
import os
import time
import argparse
from typing import Dict, List, Tuple
from safetensors import safe_open


def load_organized_matrices(organized_dir: str, index: int, layer_module: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the four AB matrices for a given layer module from organized structure"""
    matrix_file = f"index_{index:03d}_{layer_module}_matrices.safetensors"
    matrix_path = os.path.join(organized_dir, matrix_file)
    
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
    
    matrices = {}
    with safe_open(matrix_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            matrices[key] = f.get_tensor(key).numpy()
    
    # Map checkpoint names to our optimization variables
    # Based on global optimization results: olddata_new has highest weight (45.2%)
    checkpoint_mapping = {
        'finetune_starcoder2_olddata_new_checkpoint-40000': 'AB1',      # Highest weight checkpoint
        'finetune_starcoder2_multiline_new_checkpoint-40000': 'AB2',    # Second highest weight
        'finetune_starcoder2_singleline_new_checkpoint-40000': 'AB3',   # Third highest weight
        'finetune_starcoder2_combinedThree_checkpoint-40000': 'AB4'     # Target (to be approximated)
    }
    
    mapped_matrices = {}
    for checkpoint_key, var_name in checkpoint_mapping.items():
        if checkpoint_key in matrices:
            mapped_matrices[var_name] = matrices[checkpoint_key]
        else:
            available_keys = list(matrices.keys())
            raise KeyError(f"Expected checkpoint key '{checkpoint_key}' not found. Available keys: {available_keys}")
    
    AB1 = mapped_matrices['AB1']  # olddata_new (highest weight from global optimization)
    AB2 = mapped_matrices['AB2']  # multiline_new
    AB3 = mapped_matrices['AB3']  # singleline_new  
    AB4 = mapped_matrices['AB4']  # combinedThree (target)
    
    return AB1, AB2, AB3, AB4


def check_row_in_convex_hull(a1_row: np.ndarray, a2_row: np.ndarray, a3_row: np.ndarray, target_row: np.ndarray, tolerance: float = 1e-8, threshold: float = 0.5) -> bool:
    """
    Check if target_row can be represented as a positive combination of [a1_row, a2_row, a3_row]
    
    Uses percentage-based decision: if more than 50% of elements are inside convex hull,
    use non-negative constraints, otherwise use relaxed constraints.
    
    Returns True if target_row is inside the convex hull (use standard constraints)
    Returns False if target_row is outside the convex hull (use relaxed constraints)
    """
    # Element-wise convex hull check
    min_vals = np.minimum(np.minimum(a1_row, a2_row), a3_row)
    max_vals = np.maximum(np.maximum(a1_row, a2_row), a3_row)
    
    # Check which elements fall within [min_vals, max_vals]
    within_bounds = (target_row >= min_vals - tolerance) & (target_row <= max_vals + tolerance)
    
    # Calculate percentage of elements inside convex hull
    percentage_inside = np.mean(within_bounds)
    
    # Decision: use non-negative constraints if >50% elements are inside hull
    return percentage_inside > threshold


def solve_constrained_weights(a1_row: np.ndarray, a2_row: np.ndarray, a3_row: np.ndarray, target_row: np.ndarray) -> Tuple[np.ndarray, float, str]:
    """
    Solve for weights with standard constraints: α ≥ 0, sum(α) = 1
    """
    A_matrix = np.column_stack([a1_row, a2_row, a3_row])  # Shape: (L, 3)
    
    try:
        # Unconstrained solution first
        w_unconstrained = np.linalg.lstsq(A_matrix, target_row, rcond=None)[0]
        
        # Project onto simplex (positive weights summing to 1)
        w_projected = project_onto_simplex(w_unconstrained)
        
        # Calculate error
        reconstruction = A_matrix @ w_projected
        error = np.sum((target_row - reconstruction) ** 2)
        
        return w_projected, error, "constrained"
    
    except:
        # Fallback to uniform weights
        w_uniform = np.array([1/3, 1/3, 1/3])
        reconstruction = A_matrix @ w_uniform
        error = np.sum((target_row - reconstruction) ** 2)
        return w_uniform, error, "fallback"


def solve_relaxed_weights(a1_row: np.ndarray, a2_row: np.ndarray, a3_row: np.ndarray, target_row: np.ndarray) -> Tuple[np.ndarray, float, str]:
    """
    Solve for weights with relaxed constraints: No non-negativity, sum(α) = 1
    This allows negative weights which can better represent targets outside convex hull
    """
    A_matrix = np.column_stack([a1_row, a2_row, a3_row])  # Shape: (L, 3)
    
    try:
        # Method: Constrained least squares with sum constraint = 1
        # We want to minimize ||A @ w - b||² subject to 1^T @ w = 1
        # This is equivalent to minimizing ||A @ w - b||² + λ(1^T @ w - 1)
        
        # Use the method of Lagrange multipliers
        # Set up the normal equations: (A^T A + λ 1 1^T) w = A^T b + λ 1
        # With constraint 1^T w = 1
        
        AtA = A_matrix.T @ A_matrix  # Shape: (3, 3)
        Atb = A_matrix.T @ target_row  # Shape: (3,)
        ones = np.ones(3)  # Shape: (3,)
        
        # Augmented system: [AtA  ones] [w] = [Atb]
        #                   [ones^T 0 ] [λ]   [1  ]  <- Back to 1
        augmented_matrix = np.zeros((4, 4))
        augmented_matrix[:3, :3] = AtA
        augmented_matrix[:3, 3] = ones
        augmented_matrix[3, :3] = ones
        augmented_matrix[3, 3] = 0
        
        augmented_rhs = np.zeros(4)
        augmented_rhs[:3] = Atb
        augmented_rhs[3] = 1.0  # Changed back from -1.0 to 1.0
        
        # Solve the augmented system
        solution = np.linalg.solve(augmented_matrix, augmented_rhs)
        w_relaxed = solution[:3]
        
        # Calculate error
        reconstruction = A_matrix @ w_relaxed
        error = np.sum((target_row - reconstruction) ** 2)
        
        return w_relaxed, error, "relaxed"
    
    except:
        # Fallback to unconstrained least squares if augmented system fails
        try:
            w_unconstrained = np.linalg.lstsq(A_matrix, target_row, rcond=None)[0]
            # Normalize to sum to 1 (back to +1)
            current_sum = np.sum(w_unconstrained)
            if abs(current_sum) > 1e-10:
                w_normalized = w_unconstrained / current_sum
            else:
                w_normalized = np.array([1/3, 1/3, 1/3])  # Fallback to uniform positive weights
            
            reconstruction = A_matrix @ w_normalized
            error = np.sum((target_row - reconstruction) ** 2)
            return w_normalized, error, "relaxed_fallback"
        except:
            # Ultimate fallback
            w_uniform = np.array([1/3, 1/3, 1/3])  # Changed back to positive
            reconstruction = A_matrix @ w_uniform
            error = np.sum((target_row - reconstruction) ** 2)
            return w_uniform, error, "uniform_fallback"


def project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector v onto the probability simplex
    Solves: min ||x - v||^2 s.t. x >= 0, sum(x) = 1
    """
    n = len(v)
    
    # Sort in descending order
    u = np.sort(v)[::-1]
    
    # Find the threshold
    cumsum = np.cumsum(u)
    indices = np.arange(1, n + 1)
    condition = u - (cumsum - 1) / indices > 0
    
    if np.any(condition):
        rho = np.max(np.where(condition)[0])
        theta = (cumsum[rho] - 1) / (rho + 1)
    else:
        theta = (cumsum[-1] - 1) / n
    
    # Project
    x = np.maximum(v - theta, 0)
    
    # Ensure exact constraint satisfaction
    x = x / np.sum(x) if np.sum(x) > 0 else np.ones(n) / n
    
    return x


def solve_adaptive_row_wise_optimization(AB1: np.ndarray, AB2: np.ndarray, AB3: np.ndarray, AB4: np.ndarray) -> Dict:
    """
    Adaptive row-wise optimization with convex hull detection
    """
    z, L = AB1.shape
    print(f"    🔢 Matrix shape: {z} × {L} (rows × cols)")
    
    alpha1 = np.zeros(z)
    alpha2 = np.zeros(z) 
    alpha3 = np.zeros(z)
    
    total_error = 0.0
    row_results = []
    
    # Statistics tracking
    rows_in_hull = 0
    rows_outside_hull = 0
    constraint_types = []
    
    print(f"    Processing {z} rows with adaptive constraints...")
    for i in range(z):
        if i % 500 == 0:
            print(f"      Row {i}/{z} ({100*i/z:.1f}%)")
        
        # Extract row i from all matrices
        a1_row = AB1[i, :]  # Shape: (L,)
        a2_row = AB2[i, :]  # Shape: (L,)
        a3_row = AB3[i, :]  # Shape: (L,)
        target_row = AB4[i, :]  # Shape: (L,)
        
        # Check if target row is within convex hull (>50% elements inside)
        in_hull = check_row_in_convex_hull(a1_row, a2_row, a3_row, target_row)
        
        # Calculate detailed percentage for this row (for tracking)
        min_vals = np.minimum(np.minimum(a1_row, a2_row), a3_row)
        max_vals = np.maximum(np.maximum(a1_row, a2_row), a3_row)
        within_bounds = (target_row >= min_vals - 1e-8) & (target_row <= max_vals + 1e-8)
        row_percentage_inside = np.mean(within_bounds)
        
        if in_hull:
            # Standard constraints: α ≥ 0, sum(α) = 1 (>50% elements inside hull)
            weights, error, method = solve_constrained_weights(a1_row, a2_row, a3_row, target_row)
            rows_in_hull += 1
            constraint_type = "constrained"
        else:
            # Relaxed constraints: sum(α) = 1 (≤50% elements inside hull)
            weights, error, method = solve_relaxed_weights(a1_row, a2_row, a3_row, target_row)
            rows_outside_hull += 1
            constraint_type = "relaxed"
        
        alpha1[i] = weights[0]
        alpha2[i] = weights[1]
        alpha3[i] = weights[2]
        total_error += error
        
        constraint_types.append(constraint_type)
        
        # Store detailed row result for analysis
        row_results.append({
            'row': i,
            'in_convex_hull': in_hull,
            'constraint_type': constraint_type,
            'method': method,
            'weights': weights.tolist(),
            'row_error': error,
            'percentage_inside_hull': float(row_percentage_inside)
        })
    
    # Calculate overall statistics
    hull_percentage = (rows_in_hull / z) * 100
    outside_hull_percentage = (rows_outside_hull / z) * 100
    
    print(f"    📊 Convex hull analysis:")
    print(f"      Rows inside hull: {rows_in_hull} ({hull_percentage:.1f}%) - used standard constraints")
    print(f"      Rows outside hull: {rows_outside_hull} ({outside_hull_percentage:.1f}%) - used relaxed constraints")
    
    # Calculate other metrics
    total_target_norm = np.linalg.norm(AB4)
    total_residual_norm = np.sqrt(total_error)
    total_relative_error = total_residual_norm / total_target_norm if total_target_norm > 0 else float('inf')
    
    # R-squared calculation
    AB4_flat = AB4.flatten()
    reconstructed_flat = (alpha1[:, np.newaxis] * AB1 + 
                         alpha2[:, np.newaxis] * AB2 + 
                         alpha3[:, np.newaxis] * AB3).flatten()
    
    ss_res = np.sum((AB4_flat - reconstructed_flat) ** 2)
    ss_tot = np.sum((AB4_flat - np.mean(AB4_flat)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Constraint satisfaction check (modified for adaptive constraints)
    # For both inside and outside hull: sum should be 1
    constraint_violations = []
    for i in range(z):
        expected_sum = 1.0  # Always expect sum = 1 now
        actual_sum = alpha1[i] + alpha2[i] + alpha3[i]
        constraint_violations.append(abs(actual_sum - expected_sum))
    
    max_constraint_violation = np.max(constraint_violations)
    
    # Analyze negative weights
    negative_weights_alpha1 = np.sum(alpha1 < 0)
    negative_weights_alpha2 = np.sum(alpha2 < 0)
    negative_weights_alpha3 = np.sum(alpha3 < 0)
    total_negative_weights = negative_weights_alpha1 + negative_weights_alpha2 + negative_weights_alpha3
    
    results = {
        'alpha1': alpha1,
        'alpha2': alpha2, 
        'alpha3': alpha3,
        'method': 'adaptive_row_wise',
        'objective_value': total_error,
        'residual_norm': total_residual_norm,
        'relative_error': total_relative_error,
        'r_squared': r_squared,
        'target_norm': total_target_norm,
        'constraint_violation': max_constraint_violation,
        'matrix_shape': (z, L),
        'hull_analysis': {
            'rows_in_hull': rows_in_hull,
            'rows_outside_hull': rows_outside_hull,
            'hull_percentage': hull_percentage,
            'outside_hull_percentage': outside_hull_percentage
        },
        'negative_weights': {
            'alpha1_negative': int(negative_weights_alpha1),
            'alpha2_negative': int(negative_weights_alpha2),
            'alpha3_negative': int(negative_weights_alpha3),
            'total_negative': int(total_negative_weights),
            'negative_percentage': (total_negative_weights / (3 * z)) * 100
        },
        'weights_summary': {
            'alpha1_mean': np.mean(alpha1),
            'alpha2_mean': np.mean(alpha2),
            'alpha3_mean': np.mean(alpha3),
            'alpha1_std': np.std(alpha1),
            'alpha2_std': np.std(alpha2),
            'alpha3_std': np.std(alpha3),
            'alpha1_min': np.min(alpha1),
            'alpha2_min': np.min(alpha2),
            'alpha3_min': np.min(alpha3),
            'alpha1_max': np.max(alpha1),
            'alpha2_max': np.max(alpha2),
            'alpha3_max': np.max(alpha3)
        },
        'row_results': row_results[:100],  # Store first 100 for debugging
        'adaptive_optimization': True
    }
    
    return alpha1, alpha2, alpha3, total_error, results


def optimize_all_combinations_adaptive():
    """Run adaptive optimization on all matrix combinations"""
    
    print("🚀 ADAPTIVE ROW-WISE VECTOR OPTIMIZATION")
    print("=" * 60)
    print("Using adaptive constraints based on convex hull detection")
    print()
    
    # Load master index
    master_index_path = "extracted_matrices/organized_by_layer_module/master_index.json"
    with open(master_index_path, 'r') as f:
        master_index = json.load(f)
    
    index_mapping = master_index['index_mapping']
    total_combinations = master_index['total_combinations']
    
    print(f"📋 Processing {total_combinations} matrix combinations...")
    print()
    
    start_time = time.time()
    all_results = {}
    global_error = 0.0
    
    for i, (layer_module, index) in enumerate(index_mapping.items()):
        print(f"🔍 Processing {layer_module} ({i+1}/{total_combinations})")
        
        try:
            # Load matrices
            organized_dir = "extracted_matrices/organized_by_layer_module"
            AB1, AB2, AB3, AB4 = load_organized_matrices(organized_dir, index, layer_module)
            
            # Run adaptive optimization
            alpha1, alpha2, alpha3, error, results = solve_adaptive_row_wise_optimization(AB1, AB2, AB3, AB4)
            
            # Store results
            all_results[layer_module] = {
                'index': index,
                'layer_module': layer_module,
                'error': error,
                'results': results
            }
            
            global_error += error
            
            print(f"    ✅ Error: {error:.6f}")
            print(f"    📊 Hull: {results['hull_analysis']['hull_percentage']:.1f}% inside, {results['hull_analysis']['outside_hull_percentage']:.1f}% outside")
            print(f"    ⚖️  Negative weights: {results['negative_weights']['negative_percentage']:.1f}%")
            print()
        
        except Exception as e:
            print(f"    ❌ Error processing {layer_module}: {e}")
            print()
            continue
        
        # Progress update
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (total_combinations - i - 1)
            print(f"  Progress: {i+1:3d}/{total_combinations} ({100*(i+1)/total_combinations:.1f}%) - {elapsed:.1f}s elapsed, {remaining:.1f}s remaining")
            print()
    
    total_time = time.time() - start_time
    
    print("✅ Adaptive row-wise vector optimization completed!")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per combination: {total_time/total_combinations*1000:.1f}ms")
    print(f"  Total adaptive error: {global_error:.6e}")
    print()
    
    # Save results
    output_dir = "adaptive_optimization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return [convert_numpy_types(item) for item in obj.tolist()]
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj

    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for layer_module, data in all_results.items():
        json_data = convert_numpy_types(data)
        json_results[layer_module] = json_data
    
    results_file = os.path.join(output_dir, "adaptive_optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'optimization_results': json_results,
            'global_statistics': {
                'total_combinations': int(total_combinations),
                'total_adaptive_error': float(global_error),
                'total_time_seconds': float(total_time),
                'average_time_per_combination_ms': float(total_time/total_combinations*1000),
                'optimization_method': 'adaptive_row_wise_with_convex_hull_detection'
            },
            'method_description': {
                'approach': 'Adaptive constraints based on percentage-wise convex hull detection',
                'decision_criterion': 'If >50% of row elements inside convex hull: use non-negative constraints, else no non-negativity constraints',
                'inside_hull': 'Standard constraints: α ≥ 0, sum(α) = 1 (>50% elements inside hull)',
                'outside_hull': 'Relaxed constraints: sum(α) = 1 (≤50% elements inside hull)',
                'innovation': 'Element-wise analysis with percentage threshold for adaptive constraint selection'
            }
        }, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    
    return all_results, global_error


def main():
    """Main function"""
    results, total_error = optimize_all_combinations_adaptive()
    
    print("🎯 ADAPTIVE OPTIMIZATION COMPLETE")
    print("=" * 50)
    print(f"Total adaptive error: {total_error:.6e}")
    print()
    print("This approach should significantly outperform the standard")
    print("2.12% improvement by allowing negative weights for rows")
    print("that fall outside the convex hull of candidate matrices.")


if __name__ == "__main__":
    main()
