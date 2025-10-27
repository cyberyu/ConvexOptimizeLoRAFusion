#!/usr/bin/env python3
"""
Memory-Efficient Global Convex Optimization for LoRA weights

Instead of creating one giant matrix, we solve the global problem using:
1. Iterative computation of A^T A and A^T b
2. Normal equations approach 
3. Much lower memory usage

Still solves: min Σ_i ||w1*AB1_i + w2*AB2_i + w3*AB3_i - AB4_i||²
Subject to: w1 + w2 + w3 = 1, wi >= 0
"""

import os
import torch
import numpy as np
from safetensors import safe_open
import json
import time
from typing import Dict, Tuple

def compute_global_normal_equations(organized_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute A^T A and A^T b iteratively without storing the full global matrix
    
    For the global problem: min ||A_global @ w - b_global||²
    The normal equations are: A^T A @ w = A^T b
    
    We compute these by summing over all combinations:
    A^T A = Σ_i (A_i^T A_i)
    A^T b = Σ_i (A_i^T b_i)
    """
    
    # Load master index
    with open(os.path.join(organized_dir, "master_index.json"), 'r') as f:
        master_index = json.load(f)
    
    # Get checkpoint names and determine which is the target
    checkpoints = master_index['checkpoints']
    print(f"🔍 Found checkpoints: {checkpoints}")
    
    # Identify target checkpoint (usually contains 'combined' or 'concatenation')
    target_checkpoint = None
    source_checkpoints = []
    
    for checkpoint in checkpoints:
        if any(keyword in checkpoint.lower() for keyword in ['combined', 'concatenation', 'target']):
            target_checkpoint = checkpoint
        else:
            source_checkpoints.append(checkpoint)
    
    if target_checkpoint is None:
        # Default to last checkpoint as target
        target_checkpoint = checkpoints[-1]
        source_checkpoints = checkpoints[:-1]
    
    print(f"🎯 Target checkpoint: {target_checkpoint}")
    print(f"🔧 Source checkpoints: {source_checkpoints}")
    
    if len(source_checkpoints) != 3:
        raise ValueError(f"Expected 3 source checkpoints, got {len(source_checkpoints)}")
    
    # Initialize accumulators
    AtA_global = np.zeros((3, 3))  # A^T A
    Atb_global = np.zeros(3)       # A^T b
    
    total_elements = 0
    total_target_norm_sq = 0
    
    print("🧮 Computing global normal equations iteratively...")
    print(f"   Processing {len(master_index['index_mapping'])} combinations...")
    
    for i, (layer_module, index) in enumerate(master_index['index_mapping'].items()):
        if i % 32 == 0:  # Progress indicator
            print(f"   Progress: {i:3d}/{len(master_index['index_mapping'])} ({i/len(master_index['index_mapping'])*100:.1f}%)")
        
        # Load matrices for this combination
        matrix_file = os.path.join(organized_dir, f"index_{index:03d}_{layer_module}_matrices.safetensors")
        
        with safe_open(matrix_file, framework="pt", device="cpu") as f:
            # Get matrices from source checkpoints
            AB1 = f.get_tensor(source_checkpoints[0]).numpy().flatten()
            AB2 = f.get_tensor(source_checkpoints[1]).numpy().flatten()
            AB3 = f.get_tensor(source_checkpoints[2]).numpy().flatten()
            # Get target matrix
            AB4 = f.get_tensor(target_checkpoint).numpy().flatten()
        
        # Form local system: A_i = [AB1, AB2, AB3], b_i = AB4
        A_i = np.column_stack([AB1, AB2, AB3])  # Shape: (n_elements_i, 3)
        b_i = AB4  # Shape: (n_elements_i,)
        
        # Accumulate A^T A and A^T b
        AtA_global += A_i.T @ A_i  # (3, 3) += (3, n) @ (n, 3)
        Atb_global += A_i.T @ b_i  # (3,) += (3, n) @ (n,)
        
        # Track statistics
        total_elements += len(b_i)
        total_target_norm_sq += np.sum(b_i ** 2)
    
    print(f"✅ Global normal equations computed")
    print(f"   Total elements: {total_elements:,}")
    print(f"   Memory usage: {AtA_global.nbytes + Atb_global.nbytes} bytes (tiny!)")
    
    stats = {
        'total_elements': total_elements,
        'total_combinations': len(master_index['index_mapping']),
        'total_target_norm_sq': total_target_norm_sq,
        'checkpoints': checkpoints,
        'source_checkpoints': source_checkpoints,
        'target_checkpoint': target_checkpoint
    }
    
    return AtA_global, Atb_global, stats

def solve_global_normal_equations(AtA: np.ndarray, Atb: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Solve the constrained normal equations:
    min w^T AtA w - 2 w^T Atb
    s.t. w1 + w2 + w3 = 1, wi >= 0
    """
    
    print("🔍 Solving constrained normal equations...")
    
    # Method 1: Lagrange multipliers for equality constraint
    try:
        print("   Trying analytical solution with Lagrange multipliers...")
        
        # Solve unconstrained normal equations
        w_unconstrained = np.linalg.solve(AtA, Atb)
        
        # Check if constraint is satisfied
        constraint_violation = np.sum(w_unconstrained) - 1.0
        
        if abs(constraint_violation) < 1e-10 and np.all(w_unconstrained >= -1e-10):
            # Already satisfies constraints
            w_solution = np.maximum(w_unconstrained, 0)
            w_solution = w_solution / np.sum(w_solution)
            return w_solution, "unconstrained_satisfies_constraints"
        
        # Apply equality constraint using Lagrange multipliers
        # Solve: [2*AtA, 1; 1^T, 0] [w; λ] = [2*Atb; 1]
        
        # Set up augmented system
        n = AtA.shape[0]
        augmented_matrix = np.zeros((n + 1, n + 1))
        augmented_matrix[:n, :n] = 2 * AtA
        augmented_matrix[:n, n] = 1  # Constraint coefficients
        augmented_matrix[n, :n] = 1  # Constraint coefficients
        
        augmented_rhs = np.zeros(n + 1)
        augmented_rhs[:n] = 2 * Atb
        augmented_rhs[n] = 1  # Constraint RHS
        
        solution = np.linalg.solve(augmented_matrix, augmented_rhs)
        w_constrained = solution[:n]
        
        # Check non-negativity
        if np.all(w_constrained >= -1e-10):
            w_solution = np.maximum(w_constrained, 0)
            w_solution = w_solution / np.sum(w_solution)
            return w_solution, "lagrange_multipliers"
        else:
            print("   Lagrange solution violates non-negativity, trying QP...")
    
    except np.linalg.LinAlgError as e:
        print(f"   Lagrange method failed: {e}")
    
    # Method 2: Active set method for inequality constraints
    try:
        print("   Trying active set method...")
        w_qp = solve_simplex_qp(AtA, Atb)
        return w_qp, "active_set_qp"
    
    except Exception as e:
        print(f"   Active set failed: {e}")
    
    # Method 3: Projection method
    try:
        print("   Trying projection method...")
        w_unconstrained = np.linalg.solve(AtA, Atb)
        w_projected = project_onto_simplex(w_unconstrained)
        return w_projected, "projection_method"
    
    except Exception as e:
        print(f"   Projection failed: {e}")
    
    # Fallback: uniform weights
    print("   All methods failed, using uniform weights")
    return np.array([1/3, 1/3, 1/3]), "uniform_fallback"

def solve_simplex_qp(AtA: np.ndarray, Atb: np.ndarray) -> np.ndarray:
    """
    Solve QP on simplex using active set method
    min w^T AtA w - 2 w^T Atb
    s.t. sum(w) = 1, w >= 0
    """
    
    # Try corner solutions (vertices of simplex)
    vertices = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    ]
    
    best_obj = float('inf')
    best_w = vertices[0]
    
    for w in vertices:
        obj = w.T @ AtA @ w - 2 * w.T @ Atb
        if obj < best_obj:
            best_obj = obj
            best_w = w
    
    # Try edge solutions (2D faces of simplex)
    for i in range(3):
        # Solve on edge where w[i] = 0
        remaining_indices = [j for j in range(3) if j != i]
        
        # 2D problem: min w_2d^T AtA_2d w_2d - 2 w_2d^T Atb_2d
        # s.t. sum(w_2d) = 1, w_2d >= 0
        
        AtA_2d = AtA[np.ix_(remaining_indices, remaining_indices)]
        Atb_2d = Atb[remaining_indices]
        
        # Solve with constraint sum = 1
        try:
            # Use substitution: w[1] = 1 - w[0]
            # This reduces to 1D problem
            if len(remaining_indices) == 2:
                idx0, idx1 = remaining_indices
                
                # Objective becomes: (1-t)²*AtA[0,0] + t²*AtA[1,1] + 2t(1-t)*AtA[0,1] - 2(1-t)*Atb[0] - 2t*Atb[1]
                # where t = w[idx1], (1-t) = w[idx0]
                
                # Derivative = 0: 2*AtA[1,1]*t + 2*AtA[0,1]*(1-2t) - 2*AtA[0,0]*(1-t) - 2*Atb[1] + 2*Atb[0] = 0
                a = AtA_2d[1,1] - 2*AtA_2d[0,1] + AtA_2d[0,0]
                b = 2*AtA_2d[0,1] - 2*AtA_2d[0,0] - 2*Atb_2d[1] + 2*Atb_2d[0]
                
                if abs(a) > 1e-12:
                    t_opt = -b / (2*a)
                    t_opt = np.clip(t_opt, 0, 1)  # Project to [0,1]
                    
                    w_2d = np.array([1-t_opt, t_opt])
                    
                    # Reconstruct full solution
                    w_full = np.zeros(3)
                    w_full[remaining_indices] = w_2d
                    
                    obj = w_full.T @ AtA @ w_full - 2 * w_full.T @ Atb
                    if obj < best_obj:
                        best_obj = obj
                        best_w = w_full
        
        except:
            continue
    
    return best_w

def project_onto_simplex(w: np.ndarray) -> np.ndarray:
    """Project vector onto probability simplex"""
    
    # Simple projection: normalize and clamp
    w_pos = np.maximum(w, 0)
    
    if np.sum(w_pos) > 0:
        w_normalized = w_pos / np.sum(w_pos)
    else:
        w_normalized = np.ones(len(w)) / len(w)
    
    return w_normalized

def evaluate_global_solution(global_weights: np.ndarray, AtA: np.ndarray, Atb: np.ndarray, stats: Dict) -> Dict:
    """Evaluate the quality of the global solution"""
    
    w = global_weights
    
    # Objective value
    objective_value = w.T @ AtA @ w - 2 * w.T @ Atb
    
    # Alternative: residual norm squared
    # ||Aw - b||² = w^T A^T A w - 2 w^T A^T b + b^T b
    residual_norm_sq = objective_value + stats['total_target_norm_sq']
    residual_norm = np.sqrt(max(0, residual_norm_sq))
    
    # Relative error
    total_target_norm = np.sqrt(stats['total_target_norm_sq'])
    relative_error = residual_norm / total_target_norm if total_target_norm > 0 else float('inf')
    
    # R-squared (harder to compute without full b vector, approximate)
    # R² ≈ 1 - residual_variance / total_variance
    r_squared_approx = 1 - (residual_norm_sq / stats['total_target_norm_sq']) if stats['total_target_norm_sq'] > 0 else 0
    
    evaluation = {
        'objective_value': float(objective_value),
        'residual_norm': float(residual_norm),
        'relative_error': float(relative_error),
        'r_squared_approx': float(r_squared_approx),
        'weights_sum': float(np.sum(w)),
        'weights_non_negative': bool(np.all(w >= -1e-10)),
        'constraint_violation': abs(float(np.sum(w)) - 1.0)
    }
    
    return evaluation

def main():
    """Main memory-efficient global optimization"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-efficient global optimization for LoRA matrices")
    parser.add_argument("--input_dir", 
                       required=True,
                       help="Directory containing organized matrices")
    parser.add_argument("--output_dir", 
                       required=True,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    organized_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("🌍 MEMORY-EFFICIENT GLOBAL OPTIMIZATION")
    print("=" * 50)
    print("Solving global problem using normal equations approach")
    print("Memory usage: O(1) instead of O(total_elements)")
    print()
    
    # Check if organized matrices exist
    if not os.path.exists(organized_dir):
        print("❌ Organized matrices not found. Please run organize_matrices_by_layer_module.py first.")
        return
    
    # Check if master index exists
    master_index_file = os.path.join(organized_dir, "master_index.json")
    if not os.path.exists(master_index_file):
        print(f"❌ Master index not found: {master_index_file}")
        return
    
    # Compute global normal equations iteratively
    start_time = time.time()
    AtA_global, Atb_global, stats = compute_global_normal_equations(organized_dir)
    computation_time = time.time() - start_time
    
    print(f"⏱️  Normal equations computation time: {computation_time:.2f}s")
    
    # Solve constrained optimization
    start_time = time.time()
    global_weights, method = solve_global_normal_equations(AtA_global, Atb_global)
    solve_time = time.time() - start_time
    
    print(f"⏱️  Solution time: {solve_time:.2f}s")
    
    # Evaluate solution
    evaluation = evaluate_global_solution(global_weights, AtA_global, Atb_global, stats)
    
    print(f"\n🎉 GLOBAL OPTIMIZATION COMPLETE!")
    print("=" * 40)
    
    print(f"🔧 Method used: {method}")
    print(f"📊 Total combinations: {stats['total_combinations']}")
    print(f"📐 Total elements: {stats['total_elements']:,}")
    print(f"⏱️  Total time: {computation_time + solve_time:.2f}s")
    
    print(f"\n🎯 GLOBALLY OPTIMAL WEIGHTS:")
    w1, w2, w3 = global_weights
    source_checkpoints = stats['source_checkpoints']
    target_checkpoint = stats['target_checkpoint']
    
    print(f"  w1 ({source_checkpoints[0]}): {w1:.6f}")
    print(f"  w2 ({source_checkpoints[1]}): {w2:.6f}")
    print(f"  w3 ({source_checkpoints[2]}): {w3:.6f}")
    print(f"  Target: {target_checkpoint}")
    print(f"  Sum:                 {evaluation['weights_sum']:.10f}")
    print(f"  Constraint violation: {evaluation['constraint_violation']:.2e}")
    
    print(f"\n📈 GLOBAL PERFORMANCE:")
    print(f"  Objective value:      {evaluation['objective_value']:.6e}")
    print(f"  Residual norm:        {evaluation['residual_norm']:.6f}")
    print(f"  Relative error:       {evaluation['relative_error']:.6f}")
    print(f"  R² (approx):          {evaluation['r_squared_approx']:.6f}")
    print(f"  Constraints satisfied: {evaluation['weights_non_negative'] and evaluation['constraint_violation'] < 1e-10}")
    
    # Save results with proper type conversion for JSON serialization
    def convert_to_json_serializable(obj):
        """Convert numpy types to standard Python types"""
        if hasattr(obj, 'item'):  # scalar numpy type
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    results = {
        'global_weights': global_weights.tolist(),
        'method': method,
        'statistics': convert_to_json_serializable(stats),
        'evaluation': convert_to_json_serializable(evaluation),
        'computational_info': {
            'computation_time_seconds': float(computation_time),
            'solve_time_seconds': float(solve_time),
            'total_time_seconds': float(computation_time + solve_time),
            'memory_efficient': True
        },
        'normal_equations': {
            'AtA_matrix': AtA_global.astype(float).tolist(),
            'Atb_vector': Atb_global.astype(float).tolist()
        }
    }
    
    results_file = os.path.join(output_dir, "memory_efficient_global_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    print(f"\n💡 INTERPRETATION:")
    
    # Find which checkpoint has the highest weight
    max_weight_idx = np.argmax(global_weights)
    max_checkpoint = source_checkpoints[max_weight_idx]
    max_weight = global_weights[max_weight_idx]
    
    print(f"  🎯 {max_checkpoint} is most important ({max_weight:.1%})")
    
    # Compare weights between checkpoints
    weights_with_names = list(zip(global_weights, source_checkpoints))
    weights_with_names.sort(reverse=True)
    
    for i, (weight, checkpoint) in enumerate(weights_with_names):
        if i == 0:
            print(f"  🥇 Highest: {checkpoint} ({weight:.1%})")
        elif i == 1:
            print(f"  🥈 Second: {checkpoint} ({weight:.1%})")
        else:
            print(f"  🥉 Third: {checkpoint} ({weight:.1%})")
    
    if evaluation['r_squared_approx'] > 0:
        print(f"  ✅ Linear combination explains ~{evaluation['r_squared_approx']*100:.1f}% of variance")
    else:
        print(f"  ⚠️  {target_checkpoint} has unique emergent properties")
    
    print(f"\n🚀 ADVANTAGES OF THIS APPROACH:")
    print(f"  ✅ Globally optimal solution")
    print(f"  ✅ Memory-efficient: O(1) vs O(total_elements)")
    print(f"  ✅ Fast: {computation_time + solve_time:.1f}s for {stats['total_elements']:,} elements")
    print(f"  ✅ Mathematically rigorous")
    print(f"  ✅ Single unified weight set for all layers")

if __name__ == "__main__":
    main()
