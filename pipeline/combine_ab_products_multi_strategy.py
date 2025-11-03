#!/usr/bin/env python3
"""
Multi-Strategy AB Products Combination for CodeGemma-2B

This script runs multiple optimization strategies sequentially:
1. Joint Gradient - Global differentiable optimization
2. Softmax - Analytical least squares + softmax normalization  
3. Probability Simplex - Constrained optimization with non-negative weights

Each strategy saves results to separate folders for comparison.

Usage:
    python combine_ab_products_multi_strategy.py
    python combine_ab_products_multi_strategy.py --strategies joint_gradient softmax
    python combine_ab_products_multi_strategy.py --learning_rate 0.005 --max_iterations 200
"""

import torch
import json
import os
import numpy as np
import time
import argparse
import shutil
from safetensors import safe_open
import safetensors.torch
from typing import Dict, List, Tuple
from scipy.optimize import minimize

def load_ab_products(matrices_dir: str, layer_module: str) -> Dict[str, torch.Tensor]:
    """Load AB products for a specific layer module from all checkpoints"""
    matrix_file = f"{layer_module}_matrices.safetensors"
    matrix_path = os.path.join(matrices_dir, matrix_file)
    
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
    
    matrices = {}
    with safe_open(matrix_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            matrices[key] = f.get_tensor(key)
    
    return matrices

def optimize_ab_products_row_wise(ab1: torch.Tensor, ab2: torch.Tensor, ab3: torch.Tensor,
                                 ab_target: torch.Tensor, constraint_method: str = "probability_simplex") -> Tuple[np.ndarray, float]:
    """
    Row-wise optimization with different constraint methods
    
    Args:
        ab1, ab2, ab3: Source AB matrices [rows, cols]
        ab_target: Target AB matrix [rows, cols] 
        constraint_method: 'probability_simplex', 'sum_to_one', 'softmax'
    
    Returns:
        optimized_weights: [rows, 3] weight matrix
        total_error: Total optimization error
    """
    
    rows, cols = ab1.shape
    print(f"    🔧 Row-wise AB optimization: {rows}×{cols} matrix, method: {constraint_method}")
    
    # Convert to numpy
    ab1_np = ab1.numpy()
    ab2_np = ab2.numpy()
    ab3_np = ab3.numpy()
    ab_target_np = ab_target.numpy()
    
    optimized_weights = np.zeros((rows, 3))
    total_error = 0.0
    
    for row_idx in range(rows):
        ab1_row = ab1_np[row_idx, :]
        ab2_row = ab2_np[row_idx, :]
        ab3_row = ab3_np[row_idx, :]
        target_row = ab_target_np[row_idx, :]
        
        # Objective function: minimize ||α1*AB1 + α2*AB2 + α3*AB3 - target||²
        def objective(weights):
            α1, α2, α3 = weights
            predicted = α1 * ab1_row + α2 * ab2_row + α3 * ab3_row
            return np.sum((predicted - target_row) ** 2)
        
        # Initial guess
        x0 = [1/3, 1/3, 1/3]
        
        # Set up constraints based on method
        constraints = []
        bounds = None
        
        if constraint_method == "probability_simplex":
            bounds = [(0, None), (0, None), (0, None)]  # α_i >= 0
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Σα_i = 1
        elif constraint_method == "sum_to_one":
            bounds = None  # No bounds, only sum-to-one constraint
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Σα_i = 1
        elif constraint_method == "softmax":
            # Use analytical solution for this row
            AB_matrix = np.column_stack([ab1_row, ab2_row, ab3_row])
            try:
                weights = np.linalg.lstsq(AB_matrix, target_row, rcond=None)[0]
                # Apply softmax to ensure probability simplex
                weights_exp = np.exp(weights - np.max(weights))  # Numerical stability
                weights = weights_exp / np.sum(weights_exp)
                optimized_weights[row_idx] = weights
                total_error += objective(weights)
            except:
                optimized_weights[row_idx] = [1/3, 1/3, 1/3]
                total_error += objective([1/3, 1/3, 1/3])
            continue
        else:
            raise ValueError(f"Unknown constraint method: {constraint_method}")
        
        # Optimize with chosen constraints
        result = minimize(objective, x0, method='SLSQP', 
                         constraints=constraints, bounds=bounds,
                         options={'maxiter': 100, 'ftol': 1e-8})
        
        if result.success:
            if constraint_method == "probability_simplex":
                # Ensure weights are properly normalized and non-negative
                weights = np.maximum(result.x, 0)  # Clamp to non-negative
                weights = weights / np.sum(weights)  # Normalize to sum=1
            else:
                weights = result.x
                
            optimized_weights[row_idx] = weights
            total_error += result.fun
        else:
            # Fallback to uniform weights (always satisfies constraints)
            optimized_weights[row_idx] = [1/3, 1/3, 1/3]
            total_error += objective([1/3, 1/3, 1/3])
    
    # Verify constraint satisfaction
    weight_sums = np.sum(optimized_weights, axis=1)
    min_weights = np.min(optimized_weights)
    print(f"      ✅ Row-wise optimization complete: error = {total_error:.6e}")
    print(f"      Weight constraint verification: min_weight = {min_weights:.6f}, weight_sum_range = [{np.min(weight_sums):.6f}, {np.max(weight_sums):.6f}]")
    
    return optimized_weights, total_error

def optimize_ab_products_joint_gradient(ab1: torch.Tensor, ab2: torch.Tensor, ab3: torch.Tensor,
                                       ab_target: torch.Tensor, learning_rate: float = 0.01,
                                       max_iterations: int = 100) -> Tuple[np.ndarray, float]:
    """
    Joint gradient-based optimization that directly minimizes ||α1*AB1 + α2*AB2 + α3*AB3 - AB_target||²
    
    Uses differentiable optimization to find globally optimal weights.
    """
    
    rows, cols = ab1.shape
    print(f"    🔧 Joint gradient AB optimization: {rows}×{cols} matrix")
    print(f"       Learning rate: {learning_rate}, Max iterations: {max_iterations}")
    
    # Convert to torch tensors
    ab1_t = ab1.clone().detach().requires_grad_(False)
    ab2_t = ab2.clone().detach().requires_grad_(False)
    ab3_t = ab3.clone().detach().requires_grad_(False)
    ab_target_t = ab_target.clone().detach().requires_grad_(False)
    
    # Initialize trainable parameters (log space for positivity)
    log_weights = torch.zeros(rows, 3, requires_grad=True)  # [rows, 3]
    
    optimizer = torch.optim.Adam([log_weights], lr=learning_rate)
    
    errors = []
    start_time = time.time()
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # Convert log weights to normalized weights via softmax
        weights_raw = torch.softmax(log_weights, dim=1)  # [rows, 3]
        
        # Reconstruct AB matrix row by row
        ab_reconstructed = torch.zeros_like(ab1_t)
        for row_idx in range(rows):
            α1, α2, α3 = weights_raw[row_idx, 0], weights_raw[row_idx, 1], weights_raw[row_idx, 2]
            ab_reconstructed[row_idx, :] = α1 * ab1_t[row_idx, :] + α2 * ab2_t[row_idx, :] + α3 * ab3_t[row_idx, :]
        
        # Calculate loss (joint optimization objective)
        loss = torch.sum((ab_reconstructed - ab_target_t) ** 2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        error = torch.sqrt(loss).item()
        errors.append(error)
        
        if iteration % 20 == 0:
            print(f"      Iteration {iteration}: error = {error:.6f}")
        
        # Early stopping
        if iteration > 10 and abs(errors[-1] - errors[-11]) < 1e-8:
            print(f"      Converged after {iteration + 1} iterations")
            break
    
    optimization_time = time.time() - start_time
    
    # Extract final weights
    with torch.no_grad():
        weights_final = torch.softmax(log_weights, dim=1).numpy()  # [rows, 3]
        
        # Calculate final error
        ab_reconstructed_final = torch.zeros_like(ab1_t)
        for row_idx in range(rows):
            α1, α2, α3 = weights_final[row_idx, 0], weights_final[row_idx, 1], weights_final[row_idx, 2]
            ab_reconstructed_final[row_idx, :] = α1 * ab1_t[row_idx, :] + α2 * ab2_t[row_idx, :] + α3 * ab3_t[row_idx, :]
        
        final_error = torch.sum((ab_reconstructed_final - ab_target_t) ** 2).item()
    
    print(f"      ✅ Joint optimization complete: error = {final_error:.6e}")
    print(f"      Optimization time: {optimization_time:.2f}s")
    
    return weights_final, final_error

def combine_ab_products_with_strategy(matrices_dir: str, strategy: str, **strategy_kwargs) -> Dict[str, torch.Tensor]:
    """
    Combine AB products using specified optimization strategy
    
    Args:
        matrices_dir: Directory containing extracted matrices
        strategy: 'joint_gradient', 'softmax', 'probability_simplex', 'sum_to_one'
        **strategy_kwargs: Strategy-specific parameters
    
    Returns:
        combined_ab_products: Dictionary of combined AB matrices
    """
    
    print(f"🔄 Combining AB products using {strategy.upper()} strategy...")
    print(f"  📁 Matrices directory: {matrices_dir}")
    
    # Find all matrix files
    matrix_files = [f for f in os.listdir(matrices_dir) if f.endswith('_matrices.safetensors')]
    layer_modules = [f.replace('_matrices.safetensors', '') for f in matrix_files]
    
    print(f"  📊 Found {len(layer_modules)} layer modules to process")
    
    combined_ab_products = {}
    processed_count = 0
    failed_count = 0
    total_error = 0.0
    
    for layer_module in layer_modules:
        try:
            print(f"  🔧 Processing {layer_module}...")
            
            # Load AB products for all checkpoints
            ab_matrices = load_ab_products(matrices_dir, layer_module)
            
            # Get the AB products for source checkpoints
            ab1 = ab_matrices['annotated']       # AB product from annotated checkpoint
            ab2 = ab_matrices['multiline']       # AB product from multiline checkpoint
            ab3 = ab_matrices['singleline']      # AB product from singleline checkpoint
            ab_target = ab_matrices['concatenationTrained']  # Target AB matrix
            
            # Verify all matrices have the same shape
            if not (ab1.shape == ab2.shape == ab3.shape == ab_target.shape):
                print(f"❌ Shape mismatch for {layer_module}")
                print(f"  AB shapes: {ab1.shape}, {ab2.shape}, {ab3.shape}, target: {ab_target.shape}")
                failed_count += 1
                continue
            
            print(f"    📊 Matrix shape: {ab1.shape}")
            
            # Apply optimization strategy
            if strategy == 'joint_gradient':
                learning_rate = strategy_kwargs.get('learning_rate', 0.01)
                max_iterations = strategy_kwargs.get('max_iterations', 100)
                weights, error = optimize_ab_products_joint_gradient(ab1, ab2, ab3, ab_target, learning_rate, max_iterations)
                
            elif strategy in ['softmax', 'probability_simplex', 'sum_to_one']:
                weights, error = optimize_ab_products_row_wise(ab1, ab2, ab3, ab_target, strategy)
                
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Reconstruct combined AB matrix using optimized weights
            rows, cols = ab1.shape
            combined_ab = torch.zeros_like(ab1)
            for r in range(rows):
                α1, α2, α3 = weights[r, 0], weights[r, 1], weights[r, 2]
                combined_ab[r, :] = α1 * ab1[r, :] + α2 * ab2[r, :] + α3 * ab3[r, :]
            
            # Store combined AB product
            combined_ab_products[layer_module] = combined_ab
            processed_count += 1
            total_error += error
            
            if processed_count % 20 == 0:
                print(f"  📈 Processed {processed_count}/{len(layer_modules)} modules...")
        
        except Exception as e:
            print(f"❌ Error processing {layer_module}: {e}")
            failed_count += 1
            continue
    
    print(f"✅ {strategy.upper()} strategy complete:")
    print(f"  📊 Successfully processed: {processed_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  🎯 Total optimization error: {total_error:.6e}")
    
    return combined_ab_products

def convert_ab_products_to_lora_format(combined_ab_products: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert combined AB products back to LoRA A and B matrix format with FIXED SVD"""
    
    print(f"🔄 Converting {len(combined_ab_products)} AB products to LoRA A/B format...")
    
    lora_tensors = {}
    lora_rank = 8  # Standard LoRA rank
    
    conversion_stats = {
        'total_modules': len(combined_ab_products),
        'successful_conversions': 0,
        'failed_conversions': 0,
        'rank_used': lora_rank
    }
    
    for layer_module, ab_matrix in combined_ab_products.items():
        try:
            # Convert layer_XX_self_attn_Y_proj to proper LoRA key format
            parts = layer_module.split('_')
            layer_num = int(parts[1])  # layer_XX -> XX
            module_parts = parts[2:]   # self_attn_q_proj -> [self_attn, q, proj]
            
            # Reconstruct the module path correctly
            if len(module_parts) >= 3:
                attn_type = "_".join(module_parts[:2])  # "self_attn"
                proj_name = "_".join(module_parts[2:])  # "q_proj", "k_proj", etc.
                module_path = f"base_model.model.model.layers.{layer_num}.{attn_type}.{proj_name}"
            else:
                print(f"    ⚠️  Unexpected module parts for {layer_module}: {module_parts}")
                conversion_stats['failed_conversions'] += 1
                continue
            
            # Determine effective rank based on matrix dimensions
            out_features, in_features = ab_matrix.shape
            max_possible_rank = min(out_features, in_features)
            effective_rank = min(lora_rank, max_possible_rank)
            
            if effective_rank <= 0:
                print(f"    ⚠️  Skipping {layer_module}: invalid rank {effective_rank}")
                conversion_stats['failed_conversions'] += 1
                continue
            
            # SVD decomposition with FIXED handling: AB = U @ S @ V^T ≈ B @ A
            try:
                U, S, V = torch.svd(ab_matrix)
                
                # Keep only top effective_rank singular values for LoRA rank
                rank = min(effective_rank, len(S))
                U_r = U[:, :rank]  # Shape: [d_out, rank]
                S_r = S[:rank]     # Shape: [rank]
                # FIXED: PyTorch SVD returns V, not V^T, so we need to transpose and slice correctly
                Vt_r = V[:, :rank].T  # Shape: [rank, d_in]
                
                # Distribute singular values: sqrt(S) to both A and B for stability
                sqrt_S = torch.sqrt(torch.clamp(S_r, min=1e-8))  # Avoid sqrt of zero
                
                # A matrix: [rank, d_in] (low rank to high rank)
                # B matrix: [d_out, rank] (high rank to low rank)
                lora_A = sqrt_S.unsqueeze(1) * Vt_r  # Shape: [rank, d_in]
                lora_B = U_r * sqrt_S.unsqueeze(0)   # Shape: [d_out, rank]
                
            except RuntimeError as svd_error:
                print(f"    ⚠️  SVD failed for {layer_module}: {svd_error}")
                # Fallback: use random low-rank approximation
                if out_features >= effective_rank and in_features >= effective_rank:
                    lora_A = torch.randn(effective_rank, in_features) * 0.01
                    lora_B = torch.randn(out_features, effective_rank) * 0.01
                    print(f"    🔄 Using random initialization fallback")
                else:
                    print(f"    ❌ Cannot create fallback matrices for {layer_module}")
                    conversion_stats['failed_conversions'] += 1
                    continue
            
            # Store with proper LoRA key names - ensure tensors are contiguous
            lora_a_key = f"{module_path}.lora_A.weight"
            lora_b_key = f"{module_path}.lora_B.weight"
            
            # Make tensors contiguous before storing (required for safetensors)
            lora_tensors[lora_a_key] = lora_A.contiguous()
            lora_tensors[lora_b_key] = lora_B.contiguous()
            
            conversion_stats['successful_conversions'] += 1
            
            # Debug first few conversions
            if conversion_stats['successful_conversions'] <= 3:
                print(f"    ✓ {layer_module} -> A:{lora_A.shape}, B:{lora_B.shape}")
                # Verify reconstruction
                reconstructed = lora_B @ lora_A
                reconstruction_error = torch.norm(ab_matrix - reconstructed).item()
                print(f"      Reconstruction error: {reconstruction_error:.6e}")
            
        except Exception as e:
            print(f"❌ Failed to convert {layer_module}: {e}")
            conversion_stats['failed_conversions'] += 1
            continue
    
    print(f"✅ LoRA conversion complete:")
    print(f"    Successful: {conversion_stats['successful_conversions']}")
    print(f"    Failed: {conversion_stats['failed_conversions']}")
    print(f"    Total LoRA tensors: {len(lora_tensors)}")
    
    return lora_tensors, conversion_stats

def copy_configuration_files(template_path: str, output_path: str) -> List[str]:
    """Copy all necessary configuration files from template"""
    
    print(f"📄 Copying configuration files from template...")
    
    config_files = [
        'adapter_config.json',
        'tokenizer.json',
        'tokenizer.model', 
        'tokenizer_config.json',
        'special_tokens_map.json'
    ]
    
    copied_files = []
    
    for config_file in config_files:
        src_path = os.path.join(template_path, config_file)
        dst_path = os.path.join(output_path, config_file)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_files.append(config_file)
            print(f"  ✅ Copied: {config_file}")
        else:
            print(f"  ⚠️  Not found: {config_file}")
    
    print(f"  📄 Successfully copied {len(copied_files)} configuration files")
    
    return copied_files

def save_strategy_results(combined_ab_products: Dict[str, torch.Tensor], 
                         strategy: str,
                         output_base_dir: str,
                         strategy_kwargs: Dict,
                         template_path: str = None) -> str:
    """Save results for a specific strategy to its own directory"""
    
    # Create strategy-specific output directory
    output_dir = os.path.join(output_base_dir, f"combined_ab_products_codegemma2b_{strategy}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"💾 Saving {strategy.upper()} results to: {output_dir}")
    
    # Save combined AB products
    ab_products_path = os.path.join(output_dir, "combined_ab_products.safetensors")
    safetensors.torch.save_file(combined_ab_products, ab_products_path)
    print(f"  ✅ Saved {len(combined_ab_products)} combined AB products")
    
    # Convert to LoRA format and save
    lora_tensors, conversion_stats = convert_ab_products_to_lora_format(combined_ab_products)
    adapter_model_path = os.path.join(output_dir, "adapter_model.safetensors")
    safetensors.torch.save_file(lora_tensors, adapter_model_path)
    print(f"  💾 Saved LoRA adapter model: {adapter_model_path}")
    print(f"      Contains {len(lora_tensors)} tensors ({conversion_stats['successful_conversions']} modules)")
    
    # Copy configuration files if template path provided
    copied_files = []
    if template_path and os.path.exists(template_path):
        copied_files = copy_configuration_files(template_path, output_dir)
    
    # Create strategy-specific summary
    summary = {
        'combination_method': f'{strategy}_ab_products_optimization',
        'description': f'AB products combination using {strategy} optimization strategy',
        'strategy': strategy,
        'strategy_parameters': strategy_kwargs,
        'source_checkpoints': {
            'annotated': 'finetune_codegemma_2b_AnnotatedOnly_codegemmaformat_checkpoint-40000',
            'multiline': 'finetune_codegemma_2b_MultiLineOnly_codegemmaformat_checkpoint-40000', 
            'singleline': 'finetune_codegemma_2b_SingleLineOnly_codegemmaformat_checkpoint-40000'
        },
        'target_checkpoint': 'finetune_codegemma_2b_triple_codegemmaformat_checkpoint-40000',
        'output_directory': output_dir,
        'combined_modules': {
            'total_modules': len(combined_ab_products),
            'module_list': list(combined_ab_products.keys())
        },
        'conversion_statistics': conversion_stats,
        'attention_only': True,
        'configuration_files_copied': copied_files,
        'template_checkpoint_used': template_path if template_path else None,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, f"{strategy}_combination_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  📋 Saved combination summary to: {summary_path}")
    
    # Create README
    readme_content = f"""# {strategy.upper()} Strategy Results

This directory contains the results of AB products combination using the **{strategy}** optimization strategy.

## Strategy Details
- **Method**: {strategy}
- **Parameters**: {strategy_kwargs}
- **Total Modules**: {len(combined_ab_products)}
- **Successful Conversions**: {conversion_stats['successful_conversions']}

## Files
- `adapter_model.safetensors` - LoRA checkpoint for merging with base model
- `combined_ab_products.safetensors` - Raw combined AB products  
- `{strategy}_combination_summary.json` - Detailed statistics and metadata
- Configuration files copied from template

## Usage
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained('google/codegemma-2b', torch_dtype=torch.float16)

# Load this checkpoint
model = PeftModel.from_pretrained(model, "{output_dir}")

# Merge and use
merged_model = model.merge_and_unload()
```

Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  📖 Created README: {readme_path}")
    
    return output_dir

def run_multi_strategy_optimization(strategies: List[str], 
                                  matrices_dir: str,
                                  output_base_dir: str,
                                  template_path: str = None,
                                  **strategy_kwargs) -> Dict[str, str]:
    """
    Run multiple optimization strategies sequentially
    
    Returns:
        results: Dictionary mapping strategy names to output directories
    """
    
    print(f"🚀 Running Multi-Strategy AB Products Optimization")
    print(f"📊 Strategies: {strategies}")
    print(f"📁 Matrices directory: {matrices_dir}")
    print(f"📤 Output base directory: {output_base_dir}")
    print(f"🗂️  Template checkpoint: {template_path}")
    print()
    
    results = {}
    overall_start_time = time.time()
    
    for i, strategy in enumerate(strategies, 1):
        print(f"{'='*60}")
        print(f"🔄 STRATEGY {i}/{len(strategies)}: {strategy.upper()}")
        print(f"{'='*60}")
        
        strategy_start_time = time.time()
        
        try:
            # Run optimization for this strategy
            combined_ab_products = combine_ab_products_with_strategy(
                matrices_dir, strategy, **strategy_kwargs
            )
            
            # Save results for this strategy
            output_dir = save_strategy_results(
                combined_ab_products, strategy, output_base_dir, 
                strategy_kwargs, template_path
            )
            
            results[strategy] = output_dir
            
            strategy_time = time.time() - strategy_start_time
            print(f"✅ {strategy.upper()} strategy completed in {strategy_time:.2f}s")
            print(f"📁 Results saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ {strategy.upper()} strategy failed: {e}")
            import traceback
            traceback.print_exc()
            results[strategy] = None
        
        print()
    
    overall_time = time.time() - overall_start_time
    
    print(f"{'='*60}")
    print(f"🏁 MULTI-STRATEGY OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {overall_time:.2f}s")
    print(f"📊 Results summary:")
    
    for strategy, output_dir in results.items():
        if output_dir:
            print(f"  ✅ {strategy}: {output_dir}")
        else:
            print(f"  ❌ {strategy}: FAILED")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Multi-Strategy AB Products Combination')
    
    parser.add_argument('--strategies', nargs='+', 
                       choices=['joint_gradient', 'softmax', 'probability_simplex', 'sum_to_one'],
                       default=['joint_gradient', 'softmax', 'probability_simplex'],
                       help='Optimization strategies to run')
    
    # Strategy-specific parameters
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate for joint gradient optimization')
    parser.add_argument('--max_iterations', type=int, default=100,
                       help='Maximum iterations for joint gradient optimization')
    
    # Paths
    parser.add_argument('--matrices_dir', type=str, 
                       default='gemma2b_attention_only_extracted_matrices',
                       help='Directory containing extracted matrices')
    parser.add_argument('--output_base_dir', type=str, 
                       default='.',
                       help='Base directory for output results')
    parser.add_argument('--template_path', type=str,
                       default='backup/adaptive_combined_checkpoint_codegemma2b_component_specific',
                       help='Template checkpoint for configuration files')
    
    args = parser.parse_args()
    
    # Prepare strategy parameters
    strategy_kwargs = {
        'learning_rate': args.learning_rate,
        'max_iterations': args.max_iterations
    }
    
    try:
        # Run multi-strategy optimization
        results = run_multi_strategy_optimization(
            strategies=args.strategies,
            matrices_dir=args.matrices_dir,
            output_base_dir=args.output_base_dir,
            template_path=args.template_path,
            **strategy_kwargs
        )
        
        print(f"\n🎉 Multi-strategy optimization completed successfully!")
        print(f"📈 {len([r for r in results.values() if r])} out of {len(results)} strategies succeeded")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Multi-strategy optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
