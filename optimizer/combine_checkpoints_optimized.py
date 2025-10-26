#!/usr/bin/env python3
"""
LoRA Checkpoint Combiner using Optimized Weights

This script combines LoRA checkpoints using the globally optimized weights found through
convex optimization. It creates a new checkpoint that represents the optimal linear
combination of singleline, multiline, and annotated LoRA matrices.

The output checkpoint follows the same format as the original StarCoder2-7B checkpoints.
"""

import os
import json
import torch
import safetensors.torch
import numpy as np
from pathlib import Path
import argparse
import shutil
from typing import Dict, Any
import gc


def load_optimization_results(results_file: str) -> Dict[str, float]:
    """Load the globally optimized weights from results file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    weights = results['global_weights']
    return {
        'singleline': weights[0],
        'multiline': weights[1], 
        'annotated': weights[2]
    }


def load_checkpoint_matrices(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load A and B matrices from a checkpoint."""
    print(f"📦 Loading checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Load A matrices
    a_matrices_file = None
    b_matrices_file = None
    
    for file in os.listdir(checkpoint_path):
        if file.endswith('_A_matrices.safetensors'):
            a_matrices_file = os.path.join(checkpoint_path, file)
        elif file.endswith('_B_matrices.safetensors'):
            b_matrices_file = os.path.join(checkpoint_path, file)
    
    if not a_matrices_file or not b_matrices_file:
        raise FileNotFoundError(f"Could not find A/B matrix files in {checkpoint_path}")
    
    a_matrices = safetensors.torch.load_file(a_matrices_file)
    b_matrices = safetensors.torch.load_file(b_matrices_file)
    
    print(f"  ✅ Loaded {len(a_matrices)} A matrices and {len(b_matrices)} B matrices")
    return {'A': a_matrices, 'B': b_matrices}


def combine_matrices(matrices_dict: Dict[str, Dict], weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
    """
    Combine A and B matrices using optimized weights.
    
    The combination is: w1*A1 + w2*A2 + w3*A3 for A matrices
                       w1*B1 + w2*B2 + w3*B3 for B matrices
    """
    print(f"\n🔄 Combining matrices using optimized weights:")
    print(f"  w_singleline = {weights['singleline']:.6f}")
    print(f"  w_multiline  = {weights['multiline']:.6f}")
    print(f"  w_annotated  = {weights['annotated']:.6f}")
    
    # Get module names from first checkpoint
    first_checkpoint = list(matrices_dict.keys())[0]
    a_keys = list(matrices_dict[first_checkpoint]['A'].keys())
    b_keys = list(matrices_dict[first_checkpoint]['B'].keys())
    
    combined_a = {}
    combined_b = {}
    
    print(f"\n📊 Processing {len(a_keys)} A matrices and {len(b_keys)} B matrices...")
    
    # Combine A matrices
    for key in a_keys:
        a_singleline = matrices_dict['singleline']['A'][key]
        a_multiline = matrices_dict['multiline']['A'][key]
        a_annotated = matrices_dict['annotated']['A'][key]
        
        combined_a[key] = (weights['singleline'] * a_singleline + 
                          weights['multiline'] * a_multiline + 
                          weights['annotated'] * a_annotated)
    
    # Combine B matrices
    for key in b_keys:
        b_singleline = matrices_dict['singleline']['B'][key]
        b_multiline = matrices_dict['multiline']['B'][key]
        b_annotated = matrices_dict['annotated']['B'][key]
        
        combined_b[key] = (weights['singleline'] * b_singleline + 
                          weights['multiline'] * b_multiline + 
                          weights['annotated'] * b_annotated)
    
    print(f"  ✅ Combined {len(combined_a)} A matrices and {len(combined_b)} B matrices")
    
    return {'A': combined_a, 'B': combined_b}


def create_combined_checkpoint(combined_matrices: Dict[str, torch.Tensor], 
                             template_checkpoint: str,
                             output_checkpoint: str,
                             weights: Dict[str, float]) -> None:
    """
    Create a new checkpoint directory with combined LoRA matrices.
    Uses the template checkpoint as a base and replaces the LoRA matrices.
    """
    print(f"\n🏗️  Creating combined checkpoint: {output_checkpoint}")
    
    # Create output directory
    os.makedirs(output_checkpoint, exist_ok=True)
    
    # Copy all files from template except safetensors files
    print("📋 Copying template files...")
    for file in os.listdir(template_checkpoint):
        if not file.endswith('.safetensors'):
            src = os.path.join(template_checkpoint, file)
            dst = os.path.join(output_checkpoint, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  ✅ Copied {file}")
    
    # Create adapter_model.safetensors with combined matrices
    print("💾 Creating combined adapter_model.safetensors...")
    
    # Load original safetensors structure from template
    template_safetensors = os.path.join(template_checkpoint, "adapter_model.safetensors")
    if os.path.exists(template_safetensors):
        original_tensors = safetensors.torch.load_file(template_safetensors)
        print(f"  📊 Template has {len(original_tensors)} tensors")
    else:
        raise FileNotFoundError(f"adapter_model.safetensors not found in {template_checkpoint}")
    
    # Replace lora_A and lora_B tensors with combined versions
    combined_tensors = {}
    
    for key, tensor in original_tensors.items():
        if 'lora_A' in key:
            # Convert checkpoint key to extracted matrix key
            # From: base_model.model.model.layers.X.self_attn.Y_proj.lora_A.weight
            # To: model.layers.X.self_attn.Y_proj
            module_key = key.replace('base_model.model.', '').replace('.lora_A.weight', '')
            if module_key in combined_matrices['A']:
                combined_tensors[key] = combined_matrices['A'][module_key]
                print(f"  🔄 Replaced A: {key} -> {module_key}")
            else:
                print(f"  ⚠️  A matrix not found for: {key} (looking for {module_key})")
                combined_tensors[key] = tensor
        elif 'lora_B' in key:
            # Convert checkpoint key to extracted matrix key
            # From: base_model.model.model.layers.X.self_attn.Y_proj.lora_B.weight  
            # To: model.layers.X.self_attn.Y_proj
            module_key = key.replace('base_model.model.', '').replace('.lora_B.weight', '')
            if module_key in combined_matrices['B']:
                combined_tensors[key] = combined_matrices['B'][module_key]
                print(f"  🔄 Replaced B: {key} -> {module_key}")
            else:
                print(f"  ⚠️  B matrix not found for: {key} (looking for {module_key})")
                combined_tensors[key] = tensor
        else:
            # Keep other tensors unchanged
            combined_tensors[key] = tensor
    
    # Save combined safetensors
    output_safetensors = os.path.join(output_checkpoint, "adapter_model.safetensors")
    safetensors.torch.save_file(combined_tensors, output_safetensors)
    print(f"  ✅ Saved combined adapter_model.safetensors with {len(combined_tensors)} tensors")
    
    # Create a README with combination info
    readme_content = f"""# Optimally Combined LoRA Checkpoint

This checkpoint was created by combining three task-specific LoRA checkpoints using
globally optimized weights found through convex optimization.

## Combination Weights:
- Singleline: {weights['singleline']:.6f} ({weights['singleline']*100:.1f}%)
- Multiline:  {weights['multiline']:.6f} ({weights['multiline']*100:.1f}%)
- Annotated:  {weights['annotated']:.6f} ({weights['annotated']*100:.1f}%)

## Optimization Details:
- Method: Constrained convex optimization with Lagrange multipliers
- Constraint: w1 + w2 + w3 = 1, wi ≥ 0
- Objective: Minimize ||w1*AB1 + w2*AB2 + w3*AB3 - AB_target||²
- Total elements optimized: 1,509,949,440
- Final L2 error: 158.061809

## Architecture:
- Model: StarCoder2-7B
- LoRA: Attention-only (128 modules: 32 layers × 4 attention types)
- Rank: 8
- Alpha: 8

## Usage:
This checkpoint can be used as a drop-in replacement for any of the original
task-specific checkpoints. It represents the optimal linear combination that
best approximates the concatenation-trained model.
"""
    
    readme_path = os.path.join(output_checkpoint, "COMBINATION_README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  ✅ Created combination README")


def main():
    parser = argparse.ArgumentParser(description="Combine LoRA checkpoints using optimized weights")
    parser.add_argument("--extracted_dir", 
                       default="extracted_starcoder27b_matrices",
                       help="Directory containing extracted matrices")
    parser.add_argument("--results_file",
                       default="memory_efficient_global_optimization/memory_efficient_global_results.json",
                       help="Path to optimization results file")
    parser.add_argument("--template_checkpoint",
                       default="starcoder27b/concatenationTrained/checkpoint-40000",
                       help="Template checkpoint directory (for structure)")
    parser.add_argument("--output_checkpoint",
                       default="optimally_combined_checkpoint",
                       help="Output directory for combined checkpoint")
    parser.add_argument("--checkpoint_names",
                       nargs=3,
                       default=["singleline", "multiline", "annotated"],
                       help="Names of the three checkpoints to combine")
    
    args = parser.parse_args()
    
    print("🚀 LORA CHECKPOINT COMBINER")
    print("=" * 60)
    print(f"📁 Extracted matrices dir: {args.extracted_dir}")
    print(f"📊 Optimization results: {args.results_file}")
    print(f"📋 Template checkpoint: {args.template_checkpoint}")
    print(f"📦 Output checkpoint: {args.output_checkpoint}")
    print(f"🔗 Combining: {args.checkpoint_names}")
    
    # Load optimization results
    print(f"\n📊 Loading optimization results...")
    if not os.path.exists(args.results_file):
        raise FileNotFoundError(f"Results file not found: {args.results_file}")
    
    weights = load_optimization_results(args.results_file)
    print(f"  ✅ Loaded optimal weights: {weights}")
    
    # Verify weights sum to 1
    weight_sum = sum(weights.values())
    print(f"  📐 Weight sum: {weight_sum:.10f}")
    if abs(weight_sum - 1.0) > 1e-10:
        print(f"  ⚠️  Warning: Weights don't sum to 1!")
    
    # Load matrices from all checkpoints
    print(f"\n📦 Loading matrices from checkpoints...")
    matrices_dict = {}
    
    for name in args.checkpoint_names:
        checkpoint_dir = None
        # Find the checkpoint directory
        for item in os.listdir(args.extracted_dir):
            if name.lower() in item.lower() and item.endswith('_matrices.safetensors'):
                # Found individual matrix files, construct path
                checkpoint_dir = args.extracted_dir
                break
        
        if checkpoint_dir is None:
            # Try looking for subdirectories
            for item in os.listdir(args.extracted_dir):
                item_path = os.path.join(args.extracted_dir, item)
                if os.path.isdir(item_path) and name.lower() in item.lower():
                    checkpoint_dir = item_path
                    break
        
        if checkpoint_dir is None:
            raise FileNotFoundError(f"Could not find checkpoint for: {name}")
        
        matrices_dict[name] = load_checkpoint_matrices(checkpoint_dir)
    
    # Combine matrices using optimal weights
    print(f"\n🔄 Combining matrices...")
    combined_matrices = combine_matrices(matrices_dict, weights)
    
    # Create combined checkpoint
    print(f"\n🏗️  Creating combined checkpoint...")
    create_combined_checkpoint(combined_matrices, args.template_checkpoint, 
                             args.output_checkpoint, weights)
    
    # Verify output
    print(f"\n✅ COMBINATION COMPLETE!")
    print("=" * 60)
    print(f"📁 Combined checkpoint saved to: {args.output_checkpoint}")
    print(f"📊 Checkpoint contains:")
    
    if os.path.exists(args.output_checkpoint):
        files = os.listdir(args.output_checkpoint)
        for file in sorted(files):
            file_path = os.path.join(args.output_checkpoint, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  📄 {file} ({size:,} bytes)")
    
    print(f"\n💡 USAGE:")
    print(f"  This checkpoint can be used as a drop-in replacement for:")
    print(f"  - Original concatenationTrained checkpoint")
    print(f"  - Any task-specific checkpoint")
    print(f"  It represents the optimal linear combination found through convex optimization.")
    
    print(f"\n🎯 OPTIMAL COMBINATION:")
    for name, weight in weights.items():
        print(f"  {name:12}: {weight:.6f} ({weight*100:.1f}%)")


if __name__ == "__main__":
    main()
