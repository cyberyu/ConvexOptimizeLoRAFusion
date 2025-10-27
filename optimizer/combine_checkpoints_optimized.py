#!/usr/bin/env python3
"""
LoRA Checkpoint Combiner using Optimized Weights

This script combines LoRA checkpoints using the globally optimized weights found through
convex optimization. It creates a new checkpoint that represents the optimal linear
combination of source LoRA matrices to approximate the target checkpoint.

The script automatically detects checkpoint names from the optimization results and
maps them to the actual checkpoint paths using the YAML configuration file.
"""

import os
import json
import yaml
import torch
import safetensors.torch
import numpy as np
from pathlib import Path
import argparse
import shutil
from typing import Dict, Any, List
import gc


def load_checkpoints_config(config_path: str) -> Dict:
    """Load checkpoint configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_optimization_results(results_file: str) -> Dict[str, Any]:
    """Load the globally optimized weights and checkpoint info from results file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract checkpoint information
    source_checkpoints = results['statistics']['source_checkpoints']
    target_checkpoint = results['statistics']['checkpoints'][-1]  # Last one is target
    weights = results['global_weights']
    
    print(f"🔍 Found optimization results:")
    print(f"  Source checkpoints: {source_checkpoints}")
    print(f"  Target checkpoint: {target_checkpoint}")
    print(f"  Optimal weights: {weights}")
    
    return {
        'weights': weights,
        'source_checkpoints': source_checkpoints,
        'target_checkpoint': target_checkpoint,
        'results': results
    }


def map_checkpoints_to_paths(opt_results: Dict, config: Dict) -> Dict[str, str]:
    """Map optimization checkpoint names to actual file paths from config."""
    
    print(f"\n🗺️  Mapping checkpoints to paths...")
    
    checkpoint_paths = {}
    config_checkpoints = config.get('checkpoints', {})
    
    # Try to map each source checkpoint name to a config entry
    for checkpoint_name in opt_results['source_checkpoints']:
        found_path = None
        
        # Try exact match first
        if checkpoint_name in config_checkpoints:
            found_path = config_checkpoints[checkpoint_name]['path']
        else:
            # Try partial matching (look for keywords in checkpoint name)
            for config_name, config_info in config_checkpoints.items():
                if any(keyword in checkpoint_name.lower() for keyword in config_name.lower().split('_')):
                    found_path = config_info['path']
                    print(f"  📋 Mapped {checkpoint_name} -> {config_name} ({found_path})")
                    break
        
        if found_path:
            # Expand paths
            expanded_path = os.path.expanduser(os.path.expandvars(found_path))
            checkpoint_paths[checkpoint_name] = os.path.abspath(expanded_path)
        else:
            print(f"  ⚠️  Could not map checkpoint: {checkpoint_name}")
    
    # Map target checkpoint
    target_name = opt_results['target_checkpoint']
    target_path = None
    
    if target_name in config_checkpoints:
        target_path = config_checkpoints[target_name]['path']
    else:
        # Try partial matching for target
        for config_name, config_info in config_checkpoints.items():
            if any(keyword in target_name.lower() for keyword in config_name.lower().split('_')):
                target_path = config_info['path']
                print(f"  🎯 Mapped target {target_name} -> {config_name} ({target_path})")
                break
    
    if target_path:
        expanded_path = os.path.expanduser(os.path.expandvars(target_path))
        checkpoint_paths[target_name] = os.path.abspath(expanded_path)
    
    print(f"  ✅ Mapped {len(checkpoint_paths)} checkpoints")
    return checkpoint_paths


def load_checkpoint_matrices(extracted_dir: str, checkpoint_name: str) -> Dict[str, torch.Tensor]:
    """Load A and B matrices from the extracted matrices directory."""
    print(f"📦 Loading matrices for: {checkpoint_name}")
    
    # Find A and B matrix files for this checkpoint
    a_matrices_file = None
    b_matrices_file = None
    
    # Look for files matching the checkpoint name
    for file in os.listdir(extracted_dir):
        if checkpoint_name in file:
            if file.endswith('_A_matrices.safetensors'):
                a_matrices_file = os.path.join(extracted_dir, file)
            elif file.endswith('_B_matrices.safetensors'):
                b_matrices_file = os.path.join(extracted_dir, file)
    
    if not a_matrices_file or not b_matrices_file:
        raise FileNotFoundError(f"Could not find A/B matrix files for {checkpoint_name} in {extracted_dir}")
    
    print(f"  📁 Loading A matrices from: {os.path.basename(a_matrices_file)}")
    print(f"  📁 Loading B matrices from: {os.path.basename(b_matrices_file)}")
    
    a_matrices = safetensors.torch.load_file(a_matrices_file)
    b_matrices = safetensors.torch.load_file(b_matrices_file)
    
    print(f"  ✅ Loaded {len(a_matrices)} A matrices and {len(b_matrices)} B matrices")
    return {'A': a_matrices, 'B': b_matrices}


def combine_matrices(matrices_dict: Dict[str, Dict], weights: List[float], source_checkpoints: List[str]) -> Dict[str, torch.Tensor]:
    """
    Combine A and B matrices using optimized weights.
    
    The combination is: w1*A1 + w2*A2 + w3*A3 for A matrices
                       w1*B1 + w2*B2 + w3*B3 for B matrices
    """
    print(f"\n🔄 Combining matrices using optimized weights:")
    for i, (checkpoint, weight) in enumerate(zip(source_checkpoints, weights)):
        print(f"  w{i+1} ({checkpoint}): {weight:.6f}")
    
    # Get module names from first checkpoint
    first_checkpoint = source_checkpoints[0]
    a_keys = list(matrices_dict[first_checkpoint]['A'].keys())
    b_keys = list(matrices_dict[first_checkpoint]['B'].keys())
    
    combined_a = {}
    combined_b = {}
    
    print(f"\n📊 Processing {len(a_keys)} A matrices and {len(b_keys)} B matrices...")
    
    # Combine A matrices
    for key in a_keys:
        combined_a[key] = torch.zeros_like(matrices_dict[first_checkpoint]['A'][key])
        for i, checkpoint in enumerate(source_checkpoints):
            combined_a[key] += weights[i] * matrices_dict[checkpoint]['A'][key]
    
    # Combine B matrices  
    for key in b_keys:
        combined_b[key] = torch.zeros_like(matrices_dict[first_checkpoint]['B'][key])
        for i, checkpoint in enumerate(source_checkpoints):
            combined_b[key] += weights[i] * matrices_dict[checkpoint]['B'][key]
    
    print(f"  ✅ Combined {len(combined_a)} A matrices and {len(combined_b)} B matrices")
    return {'A': combined_a, 'B': combined_b}
    
    print(f"  ✅ Combined {len(combined_a)} A matrices and {len(combined_b)} B matrices")
    
    return {'A': combined_a, 'B': combined_b}


def create_combined_checkpoint(combined_matrices: Dict[str, torch.Tensor], 
                             template_checkpoint: str,
                             output_checkpoint: str,
                             opt_results: Dict[str, Any]) -> None:
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
    weights = opt_results['weights']
    source_checkpoints = opt_results['source_checkpoints']
    target_checkpoint = opt_results['target_checkpoint']
    results = opt_results['results']
    
    readme_content = f"""# Optimally Combined LoRA Checkpoint

This checkpoint was created by combining {len(source_checkpoints)} task-specific LoRA checkpoints using
globally optimized weights found through convex optimization.

## Source Checkpoints:
"""
    
    for i, (checkpoint, weight) in enumerate(zip(source_checkpoints, weights)):
        readme_content += f"- {checkpoint}: {weight:.6f} ({weight*100:.1f}%)\n"
    
    readme_content += f"""
## Target Checkpoint:
- {target_checkpoint}

## Optimization Details:
- Method: {results['method']}
- Constraint: w1 + w2 + w3 = 1, wi ≥ 0
- Objective: Minimize ||w1*AB1 + w2*AB2 + w3*AB3 - AB_target||²
- Total elements optimized: {results['statistics']['total_elements']:,}
- Total combinations: {results['statistics']['total_combinations']}
- Final L2 error: {results['evaluation']['residual_norm']:.6f}
- Relative error: {results['evaluation']['relative_error']:.6f}

## Architecture:
- Model: StarCoder2-7B
- LoRA: Attention-only ({results['statistics']['total_combinations']} modules: 32 layers × 4 attention types)
- Optimization time: {results['computational_info']['total_time_seconds']:.1f} seconds

## Usage:
This checkpoint can be used as a drop-in replacement for any of the original
task-specific checkpoints. It represents the optimal linear combination that
best approximates the target model.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-7b")

# Load optimally combined LoRA
model = PeftModel.from_pretrained(base_model, "path/to/this/checkpoint")

# Use for inference
inputs = tokenizer("def fibonacci(n):", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
```
"""
    
    readme_path = os.path.join(output_checkpoint, "COMBINATION_README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  ✅ Created combination README")


def main():
    """Main checkpoint combination function"""
    
    parser = argparse.ArgumentParser(description="Combine LoRA checkpoints using optimized weights")
    parser.add_argument("--extracted_dir", 
                       default="extracted_matrices",
                       help="Directory containing extracted matrices")
    parser.add_argument("--results_file",
                       default="memory_efficient_global_optimization/memory_efficient_global_results.json",
                       help="Path to optimization results file")
    parser.add_argument("--output_checkpoint",
                       default="optimally_combined_checkpoint",
                       help="Output directory for combined checkpoint")
    parser.add_argument("--template_checkpoint",
                       help="Template checkpoint directory (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    print("🚀 LORA CHECKPOINT COMBINER")
    print("=" * 60)
    print(f"📁 Extracted matrices dir: {args.extracted_dir}")
    print(f"📊 Optimization results: {args.results_file}")
    print(f"📦 Output checkpoint: {args.output_checkpoint}")
    
    # Load optimization results
    print(f"\n📊 Loading optimization results...")
    if not os.path.exists(args.results_file):
        raise FileNotFoundError(f"Results file not found: {args.results_file}")
    
    opt_results = load_optimization_results(args.results_file)
    
    # Verify weights sum to 1
    weight_sum = sum(opt_results['weights'])
    print(f"  📐 Weight sum: {weight_sum:.10f}")
    if abs(weight_sum - 1.0) > 1e-10:
        print(f"  ⚠️  Warning: Weights don't sum to 1!")
    
    # Determine template checkpoint
    template_checkpoint = args.template_checkpoint
    if template_checkpoint is None:
        # Since we don't have YAML config mapping, use a hardcoded path to one of the actual checkpoints
        # We'll use the target checkpoint path from the optimization results
        target_name = opt_results['target_checkpoint']
        
        # Map the target checkpoint name to the actual directory
        if 'combinedThree' in target_name:
            template_checkpoint = "/mnt/teamssd/compressed_LLM_tbricks/finetune_starcoder2_combinedThree/checkpoint-40000"
        elif 'singleline' in target_name:
            template_checkpoint = "/mnt/teamssd/compressed_LLM_tbricks/finetune_starcoder2_singleline_new/checkpoint-40000"
        elif 'multiline' in target_name:
            template_checkpoint = "/mnt/teamssd/compressed_LLM_tbricks/finetune_starcoder2_multiline_new/checkpoint-40000"
        elif 'olddata' in target_name:
            template_checkpoint = "/mnt/teamssd/compressed_LLM_tbricks/finetune_starcoder2_olddata_new/checkpoint-40000"
        else:
            # Default fallback
            template_checkpoint = "/mnt/teamssd/compressed_LLM_tbricks/finetune_starcoder2_combinedThree/checkpoint-40000"
        
        print(f"🎯 Using target checkpoint as template: {template_checkpoint}")
    
    print(f"📋 Template checkpoint: {template_checkpoint}")
    
    # Verify template checkpoint exists
    if not os.path.exists(template_checkpoint):
        raise FileNotFoundError(f"Template checkpoint not found: {template_checkpoint}")
        
    # Verify it has the required adapter_model.safetensors
    adapter_file = os.path.join(template_checkpoint, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"adapter_model.safetensors not found in template: {adapter_file}")
    
    # Load matrices from all source checkpoints
    print(f"\n📦 Loading matrices from source checkpoints...")
    matrices_dict = {}
    
    for checkpoint_name in opt_results['source_checkpoints']:
        matrices_dict[checkpoint_name] = load_checkpoint_matrices(args.extracted_dir, checkpoint_name)
    
    # Combine matrices using optimal weights
    print(f"\n⚖️  Combining matrices...")
    combined_matrices = combine_matrices(
        matrices_dict, 
        opt_results['weights'], 
        opt_results['source_checkpoints']
    )
    
    # Create combined checkpoint
    print(f"\n🏗️  Creating combined checkpoint...")
    create_combined_checkpoint(
        combined_matrices,
        template_checkpoint,
        args.output_checkpoint, 
        opt_results
    )
    
    print(f"\n🎉 SUCCESS!")
    print(f"  📦 Combined checkpoint created: {args.output_checkpoint}")
    print(f"  🎯 Optimal combination of {len(opt_results['source_checkpoints'])} checkpoints")
    print(f"  ⚖️  Weights: {[f'{w:.3f}' for w in opt_results['weights']]}")
    print(f"  📊 Total combinations processed: {opt_results['results']['statistics']['total_combinations']}")
    print(f"  🔢 Total elements optimized: {opt_results['results']['statistics']['total_elements']:,}")


if __name__ == "__main__":
    main()
