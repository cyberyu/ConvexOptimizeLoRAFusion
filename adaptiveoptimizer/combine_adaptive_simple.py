#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
import torch
import safetensors.torch
from typing import Dict

def load_organized_matrices(organized_dir: str, layer_module: str) -> Dict[str, torch.Tensor]:
    """Load matrices for a specific layer_module from organized extracted matrices."""
    # Load master index to get the file mapping
    with open(f"{organized_dir}/master_index.json") as f:
        master_index = json.load(f)
    
    # Get index for this layer module
    layer_index = master_index['index_mapping'][layer_module]
    
    # Load matrices
    safetensors_file = f"index_{layer_index:03d}_{layer_module}_matrices.safetensors"
    safetensors_path = os.path.join(organized_dir, safetensors_file)
    
    # Load the safetensors file
    matrices = safetensors.torch.load_file(safetensors_path)
    
    return matrices

def combine_adaptive_checkpoint(results_file: str, organized_dir: str, output_dir: str):
    """Create combined checkpoint using adaptive optimization results."""
    
    print(f"🚀 ADAPTIVE CHECKPOINT COMBINER")
    print("=" * 50)
    
    # Load adaptive optimization results
    print(f"📊 Loading adaptive optimization results...")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    optimization_results = data['optimization_results']
    global_stats = data['global_statistics']
    
    print(f"  ✅ Loaded {len(optimization_results)} optimized layer modules")
    print(f"  📈 Total error: {global_stats['total_adaptive_error']:.2e}")
    print(f"  ⏱️  Average time: {global_stats['average_time_per_combination_ms']:.1f}ms")
    
    # Create combined matrices
    print(f"\n🔄 Combining matrices using adaptive weights...")
    combined_matrices = {}
    modules_processed = 0
    
    for layer_module, opt_result in optimization_results.items():
        try:
            # Load matrices for this layer module
            matrices = load_organized_matrices(organized_dir, layer_module)
            
            # Extract alpha vectors from optimization results
            alpha1 = np.array(opt_result['results']['alpha1'])
            alpha2 = np.array(opt_result['results']['alpha2'])  
            alpha3 = np.array(opt_result['results']['alpha3'])
            
            # Get matrices for each checkpoint (same mapping as optimization)
            # alpha1 → annotated, alpha2 → multiline, alpha3 → singleline
            matrix_annotated = matrices['annotated']      # α1
            matrix_multiline = matrices['multiline']      # α2
            matrix_singleline = matrices['singleline']    # α3
            
            # Convert to numpy for computation
            m1_np = matrix_annotated.numpy()    # α1 matrices
            m2_np = matrix_multiline.numpy()    # α2 matrices  
            m3_np = matrix_singleline.numpy()   # α3 matrices
            
            # Verify shapes match
            z, L = m1_np.shape
            if len(alpha1) != z:
                print(f"  ⚠️  Shape mismatch for {layer_module}: alpha length {len(alpha1)} vs matrix rows {z}")
                continue
                
            # Combine matrices row-wise using adaptive weights
            combined_np = np.zeros_like(m1_np)
            for i in range(z):
                combined_np[i, :] = (alpha1[i] * m1_np[i, :] + 
                                   alpha2[i] * m2_np[i, :] + 
                                   alpha3[i] * m3_np[i, :])
            
            # Convert back to tensor and store
            combined_matrices[layer_module] = torch.from_numpy(combined_np)
            
            modules_processed += 1
            if modules_processed % 32 == 0:
                print(f"  Progress: {modules_processed}/{len(optimization_results)} modules")
                
        except Exception as e:
            print(f"  ❌ Error processing {layer_module}: {e}")
            continue
    
    print(f"✅ Successfully combined {modules_processed} modules")
    
    # Save combined matrices
    print(f"\n💾 Saving combined checkpoint...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as safetensors
    output_file = os.path.join(output_dir, "adaptive_combined_matrices.safetensors")
    safetensors.torch.save_file(combined_matrices, output_file)
    
    # Save metadata
    metadata = {
        "combination_method": "adaptive_row_wise_vector_weights",
        "source_checkpoints": ["annotated", "multiline", "singleline"],  # α1, α2, α3 order
        "total_modules": modules_processed,
        "optimization_error": global_stats['total_adaptive_error'],
        "convex_hull_threshold": 0.5,
        "adaptive_constraint_switching": True
    }
    
    metadata_file = os.path.join(output_dir, "combination_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✅ Saved combined matrices: {output_file}")
    print(f"  ✅ Saved metadata: {metadata_file}")
    print(f"\n🎯 COMBINATION COMPLETE!")

def main():
    parser = argparse.ArgumentParser(description="Combine LoRA checkpoints using adaptive optimization results")
    parser.add_argument("--results_file", required=True, help="Path to adaptive optimization results JSON")
    parser.add_argument("--organized_dir", default="extracted_starcoder27b_matrices/organized_by_layer_module", 
                       help="Directory with organized extracted matrices")
    parser.add_argument("--output", required=True, help="Output directory for combined checkpoint")
    
    args = parser.parse_args()
    
    combine_adaptive_checkpoint(args.results_file, args.organized_dir, args.output)

if __name__ == "__main__":
    main()
