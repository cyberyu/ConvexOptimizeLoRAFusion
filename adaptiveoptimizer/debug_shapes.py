#!/usr/bin/env python3

import json
import numpy as np
from safetensors import safe_open
import os

def debug_checkpoint_shapes():
    """Debug the shape mismatch issue by examining actual matrix shapes."""
    
    # Load adaptive optimization results
    with open('adaptive_optimization_results/adaptive_optimization_results.json', 'r') as f:
        data = json.load(f)
        adaptive_results = data['optimization_results']
    
    # Load master index
    with open('extracted_starcoder27b_matrices/organized_by_layer_module/master_index.json', 'r') as f:
        master_index = json.load(f)
    
    print("Debugging shape mismatch issue...")
    print(f"Found {len(adaptive_results)} optimized layer modules")
    print(f"Master index has {len(master_index['index_mapping'])} layer modules")
    
    # Check first layer module for shape consistency
    first_layer_module = list(adaptive_results.keys())[0]
    print(f"\n=== Debugging {first_layer_module} ===")
    
    # Get index for this layer module
    layer_index = master_index['index_mapping'][first_layer_module]
    print(f"Layer index: {layer_index}")
    
    # Construct file names
    safetensors_file = f"index_{layer_index:03d}_{first_layer_module}_matrices.safetensors"
    metadata_file = f"index_{layer_index:03d}_{first_layer_module}_metadata.json"
    
    print(f"Files: {safetensors_file}, {metadata_file}")
    
    # Load metadata
    metadata_path = f"extracted_starcoder27b_matrices/organized_by_layer_module/{metadata_file}"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Metadata shapes: {metadata['matrix_shapes']}")
    
    # Load actual matrices and check shapes
    safetensors_path = f"extracted_starcoder27b_matrices/organized_by_layer_module/{safetensors_file}"
    with safe_open(safetensors_path, framework="np") as f:
        print("Available keys in safetensors:", list(f.keys()))
        
        matrices = {}
        for checkpoint in ['annotated', 'concatenationTrained', 'multiline', 'singleline']:
            if checkpoint in f.keys():
                matrices[checkpoint] = f.get_tensor(checkpoint)
                print(f"  {checkpoint}: {matrices[checkpoint].shape}")
    
    # Check alpha vector lengths from optimization results
    opt_result = adaptive_results[first_layer_module]
    alpha1 = np.array(opt_result['results']['alpha1'])
    alpha2 = np.array(opt_result['results']['alpha2'])  
    alpha3 = np.array(opt_result['results']['alpha3'])
    
    print(f"\nAlpha vector lengths:")
    print(f"  alpha1: {len(alpha1)}")
    print(f"  alpha2: {len(alpha2)}")
    print(f"  alpha3: {len(alpha3)}")
    
    # Check if alpha lengths match matrix dimensions
    if matrices:
        first_matrix = list(matrices.values())[0]
        expected_rows = first_matrix.shape[0]
        print(f"\nMatrix analysis:")
        print(f"  Matrix shape: {first_matrix.shape}")
        print(f"  Expected rows: {expected_rows}")
        
        if len(alpha1) != expected_rows:
            print(f"  MISMATCH: alpha1 length {len(alpha1)} != matrix rows {expected_rows}")
        else:
            print(f"  OK: alpha1 length matches matrix rows")
            
        if len(alpha2) != expected_rows:
            print(f"  MISMATCH: alpha2 length {len(alpha2)} != matrix rows {expected_rows}")
        else:
            print(f"  OK: alpha2 length matches matrix rows")
            
        if len(alpha3) != expected_rows:
            print(f"  MISMATCH: alpha3 length {len(alpha3)} != matrix rows {expected_rows}")
        else:
            print(f"  OK: alpha3 length matches matrix rows")

if __name__ == "__main__":
    debug_checkpoint_shapes()
