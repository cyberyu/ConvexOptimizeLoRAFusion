#!/usr/bin/env python3
"""
Memory-Efficient StableCode Matrix Organizer

This script creates an organized index of StableCode matrices without duplicating data.
Instead of copying files, it creates a lookup index that points to the original files.
"""

import os
import json
import argparse
import glob
from safetensors import safe_open
from collections import defaultdict

def discover_checkpoint_files(input_dir):
    """Discover checkpoint files and determine format"""
    
    print(f"🔍 Discovering checkpoint files in: {input_dir}")
    
    # Check for enhanced extraction format
    ab_dir = os.path.join(input_dir, "ab_products")
    enhanced_format = os.path.exists(ab_dir)
    
    if enhanced_format:
        print("📊 Enhanced extraction format detected")
        layer_pattern = os.path.join(ab_dir, "layer_*_ab_matrices.safetensors")
        layer_files = glob.glob(layer_pattern)
        
        if layer_files:
            # Load first file to discover checkpoint names
            sample_file = layer_files[0]
            with safe_open(sample_file, framework="pt", device="cpu") as f:
                checkpoint_names = list(f.keys())
            
            print(f"📦 Found {len(layer_files)} enhanced layer files")
            print(f"🏷️  Discovered checkpoint names: {checkpoint_names}")
            
            return layer_files, checkpoint_names, enhanced_format
    
    # Fallback to legacy format
    print("📦 Legacy extraction format detected")
    layer_pattern = os.path.join(input_dir, "layer_*_matrices.safetensors")
    layer_files = glob.glob(layer_pattern)
    
    if not layer_files:
        raise FileNotFoundError(f"No layer matrix files found in {input_dir}")
    
    # Load first file to discover checkpoint names
    sample_file = layer_files[0]
    with safe_open(sample_file, framework="pt", device="cpu") as f:
        checkpoint_names = list(f.keys())
    
    print(f"📦 Found {len(layer_files)} legacy layer files")
    print(f"🏷️  Discovered checkpoint names: {checkpoint_names}")
    
    return layer_files, checkpoint_names, False

def create_efficient_index(input_dir, layer_files, checkpoint_names, enhanced_format, output_dir):
    """Create an efficient index without duplicating data"""
    
    print(f"📋 Creating efficient organization index...")
    
    # Extract layer module names from files
    layer_modules = []
    file_mapping = {}
    
    for layer_file in layer_files:
        filename = os.path.basename(layer_file)
        
        if enhanced_format:
            layer_module = filename.replace("_ab_matrices.safetensors", "")
        else:
            layer_module = filename.replace("_matrices.safetensors", "")
        
        layer_modules.append(layer_module)
        file_mapping[layer_module] = {
            'ab_file': layer_file if not enhanced_format else layer_file,
            'relative_ab_path': os.path.relpath(layer_file, input_dir)
        }
        
        if enhanced_format:
            # Add paths for A and B matrices
            a_file = layer_file.replace("/ab_products/", "/a_matrices/").replace("_ab_matrices.safetensors", "_a_matrices.safetensors")
            b_file = layer_file.replace("/ab_products/", "/b_matrices/").replace("_ab_matrices.safetensors", "_b_matrices.safetensors")
            combined_file = layer_file.replace("/ab_products/", "/combined_abc/").replace("_ab_matrices.safetensors", "_all_matrices.safetensors")
            
            file_mapping[layer_module].update({
                'a_file': a_file,
                'b_file': b_file,
                'combined_file': combined_file,
                'relative_a_path': os.path.relpath(a_file, input_dir),
                'relative_b_path': os.path.relpath(b_file, input_dir),
                'relative_combined_path': os.path.relpath(combined_file, input_dir)
            })
    
    layer_modules.sort()
    print(f"📊 Found {len(layer_modules)} layer modules")
    
    # Create index mapping
    index_mapping = {layer_module: i for i, layer_module in enumerate(layer_modules)}
    
    # Sample some files to get matrix info
    print(f"🔍 Sampling matrix information...")
    
    matrix_info = {}
    sample_count = min(5, len(layer_files))
    
    for i, layer_file in enumerate(layer_files[:sample_count]):
        filename = os.path.basename(layer_file)
        
        if enhanced_format:
            layer_module = filename.replace("_ab_matrices.safetensors", "")
        else:
            layer_module = filename.replace("_matrices.safetensors", "")
        
        layer_info = {'shapes': {}, 'available_checkpoints': []}
        
        with safe_open(layer_file, framework="pt", device="cpu") as f:
            for checkpoint_name in checkpoint_names:
                if checkpoint_name in f.keys():
                    tensor = f.get_tensor(checkpoint_name)
                    layer_info['shapes'][checkpoint_name] = list(tensor.shape)
                    layer_info['available_checkpoints'].append(checkpoint_name)
        
        matrix_info[layer_module] = layer_info
        print(f"  Sampled {i+1}/{sample_count}: {layer_module}")
    
    # Create efficient organization index
    efficient_index = {
        'organization_type': 'efficient_file_index',
        'description': 'Memory-efficient organization using file references instead of data duplication',
        'input_directory': input_dir,
        'enhanced_format': enhanced_format,
        'total_layer_modules': len(layer_modules),
        'checkpoint_names': checkpoint_names,
        'matrix_types': ['A', 'B', 'AB'] if enhanced_format else ['AB'],
        'index_mapping': index_mapping,
        'file_mapping': file_mapping,
        'sample_matrix_info': matrix_info,
        'access_instructions': {
            'description': 'Use the file_mapping to locate files and index_mapping for indexed access',
            'example_usage': {
                'get_file_by_layer': 'file_mapping[layer_module][matrix_type + "_file"]',
                'get_index_by_layer': 'index_mapping[layer_module]',
                'get_layer_by_index': 'list(index_mapping.keys())[index]'
            }
        }
    }
    
    if enhanced_format:
        efficient_index['directory_structure'] = {
            'ab_products': 'AB matrices for L1/L2 optimization',
            'a_matrices': 'A matrices for SVD optimization', 
            'b_matrices': 'B matrices for SVD optimization',
            'combined_abc': 'All matrices in single files'
        }
    
    # Create output directory and save index
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, "efficient_organization_index.json")
    with open(index_path, 'w') as f:
        json.dump(efficient_index, f, indent=2)
    
    print(f"💾 Efficient index saved to: {index_path}")
    
    # Create access helper script
    create_access_helper(output_dir, efficient_index)
    
    return efficient_index

def create_access_helper(output_dir, efficient_index):
    """Create a helper script for easy access to organized matrices"""
    
    helper_script = f'''#!/usr/bin/env python3
"""
Helper script for accessing efficiently organized StableCode matrices
"""

import json
import os
from safetensors import safe_open
from typing import Dict, List, Optional

class StableCodeMatrixAccessor:
    def __init__(self, index_path: str = "efficient_organization_index.json"):
        """Initialize the accessor with the organization index"""
        with open(index_path, 'r') as f:
            self.index = json.load(f)
        
        self.input_dir = self.index['input_directory']
        self.enhanced_format = self.index['enhanced_format']
        self.checkpoint_names = self.index['checkpoint_names']
        self.file_mapping = self.index['file_mapping']
        self.index_mapping = self.index['index_mapping']
        self.reverse_index = {{v: k for k, v in self.index_mapping.items()}}
    
    def get_layer_modules(self) -> List[str]:
        """Get list of all layer modules"""
        return list(self.index_mapping.keys())
    
    def get_layer_by_index(self, index: int) -> str:
        """Get layer module name by index"""
        return self.reverse_index.get(index, None)
    
    def get_index_by_layer(self, layer_module: str) -> Optional[int]:
        """Get index by layer module name"""
        return self.index_mapping.get(layer_module, None)
    
    def load_matrices(self, layer_module: str, matrix_type: str = 'AB') -> Dict:
        """
        Load matrices for a specific layer module
        
        Args:
            layer_module: Name of the layer module (e.g., 'layer_00_attn_q_proj')
            matrix_type: Type of matrix ('AB', 'A', 'B', or 'combined')
        
        Returns:
            Dictionary mapping checkpoint names to tensors
        """
        if layer_module not in self.file_mapping:
            raise ValueError(f"Layer module not found: {{layer_module}}")
        
        file_info = self.file_mapping[layer_module]
        
        # Determine file path based on matrix type
        if matrix_type == 'AB':
            file_path = file_info['ab_file']
        elif matrix_type == 'A' and self.enhanced_format:
            file_path = file_info['a_file']
        elif matrix_type == 'B' and self.enhanced_format:
            file_path = file_info['b_file']
        elif matrix_type == 'combined' and self.enhanced_format:
            file_path = file_info['combined_file']
        else:
            available_types = ['AB'] + (['A', 'B', 'combined'] if self.enhanced_format else [])
            raise ValueError(f"Matrix type '{{matrix_type}}' not available. Available: {{available_types}}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Matrix file not found: {{file_path}}")
        
        # Load matrices
        matrices = {{}}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                matrices[key] = f.get_tensor(key)
        
        return matrices
    
    def load_matrices_by_index(self, index: int, matrix_type: str = 'AB') -> Dict:
        """Load matrices by layer index"""
        layer_module = self.get_layer_by_index(index)
        if layer_module is None:
            raise ValueError(f"Invalid index: {{index}}")
        
        return self.load_matrices(layer_module, matrix_type)
    
    def get_file_path(self, layer_module: str, matrix_type: str = 'AB') -> str:
        """Get file path for a specific layer module and matrix type"""
        if layer_module not in self.file_mapping:
            raise ValueError(f"Layer module not found: {{layer_module}}")
        
        file_info = self.file_mapping[layer_module]
        
        if matrix_type == 'AB':
            return file_info['ab_file']
        elif matrix_type == 'A' and self.enhanced_format:
            return file_info['a_file']
        elif matrix_type == 'B' and self.enhanced_format:
            return file_info['b_file']
        elif matrix_type == 'combined' and self.enhanced_format:
            return file_info['combined_file']
        else:
            available_types = ['AB'] + (['A', 'B', 'combined'] if self.enhanced_format else [])
            raise ValueError(f"Matrix type '{{matrix_type}}' not available. Available: {{available_types}}")
    
    def print_summary(self):
        """Print summary of available matrices"""
        print(f"StableCode Matrix Accessor Summary")
        print(f"=" * 40)
        print(f"Total layer modules: {{len(self.index_mapping)}}")
        print(f"Checkpoints: {{self.checkpoint_names}}")
        print(f"Enhanced format: {{self.enhanced_format}}")
        
        if self.enhanced_format:
            print(f"Available matrix types: A, B, AB, combined")
        else:
            print(f"Available matrix types: AB")
        
        print(f"\\nExample usage:")
        print(f"  accessor = StableCodeMatrixAccessor()")
        print(f"  matrices = accessor.load_matrices('layer_00_attn_q_proj', 'AB')")
        print(f"  matrices_by_idx = accessor.load_matrices_by_index(0, 'AB')")

# Example usage
if __name__ == "__main__":
    accessor = StableCodeMatrixAccessor()
    accessor.print_summary()
    
    # Example: Load first layer's AB matrices
    first_layer = accessor.get_layer_by_index(0)
    print(f"\\nLoading matrices for first layer: {{first_layer}}")
    matrices = accessor.load_matrices(first_layer, 'AB')
    print(f"Loaded matrices from checkpoints: {{list(matrices.keys())}}")
    
    for checkpoint, matrix in matrices.items():
        print(f"  {{checkpoint}}: {{matrix.shape}}")
'''
    
    helper_path = os.path.join(output_dir, "matrix_accessor.py")
    with open(helper_path, 'w') as f:
        f.write(helper_script)
    
    os.chmod(helper_path, 0o755)
    print(f"🔧 Access helper created: {helper_path}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="Create memory-efficient organization index for StableCode matrices"
    )
    parser.add_argument(
        "--input", 
        default="enhanced_extracted_stablecode_matrices",
        help="Input directory containing extracted matrices"
    )
    parser.add_argument(
        "--output", 
        default="enhanced_extracted_stablecode_matrices/efficient_organization",
        help="Output directory for organization index"
    )
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    
    print("🚀 MEMORY-EFFICIENT STABLECODE MATRIX ORGANIZATION")
    print("=" * 55)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        # Discover files
        layer_files, checkpoint_names, enhanced_format = discover_checkpoint_files(input_dir)
        
        # Create efficient index
        efficient_index = create_efficient_index(
            input_dir, layer_files, checkpoint_names, enhanced_format, output_dir
        )
        
        print(f"\\n✅ EFFICIENT ORGANIZATION COMPLETE!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🔢 Total layer modules: {efficient_index['total_layer_modules']}")
        print(f"📊 Enhanced format: {enhanced_format}")
        print(f"💾 No data duplication - references original files")
        print(f"🔧 Use matrix_accessor.py for easy access")
        
    except Exception as e:
        print(f"\\n❌ Organization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())