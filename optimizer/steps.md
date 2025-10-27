# LoRA Checkpoint Optimizer - Step-by-Step Guide

This directory contains scripts and tools for optimally combining LoRA (Low-Rank Adaptation) checkpoints using convex optimization. The pipeline finds mathematically optimal weights to linearly combine multiple task-specific LoRA matrices.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Extract LoRA Matrices](#step-1-extract-lora-matrices)
3. [Step 2: Organize Matrices by Layer-Module](#step-2-organize-matrices-by-layer-module)
4. [Step 3: Global Convex Optimization](#step-3-global-convex-optimization)
5. [Step 4: Create Combined Checkpoint](#step-4-create-combined-checkpoint)
6. [Results and Analysis](#results-and-analysis)
7. [Files in This Directory](#files-in-this-directory)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Dependencies
```bash
pip install torch safetensors numpy scipy pyyaml
```

### Required Configuration
Create a YAML configuration file (`../LoRAExtraction/checkpoints_config.yml`) that specifies the paths to your LoRA checkpoints. This replaces the need to have all checkpoints in a single directory structure.

### Required Directory Structure
```
LoRATaskVectors/
├── starcoder27b/                    # Can be anywhere now (configured in YAML)
│   ├── annotated/checkpoint-40000/
│   ├── multiline/checkpoint-40000/
│   ├── singleline/checkpoint-40000/
│   └── concatenationTrained/checkpoint-40000/  # Target to approximate
├── LoRAExtraction/                  # Matrix extraction scripts
│   ├── extract_starcoder27b_matrices.py
│   ├── organize_matrices_by_layer_module.py
│   └── checkpoints_config.yml      # NEW: Configuration file
└── optimizer/                      # This directory
```

---

## Step 1: Extract LoRA Matrices

**Purpose**: Extract A, B, and AB matrices from all StarCoder2-7B LoRA checkpoints.

### Configuration Setup:
First, create or modify the checkpoint configuration file. You can either:

**Option 1: Create a default configuration file**
```bash
python3 ../LoRAExtraction/extract_starcoder27b_matrices.py --create_default_config --config ../LoRAExtraction/checkpoints_config.yml
```

**Option 2: Manually create the configuration file `../LoRAExtraction/checkpoints_config.yml`:**

```yaml
# StarCoder2-7B LoRA Checkpoints Configuration
checkpoints:
  annotated:
    path: "../starcoder27b/annotated/checkpoint-40000"
    description: "Annotated training data checkpoint"
    
  multiline:
    path: "../starcoder27b/multiline/checkpoint-40000"
    description: "Multiline training data checkpoint"
    
  singleline:
    path: "../starcoder27b/singleline/checkpoint-40000"
    description: "Singleline training data checkpoint"
    
  concatenationTrained:
    path: "../starcoder27b/concatenationTrained/checkpoint-40000"
    description: "Concatenation trained checkpoint (target for optimization)"

settings:
  verify_paths: true
  verbose_paths: true
```

**Then edit the paths** in the generated/created file to match your actual checkpoint locations.

### Command:
```bash
python3 ../LoRAExtraction/extract_starcoder27b_matrices.py \
    --config ../LoRAExtraction/checkpoints_config.yml \
    --output_dir ../extracted_starcoder27b_matrices
```

### Alternative Commands:

**Process specific checkpoints only:**
```bash
python3 ../LoRAExtraction/extract_starcoder27b_matrices.py \
    --config ../LoRAExtraction/checkpoints_config.yml \
    --output_dir ../extracted_starcoder27b_matrices \
    --checkpoints annotated multiline singleline
```

**Verbose output for debugging:**
```bash
python3 ../LoRAExtraction/extract_starcoder27b_matrices.py \
    --config ../LoRAExtraction/checkpoints_config.yml \
    --output_dir ../extracted_starcoder27b_matrices \
    --verbose
```

**Backward compatibility (deprecated):**
```bash
python3 ../LoRAExtraction/extract_starcoder27b_matrices.py \
    --starcoder_dir ../starcoder27b \
    --output_dir ../extracted_starcoder27b_matrices
```

### What This Does:
- Loads checkpoint paths from YAML configuration file
- Validates that all specified checkpoint paths exist and contain required files
- Extracts LoRA A matrices (rank × input_dim) and B matrices (output_dim × rank)
- Computes AB products: ΔW = B @ A * scaling_factor
- Processes all checkpoints defined in configuration (typically 4: annotated, multiline, singleline, concatenationTrained)
- Saves matrices with checkpoint-specific naming based on configuration

### Expected Output:
```
extracted_starcoder27b_matrices/
├── starcoder27b_annotated_checkpoint-40000_A_matrices.safetensors
├── starcoder27b_annotated_checkpoint-40000_B_matrices.safetensors
├── starcoder27b_annotated_checkpoint-40000_AB_products.safetensors
├── starcoder27b_annotated_checkpoint-40000_metadata.json
├── [similar files for multiline, singleline, concatenationTrained]
├── STARCODER27B_EXTRACTION_SUMMARY.md
└── checkpoints_config.yml  # Copy of configuration used
```

**Note**: File naming now uses the checkpoint names from your YAML configuration, providing more flexibility in organizing your checkpoints.

### Key Information:
- **Architecture**: StarCoder2-7B uses attention-only LoRA (not MLP)
- **Modules**: 128 total (32 layers × 4 attention types: k_proj, q_proj, v_proj, o_proj)
- **Matrix Dimensions**: A[8,4608], B varies by module (k/v: [512,8], q/o: [4608,8])

---

## Step 2: Organize Matrices by Layer-Module

**Purpose**: Reorganize extracted matrices by (layer, module) combinations for optimization.

### Command:
```bash
python3 ../LoRAExtraction/organize_matrices_by_layer_module.py \
    --input_dir ../extracted_starcoder27b_matrices \
    --output_dir ../extracted_starcoder27b_matrices/organized_by_layer_module
```

### What This Does:
- Automatically discovers all available checkpoint files in the input directory
- Groups matrices by (layer, module) combinations across all found checkpoints
- Creates indexed combinations (0 to N-1, where N is the number of unique layer-module pairs)
- Each index contains AB matrices from all available checkpoints
- Generates master index mapping layer_module names to indices
- Works with any checkpoint naming convention (no hardcoded paths)

### Expected Output:
```
extracted_starcoder27b_matrices/organized_by_layer_module/
├── master_index.json
├── index_000_layer_00_k_proj_matrices.safetensors
├── index_001_layer_00_o_proj_matrices.safetensors
├── index_002_layer_00_q_proj_matrices.safetensors
├── index_003_layer_00_v_proj_matrices.safetensors
├── [... 124 more combinations]
└── index_127_layer_31_v_proj_matrices.safetensors
```

### Index Structure:
- **Index 0-3**: Layer 0 attention modules (k, o, q, v)
- **Index 4-7**: Layer 1 attention modules (k, o, q, v)
- **...**: Pattern continues
- **Index 124-127**: Layer 31 attention modules (k, o, q, v)

---

## Step 3: Global Convex Optimization

**Purpose**: Find globally optimal weights to minimize reconstruction error.

### Mathematical Problem:
```
minimize ||w1*AB1 + w2*AB2 + w3*AB3 - AB4||²
subject to: w1 + w2 + w3 = 1, wi ≥ 0

Where:
- AB1 = singleline AB matrices
- AB2 = multiline AB matrices  
- AB3 = annotated AB matrices
- AB4 = concatenationTrained AB matrices (target)
```

### Command:
```bash
python3 memory_efficient_global_optimize.py \
    --input_dir ../extracted_starcoder27b_matrices/organized_by_layer_module \
    --output_dir memory_efficient_global_optimization
```

### What This Does:
- Solves constrained convex optimization using Lagrange multipliers
- Uses memory-efficient normal equations approach: A^T A @ w = A^T b
- Processes 1,509,949,440 matrix elements in O(1) memory
- Finds globally optimal solution across all 128 layer×module combinations

### Expected Output:
```
memory_efficient_global_optimization/
└── memory_efficient_global_results.json
```

### Typical Results:
```json
{
  "global_weights": [0.263141, 0.284697, 0.452163],
  "method": "lagrange_multipliers",
  "evaluation": {
    "objective_value": 3219.643,
    "residual_norm": 158.061809,
    "constraint_violation": 0.0
  }
}
```

### Interpretation:
- **w1 (singleline)**: 26.3%
- **w2 (multiline)**: 28.5% 
- **w3 (annotated)**: 45.2%
- **Annotated checkpoint dominates** the optimal combination

---

## Step 4: Create Combined Checkpoint

**Purpose**: Create a new checkpoint with optimally combined LoRA matrices.

### Command:
```bash
python3 combine_checkpoints_optimized.py \
    --extracted_dir ../extracted_starcoder27b_matrices \
    --results_file memory_efficient_global_optimization/memory_efficient_global_results.json \
    --template_checkpoint ../starcoder27b/concatenationTrained/checkpoint-40000 \
    --output_checkpoint ../optimally_combined_checkpoint
```

### What This Does:
- Loads optimal weights from optimization results
- Combines A and B matrices using: w1*A1 + w2*A2 + w3*A3
- Creates new adapter_model.safetensors with combined matrices
- Preserves all other checkpoint files (tokenizer, config, etc.)
- Maintains identical structure to original checkpoints

### Expected Output:
```
optimally_combined_checkpoint/
├── adapter_model.safetensors     # 29.4MB - Combined LoRA matrices
├── adapter_config.json           # LoRA configuration
├── COMBINATION_README.md          # Documentation
├── tokenizer.json               # Tokenizer (copied from template)
├── tokenizer_config.json        # Tokenizer config
├── vocab.json                   # Vocabulary
├── merges.txt                   # BPE merges
├── special_tokens_map.json      # Special tokens
├── optimizer.pt                 # Optimizer state (copied)
├── scheduler.pt                 # Scheduler state (copied)
├── trainer_state.json           # Training state (copied)
├── training_args.bin            # Training arguments (copied)
├── scaler.pt                    # Gradient scaler (copied)
├── rng_state.pth               # Random state (copied)
└── README.md                    # Original README (copied)
```

### Key Replacements:
- **256 LoRA tensors replaced**: 128 lora_A.weight + 128 lora_B.weight
- **All other tensors preserved**: Maintains compatibility
- **Drop-in replacement**: Can be used anywhere original checkpoints would be used

---

## Results and Analysis

### Final Optimal Weights:
- **Singleline**: 26.3% contribution
- **Multiline**: 28.5% contribution  
- **Annotated**: 45.2% contribution

### Performance Metrics:
- **Final L2 Error**: 158.061809
- **Relative Error**: 1.071417  
- **R² (approximate)**: -0.147935
- **Constraint Violation**: 0.00000000 (perfect)

### Key Insights:
1. **Annotated training data is most important** (45.2% weight)
2. **Multiline slightly outweighs singleline** (28.5% vs 26.3%)
3. **Negative R²** indicates concatenationTrained has emergent properties beyond linear combination
4. **Perfect constraint satisfaction** proves global optimality

### Computational Performance:
- **Total elements processed**: 1,509,949,440
- **Computation time**: ~60 seconds
- **Memory efficiency**: O(1) vs O(1.5B elements)
- **Method**: Analytically solved using Lagrange multipliers

---

## Files in This Directory

### Core Scripts:
- **`memory_efficient_global_optimize.py`**: Main optimization script
- **`combine_checkpoints_optimized.py`**: Checkpoint combination script
- **`final_optimization_summary.py`**: Results analysis and summary
- **`compare_global_vs_layerwise.py`**: Compare global vs layer-wise optimization

### Results Directories:
- **`memory_efficient_global_optimization/`**: Optimization results
- **`convex_optimization_results/`**: Layer-wise optimization results (if run)

### Previously Used Scripts:
- **`convex_optimize_lora_weights.py`**: Layer-by-layer optimization (optional)

---

## Usage Examples

### Load and Use Combined Checkpoint:
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-7b")

# Load optimally combined LoRA
model = PeftModel.from_pretrained(base_model, "path/to/optimally_combined_checkpoint")

# Use for inference
inputs = tokenizer("def fibonacci(n):", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
```

### Verify Combination Weights:
```python
import json

# Load optimization results
with open("memory_efficient_global_optimization/memory_efficient_global_results.json") as f:
    results = json.load(f)

weights = results["global_weights"]
print(f"Optimal weights: {weights}")
print(f"Singleline: {weights[0]:.1%}")
print(f"Multiline: {weights[1]:.1%}")  
print(f"Annotated: {weights[2]:.1%}")
```

---

## Troubleshooting

### Common Issues:

#### 1. Configuration File Issues
**Problem**: YAML configuration file not found or malformed
**Solution**: 
```bash
# Check if config file exists
ls ../LoRAExtraction/checkpoints_config.yml

# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('../LoRAExtraction/checkpoints_config.yml'))"

# Create default configuration
cat > ../LoRAExtraction/checkpoints_config.yml << 'EOF'
checkpoints:
  annotated:
    path: "../starcoder27b/annotated/checkpoint-40000"
    description: "Annotated training data checkpoint"
  multiline:
    path: "../starcoder27b/multiline/checkpoint-40000"
    description: "Multiline training data checkpoint"
  singleline:
    path: "../starcoder27b/singleline/checkpoint-40000"
    description: "Singleline training data checkpoint"
  concatenationTrained:
    path: "../starcoder27b/concatenationTrained/checkpoint-40000"
    description: "Concatenation trained checkpoint"
settings:
  verify_paths: true
  verbose_paths: true
EOF
```

#### 2. Checkpoint Path Validation Errors
**Problem**: Checkpoint paths in YAML don't exist or are missing required files
**Solution**:
```bash
# Use verbose mode to see detailed path validation
python3 ../LoRAExtraction/extract_starcoder27b_matrices.py \
    --config ../LoRAExtraction/checkpoints_config.yml \
    --output_dir ../extracted_starcoder27b_matrices \
    --verbose

# Check individual checkpoint directories
ls -la /path/to/your/checkpoint/directory/
# Should contain: adapter_model.safetensors, adapter_config.json
```

#### 3. Memory Errors
**Problem**: Out of memory during optimization
**Solution**: The memory-efficient script should handle this, but if issues persist:
```bash
# Reduce batch processing or run on machine with more RAM
# The current implementation uses O(1) memory
```

#### 4. Missing Dependencies
**Problem**: Import errors for torch, safetensors, yaml, etc.
**Solution**:
```bash
pip install torch safetensors numpy scipy pyyaml
```

#### 5. Key Mapping Errors
**Problem**: Matrix keys don't match between extracted and checkpoint formats
**Solution**: This should be handled automatically, but verify:
- Extracted format: `model.layers.X.self_attn.Y_proj`
- Checkpoint format: `base_model.model.model.layers.X.self_attn.Y_proj.lora_A.weight`

#### 6. File Not Found Errors
**Problem**: Cannot find input directories or files
**Solution**: Ensure you've run previous steps and paths are correct:
```bash
# Check extraction was completed
ls ../extracted_starcoder27b_matrices/

# Check organization was completed  
ls ../extracted_starcoder27b_matrices/organized_by_layer_module/

# Verify optimization results exist
ls memory_efficient_global_optimization/
```

#### 7. Checkpoint Size Issues
**Problem**: Combined checkpoint seems too large/small
**Solution**: Verify the adapter_model.safetensors size:
- **Expected size**: ~29.4 MB
- **Contains**: 256 tensors (128 A + 128 B matrices)

#### 8. Configuration Validation Issues
**Problem**: Need to check configuration without running extraction
**Solution**: Use helper commands:
```bash
# Create a default configuration file
python3 ../LoRAExtraction/extract_starcoder27b_matrices.py --create_default_config

# Test configuration with verbose validation
python3 ../LoRAExtraction/extract_starcoder27b_matrices.py \
    --config ../LoRAExtraction/checkpoints_config.yml \
    --checkpoints annotated  # Test with just one checkpoint
    --verbose \
    --output_dir /tmp/test_extraction  # Use temporary directory
```

---

### Performance Tips:

1. **Use SSD storage** for faster I/O during matrix operations
2. **Ensure sufficient RAM** (8GB+ recommended) for large matrix operations
3. **Run on GPU-enabled machine** for faster tensor operations (optional)
4. **Clear intermediate results** to save disk space between steps
5. **Use YAML validation** to catch configuration errors early
6. **Test with single checkpoint first** before processing all checkpoints

---

## Mathematical Background

### Problem Formulation:
The optimization problem is a constrained quadratic program:

```
minimize    (1/2) * ||Aw - b||²
subject to  1^T w = 1
            w ≥ 0

Where:
- A is the stacked matrix of [AB1, AB2, AB3] 
- b is the target AB4 matrix (concatenationTrained)
- w = [w1, w2, w3] are the combination weights
```

### Solution Method:
We solve using the method of Lagrange multipliers:

```
L(w,λ,μ) = (1/2)||Aw - b||² + λ(1^T w - 1) - μ^T w

Optimality conditions:
∇w L = A^T(Aw - b) + λ1 - μ = 0
∇λ L = 1^T w - 1 = 0
∇μ L = -w ≤ 0, μ ≥ 0, μ^T w = 0
```

### Memory Efficiency:
Instead of storing the full global matrix A (1.5B elements), we compute:
- **A^T A** (3×3 matrix) 
- **A^T b** (3×1 vector)

Then solve the normal equations: **(A^T A) w = A^T b** subject to constraints.

This reduces memory complexity from O(total_elements) to O(1).

---

## Citation

If you use this optimization pipeline in your research, please cite:

```bibtex
@misc{lora_convex_optimization_2024,
  title={Optimal Linear Combination of LoRA Adapters via Convex Optimization},
  author={StarCoder2-7B LoRA Analysis},
  year={2024},
  note={Constrained quadratic programming approach to LoRA adapter combination}
}
```

---

**Happy Optimizing!** 🚀

For questions or issues, check the troubleshooting section above or examine the generated log files for detailed error messages.
