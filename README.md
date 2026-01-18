# Fusing rCM Skip-Connections into SageAttention 2++ Kernels

**Goal**: Eliminate HBM round-trips by fusing rCM consistency boundary conditions into the attention kernel epilogue.

## Background

### rCM (Rectified Consistency Model)
- Uses skip connections with boundary conditions: `f_θ(x_t, t) = c_skip(t) * x_t + c_out(t) * F_θ(x_t, t)`
- Boundary condition at t=0: `f_θ(x_0, 0) = x_0` (enforced by `c_skip(0)=1, c_out(0)=0`)
- These coefficients prevent mode collapse and enforce consistency

### SageAttention 2++
- Quantized attention kernel (int8 matmuls)
- Writes output to HBM after attention computation

### The Problem
Current TurboDiffusion implementation:
1. Compute attention → write to HBM
2. Read from HBM → apply skip connection → write back to HBM
3. Total: 2 unnecessary HBM round-trips

### Our Solution
Fuse skip connection into attention epilogue:
1. Compute attention → apply skip in registers → write once to HBM
2. Total: 0 unnecessary round-trips

## Project Structure

```
kernels/
  ├── baseline_attention.py      # Two-pass: attention + separate skip
  └── fused_attention.py          # One-pass: skip in epilogue

benchmarks/
  ├── profile_hbm.py              # Measure HBM traffic with nsys
  └── compare_implementations.py  # Speed and accuracy benchmarks

tests/
  └── test_numerical_equivalence.py  # Verify fusion is bit-exact

blog/
  └── writeup.md                  # Technical blog post
```

## Installation

```bash
pip install torch triton numpy pytest nsight-systems
```

## Usage

```python
from kernels.fused_attention import FusedRCMAttention

# Initialize with skip coefficients
attn = FusedRCMAttention(
    dim=512,
    num_heads=8,
    c_skip=lambda t: 1.0 / (1 + t**2),  # Example schedule
    c_out=lambda t: t / np.sqrt(1 + t**2)
)

# Single-pass computation
output = attn(x, timestep=0.5, residual=x)  # Fused in one kernel
```

## Benchmarks

Run HBM profiling:
```bash
python benchmarks/profile_hbm.py --batch 32 --seq 2048 --dim 512
```

Compare implementations:
```bash
python benchmarks/compare_implementations.py
```

## Results

[To be filled after benchmarking]

- HBM traffic reduction: XX%
- Speedup: XXms baseline → XXms fused
- Numerical error: < 1e-6

## References

- [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)
- [rCM Paper](https://arxiv.org/abs/2510.08431)
- [SageAttention](https://arxiv.org/abs/2410.02367)
