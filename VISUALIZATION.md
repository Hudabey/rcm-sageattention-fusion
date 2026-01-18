# Visual Explanation: HBM Round-trip Elimination

## The Problem: Baseline (Two-Pass)

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Memory Hierarchy                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Registers/SRAM (Fast, <1 cycle)                            │
│  ┌──────────────────────────────────────┐                   │
│  │                                       │                   │
│  │  Step 1: Compute Attention            │                   │
│  │    q, k, v = split(QKV(x))           │                   │
│  │    attn = softmax(q @ k.T)           │                   │
│  │    out = attn @ v                    │                   │
│  │                                       │                   │
│  └──────────────────────────────────────┘                   │
│             │                                                 │
│             │ WRITE (400 cycles)                             │
│             ▼                                                 │
│  ┌──────────────────────────────────────┐                   │
│  │      HBM (Slow, ~400 cycles)         │                   │
│  │                                       │                   │
│  │    attn_out: [B, N, C]               │ ◄─── Stored!     │
│  │                                       │                   │
│  └──────────────────────────────────────┘                   │
│             │                                                 │
│             │ READ (400 cycles)                              │
│             ▼                                                 │
│  ┌──────────────────────────────────────┐                   │
│  │                                       │                   │
│  │  Step 2: Apply Skip Connection        │                   │
│  │    c_skip, c_out = coeffs(t)         │                   │
│  │    final = c_skip*x + c_out*attn_out │ ◄─── Read X too! │
│  │                                       │                   │
│  └──────────────────────────────────────┘                   │
│             │                                                 │
│             │ WRITE (400 cycles)                             │
│             ▼                                                 │
│  ┌──────────────────────────────────────┐                   │
│  │      HBM (Slow, ~400 cycles)         │                   │
│  │                                       │                   │
│  │    final: [B, N, C]                  │ ◄─── Stored!     │
│  │                                       │                   │
│  └──────────────────────────────────────┘                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘

Total HBM Traffic:
  • Attention: 1 write (attn_out)
  • Skip:      2 reads (x, attn_out) + 1 write (final)
  • TOTAL:     2 reads + 2 writes = 4 HBM operations

Latency: ~1600 cycles (400 × 4)
```

## The Solution: Fused (One-Pass)

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Memory Hierarchy                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Registers/SRAM (Fast, <1 cycle)                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                                                        │   │
│  │  Step 1: Compute Attention                            │   │
│  │    q, k, v = split(QKV(x))                           │   │
│  │    attn = softmax(q @ k.T)                           │   │
│  │    attn_out = attn @ v                               │   │
│  │                              │                         │   │
│  │  ┌───────────────────────────┘                        │   │
│  │  │                                                     │   │
│  │  │  Step 2: FUSED Epilogue (still in registers!)     │   │
│  │  │    c_skip, c_out = coeffs(t)                      │   │
│  │  │    final = c_skip*x + c_out*attn_out  ◄─── Fused!│   │
│  │  │                                                     │   │
│  │  └─────────────────────────────────────────┐          │   │
│  │                                             │          │   │
│  └─────────────────────────────────────────────┼──────────┘   │
│                                                │              │
│                                                │ WRITE (400)  │
│                                                ▼              │
│  ┌──────────────────────────────────────┐                   │
│  │      HBM (Slow, ~400 cycles)         │                   │
│  │                                       │                   │
│  │    final: [B, N, C]                  │ ◄─── Single Write│
│  │                                       │                   │
│  └──────────────────────────────────────┘                   │
│                                                               │
│  ✓ attn_out NEVER written to HBM!                           │
│  ✓ Skip applied while data is HOT in registers!             │
│                                                               │
└─────────────────────────────────────────────────────────────┘

Total HBM Traffic:
  • Fused:     0 reads (from HBM for skip) + 1 write (final)
  • TOTAL:     0 reads + 1 write = 1 HBM operation

Latency: ~400 cycles
Speedup: 1600 / 400 = 4x for skip operation!
```

## Key Insight

```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  The skip connection is computationally TRIVIAL:             │
│                                                               │
│    c_skip * x + c_out * attn_out                            │
│    ^^^^^   ^   ^^^^^   ^^^^^^^^^                             │
│    2 multiplications + 1 addition = 3 FLOPs                  │
│                                                               │
│  But in the baseline, we pay 1600 cycles to:                │
│    • Write attn_out to HBM      (400 cycles)                │
│    • Read x from HBM            (400 cycles)                │
│    • Read attn_out from HBM     (400 cycles)                │
│    • Write final to HBM         (400 cycles)                │
│                                                               │
│  The actual computation (3 FLOPs) takes <1 cycle!           │
│  But the memory traffic takes 1600 cycles!                   │
│                                                               │
│  Memory-bound, not compute-bound! ◄─── This is the problem  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Cache Hierarchy Benefits

```
Memory Level       Access Time    Benefit in Fused Implementation
─────────────────────────────────────────────────────────────────
Registers          <1 cycle       ✓ attn_out stays here!
L1 Cache           ~4 cycles      ✓ x likely still here from QKV
L2 Cache           ~20 cycles     ✓ Fallback if L1 evicted
HBM                ~400 cycles    ✗ Only write final result

Baseline:  Hits HBM 4 times
Fused:     Hits HBM 1 time (write only)
```

## Real-World Impact: Video Diffusion

```
Wan2.1-14B Model:
  • 32 transformer blocks
  • 81 frames
  • 4 timesteps
  • Each block has self-attention + cross-attention

Skip operations per generation:
  32 blocks × 2 attentions × 81 frames × 4 steps = 20,736 skip ops

HBM traffic saved (per generation):
  20,736 ops × 28% reduction × ~1 MB per skip ≈ 5.8 GB saved!

On RTX 5090 (1 TB/s bandwidth):
  5.8 GB / 1000 GB/s = 5.8 ms saved per video

This compounds across multiple videos! 🚀
```

## The Fusion Pattern

This pattern applies to ANY residual connection:

```python
# ❌ Baseline (two-pass)
intermediate = expensive_operation(x)
final = cheap_residual(x, intermediate)  # Requires HBM round-trip

# ✅ Fused (one-pass)  
final = fused_operation_with_residual(x)  # Residual in epilogue!
```

**Examples where this helps:**
- ResNet skip connections
- LoRA adapters
- Transformer residuals
- Any `out = f(x) + x` pattern

**Key requirement:** The residual operation must be CHEAP compared to the main operation. Otherwise, you're optimizing the wrong thing!

---

**Remember**: In GPU optimization, it's not about making computation faster—it's about moving less data!
