"""
Triton Fused Attention with rCM Skip Connection - CORRECT IMPLEMENTATION

Uses online softmax normalization (FlashAttention algorithm) to compute
attention correctly across multiple K/V blocks.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _fwd_kernel(
    Q, K, V, X_residual, Out,
    c_skip, c_out,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_xz, stride_xh, stride_xm, stride_xd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, M, N, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused attention with rCM skip using online softmax normalization.
    
    Implements FlashAttention-style algorithm:
    - Computes attention with tiled K/V
    - Uses online normalization for correct softmax
    - Applies skip connection in epilogue
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    
    # Batch and head indices
    batch_idx = pid_z // H
    head_idx = pid_z % H
    
    # Offsets for this thread block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Load Q block
    q_ptrs = Q + batch_idx * stride_qz + head_idx * stride_qh + \
             offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Initialize accumulators for online softmax
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1e-6  # Running sum
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9   # Running max
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # Attention output
    
    # Iterate over K and V blocks (online softmax)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Load K block
        k_ptrs = K + batch_idx * stride_kz + head_idx * stride_kh + \
                 offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k_mask = (offs_d[:, None] < D) & (offs_n[None, :] < N)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # Compute QK^T for this block
        qk = tl.dot(q, k, allow_tf32=True) * scale  # [BLOCK_M, BLOCK_N]
        
        # Update running max
        m_ij = tl.max(qk, axis=1)  # Max over N dimension
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Compute exp with correction factor
        alpha = tl.exp(m_i - m_i_new)  # Correction for previous blocks
        p = tl.exp(qk - m_i_new[:, None])  # Current block scores
        
        # Update running sum
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # Load V block
        v_ptrs = V + batch_idx * stride_vz + head_idx * stride_vh + \
                 offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # Update accumulator with correction
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=True)
        
        # Update max for next iteration
        m_i = m_i_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # ===== EPILOGUE: Fused rCM Skip Connection =====
    # Load residual (shaped as [B, H, N, D])
    x_ptrs = X_residual + batch_idx * stride_xz + head_idx * stride_xh + \
             offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xd
    x_res = tl.load(x_ptrs, mask=q_mask, other=0.0)
    
    # Apply skip: out = c_skip * x_res + c_out * acc
    out = c_skip * x_res + c_out * acc
    
    # ===== SINGLE HBM WRITE =====
    out_ptrs = Out + batch_idx * stride_oz + head_idx * stride_oh + \
               offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=q_mask)


class TritonFusedRCMAttention(torch.nn.Module):
    """
    Production-grade fused attention with rCM skip connection.
    
    Implements:
    - FlashAttention-style online softmax
    - rCM skip connection in epilogue
    - Single HBM write for output
    """
    
    def __init__(self, dim, num_heads=8, sigma_data=1.0):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sigma_data = sigma_data
        
        # Learnable parameters
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=True)
        self.proj = torch.nn.Linear(dim, dim)
        
    def get_skip_coefficients(self, t):
        """Compute rCM skip coefficients."""
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=torch.float32, device='cuda')
        
        sigma_d_sq = self.sigma_data ** 2
        t_sq = t ** 2
        
        c_skip = sigma_d_sq / (t_sq + sigma_d_sq)
        c_out = self.sigma_data * t / torch.sqrt(sigma_d_sq + t_sq)
        
        return c_skip, c_out
    
    def forward(self, x, t):
        """Forward pass with Triton fused kernel."""
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Get skip coefficients
        c_skip, c_out = self.get_skip_coefficients(t)
        if c_skip.dim() > 0:
            c_skip = c_skip.mean().item()
            c_out = c_out.mean().item()
        else:
            c_skip = c_skip.item()
            c_out = c_out.item()
        
        # Reshape X to match attention layout [B, H, N, D]
        x_reshaped = x.reshape(B, N, self.num_heads, self.head_dim)
        x_reshaped = x_reshaped.permute(0, 2, 1, 3).contiguous()
        
        # Prepare output
        out = torch.empty_like(x_reshaped)
        
        # Grid configuration
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = min(64, triton.next_power_of_2(self.head_dim))
        
        grid = lambda meta: (
            triton.cdiv(N, BLOCK_M),
            B * self.num_heads,
        )
        
        # Launch kernel
        _fwd_kernel[grid](
            q, k, v, x_reshaped, out,
            c_skip, c_out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            x_reshaped.stride(0), x_reshaped.stride(1), x_reshaped.stride(2), x_reshaped.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B * self.num_heads, self.num_heads, N, N, self.head_dim,
            self.scale,
            BLOCK_M, BLOCK_N, BLOCK_D,
        )
        
        # Reshape and project
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        
        return out


if __name__ == "__main__":
    print("Testing Correct Triton Implementation")
    print("=" * 60)
    
    model = TritonFusedRCMAttention(dim=512, num_heads=8).cuda()
    x = torch.randn(2, 128, 512, device='cuda')
    t = torch.tensor([0.5, 0.8], device='cuda')
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful")
    
    # Test against baseline
    from kernels.baseline_attention import BaselineAttentionWithSkip
    
    baseline = BaselineAttentionWithSkip(512, 8).cuda()
    baseline.attention.qkv.weight.data = model.qkv.weight.data.clone()
    baseline.attention.qkv.bias.data = model.qkv.bias.data.clone()
    baseline.attention.proj.weight.data = model.proj.weight.data.clone()
    baseline.attention.proj.bias.data = model.proj.bias.data.clone()
    
    baseline.eval()
    model.eval()
    
    with torch.no_grad():
        out_baseline = baseline(x, t)
        out_triton = model(x, t)
    
    max_diff = (out_baseline - out_triton).abs().max().item()
    print(f"\nNumerical verification:")
    print(f"  Max difference: {max_diff:.2e}")
    
    if max_diff < 1e-3:
        print("  ✓ Triton kernel matches baseline!")
    else:
        print(f"  ✗ Outputs differ by {max_diff}")
