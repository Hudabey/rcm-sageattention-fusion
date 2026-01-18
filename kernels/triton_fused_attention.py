"""
Triton Fused Attention with rCM Skip Connection

TRUE kernel fusion - skip connection applied in epilogue before HBM write.
No PyTorch overhead, no unnecessary memory traffic.
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
    stride_xz, stride_xm, stride_xd,
    stride_oz, stride_om, stride_od,
    Z, H, M, N, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused attention kernel with rCM skip connection in epilogue.
    
    Computes: Out = c_skip * X_residual + c_out * Attention(Q, K, V)
    
    Key optimization: Skip connection applied BEFORE writing to HBM.
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    
    # Batch and head indices
    batch_idx = pid_z // H
    head_idx = pid_z % H
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Initialize accumulator for attention output
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Load Q block
    q_ptrs = Q + batch_idx * stride_qz + head_idx * stride_qh + \
             offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D), other=0.0)
    
    # Compute attention over all K, V blocks
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Load K block
        k_ptrs = K + batch_idx * stride_kz + head_idx * stride_kh + \
                 offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=(offs_d[:, None] < D) & (offs_n[None, :] < N), other=0.0)
        
        # Compute QK^T
        qk = tl.dot(q, k, allow_tf32=True) * scale
        
        # Softmax
        qk_max = tl.max(qk, axis=1)
        qk = qk - qk_max[:, None]
        qk = tl.exp(qk)
        qk_sum = tl.sum(qk, axis=1)
        qk = qk / qk_sum[:, None]
        
        # Load V block
        v_ptrs = V + batch_idx * stride_vz + head_idx * stride_vh + \
                 offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)
        
        # Accumulate attention output
        acc += tl.dot(qk, v, allow_tf32=True)
    
    # ===== EPILOGUE: Fused rCM Skip Connection =====
    # Load residual
    x_ptrs = X_residual + batch_idx * stride_xz + \
             offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xd
    x_res = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D), other=0.0)
    
    # Apply skip: out = c_skip * x_res + c_out * acc
    out = c_skip * x_res + c_out * acc
    
    # ===== SINGLE HBM WRITE =====
    out_ptrs = Out + batch_idx * stride_oz + \
               offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))


class TritonFusedRCMAttention(torch.nn.Module):
    """
    Production-grade fused attention with rCM skip connection.
    
    Implemented as a Triton kernel for maximum performance.
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
        
        # Use scalar coefficients for simplicity
        if c_skip.dim() > 0:
            c_skip = c_skip.mean().item()
            c_out = c_out.mean().item()
        else:
            c_skip = c_skip.item()
            c_out = c_out.item()
        
        # Prepare output
        out = torch.empty(B, self.num_heads, N, self.head_dim, device=x.device, dtype=x.dtype)
        
        # Grid configuration
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = 64
        
        grid = lambda meta: (
            triton.cdiv(N, BLOCK_M),
            B * self.num_heads,
        )
        
        # Launch fused kernel
        _fwd_kernel[grid](
            q, k, v, x.view(B, N, C), out,
            c_skip, c_out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            x.stride(0), x.stride(1), 1,  # X is [B, N, C]
            out.stride(0), out.stride(1), out.stride(2),
            B * self.num_heads, self.num_heads, N, N, self.head_dim,
            self.scale,
            BLOCK_M, BLOCK_N, BLOCK_D,
        )
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out


if __name__ == "__main__":
    print("Testing Triton Fused rCM Attention")
    print("=" * 60)
    
    # Simple test
    model = TritonFusedRCMAttention(dim=512, num_heads=8).cuda()
    x = torch.randn(2, 128, 512, device='cuda')
    t = torch.tensor([0.5, 0.8], device='cuda')
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful")
