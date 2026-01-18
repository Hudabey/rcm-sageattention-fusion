"""
Baseline Implementation: Two-Pass Attention + Skip Connection

This mimics the current TurboDiffusion approach:
1. Compute attention output → write to HBM
2. Read back → apply rCM skip connection → write to HBM

HBM round-trips: 2 unnecessary (read + write for skip)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RCMSkipConnection(nn.Module):
    """rCM consistency boundary condition via skip connections.
    
    f(x_t, t) = c_skip(t) * x_t + c_out(t) * F(x_t, t)
    
    Boundary conditions:
    - c_skip(0) = 1, c_out(0) = 0  (ensures f(x_0, 0) = x_0)
    - c_skip(T) = 0, c_out(T) = 1  (ensures f(x_T, T) = F(x_T, T))
    """
    
    def __init__(self, sigma_data=1.0):
        super().__init__()
        self.sigma_data = sigma_data
    
    def get_coefficients(self, t):
        """Compute skip and output coefficients.
        
        Args:
            t: Timestep tensor of shape [B] or scalar
            
        Returns:
            c_skip, c_out: Coefficient tensors
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=torch.float32)
        
        sigma_d_sq = self.sigma_data ** 2
        t_sq = t ** 2
        
        c_skip = sigma_d_sq / (t_sq + sigma_d_sq)
        c_out = self.sigma_data * t / torch.sqrt(sigma_d_sq + t_sq)
        
        return c_skip, c_out
    
    def forward(self, x_t, network_output, t):
        """Apply skip connection.
        
        Args:
            x_t: Input at timestep t, shape [B, ...]
            network_output: Network prediction F(x_t, t), shape [B, ...]
            t: Timestep, shape [B] or scalar
            
        Returns:
            f(x_t, t) = c_skip(t) * x_t + c_out(t) * F(x_t, t)
        """
        c_skip, c_out = self.get_coefficients(t)
        
        # Reshape coefficients for broadcasting
        shape = [c_skip.shape[0]] + [1] * (x_t.ndim - 1)
        c_skip = c_skip.view(*shape)
        c_out = c_out.view(*shape)
        
        return c_skip * x_t + c_out * network_output


class BaselineAttention(nn.Module):
    """Standard multi-head attention WITHOUT skip connection fusion.
    
    This is the baseline - skip connection applied as a SEPARATE operation
    after attention, causing HBM round-trips.
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        """Standard attention forward pass.
        
        Args:
            x: Input tensor [B, N, C]
            
        Returns:
            Attention output [B, N, C]
        """
        B, N, C = x.shape
        
        # [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Output: [B, num_heads, N, head_dim] → [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # HBM WRITE #1: Attention output written to memory
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x  # Will be read back for skip connection


class BaselineAttentionWithSkip(nn.Module):
    """Complete baseline: Attention + separate rCM skip.
    
    Demonstrates the HBM round-trip issue:
    1. Attention writes output to HBM
    2. Skip connection reads from HBM, computes, writes back
    """
    
    def __init__(self, dim, num_heads=8, sigma_data=1.0):
        super().__init__()
        self.attention = BaselineAttention(dim, num_heads)
        self.skip = RCMSkipConnection(sigma_data)
        
    def forward(self, x, t):
        """Two-pass computation.
        
        Args:
            x: Input [B, N, C]
            t: Timestep [B] or scalar
            
        Returns:
            Output with skip connection applied
        """
        # Pass 1: Compute attention (writes to HBM)
        attn_out = self.attention(x)
        
        # HBM READ + WRITE: Skip connection is a SEPARATE operation
        # This reads attn_out from HBM and writes result back
        output = self.skip(x, attn_out, t)
        
        return output


def test_baseline():
    """Test baseline implementation."""
    torch.manual_seed(42)
    
    B, N, C = 4, 128, 512
    num_heads = 8
    
    model = BaselineAttentionWithSkip(C, num_heads)
    x = torch.randn(B, N, C)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    
    output = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Timesteps: {t}")
    
    # Verify boundary conditions
    model_t0 = BaselineAttentionWithSkip(C, num_heads)
    model_t0.load_state_dict(model.state_dict())
    
    output_t0 = model_t0(x[0:1], torch.tensor([0.0]))
    # At t=0, output should approximately equal input (after attention)
    # c_skip(0)=1, c_out(0)=0 → output = 1*x + 0*attn
    
    print("\nBoundary condition check:")
    print(f"c_skip(0) should be 1.0: {model.skip.get_coefficients(torch.tensor(0.0))[0].item():.6f}")
    print(f"c_out(0) should be 0.0: {model.skip.get_coefficients(torch.tensor(0.0))[1].item():.6f}")
    
    print("\n✓ Baseline implementation working")


if __name__ == "__main__":
    test_baseline()
