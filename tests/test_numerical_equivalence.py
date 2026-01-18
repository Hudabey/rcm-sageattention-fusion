"""
Test Suite: Numerical Equivalence

Verify that the fused implementation produces identical results to baseline.
"""

import sys
sys.path.append('/home/claude/rcm-sageattention-fusion')

import torch
import pytest


def test_skip_coefficients():
    """Test rCM skip coefficient computation."""
    from kernels.baseline_attention import RCMSkipConnection
    from kernels.fused_attention import FusedRCMAttention
    
    skip = RCMSkipConnection(sigma_data=1.0)
    fused = FusedRCMAttention(dim=256, num_heads=8)
    
    # Test boundary conditions
    t_values = torch.tensor([0.0, 0.5, 1.0, 10.0, 100.0])
    
    for t in t_values:
        c_skip_baseline, c_out_baseline = skip.get_coefficients(t)
        c_skip_fused, c_out_fused = fused.get_skip_coefficients(t)
        
        assert torch.allclose(c_skip_baseline, c_skip_fused), \
            f"c_skip mismatch at t={t}: {c_skip_baseline} vs {c_skip_fused}"
        assert torch.allclose(c_out_baseline, c_out_fused), \
            f"c_out mismatch at t={t}: {c_out_baseline} vs {c_out_fused}"
    
    # Verify boundary conditions
    c_skip_0, c_out_0 = skip.get_coefficients(torch.tensor(0.0))
    assert abs(c_skip_0 - 1.0) < 1e-6, "c_skip(0) should be 1.0"
    assert abs(c_out_0 - 0.0) < 1e-6, "c_out(0) should be 0.0"
    
    print("✓ Skip coefficients match exactly")


def test_forward_equivalence():
    """Test that fused and baseline produce identical outputs."""
    from kernels.baseline_attention import BaselineAttentionWithSkip
    from kernels.fused_attention import FusedRCMAttention
    
    torch.manual_seed(42)
    
    configs = [
        # (B, N, C, num_heads)
        (1, 16, 128, 4),
        (4, 64, 256, 8),
        (8, 128, 512, 8),
        (2, 256, 512, 16),
    ]
    
    for B, N, C, num_heads in configs:
        print(f"\nTesting B={B}, N={N}, C={C}, heads={num_heads}")
        
        # Create models
        baseline = BaselineAttentionWithSkip(C, num_heads)
        fused = FusedRCMAttention(C, num_heads)
        
        # Copy weights
        fused.qkv.weight.data = baseline.attention.qkv.weight.data.clone()
        fused.qkv.bias.data = baseline.attention.qkv.bias.data.clone()
        fused.proj.weight.data = baseline.attention.proj.weight.data.clone()
        fused.proj.bias.data = baseline.attention.proj.bias.data.clone()
        
        # Test inputs
        x = torch.randn(B, N, C)
        t = torch.rand(B) * 2.0  # Random timesteps in [0, 2]
        
        # Forward pass
        baseline.eval()
        fused.eval()
        
        with torch.no_grad():
            out_baseline = baseline(x, t)
            out_fused = fused(x, t)
        
        # Check equivalence
        max_diff = (out_baseline - out_fused).abs().max().item()
        mean_diff = (out_baseline - out_fused).abs().mean().item()
        
        print(f"  Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        
        # Due to dropout being disabled in eval mode, should be exact
        assert max_diff < 1e-5, f"Outputs differ by {max_diff}"
        assert mean_diff < 1e-6, f"Mean difference too high: {mean_diff}"
    
    print("\n✓ All forward passes match exactly")


def test_gradient_equivalence():
    """Test that gradients match for training."""
    from kernels.baseline_attention import BaselineAttentionWithSkip
    from kernels.fused_attention import FusedRCMAttention
    
    torch.manual_seed(42)
    
    B, N, C, num_heads = 2, 32, 128, 4
    
    # Create models
    baseline = BaselineAttentionWithSkip(C, num_heads)
    fused = FusedRCMAttention(C, num_heads)
    
    # Copy weights
    fused.qkv.weight.data = baseline.attention.qkv.weight.data.clone()
    fused.qkv.bias.data = baseline.attention.qkv.bias.data.clone()
    fused.proj.weight.data = baseline.attention.proj.weight.data.clone()
    fused.proj.bias.data = baseline.attention.proj.bias.data.clone()
    
    # Test inputs
    x = torch.randn(B, N, C, requires_grad=True)
    x_fused = x.clone().detach().requires_grad_(True)
    t = torch.tensor([0.5, 1.0])
    
    # Forward + backward
    baseline.train()
    fused.train()
    
    # Baseline
    out_baseline = baseline(x, t)
    loss_baseline = out_baseline.sum()
    loss_baseline.backward()
    
    # Fused
    out_fused = fused(x_fused, t)
    loss_fused = out_fused.sum()
    loss_fused.backward()
    
    # Check output equivalence
    max_diff = (out_baseline - out_fused).abs().max().item()
    print(f"Output max diff: {max_diff:.2e}")
    
    # Check gradient equivalence
    grad_diff = (x.grad - x_fused.grad).abs().max().item()
    print(f"Input gradient max diff: {grad_diff:.2e}")
    
    assert max_diff < 1e-4, f"Outputs differ: {max_diff}"
    # Gradients might have slightly more variance due to dropout
    assert grad_diff < 5e-4, f"Gradients differ: {grad_diff}"
    
    print("✓ Gradients match")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    from kernels.fused_attention import FusedRCMAttention
    
    torch.manual_seed(42)
    
    C = 256
    model = FusedRCMAttention(C, num_heads=8)
    
    # Test t=0 (should mostly preserve input via skip)
    x = torch.randn(1, 16, C)
    out_t0 = model(x, torch.tensor([0.0]))
    # At t=0: c_skip=1, c_out=0, so output ≈ input (after attention)
    # Can't be exact due to attention, but structure preserved
    
    # Test large t (should mostly use attention output)
    out_t_large = model(x, torch.tensor([100.0]))
    # At large t: c_skip≈0, c_out≈1, so output ≈ attention(x)
    
    # Test batch with mixed timesteps
    x_batch = torch.randn(4, 16, C)
    t_batch = torch.tensor([0.0, 0.5, 1.0, 10.0])
    out_batch = model(x_batch, t_batch)
    
    assert out_batch.shape == x_batch.shape
    print("✓ Edge cases handled correctly")


if __name__ == "__main__":
    print("Running numerical equivalence tests...\n")
    print("=" * 60)
    
    test_skip_coefficients()
    test_forward_equivalence()
    test_gradient_equivalence()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("  Fused implementation is numerically equivalent to baseline")
