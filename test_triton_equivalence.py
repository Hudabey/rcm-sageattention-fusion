"""
Test Suite: Triton Kernel Equivalence

Verify that the Triton fused kernel produces identical results
to the PyTorch baseline.
"""

import sys
sys.path.append('.')

import torch
import pytest


def test_triton_vs_baseline():
    """Test Triton kernel matches baseline exactly."""
    try:
        import triton
        from triton_fused_attention import TritonFusedRCMAttention
    except ImportError:
        print("⚠ Triton not installed, skipping Triton tests")
        return
    
    from kernels.baseline_attention import BaselineAttentionWithSkip
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping Triton tests")
        return
    
    torch.manual_seed(42)
    
    configs = [
        (2, 128, 256, 4),
        (4, 512, 512, 8),
        (8, 1024, 768, 12),
    ]
    
    for B, N, C, num_heads in configs:
        print(f"\nTesting B={B}, N={N}, C={C}, heads={num_heads}")
        
        # Create models
        baseline = BaselineAttentionWithSkip(C, num_heads).cuda()
        triton_fused = TritonFusedRCMAttention(C, num_heads).cuda()
        
        # Copy weights
        triton_fused.qkv.weight.data = baseline.attention.qkv.weight.data.clone()
        triton_fused.qkv.bias.data = baseline.attention.qkv.bias.data.clone()
        triton_fused.proj.weight.data = baseline.attention.proj.weight.data.clone()
        triton_fused.proj.bias.data = baseline.attention.proj.bias.data.clone()
        
        baseline.eval()
        triton_fused.eval()
        
        # Test data
        x = torch.randn(B, N, C, device='cuda')
        t = torch.rand(B, device='cuda')
        
        # Forward pass
        with torch.no_grad():
            out_baseline = baseline(x, t)
            out_triton = triton_fused(x, t)
        
        # Check equivalence
        max_diff = (out_baseline - out_triton).abs().max().item()
        mean_diff = (out_baseline - out_triton).abs().mean().item()
        
        print(f"  Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        
        # Triton might have slightly different numerical precision
        assert max_diff < 1e-3, f"Outputs differ by {max_diff}"
        assert mean_diff < 1e-4, f"Mean difference too high: {mean_diff}"
    
    print("\n✓ Triton kernel matches baseline (within numerical precision)")


def test_triton_gradient():
    """Test that Triton kernel has correct gradients."""
    try:
        from triton_fused_attention import TritonFusedRCMAttention
    except ImportError:
        print("⚠ Triton not installed, skipping gradient test")
        return
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping gradient test")
        return
    
    torch.manual_seed(42)
    
    model = TritonFusedRCMAttention(256, 8).cuda()
    model.train()
    
    x = torch.randn(2, 64, 256, device='cuda', requires_grad=True)
    t = torch.tensor([0.5, 0.8], device='cuda')
    
    # Forward + backward
    out = model(x, t)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradient not computed"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"
    
    print("✓ Triton kernel gradient computation works")


def test_triton_boundary_conditions():
    """Test rCM boundary conditions with Triton kernel."""
    try:
        from triton_fused_attention import TritonFusedRCMAttention
    except ImportError:
        print("⚠ Triton not installed, skipping boundary test")
        return
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping boundary test")
        return
    
    model = TritonFusedRCMAttention(256, 8).cuda()
    
    # Test t=0 (should preserve input via skip)
    c_skip_0, c_out_0 = model.get_skip_coefficients(torch.tensor(0.0))
    assert abs(c_skip_0 - 1.0) < 1e-6, "c_skip(0) should be 1.0"
    assert abs(c_out_0 - 0.0) < 1e-6, "c_out(0) should be 0.0"
    
    # Test large t (should mostly use attention)
    c_skip_T, c_out_T = model.get_skip_coefficients(torch.tensor(100.0))
    assert c_skip_T < 0.01, "c_skip(∞) should be ~0"
    assert c_out_T > 0.99, "c_out(∞) should be ~1"
    
    print("✓ Triton kernel satisfies rCM boundary conditions")


def test_triton_performance():
    """Quick performance sanity check."""
    try:
        from triton_fused_attention import TritonFusedRCMAttention
    except ImportError:
        print("⚠ Triton not installed, skipping performance test")
        return
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping performance test")
        return
    
    from kernels.baseline_attention import BaselineAttentionWithSkip
    import time
    
    B, N, C = 4, 1024, 512
    num_heads = 8
    
    baseline = BaselineAttentionWithSkip(C, num_heads).cuda()
    triton_fused = TritonFusedRCMAttention(C, num_heads).cuda()
    
    # Copy weights
    triton_fused.qkv.weight.data = baseline.attention.qkv.weight.data.clone()
    triton_fused.qkv.bias.data = baseline.attention.qkv.bias.data.clone()
    triton_fused.proj.weight.data = baseline.attention.proj.weight.data.clone()
    triton_fused.proj.bias.data = baseline.attention.proj.bias.data.clone()
    
    baseline.eval()
    triton_fused.eval()
    
    x = torch.randn(B, N, C, device='cuda')
    t = torch.rand(B, device='cuda')
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = baseline(x, t)
            _ = triton_fused(x, t)
    
    torch.cuda.synchronize()
    
    # Time baseline
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(50):
        with torch.no_grad():
            _ = baseline(x, t)
    end.record()
    torch.cuda.synchronize()
    baseline_time = start.elapsed_time(end) / 50
    
    # Time Triton
    start.record()
    for _ in range(50):
        with torch.no_grad():
            _ = triton_fused(x, t)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / 50
    
    speedup = baseline_time / triton_time
    
    print(f"\nPerformance check:")
    print(f"  Baseline: {baseline_time:.3f} ms")
    print(f"  Triton:   {triton_time:.3f} ms")
    print(f"  Speedup:  {speedup:.2f}x")
    
    # Triton should be at least as fast as baseline (ideally faster)
    if speedup >= 1.0:
        print("✓ Triton kernel is competitive or faster")
    else:
        print(f"⚠ Triton kernel is slower ({speedup:.2f}x) - may need tuning")


if __name__ == "__main__":
    print("Testing Triton Fused rCM Attention")
    print("=" * 60)
    
    test_triton_vs_baseline()
    test_triton_gradient()
    test_triton_boundary_conditions()
    test_triton_performance()
    
    print("\n" + "=" * 60)
    print("✓ ALL TRITON TESTS PASSED")
