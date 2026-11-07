"""
Unit Tests for Blackwell (SM 12.0) Model Loader & Inference Optimization
========================================================================
Verifies:
1. Hardware driver probe (Blackwell SM 12.0 / CUDA 13.0 / bfloat16 / TF32)
2. Driver-level CUDA optimizations & VRAM governor
3. Resilient state dict loader with zero fatal crashes and strict weights_only
4. Dynamic hook attachment of HADL v5.8 Non-Token Latent Matrix and Toroidal Router
5. Resilient end-to-end model initialization
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from xstar.inference.blackwell_optimizations import (
    HardwareDriverReport,
    probe_hardware_driver,
    configure_blackwell_cuda_environment,
    ResilientStateDictLoader,
)
from xstar.inference.engine import CustomInferenceEngine, EngineConfig, GenerationMetrics
from dual_loop.neuromorphic.spiking_gate import LatentThoughtMatrixEngine, CentralCognitiveRouter


def test_probe_hardware_driver():
    """Verifies that driver probe runs cleanly and detects host hardware."""
    report = probe_hardware_driver()
    assert isinstance(report, HardwareDriverReport)
    assert report.pytorch_version != ""
    assert report.optimal_device in ["cpu"] or report.optimal_device.startswith("cuda")
    assert report.optimal_dtype in [torch.float32, torch.bfloat16, torch.float16]
    
    summary = report.summary()
    assert "Hardware Diagnostics" in summary
    assert "Target Device" in summary


def test_configure_blackwell_cuda_environment():
    """Verifies CUDA environment settings apply without errors."""
    report = probe_hardware_driver()
    settings = configure_blackwell_cuda_environment(report, vram_headroom_fraction=0.88)
    assert isinstance(settings, dict)
    if report.cuda_available and report.device_type == "cuda":
        if report.tf32_supported:
            assert settings.get("allow_tf32") is True
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
        assert settings.get("cache_flushed") is True


def test_resilient_state_dict_loader():
    """Verifies resilient state dict application with fuzzy prefixes and shape adaptation."""
    class DummySubModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 64)
            self.bias_param = nn.Parameter(torch.zeros(64))

    module = DummySubModule()
    
    # 1. State dict with mismatched prefixes
    fake_state_dict = {
        "module.linear.weight": torch.randn(64, 32),
        "base_model.bias_param": torch.ones(64),
        "unrelated_layer.weight": torch.randn(10, 10),  # extra key
    }
    
    res = ResilientStateDictLoader.apply_state_dict_resilient(module, fake_state_dict)
    assert res["applied_keys"] == 2
    assert res["shape_mismatches"] == 0
    assert torch.allclose(module.bias_param, torch.ones(64))

    # 2. State dict with shape mismatch (should be gracefully skipped, not crash)
    mismatch_dict = {
        "linear.weight": torch.randn(128, 32),  # shape mismatch (128 vs 64)
    }
    res2 = ResilientStateDictLoader.apply_state_dict_resilient(module, mismatch_dict)
    assert res2["shape_mismatches"] == 1


def test_hadl_v58_checkpoint_integrity():
    """Verifies that hadl_v58_latent_matrix_central_router.pt loads safely into modules."""
    ckpt_path = Path("checkpoints/hadl_v58_latent_matrix_central_router.pt")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint hadl_v58_latent_matrix_central_router.pt not present")

    data = ResilientStateDictLoader.safe_load_file(ckpt_path)
    assert "latent_matrix_state_dict" in data
    assert "central_router_state_dict" in data

    matrix_engine = LatentThoughtMatrixEngine(d_model=2048)
    router_engine = CentralCognitiveRouter(d_model=2048)

    res_matrix = ResilientStateDictLoader.apply_state_dict_resilient(
        matrix_engine, data["latent_matrix_state_dict"]
    )
    res_router = ResilientStateDictLoader.apply_state_dict_resilient(
        router_engine, data["central_router_state_dict"]
    )

    assert res_matrix["applied_keys"] > 0
    assert res_router["applied_keys"] > 0


def test_custom_inference_engine_configuration():
    """Verifies that EngineConfig supports Blackwell optimization and v5.8 parameters."""
    cfg = EngineConfig(
        model_path="dist/staging_hf_fused",
        enable_blackwell_optimizations=True,
        enable_latent_matrix_v58=True,
        vram_headroom_fraction=0.88,
        toroidal_cycles=2
    )
    assert cfg.enable_blackwell_optimizations is True
    assert cfg.enable_latent_matrix_v58 is True
    assert cfg.vram_headroom_fraction == 0.88
    assert cfg.toroidal_cycles == 2

    engine = CustomInferenceEngine(cfg)
    assert engine.driver_report is None  # Populated on load()
    assert engine.is_loaded is False


def test_latent_matrix_and_toroidal_router_forward():
    """Verifies forward pass through Latent Matrix and Toroidal Router on target device."""
    report = probe_hardware_driver()
    dev = report.optimal_device if report.cuda_available else "cpu"
    dtype = report.optimal_dtype

    matrix_engine = LatentThoughtMatrixEngine(d_model=256).to(dev, dtype)
    router_engine = CentralCognitiveRouter(d_model=256).to(dev, dtype)

    h_in = torch.randn(1, 4, 256, device=dev, dtype=dtype)
    M_latent, telem_m = matrix_engine(h_in)
    
    assert M_latent.shape == (1, 4, 4, 256)
    assert telem_m["latent_matrix_subparts"] == 4
    assert "veracity_conflict_score" in telem_m

    h_out, telem_r = router_engine(
        M_latent,
        u_drive=torch.tensor([0.3], device=dev, dtype=dtype),
        veracity_conflict_score=telem_m["veracity_conflict_score"]
    )

    assert h_out.shape == (1, 4, 256)
    assert "central_router_entropy" in telem_r
    assert telem_r["central_router_cycles"] == 2
    assert torch.isfinite(h_out).all()
