"""
Unit Tests for HADL v5.9 Architectural Upgrades:
1. OuterDynamicScratchDimension (elastic OS working memory alongside D_static = 2048).
2. Order-Statistic Consensus Norm Bound via Mean & Median of per-token subpart minimums.
3. Closed-Form Mahalanobis Epistemic Attenuation & Skew-Symmetric Bilinear Contradiction Detection.
"""

import time
import torch
import pytest

from dual_loop.neuromorphic.pacemaker_router import (
    OuterDynamicScratchDimension,
    LatentThoughtMatrixEngine,
    CentralCognitiveRouter,
)


def test_outer_dynamic_scratch_dimension_scaling():
    """Verifies dynamic width counting, Tensor Core 32-block alignment, learned slot expansion, and nullspace projection."""
    torch.manual_seed(42)
    d_model = 256
    scratch = OuterDynamicScratchDimension(d_model=d_model, d_min=64, d_max=256, block_size=32, slot_capacity=32)

    # 1. Calm, low-dispersion input with 0 learned slots
    h_calm = torch.ones(2, 4, d_model) * 0.5 + torch.randn(2, 4, d_model) * 0.02
    u_calm = torch.zeros(2, 4, 1)
    z_calm, telem_calm = scratch.lift_to_scratch(h_calm, u_drive=u_calm, learned_slots=0)

    d_act_calm = telem_calm["active_scratch_dim"]
    assert d_act_calm % 32 == 0, f"Expected multiple of 32, got {d_act_calm}"
    assert z_calm.shape == (2, 4, d_act_calm)

    # 2. High-complexity input with strong pacemaker drive and expanded learned slots
    scratch.register_learned_slot(48)
    h_complex = torch.randn(2, 64, d_model) * 3.0
    u_high = torch.ones(2, 64, 1) * 2.0
    z_complex, telem_complex = scratch.lift_to_scratch(h_complex, u_drive=u_high)

    d_act_complex = telem_complex["active_scratch_dim"]
    assert d_act_complex % 32 == 0
    assert d_act_complex > d_act_calm, f"Expected dynamic expansion ({d_act_complex} > {d_act_calm})"
    assert z_complex.shape == (2, 64, d_act_complex)

    # 3. Write subsystem state and project back to static space strictly orthogonal to h_complex
    h_sub = torch.randn_like(h_complex)
    z_updated = scratch.write_subsystem_to_scratch(z_complex, h_sub, weight=0.4)
    delta_ortho = scratch.project_from_scratch(z_updated, h_complex)

    inner_prod = torch.abs(torch.sum(delta_ortho * h_complex, dim=-1)).mean().item()
    base_norm_sq = torch.sum(h_complex * h_complex, dim=-1).mean().item()
    assert inner_prod / base_norm_sq < 1e-5, f"Expected strict orthogonality, got {inner_prod / base_norm_sq}"


def test_order_statistic_consensus_bound_mean_median_min():
    """
    Verifies that single-subpart noise keeps eta_dyn near eta_floor (because min across subparts stays ~0),
    whereas genuine multi-subpart consensus raises both Mean(r_min) and Median(r_min) to unlock higher eta_dyn.
    """
    torch.manual_seed(7)
    B, S, R, D = 2, 16, 4, 128
    router = CentralCognitiveRouter(
        d_model=D,
        num_subparts=R,
        enable_order_statistic_bound=True,
        eta_floor=0.008,
        eta_ceil=0.065,
        kappa_consensus=0.40,
    )

    m0 = torch.randn(B, S, D)
    # Case A: Single-subpart wild noise spike (M_1 huge, M_2 and M_3 near zero)
    m1_noisy = torch.randn(B, S, D) * 5.0
    m2_quiet = torch.randn(B, S, D) * 1e-4
    m3_quiet = torch.randn(B, S, D) * 1e-4
    M_noisy = torch.stack([m0, m1_noisy, m2_quiet, m3_quiet], dim=2)

    eta_noisy, telem_noisy = router.compute_order_statistic_bound(
        M_noisy, m0, u_mean=torch.ones(B, 1) * 0.5
    )

    # Case B: Genuine multi-subpart consensus (M_1, M_2, M_3 all concertedly active at ~0.12x anchor norm)
    m1_cons = torch.randn(B, S, D) * 0.12
    m2_cons = torch.randn(B, S, D) * 0.12
    m3_cons = torch.randn(B, S, D) * 0.12
    M_cons = torch.stack([m0, m1_cons, m2_cons, m3_cons], dim=2)

    eta_cons, telem_cons = router.compute_order_statistic_bound(
        M_cons, m0, u_mean=torch.ones(B, 1) * 1.0
    )

    assert telem_noisy["r_min_mean"] < 0.005
    assert telem_noisy["r_min_median"] < 0.005
    assert telem_cons["r_min_mean"] > 0.05
    assert telem_cons["r_min_median"] > 0.05
    assert float(eta_cons.mean().item()) > float(eta_noisy.mean().item()) * 2.0, (
        f"Consensus bound ({eta_cons.mean().item():.5f}) should significantly exceed "
        f"single-subpart noise bound ({eta_noisy.mean().item():.5f})"
    )


def test_mahalanobis_epistemic_attenuation_and_skew_bilinear():
    """
    Verifies:
    1. Skew-symmetric bilinear operator J = A - A^T yields 0.0 on self-consistent sequences (z_pre == z_post)
       and non-zero on contradictory premise-conclusion sequences.
    2. Mahalanobis epistemic attenuation u_epistemic damps Out-of-Distribution (OOD) inputs smoothly
       with sub-millisecond execution time.
    """
    torch.manual_seed(99)
    D = 128
    engine = LatentThoughtMatrixEngine(d_model=D, bottleneck_dim=32, veracity_dim=32, mahalanobis_threshold=1.2)
    # Give veracity_head non-zero weights so raw_logit is active
    with torch.no_grad():
        engine.veracity_head.weight.fill_(0.25)

    # 1. Self-consistent sequence: all tokens identical -> z_pre == z_post -> z^T J z == 0
    token_vec = torch.randn(1, 1, D)
    h_consistent = token_vec.repeat(1, 8, 1)
    _, _, _ = engine.evaluate_veracity_subspace(h_consistent)
    assert abs(engine.last_bilinear_contradiction) < 1e-6, (
        f"Expected 0 bilinear contradiction on self-consistent state, got {engine.last_bilinear_contradiction}"
    )

    # 2. Contradictory sequence: premise (first 4 tokens) vs twisted conclusion (last 4 tokens)
    h_pre = torch.randn(1, 4, D)
    h_post = -1.5 * torch.roll(h_pre, shifts=1, dims=-1)
    h_contradict = torch.cat([h_pre, h_post], dim=1)
    _, _, z_prop_in = engine.evaluate_veracity_subspace(h_contradict)
    assert abs(engine.last_bilinear_contradiction) > 1e-3, (
        "Expected non-zero bilinear contradiction on twisted premise-conclusion sequence"
    )

    # 3. Calibrate Mahalanobis manifold around tight in-distribution cluster and test OOD attenuation
    with torch.no_grad():
        engine.veracity_ema_mean.copy_(z_prop_in.squeeze(0))
        engine.veracity_ema_var.fill_(0.02)  # Tight in-distribution variance
        engine.veracity_ema_steps.fill_(10)

    # In-distribution evaluation (matches mean)
    _, score_in, _ = engine.evaluate_veracity_subspace(h_contradict)
    atten_in = engine.last_epistemic_attenuation
    assert abs(atten_in - 1.0) < 1e-5, f"Expected in-distribution attenuation 1.0, got {atten_in}"

    # Out-of-Distribution evaluation (far from mean relative to variance 0.02)
    h_ood = -h_contradict * 5.0
    t0 = time.perf_counter()
    _, score_ood, _ = engine.evaluate_veracity_subspace(h_ood)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    atten_ood = engine.last_epistemic_attenuation

    assert atten_ood < 0.50, f"Expected strong OOD epistemic attenuation (< 0.50), got {atten_ood}"
    assert elapsed_ms < 15.0, f"Expected zero-latency overhead, took {elapsed_ms:.2f} ms"


def test_central_router_end_to_end_v59_orthogonality_and_telemetry():
    """Verifies end-to-end LatentThoughtMatrixEngine + CentralCognitiveRouter v5.9 telemetry and orthogonality."""
    torch.manual_seed(123)
    B, S, D = 2, 12, 256
    matrix_engine = LatentThoughtMatrixEngine(d_model=D, bottleneck_dim=64, veracity_dim=64)
    router = CentralCognitiveRouter(d_model=D, router_dim=32, disclosure_dim=64, num_cycles=2)

    h = torch.randn(B, S, D)
    u_drive = torch.ones(B, S, 1) * 0.8
    M, m_telem = matrix_engine(h)
    h_routed, r_telem = router(
        M,
        u_drive=u_drive,
        veracity_conflict_score=m_telem["veracity_conflict_score"],
        learned_slots=16,
    )

    assert h_routed.shape == (B, S, D)
    assert r_telem["orthogonal_inner_product_error"] < 1e-6
    assert r_telem["inner_product_preservation_ratio"] >= 0.99999
    assert "active_scratch_dim" in r_telem
    assert "dynamic_norm_bound" in r_telem
    assert "r_min_mean" in r_telem
    assert "r_min_median" in r_telem
    assert "mahalanobis_ood_distance" in m_telem
    assert "epistemic_attenuation" in m_telem
    assert "bilinear_contradiction_score" in m_telem


def test_spiking_brain_v59_reentry_scratch_coupling_and_slot_expansion():
    """Verifies NeuromorphicSpikingBrain Layer 7 -> Re-entry outer scratch coupling and closed-loop slot growth."""
    from dual_loop.neuromorphic.spiking_gate import NeuromorphicSpikingBrain

    torch.manual_seed(77)
    brain = NeuromorphicSpikingBrain(d_model=64)
    h7 = torch.randn(1, 6, 64) * 1.5
    h_out, info = brain.forward_layer7_transition(h7)

    assert h_out.shape == (1, 6, 64)
    assert "active_scratch_dim" in info
    assert "dynamic_norm_bound" in info
    assert info["active_scratch_dim"] >= 64

    # Deep layer re-entry should couple the outer scratch working memory
    h14 = torch.randn(1, 6, 64)
    h14_out, r_telem = brain.forward_deep_layer_reentry(h14, layer_idx=14)
    assert h14_out.shape == (1, 6, 64)
    assert r_telem.get("outer_scratch_reentry_coupled") is True
    assert r_telem.get("outer_scratch_active_dim", 0) >= 64

    # Closed-loop verified feedback should expand learned scratch slots
    c_telem = brain.verify_and_adapt_closed_loop(
        prompt="What is 2 + 2?",
        response="2 + 2 equals 4.",
        external_verdict=1.0,
    )
    assert c_telem.get("outer_scratch_learned_slots", 0) >= 1

