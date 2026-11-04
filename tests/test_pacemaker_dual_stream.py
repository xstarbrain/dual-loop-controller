"""
Unit Tests for Pre-Router Spiking Pacemaker, Concurrent Dual-Stream Synthesizer,
Tour Leader Attractor, and Dimension-Aware Personality Scaling
================================================================================
HADL v5.5 Neuromorphic Architecture Verification Suite
"""

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dual_loop.neuromorphic import (
    ModelPersonalityAdapter,
    PersonalityProfile,
    PreRouterSpikingPacemaker,
    TourLeaderAttractor,
    ConcurrentDualStreamSynthesizer,
    FastSigmoidSurrogate,
    spike_fn,
    NeuromorphicSpikingBrain,
    HADLv5SpikingBrainModelWrapper,
    attach_hadl_v5_spiking_brain,
    DirectionalSafetyProjection,
)


def test_personality_scaling_laws_across_dimensions():
    """
    Verifies physical and mathematical scaling laws:
    - tau_m(D) = tau_0 * sqrt(D / D_0)
    - V_th(D) = V_0 * sqrt(D / D_0)
    - beta(D) = beta_0^(sqrt(D_0 / D))
    - D_c(D) = min(1024, max(256, D // 2))
    - eta_plastic(D) = eta_0 * (D_0 / D)
    """
    # 1. Reference dimension (Qwen 2B: D=2048)
    p_2048 = ModelPersonalityAdapter.calibrate(2048)
    assert p_2048.d_model == 2048
    assert math.isclose(p_2048.tau_m, 20.0, abs_tol=0.1)
    assert math.isclose(p_2048.v_th, 1.0, abs_tol=0.05)
    assert math.isclose(p_2048.beta, 0.85, abs_tol=0.05)
    assert p_2048.d_bottleneck == 1024
    assert math.isclose(p_2048.eta_plastic, 0.05, abs_tol=0.01)

    # 2. Intermediate dimension (Gemma 2B: D=2304)
    p_2304 = ModelPersonalityAdapter.calibrate(2304)
    assert p_2304.d_model == 2304
    assert p_2304.tau_m > p_2048.tau_m
    assert p_2304.v_th > p_2048.v_th
    assert p_2304.d_bottleneck == 1024

    # 3. Large dimension (LLaMA-3 8B: D=4096)
    p_4096 = ModelPersonalityAdapter.calibrate(4096)
    assert p_4096.d_model == 4096
    expected_ratio = math.sqrt(4096 / 2048)  # sqrt(2) ~ 1.414
    assert math.isclose(p_4096.tau_m, 20.0 * expected_ratio, abs_tol=0.2)
    assert math.isclose(p_4096.v_th, 1.0 * expected_ratio, abs_tol=0.05)
    assert p_4096.beta > p_2048.beta  # Slower leak (longer memory) in larger dimension
    assert p_4096.eta_plastic < p_2048.eta_plastic  # Conservative plastic learning


def test_pacemaker_membrane_integration_and_spike_emission():
    """
    Verifies that early Layer 2 Pacemaker:
    1. Remains quiet (0 spikes, low driving signal) on routine calm activations.
    2. Emits action potentials s(t)=1 and strong continuous u_drive on ambiguous/high-saliency inputs.
    3. Resets membrane potential upon spike via refractory mechanism.
    """
    d_model = 256
    pacemaker = PreRouterSpikingPacemaker(d_model=d_model, target_firing_rate=0.20)
    pacemaker.eval()

    # Sub-threshold calm inputs
    h_calm = torch.randn(2, 8, d_model) * 0.01
    spikes_calm, u_drive_calm, telem_calm = pacemaker(h_calm)
    assert spikes_calm.shape == (2, 8, 1)
    assert u_drive_calm.shape == (2, 8, 1)
    assert spikes_calm.sum().item() == 0
    assert abs(telem_calm["pacemaker_u_drive_mean"]) < 0.20

    # Reset state
    pacemaker.reset_state()
    assert pacemaker.v_membrane.sum().item() == 0.0

    # High-saliency ambiguous input: strong current should trigger spikes
    h_ambiguous = torch.randn(2, 8, d_model) * 15.0
    spikes_amb, u_drive_amb, telem_amb = pacemaker(h_ambiguous)
    assert spikes_amb.sum().item() > 0
    assert telem_amb["pacemaker_spikes_fired"] > 0
    assert torch.all(u_drive_amb >= -1.0) and torch.all(u_drive_amb <= 1.0)


def test_orthogonal_symplectic_projection_inner_product_preservation():
    """
    Verifies the mathematical theorem:
        P_perp(h_S1) Delta = Delta - (<Delta, h_S1> / (||h_S1||^2 + eps)) * h_S1
    Strictly satisfies:
        <P_perp(h_S1) Delta, h_S1> = 0
        <h_S1 + P_perp(h_S1) Delta, h_S1> = ||h_S1||^2
    This guarantees zero degradation to System 1's trajectory and fluency!
    """
    B, S, D = 4, 16, 512
    h_s1 = torch.randn(B, S, D)
    delta_s2 = torch.randn(B, S, D)

    delta_perp = ConcurrentDualStreamSynthesizer.orthogonal_symplectic_project(delta_s2, h_s1)

    # 1. Orthogonality test: <delta_perp, h_s1> must be 0
    inner_products = torch.sum(delta_perp * h_s1, dim=-1)  # [B, S]
    cos_sim = inner_products / (torch.norm(delta_perp, dim=-1) * torch.norm(h_s1, dim=-1) + 1e-8)
    assert torch.allclose(cos_sim, torch.zeros_like(cos_sim), atol=1e-5)
    assert torch.allclose(inner_products, torch.zeros_like(inner_products), atol=1e-4)

    # 2. Inner product preservation: <h_s1 + delta_perp, h_s1> == ||h_s1||^2
    h_fused = h_s1 + delta_perp
    fused_inner = torch.sum(h_fused * h_s1, dim=-1)
    norm_sq = torch.sum(h_s1 * h_s1, dim=-1)
    assert torch.allclose(fused_inner, norm_sq, atol=1e-4)


def test_tour_leader_attractor_extrinsic_potential_restoration():
    """
    Verifies that Tour Leader Attractor:
    1. Exerts near-zero force (v_tour ~ 0) when representations are aligned with logic manifold.
    2. Exerts non-invasive restorative drift v_tour = -mu * nabla Phi_logic when contradiction arises.
    """
    d_model = 256
    tour_leader = TourLeaderAttractor(d_model=d_model, mu_tour=0.15)
    tour_leader.eval()

    # Create synthetic concept anchors
    anchors = F.normalize(torch.randn(10, d_model), dim=-1)

    # 1. State lying directly on the manifold (linear combination of anchors)
    weights = torch.softmax(torch.randn(2, 4, 10), dim=-1)
    h_valid = torch.matmul(weights, anchors)  # [2, 4, D] on manifold
    v_tour_valid, telem_valid = tour_leader(h_valid, concept_anchors=anchors)
    # Deadband ensures force is zero or minimal
    assert telem_valid["tour_restoring_force"] < 0.05

    # 2. Perturbed state with contradiction / defect
    perturbation = torch.randn_like(h_valid) * 5.0
    h_defective = h_valid + perturbation
    v_tour_defective, telem_defective = tour_leader(h_defective, concept_anchors=anchors)
    assert telem_defective["tour_restoring_force"] > 0.05
    assert telem_defective["tour_active"] is True
    # Restoring vector should have opposite sign of perturbation
    inner_with_perturbation = torch.sum(v_tour_defective * perturbation, dim=-1).mean().item()
    assert inner_with_perturbation < 0  # Restores back towards manifold!


def test_concurrent_dual_stream_synthesis_flow():
    """
    Tests complete ConcurrentDualStreamSynthesizer execution:
    - System 1 (fast reflex) + System 2 (Banach contraction steps K=1..3)
    - Pacemaker driving weight modulation
    - Symplectic fusion
    """
    d_model = 256
    synthesizer = ConcurrentDualStreamSynthesizer(d_model=d_model, max_deliberation_steps=3)
    synthesizer.eval()

    h_s1 = torch.randn(2, 6, d_model)
    u_drive_calm = torch.zeros(2, 6, 1)

    # When driving signal is 0, System 2 modulation is 0
    h_fused_calm, telem_calm = synthesizer(h_s1, u_drive=u_drive_calm, num_deliberation_steps=1)
    assert h_fused_calm.shape == h_s1.shape
    assert telem_calm["s2_drive_weight"] == 0.0

    # When driving signal is active, System 2 delivers orthogonal correction
    u_drive_active = torch.ones(2, 6, 1) * 0.8
    h_fused_active, telem_active = synthesizer(h_s1, u_drive=u_drive_active, num_deliberation_steps=2)
    assert telem_active["s2_drive_weight"] > 0.5
    assert telem_active["s2_delta_norm"] > 0.0
    assert telem_active["deliberation_steps_k"] == 2


def test_automatic_loading_of_all_three_checkpoints():
    """
    Verifies that NeuromorphicSpikingBrain automatically loads:
    1. vgni_lbu_autonomous_stabilized_qwen2b.pt (concept anchors & LBU weights).
    2. xstar_brain_post_20min_cumulative.pt (w_plastic matrix).
    3. polynomial_embedded_router_weights.pt (router weights).
    """
    ckpt_dir = Path("checkpoints")
    assert ckpt_dir.exists(), "checkpoints directory must exist"

    brain = NeuromorphicSpikingBrain(d_model=2048)
    loaded_status = brain.load_trained_checkpoints(ckpt_dir)

    assert loaded_status["vgni_lbu"] is True, "Failed to load VGNI+LBU checkpoint"
    assert loaded_status["plastic_cumulative"] is True, "Failed to load cumulative plastic checkpoint"
    assert loaded_status["embedded_router"] is True, "Failed to load polynomial embedded router checkpoint"

    # Verify that concept anchors were populated
    num_stored = int(brain.vgni.num_stored.item())
    assert num_stored > 0, "VGNI concept anchors should be non-zero after loading"

    # Verify that plastic weights were populated
    plastic_norm = float(torch.norm(brain.plastic_unit.w_plastic).item())
    assert plastic_norm > 0.0, "w_plastic should be non-zero"

    # Verify embedded router weights
    router_norm = float(torch.norm(brain.embedded_router.weight).item())
    assert router_norm > 0.0, "embedded router weight should be non-zero"


def test_wrapper_dual_stream_hooks_and_directional_safety():
    """
    Verifies that HADLv5SpikingBrainModelWrapper:
    1. Hooks Layer 2 (PreRouterSpikingPacemaker).
    2. Hooks Layer 7 (ConcurrentDualStreamSynthesizer).
    3. Hooks final layer with DirectionalSafetyProjection before LM Head.
    """
    # Create mock transformer model
    class MockTransformer(nn.Module):
        def __init__(self, d_model=128, num_layers=8, vocab_size=1000):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(num_layers)
            ])
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, x):
            h = x
            for layer in self.layers:
                h = layer(h)
            logits = self.lm_head(h)
            return logits

    d_model = 128
    base_model = MockTransformer(d_model=d_model, num_layers=8, vocab_size=500)
    wrapper = attach_hadl_v5_spiking_brain(
        base_model=base_model,
        target_layer_idx=5,
        pacemaker_layer_idx=1,
        d_model=d_model,
        deliberation_steps_k=2,
        auto_load_checkpoints=False,
    )

    dummy_input = torch.randn(1, 4, d_model)
    out_logits = wrapper(dummy_input)

    assert out_logits.shape == (1, 4, 500)
    assert len(wrapper.generation_history) > 0
    telem = wrapper.last_telemetry

    # Verify pacemaker telemetry was captured
    assert "pacemaker_u_drive_mean" in telem
    # Verify dual stream telemetry was captured
    assert "dual_stream_telemetry" in telem
    dual_telem = telem["dual_stream_telemetry"]
    assert "s2_delta_norm" in dual_telem


def test_directional_safety_monotonic_ranking_preservation():
    """
    Verifies that DirectionalSafetyProjection prevents candidate deliberation
    deltas from flipping confident top-1 predictions.
    """
    d_model = 64
    vocab_size = 100
    lm_head = nn.Linear(d_model, vocab_size, bias=False)
    safety = DirectionalSafetyProjection()

    # Base hidden state produces confident top-1 prediction
    h_base = torch.randn(2, d_model)
    with torch.no_grad():
        logits_base = lm_head(h_base)
        top1_base = torch.argmax(logits_base, dim=-1)

    # Harmful delta deliberately aimed at flipping top-1 to top-2
    with torch.no_grad():
        topk = torch.topk(logits_base, k=2, dim=-1)
        top1_idx = topk.indices[:, 0]
        top2_idx = topk.indices[:, 1]
        W = lm_head.weight
        # Direction that reduces margin
        d = W[top1_idx] - W[top2_idx]
        harmful_delta = -2.0 * d

    # Project out harmful component
    delta_safe, telem = safety(harmful_delta, h_base, lm_head)

    # Verify that with delta_safe, top-1 is preserved!
    h_protected = h_base + delta_safe
    logits_protected = lm_head(h_protected)
    top1_protected = torch.argmax(logits_protected, dim=-1)

    assert torch.equal(top1_base, top1_protected), "Directional safety must preserve top-1 prediction"
    assert telem["correction_magnitude"].mean().item() > 0.0


def test_cli_plan_teamwork_preview_and_boost():
    """
    Verifies that CLI commands:
    1. /boost scales Banach contraction steps K=1..3 and threshold V_th.
    2. /plan de-aliases from /goal and inspects architectural routing.
    3. /teamwork-preview runs dual-channel stream.
    """
    from unittest.mock import MagicMock
    from xstar.inference.engine import CustomInferenceEngine, EngineConfig, GenerationMetrics
    from xstar.inference.cli_app import InteractiveCLI

    cfg = EngineConfig(enable_brain=False, enable_adapter=False)
    engine = CustomInferenceEngine(cfg)
    cli = InteractiveCLI(engine)
    cli.execute_turn = MagicMock()

    # Test /boost tuning
    res1 = engine.set_boost(enabled=True, level=1)
    assert res1["boost_level"] == 1
    assert res1["deliberation_steps_k"] == 1
    res4 = engine.set_boost(enabled=True, level=4)
    assert res4["boost_level"] == 4
    assert res4["deliberation_steps_k"] == 3

    # Test /plan handling (fast route inspection without 4-phase triad)
    cli.handle_plan("Analyze logical consistency of matrix eigenvalues")
    assert cli.execute_turn.called
    plan_prompt = cli.execute_turn.call_args[0][0]
    assert "concise, direct architectural milestone" in plan_prompt

    # Test /teamwork-preview handling
    cli.engine.stream_chat = MagicMock(return_value=[("Token reflex", GenerationMetrics(s2_delta_norm=0.15, tour_restoring_force=0.08))])
    cli.handle_teamwork_preview("Check if System 1 and System 2 fuse orthogonally")
    assert any("Check if System 1" in m["content"] for m in cli.messages)

    # Test /boost status
    cli.handle_boost("status")


def test_directional_safety_projection_export_and_router_wiring():
    """
    Verifies:
    1. Clean export of DirectionalSafetyProjection from both dual_loop and dual_loop.neuromorphic.
    2. Embedded router is actively evaluated at Layer 2 driven by pacemaker signals.
    """
    from dual_loop import DirectionalSafetyProjection as DSP_root
    from dual_loop.neuromorphic import DirectionalSafetyProjection as DSP_neuro
    assert DSP_root is DSP_neuro

    brain = NeuromorphicSpikingBrain(d_model=256)
    h_layer2 = torch.randn(2, 6, 256)
    spikes, u_drive, telem = brain.forward_layer2_pacemaker(h_layer2)

    assert "router_route_mode" in telem
    assert telem["router_route_mode"] in ["bypass", "jalur_tengah", "heavy"]
    assert "router_p_bypass" in telem
    assert "router_p_hybrid" in telem
    assert "router_p_heavy" in telem
    total_p = telem["router_p_bypass"] + telem["router_p_hybrid"] + telem["router_p_heavy"]
    assert math.isclose(total_p, 1.0, abs_tol=1e-3)


def test_system2_banach_contraction_anchor_and_k3_execution():
    """
    Verifies that:
    1. K=3 deliberation steps actually execute without truncation.
    2. Conditioning on input anchor z_0 avoids collapsing to zero as K increases.
    """
    d_model = 256
    synthesizer = ConcurrentDualStreamSynthesizer(d_model=d_model, max_deliberation_steps=3)
    synthesizer.eval()

    h = torch.randn(2, 4, d_model)
    d1, norm1, steps1 = synthesizer.forward_system2_contraction(h, num_steps=1)
    d2, norm2, steps2 = synthesizer.forward_system2_contraction(h, num_steps=2)
    d3, norm3, steps3 = synthesizer.forward_system2_contraction(h, num_steps=3)

    assert steps1 == 1
    assert steps2 == 2
    assert steps3 == 3, "K=3 must execute 3 iterations without silent truncation"

    # Anchored contraction converges to a non-zero fixed point without decaying to 0
    assert norm1 > 0.1
    assert norm2 > 0.1
    assert norm3 > 0.1

    # End-to-end forward reflects K=3
    u = torch.ones(2, 4, 1) * 0.5
    _, telem = synthesizer(h, u_drive=u, num_deliberation_steps=3)
    assert telem["deliberation_steps_k"] == 3


def test_non_negative_drive_modulation_and_membrane_drift_stability():
    """
    Verifies:
    1. Subthreshold driving signals (u_drive < 0) yield non-negative drive_weight = 0.0,
       preventing inverted Banach steps.
    2. Prolonged negative sensory current does not hyperpolarize membrane potential below 0.0,
       preventing the neuron from becoming permanently inhibited/frozen.
    """
    d_model = 256
    synthesizer = ConcurrentDualStreamSynthesizer(d_model=d_model)
    synthesizer.eval()

    h = torch.randn(2, 4, d_model)
    u_neg = -torch.ones(2, 4, 1) * 0.75
    h_fused, telem = synthesizer(h, u_drive=u_neg, num_deliberation_steps=1)

    assert telem["s2_drive_weight"] == 0.0, "Negative driving signal must yield 0.0 modulation"
    # When drive weight is 0 and no concept anchors, h_fused equals h_s1 (pure System 1 reflex)
    assert torch.allclose(h_fused, h, atol=1e-4)

    # Membrane drift stability test
    pacemaker = PreRouterSpikingPacemaker(d_model=d_model)
    pacemaker.eval()
    pacemaker.w_sensor.weight.data.copy_(torch.randn(1, d_model))

    # Drive with negative sensory current over 40 tokens
    h_neg = -pacemaker.w_sensor.weight.data.unsqueeze(0) * 10.0
    for _ in range(40):
        pacemaker(h_neg)

    # Membrane potential must not drop below resting potential 0.0
    assert pacemaker.v_membrane.item() >= 0.0, "Membrane potential must not drop below 0.0"

    # Next, a high-saliency token must fire immediately without being blocked by prior negative tokens
    h_pos = pacemaker.w_sensor.weight.data.unsqueeze(0) * 15.0
    spikes, u_drive, p_telem = pacemaker(h_pos)
    assert spikes.sum().item() > 0, "High-saliency token must emit spikes without delay"


def test_wrapper_bounded_memory_and_constant_time_telemetry():
    """
    Verifies that HADLv5SpikingBrainModelWrapper:
    1. Caps generation_history to prevent unbounded memory growth.
    2. Maintains accurate running counters for spikes and step counts.
    """
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(8)])
            self.lm_head = torch.nn.Linear(64, 100, bias=False)

        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return self.lm_head(x)

    base = TinyModel()
    wrapper = attach_hadl_v5_spiking_brain(base, d_model=64, pacemaker_layer_idx=1, target_layer_idx=5, auto_load_checkpoints=False)
    x = torch.randn(1, 1, 64)

    # Simulate 300 sequential generation steps
    for _ in range(300):
        _ = wrapper(x)

    assert len(wrapper.generation_history) <= 256, "generation_history must be bounded to prevent memory leaks"
    assert wrapper.last_telemetry["generation_step_count"] == 300

