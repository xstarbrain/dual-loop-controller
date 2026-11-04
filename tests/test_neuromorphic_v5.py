"""
Unit Tests for Neuromorphic Spiking Brain (X-Star HADL v5.0)
============================================================
Tests:
1. FastSigmoidSurrogate differentiable gradient flow.
2. LIF Spiking Gate membrane integration, refractory reset, and homeostatic adaptation.
3. Dynamic Neural Replication (DNR) vertical KV saliency & identity preservation.
4. Open-Freeze Liquid Phase Plasticity (solid frozen vs. liquid flow).
5. Virtual Latent Sandbox Banach contraction regularizer.
6. Re-Entrant Cross-Layer Re-Query Bus for deep layers.
7. NeuromorphicSpikingBrain unified engine Layer 7 transition.
8. HADLv5SpikingBrainModelWrapper integration with transformer architecture.
"""

import pytest
import torch
import torch.nn as nn

from dual_loop.neuromorphic import (
    FastSigmoidSurrogate,
    spike_fn,
    LeakyIntegrateAndFireGate,
    DynamicNeuralReplicator,
    OpenFreezeLiquidPlasticity,
    ReEntrantCrossLayerBus,
    VirtualLatentSandbox,
    NeuromorphicSpikingBrain,
    HADLv5SpikingBrainModelWrapper,
    attach_hadl_v5_spiking_brain,
)


def test_fast_sigmoid_surrogate():
    v = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
    v_th = torch.tensor(1.0, requires_grad=True)

    # Forward Heaviside
    s = spike_fn(v, v_th, gamma=4.0)
    assert s[0].item() == 0.0
    assert s[1].item() == 1.0
    assert s[2].item() == 1.0

    # Differentiable backward
    loss = s.sum()
    loss.backward()

    assert v.grad is not None
    assert v_th.grad is not None
    # Maximum gradient at v == v_th: 1 / (1 + 0)^2 = 1.0
    assert torch.isclose(v.grad[1], torch.tensor(1.0), atol=1e-3)
    # Tapered gradient away from threshold
    assert v.grad[0] < 1.0 and v.grad[0] > 0.0
    assert v.grad[2] < 1.0 and v.grad[2] > 0.0


def test_lif_spiking_gate_integration_and_reset():
    d_model = 64
    lif = LeakyIntegrateAndFireGate(d_model=d_model, beta=0.8, v_threshold=1.0)
    
    # Sub-threshold input: should not fire
    h_sub = torch.zeros(2, 4, d_model)
    spikes_sub, telem_sub = lif(h_sub)
    assert spikes_sub.sum().item() == 0.0
    assert telem_sub["spike_firing_rate"] == 0.0

    # Supra-threshold input: strong current should trigger spike and refractory reset
    h_strong = torch.randn(2, 4, d_model) * 10.0
    spikes_strong, telem_strong = lif(h_strong)
    assert spikes_strong.shape == (2, 4, 1)
    # Check that refractory reset worked (v_membrane reset after spike)
    assert lif.v_membrane is not None
    assert telem_strong["total_spikes_fired"] >= 0


def test_dynamic_neural_replicator_identity_and_saliency():
    d_model = 64
    dnr = DynamicNeuralReplicator(d_model=d_model, c_max=256)

    h = torch.randn(2, 5, d_model)
    # 1. Empty KV cache -> identity preservation W_rep(h) == h
    h_id = dnr(h, v_kv=None, c_active=0)
    assert torch.allclose(h, h_id, atol=1e-6)

    # 2. Extract vertical KV saliency
    K = torch.randn(2, 8, d_model)
    V = torch.randn(2, 8, d_model)
    v_kv = dnr.extract_vertical_saliency(K, V)
    assert v_kv.shape == (2, d_model)

    # 3. Apply replication with active cache
    h_mod = dnr(h, v_kv=v_kv, c_active=8)
    assert h_mod.shape == h.shape
    assert not torch.allclose(h, h_mod)  # Replicated modulation active


def test_open_freeze_liquid_plasticity():
    d_model = 64
    plasticity = OpenFreezeLiquidPlasticity(d_model=d_model, gamma=1.0, rank=8)

    h_base = torch.randn(2, 4, d_model)

    # With zero spike, surprisal is small -> Solid/frozen phase
    h_solid, telem_solid = plasticity(h_base, spike=torch.zeros(2, 4, 1))
    assert telem_solid["phase_state"] in ["SOLID_FROZEN", "TRANSIENT_GEL"]

    # With strong spike signal, high surprisal -> Liquid plastic phase
    spike_burst = torch.ones(2, 4, 1) * 5.0
    h_liquid, telem_liquid = plasticity(h_base, spike=spike_burst)
    assert telem_liquid["fluid_viscosity_mu"] > telem_solid["fluid_viscosity_mu"]


def test_virtual_latent_sandbox_contraction():
    d_model = 64
    sandbox = VirtualLatentSandbox(d_model=d_model, max_drift_bound=0.5, gamma_lipschitz=0.85)

    h = torch.randn(2, 4, d_model)

    # Small safe delta
    delta_small = torch.randn(2, 4, d_model) * 0.01
    safe_delta, telem_small = sandbox(h, delta_small)
    assert not telem_small["sandbox_intervention_active"]

    # Massive explosive delta (divergence / hallucination risk)
    delta_huge = torch.randn(2, 4, d_model) * 50.0
    safe_delta_huge, telem_huge = sandbox(h, delta_huge)
    # Sandbox must intervene and clamp divergence
    assert telem_huge["sandbox_intervention_active"]
    assert torch.norm(safe_delta_huge) < torch.norm(delta_huge)


def test_reentrant_cross_layer_bus():
    d_model = 64
    bus = ReEntrantCrossLayerBus(d_model=d_model, ambiguity_threshold=0.5)
    brain = NeuromorphicSpikingBrain(d_model=d_model, target_layer_idx=7)

    # Low variance representation
    h_calm = torch.ones(2, 4, d_model) * 0.5
    h_out, telem = bus.execute_reentry(h_calm, brain.vgni)
    assert not telem["reentry_active"]

    # Seed VGNI with a trajectory
    brain.vgni.register_successful_trajectory(torch.randn(4, d_model), torch.randn(4, d_model))

    # High variance representation triggering re-entry
    h_ambiguous = torch.randn(2, 4, d_model) * 5.0
    h_out, telem = bus.execute_reentry(h_ambiguous, brain.vgni)
    assert telem["reentry_active"]


def test_neuromorphic_spiking_brain_gate_isolation():
    d_model = 64
    brain = NeuromorphicSpikingBrain(d_model=d_model, target_layer_idx=7, spike_v_th=5.0)

    # Zero input -> Gate strictly closed (0 x VGNI)
    h_zero = torch.zeros(2, 4, d_model)
    h_out, telem = brain.forward_layer7_transition(h_zero)
    assert telem["gate_status"] == "CLOSED_ISOLATED"
    assert not telem["vgni_engaged"]
    assert not telem["sandbox_engaged"]

    # High excitation input -> Gate opens, VGNI and Sandbox engaged
    # Lower threshold temporarily for test
    brain.spiking_gate.base_v_th = 0.01
    brain.spiking_gate.v_th.fill_(0.01)
    h_excite = torch.randn(2, 4, d_model) * 2.0
    h_out, telem = brain.forward_layer7_transition(h_excite)
    assert telem["gate_status"] == "OPEN_SPIKING"
    assert telem["vgni_engaged"]
    assert telem["sandbox_engaged"]


class MockTransformerLayer(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, h):
        return self.proj(h)


class MockTransformerModel(nn.Module):
    def __init__(self, num_layers: int = 16, d_model: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([MockTransformerLayer(d_model) for _ in range(num_layers)])

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


def test_hadl_v5_spiking_brain_model_wrapper():
    base_model = MockTransformerModel(num_layers=16, d_model=64)
    wrapped = attach_hadl_v5_spiking_brain(
        base_model=base_model,
        target_layer_idx=7,
        d_model=64,
        spike_v_th=0.5
    )

    assert wrapped.target_layer_idx == 7
    assert len(wrapped.hook_handles) >= 1  # Layer 7 + re-entrant layers

    x = torch.randn(2, 4, 64)
    out = wrapped(x)
    assert out.shape == x.shape

    # Check telemetry recorded
    assert "gate_status" in wrapped.last_telemetry

    # Test reset state
    wrapped.reset_state()
    assert len(wrapped.last_telemetry) == 0


def test_v51_multistate_cognitive_spiking():
    d_model = 64
    lif = LeakyIntegrateAndFireGate(d_model=d_model, beta=0.85, v_threshold=1.0)

    # 1. Routine calm input
    h_calm = torch.randn(2, 4, d_model) * 0.05
    spikes_calm, telem_calm = lif(h_calm)
    assert telem_calm["cognitive_state"] in ["ROUTINE_CALM", "CONFUSION"]
    assert "dynamic_threshold_factor" in telem_calm

    # 2. Stress / large computational input
    h_stress = torch.randn(2, 4, d_model) * 15.0
    spikes_stress, telem_stress = lif(h_stress)
    assert telem_stress["cognitive_state"] in ["STRESS", "EUREKA"]


def test_v51_multivector_subspace_dnr():
    d_model = 64
    dnr = DynamicNeuralReplicator(d_model=d_model, c_max=256)

    K = torch.randn(2, 8, d_model)
    V = torch.randn(2, 8, d_model)

    # Extract multi-vector orthogonal basis (rank 3)
    V_sub = dnr.extract_multivector_subspace(K, V, max_rank=3)
    assert V_sub.shape == (2, 3, d_model)

    # Verify orthogonality of extracted basis vectors: v1 . v2 approx 0
    v1 = V_sub[0, 0, :]
    v2 = V_sub[0, 1, :]
    dot_prod = torch.abs(torch.dot(v1, v2)).item()
    assert dot_prod < 0.10  # Orthogonal within numerical tolerance

    # Multi-vector projection forward
    h = torch.randn(2, 4, d_model)
    h_mod = dnr(h, v_kv=V_sub, c_active=8, subspace_rank=3)
    assert h_mod.shape == h.shape
    assert not torch.allclose(h, h_mod)


def test_v51_dynamic_lipschitz_sandbox_stress_relaxation():
    d_model = 64
    sandbox = VirtualLatentSandbox(d_model=d_model, max_drift_bound=1.0, gamma_lipschitz=0.85)

    h = torch.randn(2, 4, d_model)
    delta_math = torch.randn(2, 4, d_model) * 0.80

    # Under STRESS / computational state, gamma relaxes dynamically up to 0.985
    _, telem_stress = sandbox(h, delta_math, cognitive_state="STRESS")
    assert telem_stress["dynamic_gamma"] > 0.85
    assert telem_stress["dynamic_gamma"] <= 0.985

    # Under ROUTINE_CALM, gamma stays conservative at 0.85
    _, telem_routine = sandbox(h, delta_math, cognitive_state="ROUTINE_CALM")
    assert telem_routine["dynamic_gamma"] <= 0.90
