"""
Comprehensive Test Suite for HADL v5.7 Unified Self-Evolving Multimodal AGI
===========================================================================
Covers all 4 architectural pillars:
1. Dynamic Self-Evolving Closed-Loop Learning (IntrinsicPopperianCycleVerifier,
   SynapticEligibilityTrace, OrthogonalNullspaceProjector, OpenFreezeLiquidPlasticity,
   Shadow KV Cache multi-token slicing fix, multi-pass generate_self_evolving,
   and benchmark evaluator negative-number boundary fixes).
2. Endogenous Custom Sound Synthesis (`freq suara` — EndogenousAcousticVocalOrgan
   Spiking-Harmonic Formant Synthesizer + Self-Listening Auditory Re-Entry Loop).
3. Progressive 2D-to-3D Human-Like Visual Imagination (Progressive2Dto3DImaginationCanvas:
   1D -> 2D Retinotopic Sketchpad -> 3D Gaussian/SDF Volumetric Lifting + SO(3)
   Mental Rotation + 3D-to-2D Splatting Verification).
4. Full Custom Inference Framework Alignment (xstar/inference/engine.py,
   cognitive_triad.py, server.py).
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Dict, Any

import pytest
import torch
import torch.nn as nn

from dual_loop.verification import IntrinsicPopperianCycleVerifier
from dual_loop.nullspace_engine import (
    OrthogonalNullspaceProjector,
    SynapticEligibilityTrace,
    NullspaceMemoryEngine,
)
from dual_loop.neuromorphic.acoustic_vocal_organ import EndogenousAcousticVocalOrgan
from dual_loop.neuromorphic.latent_imagination_canvas import Progressive2Dto3DImaginationCanvas
from dual_loop.neuromorphic.spiking_gate import (
    OpenFreezeLiquidPlasticity,
    EndogenousBanachPlasticUnit,
    NeuromorphicSpikingBrain,
    attach_hadl_v5_spiking_brain,
)
from dual_loop.neuromorphic.tokenless_latent_engine import TokenlessLatentThoughtEngine
from xstar.inference.engine import CustomInferenceEngine, EngineConfig, GenerationMetrics
from xstar.inference.cognitive_triad import CognitiveTriadRunner


# ==============================================================================
# PILLAR 1: Dynamic Self-Evolving Closed-Loop Learning & Bug Fixes
# ==============================================================================


def test_evaluator_negative_number_word_boundary_fix():
    """Verifies that negative numbers (-1, -7) and MBPP equivalents match in suite_1000q_20bench."""
    from scripts.suite_1000q_20bench import build_1000q_suite

    suite = build_1000q_suite()
    by_id = {q["id"]: q for q in suite}

    # math_14: P(x) = x^2 - 8x + 8 at x = 3 -> target is "-7"
    q_neg7 = by_id["math_14"]
    assert q_neg7["target"] == "-7"
    assert q_neg7["check"]("The value of P(3) is -7.") is True
    assert q_neg7["check"]("-7") is True
    assert q_neg7["check"]("The value is 7.") is False

    # math_20: P(x) = x^2 - 7x + 9 at x = 5 -> target is "-1"
    q_neg1 = by_id["math_20"]
    assert q_neg1["target"] == "-1"
    assert q_neg1["check"]("The value of P(5) is -1.") is True
    assert q_neg1["check"]("-1") is True
    assert q_neg1["check"]("The value is 1.") is False

    # mbpp_04 / mbpp_18: concat_lists and find_index equivalents
    q_concat = by_id["mbpp_04"]
    assert q_concat["check"]("def concat_lists(l1, l2):\n    return [*l1, *l2]") is True
    q_find = by_id["mbpp_18"]
    assert q_find["check"]("def find_index(lst, x):\n    for i, v in enumerate(lst):\n        if v == x: return i\n    return -1") is True


def test_intrinsic_popperian_cycle_verifier_domains():
    """Tests IntrinsicPopperianCycleVerifier across code, math, conversational, and cycle energy."""
    verifier = IntrinsicPopperianCycleVerifier(d_model=64, tau_cycle=0.45)

    # 1. Code domain: valid vs stub vs wrong logic
    code_prompt = "Write a Python function `def string_xor(a: str, b: str) -> str:` that performs binary XOR."
    good_code = (
        "```python\n"
        "def string_xor(a: str, b: str) -> str:\n"
        "    return ''.join('1' if x != y else '0' for x, y in zip(a, b))\n"
        "```"
    )
    stub_code = "```python\ndef string_xor(a: str, b: str) -> str:\n    pass\n```"
    wrong_code = "```python\ndef string_xor(a: str, b: str) -> str:\n    return '000'\n```"

    sign_good, res_good = verifier.verify(code_prompt, good_code)
    assert sign_good == 1.0
    assert res_good["verified"] is True

    sign_stub, res_stub = verifier.verify(code_prompt, stub_code)
    assert sign_stub == -1.0
    assert res_stub["verified"] is False

    sign_wrong, res_wrong = verifier.verify(code_prompt, wrong_code)
    assert sign_wrong == -1.0
    assert res_wrong["verified"] is False

    # 2. Math domain: quadratic roots with negative numbers & polynomial evaluation
    quad_prompt = "Find the roots of x^2 + 8x + 7 = 0."
    sign_q_ok, res_quad_ok = verifier.verify(quad_prompt, "Factoring (x + 7)(x + 1) = 0 gives x = -7 and x = -1.")
    assert sign_q_ok == 1.0
    assert res_quad_ok["verified"] is True

    sign_q_bad, res_quad_bad = verifier.verify(quad_prompt, "The roots are x = 7 and x = 1.")
    assert sign_q_bad == -1.0
    assert res_quad_bad["verified"] is False
    assert res_quad_bad["correction_hint"] == "-7, -1"

    # 3. Conversational vs Binary classification
    conv_prompt = "Can you suggest 3 classic books on machine learning?"
    sign_c_bad, res_conv_bad = verifier.verify(conv_prompt, "Yes.")
    assert sign_c_bad == -1.0
    assert res_conv_bad["verified"] is False

    sign_c_ok, res_conv_ok = verifier.verify(
        conv_prompt,
        "1. Pattern Recognition and Machine Learning by Christopher Bishop\n"
        "2. The Elements of Statistical Learning by Hastie, Tibshirani, and Friedman\n"
        "3. Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.",
    )
    assert sign_c_ok == 1.0
    assert res_conv_ok["verified"] is True

    # 4. Latent Cycle-Consistency Reconstruction Energy
    torch.manual_seed(42)
    h_pre = torch.randn(1, 8, 64)
    h_post_good = h_pre * 1.02
    h_post_corrupt = -h_pre

    sign_cyc_ok, res_cycle_ok = verifier.verify(
        "Explain gravity briefly.",
        "Gravity is a fundamental interaction attracting masses.",
        h_query=h_pre,
        h_response=h_post_good,
    )
    assert sign_cyc_ok == 1.0
    assert res_cycle_ok["cycle_energy"] < 0.05
    assert res_cycle_ok["verified"] is True

    sign_cyc_bad, res_cycle_bad = verifier.verify(
        "Explain gravity briefly.",
        "Gravity is a fundamental interaction attracting masses.",
        h_query=h_pre,
        h_response=h_post_corrupt,
    )
    assert sign_cyc_bad == -1.0
    assert res_cycle_bad["cycle_energy"] > 0.45
    assert res_cycle_bad["verified"] is False


def test_synaptic_eligibility_trace_and_nullspace_zero_leakage():
    """
    Tests SynapticEligibilityTrace (and NullspaceMemoryEngine alias) with
    OrthogonalNullspaceProjector to verify:
    1. Zero interference on prior verified basis: max_basis_leakage < 1e-5
    2. Signed Hebbian (+1) consolidation vs Anti-Hebbian (-1) orthogonal deflation.
    """
    torch.manual_seed(7)
    d = 64
    trace_engine = SynapticEligibilityTrace(d_model=d, max_basis_slots=8, gamma=0.85)
    assert NullspaceMemoryEngine is SynapticEligibilityTrace

    # Step 1: Record and consolidate a verified trajectory (+1.0)
    h_pre1 = torch.randn(1, 6, d)
    h_post1 = torch.randn(1, 6, d)
    trace_engine.record_step(h_pre1, h_post1)
    delta_pos1, tel_pos1 = trace_engine.apply_signed_consolidation(dopamine_sign=1.0)
    assert tel_pos1["consolidation_mode"] == "hebbian_consolidation"
    assert tel_pos1["verified_basis_count"] == 1

    # Step 2: Consolidate a second trajectory and verify orthogonal projection against first basis
    h_pre2 = torch.randn(1, 6, d)
    h_post2 = torch.randn(1, 6, d)
    trace_engine.record_step(h_pre2, h_post2)
    delta_pos2, tel_pos2 = trace_engine.apply_signed_consolidation(dopamine_sign=1.0)
    assert tel_pos2["max_basis_leakage"] < 1e-5
    assert tel_pos2["is_strictly_orthogonal"] is True

    # Step 3: Apply negative Anti-Hebbian deflation (-1.0)
    h_falsified = torch.randn(1, 6, d)
    trace_engine.record_step(h_pre2, h_falsified)
    delta_neg, tel_neg = trace_engine.apply_signed_consolidation(dopamine_sign=-1.0)
    assert tel_neg["consolidation_mode"] == "anti_hebbian_deflation"
    assert tel_neg["falsified_count"] == 1

    # Verify deflection removes component along falsified direction
    h_deflected, defl_mag = trace_engine.deflect_from_falsified(h_falsified)
    assert defl_mag > 0.0
    assert not torch.allclose(h_deflected, h_falsified)


def test_open_freeze_liquid_plasticity_and_banach_signed_update():
    """Tests signed dopamine adaptation in OpenFreezeLiquidPlasticity & EndogenousBanachPlasticUnit."""
    torch.manual_seed(11)
    d = 64
    liquid = OpenFreezeLiquidPlasticity(d_model=d)
    banach = EndogenousBanachPlasticUnit(d_model=d, gamma_contraction=0.85)

    h = torch.randn(1, 4, d)
    spikes = torch.ones(1, 4, 1)
    h_out, l_telem = liquid(h, spike=spikes)
    assert h_out.shape == (1, 4, d)
    assert l_telem["fluid_viscosity_mu"] > 0.0

    w_before = liquid.plastic_v.weight.data.clone()
    drift_pos = liquid.adapt_online_step(h, spike=spikes, lr=0.05, dopamine_sign=1.0)
    assert drift_pos > 0.0
    w_after_pos = liquid.plastic_v.weight.data.clone()
    assert not torch.allclose(w_before, w_after_pos)

    drift_neg = liquid.adapt_online_step(h, spike=spikes, lr=0.05, dopamine_sign=-1.0)
    assert drift_neg > 0.0

    # Banach plastic unit signed closed-loop dopamine update
    h_pre = torch.randn(1, 4, d)
    h_post = torch.randn(1, 4, d)
    b_norm_pos = banach.apply_closed_loop_dopamine(h_pre, h_post, dopamine_sign=1.0)
    assert b_norm_pos > 0.0
    b_norm_neg = banach.apply_closed_loop_dopamine(h_pre, h_post, dopamine_sign=-1.0)
    assert b_norm_neg > 0.0
    assert banach.get_stats()["plasticity_step_count"] == 2


def test_shadow_kv_cache_multi_token_slicing_fix():
    """
    Tests that NeuromorphicSpikingBrain.forward_layer7_transition handles multi-token
    chunks (seq_len > 1) when _rolling_k_cache is populated without shape mismatch RuntimeError.
    """
    torch.manual_seed(19)
    d = 64
    brain = NeuromorphicSpikingBrain(d_model=d, spike_v_th=0.05)
    brain.reset_brain_state()

    # Step 1: Prefill with 10 tokens (populates _rolling_k_cache / _rolling_v_cache)
    h_prefill = torch.randn(1, 10, d) * 2.0
    u_drive1 = torch.ones(1, 10, 1) * 0.9
    out1, telem1 = brain.forward_layer7_transition(h_prefill, u_drive=u_drive1, num_deliberation_steps=1)
    assert out1.shape == (1, 10, d)

    # Step 2: Inject a 6-token code stub during decoding (seq_len=6 > 1 while _rolling_k_cache exists)
    h_stub = torch.randn(1, 6, d) * 2.0
    u_drive2 = torch.ones(1, 6, 1) * 0.9
    out2, telem2 = brain.forward_layer7_transition(h_stub, u_drive=u_drive2, num_deliberation_steps=1)
    assert out2.shape == (1, 6, d)
    assert torch.isfinite(out2).all()


def test_tokenless_engine_conversational_and_closed_loop_deflation():
    """
    Tests that TokenlessLatentThoughtEngine:
    1. Does not misclassify 'Can you suggest...' or 'Can you give...' as binary Yes/No.
    2. Suppresses <think> token when ALU result is known.
    3. Applies closed-loop anti-Hebbian penalty to falsified tokens and boosts symbolic correction hints.
    """
    engine = TokenlessLatentThoughtEngine(d_model=64)

    class DummyTokenizer:
        def __init__(self):
            self.vocab = {
                "<think>": 248068,
                "</think>": 248069,
                "Yes": 20,
                "No": 21,
                "7": 30,
                "-7": 31,
                "-": 32,
            }
            self.inv = {v: k for k, v in self.vocab.items()}

        def encode(self, text, add_special_tokens=False):
            if text in self.vocab:
                return [self.vocab[text]]
            if text.strip() in self.vocab:
                return [self.vocab[text.strip()]]
            return [99]

        def decode(self, ids):
            return "".join(self.inv.get(i, "?") for i in ids)

    tok = DummyTokenizer()
    logits = torch.zeros(1, 250000)
    logits[0, 248068] = 15.0  # <think> initially highest

    # Conversational 'Can you suggest...' should be passthrough, NOT yes/no
    clamped_conv = engine.clamp_first_token_logits(
        logits.clone(),
        tok,
        prompt_text="Can you suggest 3 books on physics?",
    )
    assert clamped_conv[0, 20].item() < 5.0

    # Closed-loop falsification penalty + correction hint
    clamped_retry = engine.clamp_first_token_logits(
        logits.clone(),
        tok,
        prompt_text="Find the roots of x^2 + 8x + 7 = 0.",
        telemetry={
            "falsified_token_ids": [30],  # Penalize wrong answer '7'
            "correction_hint": "-7",      # Boost '-7'
        },
    )
    assert clamped_retry[0, 30].item() <= -20.0
    assert clamped_retry[0, 31].item() > 15.0
    assert clamped_retry[0, 248068].item() < -50.0  # <think> suppressed


# ==============================================================================
# PILLAR 2: Endogenous Custom Sound Synthesis (`freq suara` + Self-Listening)
# ==============================================================================


def test_endogenous_acoustic_vocal_organ_synthesis_and_self_listening(tmp_path: Path):
    """
    Tests EndogenousAcousticVocalOrgan:
    1. Fundamental pitch f0 in [85, 340] Hz
    2. Formants F1..F4 in [250, 4200] Hz
    3. Valid 16-bit PCM .wav export
    4. Self-Listening Auditory Re-Entry Loop increases semantic-acoustic coherence online
    """
    torch.manual_seed(42)
    d = 128
    organ = EndogenousAcousticVocalOrgan(d_model=d, sample_rate=16000, n_harmonics=24)
    h_latent = torch.randn(1, 12, d)

    waveform, telem = organ.synthesize_waveform(
        hidden_states=h_latent,
        text="Hello X-Star endogenous voice",
        firing_rate=0.45,
        membrane_v=0.40,
        duration_sec=0.5,
        adapt_online=True,
    )
    wav_file = tmp_path / "test_freq_suara.wav"
    organ.save_wav(waveform, str(wav_file))

    # Check f0 in [85, 340] Hz
    assert 85.0 <= telem["f0_mean_hz"] <= 340.0
    assert 85.0 <= telem["f0_min_hz"] <= 340.0
    assert 85.0 <= telem["f0_max_hz"] <= 340.0

    # Check 4 formants in [250, 4200] Hz and ordered F1 < F2 < F3 < F4
    formants = telem["formants_hz"]
    assert len(formants) == 4
    for f_hz in formants:
        assert 250.0 <= f_hz <= 4200.0
    assert formants[0] < formants[1] < formants[2] < formants[3]

    # Check waveform energy and shape
    assert waveform.shape == (8000,)
    assert telem["waveform_rms"] > 0.01

    # Check exported .wav file validity
    assert wav_file.exists()
    with wave.open(str(wav_file), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 8000

    # Check Self-Listening Auditory Re-Entry Loop improved semantic-acoustic coherence
    assert telem["voice_adaptation_steps"] >= 1
    assert telem["semantic_acoustic_coherence_after"] > telem["semantic_acoustic_coherence_before"]


# ==============================================================================
# PILLAR 3: Progressive 2D-to-3D Human-Like Visual Imagination
# ==============================================================================


def test_progressive_2d_to_3d_imagination_canvas():
    """
    Tests Progressive2Dto3DImaginationCanvas:
    1. 1D -> 2D Retinotopic Sketchpad projection
    2. 2D -> 3D Gaussian/SDF Volumetric Lifting along e_3
    3. Exact SO(3) Mental Rotation (||R^T R - I||_F < 1e-5, det(R) = +1)
    4. Differentiable 3D-to-2D Splatting Verification & gradient flow
    """
    torch.manual_seed(99)
    d = 128
    canvas = Progressive2Dto3DImaginationCanvas(
        d_model=d,
        grid_size=16,
        n_primitives=8,
    )

    h_seq = torch.randn(2, 10, d, requires_grad=True)
    q, _ = torch.linalg.qr(torch.randn(d, 3))
    basis = [q[:, 0], q[:, 1], q[:, 2]]

    h_out, telem = canvas(h_seq, multivector_basis=basis)

    assert h_out.shape == (2, 10, d)
    assert telem["grid_size_2d"] == 16
    assert telem["n_3d_primitives"] == 8
    assert telem["so3_orthogonality_error"] < 1e-5
    assert abs(telem["so3_determinant"] - 1.0) < 1e-4
    assert 0.0 <= telem["consistency_2d_to_3d"] <= 1.0
    assert telem["basis_orthogonality_error"] < 1e-5

    # Verify differentiability through 1D -> 2D -> 3D -> SO(3) -> 2D splatting -> 1D
    loss = h_out.pow(2).mean()
    loss.backward()
    assert h_seq.grad is not None
    assert torch.norm(h_seq.grad).item() > 0.0


# ==============================================================================
# PILLAR 4: Full Custom Inference Framework & Multi-Pass Self-Evolving Loop
# ==============================================================================


class DummySelfEvolvingCausalLM(nn.Module):
    """
    Lightweight causal LM for testing HADLv5SpikingBrainModelWrapper closed-loop
    multi-pass self-evolving generation, voice synthesis, and 3D imagination.
    """

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_dim, "vocab_size": vocab_size})()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(10)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = h + 0.1 * torch.tanh(layer(h))
        h = self.norm(h)
        logits = self.lm_head(h) * 0.0
        logits[..., 30] = 10.0  # Initially prefer wrong answer token 30 ('7')
        return type("Out", (), {"logits": logits})()

    def generate(self, input_ids, logits_processor=None, max_new_tokens=5, **kwargs):
        out = self.forward(input_ids)
        next_token_logits = out.logits[:, -1, :].clone()
        if logits_processor is not None:
            for proc in logits_processor:
                next_token_logits = proc(input_ids, next_token_logits)
        next_tok = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        return torch.cat([input_ids, next_tok], dim=1)


def test_hadl_wrapper_multi_pass_self_evolving_correction(tmp_path: Path):
    """
    Tests that HADLv5SpikingBrainModelWrapper.generate_self_evolving:
    1. Detects a falsified Pass 1 answer (wrong math root '7' instead of '-7, -1')
    2. Applies Anti-Hebbian deflation (dopamine_sign = -1.0) and stores closed_loop_state
    3. Self-corrects on Pass 2 to emit the verified answer '-7, -1'!
    """
    torch.manual_seed(123)
    base_model = DummySelfEvolvingCausalLM(vocab_size=64, hidden_dim=64)
    # Force base model to initially prefer token 30 ("7") over token 31 ("-7, -1")
    with torch.no_grad():
        base_model.lm_head.weight.zero_()
        base_model.lm_head.weight[30] = 5.0

    wrapper = attach_hadl_v5_spiking_brain(
        base_model=base_model,
        target_layer_idx=7,
        pacemaker_layer_idx=2,
        reentrant_layers=[4, 5, 7],
        auto_load_checkpoints=False,
    )

    class SimpleMathTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __init__(self):
            self.map = {
                "<think>": 10,
                "</think>": 11,
                "7": 30,
                "-7": 31,
                "-7, -1": 31,
            }

        def encode(self, text, add_special_tokens=False):
            t = text.strip()
            if t in self.map:
                return [self.map[t]]
            return [40]

        def decode(self, ids, skip_special_tokens=True):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            out = []
            for i in ids:
                if i == 30:
                    out.append("7")
                elif i == 31:
                    out.append("-7, -1")
            return "".join(out)

    tok = SimpleMathTokenizer()
    prompt = "Find the roots of x^2 + 8x + 7 = 0."
    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)

    # Temporarily patch evaluate_cognitive_alu and evaluate_quadratic_roots on ipv6_register so Pass 1 relies on model weights (which emit wrong '7'),
    # proving that IntrinsicPopperianCycleVerifier catches the error and Pass 2 fixes it via closed_loop_state!
    orig_extract = wrapper.tokenless_engine.ipv6_register.evaluate_cognitive_alu
    orig_quad = wrapper.tokenless_engine.ipv6_register.evaluate_quadratic_roots
    wrapper.tokenless_engine.ipv6_register.evaluate_cognitive_alu = lambda p: None
    wrapper.tokenless_engine.ipv6_register.evaluate_quadratic_roots = lambda p: None
    try:
        out_ids, evo_summary = wrapper.generate_self_evolving(
            input_ids=input_ids,
            prompt_text=prompt,
            tokenizer=tok,
            max_passes=2,
            synthesize_voice=True,
            voice_output_path=str(tmp_path / "evo_voice.wav"),
            max_new_tokens=1,
        )
    finally:
        wrapper.tokenless_engine.ipv6_register.evaluate_cognitive_alu = orig_extract
        wrapper.tokenless_engine.ipv6_register.evaluate_quadratic_roots = orig_quad

    assert evo_summary["total_passes"] == 2
    assert evo_summary["passes"][0]["response"] == "7"
    assert evo_summary["passes"][0]["verified"] is False
    assert evo_summary["passes"][0]["dopamine_sign"] == -1.0

    # Pass 2 self-corrected via closed-loop anti-Hebbian penalty + Popperian correction hint!
    assert evo_summary["passes"][1]["response"] == "-7, -1"
    assert evo_summary["passes"][1]["verified"] is True
    assert evo_summary["passes"][1]["dopamine_sign"] == 1.0
    assert evo_summary["final_verified"] is True
    assert (tmp_path / "evo_voice.wav").exists()
    assert "voice_telemetry" in evo_summary


def test_custom_inference_engine_and_cognitive_triad_integration(tmp_path: Path):
    """Tests CustomInferenceEngine verify_and_adapt, synthesize_voice, and get_3d_imagination_telemetry."""
    cfg = EngineConfig(enable_online_learning=True, enable_voice=True)
    engine = CustomInferenceEngine(cfg)

    # Test standalone voice synthesis & 3D imagination telemetry before full LLM load
    wav_path = tmp_path / "engine_voice.wav"
    v_res = engine.synthesize_voice(text="Testing X-Star endogenous voice.", save_path=str(wav_path), duration_sec=0.4)
    assert wav_path.exists()
    assert 85.0 <= v_res["synthesis_telemetry"]["f0_mean_hz"] <= 340.0

    img_telem = engine.get_3d_imagination_telemetry(text="Rotate a 3D tetrahedron in latent space.")
    assert img_telem["num_3d_primitives"] == 8
    assert img_telem["so3_orthogonality_error"] < 1e-5

    # Test verify_and_adapt
    ver_bad = engine.verify_and_adapt("Find the roots of x^2 + 8x + 7 = 0.", "The roots are 7 and 1.")
    assert ver_bad["verified"] is False
    assert ver_bad["dopamine_sign"] == -1.0

    ver_ok = engine.verify_and_adapt("Find the roots of x^2 + 8x + 7 = 0.", "The roots are -7 and -1.")
    assert ver_ok["verified"] is True
    assert ver_ok["dopamine_sign"] == 1.0
