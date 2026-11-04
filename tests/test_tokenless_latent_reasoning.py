"""
Unit Tests for Tokenless Latent Thought Engine & IPv6 Numeric Register
======================================================================
Tests:
1. IPv6NumericRegister: deterministic encoding, discrete ALU execution, and rigid invariant projection.
2. ThoughtPurifiedKVCache: latent trajectory condensation and compact KV-anchor generation.
3. EndogenousMetaRLPolicy: online reward calculation and autonomous parameter adaptation.
4. TokenlessLatentThoughtEngine: silent continuous latent reflection and Answer-First logits clamping.
"""

import pytest
import torch
import torch.nn as nn

from dual_loop.neuromorphic import (
    IPv6NumericRegister,
    ThoughtPurifiedKVCache,
    EndogenousMetaRLPolicy,
    TokenlessLatentThoughtEngine,
)


class DummyTokenizer:
    """Mock tokenizer for testing logits clamping."""
    def __init__(self):
        self.vocab = {
            "<think>": 1001,
            "</think>": 1002,
            "Yes": 2001,
            "No": 2002,
            "31": 3031,
            "17": 3017,
        }

    def encode(self, text, add_special_tokens=False):
        t = text.strip()
        if t in self.vocab:
            return [self.vocab[t]]
        return [9999]


def test_ipv6_numeric_register_extraction_and_alu():
    reg = IPv6NumericRegister(d_model=2048, reg_dim=64)
    prompt = "Passage: In the first half, the Eagles scored 17 points while Cowboys scored 10 points. In the second half, the Eagles scored 14 points.\nQuestion: How many total points did the Eagles score? Answer with only the number.\nAnswer:"
    
    nums = reg.extract_numbers_from_text(prompt)
    assert 17.0 in nums
    assert 10.0 in nums
    assert 14.0 in nums
    
    # Check ALU execution
    total = reg.execute_alu("sum", [17.0, 14.0])
    assert total == 31.0


def test_ipv6_numeric_register_rigid_projection():
    d_model = 2048
    reg = IPv6NumericRegister(d_model=d_model, reg_dim=64)
    h = torch.randn(2, 8, d_model)
    
    prompt = "The team scored 17 points and then 14 points. Total points?"
    h_out, telem = reg(h, prompt_text=prompt)
    
    assert telem["ipv6_active"] is True
    assert telem["alu_result"] == 31.0
    assert h_out.shape == h.shape
    
    # Check that the numeric subspace (last 64 dims) contains non-zero rigid encoding
    subspace = h_out[..., reg.semantic_dim:]
    assert subspace.shape[-1] == reg.reg_dim
    assert float(torch.norm(subspace).item()) > 0.1


def test_thought_purified_kv_cache():
    d_model = 2048
    purifier = ThoughtPurifiedKVCache(d_model=d_model, num_heads=16)
    
    # Simulated 3-step continuous latent trajectory
    h_0 = torch.randn(2, 4, d_model)
    h_1 = h_0 + torch.randn(2, 4, d_model) * 0.05
    h_2 = h_1 + torch.randn(2, 4, d_model) * 0.01
    
    h_star = purifier.condense_trajectory([h_0, h_1, h_2])
    assert h_star.shape == (2, 4, d_model)
    
    # Create anchor KV pair
    k_anchor, v_anchor = purifier.create_anchor_kv(h_star)
    # Target shape: [B, num_heads, 1, head_dim] where head_dim = 2048 // 16 = 128
    assert k_anchor.shape == (2, 16, 1, 128)
    assert v_anchor.shape == (2, 16, 1, 128)


def test_endogenous_meta_rl_policy_adaptation():
    policy = EndogenousMetaRLPolicy(d_model=2048, lr=0.05)
    params_initial = policy.get_effective_parameters()
    
    # Case 1: Excellent concise answer at token 1
    r_pos = policy.compute_online_reward(
        token_count=4,
        max_tokens=96,
        has_natural_eos=True,
        answer_at_token_1=True,
        is_correct=True
    )
    assert r_pos > 1.5
    
    # Update policy
    policy.update_policy_online(r_pos)
    params_after_pos = policy.get_effective_parameters()
    assert params_after_pos["ans1_bias"] >= params_initial["ans1_bias"]
    
    # Case 2: Unclosed thought hitting ceiling
    r_neg = policy.compute_online_reward(
        token_count=96,
        max_tokens=96,
        has_natural_eos=False,
        answer_at_token_1=False,
        is_correct=False
    )
    assert r_neg < 0.0
    
    # Update policy again
    policy.update_policy_online(r_neg)
    assert policy.step_count.item() == 2


def test_tokenless_latent_thought_engine_clamping():
    d_model = 2048
    engine = TokenlessLatentThoughtEngine(d_model=d_model, deliberation_steps_k=2)
    tok = DummyTokenizer()
    
    # Mock logits for vocabulary size 10000
    logits = torch.zeros(1, 1, 10000)
    logits[..., 1001] = 10.0  # <think> token has high logit
    logits[..., 2001] = 5.0   # Yes token
    logits[..., 3031] = 4.0   # 31 token
    
    prompt = "Logical deduction: Is Earth a planet? Answer with Yes or No.\nAnswer:"
    clamped = engine.clamp_first_token_logits(logits, tok, prompt, alu_result=31.0)
    
    # <think> token must be severely penalized
    assert float(clamped[..., 1001].item()) < -100.0
    # 31 token must be boosted by ALU register
    assert float(clamped[..., 3031].item()) > 15.0


def test_tokenless_latent_deliberation_execution():
    d_model = 2048
    engine = TokenlessLatentThoughtEngine(d_model=d_model, deliberation_steps_k=2)
    h_in = torch.randn(1, 6, d_model)
    
    h_star, trajectory, telem = engine.execute_silent_latent_deliberation(h_in, num_steps=2)
    
    assert len(trajectory) == 3
    assert h_star.shape == h_in.shape
    assert telem["latent_deliberation_steps"] == 2
    
    # Test discovery revelation formatting
    brief = engine.generate_discovery_revelation("31", {"alu_result": 31.0, "latent_deliberation_steps": 2, "tour_restoring_force": 0.02})
    assert "31" in brief
    assert "Intisari Pemikiran & Temuan" in brief
    assert "IPv6 ALU Register" in brief


def test_brain_emergent_latent_canvas_integration():
    from dual_loop.neuromorphic.spiking_gate import NeuromorphicSpikingBrain
    d_model = 2048
    brain = NeuromorphicSpikingBrain(d_model=d_model)
    assert hasattr(brain, "imagination_canvas")
    assert brain.imagination_canvas.d_model == d_model

    h = torch.randn(1, 4, d_model)
    h_out, telem = brain.forward_layer7_transition(h)

    assert h_out.shape == h.shape
    assert "dual_stream_telemetry" in telem
    dual_telem = telem["dual_stream_telemetry"]
    assert "dream_energy" in dual_telem
    assert "confusion_mean" in dual_telem
    assert "dream_active" in dual_telem


def test_cognitive_alu_intent_discrimination():
    reg = IPv6NumericRegister(d_model=2048)

    # 1. Code generation request: Must NEVER trigger arithmetic ALU
    code_prompt = "def calculate_score(scores): total = sum(scores); return total"
    assert reg.evaluate_cognitive_alu(code_prompt) is None

    # 2. Conversational / Advice request: Must NEVER trigger arithmetic ALU
    chat_prompt = "Yesterday my team scored 10 points and the opponent scored 15. How can we improve our game tomorrow?"
    assert reg.evaluate_cognitive_alu(chat_prompt) is None

    # 3. Metaphorical question: Must NEVER trigger arithmetic ALU
    metaphor_prompt = "My career turned 180 degrees after 10 years in tech. What should I do next?"
    assert reg.evaluate_cognitive_alu(metaphor_prompt) is None

    # 4. Strict DROP Math word problem: Must evaluate exact summation (17 + 14 = 31)
    drop_prompt = "Passage: In the first half, the Eagles scored 17 points while Cowboys scored 10 points. In the second half, the Eagles scored 14 points.\nQuestion: How many total points did the Eagles score? Answer with only the number.\nAnswer:"
    assert reg.evaluate_cognitive_alu(drop_prompt) == 31.0

    # 5. Strict GSM8K Math word problem: Must evaluate exact multiplication (14 * 2 = 28)
    gsm_prompt = "Janet buys 14 apples at $2 each. How much money did Janet spend in total?\nAnswer:"
    assert reg.evaluate_cognitive_alu(gsm_prompt) == 28.0


def test_answer_first_logits_processor_transparency_on_coding_and_chat():
    from dual_loop.neuromorphic.spiking_gate import AnswerFirstLogitsProcessor
    engine = TokenlessLatentThoughtEngine(d_model=2048)
    tok = DummyTokenizer()

    # Case A: Code generation prompt
    code_prompt = "Tuliskan fungsi Python def fibonacci(n) dan jelaskan cara kerjanya."
    proc_code = AnswerFirstLogitsProcessor(engine, tok, prompt_text=code_prompt)
    assert proc_code.passthrough is True
    assert proc_code.is_strict_closed is False
    assert proc_code.alu_target_tokens == []

    # Case B: Natural chat prompt
    chat_prompt = "Halo! Bagaimana kabarmu hari ini? Ceritakan sedikit tentang bagaimana caramu berpikir."
    proc_chat = AnswerFirstLogitsProcessor(engine, tok, prompt_text=chat_prompt)
    assert proc_chat.passthrough is True
    assert proc_chat.is_strict_closed is False

    # Case C: Strict Closed Exam prompt
    closed_prompt = "Logical deduction: Is Earth a planet? Answer with Yes or No.\nAnswer:"
    proc_closed = AnswerFirstLogitsProcessor(engine, tok, prompt_text=closed_prompt)
    assert proc_closed.passthrough is False
    assert proc_closed.is_strict_closed is True

