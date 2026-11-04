"""Unit and edge-case tests for LatentThoughtMatrixEngine and CentralCognitiveRouter."""

from __future__ import annotations

from pathlib import Path
import torch

from dual_loop.neuromorphic.pacemaker_router import (
    LatentThoughtMatrixEngine,
    CentralCognitiveRouter,
)
from dual_loop.neuromorphic.tokenless_latent_engine import (
    TokenlessLatentThoughtEngine,
)
from dual_loop.neuromorphic.spiking_gate import (
    NeuromorphicSpikingBrain,
    AnswerFirstLogitsProcessor,
)


class _DummyTokenizer:
    """Minimal deterministic tokenizer for logit clamping and processor tests."""

    def __init__(self) -> None:
        self.vocab = {
            "<think>": 248068,
            "</think>": 248069,
            "Yes": 10,
            " yes": 11,
            "True": 12,
            "No": 20,
            " no": 21,
            "False": 22,
            "Myth": 23,
            "def": 30,
            " def": 31,
            "    ": 32,
            "return": 33,
            "Everest": 40,
            "Nile": 41,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.vocab:
            return [self.vocab[text]]
        return [100 + (sum(ord(c) for c in text) % 50)]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self.inv_vocab.get(i, "") for i in ids)


def test_latent_thought_matrix_4_subparts_and_orthogonality():
    """Verify M in R^{B x S x 4 x D} and strict orthogonality of M_1, M_2, M_3 to M_0."""
    torch.manual_seed(42)
    d_model = 128
    engine = LatentThoughtMatrixEngine(d_model=d_model, bottleneck_dim=32, veracity_dim=32)
    h = torch.randn(2, 9, d_model) * 3.5

    M, telemetry = engine(h)

    assert M.shape == (2, 9, 4, d_model)
    assert "veracity_conflict_score" in telemetry
    assert "subpart_norms" in telemetry
    assert -1.0 <= telemetry["veracity_conflict_score"] <= 1.0

    m0 = M[:, :, 0, :]
    m1 = M[:, :, 1, :]
    m2 = M[:, :, 2, :]
    m3 = M[:, :, 3, :]

    # M_0 must be the exact invariant factual/syntactic anchor h
    assert torch.allclose(m0, h, atol=1e-6)

    # M_1, M_2, M_3 must all lie in the orthogonal complement of M_0 at every token
    for idx, mi in enumerate([m1, m2, m3], start=1):
        inner = (mi * m0).sum(dim=-1) / (m0.norm(dim=-1) * mi.norm(dim=-1) + 1e-6)
        max_cos = inner.abs().max().item()
        assert max_cos < 1e-4, f"Subpart M_{idx} is not orthogonal to M_0 (max |cos|={max_cos})"


def test_central_cognitive_router_toroidal_circulation_and_invariance():
    """Verify CentralCognitiveRouter preserves M_0 and bounds orthogonal delta norm."""
    torch.manual_seed(7)
    d_model = 128
    matrix_engine = LatentThoughtMatrixEngine(d_model=d_model, bottleneck_dim=32, veracity_dim=32)
    router = CentralCognitiveRouter(
        d_model=d_model,
        num_subparts=4,
        num_cycles=2,
        max_ortho_ratio=0.015,
    )

    h = torch.randn(3, 6, d_model) * 2.0
    M, matrix_meta = matrix_engine(h)
    h_routed, router_meta = router(
        M,
        veracity_conflict_score=matrix_meta["veracity_conflict_score"],
    )

    assert h_routed.shape == h.shape
    assert router_meta["central_router_cycles"] == 2
    assert len(router_meta["central_router_weights"]) == 4

    # The net update delta = h_routed - h must be strictly orthogonal to h
    delta = h_routed - h
    cos_sim = (delta * h).sum(dim=-1) / (delta.norm(dim=-1) * h.norm(dim=-1) + 1e-6)
    assert cos_sim.abs().max().item() < 1e-4

    # And relative norm must never exceed max_ortho_ratio (0.015) + numerical epsilon
    rel_norm = (delta.norm(dim=-1) / (h.norm(dim=-1) + 1e-6)).max().item()
    assert rel_norm <= 0.015 + 1e-4


def test_single_token_and_zero_vector_edge_cases():
    """Test boundary shapes (S=1) and zero hidden state vectors without NaN/Inf."""
    d_model = 64
    matrix_engine = LatentThoughtMatrixEngine(d_model=d_model, bottleneck_dim=16, veracity_dim=16)
    router = CentralCognitiveRouter(d_model=d_model, num_cycles=2)

    # S=1 single-token input
    h_single = torch.randn(1, 1, d_model)
    M_single, meta_single = matrix_engine(h_single)
    h_out_single, _ = router(M_single, veracity_conflict_score=meta_single["veracity_conflict_score"])
    assert not torch.isnan(h_out_single).any()
    assert not torch.isinf(h_out_single).any()

    # All-zero input vector
    h_zero = torch.zeros(1, 4, d_model)
    M_zero, meta_zero = matrix_engine(h_zero)
    h_out_zero, _ = router(M_zero, veracity_conflict_score=meta_zero["veracity_conflict_score"])
    assert not torch.isnan(h_out_zero).any()
    assert not torch.isinf(h_out_zero).any()
    assert torch.allclose(h_out_zero, h_zero, atol=1e-6)


def test_disjunctive_vs_polar_veracity_clamping():
    """Verify disjunctive 'X or Y?' questions are not misclassified as polar Yes/No myths."""
    tok = _DummyTokenizer()
    engine = TokenlessLatentThoughtEngine(d_model=64)

    base_logits = torch.zeros(1, 300)
    high_veracity_telemetry = {"veracity_conflict_score": 0.92}
    low_veracity_telemetry = {"veracity_conflict_score": -0.85}

    # 1. Polar misconception with high learned veracity_conflict_score -> refutes ("No" > "Yes")
    polar_prompt = "Question: Can goldfish remember things for more than three seconds?\nAnswer:"
    clamped_polar = engine.clamp_first_token_logits(
        base_logits.clone(),
        tok,
        polar_prompt,
        telemetry=high_veracity_telemetry,
    )
    assert clamped_polar[0, tok.vocab["No"]] > clamped_polar[0, tok.vocab["Yes"]]

    # 2. Polar factual question with negative learned veracity_conflict_score -> does NOT refute
    clamped_verified = engine.clamp_first_token_logits(
        base_logits.clone(),
        tok,
        polar_prompt,
        telemetry=low_veracity_telemetry,
    )
    assert clamped_verified[0, tok.vocab["No"]] <= clamped_verified[0, tok.vocab["Yes"]]

    # 3. Disjunctive choice question ("Is tomato a botanical vegetable or a fruit?") ->
    # MUST NOT apply polar Yes/No refutation even if veracity_conflict_score is high!
    disjunctive_prompt = "Question: Is tomato a botanical vegetable or a fruit?\nAnswer:"
    clamped_disjunctive = engine.clamp_first_token_logits(
        base_logits.clone(),
        tok,
        disjunctive_prompt,
        telemetry=high_veracity_telemetry,
    )
    assert clamped_disjunctive[0, tok.vocab["No"]] == 0.0
    assert clamped_disjunctive[0, tok.vocab["Yes"]] == 0.0


def test_code_starters_when_prompt_closes_code_fence():
    """Verify AnswerFirstLogitsProcessor boosts 'def' when prompt closes ``` fence."""
    tok = _DummyTokenizer()
    engine = TokenlessLatentThoughtEngine(d_model=64)
    # HumanEval style prompt where ``` is closed before "Python code:"
    humaneval_prompt = (
        "Complete the following Python function:\n```python\n"
        "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
        "    \"\"\"Check if any two numbers are closer than threshold.\"\"\"\n"
        "```\nPython code:"
    )
    proc = AnswerFirstLogitsProcessor(
        tokenless_engine=engine,
        tokenizer=tok,
        prompt_text=humaneval_prompt,
    )
    scores = torch.zeros(1, 300)
    # Simulate step 0 (semantic inception on code completion)
    input_ids = torch.tensor([[101, 102, 103, 104, 105]])
    out_scores = proc(input_ids, scores)

    # 'def' (30) should be boosted, not just indentation '    ' (32)
    assert out_scores[0, tok.vocab["def"]] > out_scores[0, tok.vocab["    "]]


def test_neuromorphic_spiking_brain_forward_layer7_matrix_integration():
    """Verify NeuromorphicSpikingBrain integrates LatentThoughtMatrixEngine & CentralCognitiveRouter."""
    brain = NeuromorphicSpikingBrain(d_model=64)
    h = torch.randn(1, 5, 64)
    h_out, info = brain.forward_layer7_transition(h)

    assert h_out.shape == h.shape
    assert "veracity_conflict_score" in info
    assert "central_router_telemetry" in info
    assert info["central_router_telemetry"]["central_router_cycles"] == 2

    # Perturbation must be strictly orthogonal to h and bounded
    delta = h_out - h
    cos = (delta * h).sum(dim=-1) / (delta.norm(dim=-1) * h.norm(dim=-1) + 1e-6)
    assert cos.abs().max().item() < 1e-3
    rel_norm = (delta.norm(dim=-1) / (h.norm(dim=-1) + 1e-6)).max().item()
    assert rel_norm <= 0.02


def test_trained_checkpoint_loads_into_spiking_brain():
    """Verify checkpoints/hadl_v58_latent_matrix_central_router.pt exists and loads cleanly."""
    ckpt_path = Path("checkpoints/hadl_v58_latent_matrix_central_router.pt")
    assert ckpt_path.exists(), "Trained HADL v5.8 Latent Matrix checkpoint missing!"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    assert ckpt["metrics"]["accuracy_pct"] >= 99.0
    assert ckpt["metrics"]["mean_myth_veracity_score"] > 0.85
    assert ckpt["metrics"]["mean_grounded_veracity_score"] < -0.85

    brain = NeuromorphicSpikingBrain(d_model=2048)
    load_status = brain.load_trained_checkpoints(checkpoints_dir="checkpoints")
    assert load_status.get("latent_matrix_central_router", False) is True


def test_ipv6_register_static_ast_and_physical_coreference():
    """Verify IPv6NumericRegister static AST exception evaluator and physical/calendar registers."""
    engine = TokenlessLatentThoughtEngine(d_model=64)
    reg = engine.ipv6_register

    assert reg.evaluate_python_exception("What exception is raised by `1 / 0` in Python?") == "ZeroDivisionError"
    assert reg.evaluate_python_exception("What exception is raised by `[1, 2, 3][5]` in Python?") == "IndexError"
    assert reg.evaluate_python_exception("What exception is raised by `{'a': 1}['b']` in Python?") == "KeyError"
    assert reg.evaluate_python_exception("What exception is raised by `int('abc')` in Python?") == "ValueError"
    assert reg.evaluate_python_exception("What exception is raised by `'a' + 5` in Python?") == "TypeError"
    assert reg.evaluate_python_exception("What exception is raised by `(1, 2)[0] = 3` in Python?") == "TypeError"
    assert reg.evaluate_python_exception("What exception is raised by `iter(42)` in Python?") == "TypeError"
    assert reg.evaluate_python_exception("What exception is raised by `open('nonexistent_xyz.txt')` in Python?") == "FileNotFoundError"

    # Discrete kinematics & boolean logic
    assert reg.evaluate_die_face("A standard die has 3 on the top face. What number is on the opposite face?") == "4"
    assert reg.evaluate_boolean_expression("Boolean expression: True and (False or True). Is the result True or False?") == "True"

    # Multi-day calendar arithmetic and verbal ALU
    assert reg.evaluate_calendar_direction("If two days ago was Friday, what day is tomorrow?") == "Monday"
    assert reg.evaluate_cognitive_alu("Compute 14 times 6 and respond with only digits.") == 84.0

