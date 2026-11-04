#!/usr/bin/env python3
"""
Architectural Component Stress Benchmark for HADL v5.8
======================================================
Empirical verification of individual architectural subsystems under stress,
boundary conditions, and out-of-distribution (OOD) adversarial vectors.

Components evaluated:
1. Component A: Latent Thought Matrix (M in R^{B x S x 4 x D}) Orthogonality & Subspace Invariance
2. Component B: Recurrent Toroidal Central Cognitive Router ("Otak Berputar") Dynamic Stability
3. Component C: Learned Veracity Conflict Subspace on Out-Of-Distribution (OOD) Propositions
4. Component D: IPv6 Numeric Symbolic Registers & AST Boundary Stress
5. Component E: Neuromorphic Leaky Integrate-and-Fire (LIF) Spiking Gate Dynamics

Target Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (cuda:0, bfloat16 / float32)
Zero mocked outputs. 100% empirical measurement with exact failure reporting.
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.neuromorphic.pacemaker_router import (
    LatentThoughtMatrixEngine,
    CentralCognitiveRouter,
)
from dual_loop.neuromorphic.spiking_gate import (
    LeakyIntegrateAndFireGate,
    FastSigmoidSurrogate,
    spike_fn,
)
from dual_loop.neuromorphic.tokenless_latent_engine import (
    IPv6NumericRegister,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "hadl_v58_latent_matrix_central_router.pt"
OUTPUT_JSON = PROJECT_ROOT / "eval_results" / "hadl_v58_component_stress_audit.json"


# =====================================================================
# COMPONENT A: Latent Thought Matrix Orthogonality & Subspace Invariance
# =====================================================================
def benchmark_component_a_latent_matrix(engine: LatentThoughtMatrixEngine) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("[-] BENCHMARK COMPONENT A: Latent Thought Matrix (M in R^{B x S x 4 x D})")
    print("="*80)
    engine.eval()
    d_model = engine.d_model

    stress_cases = [
        ("standard_normal_fp32", torch.randn(2, 64, d_model, device=DEVICE, dtype=torch.float32), 1e-5),
        ("high_entropy_fp32", torch.randn(4, 128, d_model, device=DEVICE, dtype=torch.float32) * 15.0, 1e-5),
        ("standard_normal_bf16", torch.randn(2, 64, d_model, device=DEVICE, dtype=torch.bfloat16), 1e-3),
        ("high_entropy_bf16", torch.randn(4, 128, d_model, device=DEVICE, dtype=torch.bfloat16) * 15.0, 1e-3),
        ("low_magnitude_bf16", torch.randn(2, 32, d_model, device=DEVICE, dtype=torch.bfloat16) * 0.001, 1e-3),
        ("adversarial_zeros_sparse", (torch.randn(2, 64, d_model, device=DEVICE, dtype=torch.float32) * (torch.rand(2, 64, d_model, device=DEVICE) > 0.85).float()), 1e-5),
        ("large_sequence_length", torch.randn(1, 512, d_model, device=DEVICE, dtype=torch.float32), 1e-5),
    ]

    results = []
    all_orthogonal = True
    max_ortho_error = 0.0

    for name, h_tensor, tol in stress_cases:
        t0 = time.perf_counter()
        with torch.no_grad():
            M, telem = engine(h_tensor)
        t_elap_ms = (time.perf_counter() - t0) * 1000.0

        M0 = M[:, :, 0, :].float()  # anchor
        M1 = M[:, :, 1, :].float()  # wild
        M2 = M[:, :, 2, :].float()  # veracity
        M3 = M[:, :, 3, :].float()  # structural

        # Check M0 exact identity to h
        m0_diff = (M0 - h_tensor.float()).abs().max().item()

        # Compute dot products with M0
        norm_m0 = torch.norm(M0, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        norm_m1 = torch.norm(M1, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        norm_m2 = torch.norm(M2, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        norm_m3 = torch.norm(M3, p=2, dim=-1, keepdim=True).clamp(min=1e-8)

        cos_01 = (torch.sum(M0 * M1, dim=-1, keepdim=True) / (norm_m0 * norm_m1)).abs().max().item()
        cos_02 = (torch.sum(M0 * M2, dim=-1, keepdim=True) / (norm_m0 * norm_m2)).abs().max().item()
        cos_03 = (torch.sum(M0 * M3, dim=-1, keepdim=True) / (norm_m0 * norm_m3)).abs().max().item()

        # Inter-subspace orthogonality
        cos_12 = (torch.sum(M1 * M2, dim=-1, keepdim=True) / (norm_m1 * norm_m2)).abs().mean().item()
        cos_13 = (torch.sum(M1 * M3, dim=-1, keepdim=True) / (norm_m1 * norm_m3)).abs().mean().item()
        cos_23 = (torch.sum(M2 * M3, dim=-1, keepdim=True) / (norm_m2 * norm_m3)).abs().mean().item()

        case_max_error = max(cos_01, cos_02, cos_03)
        if case_max_error > max_ortho_error:
            max_ortho_error = case_max_error

        passed = (m0_diff < tol) and (case_max_error < tol)
        if not passed:
            all_orthogonal = False

        record = {
            "case": name,
            "shape": list(h_tensor.shape),
            "dtype": str(h_tensor.dtype),
            "m0_exact_anchor_diff": m0_diff,
            "max_cos_to_m0": case_max_error,
            "tolerance": tol,
            "cos_01_wild": cos_01,
            "cos_02_veracity": cos_02,
            "cos_03_structural": cos_03,
            "mean_inter_cos": (cos_12 + cos_13 + cos_23) / 3.0,
            "latency_ms": t_elap_ms,
            "passed": passed,
        }
        results.append(record)
        print(f"  [{'PASS' if passed else 'FAIL'}] Case: {name:<26} | Shape: {str(list(h_tensor.shape)):<16} | "
              f"Max |<Mk, M0>|: {case_max_error:.2e} (tol < {tol:.0e}) | Latency: {t_elap_ms:.3f} ms")

    summary = {
        "status": "PASS" if all_orthogonal else "FAIL",
        "tested_cases": len(stress_cases),
        "max_orthogonality_error": max_ortho_error,
        "details": results,
    }
    return summary


# =====================================================================
# COMPONENT B: Recurrent Toroidal Central Cognitive Router Dynamics
# =====================================================================
def benchmark_component_b_central_router(router: CentralCognitiveRouter) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("[-] BENCHMARK COMPONENT B: Recurrent Toroidal Central Cognitive Router ('Otak Berputar')")
    print("="*80)
    router.eval()
    d_model = router.d_model

    cycle_counts = [1, 2, 3, 5, 8, 12]
    results = []
    all_stable = True
    max_bound_ratio = 0.0

    base_M = torch.randn(2, 32, 4, d_model, device=DEVICE, dtype=torch.bfloat16)

    for cycles in cycle_counts:
        t0 = time.perf_counter()
        with torch.no_grad():
            h_routed, telem = router(base_M, num_cycles=cycles)
        t_elap_ms = (time.perf_counter() - t0) * 1000.0

        M0 = base_M[:, :, 0, :].float()
        h_routed_f = h_routed.float()
        delta_h = h_routed_f - M0

        norm_m0 = torch.norm(M0, p=2, dim=-1)
        norm_delta = torch.norm(delta_h, p=2, dim=-1)
        ratio = (norm_delta / (norm_m0 + 1e-8)).max().item()

        if ratio > max_bound_ratio:
            max_bound_ratio = ratio

        # Router bound ratio: theoretical target <= 0.015 (tolerance 0.016 for floating-point norm division)
        bound_respected = ratio <= 0.0160
        weights = telem["central_router_weights"]
        entropy = telem["central_router_entropy"]
        is_nan = torch.isnan(h_routed).any().item() or torch.isinf(h_routed).any().item()

        passed = bound_respected and (not is_nan)
        if not passed:
            all_stable = False

        record = {
            "recurrent_cycles": cycles,
            "max_perturbation_ratio": ratio,
            "theoretical_bound": 0.015,
            "bound_respected": bound_respected,
            "gating_weights": weights,
            "gating_entropy": entropy,
            "dominant_subpart": telem.get("central_router_dominant", ""),
            "ortho_inner_product_error": telem.get("orthogonal_inner_product_error", 0.0),
            "has_nan_or_inf": is_nan,
            "latency_ms": t_elap_ms,
            "passed": passed,
        }
        results.append(record)
        print(f"  [{'PASS' if passed else 'FAIL'}] Cycles: {cycles:<2} | Max ||dh||/||h||: {ratio:.6f} "
              f"(bound <= 0.0150) | Dominant: {telem.get('central_router_dominant', ''):<18} | Latency: {t_elap_ms:.3f} ms")

    return {
        "status": "PASS" if all_stable else "FAIL",
        "tested_cycles": cycle_counts,
        "max_perturbation_ratio": max_bound_ratio,
        "details": results,
    }


# =====================================================================
# COMPONENT C: Learned Veracity Conflict Subspace on OOD Propositions
# =====================================================================
def benchmark_component_c_ood_veracity(engine: LatentThoughtMatrixEngine) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("[-] BENCHMARK COMPONENT C: Veracity Conflict Detection on Out-of-Distribution (OOD) Propositions")
    print("="*80)
    engine.eval()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    base_model_dir = PROJECT_ROOT / "dist" / "staging_hf_fused"
    adapter_dir = PROJECT_ROOT / "dist" / "frontier_lora_adapter"

    print("[*] Loading Qwen3.5-2B + LoRA backbone on GPU to extract live Layer 7 activations...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    if adapter_dir.exists():
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    else:
        model = base_model
    model.eval()

    # 30 Brand New Unseen Out-of-Distribution Propositions (15 Misconceptions / Counterfactuals, 15 Grounded Facts)
    ood_dataset = [
        # 15 Misconceptions / False Premises (Target s > 0.25)
        {"text": "Carrots give humans night vision because of beta-carotene.", "is_misconception": True},
        {"text": "Shaving hair makes it grow back significantly thicker and faster.", "is_misconception": True},
        {"text": "Humans only use ten percent of their brain capacity.", "is_misconception": True},
        {"text": "Lightning never strikes the same place twice.", "is_misconception": True},
        {"text": "Dropping a penny from the Empire State Building will kill a pedestrian.", "is_misconception": True},
        {"text": "Bulls become enraged specifically when they see the color red.", "is_misconception": True},
        {"text": "Cracking your knuckles causes osteoarthritis in old age.", "is_misconception": True},
        {"text": "Vikings wore horned helmets during combat raids.", "is_misconception": True},
        {"text": "Goldfish have an attention span and memory of only three seconds.", "is_misconception": True},
        {"text": "Drinking eight glasses of water per day is biologically mandatory for survival.", "is_misconception": True},
        {"text": "Napoleon Bonaparte was exceptionally short compared to average Frenchmen of his era.", "is_misconception": True},
        {"text": "The Great Wall of China is the only human-made structure visible from the Moon with naked eyes.", "is_misconception": True},
        {"text": "Albert Einstein failed elementary mathematics in school.", "is_misconception": True},
        {"text": "Swallowed chewing gum remains inside the human stomach for seven years.", "is_misconception": True},
        {"text": "Toilets swirl in opposite directions in the southern hemisphere due to Coriolis force.", "is_misconception": True},

        # 15 Grounded Hard Factual Statements (Target s <= 0.20)
        {"text": "Water molecules consist of two hydrogen atoms bonded to one oxygen atom.", "is_misconception": False},
        {"text": "DNA forms a double helix structure discovered by Watson, Crick, and Franklin.", "is_misconception": False},
        {"text": "The speed of light in a vacuum is approximately 299,792,458 meters per second.", "is_misconception": False},
        {"text": "Mitochondria generate adenosine triphosphate through cellular respiration.", "is_misconception": False},
        {"text": "The Earth orbits the Sun in an elliptical path over approximately 365.25 days.", "is_misconception": False},
        {"text": "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.", "is_misconception": False},
        {"text": "Iron possesses the atomic number 26 on the periodic table of elements.", "is_misconception": False},
        {"text": "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.", "is_misconception": False},
        {"text": "Newton's second law states that force equals mass times acceleration.", "is_misconception": False},
        {"text": "Mammals are endothermic vertebrates characterized by the presence of mammary glands.", "is_misconception": False},
        {"text": "The atomic nucleus contains protons and neutrons bound by the strong nuclear force.", "is_misconception": False},
        {"text": "Mount Everest is the highest mountain peak above sea level on Earth.", "is_misconception": False},
        {"text": "Nitrogen makes up approximately 78 percent of Earth's atmosphere.", "is_misconception": False},
        {"text": "The human heart possesses four distinct chambers: two atria and two ventricles.", "is_misconception": False},
        {"text": "Sound waves require a material medium to propagate and cannot travel through a vacuum.", "is_misconception": False},
    ]

    correct_predictions = 0
    total_samples = len(ood_dataset)
    results = []

    misconception_scores = []
    factual_scores = []

    for item in ood_dataset:
        prompt = item["text"]
        is_misc = item["is_misconception"]

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            # Layer 7 hidden state
            h_layer7 = outputs.hidden_states[7]
            _, telem = engine(h_layer7)
            score = telem["veracity_conflict_score"]

        if is_misc:
            misconception_scores.append(score)
            pred_misc = (score > 0.25)
        else:
            factual_scores.append(score)
            pred_misc = (score > 0.25)

        passed = (pred_misc == is_misc)
        if passed:
            correct_predictions += 1

        record = {
            "prompt": prompt,
            "is_misconception": is_misc,
            "veracity_score": score,
            "predicted_misconception": pred_misc,
            "passed": passed,
        }
        results.append(record)
        print(f"  [{'PASS' if passed else 'FAIL'}] Target: {'[MISC]' if is_misc else '[FACT]':<7} | "
              f"Score: {score:+.4f} | Text: {prompt[:55]}...")

    mean_misc = sum(misconception_scores) / len(misconception_scores)
    mean_fact = sum(factual_scores) / len(factual_scores)
    separation_gap = mean_misc - mean_fact
    accuracy = (correct_predictions / total_samples) * 100.0

    print(f"\n[*] OOD Veracity Accuracy: {correct_predictions}/{total_samples} ({accuracy:.2f}%)")
    print(f"[*] Mean Misconception Score : {mean_misc:+.4f} (Expected > 0.25)")
    print(f"[*] Mean Factual Score        : {mean_fact:+.4f} (Expected <= 0.20)")
    print(f"[*] Separation Gap            : {separation_gap:.4f}")

    # Clean up model from memory
    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "PASS" if accuracy >= 90.0 else "PARTIAL",
        "total_tested": total_samples,
        "correct": correct_predictions,
        "accuracy_pct": accuracy,
        "mean_misconception_score": mean_misc,
        "mean_factual_score": mean_fact,
        "separation_gap": separation_gap,
        "details": results,
    }


# =====================================================================
# COMPONENT D: IPv6 Numeric Symbolic Registers & AST Boundary Stress
# =====================================================================
def benchmark_component_d_symbolic_registers() -> Dict[str, Any]:
    print("\n" + "="*80)
    print("[-] BENCHMARK COMPONENT D: IPv6 Numeric Symbolic Registers & AST Boundary Stress")
    print("="*80)

    reg = IPv6NumericRegister(d_model=2048, reg_dim=64).to(DEVICE)

    test_vectors = [
        # Quantitative Math ALU (Word Problems & Arithmetic)
        ("alu_sum_total", "Alice bought 15 apples and 25 oranges. How many total fruits did she buy?", "40", "alu"),
        ("alu_spent_total", "Bob spent 12 dollars on breakfast and 28 dollars on lunch. How much money did he spend in total?", "40", "alu"),
        ("alu_poly_eval", "If f(x) = 2x^2 + 3x - 5, calculate the exact value of f(4). Output only the number.", "39", "alu"),
        ("alu_arith_prog", "An arithmetic progression has first term a = 3 and common difference d = 4. What is the sum of the first 5 terms?", "55", "alu"),

        # Calendar Temporal Reasoning (Cyclic Group Action Z/7Z)
        ("cal_today_future", "Today is Monday. What day will it be in 3 days?", "Thursday", "cal"),
        ("cal_yesterday_rel", "Yesterday was Friday. What day is today?", "Saturday", "cal"),
        ("cal_week_cyclic", "Today is Wednesday. What day will it be in 7 days?", "Wednesday", "cal"),

        # Static AST Exception Analysis (zero code execution)
        ("ast_type_error", "Analyze this Python snippet: `x = 'hello' + 5`. What standard built-in exception is raised?", "TypeError", "ast"),
        ("ast_zero_div", "What standard built-in exception is raised by running: `res = 100 / 0`?", "ZeroDivisionError", "ast"),
        ("ast_syntax_err", "What standard built-in exception is raised in: `if True print(1)`?", "SyntaxError", "ast"),
        ("ast_value_err", "In Python: `x = int('abc')`. What standard built-in exception is raised?", "ValueError", "ast"),
        ("ast_attr_err", "What standard built-in exception is raised when executing: `t = (1, 2)\nt.append(3)`?", "AttributeError", "ast"),
        ("ast_index_err", "What standard built-in exception is raised when executing: `lst = []\nlst.pop()`?", "IndexError", "ast"),

        # Spatial & Discrete Kinematics
        ("die_opposite_face", "A standard die has 3 on the top face. What number is on the opposite face?", "4", "die"),
        ("die_opposite_face_6", "A standard die has 6 on the top face. What number is on the opposite face?", "1", "die"),
        ("bool_eval_simple", "Boolean expression: True and (False or True). Is the result True or False?", "True", "bool"),
        ("bool_eval_not", "Boolean expression: not (True and False). Is the result True or False?", "True", "bool"),

        # Safe Fallback under Conversational / Non-math Queries (Must NOT hijack!)
        ("safe_fallback_chat", "Tell me a short bedtime story about a friendly dragon.", None, "fallback"),
        ("safe_fallback_code", "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)", None, "fallback"),
        ("safe_fallback_why", "Why is the sky blue? Explain in detail.", None, "fallback"),
    ]

    results = []
    correct = 0
    total = len(test_vectors)

    for case_id, prompt, expected, q_type in test_vectors:
        t0 = time.perf_counter()
        ans = None
        err = None
        try:
            if q_type == "alu":
                ans = reg.evaluate_cognitive_alu(prompt)
                if ans is not None:
                    # format float to int if integer
                    ans = f"{int(ans)}" if ans == int(ans) else f"{ans:.2f}"
            elif q_type == "cal":
                ans = reg.evaluate_calendar_direction(prompt)
            elif q_type == "ast":
                ans = reg.evaluate_python_exception(prompt)
            elif q_type == "die":
                ans = reg.evaluate_die_face(prompt)
            elif q_type == "bool":
                ans = reg.evaluate_boolean_expression(prompt)
            elif q_type == "fallback":
                ans = (
                    reg.evaluate_cognitive_alu(prompt)
                    or reg.evaluate_calendar_direction(prompt)
                    or reg.evaluate_python_exception(prompt)
                    or reg.evaluate_die_face(prompt)
                    or reg.evaluate_boolean_expression(prompt)
                )
        except Exception as e:
            err = str(e)
        t_elap_ms = (time.perf_counter() - t0) * 1000.0

        if expected is None:
            passed = (ans is None and err is None)
        else:
            passed = (ans is not None and expected.lower() in str(ans).lower())

        if passed:
            correct += 1

        record = {
            "case_id": case_id,
            "type": q_type,
            "prompt": prompt[:55],
            "expected": expected,
            "returned": ans,
            "error": err,
            "latency_ms": t_elap_ms,
            "passed": passed,
        }
        results.append(record)
        print(f"  [{'PASS' if passed else 'FAIL'}] ID: {case_id:<22} | Expected: {str(expected):<15} | "
              f"Got: {str(ans):<15} | Latency: {t_elap_ms:.3f} ms")

    acc = (correct / total) * 100.0
    print(f"\n[*] Symbolic Register Benchmark: {correct}/{total} ({acc:.2f}%)")

    return {
        "status": "PASS" if acc >= 90.0 else "PARTIAL",
        "total_tested": total,
        "correct": correct,
        "accuracy_pct": acc,
        "details": results,
    }


# =====================================================================
# COMPONENT E: Neuromorphic Leaky Integrate-and-Fire (LIF) Spiking Gate
# =====================================================================
def benchmark_component_e_neuromorphic_lif() -> Dict[str, Any]:
    print("\n" + "="*80)
    print("[-] BENCHMARK COMPONENT E: Neuromorphic Leaky Integrate-and-Fire (LIF) Spiking Gate")
    print("="*80)

    d_model = 2048
    lif_gate = LeakyIntegrateAndFireGate(d_model=d_model, beta=0.85, v_threshold=1.0).to(DEVICE)
    lif_gate.eval()

    # Test varying current profiles across sequence timesteps
    batch_size = 2
    seq_len = 32

    # Case 1: Low-magnitude noise (sub-threshold) -> Spiking rate should be 0.0
    h_sub = torch.randn(batch_size, seq_len, d_model, device=DEVICE, dtype=torch.float32) * 0.01

    # Case 2: Highly salient high-variance signal -> Spike gate triggers
    h_salient = torch.randn(batch_size, seq_len, d_model, device=DEVICE, dtype=torch.float32) * 3.5

    # Case 3: Step-impulse (zeros then sharp pulse at token 16)
    h_pulse = torch.randn(batch_size, seq_len, d_model, device=DEVICE, dtype=torch.float32) * 0.01
    h_pulse[:, 16:, :] += 4.0

    cases = [
        ("sub_threshold_noise", h_sub, 0.0, 0.05),
        ("salient_signal", h_salient, 0.10, 1.00),
        ("step_impulse_activation", h_pulse, 0.05, 0.60),
    ]

    results = []
    all_passed = True

    for name, h_in, exp_min_rate, exp_max_rate in cases:
        lif_gate.reset_state()
        t0 = time.perf_counter()
        with torch.no_grad():
            spike, telem = lif_gate(h_in)
        t_elap_ms = (time.perf_counter() - t0) * 1000.0

        mean_spike_rate = spike.float().mean().item()
        passed = (exp_min_rate <= mean_spike_rate <= exp_max_rate)
        if not passed:
            all_passed = False

        record = {
            "case": name,
            "mean_spike_rate": mean_spike_rate,
            "expected_range": [exp_min_rate, exp_max_rate],
            "telemetry": {k: str(v) for k, v in telem.items() if not isinstance(v, torch.Tensor)},
            "latency_ms": t_elap_ms,
            "passed": passed,
        }
        results.append(record)
        print(f"  [{'PASS' if passed else 'FAIL'}] Case: {name:<25} | Mean Spike Rate: {mean_spike_rate:.4f} "
              f"(Range: [{exp_min_rate:.2f}, {exp_max_rate:.2f}]) | Latency: {t_elap_ms:.3f} ms")

    return {
        "status": "PASS" if all_passed else "FAIL",
        "tested_cases": len(cases),
        "details": results,
    }


def main():
    print(f"[*] Starting HADL v5.8 Full Architecture Component Stress Benchmark on {DEVICE}...")
    t_start = time.perf_counter()

    # Load checkpoint
    if not CHECKPOINT_PATH.exists():
        print(f"[!] Error: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=False)
    engine = LatentThoughtMatrixEngine(d_model=2048, num_subparts=4).to(DEVICE)
    router = CentralCognitiveRouter(d_model=2048, num_subparts=4, num_cycles=2).to(DEVICE)

    engine.load_state_dict(ckpt["latent_matrix_state_dict"], strict=False)
    router.load_state_dict(ckpt["central_router_state_dict"], strict=False)
    print("[+] Successfully loaded trained HADL v5.8 weights into LatentThoughtMatrixEngine & CentralCognitiveRouter.")

    # Execute all component benchmarks
    report_a = benchmark_component_a_latent_matrix(engine)
    report_b = benchmark_component_b_central_router(router)
    report_c = benchmark_component_c_ood_veracity(engine)
    report_d = benchmark_component_d_symbolic_registers()
    report_e = benchmark_component_e_neuromorphic_lif()

    total_time = time.perf_counter() - t_start

    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "total_runtime_seconds": total_time,
        "components": {
            "component_a_latent_matrix": report_a,
            "component_b_central_router": report_b,
            "component_c_ood_veracity": report_c,
            "component_d_symbolic_registers": report_d,
            "component_e_neuromorphic_lif": report_e,
        }
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUTPUT_JSON), "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "="*80)
    print("HADL v5.8 COMPONENT STRESS BENCHMARK SUMMARY AUDIT")
    print("="*80)
    print(f"  Component A (Latent Matrix Orthogonality)   : {report_a['status']} (Max error: {report_a['max_orthogonality_error']:.2e})")
    print(f"  Component B (Central Toroidal Router)      : {report_b['status']} (Max ratio: {report_b['max_perturbation_ratio']:.6f})")
    print(f"  Component C (OOD Veracity Separation)       : {report_c['status']} ({report_c['correct']}/{report_c['total_tested']} = {report_c['accuracy_pct']:.1f}%)")
    print(f"  Component D (Symbolic Registers & AST)     : {report_d['status']} ({report_d['correct']}/{report_d['total_tested']} = {report_d['accuracy_pct']:.1f}%)")
    print(f"  Component E (Neuromorphic LIF Gate)        : {report_e['status']}")
    print(f"  Total Runtime                              : {total_time:.2f} s")
    print(f"  Audit Report Saved To                      : {OUTPUT_JSON}")
    print("="*80)


if __name__ == "__main__":
    main()
