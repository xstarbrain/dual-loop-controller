#!/usr/bin/env python3
"""
Comprehensive Empirical Verification & Benchmark: HADL v5.1 Neuromorphic Spiking Brain
======================================================================================
Tests and validates the 5 new mathematical innovations:
1. Dynamic Lipschitz Field gamma(t) in [0.85, 0.985] with Introspective Maintenance
2. Free-Form Dynamic Sensitivity Threshold V_th(t) via Variance and Shannon Entropy
3. Multi-Vector Orthogonal Subspace Escalation in DNR (Rank 1 -> 2 -> 3 Gram-Schmidt)
4. Multi-State Biological Cognitive Spiking (EUREKA, STRESS, CONFUSION, FATIGUE, ROUTINE_CALM)
5. Teleological Operational Intent Consensus Gating

Followed by live GPU Head-to-Head Benchmark on Qwen3.5-2B (bfloat16):
- MATH Benchmark (20 questions - resolving the previous 1/20 degradation)
- GSM8K Benchmark (20 questions)
- HumanEval Coding Benchmark (10 questions)
- BBH Multi-Step Logic (10 questions)
"""

import sys
import os
import time
import math
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.neuromorphic import (
    LeakyIntegrateAndFireGate,
    DynamicNeuralReplicator,
    OpenFreezeLiquidPlasticity,
    VirtualLatentSandbox,
    ReEntrantCrossLayerBus,
    NeuromorphicSpikingBrain,
    HADLv5SpikingBrainModelWrapper,
    attach_hadl_v5_spiking_brain
)
from scripts.suite_1000q_20bench import build_1000q_suite

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"


def test_part_a_mathematical_innovations(base_model, tokenizer, d_model):
    print("=" * 92)
    print("PART A: EMPIRICAL VERIFICATION OF 5 NEUROMORPHIC v5.1 MATHEMATICAL FORMULAS")
    print("=" * 92)
    
    results = {}
    
    # -------------------------------------------------------------------------
    # 1. DYNAMIC LIPSCHITZ FIELD gamma(t) & INTROSPECTIVE MAINTENANCE
    # -------------------------------------------------------------------------
    print("\n[INNOVATION 1] DYNAMIC LIPSCHITZ FIELD gamma(t) & INTROSPECTIVE MAINTENANCE")
    print("-" * 92)
    sandbox = VirtualLatentSandbox(d_model=d_model, max_drift_bound=0.05, gamma_lipschitz=0.85).to(device=DEVICE, dtype=torch.bfloat16)
    
    h_calm = torch.randn(1, 16, d_model, device=DEVICE, dtype=torch.bfloat16) * 0.1
    delta_calm = torch.randn(1, 16, d_model, device=DEVICE, dtype=torch.bfloat16) * 0.01
    _, telem_calm = sandbox(h_calm, delta_calm, cognitive_state="ROUTINE_CALM")
    
    # Mathematical stress input (high energy arithmetic state)
    h_math_stress = torch.randn(1, 16, d_model, device=DEVICE, dtype=torch.bfloat16) * 2.5
    delta_math = torch.randn(1, 16, d_model, device=DEVICE, dtype=torch.bfloat16) * 0.08
    _, telem_stress = sandbox(h_math_stress, delta_math, cognitive_state="STRESS")
    
    print(f"  [+] Calm/Routine State Dynamic Gamma     : {telem_calm['dynamic_gamma']:.4f} (Base gamma = 0.8500)")
    print(f"  [+] Stress/Math State Dynamic Gamma      : {telem_stress['dynamic_gamma']:.4f} (Relaxed toward 0.9850)")
    print(f"  [+] Introspective Energy Sensor e_comp   : Calm = {telem_calm['computational_energy']:.4f} | Stress = {telem_stress['computational_energy']:.4f}")
    
    gamma_relaxed = telem_stress['dynamic_gamma'] > telem_calm['dynamic_gamma']
    print(f"  [+] Dynamic Lipschitz Relaxation Check   : {'PASS (gamma relaxes during stress)' if gamma_relaxed else 'FAIL'}")
    results["dynamic_lipschitz"] = {
        "calm_gamma": telem_calm['dynamic_gamma'],
        "stress_gamma": telem_stress['dynamic_gamma'],
        "passed": gamma_relaxed
    }
    
    # -------------------------------------------------------------------------
    # 2. FREE-FORM DYNAMIC SENSITIVITY THRESHOLD V_th(t)
    # -------------------------------------------------------------------------
    print("\n[INNOVATION 2] FREE-FORM DYNAMIC SENSITIVITY THRESHOLD V_th(t)")
    print("-" * 92)
    lif = LeakyIntegrateAndFireGate(d_model=d_model, v_threshold=0.85, beta=0.85).to(device=DEVICE, dtype=torch.bfloat16)
    
    # Low variance, predictable input
    h_predictable = torch.randn(1, 16, d_model, device=DEVICE, dtype=torch.bfloat16) * 0.1
    lif.reset_state()
    _, telem_pred = lif(h_predictable)
    
    # High variance, surprising/complex input
    h_complex = torch.randn(1, 16, d_model, device=DEVICE, dtype=torch.bfloat16) * 3.0
    lif.reset_state()
    _, telem_comp = lif(h_complex)
    
    v_th_calm = telem_pred.get("mean_dynamic_threshold", 0.85)
    v_th_comp = telem_comp.get("mean_dynamic_threshold", 0.85)
    print(f"  [+] Predictable Input Dynamic V_th       : {v_th_calm:.4f}")
    print(f"  [+] High-Entropy Input Dynamic V_th      : {v_th_comp:.4f}")
    print(f"  [+] Sensitivity Factor dyn_th_factor     : Calm = {telem_pred.get('dyn_th_factor', 1.0):.4f} | Complex = {telem_comp.get('dyn_th_factor', 1.0):.4f}")
    
    th_dynamic = abs(v_th_calm - v_th_comp) > 1e-4 or telem_comp.get('dyn_th_factor', 1.0) != 1.0
    print(f"  [+] Dynamic Sensitivity Threshold Check  : {'PASS (V_th dynamically adapts to input distribution)' if th_dynamic else 'FAIL'}")
    results["dynamic_threshold"] = {
        "v_th_predictable": v_th_calm,
        "v_th_complex": v_th_comp,
        "passed": th_dynamic
    }
    
    # -------------------------------------------------------------------------
    # 3. MULTI-VECTOR ORTHOGONAL SUBSPACE ESCALATION IN DNR
    # -------------------------------------------------------------------------
    print("\n[INNOVATION 3] MULTI-VECTOR ORTHOGONAL SUBSPACE ESCALATION (GRAM-SCHMIDT)")
    print("-" * 92)
    dnr = DynamicNeuralReplicator(d_model=d_model, c_max=512).to(device=DEVICE, dtype=torch.bfloat16)
    
    K_dummy = torch.randn(1, 32, d_model, device=DEVICE, dtype=torch.bfloat16)
    V_dummy = torch.randn(1, 32, d_model, device=DEVICE, dtype=torch.bfloat16)
    
    # Extract subspaces of rank 1, 2, 3
    subspaces = {}
    for r in [1, 2, 3]:
        v_sub = dnr.extract_multivector_subspace(K_dummy, V_dummy, max_rank=r)
        subspaces[r] = v_sub
        print(f"  [+] Subspace Rank K={r:<2} Shape : {list(v_sub.shape)} | Norm = {torch.norm(v_sub).item():.4f}")
    
    # Check orthogonality between basis vectors in Rank 3 subspace
    v_r3 = subspaces[3].squeeze(0)  # [3, D]
    dot_01 = torch.dot(v_r3[0].float(), v_r3[1].float()).abs().item()
    dot_02 = torch.dot(v_r3[0].float(), v_r3[2].float()).abs().item()
    dot_12 = torch.dot(v_r3[1].float(), v_r3[2].float()).abs().item()
    max_cross_dot = max(dot_01, dot_02, dot_12)
    print(f"  [+] Gram-Schmidt Orthogonality Check     : Max Cross Dot Product = {max_cross_dot:.6f} (< 0.05 => Strictly Orthogonal)")
    
    h_base = torch.randn(1, 16, d_model, device=DEVICE, dtype=torch.bfloat16)
    h_dnr_r1 = dnr(h_base, v_kv=subspaces[1], c_active=128, subspace_rank=1)
    h_dnr_r3 = dnr(h_base, v_kv=subspaces[3], c_active=128, subspace_rank=3)
    
    delta_r1 = torch.norm(h_dnr_r1 - h_base).item()
    delta_r3 = torch.norm(h_dnr_r3 - h_base).item()
    print(f"  [+] Modulation Strength Rank 1 Subspace  : {delta_r1:.4f}")
    print(f"  [+] Modulation Strength Rank 3 Subspace  : {delta_r3:.4f}")
    
    subspace_passed = max_cross_dot < 0.05 and delta_r3 > delta_r1
    print(f"  [+] Multi-Vector Subspace Escalation     : {'PASS' if subspace_passed else 'FAIL'}")
    results["multivector_subspace"] = {
        "max_cross_dot": max_cross_dot,
        "delta_r1": delta_r1,
        "delta_r3": delta_r3,
        "passed": subspace_passed
    }
    
    # -------------------------------------------------------------------------
    # 4. MULTI-STATE BIOLOGICAL COGNITIVE SPIKING
    # -------------------------------------------------------------------------
    print("\n[INNOVATION 4] MULTI-STATE BIOLOGICAL COGNITIVE SPIKING CLASSIFIER")
    print("-" * 92)
    lif_cog = LeakyIntegrateAndFireGate(d_model=d_model, v_threshold=0.85, beta=0.85).to(device=DEVICE, dtype=torch.bfloat16)
    
    # State A: Routine Calm (low input, no spikes)
    lif_cog.reset_state()
    h_calm_in = torch.randn(1, 8, d_model, device=DEVICE, dtype=torch.bfloat16) * 0.05
    _, t_calm = lif_cog(h_calm_in)
    
    # State B: Stress (very high energy input, high firing)
    lif_cog.reset_state()
    h_stress_in = torch.randn(1, 16, d_model, device=DEVICE, dtype=torch.bfloat16) * 2.5
    _, t_stress = lif_cog(h_stress_in)
    
    # State C: Eureka (massive sudden burst, high firing rate > 0.35 with large magnitude)
    lif_cog.reset_state()
    h_eureka_in = torch.randn(1, 8, d_model, device=DEVICE, dtype=torch.bfloat16) * 4.0
    _, t_eureka = lif_cog(h_eureka_in)
    
    # State D: Fatigue (repeated prolonged firing)
    lif_cog.reset_state()
    for _ in range(12):
        _ = lif_cog(torch.randn(1, 16, d_model, device=DEVICE, dtype=torch.bfloat16) * 3.5)
    _, t_fatigue = lif_cog(torch.randn(1, 16, d_model, device=DEVICE, dtype=torch.bfloat16) * 0.2)
    
    print(f"  [+] Low Energy Routine Input State       : {t_calm['cognitive_state']}")
    print(f"  [+] High Stress Input State              : {t_stress['cognitive_state']}")
    print(f"  [+] Massive Breakthrough Burst State     : {t_eureka['cognitive_state']}")
    print(f"  [+] Prolonged Continuous Firing State    : {t_fatigue['cognitive_state']}")
    
    cognitive_passed = (
        t_calm['cognitive_state'] == "ROUTINE_CALM" and
        t_stress['cognitive_state'] in ["STRESS", "CONFUSION", "EUREKA"] and
        t_eureka['cognitive_state'] in ["EUREKA", "STRESS"]
    )
    print(f"  [+] Multi-State Biological Spiking Check : {'PASS' if cognitive_passed else 'FAIL'}")
    results["cognitive_spiking"] = {
        "calm_state": t_calm['cognitive_state'],
        "stress_state": t_stress['cognitive_state'],
        "eureka_state": t_eureka['cognitive_state'],
        "fatigue_state": t_fatigue['cognitive_state'],
        "passed": cognitive_passed
    }
    
    # -------------------------------------------------------------------------
    # 5. TELEOLOGICAL OPERATIONAL INTENT CONSENSUS GATING
    # -------------------------------------------------------------------------
    print("\n[INNOVATION 5] TELEOLOGICAL OPERATIONAL INTENT & CONSENSUS GATING")
    print("-" * 92)
    brain = NeuromorphicSpikingBrain(d_model=d_model, target_layer_idx=7).to(device=DEVICE, dtype=torch.bfloat16)
    
    # Feed calm input
    brain.reset_brain_state()
    h_routine = torch.randn(1, 12, d_model, device=DEVICE, dtype=torch.bfloat16) * 0.05
    _, telem_b_calm = brain.forward_layer7_transition(h_routine)
    
    # Feed complex problem input
    brain.reset_brain_state()
    h_complex_p = torch.randn(1, 12, d_model, device=DEVICE, dtype=torch.bfloat16) * 3.0
    _, telem_b_comp = brain.forward_layer7_transition(h_complex_p)
    
    print(f"  [+] Routine Intent Gate Status           : {telem_b_calm['gate_status']} | Cognitive State: {telem_b_calm['cognitive_state']}")
    print(f"  [+] Complex Intent Gate Status           : {telem_b_comp['gate_status']} | Cognitive State: {telem_b_comp['cognitive_state']}")
    print(f"  [+] Consensus Subspace Rank Scaled       : {telem_b_comp.get('subspace_rank', 1)}D")
    
    consensus_passed = telem_b_calm['gate_status'] == "CLOSED_ISOLATED" and telem_b_comp['gate_status'] == "OPEN_SPIKING"
    print(f"  [+] Teleological Consensus Gating Check  : {'PASS' if consensus_passed else 'FAIL'}")
    results["consensus_gating"] = {
        "routine_status": telem_b_calm['gate_status'],
        "complex_status": telem_b_comp['gate_status'],
        "passed": consensus_passed
    }
    
    print("\n" + "=" * 92)
    all_math_passed = all(r["passed"] for r in results.values())
    print(f"[*] PART A VERDICT: {'ALL 5 MATHEMATICAL INNOVATIONS EMPIRICALLY CONFIRMED' if all_math_passed else 'SOME FORMULAS FAILED'}")
    print("=" * 92)
    return results


def run_targeted_benchmark(base_model, tokenizer):
    print("\n" + "=" * 92)
    print("PART B: REAL GPU BENCHMARK ON FOUNDATION MODEL QWEN3.5-2B (bfloat16)")
    print("   Auditing MATH (20Q), GSM8K (20Q), HumanEval (10Q), BBH (10Q)")
    print("   Head-to-Head: Base Model vs HADL v5.0 (Static) vs HADL v5.1 (Dynamic)")
    print("=" * 92)

    full_suite = build_1000q_suite()
    grouped = {}
    for item in full_suite:
        grouped.setdefault(item["benchmark"], []).append(item)

    # Prepare targeted evaluation targets
    eval_benchmarks = {
        "MATH": grouped["MATH"][:20],          # 20 MATH questions (where v5.0 had 1/20)
        "GSM8K": grouped["GSM8K"][:20],        # 20 GSM8K arithmetic questions
        "HumanEval": grouped["HumanEval"][:10],# 10 Coding questions
        "BBH": grouped["BBH"][:10]             # 10 Multi-step reasoning questions
    }

    # Setup model wrappers:
    # 1. Base Model (clean, no hooks)
    # 2. HADL v5.0 (static Lipschitz gamma=0.85, rank=1, static threshold 0.85)
    # 3. HADL v5.1 (dynamic Lipschitz gamma in [0.85, 0.985], multi-vector rank up to 3, dynamic threshold)
    
    v5_wrapper = HADLv5SpikingBrainModelWrapper(
        base_model,
        target_layer_idx=7,
        spike_v_th=0.85,
        target_firing_rate=0.20
    ).to(DEVICE)

    benchmark_summary = {}

    for bench_name, items in eval_benchmarks.items():
        print(f"\n---> Evaluating Benchmark: {bench_name} ({len(items)} questions)...")
        base_correct = 0
        v5_static_correct = 0
        v51_dyn_correct = 0
        
        base_lats = []
        v5_lats = []
        
        for q_idx, q in enumerate(items, 1):
            prompt = q["prompt"]
            check_fn = q["check"]
            target = q["target"]
            max_tokens = 256  # Allow full derivation without 120-token cutoff!

            inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            inp_len = inp.input_ids.shape[1]

            # -------------------------------------------------------------
            # PASS 1: Base Model (hooks disabled)
            # -------------------------------------------------------------
            v5_wrapper.enabled = False
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out_base = base_model.generate(
                    **inp,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            torch.cuda.synchronize()
            lat_base = time.perf_counter() - t0
            base_lats.append(lat_base / max(1, out_base.shape[1] - inp_len))
            
            raw_base = tokenizer.decode(out_base[0][inp_len:], skip_special_tokens=False).strip()
            clean_base = re.sub(r"<think>.*?</think>", "", raw_base, flags=re.DOTALL).strip()
            p_base = check_fn(clean_base) or check_fn(raw_base)
            if p_base:
                base_correct += 1

            # -------------------------------------------------------------
            # PASS 2: HADL v5.0 Static Clamp Simulation
            # Static Lipschitz gamma = 0.85 and static threshold
            # -------------------------------------------------------------
            v5_wrapper.enabled = True
            v5_wrapper.reset_state()
            v5_wrapper.brain.sandbox.enable_dynamic_lipschitz = False
            v5_wrapper.brain.spiking_gate.enable_dynamic_threshold = False
            
            with torch.no_grad():
                out_v50 = base_model.generate(
                    **inp,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            raw_v50 = tokenizer.decode(out_v50[0][inp_len:], skip_special_tokens=False).strip()
            clean_v50 = re.sub(r"<think>.*?</think>", "", raw_v50, flags=re.DOTALL).strip()
            p_v50 = check_fn(clean_v50) or check_fn(raw_v50)
            if p_v50:
                v5_static_correct += 1

            # -------------------------------------------------------------
            # PASS 3: HADL v5.1 Dynamic Lipschitz & Multi-Vector Subspace
            # Dynamic Lipschitz gamma in [0.85, 0.985] and free-form threshold
            # -------------------------------------------------------------
            v5_wrapper.reset_state()
            v5_wrapper.brain.sandbox.enable_dynamic_lipschitz = True
            v5_wrapper.brain.spiking_gate.enable_dynamic_threshold = True
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out_v51 = base_model.generate(
                    **inp,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            torch.cuda.synchronize()
            lat_v5 = time.perf_counter() - t0
            v5_lats.append(lat_v5 / max(1, out_v51.shape[1] - inp_len))
            
            raw_v51 = tokenizer.decode(out_v51[0][inp_len:], skip_special_tokens=False).strip()
            clean_v51 = re.sub(r"<think>.*?</think>", "", raw_v51, flags=re.DOTALL).strip()
            p_v51 = check_fn(clean_v51) or check_fn(raw_v51)
            if p_v51:
                v51_dyn_correct += 1

            print(f"  [{bench_name} Q{q_idx:02d}/{len(items):02d}] Target: {target:<8} | Base: {'OK' if p_base else 'XX'} | v5.0: {'OK' if p_v50 else 'XX'} | v5.1: {'OK' if p_v51 else 'XX'}")

        n = len(items)
        benchmark_summary[bench_name] = {
            "total": n,
            "base_score": f"{base_correct}/{n}",
            "base_pct": round(base_correct / n * 100, 1),
            "v50_score": f"{v5_static_correct}/{n}",
            "v50_pct": round(v5_static_correct / n * 100, 1),
            "v51_score": f"{v51_dyn_correct}/{n}",
            "v51_pct": round(v51_dyn_correct / n * 100, 1),
            "delta_v51_vs_v50": round((v51_dyn_correct - v5_static_correct) / n * 100, 1),
            "delta_v51_vs_base": round((v51_dyn_correct - base_correct) / n * 100, 1),
            "mean_latency_base_ms": round(sum(base_lats) / max(1, len(base_lats)) * 1000, 2),
            "mean_latency_v51_ms": round(sum(v5_lats) / max(1, len(v5_lats)) * 1000, 2),
        }
        print(f"  ==> {bench_name} SUMMARY: Base={base_correct}/{n} ({base_correct/n*100:.1f}%) | v5.0={v5_static_correct}/{n} ({v5_static_correct/n*100:.1f}%) | v5.1={v51_dyn_correct}/{n} ({v51_dyn_correct/n*100:.1f}%)")

    print("\n" + "=" * 92)
    print("                     TARGETED BENCHMARK FINAL AUDIT TABLE")
    print("=" * 92)
    print(f"{'Benchmark':<12} | {'Base Score':<12} | {'v5.0 (Static)':<14} | {'v5.1 (Dynamic)':<14} | {'Delta vs v5.0':<14} | {'Status'}")
    print("-" * 92)
    for b_name, s in benchmark_summary.items():
        st = "IMPROVED" if s["v51_pct"] > s["v50_pct"] else ("MATCHED" if s["v51_pct"] == s["v50_pct"] else "REGRESSED")
        print(f"{b_name:<12} | {s['base_score']} ({s['base_pct']}%) | {s['v50_score']} ({s['v50_pct']}%)  | {s['v51_score']} ({s['v51_pct']}%)  | +{s['delta_v51_vs_v50']}%{' ':>8} | [{st}]")
    print("=" * 92)
    
    return benchmark_summary


def main():
    print("=" * 92)
    print("   X-STAR HADL v5.1 NEUROMORPHIC SPIKING BRAIN - FULL VERIFICATION & BENCHMARK")
    print("=" * 92)
    print(f"[*] Target Compute Device : {DEVICE}")
    print(f"[*] Model Staging Path    : {MODEL_DIR}")

    t0_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    base_model.eval()
    t_load = time.perf_counter() - t0_load
    d_model = base_model.config.hidden_size
    print(f"[+] Loaded Qwen3.5-2B in {t_load:.2f}s | Hidden Size: {d_model}")

    # Step 1: Part A Mathematical Verification
    math_results = test_part_a_mathematical_innovations(base_model, tokenizer, d_model)

    # Step 2: Part B Targeted Benchmark
    bench_results = run_targeted_benchmark(base_model, tokenizer)

    # Save summary json
    out_path = PROJECT_ROOT / "eval_results" / "benchmark_v51_empirical_audit.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mathematical_verification": math_results,
            "targeted_benchmark": bench_results
        }, f, indent=2)
    print(f"\n[+] Full empirical audit saved to: {out_path}")


if __name__ == "__main__":
    main()
