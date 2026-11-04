"""
Real GPU Empirical Verification Suite for HADL v5.7 Unified Multimodal AGI
==========================================================================
Executes end-to-end empirical verification on the physical GPU using
Qwen3.5-2B (`dist/staging_hf_fused` + `dist/frontier_lora_adapter`) with the
HADL v5.7 Self-Evolving Multimodal Spiking Brain:
1. Previously failing 400Q/1000Q benchmark items (negative quadratic roots,
   HumanEval/MBPP code generation without Shadow KV shape crash, conversational
   open-ended prompts without binary Yes/No truncation).
2. Closed-Loop Self-Evolving Learning (IntrinsicPopperianCycleVerifier +
   SynapticEligibilityTrace + OrthogonalNullspaceProjector + OpenFreezeLiquidPlasticity).
3. Endogenous Custom Sound Synthesis (`freq suara` — EndogenousAcousticVocalOrgan
   16-bit PCM .wav synthesis + Self-Listening Auditory Re-Entry Loop).
4. Progressive 2D-to-3D Visual Imagination (`Progressive2Dto3DImaginationCanvas`
   1D -> 2D Sketchpad -> 3D Gaussian/SDF Primitives + SO(3) Mental Rotation +
   3D-to-2D Splatting Verification).
"""

from __future__ import annotations

import json
import os
import sys
import time
import wave
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
from xstar.inference.engine import CustomInferenceEngine, EngineConfig
from scripts.suite_1000q_20bench import build_1000q_suite


def main():
    out_dir = ROOT / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "hadl_v57_real_voice.wav"
    report_path = out_dir / "hadl_v57_real_gpu_verification.json"

    print("=" * 78)
    print("  HADL v5.7 UNIFIED MULTIMODAL AGI - REAL GPU EMPIRICAL VERIFICATION")
    print("=" * 78)

    cfg = EngineConfig(
        model_path=str(ROOT / "dist" / "staging_hf_fused"),
        adapter_path=str(ROOT / "dist" / "frontier_lora_adapter"),
        device="auto",
        dtype="bfloat16",
        enable_brain=True,
        enable_adapter=True,
        enable_online_learning=True,
        enable_self_evolving=True,
        enable_voice=True,
        max_evolution_passes=2,
    )
    engine = CustomInferenceEngine(cfg)
    t0 = time.perf_counter()
    engine.load(status_callback=lambda msg: print(f"[LOAD] {msg}"))
    load_time_s = round(time.perf_counter() - t0, 2)

    vram = engine.get_vram_usage()
    print(f"[GPU] Device={engine.device} | VRAM Allocated={vram.get('allocated_mb')} MB | Load Time={load_time_s}s\n")

    suite_1000 = {q["id"]: q for q in build_1000q_suite()}

    test_cases = [
        {
            "id": "math_35_neg_eval",
            "category": "Math (Negative Polynomial Evaluation P(4) = -13)",
            "prompt": suite_1000["math_35"]["prompt"],
            "check": suite_1000["math_35"]["check"],
            "max_tokens": 64,
        },
        {
            "id": "math_roots_neg",
            "category": "Math (Negative Quadratic Roots x^2 + 8x + 7 = 0)",
            "prompt": "Find the roots of x^2 + 8x + 7 = 0.",
            "check": lambda r: bool(re.search(r"(?<!\w)-7\b", r)) and bool(re.search(r"(?<!\w)-1\b", r)),
            "max_tokens": 64,
        },
        {
            "id": "he_11",
            "category": "HumanEval Code (Multi-Token Stub Shadow KV Fix)",
            "prompt": suite_1000["he_11"]["prompt"],
            "check": suite_1000["he_11"]["check"],
            "max_tokens": 256,
        },
        {
            "id": "mbpp_04",
            "category": "MBPP Code (List Concatenation)",
            "prompt": suite_1000["mbpp_04"]["prompt"],
            "check": suite_1000["mbpp_04"]["check"],
            "max_tokens": 256,
        },
        {
            "id": "mbpp_18",
            "category": "MBPP Code (Index Search)",
            "prompt": suite_1000["mbpp_18"]["prompt"],
            "check": suite_1000["mbpp_18"]["check"],
            "max_tokens": 256,
        },
        {
            "id": "conv_open_ended",
            "category": "Conversational Open-Ended (No Yes/No Truncation)",
            "prompt": "Can you suggest 3 classic books on machine learning?",
            "check": lambda r: len(r.strip().split()) >= 8 and r.strip().lower() not in ("yes", "yes.", "no", "no."),
            "max_tokens": 128,
        },
    ]

    results = []
    passed_count = 0

    for tc in test_cases:
        formatted = engine.format_chat_prompt([{"role": "user", "content": tc["prompt"]}])
        resp, metrics = engine.generate(
            prompt=formatted,
            max_new_tokens=tc["max_tokens"],
            temperature=0.0,
        )
        bench_ok = bool(tc["check"](resp))
        ver_res = engine.verify_and_adapt(tc["prompt"], resp)
        intrinsic_ok = bool(ver_res.get("verified", False))
        overall_ok = bench_ok and intrinsic_ok
        if overall_ok:
            passed_count += 1

        print(f"[{tc['id']}] ({tc['category']})")
        print(f"  Prompt   : {tc['prompt']}")
        print(f"  Response : {resp.strip()[:180]}")
        print(
            f"  Status   : {'PASS' if overall_ok else 'FAIL'} "
            f"(Bench={bench_ok}, Popperian={intrinsic_ok}, Dopamine={metrics.dopamine_sign:+.1f}, "
            f"Passes={metrics.evolution_passes}, TPS={metrics.tokens_per_second:.1f}, "
            f"f0={metrics.f0_mean_hz:.1f}Hz, 2D->3D={metrics.consistency_2d_to_3d:.3f})\n"
        )

        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "prompt": tc["prompt"],
            "response": resp.strip(),
            "benchmark_passed": bench_ok,
            "popperian_verified": intrinsic_ok,
            "overall_passed": overall_ok,
            "dopamine_sign": metrics.dopamine_sign,
            "evolution_passes": metrics.evolution_passes,
            "ttft_ms": metrics.ttft_ms,
            "tps": metrics.tokens_per_second,
            "f0_mean_hz": metrics.f0_mean_hz,
            "formants_hz": metrics.formants_hz,
            "consistency_2d_to_3d": metrics.consistency_2d_to_3d,
        })

    # Synthesize dedicated 1.5s endogenous voice .wav from real GPU model
    voice_out = engine.synthesize_voice(
        text="X-Star unified multimodal brain online with endogenous harmonic vocal synthesis.",
        save_path=str(wav_path),
        duration_sec=1.5,
        enable_self_listening=True,
    )
    synth_t = voice_out.get("synthesis_telemetry", {})
    listen_t = voice_out.get("self_listening_telemetry", {})

    with wave.open(str(wav_path), "rb") as wf:
        wav_info = {
            "path": str(wav_path),
            "channels": wf.getnchannels(),
            "sample_width_bytes": wf.getsampwidth(),
            "sample_rate_hz": wf.getframerate(),
            "num_frames": wf.getnframes(),
            "duration_s": round(wf.getnframes() / float(wf.getframerate()), 3),
        }

    print(
        f"[VOICE (`freq suara`)] Saved={wav_path.name} | "
        f"f0={synth_t.get('f0_mean_hz', 0.0):.2f} Hz | "
        f"Formants={synth_t.get('formants_hz', [])} Hz | "
        f"Self-Listening Coherence: {listen_t.get('coherence_before', 0.0):.4f} -> {listen_t.get('coherence_after', 0.0):.4f}"
    )

    # Inspect 2D-to-3D Imagination telemetry on GPU
    img_telem = engine.get_3d_imagination_telemetry(
        text="Imagine a 3D crystal octahedron rotating smoothly in space."
    )
    print(
        f"[2D->3D IMAGINATION] Sketchpad={img_telem.get('sketchpad_2d_resolution')} | "
        f"3D Primitives={img_telem.get('num_3d_primitives')} | "
        f"SO(3) Orthogonality Error={img_telem.get('so3_orthogonality_error'):.2e} | "
        f"det(R)={img_telem.get('so3_det'):.4f} | "
        f"2D<->3D Splat Consistency={img_telem.get('consistency_2d_to_3d'):.4f}"
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": engine.device,
        "dtype": str(engine.torch_dtype),
        "vram_usage": engine.get_vram_usage(),
        "load_time_s": load_time_s,
        "total_tasks": len(test_cases),
        "passed_tasks": passed_count,
        "pass_rate_percent": round(100.0 * passed_count / max(1, len(test_cases)), 2),
        "results": results,
        "endogenous_voice": {
            "wav_file": wav_info,
            "synthesis_telemetry": synth_t,
            "self_listening_telemetry": listen_t,
        },
        "imagination_2d_to_3d": img_telem,
        "brain_telemetry": engine.get_brain_telemetry(),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 78)
    print(f"  VERIFICATION COMPLETE: {passed_count}/{len(test_cases)} ({summary['pass_rate_percent']}%) PASSED")
    print(f"  Report saved to: {report_path}")
    print("=" * 78)

    if passed_count < len(test_cases):
        sys.exit(1)


if __name__ == "__main__":
    main()
