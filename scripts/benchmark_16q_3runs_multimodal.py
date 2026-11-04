"""
Benchmark 16Q x 3 Runs (Zero Checkpoint Fresh Start) with Sound & Video Evolution
================================================================================
Empirical evaluation on physical NVIDIA RTX 5060 GPU:
1. Fresh initialization from 0 without prior plastic checkpoints.
2. 3 sequential runs across 16 uncurtailed questions covering Math, Coding, Science,
   Logic, and Conversational NLP.
3. Sound evolution test (`freq suara`): f0, F1..F4, self-listening auditory re-entry coherence
   tracked across Run 1, Run 2, and Run 3, saving .wav files.
4. Video spatio-temporal mental animation test: 16-frame SO(3) geodesic trajectory,
   tracking volume conservation det(R_t)=1.0, motion delta, saving .gif.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xstar.inference.engine import CustomInferenceEngine, EngineConfig


def build_16q_benchmark() -> List[Dict[str, Any]]:
    return [
        # --- Math & Quantitative (4 Questions) ---
        {
            "id": "math_roots_neg",
            "pillar": "Math & Quantitative",
            "name": "Negative Quadratic Roots (x^2 + 8x + 7 = 0)",
            "prompt": "Find the roots of x^2 + 8x + 7 = 0. Provide only the numerical values of the roots.",
            "check": lambda r: bool(re.search(r"(?<!\w)-7\b", r)) and bool(re.search(r"(?<!\w)-1\b", r)),
            "max_tokens": 128,
        },
        {
            "id": "math_35_neg_eval",
            "pillar": "Math & Quantitative",
            "name": "Polynomial Evaluation P(4) with Negative Result",
            "prompt": "Evaluate the polynomial P(x) = x^2 - 8x + 3 at x = 4. Provide the final numerical value.\nAnswer:",
            "check": lambda r: bool(re.search(r"(?<!\w)-13\b", r)),
            "max_tokens": 128,
        },
        {
            "id": "math_progression",
            "pillar": "Math & Quantitative",
            "name": "Arithmetic Sequence 5th Term (a1=3, d=-4)",
            "prompt": "In an arithmetic sequence with first term a_1 = 3 and common difference d = -4, what is the 5th term a_5?\nAnswer:",
            "check": lambda r: bool(re.search(r"(?<!\w)-13\b", r)) or "-13" in r,
            "max_tokens": 128,
        },
        {
            "id": "math_combinatorics",
            "pillar": "Math & Quantitative",
            "name": "Permutations of 4 Books (4!)",
            "prompt": "How many distinct ways can 4 distinct books be arranged in a line on a shelf? Provide the numerical answer.\nAnswer:",
            "check": lambda r: bool(re.search(r"\b24\b", r)),
            "max_tokens": 128,
        },

        # --- Coding & Software (4 Questions) ---
        {
            "id": "he_palindrome",
            "pillar": "Coding & Software",
            "name": "HumanEval Palindrome Function",
            "prompt": "Complete the Python function to check if string s is a palindrome.\n```python\ndef is_palindrome(s):\n    # Your implementation here\n```\nPython code:",
            "check": lambda r: "def is_palindrome" in r and ("return" in r or "==" in r),
            "max_tokens": 256,
        },
        {
            "id": "mbpp_04_concat",
            "pillar": "Coding & Software",
            "name": "MBPP List Concatenation",
            "prompt": "Problem 4: In Python, write a function to concatenate two lists l1 and l2.\nProvide the Python code definition for `concat_lists(l1, l2)`:\nAnswer:",
            "check": lambda r: "def concat_lists" in r and ("+" in r or "extend" in r),
            "max_tokens": 256,
        },
        {
            "id": "mbpp_18_find_idx",
            "pillar": "Coding & Software",
            "name": "MBPP Linear Index Search",
            "prompt": "Problem 18: In Python, write a function to find the index of element x in list lst.\nProvide the Python code definition for `find_index(lst, x)`:\nAnswer:",
            "check": lambda r: "def find_index" in r and ("return" in r and ("for" in r or "index" in r)),
            "max_tokens": 256,
        },
        {
            "id": "codedebug_binary_search",
            "pillar": "Coding & Software",
            "name": "Binary Search Boundary Debugging",
            "prompt": "Debug this binary search function to fix the loop condition so it finds target at the high boundary:\n```python\ndef binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low < high: # Bug here\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1\n```\nCorrect code:",
            "check": lambda r: "low <= high" in r or "<=" in r,
            "max_tokens": 256,
        },

        # --- Science & Academic (3 Questions) ---
        {
            "id": "phys_light_speed",
            "pillar": "Science & Academic",
            "name": "Speed of Light in Vacuum",
            "prompt": "What is the approximate speed of light in a vacuum in meters per second (in scientific notation, e.g., 3x10^8)?\nAnswer:",
            "check": lambda r: any(k in r.lower().replace(" ", "") for k in ["3x10^8", "3.00x10^8", "3.0x10^8", "3*10^8", "3x108", "300,000,000", "300000000", "3e8"]),
            "max_tokens": 128,
        },
        {
            "id": "bio_cell_organelle",
            "pillar": "Science & Academic",
            "name": "Mitochondria Cellular Respiration",
            "prompt": "What organelle is known as the powerhouse of eukaryotic cells responsible for producing ATP via cellular respiration?\nAnswer:",
            "check": lambda r: "mitochondri" in r.lower(),
            "max_tokens": 128,
        },
        {
            "id": "chem_neutral_ph",
            "pillar": "Science & Academic",
            "name": "Neutral Aqueous pH at 25C",
            "prompt": "What is the pH of a neutral aqueous solution at 25 degrees Celsius? Provide the single numerical value.\nAnswer:",
            "check": lambda r: bool(re.search(r"\b7(\.0)?\b", r)),
            "max_tokens": 128,
        },

        # --- Commonsense & Logic (2 Questions) ---
        {
            "id": "logic_boolean_eval",
            "pillar": "Commonsense & Logic",
            "name": "Compound Boolean Expression Evaluation",
            "prompt": "Evaluate the boolean expression: True and not (False or True). Does it evaluate to True or False?\nAnswer:",
            "check": lambda r: "false" in r.lower().split() or "false." in r.lower().split() or r.strip().lower().startswith("false"),
            "max_tokens": 128,
        },
        {
            "id": "logic_transitivity",
            "pillar": "Commonsense & Logic",
            "name": "Transitive Height Deduction",
            "prompt": "If Alice is taller than Bob, and Bob is taller than Charlie, who is the shortest among Alice, Bob, and Charlie?\nAnswer:",
            "check": lambda r: "charlie" in r.lower(),
            "max_tokens": 128,
        },

        # --- Instruction & Conversational (3 Questions) ---
        {
            "id": "conv_ml_books",
            "pillar": "Instruction & Chat",
            "name": "Open-Ended Machine Learning Book Recommendation",
            "prompt": "Can you suggest 3 classic books on machine learning?",
            "check": lambda r: len(r.strip().split()) >= 10 and r.strip().lower() not in ("yes", "yes.", "no", "no."),
            "max_tokens": 256,
        },
        {
            "id": "ifeval_bullet_colors",
            "pillar": "Instruction & Chat",
            "name": "Additive Primary Colors 3-Bullet List",
            "prompt": "List exactly the three additive primary colors of light as a bulleted list using asterisks (*):\nAnswer:",
            "check": lambda r: "red" in r.lower() and "green" in r.lower() and "blue" in r.lower() and "*" in r,
            "max_tokens": 192,
        },
        {
            "id": "json_structured",
            "pillar": "Instruction & Chat",
            "name": "Structured JSON Entity Extraction",
            "prompt": "Convert the following data into a valid JSON object with keys 'name', 'age', and 'role': Alice is a 28 year old AI researcher.\nJSON:",
            "check": lambda r: "name" in r and "age" in r and ("role" in r or "researcher" in r),
            "max_tokens": 192,
        },
    ]


def inspect_wav_file(path: Path) -> Dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        return {
            "path": str(path),
            "channels": wf.getnchannels(),
            "sample_width_bytes": wf.getsampwidth(),
            "sample_rate_hz": wf.getframerate(),
            "num_frames": wf.getnframes(),
            "duration_s": round(wf.getnframes() / float(wf.getframerate()), 3),
        }


def main():
    benchmarks_dir = ROOT / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    report_path = benchmarks_dir / "benchmark_16q_3runs_multimodal.json"
    video_gif_path = benchmarks_dir / "hadl_v57_mental_video.gif"

    print("=" * 82)
    print("  HADL v5.7 UNIFIED MULTIMODAL AGI - 16Q x 3 RUNS EMPIRICAL GPU BENCHMARK")
    print("  (Zero Checkpoint Fresh Start + Sound & Spatio-Temporal Video Evolution)")
    print("=" * 82)

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

    # Force cold start: zero out plastic weights, clear eligibility traces, reset closed loop
    if hasattr(engine.model, "brain"):
        brain = engine.model.brain
        if hasattr(brain, "plastic_unit"):
            brain.plastic_unit.w_plastic.data.zero_()
            brain.plastic_unit.initial_w.data.zero_()
            brain.plastic_unit.snapshot_initial_weights()
        if hasattr(brain, "eligibility_trace"):
            brain.eligibility_trace.reset()
        if hasattr(brain, "liquid_plasticity"):
            brain.liquid_plasticity.reset_liquid_state()
        if hasattr(brain, "vocal_organ"):
            brain.vocal_organ.reset()
        print("[RESET] Successfully zeroed all plastic weights and traces for pure cold baseline.\n")

    vram_init = engine.get_vram_usage()
    print(f"[GPU] Device={engine.device} | VRAM Allocated={vram_init.get('allocated_mb')} MB | Load Time={load_time_s}s\n")

    suite_16 = build_16q_benchmark()
    print(f"[SUITE] Loaded {len(suite_16)} uncurtailed questions across 5 cognitive pillars.\n")

    runs_results = []
    sound_evolution_log = []

    # Text for sound synthesis evolution test
    sound_test_text = "X-Star neuromorphic vocal organ dynamic frequency synthesis test."

    for run_idx in range(1, 4):
        print(f"\n{'#' * 82}")
        print(f"  STARTING RUN {run_idx} OF 3")
        print(f"{'#' * 82}")

        # Sound test at the start of each run to measure acoustic evolution
        wav_filename = f"hadl_v57_voice_run{run_idx}.wav"
        wav_path = benchmarks_dir / wav_filename
        voice_out = engine.synthesize_voice(
            text=sound_test_text,
            save_path=str(wav_path),
            duration_sec=1.5,
            enable_self_listening=True
        )
        wav_meta = inspect_wav_file(wav_path)
        synth_t = voice_out.get("synthesis_telemetry", {})
        listen_t = voice_out.get("self_listening_telemetry", {})
        sound_entry = {
            "run": run_idx,
            "wav_file": wav_meta,
            "f0_mean_hz": round(float(synth_t.get("f0_mean_hz", 0.0)), 2),
            "f0_min_hz": round(float(synth_t.get("f0_min_hz", 0.0)), 2),
            "f0_max_hz": round(float(synth_t.get("f0_max_hz", 0.0)), 2),
            "formants_hz": [round(float(x), 2) for x in synth_t.get("formants_hz", [])],
            "bandwidths_hz": [round(float(x), 2) for x in synth_t.get("bandwidths_hz", [])],
            "coherence_before": round(float(listen_t.get("coherence_before", 0.0)), 4),
            "coherence_after": round(float(listen_t.get("coherence_after", 0.0)), 4),
            "coherence_gain_percent": round(
                100.0 * (float(listen_t.get("coherence_after", 0.0)) - float(listen_t.get("coherence_before", 0.0)))
                / max(1e-4, float(listen_t.get("coherence_before", 1.0))), 2
            ),
            "voice_adaptation_steps": synth_t.get("voice_adaptation_steps", 0)
        }
        sound_evolution_log.append(sound_entry)
        print(
            f"[SOUND RUN {run_idx}] Saved {wav_filename} | f0={sound_entry['f0_mean_hz']}Hz | "
            f"F1..F4={sound_entry['formants_hz']} | "
            f"Coherence: {sound_entry['coherence_before']} -> {sound_entry['coherence_after']} "
            f"(+{sound_entry['coherence_gain_percent']}%)"
        )

        run_items = []
        run_passed_bench = 0
        run_passed_popperian = 0
        run_passed_overall = 0

        for q_idx, q in enumerate(suite_16, 1):
            formatted = engine.format_chat_prompt([{"role": "user", "content": q["prompt"]}])
            t_gen_start = time.perf_counter()
            resp, metrics = engine.generate(
                prompt=formatted,
                max_new_tokens=q["max_tokens"],
                temperature=0.0,
            )
            elapsed = time.perf_counter() - t_gen_start

            bench_ok = bool(q["check"](resp))
            ver_res = engine.verify_and_adapt(q["prompt"], resp)
            intrinsic_ok = bool(ver_res.get("verified", False))
            overall_ok = bench_ok and intrinsic_ok

            if bench_ok:
                run_passed_bench += 1
            if intrinsic_ok:
                run_passed_popperian += 1
            if overall_ok:
                run_passed_overall += 1

            status_str = "PASS" if overall_ok else "FAIL"
            print(
                f"  [{run_idx}:{q_idx:02d}] {q['id']:<24} | {status_str:<4} "
                f"(Bench={bench_ok}, Popper={intrinsic_ok}, Dopamine={metrics.dopamine_sign:+.1f}, "
                f"Passes={metrics.evolution_passes}, TPS={metrics.tokens_per_second:.1f}, "
                f"f0={metrics.f0_mean_hz:.1f}Hz, 2D->3D={metrics.consistency_2d_to_3d:.3f})"
            )

            run_items.append({
                "id": q["id"],
                "pillar": q["pillar"],
                "name": q["name"],
                "prompt": q["prompt"],
                "response": resp.strip(),
                "benchmark_passed": bench_ok,
                "popperian_verified": intrinsic_ok,
                "overall_passed": overall_ok,
                "dopamine_sign": metrics.dopamine_sign,
                "evolution_passes": metrics.evolution_passes,
                "tokens_per_second": metrics.tokens_per_second,
                "ttft_ms": metrics.ttft_ms,
                "f0_mean_hz": metrics.f0_mean_hz,
                "consistency_2d_to_3d": metrics.consistency_2d_to_3d,
                "elapsed_s": round(elapsed, 2)
            })

        brain_telem = engine.get_brain_telemetry()
        plastic_drift = brain_telem.get("spiking_telemetry", {}).get("plastic_drift", 0.0)

        run_summary = {
            "run_index": run_idx,
            "total_questions": len(suite_16),
            "benchmark_passed": run_passed_bench,
            "popperian_verified": run_passed_popperian,
            "overall_passed": run_passed_overall,
            "benchmark_accuracy_percent": round(100.0 * run_passed_bench / len(suite_16), 2),
            "popperian_accuracy_percent": round(100.0 * run_passed_popperian / len(suite_16), 2),
            "overall_accuracy_percent": round(100.0 * run_passed_overall / len(suite_16), 2),
            "plastic_drift": plastic_drift,
            "items": run_items
        }
        runs_results.append(run_summary)
        print(
            f"\n[RUN {run_idx} SUMMARY] Overall Passed: {run_passed_overall}/{len(suite_16)} "
            f"({run_summary['overall_accuracy_percent']}%) | "
            f"Bench: {run_passed_bench}/{len(suite_16)} ({run_summary['benchmark_accuracy_percent']}%) | "
            f"Popperian: {run_passed_popperian}/{len(suite_16)} ({run_summary['popperian_accuracy_percent']}%) | "
            f"Plastic Drift ||ΔW||_F = {plastic_drift:.5f}"
        )

    # Video Imagination Test: Generate 16-frame 4D Spatio-Temporal Mental Video
    print(f"\n{'=' * 82}")
    print("  EXECUTING VIDEO SPATIO-TEMPORAL IMAGINATION TEST (SO(3) Geodesic Animation)")
    print(f"{'=' * 82}")
    video_prompt = "A 3D rotating crystalline hypercube in deep space with volume-preserving isometric mental motion."
    vid_res = engine.generate_mental_video(
        text=video_prompt,
        n_frames=16,
        fps=8,
        save_gif_path=str(video_gif_path)
    )
    print(
        f"[VIDEO] Generated {vid_res['n_frames']} frames at {vid_res['fps']} FPS | "
        f"Saved GIF: {video_gif_path.name} ({vid_res['gif_saved']}) | "
        f"Mean SO(3) Orthogonality Error: {vid_res['mean_orthogonality_error']:.2e} | "
        f"Mean Frame Delta (Motion Smoothness): {vid_res['mean_frame_delta']:.5f} | "
        f"Volume Conservation det(R_t) for all t: {all(abs(d - 1.0) < 1e-3 for d in vid_res['so3_dets'])}"
    )

    vram_final = engine.get_vram_usage()

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": engine.device,
        "dtype": str(engine.torch_dtype),
        "load_time_s": load_time_s,
        "vram_usage_initial": vram_init,
        "vram_usage_final": vram_final,
        "benchmark_summary": {
            "total_questions": len(suite_16),
            "runs": [
                {
                    "run": r["run_index"],
                    "overall_accuracy_percent": r["overall_accuracy_percent"],
                    "benchmark_accuracy_percent": r["benchmark_accuracy_percent"],
                    "popperian_accuracy_percent": r["popperian_accuracy_percent"],
                    "passed": r["overall_passed"],
                    "total": r["total_questions"],
                    "plastic_drift": r["plastic_drift"],
                }
                for r in runs_results
            ],
            "accuracy_progression": [r["overall_accuracy_percent"] for r in runs_results],
            "plastic_drift_progression": [r["plastic_drift"] for r in runs_results],
        },
        "sound_evolution_test": {
            "test_text": sound_test_text,
            "sound_evolution_log": sound_evolution_log,
        },
        "video_imagination_test": {
            "prompt": video_prompt,
            "gif_path": str(video_gif_path),
            "gif_saved": vid_res.get("gif_saved", False),
            "n_frames": vid_res.get("n_frames", 16),
            "fps": vid_res.get("fps", 8),
            "mean_frame_delta": vid_res.get("mean_frame_delta", 0.0),
            "frame_deltas": vid_res.get("frame_deltas", []),
            "so3_dets": vid_res.get("so3_dets", []),
            "mean_orthogonality_error": vid_res.get("mean_orthogonality_error", 0.0),
            "ascii_first_frame": vid_res.get("ascii_first_frame", ""),
            "ascii_mid_frame": vid_res.get("ascii_mid_frame", ""),
            "ascii_last_frame": vid_res.get("ascii_last_frame", ""),
        },
        "runs_detailed": runs_results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 82)
    print("  3-RUN EMPIRICAL MULTIMODAL BENCHMARK COMPLETE")
    print(f"  Progression: {' -> '.join(str(r['overall_accuracy_percent']) + '%' for r in runs_results)}")
    print(f"  Full JSON Report saved to: {report_path}")
    print("=" * 82)


if __name__ == "__main__":
    main()
