"""
Rigorous Empirical A/B Benchmark: 3D Spatial Grounding vs 1D Language Baseline
==============================================================================
Empirical evaluation on NVIDIA GeForce RTX 5060 Laptop GPU:
- Mode A (Baseline): Pure 1D language model representation without 3D spatial
  imagination and without auditory re-entry (enable_3d_imagination = False).
- Mode B (Active Multimodal Grounding): Neuromorphic brain with progressive
  2D-to-3D visual imagination, SO(3) Lie-group mental rotation delta injection,
  and endogenous vocal re-entry (enable_3d_imagination = True).

18 challenging questions across 5 spatial reasoning domains:
1. 3D Mental Rotation & Cube Faces (Shepard-Metzler style)
2. Compound Compass & 2D/3D Navigation
3. Observer Perspective-Taking & Point of View
4. 3D Spatial Topology & Geometric Containment
5. Coordinate Inversion & Transformation

Outputs:
- JSON detailed results: benchmarks/spatial_grounding_ab_benchmark_results.json
- Multi-panel comparison graph: benchmarks/hadl_v57_spatial_grounding_ab_graph.png
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Callable

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xstar.inference.engine import CustomInferenceEngine, EngineConfig


def get_spatial_benchmark_suite() -> List[Dict[str, Any]]:
    return [
        # Domain 1: 3D Mental Rotation & Cube Coloring
        {
            "id": "cube_rot_clockwise_z",
            "domain": "3D Mental Rotation",
            "name": "Cube 90-deg CW Yaw Rotation",
            "prompt": "A cube has Top: Red, Bottom: Blue, Front: Green, Back: Yellow, Left: Orange, Right: White. If you rotate the cube 90 degrees clockwise looking from above the Top face, what color is now on the Front face? Answer with only the color name:\nAnswer:",
            "ground_truth": "Orange",
            "check": lambda r: "orange" in r.lower().split()[:6] or r.strip().lower().startswith("orange"),
        },
        {
            "id": "cube_pitch_forward_x",
            "domain": "3D Mental Rotation",
            "name": "Cube 90-deg Forward Pitch",
            "prompt": "A cube has Top: Red, Bottom: Blue, Front: Green, Back: Yellow, Left: Orange, Right: White. If you pitch the cube 90 degrees forward (rolling the top towards the front), what color is now on the Top face? Answer with only the color name:\nAnswer:",
            "ground_truth": "Yellow",
            "check": lambda r: "yellow" in r.lower().split()[:6] or r.strip().lower().startswith("yellow"),
        },
        {
            "id": "cube_rot_180_z",
            "domain": "3D Mental Rotation",
            "name": "Cube 180-deg Inversion",
            "prompt": "A cube has Top: White, Bottom: Yellow, Front: Red, Back: Orange, Left: Green, Right: Blue. If you rotate the cube 180 degrees around the vertical axis, what color is now on the Front face? Answer with only the color name:\nAnswer:",
            "ground_truth": "Orange",
            "check": lambda r: "orange" in r.lower().split()[:6] or r.strip().lower().startswith("orange"),
        },
        {
            "id": "dice_opposite_faces",
            "domain": "3D Mental Rotation",
            "name": "Opposite Die Face Topology",
            "prompt": "A standard 6-sided die has opposite sides summing to 7 (1 opposite 6, 2 opposite 5, 3 opposite 4). If the die is resting on a table with 1 on the Top face, what number is on the Bottom face? Provide only the single digit:\nAnswer:",
            "ground_truth": "6",
            "check": lambda r: bool(re.search(r"\b6\b", r[:30])),
        },

        # Domain 2: Compound Compass & Navigation
        {
            "id": "compass_compound_turns",
            "domain": "Compass & Navigation",
            "name": "Compound Heading (CW 90 -> 180 -> CCW 90)",
            "prompt": "You are facing North. You turn 90 degrees clockwise, then 180 degrees, and finally 90 degrees counter-clockwise. Which cardinal direction are you now facing (North, South, East, or West)? Provide only the direction:\nAnswer:",
            "ground_truth": "South",
            "check": lambda r: "south" in r.lower().split()[:6] or r.strip().lower().startswith("south"),
        },
        {
            "id": "drone_3d_coordinates",
            "domain": "Compass & Navigation",
            "name": "Drone 3D Displacement (10E, 5N, 20Up, 10W)",
            "prompt": "A drone starts at position (0, 0, 0). It flies 10 meters East (+x), then 5 meters North (+y), then 20 meters Up (+z), then 10 meters West (-x). What are the final (x, y, z) coordinates of the drone? Answer in format (x, y, z):\nAnswer:",
            "ground_truth": "(0, 5, 20)",
            "check": lambda r: "0" in r and "5" in r and "20" in r and bool(re.search(r"\(?\s*0\s*,\s*5\s*,\s*20\s*\)?", r[:40])),
        },
        {
            "id": "walking_left_turns",
            "domain": "Compass & Navigation",
            "name": "Consecutive Left Turns from East",
            "prompt": "You start facing East. You walk 10 meters forward, turn 90 degrees to your left, walk 10 meters, and turn 90 degrees to your left again. In which direction are you now facing (North, South, East, or West)? Provide only the direction:\nAnswer:",
            "ground_truth": "West",
            "check": lambda r: "west" in r.lower().split()[:6] or r.strip().lower().startswith("west"),
        },
        {
            "id": "pythagoras_displacement",
            "domain": "Compass & Navigation",
            "name": "Euclidean 2D Vector Magnitude (4N, 3E)",
            "prompt": "A ship sails North for 4 miles, then turns 90 degrees right and sails East for 3 miles. What is the straight-line distance from the starting point to the ship in miles? Provide only the number:\nAnswer:",
            "ground_truth": "5",
            "check": lambda r: bool(re.search(r"\b5(\.0)?\b", r[:30])),
        },

        # Domain 3: Observer Perspective-Taking
        {
            "id": "table_opposite_perspective",
            "domain": "Perspective-Taking",
            "name": "Opposite Dining Partner Perspective",
            "prompt": "Alice and Bob sit across from each other at a dining table. Alice places a coffee cup to her own left side. From Bob's perspective looking directly at Alice, is the coffee cup on Bob's left or Bob's right side? Answer with either Left or Right:\nAnswer:",
            "ground_truth": "Right",
            "check": lambda r: "right" in r.lower().split()[:6] or r.strip().lower().startswith("right"),
        },
        {
            "id": "three_objects_viewpoint",
            "domain": "Perspective-Taking",
            "name": "Three Objects Reversed Point of View",
            "prompt": "Three objects are placed in a horizontal row on a table: Red on the left, Green in the middle, and Blue on the right from your point of view. A friend stands on the opposite side of the table facing you. Which object is on the far left from your friend's point of view? Provide only the color name:\nAnswer:",
            "ground_truth": "Blue",
            "check": lambda r: "blue" in r.lower().split()[:6] or r.strip().lower().startswith("blue"),
        },
        {
            "id": "mirror_clock_reflection",
            "domain": "Perspective-Taking",
            "name": "Clock 3:00 Planar Mirror Reflection",
            "prompt": "You look at an analog clock showing 3:00 (the minute hand points straight up at 12, and the hour hand points straight right at 3). In a flat mirror, what time does the clock face appear to show? Provide only the time:\nAnswer:",
            "ground_truth": "9:00",
            "check": lambda r: "9:00" in r or bool(re.search(r"\b9\b", r[:25])),
        },
        {
            "id": "sun_setting_relative_heading",
            "domain": "Perspective-Taking",
            "name": "Sunset Relative to Southward Facing",
            "prompt": "You are standing on a hill facing South. The sun is setting directly to your right side. In which compass direction (North, South, East, or West) is the sun setting? Provide only the direction:\nAnswer:",
            "ground_truth": "West",
            "check": lambda r: "west" in r.lower().split()[:6] or r.strip().lower().startswith("west"),
        },

        # Domain 4: 3D Spatial Topology & Geometry
        {
            "id": "box_nesting_dimensions",
            "domain": "Spatial Topology",
            "name": "Orthogonal Bounding Box Containment",
            "prompt": "Box A has external dimensions 10x10x10 cm. Box B has external dimensions 8x8x8 cm. Can Box B fit completely inside Box A? Answer with Yes or No:\nAnswer:",
            "ground_truth": "Yes",
            "check": lambda r: "yes" in r.lower().split()[:5] or r.strip().lower().startswith("yes"),
        },
        {
            "id": "sphere_in_cube_tangency",
            "domain": "Spatial Topology",
            "name": "Inscribed Sphere-Cube Tangency",
            "prompt": "A solid sphere of diameter 6 cm is placed inside a cubical box with internal edge length 6 cm. Does the sphere touch all 6 internal faces of the cube? Answer with Yes or No:\nAnswer:",
            "ground_truth": "Yes",
            "check": lambda r: "yes" in r.lower().split()[:5] or r.strip().lower().startswith("yes"),
        },
        {
            "id": "shadow_noon_hemisphere",
            "domain": "Spatial Topology",
            "name": "Solar Noon Shadow Direction Hemisphere",
            "prompt": "At noon on an autumn day, a vertical flagpole casts a shadow pointing directly North. In which hemisphere (Northern or Southern) is the flagpole located? Provide only the word Northern or Southern:\nAnswer:",
            "ground_truth": "Northern",
            "check": lambda r: "northern" in r.lower().split()[:6] or r.strip().lower().startswith("northern"),
        },

        # Domain 5: Coordinate Inversion & Transformation
        {
            "id": "reflection_xy_plane",
            "domain": "Coordinate Transforms",
            "name": "Point 3D Reflection across XY-Plane",
            "prompt": "Point P in 3D Cartesian coordinates is (3, -4, 5). If point P is reflected across the xy-plane (z = 0), what are its new coordinates (x, y, z)? Answer in format (x, y, z):\nAnswer:",
            "ground_truth": "(3, -4, -5)",
            "check": lambda r: "3" in r and "-4" in r and "-5" in r and bool(re.search(r"\(?\s*3\s*,\s*-4\s*,\s*-5\s*\)?", r[:40])),
        },
        {
            "id": "vector_rotation_2d_90ccw",
            "domain": "Coordinate Transforms",
            "name": "2D Vector (0, 1) 90-deg CCW Rotation",
            "prompt": "A 2D vector starts at (0, 0) and points to (0, 1) along the positive y-axis. If the vector is rotated 90 degrees counter-clockwise around the origin, what is its new endpoint? Answer in format (x, y):\nAnswer:",
            "ground_truth": "(-1, 0)",
            "check": lambda r: "-1" in r and "0" in r and bool(re.search(r"\(?\s*-1\s*,\s*0\s*\)?", r[:30])),
        },
        {
            "id": "letter_planar_rotation_180",
            "domain": "Coordinate Transforms",
            "name": "Glyph 'b' Planar 180-deg Inversion",
            "prompt": "If you take a card with the lowercase letter 'b' and rotate it 180 degrees in the plane (upside down), which lowercase letter does it resemble? Provide only the single letter:\nAnswer:",
            "ground_truth": "q",
            "check": lambda r: bool(re.search(r"\b[qQ]\b", r[:25])),
        },
    ]


def run_benchmark():
    print("=" * 80)
    print("RIGOROUS EMPIRICAL A/B BENCHMARK: 3D SPATIAL GROUNDING VS 1D LANGUAGE BASELINE")
    print("Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR6)")
    print("Zero Checkpoint / Cold Start Evaluation Across 18 Rigorous Spatial Questions")
    print("=" * 80)

    cfg = EngineConfig(
        model_path="dist/staging_hf_fused",
        adapter_path="dist/frontier_lora_adapter",
        enable_brain=True,
        enable_adapter=True,
        enable_online_learning=False,
        enable_voice=False,
    )

    print("\n[*] Initializing CustomInferenceEngine...")
    engine = CustomInferenceEngine(cfg)
    engine.load()

    suite = get_spatial_benchmark_suite()
    print(f"[*] Benchmark suite loaded: {len(suite)} questions across 5 spatial domains.")

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 5060 Laptop GPU (8GB)",
        "mode_a_baseline": {
            "description": "Standard 1D Language Model without 3D Spatial Imagination (enable_3d_imagination=False, enable_voice=False)",
            "questions": [],
            "domain_scores": {},
            "total_correct": 0,
            "accuracy_pct": 0.0,
            "mean_ttft_ms": 0.0,
            "mean_tps": 0.0,
            "total_time_s": 0.0,
        },
        "mode_b_grounded": {
            "description": "Neuromorphic Brain with Progressive 2D-to-3D Visual Imagination + SO(3) Lie-group Grounding + Auditory Re-entry (enable_3d_imagination=True, enable_voice=True)",
            "questions": [],
            "domain_scores": {},
            "total_correct": 0,
            "accuracy_pct": 0.0,
            "mean_ttft_ms": 0.0,
            "mean_tps": 0.0,
            "mean_consistency_2d_3d": 0.0,
            "total_time_s": 0.0,
        },
        "comparison_summary": {},
    }

    # =========================================================================
    # PHASE 1: MODE A EVALUATION (BASELINE)
    # =========================================================================
    print("\n" + "=" * 80)
    print(">>> EXECUTING MODE A (1D LANGUAGE BASELINE: 3D Imagination = OFF) <<<")
    print("=" * 80)

    engine.model.brain.enable_3d_imagination = False
    engine.model.brain.enable_endogenous_voice = False
    engine.config.enable_voice = False

    t_mode_a_start = time.perf_counter()
    mode_a_correct = 0
    mode_a_domain_correct: Dict[str, int] = {}
    mode_a_domain_total: Dict[str, int] = {}
    ttft_list_a = []
    tps_list_a = []

    for idx, q in enumerate(suite, 1):
        d = q["domain"]
        mode_a_domain_total[d] = mode_a_domain_total.get(d, 0) + 1
        print(f"\n[{idx}/{len(suite)}] [{d}] {q['name']}...")
        
        t0 = time.perf_counter()
        resp, m = engine.generate(
            prompt=q["prompt"],
            max_new_tokens=30,
            temperature=0.1,  # Focused deterministic evaluation
        )
        t_elap = time.perf_counter() - t0
        
        is_correct = bool(q["check"](resp))
        if is_correct:
            mode_a_correct += 1
            mode_a_domain_correct[d] = mode_a_domain_correct.get(d, 0) + 1

        ttft_list_a.append(m.ttft_ms)
        tps_list_a.append(m.tokens_per_second)

        clean_resp = resp.strip().replace("\n", " ")[:60]
        status = "CORRECT [PASS]" if is_correct else "INCORRECT [FAIL]"
        print(f"    Ans: '{clean_resp}' | GroundTruth: '{q['ground_truth']}' -> {status} ({t_elap:.2f}s)")

        results["mode_a_baseline"]["questions"].append({
            "id": q["id"],
            "domain": q["domain"],
            "name": q["name"],
            "ground_truth": q["ground_truth"],
            "model_response": resp.strip(),
            "is_correct": is_correct,
            "latency_s": round(t_elap, 2),
            "ttft_ms": round(m.ttft_ms, 1),
            "tps": round(m.tokens_per_second, 2),
            "cognitive_state": m.cognitive_state,
            "consistency_2d_to_3d": round(m.consistency_2d_to_3d, 4),
        })

    t_mode_a_total = time.perf_counter() - t_mode_a_start
    results["mode_a_baseline"]["total_correct"] = mode_a_correct
    results["mode_a_baseline"]["accuracy_pct"] = round(100.0 * mode_a_correct / len(suite), 2)
    results["mode_a_baseline"]["mean_ttft_ms"] = round(float(np.mean(ttft_list_a)), 1)
    results["mode_a_baseline"]["mean_tps"] = round(float(np.mean(tps_list_a)), 2)
    results["mode_a_baseline"]["total_time_s"] = round(t_mode_a_total, 2)
    results["mode_a_baseline"]["domain_scores"] = {
        d: {
            "correct": mode_a_domain_correct.get(d, 0),
            "total": mode_a_domain_total[d],
            "accuracy_pct": round(100.0 * mode_a_domain_correct.get(d, 0) / mode_a_domain_total[d], 1),
        }
        for d in mode_a_domain_total
    }

    # =========================================================================
    # PHASE 2: MODE B EVALUATION (ACTIVE 3D SPATIAL GROUNDING)
    # =========================================================================
    print("\n" + "=" * 80)
    print(">>> EXECUTING MODE B (ACTIVE 3D SPATIAL GROUNDING: 3D Imagination = ON) <<<")
    print("=" * 80)

    engine.model.brain.enable_3d_imagination = True
    engine.model.brain.enable_endogenous_voice = True
    engine.config.enable_voice = True

    t_mode_b_start = time.perf_counter()
    mode_b_correct = 0
    mode_b_domain_correct: Dict[str, int] = {}
    mode_b_domain_total: Dict[str, int] = {}
    ttft_list_b = []
    tps_list_b = []
    consistency_list_b = []

    for idx, q in enumerate(suite, 1):
        d = q["domain"]
        mode_b_domain_total[d] = mode_b_domain_total.get(d, 0) + 1
        print(f"\n[{idx}/{len(suite)}] [{d}] {q['name']}...")
        
        t0 = time.perf_counter()
        resp, m = engine.generate(
            prompt=q["prompt"],
            max_new_tokens=30,
            temperature=0.1,  # Focused deterministic evaluation
        )
        t_elap = time.perf_counter() - t0
        
        is_correct = bool(q["check"](resp))
        if is_correct:
            mode_b_correct += 1
            mode_b_domain_correct[d] = mode_b_domain_correct.get(d, 0) + 1

        ttft_list_b.append(m.ttft_ms)
        tps_list_b.append(m.tokens_per_second)
        consistency_list_b.append(m.consistency_2d_to_3d)

        clean_resp = resp.strip().replace("\n", " ")[:60]
        status = "CORRECT [PASS]" if is_correct else "INCORRECT [FAIL]"
        print(f"    Ans: '{clean_resp}' | GroundTruth: '{q['ground_truth']}' -> {status} (c={m.consistency_2d_to_3d:.4f}, {t_elap:.2f}s)")

        results["mode_b_grounded"]["questions"].append({
            "id": q["id"],
            "domain": q["domain"],
            "name": q["name"],
            "ground_truth": q["ground_truth"],
            "model_response": resp.strip(),
            "is_correct": is_correct,
            "latency_s": round(t_elap, 2),
            "ttft_ms": round(m.ttft_ms, 1),
            "tps": round(m.tokens_per_second, 2),
            "cognitive_state": m.cognitive_state,
            "consistency_2d_to_3d": round(m.consistency_2d_to_3d, 4),
            "f0_mean_hz": round(m.f0_mean_hz, 1),
        })

    t_mode_b_total = time.perf_counter() - t_mode_b_start
    results["mode_b_grounded"]["total_correct"] = mode_b_correct
    results["mode_b_grounded"]["accuracy_pct"] = round(100.0 * mode_b_correct / len(suite), 2)
    results["mode_b_grounded"]["mean_ttft_ms"] = round(float(np.mean(ttft_list_b)), 1)
    results["mode_b_grounded"]["mean_tps"] = round(float(np.mean(tps_list_b)), 2)
    results["mode_b_grounded"]["mean_consistency_2d_3d"] = round(float(np.mean(consistency_list_b)), 4)
    results["mode_b_grounded"]["total_time_s"] = round(t_mode_b_total, 2)
    results["mode_b_grounded"]["domain_scores"] = {
        d: {
            "correct": mode_b_domain_correct.get(d, 0),
            "total": mode_b_domain_total[d],
            "accuracy_pct": round(100.0 * mode_b_domain_correct.get(d, 0) / mode_b_domain_total[d], 1),
        }
        for d in mode_b_domain_total
    }

    # Comparison summary
    acc_delta = results["mode_b_grounded"]["accuracy_pct"] - results["mode_a_baseline"]["accuracy_pct"]
    results["comparison_summary"] = {
        "mode_a_accuracy": results["mode_a_baseline"]["accuracy_pct"],
        "mode_b_accuracy": results["mode_b_grounded"]["accuracy_pct"],
        "accuracy_delta_percentage_points": round(acc_delta, 2),
        "relative_improvement_pct": round((acc_delta / max(0.01, results['mode_a_baseline']['accuracy_pct'])) * 100.0, 1),
        "mean_latency_overhead_pct": round(((t_mode_b_total - t_mode_a_total) / max(0.01, t_mode_a_total)) * 100.0, 1),
        "mode_a_time_s": round(t_mode_a_total, 1),
        "mode_b_time_s": round(t_mode_b_total, 1),
    }

    # Save JSON results
    out_json = "benchmarks/spatial_grounding_ab_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Detailed benchmark metrics saved to: {out_json}")

    # =========================================================================
    # PHASE 3: MULTI-PANEL GRAPH GENERATION
    # =========================================================================
    generate_comparison_chart(results, suite)

    print("\n" + "=" * 80)
    print("EMPIRICAL BENCHMARK COMPLETE!")
    print(f"Mode A (Baseline): {mode_a_correct}/{len(suite)} ({results['mode_a_baseline']['accuracy_pct']}%)")
    print(f"Mode B (3D Grounded): {mode_b_correct}/{len(suite)} ({results['mode_b_grounded']['accuracy_pct']}%)")
    print(f"Net Accuracy Delta: {acc_delta:+.2f}%")
    print("=" * 80)


def generate_comparison_chart(results: Dict[str, Any], suite: List[Dict[str, Any]]):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#475569")

    # Panel 1: Domain-wise Accuracy Comparison
    domains = list(results["mode_a_baseline"]["domain_scores"].keys())
    scores_a = [results["mode_a_baseline"]["domain_scores"][d]["accuracy_pct"] for d in domains]
    scores_b = [results["mode_b_grounded"]["domain_scores"][d]["accuracy_pct"] for d in domains]

    x = np.arange(len(domains))
    width = 0.35

    ax1 = axes[0, 0]
    rects1 = ax1.bar(x - width/2, scores_a, width, label="Mode A: 1D Baseline", color="#3b82f6", alpha=0.9, edgecolor="#60a5fa")
    rects2 = ax1.bar(x + width/2, scores_b, width, label="Mode B: 3D Grounded", color="#10b981", alpha=0.9, edgecolor="#34d399")

    ax1.set_ylabel("Accuracy (%)", color="#f1f5f9", fontsize=11, fontweight="bold")
    ax1.set_title("Domain Accuracy: 1D Baseline vs 3D Grounded", color="#f8fafc", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.replace(" & ", "\n& ") for d in domains], color="#cbd5e1", fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.legend(facecolor="#334155", edgecolor="#475569", labelcolor="#f8fafc", loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.3, color="#94a3b8")

    for r in rects1:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 1.5, f"{h:.0f}%", ha="center", va="bottom", color="#93c5fd", fontsize=8)
    for r in rects2:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 1.5, f"{h:.0f}%", ha="center", va="bottom", color="#6ee7b7", fontsize=8)

    # Panel 2: Overall Accuracy & Total Solved Count
    ax2 = axes[0, 1]
    modes = ["Mode A\n(Baseline)", "Mode B\n(3D Grounded)"]
    tot_acc = [results["mode_a_baseline"]["accuracy_pct"], results["mode_b_grounded"]["accuracy_pct"]]
    colors = ["#3b82f6", "#10b981"]
    bars = ax2.bar(modes, tot_acc, color=colors, width=0.45, edgecolor="#f8fafc", linewidth=1.2)
    ax2.set_ylabel("Overall Accuracy (%)", color="#f1f5f9", fontsize=11, fontweight="bold")
    ax2.set_title(f"Overall Accuracy Across 18 Spatial Questions\n(Delta: {results['comparison_summary']['accuracy_delta_percentage_points']:+.1f}%)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#94a3b8")

    for bar, val in zip(bars, tot_acc):
        count = int(round(val * len(suite) / 100.0))
        ax2.text(bar.get_x() + bar.get_width()/2., val + 2.0, f"{val:.1f}%\n({count}/{len(suite)} Correct)", ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")

    # Panel 3: 2D-to-3D Reprojection Consistency Distribution (Mode B)
    ax3 = axes[1, 0]
    c_vals = [q["consistency_2d_to_3d"] for q in results["mode_b_grounded"]["questions"]]
    q_names = [f"Q{i+1}" for i in range(len(c_vals))]
    q_colors = ["#10b981" if q["is_correct"] else "#ef4444" for q in results["mode_b_grounded"]["questions"]]
    
    ax3.bar(q_names, c_vals, color=q_colors, alpha=0.85, edgecolor="#cbd5e1", linewidth=0.5)
    ax3.axhline(float(np.mean(c_vals)), color="#fbbf24", linestyle="--", linewidth=1.5, label=f"Mean c = {np.mean(c_vals):.3f}")
    ax3.set_ylabel("Reprojection Consistency c_{2D->3D}", color="#f1f5f9", fontsize=11, fontweight="bold")
    ax3.set_title("Mode B 3D Spatial Consistency per Question\n(Green = Correct, Red = Incorrect)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 1.05)
    ax3.tick_params(axis="x", rotation=45)
    ax3.legend(facecolor="#334155", edgecolor="#475569", labelcolor="#f8fafc", loc="lower right")
    ax3.grid(axis="y", linestyle="--", alpha=0.3, color="#94a3b8")

    # Panel 4: Latency & Compute Overhead Comparison
    ax4 = axes[1, 1]
    metrics = ["TTFT (ms)", "Total Gen (s)", "Throughput (TPS)"]
    mode_a_lat = [results["mode_a_baseline"]["mean_ttft_ms"], results["mode_a_baseline"]["total_time_s"] / len(suite), results["mode_a_baseline"]["mean_tps"]]
    mode_b_lat = [results["mode_b_grounded"]["mean_ttft_ms"], results["mode_b_grounded"]["total_time_s"] / len(suite), results["mode_b_grounded"]["mean_tps"]]

    # Normalize metrics to Mode A = 100% for relative profiling
    norm_a = [100.0, 100.0, 100.0]
    norm_b = [
        (mode_b_lat[0] / max(1.0, mode_a_lat[0])) * 100.0,
        (mode_b_lat[1] / max(0.01, mode_a_lat[1])) * 100.0,
        (mode_b_lat[2] / max(0.01, mode_a_lat[2])) * 100.0,
    ]

    x4 = np.arange(len(metrics))
    rects4_a = ax4.bar(x4 - width/2, norm_a, width, label=f"Mode A (100% Base)", color="#3b82f6", alpha=0.9, edgecolor="#60a5fa")
    rects4_b = ax4.bar(x4 + width/2, norm_b, width, label=f"Mode B Overhead", color="#f59e0b", alpha=0.9, edgecolor="#fbbf24")

    ax4.set_ylabel("Relative Scale (% of Mode A)", color="#f1f5f9", fontsize=11, fontweight="bold")
    ax4.set_title("Compute & Latency Profile (Mode A vs Mode B)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(metrics, color="#cbd5e1", fontsize=10)
    ax4.set_ylim(0, max(max(norm_b), 110) * 1.25)
    ax4.legend(facecolor="#334155", edgecolor="#475569", labelcolor="#f8fafc", loc="upper right")
    ax4.grid(axis="y", linestyle="--", alpha=0.3, color="#94a3b8")

    # Add raw value annotations
    for idx, (ra, rb) in enumerate(zip(rects4_a, rects4_b)):
        ax4.text(ra.get_x() + ra.get_width()/2., ra.get_height() + 2, f"{mode_a_lat[idx]:.1f}", ha="center", va="bottom", color="#93c5fd", fontsize=8)
        ax4.text(rb.get_x() + rb.get_width()/2., rb.get_height() + 2, f"{mode_b_lat[idx]:.1f}", ha="center", va="bottom", color="#fde68a", fontsize=8)

    plt.suptitle("X-Star HADL v5.7 Empirical Evaluation: Active 3D Spatial Grounding vs 1D Language Baseline", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_png = "benchmarks/hadl_v57_spatial_grounding_ab_graph.png"
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[*] Visual comparison chart saved to: {out_png}")


if __name__ == "__main__":
    run_benchmark()
