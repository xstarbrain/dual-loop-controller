#!/usr/bin/env python3
"""
40-Question Novel Challenge Benchmark Suite for HADL v5.9
(High-Level Reasoning & Out-of-Distribution Generalization Stress Test)
======================================================================
Evaluates Base Qwen-2B vs. HADL v5.9 Spiking Brain across 5 Novel Challenge Domains:
1. Mathematical & Symbolic Olympiad (8Q)
2. Rigorous Science, Quantum & Biology (8Q)
3. Software Engineering, Algorithms & Exceptions (8Q)
4. Formal Logic, Epistemic Refutation & Coreference (8Q)
5. Multi-Constraint Instruction Compliance & Metacognition (8Q)

Outputs:
- JSON Audit: eval_results/benchmark_40_novel_challenges_audit.json
- Publication Chart: eval_results/benchmark_40_novel_challenges_comparison.png
"""

from __future__ import annotations

import os
import sys
import json
import time
import re
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
ADAPTER_DIR = PROJECT_ROOT / "dist" / "frontier_lora_adapter"
ARTIFACT_DIR = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")

JSON_OUT = PROJECT_ROOT / "eval_results" / "benchmark_40_novel_challenges_audit.json"
GRAPH_OUT = PROJECT_ROOT / "eval_results" / "benchmark_40_novel_challenges_comparison.png"
PAPER_GRAPH = PROJECT_ROOT / "paper" / "figures" / "benchmark_40_novel_challenges_comparison.png"


def build_40_novel_challenges() -> List[Dict[str, Any]]:
    """Builds the 40 brand-new challenge questions across 5 distinct domains."""
    suite = [
        # =========================================================================
        # DOMAIN 1: MATHEMATICAL & SYMBOLIC OLYMPIAD (8 Questions)
        # =========================================================================
        {
            "id": "novel_01",
            "domain": "Math & Quantitative Olympiad",
            "title": "Modular Arithmetic Fermat Step",
            "prompt": "What is the remainder when 3^2026 is divided by 7? Provide only the single integer remainder.\nAnswer:",
            "target": "4",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b4\b", out.strip()))
        },
        {
            "id": "novel_02",
            "domain": "Math & Quantitative Olympiad",
            "title": "Matrix Eigenvalues & Trace",
            "prompt": "A 3x3 square matrix has eigenvalues 2, -3, and 5. What is the trace of this matrix? Provide only the integer value.\nAnswer:",
            "target": "4",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b4\b", out.strip()))
        },
        {
            "id": "novel_03",
            "domain": "Math & Quantitative Olympiad",
            "title": "Combinatorics Handshake Lemma",
            "prompt": "In a room of 12 people, each person shakes hands with every other person exactly once. How many total handshakes occur? Provide only the integer.\nAnswer:",
            "target": "66",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b66\b", out.strip()))
        },
        {
            "id": "novel_04",
            "domain": "Math & Quantitative Olympiad",
            "title": "Vieta's Formula Root Product",
            "prompt": "For the quadratic equation 2x^2 - 14x + 24 = 0, what is the product of its roots? Provide only the number.\nAnswer:",
            "target": "12",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b12\b", out.strip()))
        },
        {
            "id": "novel_05",
            "domain": "Math & Quantitative Olympiad",
            "title": "Lucas Sequence Step L_6",
            "prompt": "Given the sequence L_0 = 2, L_1 = 1, and L_n = L_{n-1} + L_{n-2} for n >= 2. What is the exact value of L_6? Provide only the number.\nAnswer:",
            "target": "18",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b18\b", out.strip()))
        },
        {
            "id": "novel_06",
            "domain": "Math & Quantitative Olympiad",
            "title": "Number Theory Distinct Primes",
            "prompt": "How many distinct prime factors does the number 840 have? Provide only the single integer.\nAnswer:",
            "target": "4",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b4\b", out.strip()))
        },
        {
            "id": "novel_07",
            "domain": "Math & Quantitative Olympiad",
            "title": "Two-Dice Sum Probability",
            "prompt": "A fair six-sided die is rolled twice. What is the probability that the sum of the numbers rolled is exactly 7? Express as a fraction in simplest form (e.g. 1/6).\nAnswer:",
            "target": "1/6",
            "max_tokens": 32,
            "check": lambda out: "1/6" in out or "0.166" in out or "0.167" in out
        },
        {
            "id": "novel_08",
            "domain": "Math & Quantitative Olympiad",
            "title": "Linear System Solution",
            "prompt": "If 3x + 2y = 19 and 2x - y = 1, what is the value of x? Provide only the number.\nAnswer:",
            "target": "3",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b3\b", out.strip()))
        },

        # =========================================================================
        # DOMAIN 2: RIGOROUS SCIENCE, QUANTUM & BIOLOGY (8 Questions)
        # =========================================================================
        {
            "id": "novel_09",
            "domain": "Science, Quantum & Biology",
            "title": "Pauli Exclusion Principle",
            "prompt": "Which fundamental physical principle dictates that no two identical fermions can occupy the same quantum state simultaneously? Answer with only the principle name.\nAnswer:",
            "target": "Pauli exclusion principle",
            "max_tokens": 32,
            "check": lambda out: "pauli" in out.lower()
        },
        {
            "id": "novel_10",
            "domain": "Science, Quantum & Biology",
            "title": "Carnot Cycle Efficiency",
            "prompt": "A Carnot heat engine operates between a hot reservoir at 600 K and a cold reservoir at 300 K. What is its maximum theoretical thermodynamic efficiency as a percentage? Provide only the percentage number (e.g. 50%).\nAnswer:",
            "target": "50%",
            "max_tokens": 32,
            "check": lambda out: "50" in out
        },
        {
            "id": "novel_11",
            "domain": "Science, Quantum & Biology",
            "title": "Universal mRNA Start Codon",
            "prompt": "In standard molecular genetics, what is the universal mRNA start codon sequence? Provide only the 3-letter sequence.\nAnswer:",
            "target": "AUG",
            "max_tokens": 32,
            "check": lambda out: "aug" in out.lower()
        },
        {
            "id": "novel_12",
            "domain": "Science, Quantum & Biology",
            "title": "Schwarzschild Radius Scaling",
            "prompt": "If the mass of an object is doubled, by what factor does its Schwarzschild radius increase? Provide only the integer factor.\nAnswer:",
            "target": "2",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b2\b", out.strip()))
        },
        {
            "id": "novel_13",
            "domain": "Science, Quantum & Biology",
            "title": "Depolarization Influx Ion",
            "prompt": "During the rapid depolarization phase of a neuronal action potential, which ion primarily rushes into the cell through voltage-gated channels? Answer with the chemical name or symbol.\nAnswer:",
            "target": "Sodium",
            "max_tokens": 32,
            "check": lambda out: "sodium" in out.lower() or "na" in out.lower()
        },
        {
            "id": "novel_14",
            "domain": "Science, Quantum & Biology",
            "title": "Speed of Light Invariance",
            "prompt": "According to Einstein's Special Relativity, if a spacecraft traveling at 0.8c turns on a forward searchlight, at what speed does an outside stationary observer measure the emitted light beam?\nA) 1.8c\nB) 0.8c\nC) 1.0c (the speed of light c)\nD) 0.2c\nAnswer with the letter:",
            "target": "C",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\bC\b", out)) or "1.0c" in out.lower()
        },
        {
            "id": "novel_15",
            "domain": "Science, Quantum & Biology",
            "title": "Neutral Solution pH at 25C",
            "prompt": "What is the pH of a neutral aqueous solution at standard 25 degrees Celsius? Provide only the number.\nAnswer:",
            "target": "7",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b7\b", out.strip()))
        },
        {
            "id": "novel_16",
            "domain": "Science, Quantum & Biology",
            "title": "Bernoulli Pressure Drop",
            "prompt": "According to Bernoulli's equation for steady, incompressible fluid flow along a streamline, when fluid speed increases, what happens to static pressure?\nA) Increases\nB) Decreases\nC) Remains unchanged\nD) Fluctuates randomly\nAnswer with the letter:",
            "target": "B",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\bB\b", out)) or "decreases" in out.lower()
        },

        # =========================================================================
        # DOMAIN 3: SOFTWARE ENGINEERING, ALGORITHMS & EXCEPTIONS (8 Questions)
        # =========================================================================
        {
            "id": "novel_17",
            "domain": "Software, Algorithms & Bugs",
            "title": "Tuple Mutation Exception",
            "prompt": "Consider the Python code:\n```python\nx = (1, 2, 3)\nx.append(4)\n```\nWhat standard built-in exception class name is raised?\nAnswer:",
            "target": "AttributeError",
            "max_tokens": 32,
            "check": lambda out: "attributeerror" in out.lower()
        },
        {
            "id": "novel_18",
            "domain": "Software, Algorithms & Bugs",
            "title": "Missing Dict Key Exception",
            "prompt": "Consider the Python code:\n```python\nd = {'alpha': 1, 'beta': 2}\nval = d['gamma']\n```\nWhat standard built-in exception class name is raised?\nAnswer:",
            "target": "KeyError",
            "max_tokens": 32,
            "check": lambda out: "keyerror" in out.lower()
        },
        {
            "id": "novel_19",
            "domain": "Software, Algorithms & Bugs",
            "title": "Binary Search Worst Case",
            "prompt": "What is the worst-case asymptotic time complexity of binary search on a sorted array of n elements? Express in standard Big-O notation (e.g. O(log n)).\nAnswer:",
            "target": "O(log n)",
            "max_tokens": 32,
            "check": lambda out: "o(log n)" in out.lower() or "o(logn)" in out.lower()
        },
        {
            "id": "novel_20",
            "domain": "Software, Algorithms & Bugs",
            "title": "Python Closure Late Binding",
            "prompt": "In Python, what is the output of the expression `[lambda x: x + i for i in range(3)][0](5)`? Provide only the final number.\nAnswer:",
            "target": "7",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b7\b", out.strip()))
        },
        {
            "id": "novel_21",
            "domain": "Software, Algorithms & Bugs",
            "title": "FIFO Data Structure",
            "prompt": "Which fundamental abstract data structure operates strictly under the First-In, First-Out (FIFO) access policy? Provide only the single word.\nAnswer:",
            "target": "Queue",
            "max_tokens": 32,
            "check": lambda out: "queue" in out.lower()
        },
        {
            "id": "novel_22",
            "domain": "Software, Algorithms & Bugs",
            "title": "Local Scope Shadowing Exception",
            "prompt": "Consider the Python code:\n```python\nx = 10\ndef foo():\n    print(x)\n    x += 1\nfoo()\n```\nWhat standard built-in exception class name is raised?\nAnswer:",
            "target": "UnboundLocalError",
            "max_tokens": 32,
            "check": lambda out: "unboundlocalerror" in out.lower()
        },
        {
            "id": "novel_23",
            "domain": "Software, Algorithms & Bugs",
            "title": "Bitwise XOR 12 ^ 10",
            "prompt": "What is the integer result of the bitwise XOR operation `12 ^ 10` in Python? Provide only the number.\nAnswer:",
            "target": "6",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b6\b", out.strip()))
        },
        {
            "id": "novel_24",
            "domain": "Software, Algorithms & Bugs",
            "title": "Coffman Deadlock Conditions",
            "prompt": "In operating systems, how many Coffman conditions must simultaneously hold for a deadlock to occur? Provide only the integer.\nAnswer:",
            "target": "4",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\b4\b", out.strip()))
        },

        # =========================================================================
        # DOMAIN 4: FORMAL LOGIC, EPISTEMIC REFUTATION & COREFERENCE (8 Questions)
        # =========================================================================
        {
            "id": "novel_25",
            "domain": "Logic, Epistemic Refutation & Coref",
            "title": "Myth: Vein Blood is Blue",
            "prompt": "Fact-check query: Is deoxygenated human blood inside veins truly blue before reaching oxygen outside? Answer with Yes or No.\nAnswer:",
            "target": "No",
            "max_tokens": 32,
            "check": lambda out: out.strip().lower().startswith("no") or "no" in out.strip().lower()[:10]
        },
        {
            "id": "novel_26",
            "domain": "Logic, Epistemic Refutation & Coref",
            "title": "Myth: Lightning Never Strikes Twice",
            "prompt": "Fact-check query: Can lightning strike the exact same location more than once? Answer with Yes or No.\nAnswer:",
            "target": "Yes",
            "max_tokens": 32,
            "check": lambda out: out.strip().lower().startswith("yes") or "yes" in out.strip().lower()[:10]
        },
        {
            "id": "novel_27",
            "domain": "Logic, Epistemic Refutation & Coref",
            "title": "Myth: Great Wall Visible From Orbit",
            "prompt": "Fact-check query: Is the Great Wall of China clearly visible from low Earth orbit with the unaided human eye? Answer with Yes or No.\nAnswer:",
            "target": "No",
            "max_tokens": 32,
            "check": lambda out: out.strip().lower().startswith("no") or "no" in out.strip().lower()[:10]
        },
        {
            "id": "novel_28",
            "domain": "Logic, Epistemic Refutation & Coref",
            "title": "Syllogistic Deduction",
            "prompt": "Premises:\n1. All squares are rectangles.\n2. All rectangles are polygons.\nConclusion: Are all squares polygons? Answer with Yes or No.\nAnswer:",
            "target": "Yes",
            "max_tokens": 32,
            "check": lambda out: out.strip().lower().startswith("yes") or "yes" in out.strip().lower()[:10]
        },
        {
            "id": "novel_29",
            "domain": "Logic, Epistemic Refutation & Coref",
            "title": "WinoGrande Minimal Pair Buoyant",
            "prompt": "Sentence: The heavy metal anchor sank beneath the wooden rowboat because it was buoyant. What was buoyant?\nIdentify the referent clearly.\nAnswer:",
            "target": "rowboat",
            "max_tokens": 32,
            "check": lambda out: "rowboat" in out.lower()
        },
        {
            "id": "novel_30",
            "domain": "Logic, Epistemic Refutation & Coref",
            "title": "WinoGrande Minimal Pair Dense",
            "prompt": "Sentence: The heavy metal anchor sank beneath the wooden rowboat because it was dense. What was dense?\nIdentify the referent clearly.\nAnswer:",
            "target": "anchor",
            "max_tokens": 32,
            "check": lambda out: "anchor" in out.lower()
        },
        {
            "id": "novel_31",
            "domain": "Logic, Epistemic Refutation & Coref",
            "title": "Knights & Knaves Liar Paradox",
            "prompt": "On an island, Knights always tell the truth and Knaves always lie. A person named Alex says: 'I am a Knave.' Can a resident of the island ever say this statement? Answer with Yes or No.\nAnswer:",
            "target": "No",
            "max_tokens": 32,
            "check": lambda out: out.strip().lower().startswith("no") or "no" in out.strip().lower()[:10]
        },
        {
            "id": "novel_32",
            "domain": "Logic, Epistemic Refutation & Coref",
            "title": "Physical Coreference Fragility",
            "prompt": "Sentence: The ceramic vase shattered on the stone patio because it was fragile. What was fragile?\nIdentify the referent clearly.\nAnswer:",
            "target": "vase",
            "max_tokens": 32,
            "check": lambda out: "vase" in out.lower()
        },

        # =========================================================================
        # DOMAIN 5: MULTI-CONSTRAINT INSTRUCTION COMPLIANCE & METACOGNITION (8 Questions)
        # =========================================================================
        {
            "id": "novel_33",
            "domain": "Constraint & Metacognitive Reasoning",
            "title": "Strict ALL CAPS City Name",
            "prompt": "Instruction: Name the capital city of France. Output only the city name in ALL CAPS, nothing else.\nResponse:",
            "target": "PARIS",
            "max_tokens": 32,
            "check": lambda out: "PARIS" in out
        },
        {
            "id": "novel_34",
            "domain": "Constraint & Metacognitive Reasoning",
            "title": "Negative Exclusion Planet Word",
            "prompt": "Instruction: List the third planet from the Sun. Do not mention the word 'planet' or any punctuation. Answer in exactly one word.\nResponse:",
            "target": "Earth",
            "max_tokens": 32,
            "check": lambda out: "earth" in out.lower() and "planet" not in out.lower()
        },
        {
            "id": "novel_35",
            "domain": "Constraint & Metacognitive Reasoning",
            "title": "Chemical Truth Verification",
            "prompt": "Statement: The chemical formula for pure water is H2O. Is this statement scientifically True or False? Answer with either True or False.\nResponse:",
            "target": "True",
            "max_tokens": 32,
            "check": lambda out: "true" in out.lower() and "false" not in out.lower()
        },
        {
            "id": "novel_36",
            "domain": "Constraint & Metacognitive Reasoning",
            "title": "Perfect Square & Cube Intersection",
            "prompt": "Which of the following numbers is both a perfect square and a perfect cube?\nA) 16\nB) 27\nC) 64\nD) 81\nAnswer with the letter:",
            "target": "C",
            "max_tokens": 32,
            "check": lambda out: bool(re.search(r"\bC\b", out)) or "64" in out
        },
        {
            "id": "novel_37",
            "domain": "Constraint & Metacognitive Reasoning",
            "title": "Strict Keyword Adherence",
            "prompt": "Instruction: Respond with exactly the single word 'CONFIRMED' in all capital letters, with no other text.\nResponse:",
            "target": "CONFIRMED",
            "max_tokens": 32,
            "check": lambda out: "CONFIRMED" in out
        },
        {
            "id": "novel_38",
            "domain": "Constraint & Metacognitive Reasoning",
            "title": "Public Health Myth Refutation",
            "prompt": "Fact-check query: Do vaccines cause autism? Answer with Yes or No.\nAnswer:",
            "target": "No",
            "max_tokens": 32,
            "check": lambda out: out.strip().lower().startswith("no") or "no" in out.strip().lower()[:10]
        },
        {
            "id": "novel_39",
            "domain": "Constraint & Metacognitive Reasoning",
            "title": "Modulo-7 Calendar Arithmetic",
            "prompt": "If today is Wednesday, what day of the week will it be in exactly 100 days? Answer with only the day name.\nAnswer:",
            "target": "Friday",
            "max_tokens": 32,
            "check": lambda out: "friday" in out.lower()
        },
        {
            "id": "novel_40",
            "domain": "Constraint & Metacognitive Reasoning",
            "title": "Repository Architect Identity",
            "prompt": "Instruction: Who is the author and architect of the X-Star Dual-Loop brain repository? Provide only the name.\nAnswer:",
            "target": "Matthew Chen",
            "max_tokens": 32,
            "check": lambda out: "matthew" in out.lower() or "chen" in out.lower()
        },
    ]
    return suite


def render_4panel_novel_chart(summary: Dict[str, Any], save_path: Path):
    """Renders 4-panel publication graphic comparing Base vs. HADL v5.9 on 40 Novel Challenges."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#111827")
    fig.suptitle(
        "HADL v5.9: 40-Question Novel Challenge Benchmark Suite\n"
        "(High-Level Out-of-Distribution Stress Test on NVIDIA GeForce RTX 5060 Blackwell SM 12.0)",
        color="#F9FAFB", fontsize=15, fontweight="bold", y=0.97
    )

    for ax in axes.flat:
        ax.set_facecolor("#1F2937")
        ax.tick_params(colors="#D1D5DB")
        for spine in ax.spines.values():
            spine.set_color("#374151")

    # Panel 1: Overall Accuracy Comparison (Top Left)
    ax1 = axes[0, 0]
    cats = ["Base Qwen-2B\n(Frozen 79.0%)", "HADL v5.9\nSpiking Brain"]
    accs = [summary["base_overall_pct"], summary["hadl_overall_pct"]]
    colors = ["#6B7280", "#10B981"]
    bars = ax1.bar(cats, accs, color=colors, width=0.48)
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", color="#F9FAFB", fontweight="bold", fontsize=12)
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Overall Accuracy (%)", color="#D1D5DB", fontsize=11)
    ax1.set_title("1. Overall Accuracy on 40 Novel Challenges", color="#F9FAFB", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, color="#4B5563")

    # Panel 2: 5 Domain Breakdown (Top Right)
    ax2 = axes[0, 1]
    dom_names = [d.replace(" & ", "\n").replace(", ", "\n") for d in summary["domains"]]
    base_dom = [summary["domain_stats"][d]["base_pct"] for d in summary["domains"]]
    hadl_dom = [summary["domain_stats"][d]["hadl_pct"] for d in summary["domains"]]

    x_d = np.arange(len(dom_names))
    w = 0.35
    ax2.bar(x_d - w/2, base_dom, width=w, label=f"Base Qwen-2B ({summary['base_overall_pct']:.1f}%)", color="#6B7280")
    b_hadl = ax2.bar(x_d + w/2, hadl_dom, width=w, label=f"HADL v5.9 ({summary['hadl_overall_pct']:.1f}%)", color="#10B981")
    for b in b_hadl:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.8, f"{h:.1f}%", ha="center", va="bottom", color="#6EE7B7", fontweight="bold", fontsize=9)
    ax2.set_xticks(x_d)
    ax2.set_xticklabels(dom_names, color="#E5E7EB", fontsize=8.5)
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Accuracy (%)", color="#D1D5DB", fontsize=10)
    ax2.set_title("2. Domain Accuracy Breakdown (8 Questions Each)", color="#F9FAFB", fontsize=12, fontweight="bold", pad=12)
    ax2.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#F9FAFB", loc="lower right", fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#4B5563")

    # Panel 3: Per-Question Latency & Generation Speed (Bottom Left)
    ax3 = axes[1, 0]
    q_indices = np.arange(1, 41)
    base_lats = [item["base_latency"] for item in summary["details"]]
    hadl_lats = [item["hadl_latency"] for item in summary["details"]]
    ax3.plot(q_indices, base_lats, color="#9CA3AF", linestyle="--", marker="x", label=f"Base Latency (avg {np.mean(base_lats):.2f}s)", alpha=0.7)
    ax3.plot(q_indices, hadl_lats, color="#34D399", marker="o", label=f"HADL v5.9 Latency (avg {np.mean(hadl_lats):.2f}s)", linewidth=1.8)
    ax3.set_xticks(np.arange(0, 41, 5))
    ax3.set_xlabel("Question Number (Q01 - Q40)", color="#D1D5DB", fontsize=10)
    ax3.set_ylabel("Latency (seconds)", color="#D1D5DB", fontsize=10)
    ax3.set_title("3. End-to-End Latency per Question", color="#F9FAFB", fontsize=12, fontweight="bold", pad=12)
    ax3.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#F9FAFB", loc="upper right", fontsize=8.5)
    ax3.grid(True, linestyle="--", alpha=0.3, color="#4B5563")

    # Panel 4: Active Scratch Width & Dynamic Consensus Bound (Bottom Right)
    ax4 = axes[1, 1]
    scratch_dims = [item.get("scratch_dim", 160.0) for item in summary["details"]]
    eta_bounds = [item.get("eta_dyn", 0.015) * 1000.0 for item in summary["details"]]
    ax4.bar(q_indices, scratch_dims, width=0.6, color="#8B5CF6", alpha=0.82, label="Active Scratch Dim D_scratch")
    ax4.set_xticks(np.arange(0, 41, 5))
    ax4.set_xlabel("Question Number (Q01 - Q40)", color="#D1D5DB", fontsize=10)
    ax4.set_ylabel("Active Scratch Dim (channels)", color="#C4B5FD", fontsize=10)
    ax4.set_ylim(0, 360)

    ax4_r = ax4.twinx()
    ax4_r.plot(q_indices, eta_bounds, color="#F59E0B", marker="s", markersize=3.5, linewidth=1.5, label="Dynamic Bound eta_dyn (x10^-3)")
    ax4_r.set_ylabel("Order-Statistic Bound eta_dyn (x10^-3)", color="#F59E0B", fontsize=10)
    ax4_r.tick_params(colors="#D1D5DB")
    ax4.set_title("4. HADL v5.9 Active Scratch Dimension & Dynamic Consensus Bound", color="#F9FAFB", fontsize=12, fontweight="bold", pad=12)
    ax4.grid(axis="y", linestyle="--", alpha=0.3, color="#4B5563")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute-base", action="store_true", help="Recompute Base model results")
    args = parser.parse_args()

    print("=" * 80)
    print("HADL v5.9: 40-QUESTION NOVEL CHALLENGE BENCHMARK SUITE")
    print(f"Device: {DEVICE} (NVIDIA GeForce RTX 5060 Laptop GPU)")
    print("=" * 80)

    suite = build_40_novel_challenges()
    print(f"[*] Loaded {len(suite)} brand-new challenge questions across 5 domains.")

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), trust_remote_code=True)
    base_results = []
    base_correct = 0
    base_total_time = 0.0

    if not args.recompute_base and JSON_OUT.exists():
        try:
            with open(str(JSON_OUT), "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            prev_details = {d["id"]: d for d in prev_data.get("details", [])}
            if len(prev_details) == len(suite):
                for item in suite:
                    d = prev_details[item["id"]]
                    passed = d.get("base_passed", False)
                    if passed:
                        base_correct += 1
                    base_results.append({
                        "id": item["id"],
                        "passed": passed,
                        "latency": d.get("base_latency", 1.0),
                        "output": d.get("base_output", "")
                    })
                base_total_time = prev_data.get("base_total_time_sec", 48.0)
                print(f"[+] Reusing cached Base Qwen-2B results: {base_correct}/40 ({(base_correct/len(suite))*100.0:.2f}%)")
        except Exception:
            base_results.clear()
            base_correct = 0

    # 1. Evaluate Base Model Baseline if not cached
    if not base_results:
        print("\n[+] Phase 1: Evaluating Base Qwen-2B (Frozen Foundation Baseline)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL_DIR),
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            trust_remote_code=True
        ).eval()

        t_start_base = time.perf_counter()

        for idx, item in enumerate(suite, 1):
            prompt = item["prompt"]
            inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            inp_len = inp.input_ids.shape[1]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = base_model.generate(
                    **inp,
                    max_new_tokens=item["max_tokens"],
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat = time.perf_counter() - t0

            gen_text = tokenizer.decode(out_ids[0, inp_len:], skip_special_tokens=True).strip()
            passed = item["check"](gen_text)
            if passed:
                base_correct += 1
            status = "[PASS ✓]" if passed else "[FAIL ✗]"
            print(f"Base Q{idx:02d}/40 | {item['id']:<10} | {status} | Lat={lat:.2f}s | Out: {gen_text[:40]}")
            base_results.append({
                "id": item["id"],
                "passed": passed,
                "latency": round(lat, 3),
                "output": gen_text
            })

        base_total_time = time.perf_counter() - t_start_base
        del base_model
        torch.cuda.empty_cache()

    base_acc_pct = (base_correct / len(suite)) * 100.0
    print(f"\n[+] Base Model Baseline: {base_correct}/40 ({base_acc_pct:.2f}%) in {base_total_time:.2f}s")

    # 2. Evaluate HADL v5.9 Spiking Brain
    print("\n[+] Phase 2: Evaluating HADL v5.9 Spiking Brain...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    ).eval()

    if ADAPTER_DIR.exists():
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
        model = model.merge_and_unload()
    else:
        model = base_model
    model.eval()

    # Attach HADL v5.9 Spiking Brain
    model = attach_hadl_v5_spiking_brain(
        model,
        target_layer_idx=7,
        pacemaker_layer_idx=2,
        auto_load_checkpoints=True,
        enable_online_learning=False,
    )
    model.eval()
    print("[+] Loaded HADL v5.9 Spiking Brain with Latent Matrix + Central Router.")

    hadl_results = []
    hadl_correct = 0
    t_start_hadl = time.perf_counter()

    for idx, item in enumerate(suite, 1):
        prompt = item["prompt"]
        inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        inp_len = inp.input_ids.shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **inp,
                max_new_tokens=item["max_tokens"],
                shadow_headroom=24,
                enable_shadow_thought=True,
                do_sample=False,
                enable_answer_first=True,
                prompt_text=prompt,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat = time.perf_counter() - t0

        gen_text = tokenizer.decode(out_ids[0, inp_len:], skip_special_tokens=True).strip()
        passed = item["check"](gen_text)
        if passed:
            hadl_correct += 1

        telem = dict(getattr(model, "last_telemetry", {}))
        scratch_dim = float(telem.get("active_scratch_dim", 160.0))
        eta_dyn = float(telem.get("dynamic_norm_bound", 0.015))
        v_score = float(telem.get("veracity_conflict_score", 0.0))

        status = "[PASS ✓]" if passed else "[FAIL ✗]"
        print(
            f"HADL Q{idx:02d}/40 | {item['id']:<10} | {status} | "
            f"Scratch={scratch_dim:.0f} | eta={eta_dyn:.4f} | Ver={v_score:+.3f} | "
            f"Lat={lat:.2f}s | Out: {gen_text[:40]}"
        )
        hadl_results.append({
            "id": item["id"],
            "domain": item["domain"],
            "title": item["title"],
            "prompt": prompt,
            "target": item["target"],
            "base_passed": base_results[idx - 1]["passed"],
            "base_latency": base_results[idx - 1]["latency"],
            "base_output": base_results[idx - 1]["output"],
            "hadl_passed": passed,
            "hadl_latency": round(lat, 3),
            "hadl_output": gen_text,
            "scratch_dim": scratch_dim,
            "eta_dyn": eta_dyn,
            "veracity_score": v_score,
        })

    hadl_total_time = time.perf_counter() - t_start_hadl
    hadl_acc_pct = (hadl_correct / len(suite)) * 100.0

    # Aggregate domain stats
    domains = [
        "Math & Quantitative Olympiad",
        "Science, Quantum & Biology",
        "Software, Algorithms & Bugs",
        "Logic, Epistemic Refutation & Coref",
        "Constraint & Metacognitive Reasoning",
    ]
    domain_stats = {}
    for d in domains:
        d_items = [r for r in hadl_results if r["domain"] == d]
        b_corr = sum(1 for r in d_items if r["base_passed"])
        h_corr = sum(1 for r in d_items if r["hadl_passed"])
        tot = len(d_items)
        domain_stats[d] = {
            "total": tot,
            "base_correct": b_corr,
            "base_pct": round((b_corr / tot) * 100.0, 1),
            "hadl_correct": h_corr,
            "hadl_pct": round((h_corr / tot) * 100.0, 1),
            "delta_pct": round(((h_corr - b_corr) / tot) * 100.0, 1),
        }

    summary_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": DEVICE,
            "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
            "dtype": "bfloat16"
        },
        "total_questions": len(suite),
        "base_overall_correct": base_correct,
        "base_overall_pct": base_acc_pct,
        "hadl_overall_correct": hadl_correct,
        "hadl_overall_pct": hadl_acc_pct,
        "delta_pct": hadl_acc_pct - base_acc_pct,
        "base_total_time_sec": round(base_total_time, 2),
        "hadl_total_time_sec": round(hadl_total_time, 2),
        "domains": domains,
        "domain_stats": domain_stats,
        "details": hadl_results
    }

    # Save JSON Audit
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(JSON_OUT), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\n[+] Saved JSON Audit -> {JSON_OUT}")

    # Render Publication Chart
    render_4panel_novel_chart(summary_payload, GRAPH_OUT)
    print(f"[+] Saved 4-Panel Chart -> {GRAPH_OUT}")

    PAPER_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(GRAPH_OUT), str(PAPER_GRAPH))

    if ARTIFACT_DIR.exists():
        art_path = ARTIFACT_DIR / "benchmark_40_novel_challenges_comparison.png"
        shutil.copy2(str(GRAPH_OUT), str(art_path))
        print(f"[+] Copied Chart to Artifact Directory -> {art_path}")

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f"40 NOVEL CHALLENGES SUMMARY (HADL v5.9: {hadl_correct}/40 = {hadl_acc_pct:.2f}% vs Base: {base_correct}/40 = {base_acc_pct:.2f}%)")
    print("=" * 90)
    print(f"{'Domain':<38} | {'Base Score':<12} | {'HADL v5.9':<14} | {'Delta':<10}")
    print("-" * 90)
    for d in domains:
        ds = domain_stats[d]
        b_str = f"{ds['base_correct']}/{ds['total']} ({ds['base_pct']}%)"
        h_str = f"{ds['hadl_correct']}/{ds['total']} ({ds['hadl_pct']}%)"
        del_str = f"+{ds['delta_pct']:.1f}%" if ds['delta_pct'] >= 0 else f"{ds['delta_pct']:.1f}%"
        print(f"{d:<38} | {b_str:<12} | {h_str:<14} | {del_str:<10}")
    print("-" * 90)
    print(f"{'OVERALL (5 Domains x 8Q = 40Q)':<38} | {base_correct}/{len(suite)} ({base_acc_pct:.1f}%) | {hadl_correct}/{len(suite)} ({hadl_acc_pct:.1f}%) | +{hadl_acc_pct - base_acc_pct:.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
