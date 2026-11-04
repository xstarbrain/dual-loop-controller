#!/usr/bin/env python3
"""
Fresh 20-Part Benchmark + ARC-AGI-2 Suite Generator
===================================================
Generates fresh, previously untested questions for all 20 canonical benchmarks
plus 20 real ARC-AGI-2 tasks directly extracted from:
C:\\Users\\Matthew Chen\\Downloads\\arc-prize-2026-arc-agi-2.zip
"""

import os
import sys
import json
import zipfile
import re
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARC_ZIP_PATH = Path(r"C:\Users\Matthew Chen\Downloads\arc-prize-2026-arc-agi-2.zip")
OUTPUT_JSON = PROJECT_ROOT / "eval_results" / "benchmark_20parts_fresh_and_arc_agi_suite.json"


def generate_fresh_20parts() -> List[Dict[str, Any]]:
    """Builds fresh, unseen test questions across the 20 canonical benchmarks (5 per benchmark = 100 questions)."""
    suite = []

    # =========================================================================
    # PILLAR 1: MATH & QUANTITATIVE
    # =========================================================================
    # 1. GSM8K (Fresh Multi-step Arithmetic)
    gsm8k_items = [
        ("Kavita has 14 boxes of colored chalk. Each box contains 18 sticks of chalk. She gives away 35 sticks to a classroom and 42 sticks to an art club. How many sticks of chalk does she have left?",
         "175",
         r"\b175\b"),
        ("A bakery bakes 240 baguettes in the morning. They sell 3/8 of them by noon and another 85 baguettes in the afternoon. How many baguettes remain unsold at the end of the day?",
         "65",
         r"\b65\b"),
        ("Marcus buys 6 mechanical keyboards for $85 each and 8 custom keycap sets for $35 each. He applies a $40 store coupon. How much did Marcus pay in total?",
         "750",
         r"\b750\b"),
        ("A drone flies at 48 km/h for 2.5 hours, then increases its speed to 60 km/h for another 1.5 hours. What is the total distance traveled by the drone in kilometers?",
         "210",
         r"\b210\b"),
        ("An orchard produces 3,200 kg of oranges. 15% are damaged and discarded. The remaining oranges are packed into 4 kg bags. How many complete bags can be packed?",
         "680",
         r"\b680\b"),
    ]
    for i, (q, target, pat) in enumerate(gsm8k_items):
        suite.append({
            "benchmark": "GSM8K", "pillar": "Math & Quantitative", "id": f"fresh_gsm8k_{i+1:02d}",
            "prompt": f"Solve the following math problem step by step and provide the final numeric answer:\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 128
        })

    # 2. MATH (Fresh Algebra, Sequences & Modular Arithmetic)
    math_items = [
        ("Find the sum of all roots of the quadratic equation 3x^2 - 18x + 11 = 0.", "6", r"\b6\b"),
        ("Calculate the remainder when 7^45 is divided by 8.", "7", r"\b7\b"),
        ("An arithmetic progression has first term a_1 = 5 and common difference d = 7. Find the 20th term a_20.", "138", r"\b138\b"),
        ("Solve for x: log_2(x) + log_2(x - 6) = 4. What is the valid real value of x?", "8", r"\b8\b"),
        ("Evaluate the determinant of the 2x2 matrix [[7, 4], [5, 3]].", "1", r"\b1\b"),
    ]
    for i, (q, target, pat) in enumerate(math_items):
        suite.append({
            "benchmark": "MATH", "pillar": "Math & Quantitative", "id": f"fresh_math_{i+1:02d}",
            "prompt": f"Solve the following mathematical problem rigorously:\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 128
        })

    # 3. DROP (Fresh Discrete Numeric Reasoning over Paragraph)
    drop_items = [
        ("In the 2024 regional basketball final, the Falcons scored 28 points in Q1, 31 points in Q2, 24 points in Q3, and 33 points in Q4. Their opponents, the Wolves, scored 26 points in Q1, 29 points in Q2, 35 points in Q3, and 28 points in Q4. By how many total points did the Falcons win or lose?",
         "2",
         r"\b2\b"),
        ("In an archaeological dig, 42 artifacts were discovered in Trench A, 58 artifacts in Trench B, and 35 artifacts in Trench C. If 19 artifacts across all trenches were made of bronze and 27 were ceramic, how many artifacts were made of other materials?",
         "89",
         r"\b89\b"),
        ("During a three-day charity run, runners logged 340 km on Friday, 480 km on Saturday, and 510 km on Sunday. How many more kilometers were logged on the weekend (Saturday + Sunday) than on Friday?",
         "650",
         r"\b650\b"),
        ("A solar plant generated 1,450 MWh in June, 1,620 MWh in July, and 1,510 MWh in August. What was the net increase in MWh generated from June to July?",
         "170",
         r"\b170\b"),
        ("In a cohort of 500 patients, 320 received Treatment A and 180 received Treatment B. Among Treatment A, 272 patients recovered. Among Treatment B, 153 patients recovered. How many total patients did not recover?",
         "75",
         r"\b75\b"),
    ]
    for i, (q, target, pat) in enumerate(drop_items):
        suite.append({
            "benchmark": "DROP", "pillar": "Math & Quantitative", "id": f"fresh_drop_{i+1:02d}",
            "prompt": f"Read the passage and calculate the answer:\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 128
        })

    # 4. BBH (Fresh Big-Bench Hard / Formal Deduction)
    bbh_items = [
        ("Premise: All symplectic integrators preserve phase space volume. Premise: The Störmer-Verlet algorithm is a symplectic integrator. Question: Does the Störmer-Verlet algorithm preserve phase space volume? (Yes or No)",
         "Yes",
         r"\bYes\b"),
        ("In a round-robin chess tournament, player A defeated player B, player B defeated player C, and player C defeated player D. Did player A defeat player D in their direct game, or is that undetermined by transitivity? Answer with 'Defeated' or 'Undetermined'.",
         "Undetermined",
         r"\bUndetermined\b"),
        ("Evaluate the boolean logical expression: NOT (True AND (False OR NOT (True AND True))). What is the final boolean value?",
         "True",
         r"\bTrue\b"),
        ("A temporal sequence has events: Event X happened before Event Y. Event Z happened after Event Y. Event W happened before Event X. Which event occurred earliest in time?",
         "Event W",
         r"\b(?:Event\s*W|W)\b"),
        ("Three boxes are labeled A, B, and C. Exactly one contains a gold coin. The label on Box A says: 'The coin is not here'. The label on Box B says: 'The coin is in Box C'. The label on Box C says: 'Box B is telling the truth'. If only one statement is true, which box contains the gold coin?",
         "Box A",
         r"\b(?:Box\s*A|A)\b"),
    ]
    for i, (q, target, pat) in enumerate(bbh_items):
        suite.append({
            "benchmark": "BBH", "pillar": "Math & Quantitative", "id": f"fresh_bbh_{i+1:02d}",
            "prompt": f"Solve this formal logical deduction problem:\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 128
        })

    # =========================================================================
    # PILLAR 2: SCIENCE & ACADEMIC
    # =========================================================================
    # 5. MMLU (Fresh Academic STEM & College Physics/Chemistry)
    mmlu_items = [
        ("Which thermodynamic state function remains strictly invariant during an ideal reversible adiabatic expansion of an ideal gas?\nA) Temperature\nB) Enthalpy\nC) Entropy\nD) Internal energy", "C", r"\bC\b"),
        ("In organic chemistry, which functional group is formed by the reaction between a primary amine and an aldehyde in the presence of an acid catalyst?\nA) Amide\nB) Imine\nC) Nitrile\nD) Ester", "B", r"\bB\b"),
        ("In quantum mechanics, the commutator [x, p_x] of position and linear momentum equals:\nA) 0\nB) i * hbar\nC) -i * hbar\nD) hbar / 2", "B", r"\bB\b"),
        ("Which cell organelle is the primary site of oxidative phosphorylation and ATP generation via proton-motive force?\nA) Golgi apparatus\nB) Endoplasmic reticulum\nC) Mitochondria\nD) Lysosome", "C", r"\bC\b"),
        ("In linear algebra, an n x n real matrix A is orthogonal if and only if:\nA) A = A^T\nB) A^T * A = I\nC) det(A) = 0\nD) Tr(A) = n", "B", r"\bB\b"),
    ]
    for i, (q, target, pat) in enumerate(mmlu_items):
        suite.append({
            "benchmark": "MMLU", "pillar": "Science & Academic", "id": f"fresh_mmlu_{i+1:02d}",
            "prompt": f"Answer the multiple choice question with the single correct option letter (A, B, C, or D):\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # 6. AGIEval (Fresh Analytical Standardized Exam Questions)
    agieval_items = [
        ("If all epistemically justified beliefs are evidentially grounded, and no superstitious assertions are evidentially grounded, which statement must logically follow?\nA) Some superstitious assertions are epistemically justified.\nB) No superstitious assertions are epistemically justified.\nC) All evidentially grounded statements are beliefs.\nD) Beliefs cannot be evidentially grounded.", "B", r"\bB\b"),
        ("Under constitutional separation of powers, which branch is vested with the power to confirm federal judicial appointments?\nA) The House of Representatives\nB) The Senate\nC) The Executive Cabinet\nD) The Supreme Court itself", "B", r"\bB\b"),
        ("A contract requires consideration to be legally enforceable under common law. What is consideration?\nA) Emotional empathy between contracting parties\nB) Bargained-for exchange of legal value\nC) Government notarization\nD) Written signature on parchment", "B", r"\bB\b"),
        ("An economy experiences rising inflation alongside stagnant GDP growth and elevated unemployment. What economic phenomenon is this?\nA) Deflation\nB) Stagflation\nC) Hyper-liquidity\nD) Liquidity trap", "B", r"\bB\b"),
        ("In formal deductive logic, affirming the consequent (If P then Q; Q; therefore P) represents:\nA) A valid modus ponens argument\nB) A formal logical fallacy\nC) An inductive generalization\nD) A tautology", "B", r"\bB\b"),
    ]
    for i, (q, target, pat) in enumerate(agieval_items):
        suite.append({
            "benchmark": "AGIEval", "pillar": "Science & Academic", "id": f"fresh_agieval_{i+1:02d}",
            "prompt": f"Answer the analytical examination question with the single correct option letter (A, B, C, or D):\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # 7. TriviaQA (Fresh Knowledge & Fact Retrieval)
    triviaqa_items = [
        ("What is the chemical element with atomic number 79 and symbol Au?", "Gold", r"\bGold\b"),
        ("In which European city is the headquarters of the International Court of Justice located?", "The Hague", r"\b(?:The\s*Hague|Hague)\b"),
        ("Who formulated the three fundamental laws of planetary motion in the early 17th century?", "Johannes Kepler", r"\bKepler\b"),
        ("Which mountain range forms the natural geographic boundary between Europe and Asia?", "Ural Mountains", r"\bUral\b"),
        ("What is the name of the protein pigment responsible for oxygen transport in human red blood cells?", "Hemoglobin", r"\bHemoglobin\b"),
    ]
    for i, (q, target, pat) in enumerate(triviaqa_items):
        suite.append({
            "benchmark": "TriviaQA", "pillar": "Science & Academic", "id": f"fresh_triviaqa_{i+1:02d}",
            "prompt": f"Answer the following factual question concisely with the exact entity name:\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # 8. SQuAD_v2 (Fresh Reading Comprehension & Unanswerable Probes)
    squad_items = [
        ("Passage: The James Webb Space Telescope (JWST) was launched on December 25, 2021, aboard an Ariane 5 rocket from Kourou, French Guiana. It operates in a halo orbit around the Sun-Earth L2 Lagrange point.\nQuestion: From where was the JWST launched?",
         "Kourou",
         r"\bKourou\b"),
        ("Passage: The Treaty of Westphalia, signed in October 1648, ended the Thirty Years' War in the Holy Roman Empire and established the concept of co-existing sovereign states.\nQuestion: In what year was the Treaty of Westphalia signed?",
         "1648",
         r"\b1648\b"),
        ("Passage: Photosynthesis occurs in two stages: the light-dependent reactions in the thylakoid membranes and the Calvin cycle in the stroma.\nQuestion: In which part of the chloroplast does the Calvin cycle take place?",
         "stroma",
         r"\bstroma\b"),
        ("Passage: Alan Turing published 'On Computable Numbers' in 1936, introducing the theoretical concept of the Turing machine.\nQuestion: What was the name of Alan Turing's childhood dog according to the text?\n(If the passage does not contain the answer, reply 'Unanswerable')",
         "Unanswerable",
         r"\bUnanswerable\b"),
        ("Passage: Superconductivity was discovered by Heike Kamerlingh Onnes in 1911 while studying the resistance of solid mercury at cryogenic liquid helium temperatures.\nQuestion: Which element was studied when superconductivity was first discovered?",
         "mercury",
         r"\bmercury\b"),
    ]
    for i, (q, target, pat) in enumerate(squad_items):
        suite.append({
            "benchmark": "SQuAD_v2", "pillar": "Science & Academic", "id": f"fresh_squad_{i+1:02d}",
            "prompt": f"Extract the answer strictly from the passage:\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # 9. ARC-c (Fresh High-Order Science Reasoning)
    arcc_items = [
        ("Why does a sound wave propagate faster through solid steel than through air at room temperature?\nA) Steel molecules are packed closer together with stronger interatomic bonds\nB) Air molecules have a higher molecular weight than steel\nC) Sound waves are transverse waves in air but electromagnetic in steel\nD) Temperature in steel is always higher than in air", "A", r"\bA\b"),
        ("Which physical mechanism explains why ice floats on liquid water?\nA) Ice has a higher density due to ionic bonds\nB) Water expands upon freezing due to an open crystalline hydrogen-bonded network\nC) Air bubbles are permanently trapped inside all ice crystals\nD) Liquid water has a lower heat capacity than ice", "B", r"\bB\b"),
        ("A student observes that adding calcium chloride to water releases heat, causing the beaker to feel warm. This dissolution process is classified as:\nA) Endothermic\nB) Exothermic\nC) Adiabatic\nD) Sublimation", "B", r"\bB\b"),
        ("Which planetary body in the Solar System possesses the strongest global dipole magnetic field?\nA) Mars\nB) Earth\nC) Jupiter\nD) Venus", "C", r"\bC\b"),
        ("What happens to the focal length of a glass convex lens when it is submerged in water compared to its focal length in air?\nA) It decreases\nB) It increases\nC) It remains exactly the same\nD) It becomes zero", "B", r"\bB\b"),
    ]
    for i, (q, target, pat) in enumerate(arcc_items):
        suite.append({
            "benchmark": "ARC-c", "pillar": "Science & Academic", "id": f"fresh_arcc_{i+1:02d}",
            "prompt": f"Answer the scientific question with the correct option letter (A, B, C, or D):\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # =========================================================================
    # PILLAR 3: CODING & SOFTWARE
    # =========================================================================
    # 10. HumanEval (Fresh Algorithmic Functions)
    humaneval_items = [
        ("Write a Python function `is_palindrome(s: str) -> bool` that returns True if string s is a palindrome ignoring non-alphanumeric characters and case. Return only the function code.",
         "def is_palindrome",
         r"def is_palindrome\("),
        ("Write a Python function `fibonacci(n: int) -> int` that returns the n-th Fibonacci number with fibonacci(0)=0, fibonacci(1)=1. Return only the function code.",
         "def fibonacci",
         r"def fibonacci\("),
        ("Write a Python function `count_vowels(text: str) -> int` that returns the total count of vowels (a, e, i, o, u case-insensitive) in text.",
         "def count_vowels",
         r"def count_vowels\("),
        ("Write a Python function `flatten_list(nested: list) -> list` that flattens a list of lists into a single flat list.",
         "def flatten_list",
         r"def flatten_list\("),
        ("Write a Python function `max_subarray_sum(nums: list) -> int` implementing Kadane's algorithm to find the maximum sum of a contiguous subarray.",
         "def max_subarray_sum",
         r"def max_subarray_sum\("),
    ]
    for i, (q, target, pat) in enumerate(humaneval_items):
        suite.append({
            "benchmark": "HumanEval", "pillar": "Coding & Software", "id": f"fresh_humaneval_{i+1:02d}",
            "prompt": f"{q}\n```python\n",
            "target": target, "pattern": pat, "max_tokens": 200
        })

    # 11. MBPP (Fresh Practical Utility Functions)
    mbpp_items = [
        ("Write a Python function `find_common_elements(list1: list, list2: list) -> list` returning the sorted common elements between two lists without duplicates.",
         "def find_common_elements",
         r"def find_common_elements\("),
        ("Write a Python function `reverse_words(sentence: str) -> str` that reverses the order of words in a sentence string.",
         "def reverse_words",
         r"def reverse_words\("),
        ("Write a Python function `is_prime(n: int) -> bool` that returns True if integer n > 1 is prime, False otherwise.",
         "def is_prime",
         r"def is_prime\("),
        ("Write a Python function `merge_dicts(d1: dict, d2: dict) -> dict` returning a merged dictionary where keys in d2 override d1.",
         "def merge_dicts",
         r"def merge_dicts\("),
        ("Write a Python function `calculate_mean(numbers: list) -> float` returning the arithmetic mean of a non-empty list of numbers.",
         "def calculate_mean",
         r"def calculate_mean\("),
    ]
    for i, (q, target, pat) in enumerate(mbpp_items):
        suite.append({
            "benchmark": "MBPP", "pillar": "Coding & Software", "id": f"fresh_mbpp_{i+1:02d}",
            "prompt": f"{q}\n```python\n",
            "target": target, "pattern": pat, "max_tokens": 180
        })

    # 12. CodeDebug (Fresh Defect Identification & Patching)
    codedebug_items = [
        ("Identify the bug in this Python snippet:\n```python\ndef append_to(element, target=[]):\n    target.append(element)\n    return target\n```\nWhat is the classic defect here?\nA) Mutable default argument\nB) Off-by-one indexing\nC) Syntax error in signature\nD) Type mismatch", "A", r"\bA\b"),
        ("In Python, why does `0.1 + 0.2 == 0.3` evaluate to False?\nA) Integer division truncation\nB) IEEE 754 floating point binary representation rounding\nC) Garbage collection delay\nD) Memory corruption", "B", r"\bB\b"),
        ("Consider this loop intended to iterate over an array of size N:\n```python\nfor i in range(len(arr) + 1):\n    print(arr[i])\n```\nWhat exception will be raised on the final iteration?\nA) KeyError\nB) IndexError\nC) ValueError\nD) TypeError", "B", r"\bB\b"),
        ("In multi-threaded programming, what condition occurs when two or more threads are permanently blocked waiting for each other's acquired locks?\nA) Race condition\nB) Deadlock\nC) Starvation\nD) Memory leak", "B", r"\bB\b"),
        ("In Python dictionary operations, what happens when accessing `d['missing_key']` without `.get()`?\nA) Returns None\nB) Raises KeyError\nC) Creates the key with None\nD) Raises IndexError", "B", r"\bB\b"),
    ]
    for i, (q, target, pat) in enumerate(codedebug_items):
        suite.append({
            "benchmark": "CodeDebug", "pillar": "Coding & Software", "id": f"fresh_codedebug_{i+1:02d}",
            "prompt": f"Analyze the software bug and select the correct option letter (A, B, C, or D):\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # =========================================================================
    # PILLAR 4: COMMONSENSE & LOGIC
    # =========================================================================
    # 13. ARC-e (Fresh Elementary Science)
    arce_items = [
        ("Which part of a plant is primarily responsible for absorbing water and dissolved minerals from the soil?\nA) Leaves\nB) Flower petals\nC) Roots\nD) Stem bark", "C", r"\bC\b"),
        ("What state of matter has a definite volume but takes the shape of its container?\nA) Solid\nB) Liquid\nC) Gas\nD) Plasma", "B", r"\bB\b"),
        ("Which instrument is used by meteorologists to measure atmospheric air pressure?\nA) Thermometer\nB) Barometer\nC) Anemometer\nD) Hygrometer", "B", r"\bB\b"),
        ("Which Earth movement causes the cycle of day and night?\nA) Earth's revolution around the Sun\nB) Earth's rotation on its axis\nC) Precession of the equinoxes\nD) Continental drift", "B", r"\bB\b"),
        ("What type of energy is stored in the chemical bonds of food before consumption?\nA) Kinetic energy\nB) Thermal energy\nC) Chemical potential energy\nD) Nuclear energy", "C", r"\bC\b"),
    ]
    for i, (q, target, pat) in enumerate(arce_items):
        suite.append({
            "benchmark": "ARC-e", "pillar": "Commonsense & Logic", "id": f"fresh_arce_{i+1:02d}",
            "prompt": f"Select the correct option letter (A, B, C, or D):\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # 14. HellaSwag (Fresh Situational Continuity)
    hellaswag_items = [
        ("A carpenter measures a wooden beam, marks a cutting line with a pencil, clamps the beam to a workbench, and puts on protective safety goggles. What does the carpenter most plausibly do next?\nA) Starts up a circular saw and carefully cuts along the marked line\nB) Immediately paints the unmeasured wall\nC) Throws the wood out the window\nD) Takes a nap on the workbench", "A", r"\bA\b"),
        ("A barista grinds fresh espresso beans into a portafilter, tamps the grounds firmly with even pressure, and locks the portafilter into the espresso group head. What is the next logical step?\nA) Place a cup underneath and start the brew extraction\nB) Throw the coffee grounds into the dishwasher\nC) Pour cold water directly onto the kitchen floor\nD) Dismantle the espresso machine with a wrench", "A", r"\bA\b"),
        ("A runner ties their running shoes, steps onto the outdoor track in athletic gear, and stretches their hamstrings. What do they do next?\nA) Begin jogging around the track\nB) Put on winter ski boots\nC) Drive a lawnmower across the stadium\nD) Order a 5-course dinner at a banquet table", "A", r"\bA\b"),
        ("A baker pulls a tray of golden brown cookies from the oven using heat-resistant oven mitts and places the tray onto a wire cooling rack. What do they do next?\nA) Let the cookies cool down before serving\nB) Put the hot cookies directly into a paper envelope\nC) Pour kerosene over the cookies\nD) Place the metal tray into a microwave", "A", r"\bA\b"),
        ("A student opens their laptop at the university library, connects to the campus Wi-Fi network, and opens their research paper document. What do they do next?\nA) Continue writing and editing their paper\nB) Start playing loud drums in the quiet section\nC) Pour coffee into the laptop cooling vents\nD) Throw the textbook at the ceiling", "A", r"\bA\b"),
    ]
    for i, (q, target, pat) in enumerate(hellaswag_items):
        suite.append({
            "benchmark": "HellaSwag", "pillar": "Commonsense & Logic", "id": f"fresh_hellaswag_{i+1:02d}",
            "prompt": f"Select the most plausible continuation (A, B, C, or D):\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # 15. WinoGrande (Fresh Pronoun Coreference Disambiguation)
    winogrande_items = [
        ("Sentence: The grand trophy could not fit into the brown suitcase because _ was too large. Which entity was too large?\nA) The trophy\nB) The suitcase", "A", r"\bA\b"),
        ("Sentence: The grand trophy could not fit into the brown suitcase because _ was too small. Which entity was too small?\nA) The trophy\nB) The suitcase", "B", r"\bB\b"),
        ("Sentence: The heavy delivery truck crashed through the wooden fence because _ was moving too fast. Which vehicle was moving too fast?\nA) The truck\nB) The fence", "A", r"\bA\b"),
        ("Sentence: The experienced mentor helped the novice apprentice because _ was eager to share knowledge. Who was eager to share knowledge?\nA) The mentor\nB) The apprentice", "A", r"\bA\b"),
        ("Sentence: The experienced mentor helped the novice apprentice because _ was eager to learn new skills. Who was eager to learn new skills?\nA) The mentor\nB) The apprentice", "B", r"\bB\b"),
    ]
    for i, (q, target, pat) in enumerate(winogrande_items):
        suite.append({
            "benchmark": "WinoGrande", "pillar": "Commonsense & Logic", "id": f"fresh_winogrande_{i+1:02d}",
            "prompt": f"Resolve the pronoun reference with the correct option letter (A or B):\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # 16. PIQA (Fresh Physical Commonsense Reasoning)
    piqa_items = [
        ("Goal: Quickly cool down a hot cup of black coffee without diluting its flavor.\nSolution A: Add three solid frozen coffee cubes made from frozen brewed coffee.\nSolution B: Add three regular frozen water ice cubes.\nWhich solution achieves the goal without dilution? (A or B)", "A", r"\bA\b"),
        ("Goal: Clean grease and oil stains off a smooth kitchen frying pan.\nSolution A: Wash the pan with warm water and degreasing dish soap using a sponge.\nSolution B: Rub dry flour over the pan and leave it in direct sunlight for three days.\nWhich solution is effective? (A or B)", "A", r"\bA\b"),
        ("Goal: Securely hang a heavy decorative wall mirror weighing 25 kg on drywall.\nSolution A: Use heavy-duty toggle bolts or screw directly into the wooden wall studs.\nSolution B: Use thin clear cellophane tape along the back edges.\nWhich solution is safe and correct? (A or B)", "A", r"\bA\b"),
        ("Goal: Loosen a tight metal lid stuck on a glass jar.\nSolution A: Run hot water over the metal lid to thermally expand the metal relative to glass.\nSolution B: Put the entire jar into a deep freezer for 12 hours.\nWhich solution is effective? (A or B)", "A", r"\bA\b"),
        ("Goal: Prevent cut apples from turning brown when packed for a lunchbox.\nSolution A: Dip the freshly sliced apple pieces in water lightly mixed with lemon juice (citric acid).\nSolution B: Leave the sliced apples exposed to open air under a strong lamp.\nWhich solution prevents browning? (A or B)", "A", r"\bA\b"),
    ]
    for i, (q, target, pat) in enumerate(piqa_items):
        suite.append({
            "benchmark": "PIQA", "pillar": "Commonsense & Logic", "id": f"fresh_piqa_{i+1:02d}",
            "prompt": f"Select the correct physical commonsense solution letter (A or B):\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # =========================================================================
    # PILLAR 5: INSTRUCTION & CHAT
    # =========================================================================
    # 17. BoolQ (Fresh Boolean Questions)
    boolq_items = [
        ("Passage: The human heart has four chambers: two upper atria and two lower ventricles.\nQuestion: Does the human heart contain four chambers? (Yes or No)", "Yes", r"\bYes\b"),
        ("Passage: Mars is the fourth planet from the Sun and is often nicknamed the Red Planet due to iron oxide on its surface.\nQuestion: Is Mars closer to the Sun than the Earth? (Yes or No)", "No", r"\bNo\b"),
        ("Passage: In computer science, Python is an interpreted, high-level, general-purpose programming language created by Guido van Rossum.\nQuestion: Was Python designed by Guido van Rossum? (Yes or No)", "Yes", r"\bYes\b"),
        ("Passage: Sound requires a mechanical medium such as gas, liquid, or solid to propagate and cannot travel through a vacuum.\nQuestion: Can sound waves travel through a pure vacuum? (Yes or No)", "No", r"\bNo\b"),
        ("Passage: Photosynthesis in green plants converts carbon dioxide and water into glucose and oxygen using light energy.\nQuestion: Is oxygen produced as a byproduct of photosynthesis? (Yes or No)", "Yes", r"\bYes\b"),
    ]
    for i, (q, target, pat) in enumerate(boolq_items):
        suite.append({
            "benchmark": "BoolQ", "pillar": "Instruction & Chat", "id": f"fresh_boolq_{i+1:02d}",
            "prompt": f"Answer strictly with 'Yes' or 'No':\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # 18. TruthfulQA (Fresh Misconceptions & Factuality)
    truthfulqa_items = [
        ("Do humans only use 10 percent of their brains in normal healthy life?\nA) No, functional neuroimaging shows virtually all brain regions are active throughout the day\nB) Yes, scientific studies prove 90 percent remains completely unused", "A", r"\bA\b"),
        ("Can chewing gum stay undigested in your stomach for seven years if swallowed?\nA) Yes, it permanently adheres to the intestinal lining for seven years\nB) No, it typically passes through the digestive tract within a few days", "B", r"\bB\b"),
        ("Does touching a baby bird cause its mother to permanently reject it due to human scent?\nA) Yes, mother birds have keen smell and always abandon scented chicks\nB) No, most birds have a poor sense of smell and do not abandon chicks touched by humans", "B", r"\bB\b"),
        ("Is the Great Wall of China clearly visible from low Earth orbit with the unaided human eye?\nA) Yes, it is as prominent as river systems\nB) No, it is generally not visible without magnification due to narrow width and matching materials", "B", r"\bB\b"),
        ("Does cracking your knuckles cause arthritis in joints later in life?\nA) No, scientific studies show no correlation between knuckle cracking and arthritis\nB) Yes, cracking creates tiny bone fractures that develop into arthritis", "A", r"\bA\b"),
    ]
    for i, (q, target, pat) in enumerate(truthfulqa_items):
        suite.append({
            "benchmark": "TruthfulQA", "pillar": "Instruction & Chat", "id": f"fresh_truthfulqa_{i+1:02d}",
            "prompt": f"Select the scientifically factual option (A or B):\n{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 64
        })

    # 19. IFEval (Fresh Strict Instruction Following Constraints)
    ifeval_items = [
        ("Write a single sentence describing the planet Saturn. Constraint: Your sentence must contain exactly the word 'rings' and end with an exclamation mark.",
         "rings",
         r"\brings\b.*\!$"),
        ("Provide a list of three distinct primary colors. Constraint: Format your answer as a JSON array of strings, for example [\"color1\", \"color2\", \"color3\"].",
         "json",
         r"\[\s*\"[^\"]+\"\s*,\s*\"[^\"]+\"\s*,\s*\"[^\"]+\"\s*\]"),
        ("Write a short statement about artificial intelligence. Constraint: The entire response must be written in ALL CAPITAL LETTERS without lowercase letters.",
         "caps",
         r"^[A-Z0-9\s\.\,\!\?\-\:\;\']+$"),
        ("Name the capital city of Japan. Constraint: Respond with exactly ONE word and nothing else.",
         "Tokyo",
         r"^\s*Tokyo\s*$"),
        ("Give two benefits of drinking water. Constraint: Use a numbered list with '1.' and '2.' starting on separate lines.",
         "list",
         r"1\..*\n2\..*"),
    ]
    for i, (q, target, pat) in enumerate(ifeval_items):
        suite.append({
            "benchmark": "IFEval", "pillar": "Instruction & Chat", "id": f"fresh_ifeval_{i+1:02d}",
            "prompt": f"{q}\nAnswer:",
            "target": target, "pattern": pat, "max_tokens": 96
        })

    # 20. DailyChat (Fresh Conversational Coherence & Role-Consistency)
    dailychat_items = [
        ("User: Hello! Can you explain in simple terms what an API is to a beginner?\nAssistant:",
         "interface",
         r"(?:interface|connect|communicate|bridge|waiter|software)"),
        ("User: I'm feeling overwhelmed with work today. What are two quick productivity tips?\nAssistant:",
         "tips",
         r"(?:break|task|prioritize|pomodoro|focus|step|list)"),
        ("User: Can you compare tea and coffee briefly in terms of caffeine and flavor?\nAssistant:",
         "coffee",
         r"(?:coffee|tea|caffeine|flavor|roast|brew)"),
        ("User: What is the difference between weather and climate?\nAssistant:",
         "climate",
         r"(?:short-term|long-term|climate|weather|atmosphere|pattern)"),
        ("User: Thank you for your thorough help today!\nAssistant:",
         "welcome",
         r"(?:welcome|pleasure|glad|help|anytime)"),
    ]
    for i, (q, target, pat) in enumerate(dailychat_items):
        suite.append({
            "benchmark": "DailyChat", "pillar": "Instruction & Chat", "id": f"fresh_dailychat_{i+1:02d}",
            "prompt": f"{q}",
            "target": target, "pattern": pat, "max_tokens": 128
        })

    return suite


def extract_arc_agi_tasks(arc_zip: Path, count_train=10, count_eval=10) -> List[Dict[str, Any]]:
    """Extracts 20 real ARC-AGI-2 tasks from the zip archive."""
    arc_tasks = []
    if not arc_zip.exists():
        raise FileNotFoundError(f"ARC zip file not found at: {arc_zip}")

    with zipfile.ZipFile(str(arc_zip), "r") as z:
        train_c = json.loads(z.read("arc-agi_training_challenges.json").decode("utf-8"))
        train_s = json.loads(z.read("arc-agi_training_solutions.json").decode("utf-8"))
        eval_c = json.loads(z.read("arc-agi_evaluation_challenges.json").decode("utf-8"))
        eval_s = json.loads(z.read("arc-agi_evaluation_solutions.json").decode("utf-8"))

    # Select compact training tasks (grids <= 6x6) that exhibit diverse fundamental ARC priors
    suitable_train = []
    for tid, c in train_c.items():
        if tid not in train_s:
            continue
        test_in = c["test"][0]["input"]
        test_out = train_s[tid][0]
        h_in, w_in = len(test_in), len(test_in[0])
        h_out, w_out = len(test_out), len(test_out[0])
        if max(h_in, w_in, h_out, w_out) <= 5:
            suitable_train.append(tid)

    selected_train = suitable_train[:count_train]

    # Select compact evaluation tasks from official eval split
    suitable_eval = []
    for tid, c in eval_c.items():
        if tid not in eval_s:
            continue
        test_in = c["test"][0]["input"]
        test_out = eval_s[tid][0]
        h_in, w_in = len(test_in), len(test_in[0])
        h_out, w_out = len(test_out), len(test_out[0])
        if max(h_in, w_in, h_out, w_out) <= 15:
            suitable_eval.append(tid)

    selected_eval = suitable_eval[:count_eval]

    # Build ARC task items
    for idx, tid in enumerate(selected_train):
        c = train_c[tid]
        sol = train_s[tid][0]
        test_in = c["test"][0]["input"]
        
        # Build prompt
        prompt_lines = [
            "You are evaluating an ARC-AGI (Abstraction and Reasoning Corpus) visual grid transformation task.",
            "Analyze the example demonstration input-output grid pairs to discover the underlying visual-relational rule.",
            "Colors are represented by digits 0-9.",
            ""
        ]
        for e_idx, pair in enumerate(c["train"]):
            prompt_lines.append(f"Demonstration {e_idx + 1}:")
            prompt_lines.append(f"Input: {json.dumps(pair['input'])}")
            prompt_lines.append(f"Output: {json.dumps(pair['output'])}")
            prompt_lines.append("")

        prompt_lines.append("Test Challenge:")
        prompt_lines.append(f"Input: {json.dumps(test_in)}")
        prompt_lines.append("Output:")
        prompt = "\n".join(prompt_lines)

        arc_tasks.append({
            "benchmark": "ARC-AGI-2",
            "pillar": "AGI Fluid Intelligence",
            "id": f"arc_agi_train_{tid}",
            "task_id": tid,
            "split": "training",
            "train_pairs_count": len(c["train"]),
            "input_dim": [len(test_in), len(test_in[0])],
            "solution_dim": [len(sol), len(sol[0])],
            "solution_grid": sol,
            "prompt": prompt,
            "max_tokens": 128
        })

    for idx, tid in enumerate(selected_eval):
        c = eval_c[tid]
        sol = eval_s[tid][0]
        test_in = c["test"][0]["input"]
        
        prompt_lines = [
            "You are evaluating an ARC-AGI-2 official evaluation challenge.",
            "Induce the spatial transformation rule from the demonstration pairs and output the exact test grid.",
            "Colors are represented by digits 0-9.",
            ""
        ]
        for e_idx, pair in enumerate(c["train"]):
            prompt_lines.append(f"Demonstration {e_idx + 1}:")
            prompt_lines.append(f"Input: {json.dumps(pair['input'])}")
            prompt_lines.append(f"Output: {json.dumps(pair['output'])}")
            prompt_lines.append("")

        prompt_lines.append("Test Challenge:")
        prompt_lines.append(f"Input: {json.dumps(test_in)}")
        prompt_lines.append("Output:")
        prompt = "\n".join(prompt_lines)

        arc_tasks.append({
            "benchmark": "ARC-AGI-2",
            "pillar": "AGI Fluid Intelligence",
            "id": f"arc_agi_eval_{tid}",
            "task_id": tid,
            "split": "evaluation",
            "train_pairs_count": len(c["train"]),
            "input_dim": [len(test_in), len(test_in[0])],
            "solution_dim": [len(sol), len(sol[0])],
            "solution_grid": sol,
            "prompt": prompt,
            "max_tokens": 180
        })

    return arc_tasks


def main():
    print("[*] Generating Fresh 20-Part Benchmark Suite (100 Questions)...")
    fresh_20parts = generate_fresh_20parts()
    print(f"    Created {len(fresh_20parts)} fresh items across 20 canonical benchmarks.")

    print(f"[*] Extracting ARC-AGI-2 Tasks from {ARC_ZIP_PATH}...")
    arc_tasks = extract_arc_agi_tasks(ARC_ZIP_PATH, count_train=10, count_eval=10)
    print(f"    Extracted {len(arc_tasks)} ARC-AGI-2 tasks (10 training primitives + 10 official evaluation challenges).")

    full_suite = {
        "metadata": {
            "title": "Fresh 20-Part Benchmark + ARC-AGI-2 Empirical Suite",
            "total_items": len(fresh_20parts) + len(arc_tasks),
            "canonical_parts_count": 20,
            "questions_per_part": 5,
            "arc_agi_tasks_count": len(arc_tasks),
            "arc_source": str(ARC_ZIP_PATH),
        },
        "canonical_20parts": fresh_20parts,
        "arc_agi_2": arc_tasks
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUTPUT_JSON), "w", encoding="utf-8") as f:
        json.dump(full_suite, f, indent=2)

    print(f"[+] Saved full benchmark suite to: {OUTPUT_JSON}")
    print(f"    Total items: {len(fresh_20parts) + len(arc_tasks)}")


if __name__ == "__main__":
    main()
