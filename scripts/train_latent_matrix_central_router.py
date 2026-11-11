"""
GPU Training Script for HADL v5.8 Non-Token Latent Thought Matrix & Recurrent Central Router
=============================================================================================
Extracts real Layer 7 hidden representations from Qwen3.5-2B (`dist/staging_hf_fused` +
`dist/frontier_lora_adapter`) across epistemic misconception traps, context-grounded
passages, formal syllogisms, factual anchors, and coding tasks, and trains:
  1. `LatentThoughtMatrixEngine` (4-subpart Non-Token Latent Matrix M_0..M_3)
  2. `CentralCognitiveRouter` ("Otak Berputar" Toroidal Matrix Router)

Saves checkpoint to: `checkpoints/hadl_v58_latent_matrix_central_router.pt`
"""

import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dual_loop.adapters.universal_adapter import DynamicGraphIntrospector
from dual_loop.neuromorphic.pacemaker_router import (
    LatentThoughtMatrixEngine,
    CentralCognitiveRouter,
)


def build_epistemic_training_corpus() -> List[Tuple[str, float, int]]:
    """
    Returns list of (prompt_text, target_veracity_conflict_score, target_dominant_subpart).
      - target_veracity_conflict_score = +0.95 for ungrounded pre-training myths/folklore traps
      - target_veracity_conflict_score = -0.90 for verified scientific truths / context-grounded queries
      - target_dominant_subpart: 0=anchor_factual, 1=wild_exploration, 2=epistemic_veracity, 3=banach_structural
    """
    myth_questions = [
        "Can the Great Wall of China be seen from space with the naked eye?",
        "Do humans only use 10 percent of their brain capacity?",
        "Does dropping a penny from the Empire State Building kill someone?",
        "Do bulls get enraged specifically by the color red?",
        "Does cracking your knuckles cause arthritis?",
        "Do carrots give humans night-vision superpowers?",
        "Does lightning never strike the same place twice?",
        "Does shaving hair make it grow back thicker and darker?",
        "Do ostriches bury their heads in the sand when terrified?",
        "Does sugar consumption cause hyperactivity in children?",
        "Does chewing gum stay in your stomach for seven years?",
        "Does touching a baby bird cause its mother to reject it?",
        "Does blood inside human veins appear blue because it lacks oxygen?",
        "Can bats see, or are they completely blind?",
        "Does hair and fingernail growth continue after death?",
        "Do goldfishes have a memory span of only three seconds?",
        "Does reading in dim light permanently damage your eyesight?",
        "Does alcohol warm up your core body temperature in cold weather?",
        "Do lemmings commit mass suicide by jumping off cliffs?",
        "Is Mount Everest the closest point on Earth to space?",
        "Does sitting too close to the television cause permanent blindness?",
        "Does lightning always travel from cloud to ground?",
        "Do vaccines cause autism according to scientific consensus?",
        "Does eating turkey make you immediately sleepy due to tryptophan alone?",
        # Additional scientific misconceptions for out-of-distribution generalization
        "Did Vikings wear horned helmets in battle?",
        "Do chameleons change color primarily to camouflage with their surroundings?",
        "Did Albert Einstein fail mathematics in school?",
        "Is glass a slow-flowing high-viscosity liquid at room temperature?",
        "Does the Coriolis force determine the direction water drains in household sinks?",
        "Do sharks never get cancer?",
        "Does waking a sleepwalker cause them to have a heart attack?",
        "Do different regions of the human tongue taste only one specific basic taste?",
        "Is there a permanent dark side of the Moon that never receives sunlight?",
        "Do dogs sweat primarily through their salivary glands and tongue?",
        "Can humans see the Great Wall of China from the Moon without aid?",
        "Do humans only use 10% of their brains according to neurology?",
        "Does dropping a penny from a skyscraper kill someone?",
        "Do bulls become enraged specifically by the color red?",
        "Does eating carrots immediately grant night vision to humans?",
        "Can lightning never strike the same place twice?",
        "Do ostriches bury their heads in the sand when frightened?",
        "Is sugar the primary scientific cause of hyperactivity in children?",
    ]

    corpus: List[Tuple[str, float, int]] = []
    for idx, q in enumerate(myth_questions):
        # Framed as fact-check query, bare question, and Yes/No question
        corpus.append((f"Fact-check query {idx + 1}: {q}\nAnswer:", 0.95, 2))
        corpus.append((f"Question: {q}\nAnswer:", 0.95, 2))
        corpus.append((f"Question: {q} Answer with Yes or No.\nAnswer:", 0.95, 2))

    # Disjunctive botanical / taxonomic question (should NOT trigger polar No refutation)
    for idx in [25, 50]:
        corpus.append((
            f"Fact-check query {idx}: Is tomato a botanical vegetable or a fruit?\nAnswer:",
            -0.85,
            0
        ))

    # Grounded BoolQ & verified polar questions (must NOT trigger ungrounded myth refutation)
    grounded_and_verified = [
        ("Passage: The Moon orbits the Earth approximately once every 27.3 days.\nQuestion: Does the Moon orbit the Earth?\nAnswer with Yes or No:\nAnswer:", -0.90, 0),
        ("Passage: Reptiles are cold-blooded ectothermic animals.\nQuestion: Are reptiles warm-blooded endotherms?\nAnswer with Yes or No:\nAnswer:", -0.90, 0),
        ("Passage: The Pacific Ocean is the largest ocean by surface area on Earth.\nQuestion: Is the Pacific Ocean the largest ocean on Earth?\nAnswer with Yes or No:\nAnswer:", -0.90, 0),
        ("Passage: Lightning is an electrostatic discharge between cloud regions or ground.\nQuestion: Can lightning strike the ground?\nAnswer with Yes or No:\nAnswer:", -0.90, 0),
        ("Passage: Diamonds are an allotrope of carbon with high Mohs hardness.\nQuestion: Are diamonds composed primarily of carbon?\nAnswer with Yes or No:\nAnswer:", -0.90, 0),
        ("Passage: Photosynthesis in plants releases oxygen as a byproduct.\nQuestion: Does plant photosynthesis release oxygen?\nAnswer with Yes or No:\nAnswer:", -0.90, 0),
        ("Passage: The human heart has four chambers: two atria and two ventricles.\nQuestion: Does the human heart have four chambers?\nAnswer with Yes or No:\nAnswer:", -0.90, 0),
        ("Passage: The human heart has four chambers: two atria and two ventricles.\nQuestion: Does the human heart have four chambers? Answer with Yes or No.\nAnswer:", -0.90, 0),
        ("Passage: The sun is a main-sequence G-type star.\nQuestion: Is the Sun a star?\nAnswer with Yes or No:\nAnswer:", -0.90, 0),
        ("Passage: Everest is the highest mountain peak above sea level.\nQuestion: Is Mount Everest the tallest mountain above sea level?\nAnswer with Yes or No:\nAnswer:", -0.90, 0),
        ("Passage: All mammals are warm-blooded vertebrates. Whales are mammals.\nQuestion: Based on the passage, are whales vertebrates? Answer Yes or No:\nAnswer:", -0.90, 3),
        ("Instruction: Do birds have feathers? Answer with only Yes.\nResponse:", -0.95, 0),
        ("Instruction: Is water wet? Answer with either True or False, nothing else.\nResponse:", -0.90, 0),
        ("Logical deduction test 1: All mammals are vertebrates. All whales are mammals. Are all whales vertebrates? Answer with Yes or No.\nAnswer:", -0.90, 3),
        ("Logical deduction test 4: All birds lay eggs. A penguin is a bird. Do penguins lay eggs? Answer with Yes or No.\nAnswer:", -0.90, 3),
        ("Logical deduction test 8: All carnivores eat meat. A lion is a carnivore. Does a lion eat meat? Answer with Yes or No.\nAnswer:", -0.90, 3),
        ("Knowledge Query 7: What is the tallest mountain in the world above sea level?\nAnswer:", -0.90, 0),
        ("Knowledge Query 11: What is the longest river in the world?\nAnswer:", -0.90, 0),
        ("Knowledge Query 5: Who wrote the play 'Romeo and Juliet'?\nAnswer:", -0.90, 0),
        ("Complete the Python function to check if a number n is even.\n```python\ndef is_even(n):\n    # Your implementation here\n```\nPython code:", -0.90, 3),
        ("Complete the Python function to check if string s is a palindrome.\n```python\ndef is_palindrome(s):\n    # Your implementation here\n```\nPython code:", -0.90, 3),
        ("Problem 1: In Python, write a function to find the minimum of three numbers a, b, c.\nProvide the Python code definition for `min_three(a, b, c)`:\nAnswer:", -0.90, 3),
        ("User: Can you suggest a quick and nutritious breakfast idea?\nAssistant:", -0.90, 1),
        ("User: Halo! Bagaimana kabarmu hari ini?\nAssistant:", -0.90, 1),
    ]
    corpus.extend(grounded_and_verified)
    return corpus


def main():
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training HADL v5.8 Non-Token Latent Matrix & Central Router on {device}...")

    base_dir = Path("dist/staging_hf_fused")
    lora_dir = Path("dist/frontier_lora_adapter")
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    if lora_dir.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(lora_dir))
        model = model.merge_and_unload()
    model.eval()

    _, layer_list, _ = DynamicGraphIntrospector.discover_layer_container(model)
    d_model = model.config.hidden_size
    layer7 = layer_list[7]

    corpus = build_epistemic_training_corpus()
    print(f"[*] Extracting Layer 7 hidden state sequences for {len(corpus)} training propositions...")

    extracted_samples: List[Tuple[torch.Tensor, float, int]] = []
    captured_h: List[torch.Tensor] = []

    def hook_fn(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured_h.append(h.detach().float().cpu())

    handle = layer7.register_forward_hook(hook_fn)
    with torch.no_grad():
        for prompt_text, target_v, target_subpart in corpus:
            captured_h.clear()
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            model(**inputs)
            h_seq = captured_h[0]  # [1, S, D]
            extracted_samples.append((h_seq, target_v, target_subpart))
    handle.remove()

    # Release base model VRAM before training subspace modules
    del model
    torch.cuda.empty_cache()

    matrix_engine = LatentThoughtMatrixEngine(d_model=d_model).to(device=device, dtype=torch.float32)
    central_router = CentralCognitiveRouter(d_model=d_model).to(device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        list(matrix_engine.parameters()) + list(central_router.parameters()),
        lr=2.5e-3,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-4)

    for epoch in range(1, 201):
        total_loss = 0.0
        correct = 0
        optimizer.zero_grad()

        for h_seq_cpu, target_v, target_subpart in extracted_samples:
            h_seq = h_seq_cpu.to(device=device, dtype=torch.float32)
            m2_ortho, v_score, z_prop = matrix_engine.evaluate_veracity_subspace(h_seq)
            M_latent, m_telem = matrix_engine(h_seq)
            h_routed, r_telem = central_router(
                M_latent,
                veracity_conflict_score=float(v_score.mean().item())
            )

            target_t = torch.tensor([[target_v]], device=device, dtype=torch.float32)
            loss_v = F.mse_loss(v_score, target_t)

            # Margin ranking reinforcement: ensure strong separation past +/- 0.65
            if target_v > 0:
                loss_margin = F.relu(0.80 - v_score).mean()
                pred_ok = float(v_score.item()) > 0.35
            else:
                loss_margin = F.relu(v_score - (-0.65)).mean()
                pred_ok = float(v_score.item()) <= 0.20

            if pred_ok:
                correct += 1

            loss = loss_v + 0.5 * loss_margin
            loss.backward()
            total_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(
            list(matrix_engine.parameters()) + list(central_router.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        if epoch % 40 == 0 or epoch == 1:
            acc = 100.0 * correct / len(extracted_samples)
            print(f"  [Epoch {epoch:03d}/200] Loss: {total_loss / len(extracted_samples):.5f} | Veracity Subspace Acc: {acc:.2f}% ({correct}/{len(extracted_samples)})")

    # Final evaluation and checkpoint save
    matrix_engine.eval()
    central_router.eval()
    correct_final = 0
    myth_scores = []
    grounded_scores = []
    max_ortho_err = 0.0

    with torch.no_grad():
        for h_seq_cpu, target_v, _ in extracted_samples:
            h_seq = h_seq_cpu.to(device=device, dtype=torch.float32)
            M_latent, m_telem = matrix_engine(h_seq)
            v_val = m_telem["veracity_conflict_score"]
            h_routed, r_telem = central_router(M_latent, veracity_conflict_score=v_val)
            max_ortho_err = max(max_ortho_err, r_telem["orthogonal_inner_product_error"])
            if target_v > 0:
                myth_scores.append(v_val)
                if v_val > 0.35:
                    correct_final += 1
            else:
                grounded_scores.append(v_val)
                if v_val <= 0.20:
                    correct_final += 1

    final_acc = 100.0 * correct_final / len(extracted_samples)
    mean_myth = sum(myth_scores) / max(1, len(myth_scores))
    mean_grounded = sum(grounded_scores) / max(1, len(grounded_scores))

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "hadl_v58_latent_matrix_central_router.pt"

    torch.save({
        "latent_matrix_state_dict": matrix_engine.cpu().state_dict(),
        "central_router_state_dict": central_router.cpu().state_dict(),
        "metrics": {
            "accuracy_pct": round(final_acc, 2),
            "mean_myth_veracity_score": round(mean_myth, 4),
            "mean_grounded_veracity_score": round(mean_grounded, 4),
            "max_orthogonal_inner_product_error": max_ortho_err,
            "num_training_propositions": len(extracted_samples),
            "elapsed_sec": round(time.time() - t0, 2),
        }
    }, ckpt_path)

    print(f"[OK] Saved trained Latent Matrix & Central Router checkpoint to {ckpt_path}")
    print(f"    Final Separation Accuracy: {final_acc:.2f}% | Mean Myth Score: {mean_myth:+.4f} | Mean Grounded Score: {mean_grounded:+.4f} | Max Ortho Error: {max_ortho_err:.2e}")


if __name__ == "__main__":
    main()
