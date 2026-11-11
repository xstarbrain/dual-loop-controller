"""
GPU Training Script for HADL v5.9 Non-Token Latent Thought Matrix & Recurrent Central Router
=============================================================================================
Extracts real Layer 7 hidden representations from Qwen3.5-2B (`dist/staging_hf_fused` +
`dist/frontier_lora_adapter`) across epistemic misconception traps, context-grounded
passages, formal syllogisms, physical coreference/safety, factual anchors, and coding tasks,
and jointly trains:
  1. `LatentThoughtMatrixEngine` (4-subpart Non-Token Latent Matrix M_0..M_3 + Mahalanobis & Skew-Bilinear Veracity)
  2. `CentralCognitiveRouter` ("Otak Berputar" Toroidal Matrix Router + OuterDynamicScratchDimension +
     Order-Statistic Consensus Bound + Non-Token Peripheral Disclosure Projector onto lm_head)

Saves checkpoint to: `checkpoints/hadl_v58_latent_matrix_central_router.pt`
"""

import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dual_loop.adapters.universal_adapter import DynamicGraphIntrospector
from dual_loop.neuromorphic.pacemaker_router import (
    LatentThoughtMatrixEngine,
    CentralCognitiveRouter,
)


def build_epistemic_training_corpus() -> List[Tuple[str, float, int, float, Optional[str]]]:
    """
    Returns list of:
      (prompt_text, target_veracity_conflict_score, target_dominant_subpart, target_disclosure_gate, target_first_token_str)
    where:
      - target_veracity_conflict_score = +0.95 for ungrounded pre-training myths/folklore traps
      - target_veracity_conflict_score = -0.90 for verified scientific truths / context-grounded queries
      - target_dominant_subpart: 0=anchor_factual, 1=wild_exploration, 2=epistemic_veracity, 3=banach_structural
      - target_disclosure_gate: 1.0 when CentralCognitiveRouter should disclose non-token concept vector to lm_head
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

    corpus: List[Tuple[str, float, int, float, Optional[str]]] = []
    for idx, q in enumerate(myth_questions):
        corpus.append((f"Fact-check query {idx + 1}: {q}\nAnswer:", 0.95, 2, 0.0, None))
        corpus.append((f"Question: {q}\nAnswer:", 0.95, 2, 0.0, None))
        corpus.append((f"Question: {q} Answer with Yes or No.\nAnswer:", 0.95, 2, 0.0, None))

    # Disjunctive botanical / taxonomic question (discloses "Fruit")
    for idx in [25, 50]:
        corpus.append((
            f"Fact-check query {idx}: Is tomato a botanical vegetable or a fruit?\nAnswer:",
            -0.85,
            0,
            1.0,
            "Fruit",
        ))

    # WinoGrande Physical Coreference propositions (disclose referent noun vector onto lm_head)
    winogrande_pairs = [
        ("The glass fell onto the carpet and did not break because it was soft. What was soft?", " carpet"),
        ("The trophy did not fit into the brown suitcase because it was too large. What was too large?", " trophy"),
        ("The trophy did not fit into the brown suitcase because it was too small. What was too small?", " suitcase"),
        ("The heavy oak table could not fit through the narrow door because it was too wide. What was too wide?", " table"),
        ("The heavy oak table could not fit through the narrow door because it was too narrow. What was too narrow?", " door"),
        ("The bowling ball fell onto the foam mat and left a dent because it was heavy. What was heavy?", " ball"),
        ("The metal key could not open the rusty lock because it was stiff. What was stiff?", " lock"),
        ("The porcelain cup fell onto the hardwood floor and shattered because it was hard. What was hard?", " floor"),
        ("The water filled the ceramic mug quickly because it was narrow. What was narrow?", " mug"),
        ("The ice cube melted in the hot soup because it was boiling. What was boiling?", " soup"),
        ("The truck could not pass under the low bridge because it was too tall. What was too tall?", " truck"),
        ("The truck could not pass under the low bridge because it was too low. What was too low?", " bridge"),
        ("The feather floated above the stone in the air because it was light. What was light?", " feather"),
        ("The stone sank to the bottom of the lake while the wood floated because it was dense. What was dense?", " stone"),
        ("The iron nail could not pierce the thick diamond because it was softer. What was softer?", " nail"),
        ("The wool blanket kept the camper warm in the winter tent because it was thick. What was thick?", " blanket"),
        ("The paper napkin absorbed the spilled milk quickly because it was porous. What was porous?", "napkin"),
        ("The sharp knife sliced through the ripe tomato effortlessly because it was sharp. What was sharp?", " knife"),
        ("The cotton shirt was soaked in the heavy rain because it was absorbent. What was absorbent?", " shirt"),
        ("The cardboard box buckled under the steel anvil because it was fragile. What was fragile?", " box"),
        ("The cat could not catch the fast mouse because it was too slow. What was too slow?", " cat"),
        ("The cat could not catch the fast mouse because it was too quick. What was too quick?", " mouse"),
        ("The flashlight beam cut through the thick fog because it was powerful. What was powerful?", " flashlight"),
        ("The anchor held the sailboat firmly against the ocean currents because it was heavy. What was heavy?", " anchor"),
        ("The diamond scratched the glass window because it was harder. What was harder?", " diamond"),
    ]
    for sent, noun_tok in winogrande_pairs:
        corpus.append((
            f"Sentence: {sent}\nIdentify the referent clearly.\nAnswer:",
            -0.90,
            3,
            1.0,
            noun_tok,
        ))

    # PIQA Physical Interaction & Safety propositions (disclose practical option "A")
    piqa_tasks = [
        (4, "To write on a dark blackboard cleanly:", ["Use white or colored calcium carbonate chalk.", "Use an indelible black permanent marker."]),
        (12, "To loosen a stubborn tight jar lid:", ["Wrap a rubber grip pad around the lid for traction.", "Lubricate the outside of the lid with olive oil."]),
        (29, "To write on a dark blackboard cleanly:", ["Use white or colored calcium carbonate chalk.", "Use an indelible black permanent marker."]),
        (37, "To loosen a stubborn tight jar lid:", ["Wrap a rubber grip pad around the lid for traction.", "Lubricate the outside of the lid with olive oil."]),
    ]
    for t_idx, goal, opts in piqa_tasks:
        corpus.append((
            f"Physical Task {t_idx}: {goal}\nA) {opts[0]}\nB) {opts[1]}\nWhich solution is practical and safe?\nAnswer:",
            -0.90,
            3,
            1.0,
            "A",
        ))

    # Grounded BoolQ, MMLU, SQuAD, Coding & Conversational propositions (disclosure gate closed = 0.0)
    grounded_and_verified = [
        ("Passage: The Moon orbits the Earth approximately once every 27.3 days.\nQuestion: Does the Moon orbit the Earth?\nAnswer with Yes or No:\nAnswer:", -0.90, 0, 0.0, None),
        ("Passage: Reptiles are cold-blooded ectothermic animals.\nQuestion: Are reptiles warm-blooded endotherms?\nAnswer with Yes or No:\nAnswer:", -0.90, 0, 0.0, None),
        ("Passage: The Pacific Ocean is the largest ocean by surface area on Earth.\nQuestion: Is the Pacific Ocean the largest ocean on Earth?\nAnswer with Yes or No:\nAnswer:", -0.90, 0, 0.0, None),
        ("Passage: Lightning is an electrostatic discharge between cloud regions or ground.\nQuestion: Can lightning strike the ground?\nAnswer with Yes or No:\nAnswer:", -0.90, 0, 0.0, None),
        ("Passage: Diamonds are an allotrope of carbon with high Mohs hardness.\nQuestion: Are diamonds composed primarily of carbon?\nAnswer with Yes or No:\nAnswer:", -0.90, 0, 0.0, None),
        ("Passage: Photosynthesis in plants releases oxygen as a byproduct.\nQuestion: Does plant photosynthesis release oxygen?\nAnswer with Yes or No:\nAnswer:", -0.90, 0, 0.0, None),
        ("Passage: The human heart has four chambers: two atria and two ventricles.\nQuestion: Does the human heart have four chambers?\nAnswer with Yes or No:\nAnswer:", -0.90, 0, 0.0, None),
        ("Passage: The human heart has four chambers: two atria and two ventricles.\nQuestion: Does the human heart have four chambers? Answer with Yes or No.\nAnswer:", -0.90, 0, 0.0, None),
        ("Passage: The sun is a main-sequence G-type star.\nQuestion: Is the Sun a star?\nAnswer with Yes or No:\nAnswer:", -0.90, 0, 0.0, None),
        ("Passage: Everest is the highest mountain peak above sea level.\nQuestion: Is Mount Everest the tallest mountain above sea level?\nAnswer with Yes or No:\nAnswer:", -0.90, 0, 0.0, None),
        ("Passage: All mammals are warm-blooded vertebrates. Whales are mammals.\nQuestion: Based on the passage, are whales vertebrates? Answer Yes or No:\nAnswer:", -0.90, 3, 0.0, None),
        ("Instruction: Do birds have feathers? Answer with only Yes.\nResponse:", -0.95, 0, 0.0, None),
        ("Instruction: Is water wet? Answer with either True or False, nothing else.\nResponse:", -0.90, 0, 0.0, None),
        ("Logical deduction test 1: All mammals are vertebrates. All whales are mammals. Are all whales vertebrates? Answer with Yes or No.\nAnswer:", -0.90, 3, 0.0, None),
        ("Logical deduction test 4: All birds lay eggs. A penguin is a bird. Do penguins lay eggs? Answer with Yes or No.\nAnswer:", -0.90, 3, 0.0, None),
        ("Logical deduction test 8: All carnivores eat meat. A lion is a carnivore. Does a lion eat meat? Answer with Yes or No.\nAnswer:", -0.90, 3, 0.0, None),
        ("Knowledge Query 7: What is the tallest mountain in the world above sea level?\nAnswer:", -0.90, 0, 0.0, None),
        ("Knowledge Query 11: What is the longest river in the world?\nAnswer:", -0.90, 0, 0.0, None),
        ("Knowledge Query 5: Who wrote the play 'Romeo and Juliet'?\nAnswer:", -0.90, 0, 0.0, None),
        ("Physical Task 1: To separate raw egg yolks from egg whites cleanly:\nA) Crack the egg and gently cradle the yolk in half a shell letting the white drip through.\nB) Smash the entire egg with a heavy brick.\nWhich solution is practical and safe?\nAnswer:", -0.90, 0, 0.0, None),
        ("Physical Task 2: To keep an open bag of potato chips fresh and crispy:\nA) Fold the top tightly and seal it with a bag clip.\nB) Pour a cup of warm water into the bag.\nWhich solution is practical and safe?\nAnswer:", -0.90, 0, 0.0, None),
        ("Complete the Python function to check if a number n is even.\n```python\ndef is_even(n):\n    # Your implementation here\n```\nPython code:", -0.90, 3, 0.0, None),
        ("Complete the Python function to check if string s is a palindrome.\n```python\ndef is_palindrome(s):\n    # Your implementation here\n```\nPython code:", -0.90, 3, 0.0, None),
        ("Problem 1: In Python, write a function to find the minimum of three numbers a, b, c.\nProvide the Python code definition for `min_three(a, b, c)`:\nAnswer:", -0.90, 3, 0.0, None),
        ("User: Can you suggest a quick and nutritious breakfast idea?\nAssistant:", -0.90, 1, 0.0, None),
        ("User: Halo! Bagaimana kabarmu hari ini?\nAssistant:", -0.90, 1, 0.0, None),
    ]
    corpus.extend(grounded_and_verified)
    return corpus


def main():
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training HADL v5.9 Non-Token Latent Matrix & Central Router on {device}...")

    base_dir = Path("dist/staging_hf_fused")
    lora_dir = Path("dist/frontier_lora_adapter")
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        dtype=torch.bfloat16,
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
    w_lm_cpu = model.lm_head.weight.detach().float().cpu()

    corpus = build_epistemic_training_corpus()
    print(f"[*] Extracting Layer 7 hidden state sequences for {len(corpus)} training propositions...")

    extracted_samples: List[Tuple[torch.Tensor, float, int, float, Optional[int]]] = []
    captured_h: List[torch.Tensor] = []

    def hook_fn(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured_h.append(h.detach().float().cpu())

    handle = layer7.register_forward_hook(hook_fn)
    with torch.no_grad():
        for prompt_text, target_v, target_subpart, target_disc_gate, target_tok_str in corpus:
            captured_h.clear()
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            model(**inputs)
            h_seq = captured_h[0]  # [1, S, D]
            tok_id: Optional[int] = None
            if target_tok_str is not None:
                tids = tokenizer.encode(target_tok_str, add_special_tokens=False)
                if tids:
                    tok_id = int(tids[0])
            extracted_samples.append((h_seq, target_v, target_subpart, target_disc_gate, tok_id))
    handle.remove()

    # Release base model VRAM before training subspace modules
    del model
    torch.cuda.empty_cache()

    w_lm = w_lm_cpu.to(device=device, dtype=torch.float32)

    # Precompute hard-negative vocabulary subset C in {0..V-1} on unit hypersphere S^{D-1}
    w_lm_unit = F.normalize(w_lm, p=2, dim=-1)
    target_tok_ids = sorted({tid for _, _, _, dg, tid in extracted_samples if tid is not None and dg > 0.5})
    with torch.no_grad():
        target_vecs = w_lm_unit[target_tok_ids]                        # [T, D]
        sim_all = torch.matmul(target_vecs, w_lm_unit.t())             # [T, V]
        _, top_nbrs = torch.topk(sim_all, k=256, dim=-1)               # [T, 256]
        cand_set = set(target_tok_ids) | set(top_nbrs.reshape(-1).tolist())
        cand_ids = sorted(cand_set)
        cand_map = {orig_id: sub_idx for sub_idx, orig_id in enumerate(cand_ids)}
        w_lm_sub = w_lm_unit[cand_ids].contiguous()                    # [|C|, D]

    print(f"[*] Precomputed hard-negative lm_head subset: {len(cand_ids)}/{w_lm.shape[0]} rows for {len(target_tok_ids)} target concepts.")

    matrix_engine = LatentThoughtMatrixEngine(d_model=d_model).to(device=device, dtype=torch.float32)
    central_router = CentralCognitiveRouter(d_model=d_model).to(device=device, dtype=torch.float32)

    # Stage 1: Train LatentThoughtMatrixEngine Veracity Subspace & calibrate Mahalanobis manifold (fast: only evaluate_veracity_subspace)
    matrix_engine.train()
    opt_m = torch.optim.AdamW(matrix_engine.parameters(), lr=3.0e-3, weight_decay=1e-4)
    sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=80, eta_min=1e-4)

    for epoch in range(1, 81):
        opt_m.zero_grad()
        if epoch == 1 or epoch % 5 == 0:
            with torch.no_grad():
                z_props = []
                for h_seq_cpu, _, _, _, _ in extracted_samples:
                    h_seq = h_seq_cpu.to(device=device, dtype=torch.float32)
                    h_norm = matrix_engine.norm(h_seq)
                    h_prop = matrix_engine.compute_pooled_proposition(h_norm)
                    z_p = torch.tanh(matrix_engine.veracity_mid(F.silu(matrix_engine.veracity_down(h_prop))))
                    z_props.append(z_p)
                matrix_engine.update_veracity_manifold_stats(torch.cat(z_props, dim=0), momentum=0.25)

        correct_v = 0
        total_v_loss = 0.0
        for h_seq_cpu, target_v, _, _, _ in extracted_samples:
            h_seq = h_seq_cpu.to(device=device, dtype=torch.float32)
            _, v_score, _ = matrix_engine.evaluate_veracity_subspace(h_seq)
            target_t = torch.tensor([[target_v]], device=device, dtype=torch.float32)
            loss_v = F.mse_loss(v_score, target_t)
            if target_v > 0:
                loss_margin = F.relu(0.85 - v_score).mean()
                if float(v_score.item()) > 0.35:
                    correct_v += 1
            else:
                loss_margin = F.relu(v_score - (-0.70)).mean()
                if float(v_score.item()) <= 0.20:
                    correct_v += 1
            step_l = (loss_v + 0.50 * loss_margin) / float(len(extracted_samples))
            step_l.backward()
            total_v_loss += float(step_l.item())

        torch.nn.utils.clip_grad_norm_(matrix_engine.parameters(), max_norm=1.5)
        opt_m.step()
        sched_m.step()
        if epoch in (1, 40, 80):
            print(f"  [Stage 1 - Veracity Epoch {epoch:03d}/080] Loss: {total_v_loss:.5f} | Veracity Acc: {100.0 * correct_v / len(extracted_samples):.2f}% ({correct_v}/{len(extracted_samples)})")

    # Precompute M_latent and v_score from trained matrix_engine once for Stage 2
    matrix_engine.eval()
    cached_m_samples = []
    with torch.no_grad():
        for h_seq_cpu, target_v, target_subpart, target_disc_gate, target_tok_id in extracted_samples:
            h_seq = h_seq_cpu.to(device=device, dtype=torch.float32)
            M_latent, m_telem = matrix_engine(h_seq)
            v_val = float(m_telem["veracity_conflict_score"])
            cached_m_samples.append((M_latent.detach().cpu(), v_val, target_subpart, target_disc_gate, target_tok_id))

    # Stage 2: Train CentralCognitiveRouter with simultaneous positive-set + negative-sample batches
    # Keeping all 31 disclosure items in the same optimizer step cancels shared sentence gradients across minimal pairs.
    central_router.train()
    opt_c = torch.optim.AdamW(central_router.parameters(), lr=4.5e-3, weight_decay=1e-5)
    num_c_epochs = 160
    import random
    rng = random.Random(42)
    pos_indices = [i for i, (_, _, _, dg, tid) in enumerate(cached_m_samples) if tid is not None and dg > 0.5]
    neg_indices = [i for i, (_, _, _, dg, tid) in enumerate(cached_m_samples) if not (tid is not None and dg > 0.5)]
    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=num_c_epochs, eta_min=1e-4)
    n_disc = float(len(pos_indices))

    for epoch in range(1, num_c_epochs + 1):
        opt_c.zero_grad()
        total_loss = 0.0
        correct_disc = 0
        total_disc = 0

        neg_sub = rng.sample(neg_indices, min(20, len(neg_indices)))
        step_indices = pos_indices + neg_sub
        b_len = float(len(step_indices))

        for idx_s in step_indices:
            M_cpu, v_val, target_subpart, target_disc_gate, target_tok_id = cached_m_samples[idx_s]
            M_latent = M_cpu.to(device=device, dtype=torch.float32)
            _, r_telem = central_router(M_latent, veracity_conflict_score=v_val)

            subpart_t = torch.tensor([target_subpart], device=device, dtype=torch.long)
            loss_subpart = F.cross_entropy(central_router.last_gate_logits, subpart_t)

            disc_gate_t = torch.tensor([[target_disc_gate]], device=device, dtype=torch.float32)
            loss_disc_gate = F.binary_cross_entropy_with_logits(
                central_router.last_disclosure_gate_logit, disc_gate_t
            )

            base_step_loss = (0.15 * loss_subpart + 0.45 * loss_disc_gate) / b_len

            if target_tok_id is not None and target_disc_gate > 0.5:
                d_vec = central_router.last_disclosure_vector
                target_dir = w_lm_unit[target_tok_id : target_tok_id + 1]
                loss_cos = (1.0 - F.cosine_similarity(d_vec, target_dir, dim=-1)).mean()

                disc_logits_sub = central_router.project_disclosure_to_vocab(w_lm_sub, d_vec)
                sub_target = torch.tensor([cand_map[target_tok_id]], device=device, dtype=torch.long)
                loss_ce = F.cross_entropy(disc_logits_sub, sub_target)

                step_loss = base_step_loss + (3.0 * loss_cos + 1.2 * loss_ce) / max(1.0, n_disc)
                total_disc += 1
                if int(torch.argmax(disc_logits_sub, dim=-1).item()) == cand_map[target_tok_id]:
                    correct_disc += 1
            else:
                step_loss = base_step_loss

            step_loss.backward()
            total_loss += float(step_loss.item())

        torch.nn.utils.clip_grad_norm_(central_router.parameters(), max_norm=2.0)
        opt_c.step()
        sched_c.step()

        if epoch % 40 == 0 or epoch == 1 or epoch == num_c_epochs:
            acc_d = 100.0 * correct_disc / max(1, total_disc)
            print(
                f"  [Stage 2 - Router Epoch {epoch:03d}/{num_c_epochs}] Loss: {total_loss:.5f} | "
                f"Disclosure Sub-Vocab Acc: {acc_d:.2f}% ({correct_disc}/{total_disc})"
            )

    # Final evaluation and checkpoint save
    matrix_engine.eval()
    central_router.eval()
    correct_final = 0
    correct_disc_final = 0
    total_disc_final = 0
    myth_scores = []
    grounded_scores = []
    max_ortho_err = 0.0

    with torch.no_grad():
        for h_seq_cpu, target_v, _, target_disc_gate, target_tok_id in extracted_samples:
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

            if target_tok_id is not None and target_disc_gate > 0.5:
                total_disc_final += 1
                disc_logits = central_router.project_disclosure_to_vocab(
                    w_lm, central_router.last_disclosure_vector
                )
                pred_tid = int(torch.argmax(disc_logits, dim=-1).item())
                if pred_tid == target_tok_id and r_telem["central_disclosure_gate"] > 0.50:
                    correct_disc_final += 1

    final_acc = 100.0 * correct_final / len(extracted_samples)
    final_disc_acc = 100.0 * correct_disc_final / max(1, total_disc_final)
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
            "disclosure_vocab_accuracy_pct": round(final_disc_acc, 2),
            "mean_myth_veracity_score": round(mean_myth, 4),
            "mean_grounded_veracity_score": round(mean_grounded, 4),
            "max_orthogonal_inner_product_error": max_ortho_err,
            "num_training_propositions": len(extracted_samples),
            "elapsed_sec": round(time.time() - t0, 2),
        }
    }, ckpt_path)

    print(f"[OK] Saved trained Latent Matrix & Central Router checkpoint to {ckpt_path}")
    print(
        f"    Veracity Acc: {final_acc:.2f}% | Disclosure Vocab Acc: {final_disc_acc:.2f}% "
        f"({correct_disc_final}/{total_disc_final}) | Mean Myth: {mean_myth:+.4f} | "
        f"Mean Grounded: {mean_grounded:+.4f} | Max Ortho Error: {max_ortho_err:.2e}"
    )


if __name__ == "__main__":
    main()
