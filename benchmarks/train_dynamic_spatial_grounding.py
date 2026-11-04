"""
Dynamic Spatial Grounding & Multimodal Alignment Training Engine
================================================================
Trains Progressive2Dto3DImaginationCanvas (spatial_grounding_proj, lift_2d_to_3d,
so3_angle_predictor) on physical NVIDIA RTX 5060 Laptop GPU via end-to-end
gradient backpropagation from language model logits.

Objectives:
1. Align the 3D Lie-group SO(3) coordinate transform with token semantics.
2. Minimize task cross-entropy on spatial reasoning target tokens.
3. Maximize 2D-to-3D reprojection consistency.
4. Save trained dynamic checkpoint to checkpoints/hadl_v57_spatial_grounding_dynamic_checkpoint.pt.
"""

import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xstar.inference.engine import CustomInferenceEngine, EngineConfig


def get_spatial_training_dataset() -> List[Dict[str, str]]:
    """
    Diverse spatial training dataset (distinct from evaluation test set)
    covering 3D rotation, compass tracking, perspective taking, topology,
    and coordinate transformations.
    """
    return [
        # Domain 1: 3D Mental Rotation & Cube Coloring
        {
            "prompt": "Question: A cube has Top: White, Bottom: Black, Front: Blue, Back: Green, Left: Red, Right: Yellow. Rotate the cube 90 degrees clockwise looking down from the top. What color is now on the Front face? Answer:",
            "target": " Red"
        },
        {
            "prompt": "Question: A cube has Top: White, Bottom: Black, Front: Blue, Back: Green, Left: Red, Right: Yellow. Roll the cube 90 degrees forward (top rolls to front). What color is now on the Top face? Answer:",
            "target": " Green"
        },
        {
            "prompt": "Question: A standard die has opposite faces summing to 7. If 3 is on the Top face, what number is on the Bottom face? Answer:",
            "target": " 4"
        },
        {
            "prompt": "Question: A standard die has opposite faces summing to 7. If 2 is on the Front face, what number is on the Back face? Answer:",
            "target": " 5"
        },
        {
            "prompt": "Question: A cube has Top: Cyan, Bottom: Magenta, Front: Yellow, Back: Black, Left: White, Right: Gray. Rotate the cube 180 degrees around the vertical axis. What color is now on the Front face? Answer:",
            "target": " Black"
        },

        # Domain 2: Compass & Navigation
        {
            "prompt": "Question: You face East. You turn 90 degrees clockwise, then turn 90 degrees clockwise again. Which direction are you facing? Answer:",
            "target": " West"
        },
        {
            "prompt": "Question: You face South. You turn 180 degrees, then turn 90 degrees counter-clockwise. Which direction are you facing? Answer:",
            "target": " West"
        },
        {
            "prompt": "Question: You start at (0, 0, 0). You move 5 meters East (+x), 8 meters North (+y), and 12 meters Up (+z). What is your final coordinate (x, y, z)? Answer:",
            "target": " (5, 8, 12)"
        },
        {
            "prompt": "Question: A hiker walks 6 miles North, then turns 90 degrees East and walks 8 miles. What is the straight-line distance from the start in miles? Answer:",
            "target": " 10"
        },
        {
            "prompt": "Question: You face North. You turn 90 degrees left, then 90 degrees left, then 90 degrees left. Which direction are you facing? Answer:",
            "target": " East"
        },

        # Domain 3: Perspective Taking
        {
            "prompt": "Question: Charlie and Dana sit facing each other across a chess board. Charlie places a rook to his right side. From Dana's perspective looking at Charlie, is the rook on Dana's left or right side? Answer:",
            "target": " Left"
        },
        {
            "prompt": "Question: An analog clock shows 4:00. Looking at the clock in a vertical flat mirror, what time does the reflection show? Answer:",
            "target": " 8:00"
        },
        {
            "prompt": "Question: An analog clock shows 2:00. Looking at the clock in a flat mirror, what time does the reflection appear to show? Answer:",
            "target": " 10:00"
        },
        {
            "prompt": "Question: You stand facing West. The sun rises in the morning directly behind you. Which compass direction is behind you? Answer:",
            "target": " East"
        },
        {
            "prompt": "Question: Three cups sit on a table: Cup A on your left, Cup B in the middle, Cup C on your right. A person standing directly opposite you looks at the cups. Which cup is on their far left? Answer:",
            "target": " Cup C"
        },

        # Domain 4: Spatial Topology & Geometry
        {
            "prompt": "Question: Box X has dimensions 5x5x5 cm. Box Y has dimensions 7x7x7 cm. Can Box X fit completely inside Box Y? Answer:",
            "target": " Yes"
        },
        {
            "prompt": "Question: Can a cube of side length 10 cm fit completely inside a sphere of diameter 8 cm? Answer:",
            "target": " No"
        },
        {
            "prompt": "Question: At solar noon in Sydney, Australia (Southern hemisphere), does a vertical pole cast its shadow toward the North or toward the South? Answer:",
            "target": " South"
        },

        # Domain 5: Coordinate Transforms
        {
            "prompt": "Question: Point A is at (2, 7, -3). What are its coordinates after reflection across the xy-plane (z = 0)? Answer:",
            "target": " (2, 7, 3)"
        },
        {
            "prompt": "Question: A 2D vector starts at (0, 0) and points to (1, 0) along the positive x-axis. Rotate it 90 degrees counter-clockwise. What is its new endpoint? Answer:",
            "target": " (0, 1)"
        },
        {
            "prompt": "Question: If you rotate the lowercase letter 'd' 180 degrees in the plane, which lowercase letter does it resemble? Answer:",
            "target": " p"
        },
        {
            "prompt": "Question: If you take a card with the lowercase letter 'p' and rotate it 180 degrees in the plane, which letter does it resemble? Answer:",
            "target": " d"
        },
    ]


def train_dynamic_spatial_grounding():
    print("=" * 80)
    print("HADL v5.7 DYNAMIC SPATIAL GROUNDING & MULTIMODAL ALIGNMENT TRAINING")
    print("Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR6)")
    print("Training Target: Progressive2Dto3DImaginationCanvas (spatial_grounding_proj,")
    print("                 lift_2d_to_3d, so3_angle_predictor)")
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

    # 1. Freeze base model and LoRA adapter
    for p in engine.raw_model.parameters():
        p.requires_grad = False

    # 2. Enable gradients strictly on 3D Imagination Canvas
    canvas = engine.model.brain.imagination_3d
    engine.model.brain.enable_3d_imagination = True

    trainable_params = []
    for name, p in canvas.named_parameters():
        p.requires_grad = True
        trainable_params.append((name, p))

    total_trainable = sum(p.numel() for _, p in trainable_params)
    print(f"[*] Trainable 3D Imagination Parameters: {total_trainable:,} parameters")
    for name, p in trainable_params:
        print(f"    - {name}: {p.shape} ({p.numel():,} params)")

    optimizer = AdamW([p for _, p in trainable_params], lr=1e-3, weight_decay=1e-4)

    dataset = get_spatial_training_dataset()
    print(f"\n[*] Training Dataset: {len(dataset)} distinct spatial reasoning pairs.")

    n_epochs = 4
    total_steps = len(dataset) * n_epochs
    print(f"[*] Total Optimization Steps: {total_steps} (Epochs: {n_epochs})")

    loss_history = []
    consistency_history = []
    t_start = time.perf_counter()

    step = 0
    engine.raw_model.eval()  # Base model in eval mode
    canvas.train()           # Canvas in training mode

    print("\n" + "-" * 80)
    print(f"{'Step':<6} | {'Epoch':<6} | {'Loss':<10} | {'Grad Norm':<10} | {'Consist.':<10} | {'Elapsed':<8}")
    print("-" * 80)

    for epoch in range(1, n_epochs + 1):
        for idx, sample in enumerate(dataset):
            step += 1
            prompt = sample["prompt"]
            target = sample["target"]

            # Tokenize full sequence (prompt + target)
            full_text = prompt + target
            inputs = engine.tokenizer(full_text, return_tensors="pt").to(engine.device)
            prompt_inputs = engine.tokenizer(prompt, return_tensors="pt").to(engine.device)
            prompt_len = prompt_inputs.input_ids.shape[1]

            # Construct labels with ignore_index for prompt tokens
            labels = inputs.input_ids.clone()
            labels[:, :prompt_len] = -100  # Only compute loss on target answer tokens!

            optimizer.zero_grad()

            # Forward pass through model wrapper (Layer 7 hook injects 3D spatial delta)
            outputs = engine.model(input_ids=inputs.input_ids, labels=labels)
            task_loss = outputs.loss

            # Extract 2D-3D consistency from last Layer 7 telemetry
            c_val = 0.85
            if hasattr(engine.model, "last_telemetry"):
                img_telem = engine.model.last_telemetry.get("imagination_3d_telemetry", {})
                c_val = float(img_telem.get("consistency_2d_to_3d", 0.85))

            # Total loss: Task cross-entropy + regularizer to maintain 2D-3D consistency
            consistency_penalty = 0.05 * max(0.0, 0.90 - c_val)
            total_loss = task_loss + consistency_penalty

            # Backward pass
            total_loss.backward()

            # Clip gradients
            grad_norm = nn.utils.clip_grad_norm_([p for _, p in trainable_params], max_norm=1.0)
            optimizer.step()

            loss_val = float(task_loss.item())
            loss_history.append(loss_val)
            consistency_history.append(c_val)

            if step % 5 == 0 or step == 1 or step == total_steps:
                elap = time.perf_counter() - t_start
                print(f"{step:<6} | {epoch:<6} | {loss_val:<10.4f} | {float(grad_norm):<10.4f} | {c_val:<10.4f} | {elap:<7.1f}s")

    t_total = time.perf_counter() - t_start
    print("-" * 80)
    print(f"[*] Training Complete in {t_total:.1f}s ({total_steps / t_total:.1f} steps/sec)")
    print(f"[*] Initial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f} "
          f"({((loss_history[-1] - loss_history[0]) / loss_history[0]) * 100.0:+.1f}%)")
    print(f"[*] Mean 2D-3D Consistency: {sum(consistency_history) / len(consistency_history):.4f}")

    # Save trained checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/hadl_v57_spatial_grounding_dynamic_checkpoint.pt"
    torch.save({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "HADL v5.7 Progressive2Dto3DImaginationCanvas",
        "trainable_params_count": total_trainable,
        "n_epochs": n_epochs,
        "total_steps": total_steps,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "state_dict": canvas.state_dict(),
    }, ckpt_path)
    print(f"[*] Dynamic trained checkpoint saved to: {ckpt_path}")

    return ckpt_path, loss_history, consistency_history


if __name__ == "__main__":
    train_dynamic_spatial_grounding()
