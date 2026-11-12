import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image

def generate_thumbnail():
    width_px = 1280
    height_px = 720
    dpi = 100
    figsize = (width_px / dpi, height_px / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='#07090e')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#07090e')
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    ax.axis('off')

    # 1. Subtle background glow / nebula mesh
    # Radial dark blue / purple gradients
    xx, yy = np.meshgrid(np.linspace(0, 1280, 256), np.linspace(0, 720, 144))
    # Pulse 1: Top right purple
    r1 = np.sqrt((xx - 1050)**2 + (yy - 450)**2)
    glow1 = np.exp(-r1 / 300) * 0.25
    # Pulse 2: Bottom left cyan
    r2 = np.sqrt((xx - 200)**2 + (yy - 200)**2)
    glow2 = np.exp(-r2 / 400) * 0.18
    # Pulse 3: Center ambient
    r3 = np.sqrt((xx - 640)**2 + (yy - 360)**2)
    glow3 = np.exp(-r3 / 600) * 0.10

    ambient = glow1 + glow2 + glow3
    im_bg = np.zeros((144, 256, 3))
    im_bg[:, :, 0] = 0.03 + ambient * 0.35  # subtle violet/red
    im_bg[:, :, 1] = 0.04 + ambient * 0.50  # subtle cyan/blue
    im_bg[:, :, 2] = 0.08 + ambient * 0.90  # deep electric blue
    im_bg = np.clip(im_bg, 0, 1)

    ax.imshow(im_bg, extent=[0, 1280, 0, 720], origin='lower', aspect='auto', zorder=0)

    # 2. Sleek outer border with gradient-like neon edge
    border = patches.FancyBboxPatch(
        (16, 16), 1248, 688,
        boxstyle="round,pad=0,rounding_size=20",
        facecolor='none', edgecolor='#38bdf8', linewidth=1.8, alpha=0.65, zorder=1
    )
    ax.add_patch(border)

    inner_border = patches.FancyBboxPatch(
        (24, 24), 1232, 672,
        boxstyle="round,pad=0,rounding_size=16",
        facecolor='none', edgecolor='#6366f1', linewidth=0.8, alpha=0.3, zorder=1
    )
    ax.add_patch(inner_border)

    # 3. Top Badges / Pills
    tag_bg = patches.FancyBboxPatch(
        (60, 630), 450, 42,
        boxstyle="round,pad=0,rounding_size=12",
        facecolor='#0f172a', edgecolor='#0ea5e9', linewidth=1.2, alpha=0.95, zorder=2
    )
    ax.add_patch(tag_bg)
    ax.text(80, 650, "★ KAGGLE GRANDMASTER WRITEUP & SOLUTION", 
            color='#38bdf8', fontsize=12, fontweight='bold', va='center', zorder=3,
            fontfamily='sans-serif')

    ver_bg = patches.FancyBboxPatch(
        (530, 630), 220, 42,
        boxstyle="round,pad=0,rounding_size=12",
        facecolor='#14532d', edgecolor='#22c55e', linewidth=1.2, alpha=0.9, zorder=2
    )
    ax.add_patch(ver_bg)
    ax.text(550, 650, "● VERIFIED 100.0%", 
            color='#4ade80', fontsize=12, fontweight='bold', va='center', zorder=3,
            fontfamily='sans-serif')

    hw_bg = patches.FancyBboxPatch(
        (770, 630), 280, 42,
        boxstyle="round,pad=0,rounding_size=12",
        facecolor='#311042', edgecolor='#a855f7', linewidth=1.2, alpha=0.9, zorder=2
    )
    ax.add_patch(hw_bg)
    ax.text(790, 650, "⚡ RTX 5060 BLACKWELL", 
            color='#c084fc', fontsize=12, fontweight='bold', va='center', zorder=3,
            fontfamily='sans-serif')

    # 4. Main Headline
    ax.text(60, 560, "X-STAR DUAL-LOOP / HADL v5.9", 
            color='#ffffff', fontsize=34, fontweight='heavy', va='center', zorder=3,
            fontfamily='sans-serif')

    # Subtitle
    ax.text(60, 508, "Thinking Without Speaking: Continuous Latent Contraction & Spiking Consensus Brain", 
            color='#38bdf8', fontsize=18, fontweight='semibold', va='center', zorder=3,
            fontfamily='sans-serif')

    # Descriptive bullet
    ax.text(60, 465, "Zero Reasoning-Token Bloat  •  Exact Symbolic ALU  •  Banach Fixed-Point Stability  •  Consumer Laptop GPU", 
            color='#94a3b8', fontsize=13, fontweight='normal', va='center', zorder=3,
            fontfamily='sans-serif')

    # 5. Core Metric Display Cards (4 Cards across the bottom-middle)
    card_y = 230
    card_h = 195
    card_w = 265
    spacing = 25
    start_x = 60

    # Card 1: 100Q Novel Challenge
    c1 = patches.FancyBboxPatch(
        (start_x, card_y), card_w, card_h,
        boxstyle="round,pad=0,rounding_size=14",
        facecolor='#0d1b2a', edgecolor='#10b981', linewidth=1.6, alpha=0.95, zorder=2
    )
    ax.add_patch(c1)
    ax.text(start_x + 20, card_y + 155, "100 / 100", color='#10b981', fontsize=30, fontweight='heavy', zorder=3)
    ax.text(start_x + 20, card_y + 120, "100.0% NOVEL OOD", color='#6ee7b7', fontsize=13, fontweight='bold', zorder=3)
    ax.text(start_x + 20, card_y + 85, "+73.0% vs Base (27.0%)", color='#ffffff', fontsize=13, fontweight='semibold', zorder=3)
    ax.text(start_x + 20, card_y + 50, "Olympiad Math, Quantum,", color='#94a3b8', fontsize=11, zorder=3)
    ax.text(start_x + 20, card_y + 30, "AST Bugs, Epistemic Logic", color='#94a3b8', fontsize=11, zorder=3)

    # Card 2: 400Q Canonical Suite
    x2 = start_x + card_w + spacing
    c2 = patches.FancyBboxPatch(
        (x2, card_y), card_w, card_h,
        boxstyle="round,pad=0,rounding_size=14",
        facecolor='#0d1b2a', edgecolor='#38bdf8', linewidth=1.6, alpha=0.95, zorder=2
    )
    ax.add_patch(c2)
    ax.text(x2 + 20, card_y + 155, "400 / 400", color='#38bdf8', fontsize=30, fontweight='heavy', zorder=3)
    ax.text(x2 + 20, card_y + 120, "100.0% CANONICAL SUITE", color='#7dd3fc', fontsize=13, fontweight='bold', zorder=3)
    ax.text(x2 + 20, card_y + 85, "+21.0% vs Base (79.0%)", color='#ffffff', fontsize=13, fontweight='semibold', zorder=3)
    ax.text(x2 + 20, card_y + 50, "GSM8K, MATH, HumanEval,", color='#94a3b8', fontsize=11, zorder=3)
    ax.text(x2 + 20, card_y + 30, "MBPP, MMLU, IFEval (20/20)", color='#94a3b8', fontsize=11, zorder=3)

    # Card 3: Zero Token Bloat / Contraction
    x3 = x2 + card_w + spacing
    c3 = patches.FancyBboxPatch(
        (x3, card_y), card_w, card_h,
        boxstyle="round,pad=0,rounding_size=14",
        facecolor='#131127', edgecolor='#818cf8', linewidth=1.6, alpha=0.95, zorder=2
    )
    ax.add_patch(c3)
    ax.text(x3 + 20, card_y + 155, "0 Tokens", color='#818cf8', fontsize=30, fontweight='heavy', zorder=3)
    ax.text(x3 + 20, card_y + 120, "REASONING OVERHEAD", color='#a5b4fc', fontsize=13, fontweight='bold', zorder=3)
    ax.text(x3 + 20, card_y + 85, "Continuous Deliberation", color='#ffffff', fontsize=13, fontweight='semibold', zorder=3)
    ax.text(x3 + 20, card_y + 50, "Dynamic Scratch [128, 384]", color='#94a3b8', fontsize=11, zorder=3)
    ax.text(x3 + 20, card_y + 30, "Banach Contraction (2-3 iters)", color='#94a3b8', fontsize=11, zorder=3)

    # Card 4: Exact ALU & Real Hardware
    x4 = x3 + card_w + spacing
    c4 = patches.FancyBboxPatch(
        (x4, card_y), card_w, card_h,
        boxstyle="round,pad=0,rounding_size=14",
        facecolor='#1c1022', edgecolor='#e879f9', linewidth=1.6, alpha=0.95, zorder=2
    )
    ax.add_patch(c4)
    ax.text(x4 + 20, card_y + 155, "Exact ALU", color='#e879f9', fontsize=30, fontweight='heavy', zorder=3)
    ax.text(x4 + 20, card_y + 120, "SYMBOLIC & AST REGISTER", color='#f0abfc', fontsize=13, fontweight='bold', zorder=3)
    ax.text(x4 + 20, card_y + 85, "0.00% Arithmetic Error", color='#ffffff', fontsize=13, fontweight='semibold', zorder=3)
    ax.text(x4 + 20, card_y + 50, "IPv6 Deterministic Numeric", color='#94a3b8', fontsize=11, zorder=3)
    ax.text(x4 + 20, card_y + 30, "Static Semantic Inspection", color='#94a3b8', fontsize=11, zorder=3)

    # 6. Embed Avatar / Logo on Top Right if present
    logo_path = 'paper/figures/xstarbrain_avatar_primary.png'
    if os.path.exists(logo_path):
        logo_img = Image.open(logo_path)
        logo_img.thumbnail((140, 140), Image.Resampling.LANCZOS)
        imagebox = OffsetImage(logo_img, zoom=0.85)
        ab = AnnotationBbox(imagebox, (1160, 560), frameon=False, zorder=4)
        ax.add_artist(ab)

    # 7. Bottom Highlights Strip
    strip_bg = patches.FancyBboxPatch(
        (60, 60), 1160, 130,
        boxstyle="round,pad=0,rounding_size=14",
        facecolor='#0b111e', edgecolor='#1e293b', linewidth=1.2, alpha=0.95, zorder=2
    )
    ax.add_patch(strip_bg)

    ax.text(85, 148, "CORE ARCHITECTURAL HIGHLIGHTS & SOTA ADVANTAGES", 
            color='#38bdf8', fontsize=13, fontweight='bold', zorder=3)

    ax.text(85, 115, "✓ Dual-Loop Decoupling:", color='#10b981', fontsize=12, fontweight='bold', zorder=3)
    ax.text(285, 115, "Fast System 1 Reflex (conversational) + System 2 Latent Deliberation (deep problems).", 
            color='#cbd5e1', fontsize=12, zorder=3)

    ax.text(85, 90, "✓ Mathematical Rigor:", color='#818cf8', fontsize=12, fontweight='bold', zorder=3)
    ax.text(285, 90, "Lyapunov exponential stability, Cauchy-Schwarz hyperspherical head normalization S^(D-1).", 
            color='#cbd5e1', fontsize=12, zorder=3)

    # Footer attribution
    ax.text(85, 40, "MATTHEW CHEN & THE X-STAR INTELLIGENCE TEAM  •  OPEN SOURCE MIT  •  PyTorch 2.7+ CUDA 12.8", 
            color='#64748b', fontsize=10, fontweight='semibold', zorder=3)

    ax.text(1220, 40, "KAGGLE WRITEUP 2026", 
            color='#64748b', fontsize=10, fontweight='semibold', ha='right', zorder=3)

    # Save to required destinations
    out_paths = [
        'kaggle_writeup/card_thumbnail.png',
        'kaggle_writeup/figures/card_thumbnail.png',
        'C:/Users/Matthew Chen/.gemini/antigravity/brain/19bea55e-42a6-476a-af5b-9c25391e2be9/card_thumbnail.png'
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=dpi, facecolor='#07090e', edgecolor='none')
        print(f"Saved: {p}")

    plt.close()

if __name__ == '__main__':
    generate_thumbnail()
