"""
X-Star Brain Profile Avatar Generator
Creates an ultra-crisp, mathematically precise 1024x1024 avatar emblem
representing:
- X-Star Autonomous Brain
- Non-Token Continuous Latent Cognition
- Recurrent Toroidal Brain ("Otak Berputar")
- 4-Spoke Latent Matrix ($M_0, M_1, M_2, M_3$)
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_radial_gradient(size, inner_color, outer_color, center=None, radius=None):
    W, H = size
    if center is None:
        cx, cy = W / 2, H / 2
    else:
        cx, cy = center
    if radius is None:
        radius = math.hypot(cx, cy)
    
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    t = np.clip(dist / radius, 0.0, 1.0)
    
    r = (1 - t) * inner_color[0] + t * outer_color[0]
    g = (1 - t) * inner_color[1] + t * outer_color[1]
    b = (1 - t) * inner_color[2] + t * outer_color[2]
    a = (1 - t) * inner_color[3] + t * outer_color[3] if len(inner_color) == 4 else np.full_like(t, 255)
    
    img_array = np.dstack((r, g, b, a)).astype(np.uint8)
    return Image.fromarray(img_array)

def render_avatar(size=1024, supersample=2):
    S = size * supersample
    center = S / 2.0
    
    # 1. Base Background with Deep Quantum Gradient
    # Deep midnight slate (#070b14) to obsidian black (#020408) with subtle central cyan glow (#0e1e38)
    bg = create_radial_gradient(
        (S, S),
        inner_color=(16, 28, 52, 255),
        outer_color=(4, 6, 12, 255),
        center=(center, center),
        radius=S * 0.7
    )
    
    # Glow overlay layer
    glow_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    
    # Central ambient glow
    central_glow = create_radial_gradient(
        (S, S),
        inner_color=(0, 240, 255, 60),
        outer_color=(0, 0, 0, 0),
        center=(center, center),
        radius=S * 0.35
    )
    glow_layer = Image.alpha_composite(glow_layer, central_glow)
    
    # Vector Geometry Layer
    geom_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(geom_layer)
    
    # 2. Recurrent Toroidal Brain ("Otak Berputar" / Continuous Dynamic Vortex)
    # Render nested concentric/eccentric elliptical orbital rings rotated at complementary angles
    num_torus_rings = 4  # Represents the 4 subparts of Rank-4 Latent Thought Matrix
    ring_angles = [25, 70, 115, 160]
    ring_colors = [
        (0, 242, 254, 220),   # Electric Cyan (M_0 Anchor)
        (79, 172, 254, 210),  # Radiant Blue (M_1 Wild/Lateral)
        (139, 92, 246, 220),  # Royal Violet (M_2 Epistemic Veracity)
        (236, 72, 153, 200)   # Fuchsia Neon (M_3 Syllogistic Deduction)
    ]
    
    ring_rx = S * 0.38
    ring_ry = S * 0.16
    
    # Draw Torus Ring Loops
    for idx, (angle_deg, color) in enumerate(zip(ring_angles, ring_colors)):
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        # Approximate ellipse with smooth polygon points
        points = []
        num_points = 240
        for i in range(num_points):
            theta = 2.0 * math.pi * i / num_points
            # parametric ellipse
            x0 = ring_rx * math.cos(theta)
            y0 = ring_ry * math.sin(theta)
            # rotate
            xr = x0 * cos_a - y0 * sin_a + center
            yr = x0 * sin_a + y0 * cos_a + center
            points.append((xr, yr))
        
        # Draw soft outer glow for each ring
        ring_glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        rg_draw = ImageDraw.Draw(ring_glow)
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            rg_draw.line([p1, p2], fill=(color[0], color[1], color[2], 120), width=int(12 * supersample))
        ring_glow = ring_glow.filter(ImageFilter.GaussianBlur(radius=8 * supersample))
        geom_layer = Image.alpha_composite(geom_layer, ring_glow)
        draw = ImageDraw.Draw(geom_layer)
        
        # Draw crisp high-intensity core line
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            # Variable alpha along the loop to show dynamic recirculation
            phase = math.sin(2.0 * math.pi * i / num_points + idx * 1.2)
            alpha = int(140 + 115 * ((phase + 1) / 2))
            draw.line([p1, p2], fill=(color[0], color[1], color[2], alpha), width=int(3.5 * supersample))
        
        # Add Spiking Neuromorphic Nodes along the ring
        num_nodes = 3
        for k in range(num_nodes):
            node_idx = int((k * (num_points // num_nodes) + idx * 23) % num_points)
            nx, ny = points[node_idx]
            node_r = 7.0 * supersample
            # Glow behind node
            draw.ellipse([nx - node_r * 2.2, ny - node_r * 2.2, nx + node_r * 2.2, ny + node_r * 2.2],
                         fill=(color[0], color[1], color[2], 90))
            # White-hot node center
            draw.ellipse([nx - node_r, ny - node_r, nx + node_r, ny + node_r],
                         fill=(255, 255, 255, 240))
            draw.ellipse([nx - node_r * 0.5, ny - node_r * 0.5, nx + node_r * 0.5, ny + node_r * 0.5],
                         fill=(255, 255, 255, 255))

    # 3. Central "X-Star" Stellar Core (The 4-Spoke Continuous Latent Engine)
    # A geometric 4-pointed radiant hyper-diamond star with curved parabolic spines
    star_glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sg_draw = ImageDraw.Draw(star_glow)
    
    def generate_astroid_star(cx, cy, r_outer, r_inner, rotation_deg=0):
        # 4-pointed star with concave astroid curves
        rot = math.radians(rotation_deg)
        curve_pts = []
        for q in range(4):
            t_rad = rot + q * (math.pi / 2.0)
            t_next_rad = rot + (q + 1) * (math.pi / 2.0)
            
            p_tip = (cx + r_outer * math.cos(t_rad), cy + r_outer * math.sin(t_rad))
            p_next_tip = (cx + r_outer * math.cos(t_next_rad), cy + r_outer * math.sin(t_next_rad))
            # Concave control point towards center
            c_angle = (t_rad + t_next_rad) / 2.0
            p_ctrl = (cx + r_inner * math.cos(c_angle), cy + r_inner * math.sin(c_angle))
            
            # Subdivide quadratic Bezier curve
            for step in range(25):
                u = step / 25.0
                bx = (1 - u)**2 * p_tip[0] + 2 * (1 - u) * u * p_ctrl[0] + u**2 * p_next_tip[0]
                by = (1 - u)**2 * p_tip[1] + 2 * (1 - u) * u * p_ctrl[1] + u**2 * p_next_tip[1]
                curve_pts.append((bx, by))
        return curve_pts

    # Outer Ambient Star Halo
    star_pts_outer = generate_astroid_star(center, center, r_outer=S * 0.32, r_inner=S * 0.065, rotation_deg=45)
    sg_draw.polygon(star_pts_outer, fill=(0, 240, 255, 140))
    star_glow = star_glow.filter(ImageFilter.GaussianBlur(radius=20 * supersample))
    geom_layer = Image.alpha_composite(geom_layer, star_glow)
    draw = ImageDraw.Draw(geom_layer)
    
    # Intermediate Star
    star_pts_mid = generate_astroid_star(center, center, r_outer=S * 0.28, r_inner=S * 0.055, rotation_deg=45)
    draw.polygon(star_pts_mid, fill=(0, 220, 255, 200))
    
    # Diamond Core Inlay (rotated 0 deg to create multi-dimensional X structure)
    core_pts = generate_astroid_star(center, center, r_outer=S * 0.17, r_inner=S * 0.038, rotation_deg=0)
    draw.polygon(core_pts, fill=(180, 140, 255, 230))
    
    # Center Ultra-Bright White/Cyan Nucleus
    draw.ellipse([center - S * 0.045, center - S * 0.045, center + S * 0.045, center + S * 0.045],
                 fill=(255, 255, 255, 255))
    
    # 4. Subtle Outer Precision Calibration Ring (Cybernetic Research Aesthetic)
    # Thin segmented coordinate ring enclosing the emblem comfortably within circular avatar frame
    coord_r = S * 0.44
    draw.ellipse([center - coord_r, center - coord_r, center + coord_r, center + coord_r],
                 outline=(0, 240, 255, 45), width=int(1.5 * supersample))
    
    # 4 Cardinal Tick Marks at cardinal axes
    tick_len = S * 0.02
    for angle in [0, 90, 180, 270]:
        rad = math.radians(angle)
        x1 = center + (coord_r - tick_len) * math.cos(rad)
        y1 = center + (coord_r - tick_len) * math.sin(rad)
        x2 = center + (coord_r + tick_len) * math.cos(rad)
        y2 = center + (coord_r + tick_len) * math.sin(rad)
        draw.line([(x1, y1), (x2, y2)], fill=(0, 240, 255, 120), width=int(2.0 * supersample))
    
    # Composite all layers together
    final_img = Image.alpha_composite(bg, glow_layer)
    final_img = Image.alpha_composite(final_img, geom_layer)
    
    # Downscale with high-quality Lanczos resampling
    final_img = final_img.resize((size, size), Image.Resampling.LANCZOS)
    return final_img

def create_circular_preview(avatar_img):
    """Creates a preview showing how the avatar appears when cropped into a GitHub/Gmail circular profile."""
    size = avatar_img.size[0]
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    # Circle with 10px margin
    margin = int(size * 0.02)
    mask_draw.ellipse([margin, margin, size - margin, size - margin], fill=255)
    
    circle_avatar = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    circle_avatar.paste(avatar_img, (0, 0), mask=mask)
    
    # Draw a sleek circular border
    preview_draw = ImageDraw.Draw(circle_avatar)
    preview_draw.ellipse([margin, margin, size - margin, size - margin], outline=(0, 240, 255, 180), width=3)
    return circle_avatar

if __name__ == "__main__":
    print("Generating ultra-crisp vector-grade avatar...")
    avatar = render_avatar(size=1024, supersample=2)
    
    # Save main master avatar
    out_path_full = "paper/figures/xstarbrain_avatar_vector.png"
    avatar.save(out_path_full, "PNG", optimize=True)
    print(f"Saved master avatar to {out_path_full}")
    
    # Save circular preview
    circle_preview = create_circular_preview(avatar)
    preview_path = "paper/figures/xstarbrain_avatar_circular_preview.png"
    circle_preview.save(preview_path, "PNG", optimize=True)
    print(f"Saved circular preview to {preview_path}")
    
    # Save 512, 256, 128, 64, 32 versions to verify crisp icon scaling
    for dim in [512, 256, 128, 64, 32]:
        scaled = avatar.resize((dim, dim), Image.Resampling.LANCZOS)
        scaled.save(f"paper/figures/xstarbrain_avatar_{dim}x{dim}.png", "PNG", optimize=True)
    print("Successfully generated all resolution scales!")
