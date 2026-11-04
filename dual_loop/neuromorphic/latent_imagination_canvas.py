import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List


class Progressive2Dto3DImaginationCanvas(nn.Module):
    """
    Progressive 2D-to-3D Human-Like Visual Imagination Organ
    (1D Symbolic -> 2D Retinotopic Sketchpad -> 3D Gaussian/SDF Volumetric Lifting
     -> SO(3) Mental Rotation -> Differentiable 3D-to-2D Splatting Verification).

    Bridges the gap between 1D token sequences and human-like 3D spatial mental
    models inside the Layer 7 neuromorphic transition:
    1. 2D Retinotopic Contour Sketchpad (C_2D in R^{B x H x W}):
       Projects token states onto the orthonormal planar basis (e_1, e_2) extracted
       by DynamicNeuralReplicator.extract_multivector_subspace and splats them onto
       an H x W retinotopic grid with contour sharpening.
    2. 2D-to-3D Volumetric Lifting (M 3D Gaussian/SDF Primitives):
       Lifts salient 2D regions along the orthogonal depth basis e_3 into M 3D
       volumetric primitives parameterized by:
         - 3D centers mu_i in [-1, 1]^3
         - 3D SPD covariance scales s_i in (0.08, 0.65)^3
         - Volumetric density/opacity alpha_i in (0, 1)
         - Signed Distance Field (SDF) zero-isosurface radius r_i
    3. Isometric SO(3) Mental Rotation:
       Applies exact 3D orthogonal rotation matrices R in SO(3) (R^T R = I_3, det R = +1)
       to inspect the 3D mental model from multiple viewpoints, coupled with
       UnitaryTrigonometricRotator in feature space.
    4. 3D-to-2D Splatting Reprojection & Grounding Verification:
       Alpha-composites the 3D primitives back onto 2D, verifies structural
       coherence against C_2D, and injects the verified 3D spatial delta back
       into Layer 7.
    """
    def __init__(
        self,
        d_model: int = 2048,
        grid_size: int = 16,
        n_primitives: int = 8
    ):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.n_primitives = n_primitives

        # Fallback learned 3D subspace basis [3, D] if multivector basis has rank < 3
        self.fallback_basis = nn.Parameter(torch.randn(3, d_model) * 0.02)

        # 2D Retinotopic Sketchpad -> M 3D Primitive Parameters:
        # Each primitive has 8 geometric params: [mu_x, mu_y, mu_z, log_sx, log_sy, log_sz, logit_alpha, sdf_radius]
        self.lift_2d_to_3d = nn.Sequential(
            nn.Linear(grid_size * grid_size + 3, 128),
            nn.GELU(),
            nn.Linear(128, n_primitives * 8)
        )
        nn.init.normal_(self.lift_2d_to_3d[-1].weight, std=0.02)
        nn.init.zeros_(self.lift_2d_to_3d[-1].bias)

        # Bipartite Conformal Rotoreflector:
        # Part 1: Mental rotation angle predictor: [yaw, pitch, roll] in [-pi/2, pi/2]
        self.so3_angle_predictor = nn.Linear(d_model, 3)
        nn.init.normal_(self.so3_angle_predictor.weight, std=0.01)
        nn.init.zeros_(self.so3_angle_predictor.bias)

        # Part 2: Householder reflection plane normal vector n in S^2 and soft parity s in [-1, 1]
        self.reflection_normal_predictor = nn.Linear(d_model, 3)
        nn.init.normal_(self.reflection_normal_predictor.weight, std=0.01)
        nn.init.zeros_(self.reflection_normal_predictor.bias)

        self.parity_gate_predictor = nn.Linear(d_model, 1)
        nn.init.normal_(self.parity_gate_predictor.weight, std=0.01)
        # Initialize bias to +2.0 so soft parity s = tanh(2.0) ≈ +0.96 (default proper rotation)
        nn.init.constant_(self.parity_gate_predictor.bias, 2.0)

        # Part 3: Dynamic Anisotropic Conformal Scale delta in [-0.5, 0.5]^3
        self.scale_predictor = nn.Linear(d_model, 3)
        nn.init.normal_(self.scale_predictor.weight, std=0.01)
        nn.init.zeros_(self.scale_predictor.bias)

        # 6 Canonical polyhedral face normals: Top, Bottom, Front, Back, Left, Right
        canonical_faces = torch.tensor([
            [ 0.0,  0.0,  1.0],  # 0: Top
            [ 0.0,  0.0, -1.0],  # 1: Bottom
            [ 0.0,  1.0,  0.0],  # 2: Front
            [ 0.0, -1.0,  0.0],  # 3: Back
            [-1.0,  0.0,  0.0],  # 4: Left
            [ 1.0,  0.0,  0.0],  # 5: Right
        ], dtype=torch.float32)
        self.register_buffer("canonical_faces", canonical_faces)

        # Back-projection from verified (2D sketchpad + rotated 2D splat + 3D primitives + 6 face orientations) -> R^D
        # Dimensions: grid_size^2 (256) + grid_size^2 (256) + primitives*8 (64) + 6 faces*3 (18) = 594
        self.spatial_grounding_proj = nn.Linear(grid_size * grid_size * 2 + n_primitives * 8 + 18, d_model)
        nn.init.normal_(self.spatial_grounding_proj.weight, std=0.01)
        nn.init.zeros_(self.spatial_grounding_proj.bias)

        # Isometric feature-space mental rotation via UnitaryTrigonometricRotator
        from dual_loop.experimental_cognitive_engine import UnitaryTrigonometricRotator
        self.unitary_rotator = UnitaryTrigonometricRotator(d_model=d_model)

        # Precompute normalized 2D coordinate grid [-1, 1]^2 of shape [H, W, 2]
        coords = torch.linspace(-1.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        grid_xy = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        self.register_buffer("grid_xy", grid_xy)

        # Structured 3D polyhedral spatial anchors [8, 3] for non-degenerate geometric primitive distribution
        polyhedral_anchors = torch.tensor([
            [ 0.55,  0.00,  0.15],
            [ 0.28,  0.48, -0.15],
            [-0.28,  0.48,  0.15],
            [-0.55,  0.00, -0.15],
            [-0.28, -0.48,  0.15],
            [ 0.28, -0.48, -0.15],
            [ 0.00,  0.00,  0.60],
            [ 0.00,  0.00, -0.60],
        ], dtype=torch.float32)
        self.register_buffer("polyhedral_anchors", polyhedral_anchors)

        # Anisotropic 3D scale anchor biases [8, 3]
        scale_anchor = torch.tensor([
            [ 0.40, -0.20, -0.30],
            [-0.20,  0.40, -0.30],
            [ 0.40, -0.20, -0.30],
            [-0.20,  0.40, -0.30],
            [ 0.10,  0.10, -0.20],
            [ 0.10,  0.10, -0.20],
            [-0.10, -0.10,  0.50],
            [-0.10, -0.10,  0.50],
        ], dtype=torch.float32)
        self.register_buffer("scale_anchor", scale_anchor)

        # Visual Ingestion Linear Layer: projects external photo / camera / video frame into R^D
        self.visual_to_latent = nn.Linear(grid_size * grid_size, d_model)
        nn.init.orthogonal_(self.visual_to_latent.weight, gain=0.1)
        nn.init.zeros_(self.visual_to_latent.bias)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for p in self.parameters():
            if p.is_floating_point() and p.dtype != torch.float32:
                p.data = p.data.float()
        for m in self.modules():
            for b_name, b in m._buffers.items():
                if b is not None and b.is_floating_point() and b.dtype != torch.float32:
                    m._buffers[b_name] = b.float()
        return self

    def _ensure_orthonormal_3d_basis(
        self,
        hidden_states: torch.Tensor,
        multivector_basis: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Returns an orthonormal 3D basis E = [e_1, e_2, e_3] of shape [B, 3, D]
        satisfying E @ E^T = I_3 via Gram-Schmidt / QR decomposition.
        """
        B = hidden_states.size(0)
        device = hidden_states.device
        raw_vectors = []

        if multivector_basis is not None:
            for v in multivector_basis[:3]:
                if isinstance(v, torch.Tensor) and v.numel() >= self.d_model:
                    vf = v.detach().to(device=device, dtype=torch.float32)
                    if vf.dim() == 1:
                        vf = vf.unsqueeze(0).expand(B, -1)
                    elif vf.dim() == 3:
                        vf = vf[:, -1, :]
                    raw_vectors.append(vf[:, :self.d_model])

        # Fill remaining basis axes from token trajectory variance or fallback_basis
        h_f = hidden_states.detach().float()
        if h_f.dim() == 3 and h_f.size(1) >= 2 and len(raw_vectors) < 3:
            raw_vectors.append(h_f[:, -1, :] - h_f[:, 0, :])
        if h_f.dim() == 3 and h_f.size(1) >= 3 and len(raw_vectors) < 3:
            mid = h_f.size(1) // 2
            raw_vectors.append(h_f[:, mid, :] - 0.5 * (h_f[:, -1, :] + h_f[:, 0, :]))

        fb = self.fallback_basis.to(device=device, dtype=torch.float32)
        idx = 0
        while len(raw_vectors) < 3:
            raw_vectors.append(fb[idx % 3].unsqueeze(0).expand(B, -1))
            idx += 1

        # Stack [B, D, 3] and perform thin QR orthonormalization
        M_mat = torch.stack(raw_vectors[:3], dim=-1)  # [B, D, 3]
        # Add tiny identity jitter for numerical stability if vectors are collinear
        M_mat = M_mat + 1e-5 * fb.t().unsqueeze(0)
        Q, _ = torch.linalg.qr(M_mat, mode="reduced")  # [B, D, 3]
        return Q.transpose(1, 2)  # [B, 3, D]

    @staticmethod
    def build_so3_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
        """
        Builds exact 3D rotation matrices R in SO(3) [B, 3, 3] from Euler angles
        [yaw, pitch, roll] in radians via R = R_z(yaw) @ R_y(pitch) @ R_x(roll).
        Guarantees R^T R = I_3 and det(R) = +1.
        """
        yaw = angles[:, 0]
        pitch = angles[:, 1]
        roll = angles[:, 2]

        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cr, sr = torch.cos(roll), torch.sin(roll)

        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)

        Rz = torch.stack([
            torch.stack([cy, -sy, zeros], dim=-1),
            torch.stack([sy,  cy, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
        ], dim=-2)

        Ry = torch.stack([
            torch.stack([cp, zeros, sp], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sp, zeros, cp], dim=-1)
        ], dim=-2)

        Rx = torch.stack([
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cr, -sr], dim=-1),
            torch.stack([zeros, sr,  cr], dim=-1)
        ], dim=-2)

        return torch.bmm(torch.bmm(Rz, Ry), Rx)  # [B, 3, 3]

    @staticmethod
    def build_householder_reflector(normal: torch.Tensor, parity: torch.Tensor) -> torch.Tensor:
        """
        Builds generalized Householder reflection / pinor matrices H in O(3) [B, 3, 3].
        H(n, s) = I_3 - (1 - s) * (n n^T) / (||n||^2 + eps)
        When s = +1 (proper parity): H = I_3 (pure rotation, det = +1).
        When s = -1 (improper parity): H = I - 2 n n^T (pure Householder reflection across plane n, det = -1).
        Smoothly interpolates det(H) between +1 and -1 for end-to-end differentiability.
        """
        B = normal.size(0)
        device = normal.device
        dtype = normal.dtype

        # Normalize normal vector to unit sphere S^2
        n_norm = F.normalize(normal.float(), p=2, dim=-1, eps=1e-8)  # [B, 3]
        # Outer product n n^T: [B, 3, 3]
        nnT = torch.bmm(n_norm.unsqueeze(-1), n_norm.unsqueeze(1))

        # Identity I_3 [B, 3, 3]
        I3 = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

        # Parity factor (1 - s): s in [-1, 1], so (1 - s) in [0, 2]
        factor = (1.0 - parity.float().view(B, 1, 1))  # [B, 1, 1]

        H = I3 - factor * nnT
        return H.to(dtype=dtype)

    def project_1d_to_2d_sketchpad(
        self,
        hidden_states: torch.Tensor,
        basis_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stage 1: Projects 1D token sequence [B, L, D] onto the 2D planar basis (e_1, e_2)
        to form the 2D Retinotopic Contour Sketchpad C_2D [B, H, W], and extracts
        mean depth moments along e_3 [B, 3] and raw 3D token coordinates [B, L, 3].
        """
        if hidden_states.dim() == 2:
            h_seq = hidden_states.unsqueeze(1)
        else:
            h_seq = hidden_states

        B, L, D = h_seq.shape
        H = W = self.grid_size
        h_f = h_seq.float()
        basis_f = basis_3d.to(device=h_f.device, dtype=torch.float32)  # [B, 3, D]

        # Project tokens onto (e_1, e_2, e_3): coords_3d is [B, L, 3]
        coords_3d = torch.bmm(h_f, basis_f.transpose(1, 2))  # [B, L, 3]
        coords_norm = torch.tanh(coords_3d / math.sqrt(float(D) * 0.05 + 1.0))  # [B, L, 3] in (-1, 1)

        u = coords_norm[:, :, 0].unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        v = coords_norm[:, :, 1].unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        z = coords_norm[:, :, 2]                              # [B, L]

        # Token salience weight from L2 norm
        salience = torch.sigmoid(h_f.norm(dim=-1) / math.sqrt(float(D))).unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]

        grid = self.grid_xy.to(device=h_f.device, dtype=torch.float32)  # [H, W, 2]
        gx = grid[:, :, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        gy = grid[:, :, 1].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Gaussian splat of 1D tokens onto 2D retinotopic plane (e_1, e_2)
        dist_sq = (gx - u) ** 2 + (gy - v) ** 2  # [B, L, H, W]
        sigma_2d = 0.22
        splat_2d = (salience * torch.exp(-dist_sq / (2.0 * sigma_2d ** 2))).sum(dim=1)  # [B, H, W]

        # Normalize sketchpad to [0, 1]
        max_val = splat_2d.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        sketchpad_2d = splat_2d / max_val

        # Depth summary moments along e_3: [mean_z, std_z, max_abs_z]
        z_mean = z.mean(dim=-1, keepdim=True)
        z_std = z.std(dim=-1, unbiased=False, keepdim=True) if L > 1 else torch.zeros_like(z_mean)
        z_max = z.abs().amax(dim=-1, keepdim=True)
        depth_moments = torch.cat([z_mean, z_std, z_max], dim=-1)  # [B, 3]

        return sketchpad_2d, depth_moments, coords_3d

    def lift_2d_to_3d_primitives(
        self,
        sketchpad_2d: torch.Tensor,
        depth_moments: torch.Tensor,
        coords_3d: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 2: Lifts the 2D retinotopic sketchpad + e_3 depth moments + 3D token
        coordinates into M 3D Gaussian / SDF volumetric primitives.
        """
        B = sketchpad_2d.size(0)
        w_dev = self.lift_2d_to_3d[0].weight.device
        flat_2d = sketchpad_2d.reshape(B, -1).to(device=w_dev, dtype=torch.float32)
        dm = depth_moments.to(device=w_dev, dtype=torch.float32)
        lift_in = torch.cat([flat_2d, dm], dim=-1)  # [B, H*W + 3]

        raw_params = self.lift_2d_to_3d(lift_in).view(B, self.n_primitives, 8)

        # 1. 3D centers mu_i in [-1, 1]^3 derived from token centroids + polyhedral anchors
        if coords_3d is not None and coords_3d.size(1) > 0:
            c_pool = F.adaptive_avg_pool1d(coords_3d.transpose(1, 2), self.n_primitives).transpose(1, 2)
            c_pool = c_pool.to(device=w_dev, dtype=torch.float32)
            token_influence = 0.25 * torch.tanh(c_pool)
        else:
            token_influence = torch.zeros(B, self.n_primitives, 3, device=w_dev)

        anchor_offsets = self.polyhedral_anchors.unsqueeze(0).expand(B, -1, -1).to(device=w_dev)
        centers_3d = 0.65 * torch.tanh(token_influence + anchor_offsets + raw_params[:, :, 0:3])  # [B, M, 3]

        # 2. 3D anisotropic positive-definite scales s_i in [0.08, 0.18]^3
        scale_bias = self.scale_anchor.unsqueeze(0).expand(B, -1, -1).to(device=w_dev)
        scales_3d = 0.08 + 0.10 * torch.sigmoid(raw_params[:, :, 3:6] + scale_bias)  # [B, M, 3]

        # 3. Opacity / density alpha_i in [0.10, 0.95]
        opacities = 0.10 + 0.85 * torch.sigmoid(raw_params[:, :, 6] + 0.5)    # [B, M]

        # 4. Signed Distance Field (SDF) sphere/ellipsoid radius r_i in [0.10, 0.55]
        sdf_radii = 0.10 + 0.45 * torch.sigmoid(raw_params[:, :, 7])         # [B, M]

        # Construct diagonal 3D covariance matrices Sigma_i in SPD(3) [B, M, 3, 3]
        cov_3d = torch.diag_embed(scales_3d ** 2)

        return {
            "centers_3d": centers_3d,
            "scales_3d": scales_3d,
            "cov_3d": cov_3d,
            "opacities": opacities,
            "sdf_radii": sdf_radii,
            "raw_params": raw_params
        }

    def splat_3d_to_2d(
        self,
        centers_3d: torch.Tensor,
        cov_3d: torch.Tensor,
        opacities: torch.Tensor,
        grid_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Stage 4: Differentiable depth-sorted orthographic 3D-to-2D Gaussian splatting
        with front-to-back alpha compositing and depth shading onto [-1, 1]^{H x W}.
        """
        B, M, _ = centers_3d.shape
        H = W = grid_size if grid_size is not None else self.grid_size
        device = centers_3d.device

        # Sort primitives front-to-back by depth z (ascending z)
        z_vals = centers_3d[:, :, 2]  # [B, M]
        sort_idx = torch.argsort(z_vals, dim=1)  # [B, M]

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, M)
        mu_sorted = centers_3d[batch_idx, sort_idx]       # [B, M, 3]
        cov_sorted = cov_3d[batch_idx, sort_idx]          # [B, M, 3, 3]
        alpha_sorted = opacities[batch_idx, sort_idx]     # [B, M]

        # Extract 2D marginal covariance Sigma_2D = cov_sorted[:, :, :2, :2] + 1e-4 I_2
        cov_2d = cov_sorted[:, :, :2, :2] + 1e-4 * torch.eye(2, device=device).view(1, 1, 2, 2)
        det_2d = (cov_2d[:, :, 0, 0] * cov_2d[:, :, 1, 1] - cov_2d[:, :, 0, 1] * cov_2d[:, :, 1, 0]).clamp(min=1e-6)
        inv_00 = cov_2d[:, :, 1, 1] / det_2d
        inv_11 = cov_2d[:, :, 0, 0] / det_2d
        inv_01 = -cov_2d[:, :, 0, 1] / det_2d

        if H == self.grid_size:
            grid = self.grid_xy.to(device=device, dtype=torch.float32)  # [H, W, 2]
        else:
            coords = torch.linspace(-1.0, 1.0, H, device=device)
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]

        dx = grid[:, :, 0].view(1, 1, H, W) - mu_sorted[:, :, 0].view(B, M, 1, 1)
        dy = grid[:, :, 1].view(1, 1, H, W) - mu_sorted[:, :, 1].view(B, M, 1, 1)

        mahalanobis = (
            inv_00.view(B, M, 1, 1) * (dx ** 2)
            + 2.0 * inv_01.view(B, M, 1, 1) * dx * dy
            + inv_11.view(B, M, 1, 1) * (dy ** 2)
        )
        gauss_2d = torch.exp(-0.5 * mahalanobis.clamp(min=0.0, max=50.0))  # [B, M, H, W]

        # Depth shading: primitives closer to camera (higher z) are brighter, rear are darker
        z_clamped = mu_sorted[:, :, 2].clamp(-0.85, 0.85)
        shading = (0.35 + 0.65 * ((z_clamped + 0.85) / 1.70)).view(B, M, 1, 1)

        # Front-to-back alpha compositing
        alpha_map = (alpha_sorted.view(B, M, 1, 1) * gauss_2d).clamp(min=0.0, max=0.99)
        transmittance = torch.cumprod(1.0 - alpha_map + 1e-6, dim=1)
        # Shift transmittance right by 1 (T_0 = 1.0)
        T_shifted = torch.cat([torch.ones(B, 1, H, W, device=device), transmittance[:, :-1, :, :]], dim=1)

        rendered_2d = (T_shifted * alpha_map * shading).sum(dim=1).clamp(0.0, 1.0)  # [B, H, W]
        return rendered_2d

    def evaluate_sdf_grid(
        self,
        centers_3d: torch.Tensor,
        sdf_radii: torch.Tensor,
        res: int = 8
    ) -> torch.Tensor:
        """
        Evaluates the smooth minimum (union) Signed Distance Field (SDF) on a 3D
        voxel grid [B, res, res, res]:
            SDF(p) = -1/k * log sum_i exp(-k * (||p - mu_i||_2 - r_i))
        """
        device = centers_3d.device
        B, M, _ = centers_3d.shape
        lin = torch.linspace(-1.0, 1.0, res, device=device)
        zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
        pts = torch.stack([xx, yy, zz], dim=-1).view(1, -1, 1, 3)  # [1, V, 1, 3]

        mu = centers_3d.unsqueeze(1)  # [B, 1, M, 3]
        r = sdf_radii.unsqueeze(1)    # [B, 1, M]
        dists = torch.norm(pts - mu, dim=-1) - r  # [B, V, M]

        # Smooth union (softmin) with k=8.0
        k_smooth = 8.0
        sdf_vals = -torch.logsumexp(-k_smooth * dists, dim=-1) / k_smooth  # [B, V]
        return sdf_vals.view(B, res, res, res)

    def forward(
        self,
        hidden_states: torch.Tensor,
        multivector_basis: Optional[List[torch.Tensor]] = None,
        mental_rotation_angles: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes the full Progressive 1D -> 2D Retinotopic Sketchpad -> 3D Volumetric
        Lifting -> SO(3) Mental Rotation -> 3D-to-2D Splatting Verification pipeline.

        Returns:
            grounded_states: [B, L, D] or [B, D] spatially grounded hidden states.
            telemetry: Diagnostic dictionary containing 2D sketchpad, 3D primitives,
                       SO(3) rotation metrics, and 2D<->3D consistency score.
        """
        orig_dim = hidden_states.dim()
        if orig_dim == 2:
            h_seq = hidden_states.unsqueeze(1)
        else:
            h_seq = hidden_states

        B, L, D = h_seq.shape
        w_dev = self.so3_angle_predictor.weight.device

        # 1. Orthonormal 3D basis (e_1, e_2, e_3) via Gram-Schmidt QR
        basis_3d = self._ensure_orthonormal_3d_basis(h_seq, multivector_basis)  # [B, 3, D]

        # 2. Project 1D token sequence onto 2D Retinotopic Contour Sketchpad (e_1, e_2)
        sketchpad_2d, depth_moments, coords_3d = self.project_1d_to_2d_sketchpad(h_seq, basis_3d)  # [B, H, W], [B, 3], [B, L, 3]

        # 3. Lift 2D Sketchpad + depth moments + 3D token coordinates into M 3D Gaussian/SDF primitives
        prim_3d = self.lift_2d_to_3d_primitives(sketchpad_2d, depth_moments, coords_3d)
        centers_3d = prim_3d["centers_3d"]    # [B, M, 3]
        cov_3d = prim_3d["cov_3d"]            # [B, M, 3, 3]
        opacities = prim_3d["opacities"]      # [B, M]
        sdf_radii = prim_3d["sdf_radii"]      # [B, M]

        # 4. Bipartite Conformal Rotoreflector: ConfO(3) = R^+ x O(3)
        h_pool = h_seq[:, -1, :].to(device=w_dev, dtype=torch.float32)
        if mental_rotation_angles is None:
            # Predict mental inspection angles in [-pi, pi]
            angles = math.pi * torch.tanh(self.so3_angle_predictor(h_pool))  # [B, 3]
        else:
            angles = mental_rotation_angles.to(device=w_dev, dtype=torch.float32)

        # Part 1: SO(3) Rotation
        R_so3 = self.build_so3_rotation_matrix(angles)  # [B, 3, 3]

        # Part 2: Householder Reflection Plane Normal n and Soft Parity s in [-1, 1]
        n_raw = self.reflection_normal_predictor(h_pool)  # [B, 3]
        s_parity = torch.tanh(self.parity_gate_predictor(h_pool))  # [B, 1]
        H_ref = self.build_householder_reflector(n_raw, s_parity)  # [B, 3, 3]

        # Compose Rotoreflector: M_rotoref = H @ R in O(3)
        M_rotoref = torch.bmm(H_ref, R_so3)  # [B, 3, 3]

        # Part 3: Dynamic Anisotropic Conformal Scale delta in [-0.4, 0.4]^3
        scale_delta = 0.40 * torch.tanh(self.scale_predictor(h_pool))  # [B, 3]
        scale_diag = torch.exp(scale_delta)  # [B, 3]
        S_scale = torch.diag_embed(scale_diag)  # [B, 3, 3]

        # Full Conformal Transformation Matrix M_total = S @ M_rotoref
        M_total = torch.bmm(S_scale, M_rotoref)  # [B, 3, 3]

        # Transform 3D centers: mu_trans = mu @ M_total^T
        centers_rot = torch.bmm(centers_3d, M_total.transpose(1, 2))  # [B, M, 3]

        # Transform 3D covariances: Sigma_trans = M_total @ Sigma_i @ M_total^T
        M_exp = M_total.unsqueeze(1).expand(B, self.n_primitives, 3, 3).reshape(B * self.n_primitives, 3, 3)
        cov_flat = cov_3d.reshape(B * self.n_primitives, 3, 3)
        cov_rot = torch.bmm(torch.bmm(M_exp, cov_flat), M_exp.transpose(1, 2)).view(B, self.n_primitives, 3, 3)

        # Transform 6 canonical face normals: v'_faces = v_faces @ M_rotoref^T
        canon_faces = self.canonical_faces.unsqueeze(0).expand(B, -1, -1).to(device=w_dev)  # [B, 6, 3]
        faces_rot = torch.bmm(canon_faces, M_rotoref.transpose(1, 2))  # [B, 6, 3]
        faces_flat = faces_rot.reshape(B, 18)  # [B, 18]

        # 5. Differentiable 3D-to-2D Splatting (canonical view & mentally transformed view)
        splat_canonical_2d = self.splat_3d_to_2d(centers_3d, cov_3d, opacities)  # [B, H, W]
        splat_rotated_2d = self.splat_3d_to_2d(centers_rot, cov_rot, opacities)  # [B, H, W]

        # 6. Evaluate 3D Signed Distance Field (SDF) voxel volume [B, 8, 8, 8]
        sdf_volume = self.evaluate_sdf_grid(centers_rot, sdf_radii, res=8)

        # 7. Compute 2D <-> 3D Reprojection Consistency Score
        sk_flat = F.normalize(sketchpad_2d.to(w_dev).reshape(B, -1), p=2, dim=-1)
        sp_flat = F.normalize(splat_canonical_2d.reshape(B, -1), p=2, dim=-1)
        consistency_2d_3d = F.cosine_similarity(sk_flat, sp_flat, dim=-1)  # [B]

        # 8. Synthesize verified 3D spatial grounding delta back into R^D
        #    and apply isometric feature-space rotation via UnitaryTrigonometricRotator
        rot_flat = splat_rotated_2d.reshape(B, -1)
        prim_flat = prim_3d["raw_params"].reshape(B, -1)
        # Dimensions: 256 + 256 + 64 + 18 = 594
        ground_in = torch.cat([sk_flat, rot_flat, prim_flat, faces_flat], dim=-1)
        delta_3d_raw = self.spatial_grounding_proj(ground_in)  # [B, D]
        delta_3d, givens_angles = self.unitary_rotator(delta_3d_raw, h_pool)

        # Gate by 2D-3D consistency score
        gate_3d = torch.sigmoid(4.0 * (consistency_2d_3d - 0.1)).unsqueeze(-1)  # [B, 1]
        gated_delta = (0.04 * gate_3d * delta_3d).to(device=h_seq.device, dtype=h_seq.dtype)

        grounded_seq = h_seq + gated_delta.unsqueeze(1)

        # Verify O(3) orthogonality ||M_rotoref^T M_rotoref - I||_F and det(M_rotoref)
        with torch.no_grad():
            eye3 = torch.eye(3, device=w_dev).unsqueeze(0)
            so3_ortho_err = float(torch.norm(torch.bmm(R_so3.transpose(1, 2), R_so3) - eye3).item())
            o3_ortho_err = float(torch.norm(torch.bmm(M_rotoref.transpose(1, 2), M_rotoref) - eye3).item())
            so3_det = float(torch.linalg.det(R_so3).mean().item())
            rotoref_det = float(torch.linalg.det(M_rotoref).mean().item())
            basis_ortho_err = float(
                torch.norm(torch.bmm(basis_3d, basis_3d.transpose(1, 2)) - torch.eye(3, device=basis_3d.device).unsqueeze(0)).item()
            )

        telemetry = {
            "imagination_stage": "1D_to_2D_sketchpad_to_3D_volumetric_ConfO3",
            "grid_size_2d": self.grid_size,
            "n_3d_primitives": self.n_primitives,
            "basis_rank_3d": 3,
            "basis_orthogonality_error": basis_ortho_err,
            "so3_orthogonality_error": so3_ortho_err,
            "o3_orthogonality_error": o3_ortho_err,
            "so3_determinant": so3_det,
            "so3_det": so3_det,
            "rotoreflector_determinant": rotoref_det,
            "soft_parity": float(s_parity.mean().item()),
            "mental_rotation_angles_rad": [round(float(x), 4) for x in angles[0].detach().cpu().tolist()],
            "conformal_scales": [round(float(x), 4) for x in scale_diag[0].detach().cpu().tolist()],
            "unitary_givens_mean_angle_rad": float(givens_angles.abs().mean().item()),
            "consistency_2d_to_3d": float(consistency_2d_3d.mean().item()),
            "sdf_isosurface_occupancy": float((sdf_volume < 0.0).float().mean().item()),
            "faces_transformed": faces_rot.detach(),
            "basis_3d": basis_3d.detach(),
            "sketchpad_2d": sketchpad_2d.detach(),
            "splat_canonical_2d": splat_canonical_2d.detach(),
            "splat_rotated_2d": splat_rotated_2d.detach(),
            "centers_3d": centers_rot.detach(),
            "scales_3d": prim_3d["scales_3d"].detach(),
            "opacities_3d": opacities.detach(),
            "sdf_volume_3d": sdf_volume.detach()
        }

        if orig_dim == 2:
            return grounded_seq.squeeze(1), telemetry
        return grounded_seq, telemetry

    @staticmethod
    def render_ascii_2d(grid_2d: torch.Tensor) -> str:
        """
        Renders a [H, W] 2D sketchpad or 3D splat tensor into an ASCII heatmap.
        """
        shades = " .:-=+*#%@"
        g = grid_2d.detach().cpu().float()
        if g.dim() == 3:
            g = g[0]
        g_min, g_max = float(g.min().item()), float(g.max().item())
        span = max(1e-6, g_max - g_min)
        lines = []
        for row in g:
            line = "".join(shades[min(len(shades) - 1, int(((float(val) - g_min) / span) * (len(shades) - 1)))] for val in row)
            lines.append(line)
        return "\n".join(lines)

    def generate_mental_video_trajectory(
        self,
        hidden_states: torch.Tensor,
        n_frames: int = 16,
        fps: int = 8,
        save_gif_path: Optional[str] = None,
        multivector_basis: Optional[List[torch.Tensor]] = None,
        render_res: int = 64
    ) -> Dict[str, Any]:
        """
        Stage 5 (Video Imagination): Simulates a continuous 4D spatio-temporal mental video
        trajectory by rolling out the 3D Gaussian/SDF volumetric primitives over T frames
        along a continuous SO(3) geodesic manifold R(t) in SO(3).

        For each frame t in 0..T-1:
        1. R(t) = exp(t/T * omega_hat) preserves volume (det R(t) = +1.0) and orthogonality.
        2. Centers mu_i(t) and covariance Sigma_i(t) rotate in 3D mental space.
        3. Differentiable 3D-to-2D splatting projects the rotated primitives onto the retinotopic canvas.
        4. Frame-to-frame temporal coherence Delta_t is tracked to measure motion smoothness.
        5. Saves an animated GIF if save_gif_path is provided.
        """
        orig_dim = hidden_states.dim()
        if orig_dim == 2:
            h_seq = hidden_states.unsqueeze(1)
        else:
            h_seq = hidden_states

        B, L, D = h_seq.shape
        w_dev = self.so3_angle_predictor.weight.device

        # 1. 3D Orthonormal basis
        basis_3d = self._ensure_orthonormal_3d_basis(h_seq, multivector_basis)

        # 2. 2D Retinotopic Sketchpad
        sketchpad_2d, depth_moments, coords_3d = self.project_1d_to_2d_sketchpad(h_seq, basis_3d)

        # 3. 3D Primitives derived from token coordinates + polyhedral anchors
        prim_3d = self.lift_2d_to_3d_primitives(sketchpad_2d, depth_moments, coords_3d)
        centers_3d = prim_3d["centers_3d"]  # [B, M, 3]
        cov_3d = prim_3d["cov_3d"]          # [B, M, 3, 3]
        opacities = prim_3d["opacities"]    # [B, M]

        # 4. Predict angular velocity vector omega in [-pi, pi]^3 with rich compound 3D tilt
        h_pool = h_seq[:, -1, :].to(device=w_dev, dtype=torch.float32)
        base_omega = math.pi * torch.tanh(self.so3_angle_predictor(h_pool))  # [B, 3]
        omega_bias = torch.tensor([0.55, 0.95, 0.35], device=w_dev).unsqueeze(0)
        omega_effective = 0.5 * base_omega + 0.5 * math.pi * F.normalize(omega_bias, p=2, dim=-1)

        frames_2d = []
        frame_deltas = []
        pixel_diffs = []
        so3_dets = []
        so3_ortho_errs = []
        prev_splat = None
        prev_pil = None

        for t in range(n_frames):
            frac = float(t) / float(max(1, n_frames))
            # Smooth full 360-degree rotation around model's predicted 3D inspection axis
            yaw = 2.0 * math.pi * frac
            pitch = float(omega_effective[0, 1].item())
            roll = float(omega_effective[0, 2].item())
            angles_t = torch.tensor([[yaw, pitch, roll]], device=w_dev)
            R_t = self.build_so3_rotation_matrix(angles_t)  # [B, 3, 3]

            # Rotate 3D centers mu(t) = mu @ R(t)^T and covariance Sigma(t) = R(t) @ Sigma @ R(t)^T
            centers_t = torch.bmm(centers_3d, R_t.transpose(1, 2))  # [B, M, 3]
            R_t_expand = R_t.unsqueeze(1).expand(-1, self.n_primitives, -1, -1)  # [B, M, 3, 3]
            cov_t = torch.matmul(torch.matmul(R_t_expand, cov_3d), R_t_expand.transpose(-2, -1))

            # Splat rotated 3D primitives onto 2D canvas at render_res resolution
            splat_t = self.splat_3d_to_2d(centers_t, cov_t, opacities, grid_size=render_res)  # [B, render_res, render_res]
            frames_2d.append(splat_t[0].detach().cpu())

            # Verification metrics
            with torch.no_grad():
                det_val = float(torch.linalg.det(R_t).mean().item())
                ortho_val = float(torch.norm(torch.bmm(R_t.transpose(1, 2), R_t) - torch.eye(3, device=w_dev).unsqueeze(0)).item())
                so3_dets.append(round(det_val, 4))
                so3_ortho_errs.append(ortho_val)

            if prev_splat is not None:
                delta = float(torch.norm(splat_t[0] - prev_splat).item()) / (render_res * render_res)
                frame_deltas.append(round(delta, 5))
            else:
                frame_deltas.append(0.0)
            prev_splat = splat_t[0].detach()

        # Save animated GIF if path given
        gif_saved = False
        if save_gif_path:
            try:
                from PIL import Image
                import numpy as np

                pil_frames = []
                for f_t in frames_2d:
                    f_np = (f_t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(f_np, mode="L").resize((128, 128), Image.Resampling.BILINEAR)
                    pil_frames.append(img)
                    if prev_pil is not None:
                        diff = float(np.mean(np.abs(np.array(img).astype(int) - np.array(prev_pil).astype(int))))
                        pixel_diffs.append(round(diff, 2))
                    prev_pil = img

                if pil_frames:
                    from pathlib import Path
                    Path(save_gif_path).parent.mkdir(parents=True, exist_ok=True)
                    pil_frames[0].save(
                        save_gif_path,
                        save_all=True,
                        append_images=pil_frames[1:],
                        duration=int(1000 / max(1, fps)),
                        loop=0
                    )
                    gif_saved = True
            except Exception as e:
                import logging
                logging.getLogger("XStar").warning(f"Failed to save animated mental video GIF: {e}")

        return {
            "n_frames": n_frames,
            "fps": fps,
            "gif_saved": gif_saved,
            "gif_path": str(save_gif_path) if gif_saved else None,
            "frame_deltas": frame_deltas,
            "mean_frame_delta": round(float(sum(frame_deltas) / max(1, len(frame_deltas))), 5),
            "pixel_diffs": pixel_diffs,
            "mean_pixel_diff": round(float(sum(pixel_diffs) / max(1, len(pixel_diffs))), 2) if pixel_diffs else 0.0,
            "so3_dets": so3_dets,
            "so3_orthogonality_errors": so3_ortho_errs,
            "mean_orthogonality_error": float(sum(so3_ortho_errs) / max(1, len(so3_ortho_errs))),
            "ascii_first_frame": self.render_ascii_2d(frames_2d[0]),
            "ascii_mid_frame": self.render_ascii_2d(frames_2d[n_frames // 2]),
            "ascii_last_frame": self.render_ascii_2d(frames_2d[-1]),
        }

    def ingest_external_visual_frames(
        self,
        visual_input: torch.Tensor,
        target_hidden_states: Optional[torch.Tensor] = None,
        adapt_online: bool = True,
        lr_visual: float = 0.05
    ) -> Dict[str, Any]:
        """
        External Visual Perception Ingestion (Camera / Photo / Video Frame).
        Projects an external visual image tensor into the model's shared latent space R^D
        via linear projection, generates the model's internal 3D mental imagination of it,
        and executes Analysis-by-Synthesis verification.
        """
        device = self.visual_to_latent.weight.device

        # Normalize visual input shape to [B, H, W]
        v_in = visual_input.to(device=device, dtype=torch.float32)
        if v_in.dim() == 2:
            v_in = v_in.unsqueeze(0)  # [1, H, W]
        elif v_in.dim() == 4:
            # [B, C, H, W] -> convert to grayscale [B, H, W]
            if v_in.size(1) == 3:
                v_in = 0.299 * v_in[:, 0] + 0.587 * v_in[:, 1] + 0.114 * v_in[:, 2]
            else:
                v_in = v_in[:, 0]

        B, H, W = v_in.shape
        # Interpolate to canvas grid_size [B, grid_size, grid_size]
        if H != self.grid_size or W != self.grid_size:
            v_grid = F.interpolate(v_in.unsqueeze(1), size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False).squeeze(1)
        else:
            v_grid = v_in

        v_grid_norm = (v_grid - v_grid.amin(dim=(-2, -1), keepdim=True)) / (v_grid.amax(dim=(-2, -1), keepdim=True) - v_grid.amin(dim=(-2, -1), keepdim=True) + 1e-6)

        # 1. Linear Projection from Visual Manifold -> Latent Space R^D
        v_flat = v_grid_norm.reshape(B, self.grid_size * self.grid_size)
        h_visual = self.visual_to_latent(v_flat)  # [B, D]

        # 2. Generate 3D Mental Model from Visual Latent State
        grounded_states, telem = self.forward(h_visual.unsqueeze(1))
        splat_reconstruction = telem["splat_canonical_2d"]  # [B, H, W]

        # 3. Analysis-by-Synthesis Verification
        recon_err = float(F.mse_loss(splat_reconstruction.to(device), v_grid_norm).item())
        cos_sim = 1.0
        if target_hidden_states is not None:
            t_norm = F.normalize(target_hidden_states.reshape(B, -1).to(device), p=2, dim=-1)
            h_norm = F.normalize(h_visual, p=2, dim=-1)
            cos_sim = float(F.cosine_similarity(h_norm, t_norm, dim=-1).mean().item())

        # 4. Online Hebbian / Synaptic Alignment of Visual Projection
        if adapt_online and target_hidden_states is not None:
            with torch.no_grad():
                err_latent = (t_norm - h_norm)  # [B, D]
                dW = lr_visual * torch.matmul(err_latent.t(), v_flat) / max(1, B)
                self.visual_to_latent.weight.add_(dW.to(dtype=self.visual_to_latent.weight.dtype))

        return {
            "visual_ingested": True,
            "visual_grid_resolution": (self.grid_size, self.grid_size),
            "h_visual": h_visual.detach(),
            "reconstruction_error": recon_err,
            "latent_alignment_cosine": cos_sim,
            "imagination_telemetry": telem,
            "ascii_visual_input": self.render_ascii_2d(v_grid_norm[0]),
            "ascii_mental_reconstruction": self.render_ascii_2d(splat_reconstruction[0]),
        }
