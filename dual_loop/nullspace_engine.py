import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class OrthogonalNullspaceProjector(nn.Module):
    """
    Orthogonal Nullspace Synthesis (Gram-Schmidt Latent Projection).
    
    Prevents deep learning representations from falling back into the dense
    prior training attractor manifold when encountering unprecedented concepts or anomalies.
    
    Mathematical Formulation:
        Given known knowledge basis V_known = [m_1, ..., m_M] in R^{D x M} from memory/context:
        1. Compute orthonormal basis Q in R^{D x M} via thin QR decomposition:
           V_known = Q R, where Q^T Q = I_M.
        2. Projection onto known subspace:
           P x = Q (Q^T x)
        3. Orthogonal Nullspace Projection Operator:
           x_perp = (I - P) x = x - Q (Q^T x)
           
        Geometric Guarantee:
           Q^T x_perp = Q^T x - (Q^T Q) Q^T x = Q^T x - I Q^T x = 0
           For any v in span(V_known), <x_perp, v> = 0 identically.
           
    This guarantees that the synthesized prototype occupies an entirely new, unassigned
    orthogonal coordinate in latent space without interfering with existing concepts.
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-7,
        normalize_output: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.normalize_output = normalize_output

    def project_to_nullspace(
        self,
        candidate_vector: torch.Tensor,
        basis_memory: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Projects candidate_vector into the orthogonal nullspace of basis_memory.
        
        Args:
            candidate_vector: [B, D] or [B, 1, D] Novel concept proposal or discrepancy.
            basis_memory: [B, M, D] Memory slots spanning the known manifold.
            
        Returns:
            h_novel: [B, D] or [B, 1, D] Orthogonally purified concept vector.
            telemetry: Detailed orthogonality metrics.
        """
        orig_dim = candidate_vector.dim()
        if orig_dim == 3:
            x = candidate_vector.squeeze(1) # [B, D]
        else:
            x = candidate_vector # [B, D]
            
        B, M, D = basis_memory.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure float32 for high precision QR decomposition regardless of mixed dtypes
        needs_cast = (dtype != torch.float32 and dtype != torch.float64)
        x_calc = x.float()
        basis_calc = basis_memory.to(device=device, dtype=torch.float32)
        
        # Transpose basis to [B, D, M] for QR decomposition
        V = basis_calc.transpose(1, 2) # [B, D, M]
        
        try:
            # Thin QR decomposition: Q is [B, D, M] with orthonormal columns
            Q, _ = torch.linalg.qr(V, mode="reduced")
        except Exception:
            # Fallback if SVD/QR fails: identity projection
            Q = V / (torch.norm(V, dim=1, keepdim=True) + self.eps)
            
        # Compute projection onto known subspace: x_parallel = Q (Q^T x)
        # x is [B, D, 1], Q^T x is [B, M, 1]
        x_expanded = x_calc.unsqueeze(-1) # [B, D, 1]
        alpha = torch.bmm(Q.transpose(1, 2), x_expanded) # [B, M, 1]
        x_parallel = torch.bmm(Q, alpha).squeeze(-1) # [B, D]
        
        # Orthogonal component: x_perp = x - x_parallel
        x_perp = x_calc - x_parallel # [B, D]
        
        # Compute residual norm and original norm
        orig_norm = torch.norm(x_calc, dim=-1, keepdim=True) + self.eps # [B, 1]
        perp_norm = torch.norm(x_perp, dim=-1, keepdim=True) + self.eps # [B, 1]
        
        # Degeneracy check: if candidate was almost entirely in span(V), handle gently
        orthogonality_ratio = (perp_norm / orig_norm).squeeze(-1) # [B]
        
        if self.normalize_output:
            # Scale purified vector to preserve the original energy magnitude
            x_novel = x_perp * (orig_norm / perp_norm)
        else:
            x_novel = x_perp
            
        if needs_cast:
            x_novel = x_novel.to(dtype=dtype)
            
        # Verify maximum inner product against basis Q for diagnostic telemetry
        with torch.no_grad():
            inner_prods = torch.bmm(Q.transpose(1, 2), x_perp.unsqueeze(-1)).squeeze(-1) # [B, M]
            max_leakage = float(torch.max(torch.abs(inner_prods)).item())
            
        telemetry = {
            "max_basis_leakage": max_leakage,
            "orthogonality_ratio": float(orthogonality_ratio.mean().item()),
            "is_strictly_orthogonal": bool(max_leakage < 1e-4)
        }
        
        if orig_dim == 3:
            return x_novel.unsqueeze(1), telemetry
        return x_novel, telemetry

    def forward(
        self,
        candidate_vector: torch.Tensor,
        basis_memory: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.project_to_nullspace(candidate_vector, basis_memory)


class SynapticEligibilityTrace(nn.Module):
    """
    Continuous Synaptic Eligibility Trace & Anti-Hebbian Deflation Buffer.

    Records pre-synaptic and post-synaptic trajectory outer products during
    a forward generation pass:
        E_t = gamma * E_{t-1} + normalize(h_post) (x) normalize(h_pre)^T

    Upon receiving the intrinsic Popperian-Cycle verification signal c in {-1, +1}:
    - c = +1 (Verified): Consolidates the trace into the synaptic delta while
      projecting out components that would interfere with previously verified
      knowledge basis V_known via OrthogonalNullspaceProjector.
    - c = -1 (Falsified): Performs Anti-Hebbian Orthogonal Deflation (-eta * E_t)
      and stores the falsified direction so subsequent passes actively deflect
      into an orthogonal trajectory.
    """
    def __init__(
        self,
        d_model: int,
        max_basis_slots: int = 16,
        gamma: float = 0.85,
        lr_pos: float = 0.05,
        lr_neg: float = 0.08
    ):
        super().__init__()
        self.d_model = d_model
        self.max_basis_slots = max_basis_slots
        self.gamma = gamma
        self.lr_pos = lr_pos
        self.lr_neg = lr_neg

        self.nullspace_projector = OrthogonalNullspaceProjector(
            d_model=d_model,
            normalize_output=False
        )

        # Low-rank eligibility trace factors [r, D] for O(D) efficiency on 8GB VRAM
        self.rank = 8
        self.register_buffer("trace_pre", torch.zeros(self.rank, d_model))
        self.register_buffer("trace_post", torch.zeros(self.rank, d_model))
        self.register_buffer("trace_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("prompt_h_pre", torch.zeros(1, d_model))
        self.register_buffer("last_h_pre", torch.zeros(1, d_model))
        self.register_buffer("last_h_post", torch.zeros(1, d_model))

        # Verified knowledge basis [1, M, D] protected by OrthogonalNullspaceProjector
        self.register_buffer("verified_basis", torch.zeros(1, max_basis_slots, d_model))
        self.register_buffer("basis_count", torch.tensor(0, dtype=torch.long))

        # Deflation memory of recently falsified hidden states [1, K, D]
        self.register_buffer("falsified_directions", torch.zeros(1, 8, d_model))
        self.register_buffer("falsified_count", torch.tensor(0, dtype=torch.long))

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

    @torch.no_grad()
    def record_step(self, h_pre: torch.Tensor, h_post: torch.Tensor) -> None:
        """
        Records a forward activation pair (h_pre, h_post) into the eligibility trace.
        Accepts [B, L, D] or [B, D]. Preserves prompt context in prompt_h_pre and
        integrates decaying temporal eligibility traces trace_pre / trace_post.
        """
        is_prefill = (h_pre.dim() == 3 and h_pre.size(1) > 1)
        if h_pre.dim() == 3:
            pre_vec = h_pre.detach().float().mean(dim=1).mean(dim=0, keepdim=True) if is_prefill else h_pre[:, -1, :].detach().float().mean(dim=0, keepdim=True)
        else:
            pre_vec = h_pre.detach().float().mean(dim=0, keepdim=True)

        if h_post.dim() == 3:
            post_vec = h_post[:, -1, :].detach().float().mean(dim=0, keepdim=True)
        else:
            post_vec = h_post.detach().float().mean(dim=0, keepdim=True)

        pre_norm = F.normalize(pre_vec, p=2, dim=-1)
        post_norm = F.normalize(post_vec, p=2, dim=-1)

        step_count = int(self.trace_ptr.item())
        idx = step_count % self.rank
        self.trace_pre.mul_(self.gamma)
        self.trace_post.mul_(self.gamma)
        self.trace_pre[idx].copy_(pre_norm.squeeze(0).to(self.trace_pre.device))
        self.trace_post[idx].copy_(post_norm.squeeze(0).to(self.trace_post.device))
        self.trace_ptr.add_(1)

        if is_prefill or step_count == 0 or float(self.prompt_h_pre.norm().item()) < 1e-6:
            self.prompt_h_pre.copy_(pre_norm.to(self.prompt_h_pre.device))

        active_r = min(step_count + 1, self.rank)
        trace_pre_mean = self.trace_pre[:active_r].mean(dim=0, keepdim=True)
        trace_post_mean = self.trace_post[:active_r].mean(dim=0, keepdim=True)

        blended_pre = F.normalize(
            0.7 * self.prompt_h_pre + 0.3 * trace_pre_mean.to(self.last_h_pre.device),
            p=2,
            dim=-1
        )
        blended_post = F.normalize(
            0.7 * post_norm.to(self.last_h_post.device) + 0.3 * trace_post_mean.to(self.last_h_post.device),
            p=2,
            dim=-1
        )
        self.last_h_pre.copy_(blended_pre)
        self.last_h_post.copy_(blended_post)

    @torch.no_grad()
    def apply_signed_consolidation(
        self,
        dopamine_sign: float,
        candidate_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Applies signed Hebbian (+1) or Anti-Hebbian (-1) consolidation protected
        by OrthogonalNullspaceProjector.

        Returns:
            delta_vec: [1, D] Nullspace-projected plasticity correction vector.
            telemetry: Diagnostic dictionary.
        """
        device = self.last_h_post.device
        h_cand = (
            candidate_state.detach().float().reshape(-1, self.d_model).mean(dim=0, keepdim=True).to(device)
            if candidate_state is not None
            else self.last_h_post.clone()
        )

        # Project candidate against verified basis if basis is non-empty
        n_basis = int(self.basis_count.item())
        if n_basis > 0:
            active_slots = min(n_basis, self.max_basis_slots)
            basis_slice = self.verified_basis[:, :active_slots, :]
            h_purified, null_tel = self.nullspace_projector.project_to_nullspace(
                h_cand, basis_slice
            )
            # Second-pass Gram-Schmidt re-orthogonalization ("twice is enough")
            # to eliminate float32 drift before normalizing into verified_basis
            h_purified, _ = self.nullspace_projector.project_to_nullspace(
                h_purified, basis_slice
            )
        else:
            h_purified = h_cand
            null_tel = {
                "max_basis_leakage": 0.0,
                "orthogonality_ratio": 1.0,
                "is_strictly_orthogonal": True
            }

        orth_ratio = float(null_tel.get("orthogonality_ratio", 1.0))

        if dopamine_sign >= 0.0:
            # Positive Hebbian consolidation: store in verified basis only if
            # candidate has a non-trivial orthogonal component (prevents amplifying
            # float32 rounding noise when consolidating an already-stored direction).
            if orth_ratio >= 1e-3 and float(h_purified.norm().item()) > 1e-6:
                slot = n_basis % self.max_basis_slots
                if n_basis >= self.max_basis_slots:
                    # Re-project against the M-1 retained slots when wrapping around
                    keep_indices = [i for i in range(self.max_basis_slots) if i != slot]
                    if keep_indices:
                        retained_basis = self.verified_basis[:, keep_indices, :]
                        h_slot, _ = self.nullspace_projector.project_to_nullspace(h_cand, retained_basis)
                        h_slot, _ = self.nullspace_projector.project_to_nullspace(h_slot, retained_basis)
                        if float(h_slot.norm().item()) > 1e-6:
                            self.verified_basis[0, slot].copy_(F.normalize(h_slot.squeeze(0), p=2, dim=-1))
                            self.basis_count.add_(1)
                else:
                    self.verified_basis[0, slot].copy_(F.normalize(h_purified.squeeze(0), p=2, dim=-1))
                    self.basis_count.add_(1)
                delta_vec = self.lr_pos * float(dopamine_sign) * h_purified
            else:
                # Direction is already consolidated in verified_basis
                delta_vec = self.lr_pos * float(dopamine_sign) * h_cand
            mode = "hebbian_consolidation"
        else:
            # Negative Anti-Hebbian Orthogonal Deflation: record falsified direction & deflate
            if float(h_cand.norm().item()) > 1e-6:
                f_idx = int(self.falsified_count.item()) % self.falsified_directions.size(1)
                self.falsified_directions[0, f_idx].copy_(F.normalize(h_cand.squeeze(0), p=2, dim=-1))
                self.falsified_count.add_(1)
            delta_vec = -self.lr_neg * abs(float(dopamine_sign)) * (
                h_purified if orth_ratio >= 1e-3 else h_cand
            )
            mode = "anti_hebbian_deflation"

        telemetry = {
            "consolidation_mode": mode,
            "dopamine_sign": float(dopamine_sign),
            "verified_basis_count": int(self.basis_count.item()),
            "falsified_count": int(self.falsified_count.item()),
            "delta_norm": float(delta_vec.norm().item()),
            **null_tel
        }
        return delta_vec, telemetry

    @torch.no_grad()
    def deflect_from_falsified(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Projects out components aligned with recently falsified trajectories so
        subsequent passes explore an orthogonal reasoning subspace.
        """
        n_false = int(self.falsified_count.item())
        if n_false == 0:
            return x, 0.0

        active_k = min(n_false, self.falsified_directions.size(1))
        false_basis = self.falsified_directions[:, :active_k, :].to(device=x.device, dtype=torch.float32)

        orig_dtype = x.dtype
        x_f = x.float()
        if x_f.dim() == 3:
            # Deflect last token / sequence
            last_tok = x_f[:, -1, :]
            purified, tel = self.nullspace_projector.project_to_nullspace(last_tok, false_basis.expand(x_f.size(0), -1, -1))
            out = x_f.clone()
            out[:, -1, :] = 0.7 * purified + 0.3 * last_tok
            deflection_mag = float((out[:, -1, :] - last_tok).norm(dim=-1).mean().item())
            return out.to(orig_dtype), deflection_mag
        else:
            purified, tel = self.nullspace_projector.project_to_nullspace(x_f, false_basis.expand(x_f.size(0), -1, -1))
            out = 0.7 * purified + 0.3 * x_f
            deflection_mag = float((out - x_f).norm(dim=-1).mean().item())
            return out.to(orig_dtype), deflection_mag

    @torch.no_grad()
    def reset_falsified(self) -> None:
        """Clears transient falsified directions after successful task resolution."""
        self.falsified_directions.zero_()
        self.falsified_count.zero_()

    @torch.no_grad()
    def reset(self) -> None:
        """Resets all episodic traces and nullspace bases to pure baseline."""
        self.reset_falsified()
        self.trace_pre.zero_()
        self.trace_post.zero_()
        self.trace_ptr.zero_()
        self.prompt_h_pre.zero_()
        self.last_h_pre.zero_()
        self.last_h_post.zero_()
        self.verified_basis.zero_()
        self.basis_count.zero_()

    reset_episode = reset


# Alias for backward/unified compatibility
NullspaceMemoryEngine = SynapticEligibilityTrace

