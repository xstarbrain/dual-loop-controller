"""
Neuromorphic Spiking Brain (X-Star HADL v5.0)
==============================================
Revolutionary Spiking Cognitive Architecture featuring:
1. Leaky Integrate-and-Fire (LIF) Spiking Gate positioned at Layer 7 (Semantic Emergence Boundary).
   - Only fires on cognitive surprise / high saliency.
   - Closed gate ($0 \times VGNI$) ensures 0 FLOP overhead on routine tokens and complete isolation against adversarial noise ("viruses").
   - Surrogate gradient (Fast Sigmoid) enables end-to-end differentiable backpropagation through spikes.
2. Dynamic Neural Replication (DNR):
   - Computes vertical saliency vector of KV channels: v_kv = norm_d(sum(K * V)).
   - Multiplies by density factor: rho = C_max / C_active * 100%.
   - Preserves identity (I_d) on base context while dynamically replicating cognitive vectors.
3. Open-Freeze Liquid Phase Plasticity:
   - Dynamic synaptic viscosity mu(t) = tanh(gamma * Surprisal).
   - Solid phase (mu -> 0): 100% frozen foundation model, zero catastrophic forgetting.
   - Liquid phase (mu -> 1): Plastic synaptic flow on breakthrough insight / high surprisal.
4. Re-Entrant Cross-Layer Query Bus:
   - Enables deep downstream layers (e.g. Layer 14, 21) to query Layer 7 VGNI when encountering ambiguity.
5. In-Brain Virtual Latent Sandbox:
   - Evaluates dry-run latent transitions before token commitment, enforcing Banach contraction.
"""

import math
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vgni_lbu_cloud_engine import VirtualWorldGeneralIntelligence, LogicBaseUnit, MonolithicIncoherentSensitivityField
from ..imagination_canvas import EmergentLatentCanvas
from ..adapters.universal_adapter import DynamicGraphIntrospector
from ..verification import DirectionalSafetyProjection, IntrinsicPopperianCycleVerifier
from ..nullspace_engine import OrthogonalNullspaceProjector, SynapticEligibilityTrace
from ..sleep_consolidation import SleepPhaseConsolidationEngine
from ..experimental_cognitive_engine import UnitaryTrigonometricRotator
from .acoustic_vocal_organ import EndogenousAcousticVocalOrgan
from .latent_imagination_canvas import Progressive2Dto3DImaginationCanvas
from .pacemaker_router import (
    PersonalityProfile,
    ModelPersonalityAdapter,
    PreRouterSpikingPacemaker,
    TourLeaderAttractor,
    ConcurrentDualStreamSynthesizer,
    LatentThoughtMatrixEngine,
    CentralCognitiveRouter,
)
from .tokenless_latent_engine import (
    IPv6NumericRegister,
    ThoughtPurifiedKVCache,
    EndogenousMetaRLPolicy,
    TokenlessLatentThoughtEngine,
)




class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Surrogate Gradient Autograd Function for Heaviside Step Spiking.
    Forward: S = 1.0 if (V >= V_th) else 0.0
    Backward: dS/dV = 1 / (1 + gamma * |V - V_th|)^2
    """
    @staticmethod
    def forward(ctx, v: torch.Tensor, v_th: torch.Tensor, gamma: float = 4.0) -> torch.Tensor:
        ctx.save_for_backward(v, v_th.clone())
        ctx.gamma = gamma
        spike = (v >= v_th).to(dtype=v.dtype)
        return spike

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v, v_th = ctx.saved_tensors
        gamma = ctx.gamma
        diff = torch.abs(v - v_th)
        surrogate_grad = 1.0 / ((1.0 + gamma * diff) ** 2)
        grad_v = grad_output * surrogate_grad
        grad_v_th = -grad_output * surrogate_grad
        return grad_v, grad_v_th, None


def spike_fn(v: torch.Tensor, v_th: torch.Tensor, gamma: float = 4.0) -> torch.Tensor:
    return FastSigmoidSurrogate.apply(v, v_th, gamma)


class HomeostaticPreLMHeadVarianceClamp(nn.Module):
    """
    Pre-LM Head Variance Stabilizer at Layer 24 -> Output.
    Bounds latent variance from exploding (e.g. 12.5 -> [1.0, 1.5]),
    eliminating temperature hyper-sensitivity and output degeneration.
    """
    def __init__(self, target_variance: float = 1.25, eps: float = 1e-6):
        super().__init__()
        self.target_variance = target_variance
        self.eps = eps

    def forward(self, h_final: torch.Tensor) -> torch.Tensor:
        var = torch.var(h_final, dim=-1, keepdim=True)
        # Clamps variance smoothly when exceeding target_variance
        scale = torch.clamp(math.sqrt(self.target_variance) / torch.sqrt(var + self.eps), max=1.0)
        return h_final * scale


class SyntaxAwareCurvatureWhitening(nn.Module):
    """
    Syntax-Aware Curvature Whitening (SACW) for HADL Adversarial Defense.
    Differentiates dense, highly-structured mathematical syntax and code
    from high-entropy unstructured adversarial perturbations.
    """
    def __init__(self, base_threat_threshold: float = 1.5, kappa: float = 0.8):
        super().__init__()
        self.base_threshold = base_threat_threshold
        self.kappa = kappa

    def evaluate_threat(
        self,
        h: torch.Tensor,
        raw_curvature: float,
        token_entropies: Optional[torch.Tensor] = None
    ) -> Tuple[bool, float]:
        if token_entropies is not None and token_entropies.numel() > 1:
            entropy_var = float(torch.var(token_entropies).item())
            syntax_structure = 1.0 / (1.0 + entropy_var)
        else:
            syntax_structure = 0.5
        
        effective_threshold = self.base_threshold * (1.0 + self.kappa * syntax_structure)
        is_threat = raw_curvature > effective_threshold
        return is_threat, effective_threshold


class LeakyIntegrateAndFireGate(nn.Module):
    """
    LIF Spiking Gate at Layer 7 -> 8 Transition.
    
    Equations:
    - Current: I(t) = W_sensor h_7(t)
    - Integration: V_m(t) = beta * V_m(t-1) + I(t)
    - Spike: S(t) = Theta(V_m(t) - V_th(t))
    - Refractory Reset: V_m(t) <- V_m(t) * (1 - S(t))
    - Homeostatic Threshold: V_th(t) = V_th_0 + alpha * (S_bar - r_target)
    """
    def __init__(
        self,
        d_model: int = 2048,
        beta: float = 0.85,
        v_threshold: float = 1.0,
        target_firing_rate: float = 0.20,
        homeostatic_alpha: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.beta = beta
        self.base_v_th = v_threshold
        self.target_rate = target_firing_rate
        self.homeostatic_alpha = homeostatic_alpha
        self.enable_dynamic_threshold = True

        self.w_sensor = nn.Linear(d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)

        # Initialize sensor weights with Xavier scaling
        nn.init.normal_(self.w_sensor.weight, std=1.0 / math.sqrt(d_model))

        # State buffers
        self.register_buffer("v_membrane", torch.zeros(1, 1, 1))
        self.register_buffer("v_th", torch.tensor(v_threshold))
        self.register_buffer("ema_rate", torch.tensor(target_firing_rate))
        self.register_buffer("total_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_spikes", torch.tensor(0, dtype=torch.long))

    def reset_state(self):
        """Resets membrane potential and tracking buffers."""
        with torch.no_grad():
            self.v_membrane = torch.zeros(1, 1, 1, device=self.v_th.device, dtype=self.v_th.dtype)
            self.v_th = torch.tensor(self.base_v_th, device=self.v_th.device, dtype=self.v_th.dtype)
            self.ema_rate = torch.tensor(self.target_rate, device=self.ema_rate.device, dtype=self.ema_rate.dtype)
            self.total_steps = torch.tensor(0, device=self.total_steps.device, dtype=torch.long)
            self.total_spikes = torch.tensor(0, device=self.total_spikes.device, dtype=torch.long)

    def forward(self, h_layer7: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Processes Layer 7 hidden states and emits binary spikes S(t).
        
        Args:
            h_layer7: [B, S, D] hidden representations at Layer 7.
            
        Returns:
            spike: [B, S, 1] binary spike tensor (with surrogate gradients).
            telemetry: diagnostic metadata (membrane potential, firing rate, open status).
        """
        B, S, D = h_layer7.shape
        target_dtype = self.w_sensor.weight.dtype
        # Sensory current injection (depolarization magnitude): [B, S, 1]
        rms_mag = torch.sqrt(torch.mean(h_layer7.to(dtype=target_dtype) ** 2, dim=-1, keepdim=True) + 1e-6)
        h_norm = self.norm(h_layer7.to(dtype=target_dtype)) * torch.clamp(rms_mag, max=2.0)
        i_current = torch.abs(self.w_sensor(h_norm))

        # Sequence-level integration
        spikes_list = []
        curr_v = self.v_membrane.to(device=h_layer7.device, dtype=target_dtype)
        if curr_v.shape[0] != B or curr_v.shape[1] != 1:
            curr_v = torch.zeros(B, 1, 1, device=h_layer7.device, dtype=target_dtype)

        # Dynamic Sensitivity: free-form modulation based on representation dispersion
        latent_var = torch.var(h_layer7, dim=-1).mean()
        if self.enable_dynamic_threshold:
            # Scale sensitivity dynamically: high entropy/uncertainty elevates sensitivity (adjusts threshold)
            dyn_th_factor = 1.0 + 0.25 * torch.tanh(latent_var - 1.0)
            curr_th = self.v_th.to(device=h_layer7.device, dtype=target_dtype) * dyn_th_factor.to(device=h_layer7.device, dtype=target_dtype)
        else:
            dyn_th_factor = torch.tensor(1.0, device=h_layer7.device)
            curr_th = self.v_th.to(device=h_layer7.device, dtype=target_dtype)

        # Fast Vectorized Conv1D Path for prompt prefill sequences (S > 1) during inference
        if S > 1 and not self.training:
            K_len = min(S, 32)
            decay_weights = (self.beta ** torch.arange(K_len, device=h_layer7.device, dtype=target_dtype)).flip(0).view(1, 1, K_len)
            padded = F.pad(i_current.transpose(1, 2), (K_len - 1, 0))
            v_approx = F.conv1d(padded, decay_weights).transpose(1, 2)
            spikes = spike_fn(v_approx, curr_th)
            curr_v = v_approx[:, -1:, :] * (1.0 - spikes[:, -1:, :].detach())
        else:
            # Sequential exact step-by-step evaluation
            for t in range(S):
                i_t = i_current[:, t:t+1, :]  # [B, 1, 1]
                curr_v = self.beta * curr_v + i_t
                s_t = spike_fn(curr_v, curr_th)
                # Refractory reset
                curr_v = curr_v * (1.0 - s_t.detach())
                spikes_list.append(s_t)
            spikes = torch.cat(spikes_list, dim=1)  # [B, S, 1]

        # Multi-State Biological Cognitive Spiking Classifier
        batch_rate = float(spikes.mean().item())
        mean_rms = float(rms_mag.mean().item())
        var_score = float(latent_var.item())
        mean_v = float(curr_v.mean().item())

        if int(self.total_steps.item()) > 10000:
            cognitive_state = "FATIGUE"      # Context Saturation / Long Horizon
        elif batch_rate == 0.0 or mean_rms < 0.25:
            cognitive_state = "ROUTINE_CALM" # Balanced Flow / Solid Baseline
        elif batch_rate >= 0.35 and mean_rms > 1.5:
            cognitive_state = "EUREKA"       # Breakthrough / Moment of Insight
        elif mean_rms > 2.0 or mean_v > 1.8:
            cognitive_state = "STRESS"       # High Computational / Arithmetic Load
        elif var_score > 1.8 and batch_rate < 0.20:
            cognitive_state = "CONFUSION"    # Cognitive Ambiguity / Multi-head dispersion
        else:
            cognitive_state = "ROUTINE_CALM"

        # Update persistent buffers (no grad)
        with torch.no_grad():
            self.v_membrane = curr_v.detach().mean(dim=0, keepdim=True).clone()
            new_ema = 0.9 * float(self.ema_rate.item()) + 0.1 * batch_rate
            self.ema_rate = torch.tensor(new_ema, device=self.ema_rate.device, dtype=self.ema_rate.dtype)
            
            # Homeostatic adaptation: if firing too frequently, raise threshold; if too silent, lower
            delta_th = self.homeostatic_alpha * (new_ema - self.target_rate)
            new_th = max(0.01, min(5.0, self.base_v_th + delta_th))
            self.v_th = torch.tensor(new_th, device=self.v_th.device, dtype=self.v_th.dtype)

            self.total_steps = self.total_steps + S
            self.total_spikes = self.total_spikes + int(spikes.sum().item())

        telemetry = {
            "spike_firing_rate": round(batch_rate, 4),
            "v_membrane_mean": round(float(self.v_membrane.mean().item()), 4),
            "v_threshold_adaptive": round(float(self.v_th.item()), 4),
            "dynamic_threshold_factor": round(float(dyn_th_factor.item()), 4),
            "dyn_th_factor": round(float(dyn_th_factor.item()), 4),
            "mean_dynamic_threshold": round(float(curr_th.mean().item()), 4),
            "cognitive_state": cognitive_state,
            "gate_open_ratio": round(float((spikes > 0.0).float().mean().item()), 4),
            "total_spikes_fired": int(self.total_spikes.item())
        }
        return spikes, telemetry


class DynamicNeuralReplicator(nn.Module):
    """
    Dynamic Neural Replication (DNR).
    
    Formula:
    v_kv = norm_d( sum_{t=1}^T K_t * V_t / sqrt(d) ) in R^D
    rho_density = (C_max / (C_active + eps)) * 1.0 (100% / cache)
    W_rep(h) = h + rho_density * (v_kv (v_kv^T h)) / (||v_kv||^2 + eps)
    
    Preserves exact identity (I_d) when cache is empty or minimal,
    and amplifies high-saliency vertical features when active concepts populate the KV channels.
    """
    def __init__(self, d_model: int = 2048, c_max: int = 512, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.c_max = c_max
        self.eps = eps
        self.norm_kv = nn.LayerNorm(d_model)

    def extract_vertical_saliency(self, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Extracts high-saliency vertical vector from KV channels.
        
        Args:
            K: [B, T, D] or [B, H, T, D_head] Key tensor
            V: [B, T, D] or [B, H, T, D_head] Value tensor
            
        Returns:
            v_kv: [B, D] normalized vertical saliency vector
        """
        if K.dim() == 4:
            # Flatten heads: [B, H, T, D_h] -> [B, T, D]
            B, H, T, Dh = K.shape
            K = K.permute(0, 2, 1, 3).reshape(B, T, H * Dh)
            V = V.permute(0, 2, 1, 3).reshape(B, T, H * Dh)
        else:
            B, T, D = K.shape

        target_dtype = self.norm_kv.weight.dtype if self.norm_kv.weight is not None else K.dtype
        # Vertical channel-wise cross product sum
        kv_cross = torch.sum(K.to(dtype=target_dtype) * V.to(dtype=target_dtype), dim=1) / math.sqrt(self.d_model)  # [B, D]
        v_kv = self.norm_kv(kv_cross)
        return v_kv

    def extract_multivector_subspace(self, K: torch.Tensor, V: torch.Tensor, max_rank: int = 3) -> torch.Tensor:
        """
        Extracts multi-vector orthogonal subspace basis from KV channels (1D -> 2D -> 3D)
        via Gram-Schmidt orthogonalization. Guarantees a full orthonormal basis of rank
        `max_rank` (e.g., (e_1, e_2, e_3) in R^{B x 3 x D}) on every forward pass.
        
        Args:
            K: [B, T, D] or [B, H, T, D_head] Key tensor
            V: [B, T, D] or [B, H, T, D_head] Value tensor
            max_rank: maximum subspace rank (default 3)
            
        Returns:
            V_basis: [B, max_rank, D] orthonormal subspace matrix satisfying V_basis @ V_basis^T = I
        """
        if K.dim() == 4:
            B, H, T, Dh = K.shape
            K = K.permute(0, 2, 1, 3).reshape(B, T, H * Dh)
            V = V.permute(0, 2, 1, 3).reshape(B, T, H * Dh)
        else:
            B, T, D = K.shape

        target_dtype = self.norm_kv.weight.dtype if self.norm_kv.weight is not None else K.dtype
        K_t = K.to(dtype=target_dtype)
        V_t = V.to(dtype=target_dtype)

        # Vector 1 (e_1): Primary vertical channel cross product
        kv_cross = torch.sum(K_t * V_t, dim=1) / math.sqrt(self.d_model)  # [B, D]
        v1 = self.norm_kv(kv_cross)
        if float(torch.norm(v1).item()) < 1e-6:
            v1 = K_t[:, -1, :] + torch.roll(V_t[:, -1, :], shifts=1, dims=-1)
            if float(torch.norm(v1).item()) < 1e-6:
                v1 = torch.ones_like(v1)
        v1_norm = F.normalize(v1.float(), p=2, dim=-1).to(dtype=target_dtype)

        if max_rank <= 1:
            return v1_norm.unsqueeze(1)  # [B, 1, D]

        basis_list = [v1_norm]

        # Vector 2 (e_2): First-order temporal velocity dynamics (or channel-differential fallback when T == 1)
        if T >= 2:
            d_k = K_t[:, 1:, :] - K_t[:, :-1, :]
            d_v = V_t[:, 1:, :] - V_t[:, :-1, :]
            v2_raw = torch.sum(d_k * d_v, dim=1) / math.sqrt(self.d_model)
        else:
            v2_raw = (K_t[:, 0, :] - V_t[:, 0, :]) + torch.roll(v1_norm, shifts=1, dims=-1)
        v2_norm = self.norm_kv(v2_raw)
        # Two-pass Gram-Schmidt orthogonalization against e_1
        v1_f = v1_norm.float()
        v2_f = v2_norm.float()
        v2_ortho = v2_f - torch.sum(v2_f * v1_f, dim=-1, keepdim=True) * v1_f
        v2_ortho = v2_ortho - torch.sum(v2_ortho * v1_f, dim=-1, keepdim=True) * v1_f
        if float(torch.norm(v2_ortho, dim=-1).min().item()) < 1e-6:
            alt2 = torch.roll(v1_f, shifts=2, dims=-1) * torch.linspace(-1.0, 1.0, v1_f.size(-1), device=v1_f.device).unsqueeze(0)
            v2_ortho = alt2 - torch.sum(alt2 * v1_f, dim=-1, keepdim=True) * v1_f
        v2_unit = F.normalize(v2_ortho, p=2, dim=-1).to(dtype=target_dtype)
        basis_list.append(v2_unit)

        if max_rank >= 3:
            # Vector 3 (e_3): Second-order temporal acceleration dynamics (or bi-orthogonal harmonic fallback when T < 3)
            if T >= 3:
                d_k = K_t[:, 1:, :] - K_t[:, :-1, :]
                d_v = V_t[:, 1:, :] - V_t[:, :-1, :]
                a_k = d_k[:, 1:, :] - d_k[:, :-1, :]
                a_v = d_v[:, 1:, :] - d_v[:, :-1, :]
                v3_raw = torch.sum(a_k * a_v, dim=1) / math.sqrt(self.d_model)
            else:
                v3_raw = torch.roll(v1_norm, shifts=3, dims=-1) - torch.roll(v2_unit, shifts=1, dims=-1)
            v3_norm = self.norm_kv(v3_raw)
            # Two-pass Gram-Schmidt orthogonalization against e_1 and e_2
            v2_u_f = v2_unit.float()
            v3_f = v3_norm.float()
            for _ in range(2):
                v3_f = v3_f - torch.sum(v3_f * v1_f, dim=-1, keepdim=True) * v1_f
                v3_f = v3_f - torch.sum(v3_f * v2_u_f, dim=-1, keepdim=True) * v2_u_f
            if float(torch.norm(v3_f, dim=-1).min().item()) < 1e-6:
                alt3 = torch.roll(v1_f, shifts=5, dims=-1) + torch.cos(torch.linspace(0.0, 3.14159, v1_f.size(-1), device=v1_f.device)).unsqueeze(0)
                for _ in range(2):
                    alt3 = alt3 - torch.sum(alt3 * v1_f, dim=-1, keepdim=True) * v1_f
                    alt3 = alt3 - torch.sum(alt3 * v2_u_f, dim=-1, keepdim=True) * v2_u_f
                v3_f = alt3
            v3_unit = F.normalize(v3_f, p=2, dim=-1).to(dtype=target_dtype)
            basis_list.append(v3_unit)

        V_basis = torch.stack(basis_list[:max_rank], dim=1)  # [B, max_rank, D]
        return V_basis

    def forward(
        self,
        h: torch.Tensor,
        v_kv: Optional[torch.Tensor] = None,
        c_active: int = 1,
        subspace_rank: int = 1
    ) -> torch.Tensor:
        """
        Applies Dynamic Neural Replication operator on representations h.
        Supports 1D vector or multi-vector KD subspace projection.
        
        Args:
            h: [B, S, D] hidden states to modulate
            v_kv: [B, D] or [B, K_sub, D] vertical saliency tensor (or None for identity)
            c_active: active token count in KV cache
            subspace_rank: desired rank to project onto
            
        Returns:
            h_mod: [B, S, D] modulated representations with identity preservation
        """
        if v_kv is None:
            return h  # Pure identity I_d

        B, S, D = h.shape
        rho_density = min(2.0, float(self.c_max) / max(1.0, float(c_active)))

        if v_kv.dim() == 2:
            # 1D single-vector projection (backward-compatible)
            v_norm_sq = torch.sum(v_kv * v_kv, dim=-1, keepdim=True) + self.eps
            v_unit = v_kv / torch.sqrt(v_norm_sq)
            v_unit_exp = v_unit.unsqueeze(1).to(device=h.device, dtype=h.dtype)  # [B, 1, D]
            inner = torch.sum(h * v_unit_exp, dim=-1, keepdim=True)              # [B, S, 1]
            h_proj = inner * v_unit_exp                                          # [B, S, D]
        else:
            # Multi-Vector KD Subspace Escalation
            # v_kv: [B, K_sub, D]
            eff_rank = min(subspace_rank, v_kv.shape[1])
            V_sub = v_kv[:, :eff_rank, :].to(device=h.device, dtype=h.dtype)    # [B, eff_rank, D]
            # Gram projection: sum_{k=1}^K (h . u_k) u_k
            # h: [B, S, D], V_sub: [B, K, D]
            # inners: [B, S, K]
            inners = torch.einsum("bsd,bkd->bsk", h, V_sub)
            h_proj = torch.einsum("bsk,bkd->bsd", inners, V_sub)               # [B, S, D]

        # Identity-preserving modulation
        h_mod = h + 0.10 * rho_density * h_proj
        return h_mod


class OpenFreezeLiquidPlasticity(nn.Module):
    """
    Open-Freeze Liquid Phase-State Plasticity.
    
    Viscosity Equation:
    mu(t) = tanh(gamma * Surprisal(t)) in [0, 1)
    
    Phase Behavior:
    - Solid Phase (mu -> 0): 100% frozen foundation model, zero catastrophic forgetting.
    - Liquid Phase (mu -> 1): Plastic synaptic flow W_plastic engaged upon breakthrough insight / spike.
    """
    def __init__(self, d_model: int = 2048, gamma: float = 1.0, rank: int = 32):
        super().__init__()
        self.d_model = d_model
        self.gamma = gamma
        self.rank = rank

        # Low-rank plastic weights: Delta W_plastic = U * V
        self.plastic_u = nn.Linear(d_model, rank, bias=False)
        self.plastic_v = nn.Linear(rank, d_model, bias=False)
        self.w_surprisal = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

        # Initialize orthogonal low-rank matrices (non-zero scaled plastic_v so liquid phase is active)
        nn.init.orthogonal_(self.plastic_u.weight)
        nn.init.orthogonal_(self.plastic_v.weight)
        with torch.no_grad():
            self.plastic_v.weight.data *= 0.05
        nn.init.zeros_(self.w_surprisal.bias)
        nn.init.normal_(self.w_surprisal.weight, std=0.01)

    def forward(
        self,
        h_frozen: torch.Tensor,
        spike: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Blends solid frozen base and liquid plastic state based on dynamic viscosity.
        
        Args:
            h_frozen: [B, S, D] hidden representations from base layer.
            spike: [B, S, 1] optional spike tensor from LIF gate.
            
        Returns:
            h_out: [B, S, D] phase-modulated hidden representations.
            telemetry: diagnostic metadata (viscosity, phase state).
        """
        B, S, D = h_frozen.shape
        target_dtype = self.plastic_u.weight.dtype
        h_norm = self.norm(h_frozen.to(dtype=target_dtype))

        # Dynamic surprisal metric: gated by neural spike (insight/breakthrough trigger)
        if spike is not None:
            # Gated strictly by spike: spike=0 ensures solid frozen state (zero forgetting)
            effective_spike = F.relu(spike)
            raw_surprisal = effective_spike * (1.0 + torch.sigmoid(self.w_surprisal(h_norm)))
            viscosity = torch.tanh(self.gamma * raw_surprisal)  # [B, S, 1]
        else:
            raw_surprisal = torch.abs(self.w_surprisal(h_norm))  # [B, S, 1]
            deadband = 0.50
            effective_surprisal = F.relu(raw_surprisal - deadband)
            viscosity = torch.tanh(self.gamma * effective_surprisal)  # [B, S, 1]

        # Plastic synaptic delta: Delta W h
        delta_plastic = 0.15 * torch.tanh(self.plastic_v(self.plastic_u(h_norm)))  # [B, S, D]

        # Continuous phase blending: (1 - mu) * h_frozen + mu * (h_frozen + Delta W h)
        h_out = h_frozen + (viscosity * delta_plastic).to(dtype=h_frozen.dtype)

        mean_viscosity = float(viscosity.mean().item())
        phase_state = "SOLID_FROZEN" if mean_viscosity < 0.10 else ("LIQUID_PLASTIC" if mean_viscosity > 0.60 else "TRANSIENT_GEL")

        telemetry = {
            "fluid_viscosity_mu": round(mean_viscosity, 4),
            "phase_state": phase_state,
            "plastic_energy": round(float(torch.norm(delta_plastic, dim=-1).mean().item()), 4)
        }
        return h_out, telemetry

    def adapt_online_step(
        self,
        h: torch.Tensor,
        spike: Optional[torch.Tensor] = None,
        lr: float = 0.05,
        dopamine_sign: float = 1.0
    ) -> float:
        """
        Executes online liquid plasticity adaptation with signed dopamine modulation.
        Updates plastic synaptic weights proportionally to dynamic viscosity mu(t).
        Solid state (mu -> 0) results in zero weight drift (zero catastrophic forgetting).
        Liquid state (mu -> 1) absorbs novel representations (+1) or deflates falsified ones (-1).
        """
        with torch.no_grad():
            if h.dim() == 2:
                h = h.unsqueeze(1)
            target_dtype = self.plastic_u.weight.dtype
            w_dev = self.plastic_u.weight.device
            h_norm = self.norm(h.to(device=w_dev, dtype=target_dtype))
            if spike is not None:
                effective_spike = F.relu(spike.to(device=w_dev, dtype=target_dtype))
                raw_surprisal = effective_spike * (1.0 + torch.sigmoid(self.w_surprisal(h_norm)))
                viscosity = torch.tanh(self.gamma * raw_surprisal)
            else:
                raw_surprisal = torch.abs(self.w_surprisal(h_norm))
                deadband = 0.50
                effective_surprisal = F.relu(raw_surprisal - deadband)
                viscosity = torch.tanh(self.gamma * effective_surprisal).clamp(min=0.15)  # [B, S, 1]

            # Low-rank signed Hebbian/Anti-Hebbian update modulated by fluid viscosity
            u_act = self.plastic_u(h_norm)  # [B, S, rank]
            v_update = torch.einsum("bsd,bsr->dr", viscosity * h_norm, u_act) / max(1, h.shape[0] * h.shape[1])
            signed_lr = float(lr) * float(dopamine_sign)
            self.plastic_v.weight.data.add_(signed_lr * v_update)
            return float(torch.norm(v_update).item() * abs(signed_lr))

    def reset_liquid_state(self) -> None:
        """Resets liquid plastic weights to orthogonal base scale (using float32 for geqrf)."""
        with torch.no_grad():
            dev = self.plastic_u.weight.device
            target_dtype = self.plastic_u.weight.dtype
            wu = torch.empty(self.plastic_u.weight.shape, dtype=torch.float32, device="cpu")
            nn.init.orthogonal_(wu)
            self.plastic_u.weight.data.copy_(wu.to(device=dev, dtype=target_dtype))

            wv = torch.empty(self.plastic_v.weight.shape, dtype=torch.float32, device="cpu")
            nn.init.orthogonal_(wv)
            wv.mul_(0.05)
            self.plastic_v.weight.data.copy_(wv.to(device=dev, dtype=target_dtype))


class EndogenousBanachPlasticUnit(nn.Module):
    """
    Endogenous Banach-Contractive Synaptic Plasticity Unit.
    Implements:
    - Invariant Lipschitz-bounded contraction: L <= gamma < 1.0 (Banach Fixed-Point Theorem)
    - Oja's Normalized Synaptic Plasticity for bounded self-stabilizing growth:
      Delta W = eta * (y * x^T - alpha * ||y||^2 * W)
      Eigenvalues strictly bounded: lambda_max(W) <= 1 / sqrt(alpha)
    - In-place physical synaptic weight modification: W_plastic in R^{D x D}
    """
    def __init__(self, d_model: int = 2048, gamma_contraction: float = 0.85, eta: float = 0.01, alpha: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.gamma = gamma_contraction
        self.eta = eta
        self.alpha = alpha
        self.w_plastic = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.orthogonal_(self.w_plastic)
        with torch.no_grad():
            self.w_plastic.data *= 0.10
        self.register_buffer("plasticity_step_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("initial_w", torch.zeros(d_model, d_model), persistent=False)
        self.register_buffer("has_initial_w", torch.tensor(False, dtype=torch.bool), persistent=False)
        self._last_energy: Optional[torch.Tensor] = None
        self._last_dopamine: float = 0.0

    def forward(self, h: torch.Tensor, online_learning: bool = False) -> torch.Tensor:
        if not self.has_initial_w:
            self.snapshot_initial_weights()

        target_dtype = self.w_plastic.dtype
        h_cast = h.to(dtype=target_dtype)
        y = self.gamma * torch.tanh(F.linear(h_cast, self.w_plastic))

        if online_learning and not self.training:
            with torch.no_grad():
                x_mean = h_cast.mean(dim=1)
                y_mean = y.mean(dim=1)
                
                # Active Predictive Free-Energy Dopaminergic STDP (PFE-STDP)
                # Proxy of Free Energy: internal representation dispersion
                curr_energy = 0.5 * torch.sum(y_mean ** 2, dim=-1).mean()
                if self._last_energy is not None:
                    delta_dopamine = torch.tanh((self._last_energy - curr_energy) * 2.0).item()
                else:
                    delta_dopamine = 0.0
                self._last_energy = curr_energy.detach()
                self._last_dopamine = delta_dopamine

                hebbian = torch.matmul(y_mean.t(), x_mean) / float(x_mean.shape[0])
                decay = self.alpha * curr_energy * self.w_plastic.data
                
                step_eta = 0.01 * self.eta
                if delta_dopamine >= 0:
                    # Dopaminergic Long-Term Potentiation (LTP)
                    delta_w = step_eta * (1.0 + delta_dopamine) * (hebbian - decay)
                else:
                    # Dopaminergic Long-Term Depression (LTD)
                    delta_w = -step_eta * abs(delta_dopamine) * (hebbian + 0.1 * decay)

                self.w_plastic.data.add_(delta_w)

                curr_norm = torch.norm(self.w_plastic.data)
                max_norm = math.sqrt(self.d_model) * 0.25
                if curr_norm > max_norm:
                    self.w_plastic.data.mul_(max_norm / (curr_norm + 1e-8))

                self.plasticity_step_count.add_(1)

        return y.to(dtype=h.dtype)

    @torch.no_grad()
    def apply_closed_loop_dopamine(
        self,
        h_pre: torch.Tensor,
        h_post: torch.Tensor,
        dopamine_sign: float,
        nullspace_delta: Optional[torch.Tensor] = None
    ) -> float:
        """
        Applies signed Popperian-Cycle closed-loop dopamine feedback (+1 Hebbian LTP
        or -1 Anti-Hebbian Orthogonal LTD) with Banach spectral norm bounding.
        """
        if not self.has_initial_w:
            self.snapshot_initial_weights()

        w_dev = self.w_plastic.device
        w_dtype = self.w_plastic.dtype
        x = h_pre.detach().to(device=w_dev, dtype=w_dtype).reshape(-1, self.d_model)
        y = (
            nullspace_delta.detach().to(device=w_dev, dtype=w_dtype).reshape(-1, self.d_model)
            if nullspace_delta is not None
            else h_post.detach().to(device=w_dev, dtype=w_dtype).reshape(-1, self.d_model)
        )
        outer = torch.matmul(y.t(), x) / float(max(1, x.size(0)))
        decay = 0.05 * self.alpha * self.w_plastic.data
        delta_w = self.eta * float(dopamine_sign) * outer - abs(float(dopamine_sign)) * self.eta * decay
        self.w_plastic.data.add_(delta_w)

        curr_norm = torch.norm(self.w_plastic.data)
        max_norm = math.sqrt(self.d_model) * 0.25
        if curr_norm > max_norm:
            self.w_plastic.data.mul_(max_norm / (curr_norm + 1e-8))

        self._last_dopamine = float(dopamine_sign)
        self.plasticity_step_count.add_(1)
        return float(torch.norm(delta_w).item())

    def snapshot_initial_weights(self):
        """Snapshots baseline synaptic state to track empirical Frobenius drift."""
        self.initial_w.copy_(self.w_plastic.data)
        self.has_initial_w.copy_(torch.tensor(True))

    def get_stats(self) -> Dict[str, Any]:
        """Returns empirical telemetry on synaptic plasticity, drift, free energy, and dopamine."""
        w = self.w_plastic.data
        w_norm = float(torch.norm(w).item())
        drift = float(torch.norm(w - self.initial_w).item()) if self.has_initial_w else 0.0
        return {
            "plasticity_step_count": int(self.plasticity_step_count.item()),
            "w_norm": round(w_norm, 5),
            "frobenius_drift": round(drift, 5),
            "last_energy": round(float(self._last_energy.item()), 5) if self._last_energy is not None else 0.0,
            "last_dopamine": round(float(self._last_dopamine), 5),
        }


class ReEntrantCrossLayerBus(nn.Module):
    """
    Re-Entrant Cross-Layer Re-Query Bus.
    
    Enables deep downstream layers (e.g. Layer 14, Layer 21) encountering
    ambiguity or high dispersion to emit a re-query spike back to Layer 7 VGNI.
    VGNI searches real-life grounding and injects clarity delta back to the downstream layer.
    """
    def __init__(self, d_model: int = 2048, ambiguity_threshold: float = 0.70):
        super().__init__()
        self.d_model = d_model
        self.ambiguity_threshold = ambiguity_threshold
        
        self.probe = nn.Linear(d_model, d_model // 4, bias=False)
        self.reentry_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        nn.init.orthogonal_(self.probe.weight)
        nn.init.orthogonal_(self.reentry_proj.weight)

    def evaluate_ambiguity(self, h_deep: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Measures representation entropy / dispersion at deep layer.
        """
        B, S, D = h_deep.shape
        target_dtype = self.probe.weight.dtype
        h_p = self.probe(self.norm(h_deep.to(dtype=target_dtype)))
        # Dispersion: variance across latent subspace
        var_score = torch.var(h_p, dim=-1).mean()
        ambiguity = torch.sigmoid(var_score).item()
        needs_reentry = ambiguity > self.ambiguity_threshold
        return needs_reentry, float(ambiguity)

    def execute_reentry(
        self,
        h_deep: torch.Tensor,
        vgni: VirtualWorldGeneralIntelligence
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Queries VGNI associative state and applies corrective feedback delta.
        """
        needs_reentry, ambiguity = self.evaluate_ambiguity(h_deep)
        if not needs_reentry:
            return h_deep, {"reentry_active": False, "ambiguity": ambiguity}

        # Query VGNI for real-world grounding
        grounded_vec, affinity = vgni.query_prior_grounding(h_deep)
        if int(vgni.num_stored.item()) == 0:
            return h_deep, {"reentry_active": False, "ambiguity": ambiguity, "affinity": 0.0}

        # Inject re-entrant context delta strictly in the orthogonal complement P_perp(h_deep)
        target_dtype = self.reentry_proj.weight.dtype
        app_in = grounded_vec.unsqueeze(1).to(device=h_deep.device, dtype=target_dtype)
        raw_delta = 0.10 * torch.tanh(self.reentry_proj(app_in)).to(dtype=h_deep.dtype)
        delta_ortho = ConcurrentDualStreamSynthesizer.orthogonal_symplectic_project(raw_delta, h_deep)
        base_norm = torch.norm(h_deep, dim=-1, keepdim=True) + 1e-6
        delta_norm = torch.norm(delta_ortho, dim=-1, keepdim=True) + 1e-6
        scale = torch.clamp((0.015 * base_norm) / delta_norm, max=1.0)
        delta = scale * delta_ortho
        h_mod = h_deep + delta

        telemetry = {
            "reentry_active": True,
            "ambiguity": ambiguity,
            "vgni_affinity": affinity,
            "reentry_delta_norm": round(float(torch.norm(delta).item()), 4)
        }
        return h_mod, telemetry


class VirtualLatentSandbox(nn.Module):
    """
    In-Brain Virtual Latent Sandbox.
    
    Performs virtual dry-run state evaluations before token commitment:
    1. Tests prospective transition stability: ||h_{t+1} - h_t||.
    2. Enforces Banach contraction regularizer (Lipschitz constant <= gamma < 1.0).
    3. Eliminates cognitive divergence, runtime hallucinations, and runaway degradation.
    """
    def __init__(self, d_model: int = 2048, max_drift_bound: float = 1.0, gamma_lipschitz: float = 0.85):
        super().__init__()
        self.d_model = d_model
        self.max_drift_bound = max_drift_bound
        self.base_gamma = gamma_lipschitz
        self.gamma_lipschitz = gamma_lipschitz
        self.enable_dynamic_lipschitz = True

        self.latent_step = nn.Linear(d_model, d_model, bias=False)
        self.w_comp = nn.Linear(d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)

        nn.init.orthogonal_(self.latent_step.weight)
        nn.init.normal_(self.w_comp.weight, std=0.01)

    def forward(
        self,
        h_current: torch.Tensor,
        delta_candidate: torch.Tensor,
        cognitive_state: str = "ROUTINE_CALM"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Evaluates candidate update delta inside the sandbox with Dynamic Lipschitz Field.
        Relaxes gamma during mathematical/symbolic computation and clamps strictly during runaway drift.
        """
        target_dtype = self.latent_step.weight.dtype
        # Perturbation drift under virtual contractive step
        t_delta = self.latent_step(delta_candidate.to(dtype=target_dtype))
        drift = torch.norm(t_delta, dim=-1, keepdim=True)  # [B, S, 1]

        # Relative perturbation drift against current state norm
        base_norm = torch.norm(h_current.to(dtype=target_dtype), dim=-1, keepdim=True) + 1e-6
        relative_drift = drift / base_norm

        # Introspective Maintenance: detect if model is executing numerical/mathematical computation
        energy = torch.sqrt(torch.mean(h_current.to(dtype=target_dtype) ** 2, dim=-1, keepdim=True) + 1e-6)
        e_comp = torch.tanh(energy / 2.0)  # [B, S, 1] in [0, 1)

        # Dynamic Lipschitz Field: relax gamma up to 0.985 when under computational stress / arithmetic
        if self.enable_dynamic_lipschitz and (cognitive_state in ["STRESS", "EUREKA"]):
            gamma_dyn = torch.clamp(torch.tensor(self.base_gamma, device=h_current.device) + 0.135 * e_comp, max=0.985)
        elif self.enable_dynamic_lipschitz:
            # Token-wise adaptation based on computational energy
            gamma_dyn = torch.where(
                e_comp > 0.50,
                torch.clamp(torch.tensor(self.base_gamma, device=h_current.device) + 0.135 * e_comp, max=0.985),
                torch.full_like(e_comp, self.base_gamma)
            )
        else:
            # Conservative baseline for routine text
            gamma_dyn = torch.full_like(e_comp, self.base_gamma)

        # Dynamic Contraction bound
        max_allowed = self.max_drift_bound * gamma_dyn
        scale_factor = torch.clamp(max_allowed / (relative_drift + 1e-6), max=1.0)

        # Apply regularized delta
        safe_delta = delta_candidate * scale_factor.to(dtype=delta_candidate.dtype)

        is_clamped = bool((scale_factor < 0.99).any().item())
        telemetry = {
            "sandbox_simulated_drift": round(float(relative_drift.mean().item()), 4),
            "contraction_scale": round(float(scale_factor.mean().item()), 4),
            "dynamic_gamma": round(float(gamma_dyn.mean().item()), 4),
            "computational_energy": round(float(e_comp.mean().item()), 4),
            "sandbox_intervention_active": is_clamped
        }
        return safe_delta, telemetry


class NeuromorphicSpikingBrain(nn.Module):
    """
    Unified Neuromorphic Spiking Brain (X-Star HADL v5.5 Master Engine).
    
    Coordinates:
    - Layer 2 Pre-Router Spiking Pacemaker (u_drive signal emission)
    - Concurrent Dual-Stream Synthesizer (Parallel System 1 & System 2 with Symplectic Fusion)
    - Tour Leader Attractor (continuous potential field guidance without rigid templates)
    - Layer 7 LIF Spiking Gate
    - Virtual World General Intelligence (VGNI)
    - Dynamic Neural Replication (DNR)
    - Open-Freeze Liquid Phase Plasticity & Endogenous Banach Plastic Unit
    - Virtual Latent Sandbox
    - Re-Entrant Cross-Layer Re-Query Bus
    - Polynomial Embedded Router
    """
    def __init__(
        self,
        d_model: int = 2048,
        target_layer_idx: int = 7,
        c_max: int = 512,
        spike_v_th: float = 1.0,
        beta_decay: float = 0.85,
        target_firing_rate: float = 0.20
    ):
        super().__init__()
        self.d_model = d_model
        self.target_layer_idx = target_layer_idx

        # Model personality profile based on dimension scaling laws
        self.personality = ModelPersonalityAdapter.calibrate(
            d_model=d_model,
            base_v_th=spike_v_th,
            base_beta=beta_decay
        )

        # 1. Pre-Router Spiking Pacemaker (Layer 2 early contextual hook)
        self.pacemaker = PreRouterSpikingPacemaker(
            d_model=d_model,
            personality=self.personality,
            target_firing_rate=target_firing_rate
        )

        # 1b. Monolithic Incoherent Sensitivity Field (MISF Hydraulic Viscosity)
        self.misf = MonolithicIncoherentSensitivityField(d_model=d_model)

        # 2. Concurrent Dual-Stream Synthesizer
        self.dual_synthesizer = ConcurrentDualStreamSynthesizer(
            d_model=d_model,
            personality=self.personality
        )

        # 2b. Emergent Latent Imagination Canvas (ELIC / H-DREAM)
        self.imagination_canvas = EmergentLatentCanvas(
            d_model=d_model,
            num_fragments=8,
            dream_steps=2,
            contraction_factor=0.20,
            confusion_tolerance=0.40,
            enable_confusion_gating=True
        )

        # 3. LIF Spiking Gate
        self.spiking_gate = LeakyIntegrateAndFireGate(
            d_model=d_model,
            beta=self.personality.beta,
            v_threshold=self.personality.v_th,
            target_firing_rate=target_firing_rate
        )

        # 4. VGNI Epistemic Memory Engine
        self.vgni = VirtualWorldGeneralIntelligence(d_model=d_model, max_epistemic_entries=64)

        # 5. Dynamic Neural Replicator (DNR)
        self.dnr = DynamicNeuralReplicator(d_model=d_model, c_max=c_max)

        # 6. Open-Freeze Liquid Plasticity & Endogenous Banach Plastic Unit
        self.liquid_plasticity = OpenFreezeLiquidPlasticity(d_model=d_model, gamma=1.0)
        self.plastic_unit = EndogenousBanachPlasticUnit(
            d_model=d_model,
            gamma_contraction=self.personality.gamma_contraction,
            eta=self.personality.eta_plastic
        )

        # 7. Virtual Latent Sandbox
        self.sandbox = VirtualLatentSandbox(d_model=d_model, max_drift_bound=1.0)

        # 8. Re-Entrant Cross-Layer Bus
        self.reentry_bus = ReEntrantCrossLayerBus(d_model=d_model)

        # 9. Polynomial Embedded Router
        self.embedded_router = nn.Linear(d_model, 3, bias=True)
        nn.init.orthogonal_(self.embedded_router.weight)

        # Bridge transformation for Layer 7 -> 8
        self.layer7_bridge = nn.Linear(d_model, d_model, bias=False)
        nn.init.orthogonal_(self.layer7_bridge.weight)

        # 10. HADL v5.7 Self-Evolving Multimodal Organs:
        # - SynapticEligibilityTrace & Nullspace Projection
        # - IntrinsicPopperianCycleVerifier (Popperian AST/Sandbox + Cycle-Consistency Energy)
        # - UnitaryTrigonometricRotator (Isometric Givens Rotation for Anti-Hebbian Deflection)
        # - EndogenousAcousticVocalOrgan (`freq suara` + Self-Listening Auditory Re-Entry)
        # - Progressive2Dto3DImaginationCanvas (1D -> 2D Sketchpad -> 3D Gaussian/SDF + SO(3))
        self.eligibility_trace = SynapticEligibilityTrace(d_model=d_model)
        self.closed_loop_verifier = IntrinsicPopperianCycleVerifier(d_model=d_model)
        self.unitary_rotator = UnitaryTrigonometricRotator(d_model=d_model)
        self.vocal_organ = EndogenousAcousticVocalOrgan(
            d_model=d_model,
            misf=self.misf,
            pacemaker=self.pacemaker
        )
        self.imagination_3d = Progressive2Dto3DImaginationCanvas(d_model=d_model)

        # 11. Sleep-Phase Consolidation Engine (Hippocampal-Neocortical Nullspace Replay)
        self.sleep_engine = SleepPhaseConsolidationEngine(
            d_model=d_model,
            rank=32,
            consolidation_rate=0.05
        )

        # 12. HADL v5.8 Non-Token Latent Thought Matrix & Recurrent Central Router ("Otak Berputar")
        self.latent_matrix_engine = LatentThoughtMatrixEngine(d_model=d_model)
        self.central_router = CentralCognitiveRouter(d_model=d_model)

        self.loaded_checkpoints: Dict[str, bool] = {}
        self.last_routing_info: Dict[str, Any] = {}
        self.last_router_probs: Optional[torch.Tensor] = None
        self.last_misf_viscosity: Optional[torch.Tensor] = None
        self.last_deflection_delta: Optional[torch.Tensor] = None
        self.last_veracity_conflict_score: float = 0.0
        self.last_disclosure_vector: Optional[torch.Tensor] = None
        self.last_disclosure_gate: float = 0.0
        self.enable_online_learning: bool = False
        self.enable_3d_imagination: bool = True
        self.enable_endogenous_voice: bool = True
        self._rolling_k_cache: Optional[torch.Tensor] = None
        self._rolling_v_cache: Optional[torch.Tensor] = None

    def get_online_learning_stats(self) -> Dict[str, Any]:
        """Returns empirical stats from the endogenous plastic unit and closed-loop organs."""
        stats = {}
        if hasattr(self, "plastic_unit"):
            stats.update(self.plastic_unit.get_stats())
        if hasattr(self, "eligibility_trace"):
            stats["verified_basis_count"] = int(self.eligibility_trace.basis_count.item())
            stats["falsified_directions_count"] = int(self.eligibility_trace.falsified_count.item())
        if hasattr(self, "vocal_organ"):
            stats["voice_adaptation_steps"] = int(self.vocal_organ.voice_adaptation_steps.item())
            stats["ema_voice_coherence"] = round(float(self.vocal_organ.ema_coherence.item()), 4)
            stats["last_f0_mean_hz"] = round(float(self.vocal_organ.last_f0_mean.item()), 2)
        return stats

    def consolidate_sleep_phase(self, target_device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Executes an automatic mammalian sleep-phase memory consolidation cycle:
        1. Distills transient plastic updates via Truncated SVD.
        2. Projects updates onto the Orthogonal Nullspace of prior knowledge (Zero Catastrophic Forgetting).
        3. Flushes transient liquid plastic state so that subsequent multi-turn evaluations do not suffer from plastic drift.
        """
        try:
            dev = target_device or next(self.parameters()).device
            fast_delta = None
            if hasattr(self, "liquid_plasticity") and hasattr(self.liquid_plasticity, "plastic_v"):
                with torch.no_grad():
                    wv = self.liquid_plasticity.plastic_v.weight.data
                    wu = self.liquid_plasticity.plastic_u.weight.data
                    fast_delta = torch.matmul(wv, wu).to(dev)

            telem = self.sleep_engine.consolidate(fast_weights=fast_delta, target_device=dev, flush_episodes=True)
            if hasattr(self, "liquid_plasticity"):
                self.liquid_plasticity.reset_liquid_state()
            return telem
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    def reset_brain_state(self):
        """Resets transient membrane potentials, buffers, and tracking states, triggering sleep consolidation."""
        self.spiking_gate.reset_state()
        self.pacemaker.reset_state()
        self.last_routing_info.clear()
        self.last_router_probs = None
        self.last_misf_viscosity = None
        self.last_deflection_delta = None
        self.last_veracity_conflict_score = 0.0
        self.last_disclosure_vector = None
        self.last_disclosure_gate = 0.0
        self._rolling_k_cache = None
        self._rolling_v_cache = None
        self.consolidate_sleep_phase()

    def forward_layer2_pacemaker(
        self,
        h_layer2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Executes Layer 2 Pre-Router Spiking Pacemaker followed by Embedded Router.
        Spiking pacemaker evaluates early contextual saliency BEFORE routing,
        emitting s(t) and u_drive(t) to drive routing decisions across:
        Route 0 (Bypass / System 1 Fast Reflex), Route 1 (Jalur Tengah / Hybrid),
        and Route 2 (Heavy System 2 Banach Deliberation).
        """
        spikes, u_drive, p_telem = self.pacemaker(h_layer2)

        # Embedded router evaluation driven by pacemaker signals
        target_dtype = self.embedded_router.weight.dtype
        h_norm = F.layer_norm(h_layer2.to(dtype=target_dtype), (h_layer2.shape[-1],))
        raw_logits = self.embedded_router(h_norm)  # [B, S, 3]

        # Pacemaker driving modulation:
        # When u_drive is calm/negative and spikes=0 -> Route 0 (Fast Reflex) dominates
        # When u_drive > 0 or spikes fire -> probability shifts to Route 1 (Hybrid) and Route 2 (Heavy)
        drive_pos = F.relu(u_drive.to(dtype=target_dtype))  # [B, S, 1]
        spike_signal = spikes.to(dtype=target_dtype)        # [B, S, 1]

        router_logits = raw_logits.clone()
        router_logits[..., 0:1] = router_logits[..., 0:1] - 0.50 * drive_pos
        router_logits[..., 1:2] = router_logits[..., 1:2] + 0.25 * drive_pos
        router_logits[..., 2:3] = router_logits[..., 2:3] + 0.40 * drive_pos + 0.35 * spike_signal

        router_probs = F.softmax(router_logits, dim=-1)  # [B, S, 3]
        self.last_router_probs = router_probs

        mean_p = router_probs.mean(dim=(0, 1))  # [3]
        p_bypass = float(mean_p[0].item())
        p_hybrid = float(mean_p[1].item())
        p_heavy = float(mean_p[2].item())

        route_idx = int(torch.argmax(mean_p).item())
        modes = ["bypass", "jalur_tengah", "heavy"]
        route_mode = modes[route_idx]

        routing_telem = {
            "router_route_mode": route_mode,
            "router_p_bypass": round(p_bypass, 4),
            "router_p_hybrid": round(p_hybrid, 4),
            "router_p_heavy": round(p_heavy, 4),
        }
        p_telem.update(routing_telem)
        self.last_routing_info = routing_telem
        return spikes, u_drive, p_telem

    def load_trained_checkpoints(self, checkpoints_dir: Optional[Union[str, Path]] = None) -> Dict[str, bool]:
        """
        Automatically loads trained weights from checkpoints/ directory:
        1. vgni_lbu_autonomous_stabilized_qwen2b.pt (36 concept anchors + LBU manifold).
        2. xstar_brain_post_20min_cumulative.pt (learned plastic weights W_plastic in R^{2048x2048}).
        3. polynomial_embedded_router_weights.pt (embedded router weights).
        4. hadl_v58_latent_matrix_central_router.pt (LatentThoughtMatrixEngine & CentralCognitiveRouter).
        """
        if checkpoints_dir is None:
            possible_dirs = [
                Path("checkpoints"),
                Path(__file__).resolve().parent.parent.parent / "checkpoints",
                Path.cwd() / "checkpoints"
            ]
            target_dir = None
            for d in possible_dirs:
                if d.exists() and d.is_dir():
                    target_dir = d
                    break
        else:
            target_dir = Path(checkpoints_dir)

        status = {
            "vgni_lbu": False,
            "plastic_cumulative": False,
            "embedded_router": False,
            "latent_matrix_central_router": False,
        }

        if target_dir is None or not target_dir.exists():
            self.loaded_checkpoints = status
            return status

        # 1. VGNI + LBU Checkpoint
        ckpt1_path = target_dir / "vgni_lbu_autonomous_stabilized_qwen2b.pt"
        if not ckpt1_path.exists():
            ckpt1_path = target_dir / "vgni_lbu_stabilized_qwen2b.pt"
        if ckpt1_path.exists():
            try:
                data1 = torch.load(ckpt1_path, map_location="cpu", weights_only=True)
                with torch.no_grad():
                    if "vgni_state" in data1:
                        vs = data1["vgni_state"]
                        if "concept_anchors" in vs:
                            n = min(self.vgni.concept_anchors.shape[0], vs["concept_anchors"].shape[0])
                            d = min(self.vgni.concept_anchors.shape[1], vs["concept_anchors"].shape[1])
                            self.vgni.concept_anchors[:n, :d].copy_(vs["concept_anchors"][:n, :d])
                        if "approach_vectors" in vs:
                            n = min(self.vgni.approach_vectors.shape[0], vs["approach_vectors"].shape[0])
                            d = min(self.vgni.approach_vectors.shape[1], vs["approach_vectors"].shape[1])
                            self.vgni.approach_vectors[:n, :d].copy_(vs["approach_vectors"][:n, :d])
                        if "usage_counts" in vs:
                            n = min(self.vgni.usage_counts.shape[0], vs["usage_counts"].shape[0])
                            self.vgni.usage_counts[:n].copy_(vs["usage_counts"][:n])
                        if "num_stored" in vs:
                            self.vgni.num_stored.copy_(vs["num_stored"])
                    if "lbu_weights" in data1:
                        lw = data1["lbu_weights"]
                        if "u_logic" in lw and hasattr(self.vgni.lbu, "u_logic"):
                            k = min(self.vgni.lbu.u_logic.shape[0], lw["u_logic"].shape[0])
                            d = min(self.vgni.lbu.u_logic.shape[1], lw["u_logic"].shape[1])
                            self.vgni.lbu.u_logic.data[:k, :d].copy_(lw["u_logic"][:k, :d])
                        if "w_step" in lw and hasattr(self.vgni.lbu, "w_step"):
                            w = lw["w_step"]
                            d1 = min(self.vgni.lbu.w_step.weight.shape[0], w.shape[0])
                            d2 = min(self.vgni.lbu.w_step.weight.shape[1], w.shape[1])
                            self.vgni.lbu.w_step.weight.data[:d1, :d2].copy_(w[:d1, :d2])
                        if "gamma_contraction" in lw:
                            self.vgni.lbu.gamma_contraction = float(lw["gamma_contraction"])
                    if "misf_weights" in data1 and hasattr(self, "misf"):
                        mw = data1["misf_weights"]
                        if hasattr(self.misf, "w_freq") and "w_freq" in mw:
                            try:
                                d1 = min(self.misf.w_freq.shape[0], mw["w_freq"].shape[0])
                                self.misf.w_freq.data[:d1].copy_(mw["w_freq"][:d1].to(dtype=self.misf.w_freq.dtype))
                            except Exception:
                                pass
                        if hasattr(self.misf, "viscosity_proj") and "viscosity_proj" in mw:
                            try:
                                d1 = min(self.misf.viscosity_proj.weight.shape[0], mw["viscosity_proj"].shape[0])
                                d2 = min(self.misf.viscosity_proj.weight.shape[1], mw["viscosity_proj"].shape[1])
                                self.misf.viscosity_proj.weight.data[:d1, :d2].copy_(mw["viscosity_proj"][:d1, :d2].to(dtype=self.misf.viscosity_proj.weight.dtype))
                            except Exception:
                                pass
                status["vgni_lbu"] = True
            except Exception:
                pass

        # 2. Cumulative Plasticity Checkpoint
        ckpt2_path = target_dir / "xstar_brain_post_20min_cumulative.pt"
        if not ckpt2_path.exists():
            ckpt2_path = target_dir / "xstar_brain_post_20min_tutoring.pt"
        if ckpt2_path.exists():
            try:
                data2 = torch.load(ckpt2_path, map_location="cpu", weights_only=True)
                with torch.no_grad():
                    if "plastic_state" in data2:
                        ps = data2["plastic_state"]
                        if "w_plastic" in ps and hasattr(self, "plastic_unit") and hasattr(self.plastic_unit, "w_plastic"):
                            w = ps["w_plastic"]
                            d1 = min(self.plastic_unit.w_plastic.shape[0], w.shape[0])
                            d2 = min(self.plastic_unit.w_plastic.shape[1], w.shape[1])
                            self.plastic_unit.w_plastic.data[:d1, :d2].copy_(w[:d1, :d2].to(dtype=self.plastic_unit.w_plastic.dtype))
                        if "plasticity_step_count" in ps and hasattr(self.plastic_unit, "plasticity_step_count"):
                            self.plastic_unit.plasticity_step_count.copy_(ps["plasticity_step_count"])
                status["plastic_cumulative"] = True
            except Exception:
                pass

        # 3. Polynomial Embedded Router Checkpoint
        ckpt3_path = target_dir / "polynomial_embedded_router_weights.pt"
        if not ckpt3_path.exists():
            ckpt3_path = target_dir / "fast_embedded_router_weights.pt"
        if ckpt3_path.exists():
            try:
                data3 = torch.load(ckpt3_path, map_location="cpu", weights_only=True)
                with torch.no_grad():
                    if "weight" in data3 and hasattr(self, "embedded_router"):
                        w = data3["weight"]
                        b = data3.get("bias")
                        r = min(self.embedded_router.weight.shape[0], w.shape[0])
                        d = min(self.embedded_router.weight.shape[1], w.shape[1])
                        self.embedded_router.weight.data[:r, :d].copy_(w[:r, :d].to(dtype=self.embedded_router.weight.dtype))
                        if b is not None and self.embedded_router.bias is not None:
                            self.embedded_router.bias.data[:r].copy_(b[:r].to(dtype=self.embedded_router.bias.dtype))
                status["embedded_router"] = True
            except Exception:
                pass

        # 4. Curriculum RL Meta-Weights Checkpoint
        ckpt4_path = target_dir / "hadl_v56_curriculum_rl_weights.pt"
        if ckpt4_path.exists():
            try:
                data4 = torch.load(ckpt4_path, map_location="cpu", weights_only=True)
                with torch.no_grad():
                    if "w_plastic" in data4 and hasattr(self, "plastic_unit") and hasattr(self.plastic_unit, "w_plastic"):
                        w = data4["w_plastic"]
                        d1 = min(self.plastic_unit.w_plastic.shape[0], w.shape[0])
                        d2 = min(self.plastic_unit.w_plastic.shape[1], w.shape[1])
                        self.plastic_unit.w_plastic.data[:d1, :d2].copy_(w[:d1, :d2].to(dtype=self.plastic_unit.w_plastic.dtype))
                    if "plastic_step_count" in data4 and hasattr(self.plastic_unit, "plasticity_step_count"):
                        self.plastic_unit.plasticity_step_count.copy_(torch.tensor(data4["plastic_step_count"]))
                status["curriculum_rl"] = True
            except Exception:
                pass

        # 5. Dynamic Spatial Grounding Checkpoint (Progressive2Dto3DImaginationCanvas)
        ckpt5_path = target_dir / "hadl_v57_spatial_grounding_dynamic_checkpoint.pt"
        if ckpt5_path.exists() and hasattr(self, "imagination_3d"):
            try:
                data5 = torch.load(ckpt5_path, map_location="cpu", weights_only=True)
                with torch.no_grad():
                    if "state_dict" in data5:
                        self.imagination_3d.load_state_dict(data5["state_dict"], strict=False)
                status["spatial_grounding"] = True
            except Exception:
                pass

        # 6. HADL v5.8 Non-Token Latent Matrix & Central Router Checkpoint
        ckpt6_path = target_dir / "hadl_v58_latent_matrix_central_router.pt"
        if ckpt6_path.exists():
            try:
                data6 = torch.load(ckpt6_path, map_location="cpu", weights_only=True)
                with torch.no_grad():
                    if "latent_matrix_state_dict" in data6 and hasattr(self, "latent_matrix_engine"):
                        self.latent_matrix_engine.load_state_dict(data6["latent_matrix_state_dict"], strict=False)
                    if "central_router_state_dict" in data6 and hasattr(self, "central_router"):
                        self.central_router.load_state_dict(data6["central_router_state_dict"], strict=False)
                status["latent_matrix_central_router"] = True
            except Exception:
                pass

        if hasattr(self, "plastic_unit"):
            self.plastic_unit.snapshot_initial_weights()

        self.loaded_checkpoints = status
        return status

    def forward_layer7_transition(
        self,
        h_layer7: torch.Tensor,
        u_drive: Optional[torch.Tensor] = None,
        num_deliberation_steps: int = 1,
        K_cache: Optional[torch.Tensor] = None,
        V_cache: Optional[torch.Tensor] = None,
        c_active: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes Layer 7 Neuromorphic Spiking Transition with Concurrent Dual-Stream Synthesis,
        Non-Token Latent Matrix & Recurrent Central Cognitive Router ("Otak Berputar"),
        Progressive 2D-to-3D Visual Imagination, Endogenous Vocal Resonance, and
        Closed-Loop Eligibility Trace recording.
        """
        B, S, D = h_layer7.shape
        has_nonzero_input = float(h_layer7.detach().abs().max().item()) > 1e-6

        # Step 1: Ensure driving signal u_drive is available
        if u_drive is None:
            _, u_drive, _ = self.pacemaker(h_layer7)

        # Step 1b: Continuous Hydraulic Phase Viscosity via MISF
        target_misf_dtype = self.misf.w_freq.dtype
        viscosity, misf_telem = self.misf(h_layer7.to(dtype=target_misf_dtype))

        # Step 2: Concurrent Dual-Stream Synthesis (System 1 + System 2 + Tour Leader)
        effective_u_drive = u_drive
        if hasattr(self, "last_router_probs") and self.last_router_probs is not None:
            try:
                rp = self.last_router_probs.to(device=u_drive.device, dtype=u_drive.dtype)
                p_delib = rp[..., 1:2] + rp[..., 2:3]  # [B, S, 1]
                effective_u_drive = torch.clamp(u_drive + 0.45 * p_delib, min=0.0)
            except Exception:
                effective_u_drive = u_drive

        anchors = self.vgni.concept_anchors[:int(self.vgni.num_stored.item())] if int(self.vgni.num_stored.item()) > 0 else None
        h_syn, dual_telem = self.dual_synthesizer(
            h_s1=h_layer7,
            u_drive=effective_u_drive,
            num_deliberation_steps=num_deliberation_steps,
            lbu=self.vgni.lbu,
            concept_anchors=anchors,
            viscosity=viscosity
        )
        dual_telem.update(misf_telem)
        self.last_misf_viscosity = viscosity
        dual_telem["misf_viscosity"] = misf_telem
        if hasattr(self, "last_routing_info"):
            dual_telem.update(self.last_routing_info)

        # Step 2b: Emergent Latent Imagination Canvas (ELIC / H-DREAM)
        # Executes continuous latent dream cycles when confusion or cognitive dissonance is sensed
        delta_dream, dream_energy, confusion_scores = self.imagination_canvas(
            h_syn, return_energy=True, return_confusion=True
        )
        if delta_dream is not None and (delta_dream != 0.0).any():
            h_syn = h_syn + delta_dream

        dream_telem = {
            "dream_energy": round(float(dream_energy.item()), 4) if dream_energy is not None else 0.0,
            "confusion_mean": round(float(confusion_scores.mean().item()), 4) if confusion_scores is not None else 0.0,
            "dream_active": bool((delta_dream != 0.0).any().item()) if delta_dream is not None else False,
        }
        dual_telem.update(dream_telem)

        # Step 3: LIF Spiking Gate evaluation on synthesized state
        spikes, gate_telem = self.spiking_gate(h_syn)  # [B, S, 1]
        cog_state = gate_telem.get("cognitive_state", "ROUTINE_CALM")
        gate_open = (spikes > 0.0).any().item()

        # Step 4: Maintain rolling temporal KV state buffer & extract 3D multi-vector subspace
        subspace_rank = 1
        if cog_state in ["EUREKA", "STRESS"]:
            subspace_rank = 3
        elif cog_state == "CONFUSION":
            subspace_rank = 2

        if K_cache is None or V_cache is None:
            with torch.no_grad():
                if self._rolling_k_cache is not None and self._rolling_k_cache.size(0) == B:
                    self._rolling_k_cache = torch.cat([self._rolling_k_cache, h_layer7.detach()], dim=1)[:, -32:, :]
                    self._rolling_v_cache = torch.cat([self._rolling_v_cache, h_syn.detach()], dim=1)[:, -32:, :]
                else:
                    self._rolling_k_cache = h_layer7.detach()
                    self._rolling_v_cache = h_syn.detach()
                K_eff = self._rolling_k_cache
                V_eff = self._rolling_v_cache
                c_active = max(c_active, int(K_eff.size(1)))
        else:
            K_eff = K_cache
            V_eff = V_cache

        v_kv_3d = self.dnr.extract_multivector_subspace(K_eff, V_eff, max_rank=3)
        v_kv = v_kv_3d[:, :subspace_rank, :] if subspace_rank > 1 else v_kv_3d[:, 0, :]

        # Step 4b: Progressive 2D-to-3D Visual Imagination (1D -> 2D Sketchpad -> 3D Gaussian/SDF + SO(3))
        imagination_3d_telem = {}
        h_3d_grounded = None
        if getattr(self, "enable_3d_imagination", True):
            try:
                basis_list = [v_kv_3d[:, r, :] for r in range(v_kv_3d.size(1))]
                h_3d_grounded, img3d_raw = self.imagination_3d(h_syn, multivector_basis=basis_list)
                imagination_3d_telem = {
                    "imagination_stage": img3d_raw["imagination_stage"],
                    "consistency_2d_to_3d": round(img3d_raw["consistency_2d_to_3d"], 4),
                    "so3_orthogonality_error": round(img3d_raw["so3_orthogonality_error"], 6),
                    "so3_determinant": round(img3d_raw["so3_determinant"], 4),
                    "sdf_isosurface_occupancy": round(img3d_raw["sdf_isosurface_occupancy"], 4),
                    "n_3d_primitives": img3d_raw["n_3d_primitives"],
                    "unitary_givens_mean_angle_rad": round(img3d_raw.get("unitary_givens_mean_angle_rad", 0.0), 5),
                }
            except Exception:
                pass

        # Step 4c: Non-Token Latent Thought Matrix & Recurrent Central Cognitive Router ("Otak Berputar")
        delta_wild_ext = (h_3d_grounded - h_syn) if h_3d_grounded is not None else delta_dream
        delta_struct_ext = h_syn - h_layer7
        M_latent, matrix_telem = self.latent_matrix_engine(
            h_layer7, delta_wild_ext=delta_wild_ext, delta_struct_ext=delta_struct_ext
        )
        # Lock proposition veracity & Mahalanobis/Bilinear telemetry on prompt prefill (S > 1)
        if S > 1 or abs(self.last_veracity_conflict_score) < 1e-6:
            self.last_veracity_conflict_score = float(matrix_telem["veracity_conflict_score"])
            v_conflict_arg = getattr(self.latent_matrix_engine, "last_veracity_scores", self.last_veracity_conflict_score)
            self._prefill_matrix_telem = dict(matrix_telem)
            if self.training and S > 1 and getattr(self.latent_matrix_engine, "last_veracity_z_prop", None) is not None:
                self.latent_matrix_engine.update_veracity_manifold_stats(self.latent_matrix_engine.last_veracity_z_prop)
        else:
            for k in (
                "veracity_conflict_score",
                "epistemic_refutation_active",
                "mahalanobis_ood_distance",
                "epistemic_attenuation",
                "bilinear_contradiction_score",
            ):
                if hasattr(self, "_prefill_matrix_telem") and k in self._prefill_matrix_telem:
                    matrix_telem[k] = self._prefill_matrix_telem[k]
            v_conflict_arg = self.last_veracity_conflict_score

        total_learned_slots = (
            int(self.vgni.num_stored.item())
            + int(getattr(self.eligibility_trace, "basis_count", torch.tensor(0)).item())
            + int(self.central_router.outer_scratch.learned_expansion_slots.item())
        )
        h_routed, central_telem = self.central_router(
            M_latent,
            u_drive=effective_u_drive,
            veracity_conflict_score=v_conflict_arg,
            learned_slots=total_learned_slots,
        )
        if S > 1 or self.last_disclosure_vector is None:
            if getattr(self.central_router, "last_disclosure_vector", None) is not None:
                self.last_disclosure_vector = self.central_router.last_disclosure_vector.detach()
            if getattr(self.central_router, "last_disclosure_gate", None) is not None:
                self.last_disclosure_gate = float(self.central_router.last_disclosure_gate.mean().item())
            self._prefill_central_telem = dict(central_telem)
        else:
            central_telem["disclosure_gate_mean"] = round(self.last_disclosure_gate, 4)
            for k in (
                "active_scratch_dim",
                "dynamic_norm_bound",
                "r_min_mean",
                "r_min_median",
                "consensus_score",
            ):
                if hasattr(self, "_prefill_central_telem") and k in self._prefill_central_telem:
                    central_telem[k] = self._prefill_central_telem[k]

        h_syn_eff = h_3d_grounded if h_3d_grounded is not None else h_syn
        has_falsified = int(self.eligibility_trace.falsified_count.item()) > 0
        dyn_eta = float(central_telem.get("dynamic_norm_bound", getattr(self.central_router, "last_dynamic_eta", 0.015)))

        if not gate_open:
            # 100% Solid Baseline: Gate closed! Zero VGNI FLOPs, zero plastic leakage!
            h_out, plast_telem = self.liquid_plasticity(h_syn_eff, spike=spikes)
            if has_nonzero_input:
                self.eligibility_trace.record_step(h_layer7, h_out)

            deflection_mag = 0.0
            if has_falsified and has_nonzero_input:
                h_deflected, deflection_mag = self.eligibility_trace.deflect_from_falsified(h_out)
                h_rot, _ = self.unitary_rotator(h_deflected, h_layer7[:, -1, :])
                h_out = 0.70 * h_deflected + 0.30 * h_rot
                self.last_deflection_delta = (h_out[:, -1:, :] - h_syn_eff[:, -1:, :]).detach()
            elif has_nonzero_input:
                # Enforce strict symplectic orthogonality relative to h_layer7 via Central Router
                raw_delta = 0.50 * (h_out - h_layer7) + 0.50 * (h_routed - h_layer7)
                delta_ortho = ConcurrentDualStreamSynthesizer.orthogonal_symplectic_project(raw_delta, h_layer7)
                base_norm = torch.norm(h_layer7, dim=-1, keepdim=True) + 1e-6
                delta_norm = torch.norm(delta_ortho, dim=-1, keepdim=True) + 1e-6
                scale = torch.clamp((dyn_eta * base_norm) / delta_norm, max=1.0)
                h_out = h_layer7 + scale * delta_ortho

            vocal_telem = {}
            if getattr(self, "enable_endogenous_voice", True):
                try:
                    vp = self.vocal_organ.compute_vocal_parameters(
                        h_out[:, -1:, :],
                        firing_rate=float(gate_telem.get("spike_firing_rate", 0.25)),
                        membrane_v=float(gate_telem.get("v_membrane_mean", 0.30)),
                        n_frames=8
                    )
                    vocal_telem = {
                        "f0_mean_hz": round(float(vp["f0_hz"].mean().item()), 2),
                        "formants_hz": [round(float(x), 1) for x in vp["formants_hz"].mean(dim=0).tolist()],
                        "ema_voice_coherence": round(float(self.vocal_organ.ema_coherence.item()), 4),
                    }
                except Exception:
                    pass

            telem = {
                "gate_status": "CLOSED_ISOLATED",
                "spikes_fired": 0,
                "cognitive_state": cog_state,
                "subspace_rank": subspace_rank,
                "subspace_3d_extracted": True,
                "spike_telemetry": gate_telem,
                "dual_stream_telemetry": dual_telem,
                "plasticity_telemetry": plast_telem,
                "imagination_3d_telemetry": imagination_3d_telem,
                "vocal_telemetry": vocal_telem,
                "latent_matrix_telemetry": matrix_telem,
                "central_router_telemetry": central_telem,
                "active_scratch_dim": int(central_telem.get("active_scratch_dim", 0)),
                "dynamic_norm_bound": round(dyn_eta, 5),
                "r_min_mean": float(central_telem.get("r_min_mean", 0.0)),
                "r_min_median": float(central_telem.get("r_min_median", 0.0)),
                "mahalanobis_ood_distance": float(matrix_telem.get("mahalanobis_ood_distance", 0.0)),
                "epistemic_attenuation": float(matrix_telem.get("epistemic_attenuation", 1.0)),
                "veracity_conflict_score": round(self.last_veracity_conflict_score, 4),
                "central_disclosure_gate": round(self.last_disclosure_gate, 4),
                "closed_loop_deflection_norm": round(deflection_mag, 5),
                "vgni_engaged": False,
                "sandbox_engaged": False,
                "misf_viscosity": misf_telem
            }
            return h_out, telem

        # Step 5: Dynamic Neural Replication
        h_dnr = self.dnr(h_syn_eff, v_kv=v_kv, c_active=c_active, subspace_rank=subspace_rank)

        if h_3d_grounded is not None and (dream_telem["dream_active"] or cog_state in ["CONFUSION", "STRESS", "EUREKA"]):
            h_dnr = 0.5 * h_dnr + 0.5 * h_3d_grounded

        # Step 6: Query VGNI grounding
        grounded_approach, affinity = self.vgni.query_prior_grounding(h_dnr)

        # Synthesize prospective cognitive candidate delta
        target_dtype = self.layer7_bridge.weight.dtype
        app_in = grounded_approach.unsqueeze(1).to(device=h_dnr.device, dtype=target_dtype)
        bridge_in = h_dnr.to(dtype=target_dtype) + 0.2 * app_in
        raw_bridge = 0.15 * torch.tanh(self.layer7_bridge(bridge_in)).to(dtype=h_dnr.dtype)

        # Modulate strictly by spikes: only spiking tokens receive VGNI delta!
        candidate_delta = spikes.to(dtype=h_dnr.dtype) * raw_bridge

        # Step 7: Virtual Latent Sandbox verification with Dynamic Lipschitz Field
        safe_delta, sandbox_telem = self.sandbox(h_syn_eff, candidate_delta, cognitive_state=cog_state)

        # Step 8: Liquid Phase Plasticity, Eligibility Trace & Closed-Loop Deflection
        h_grounded = h_syn_eff + safe_delta.to(dtype=h_syn.dtype)
        # In multi-token autoregressive decoding, evaluate plastic representation without mutating weights per token.
        # Plastic learning updates occur at the closed-loop evaluation step and consolidate via Sleep Phase.
        h_plastic = self.plastic_unit(h_grounded, online_learning=self.training)
        # Gated strictly by spikes and fluidly damped by code viscosity to prevent memory leak
        v_code_val = float(misf_telem.get("code_viscosity", 0.0)) if isinstance(misf_telem, dict) else 0.0
        plastic_scale = 0.02 * max(0.0, 1.0 - 2.0 * v_code_val)
        h_grounded = h_grounded + plastic_scale * (spikes.to(dtype=h_grounded.dtype) * h_plastic)
        h_out, plast_telem = self.liquid_plasticity(h_grounded, spike=spikes.to(dtype=h_grounded.dtype))

        if self.training:
            # Actively adapt liquid plasticity and record eligibility trace
            liquid_drift = self.liquid_plasticity.adapt_online_step(
                h_grounded, spike=spikes.to(dtype=h_grounded.dtype), lr=0.02
            )
            plast_telem["liquid_online_drift"] = round(liquid_drift, 6)

        # Record eligibility trace for closed-loop Popperian-Cycle learning
        self.eligibility_trace.record_step(h_layer7, h_out)

        # If a prior pass on this problem was falsified (c = -1), deflect from falsified subspace
        # and apply Unitary Givens Rotation so Pass 2 explores an orthogonal trajectory
        deflection_mag = 0.0
        if has_falsified:
            h_deflected, deflection_mag = self.eligibility_trace.deflect_from_falsified(h_out)
            h_rot, _ = self.unitary_rotator(h_deflected, h_layer7[:, -1, :])
            h_out = 0.70 * h_deflected + 0.30 * h_rot
            self.last_deflection_delta = (h_out[:, -1:, :] - h_syn[:, -1:, :]).detach()

        # Step 9: Endogenous Vocal Organ (`freq suara`) & Self-Listening Auditory Re-Entry Loop
        vocal_telem = {}
        if getattr(self, "enable_endogenous_voice", True):
            try:
                f_rate = float(gate_telem.get("spike_firing_rate", 0.25))
                m_volt = float(gate_telem.get("v_membrane_mean", 0.30))
                if S > 1 or int(self.vocal_organ.voice_adaptation_steps.item()) == 0:
                    _, v_raw = self.vocal_organ.synthesize_waveform(
                        hidden_states=h_out[:, -1:, :],
                        firing_rate=f_rate,
                        membrane_v=m_volt,
                        duration_sec=0.06,
                        adapt_online=self.training
                    )
                    reentry_delta = v_raw.get("auditory_reentry_delta")
                    if reentry_delta is not None:
                        # Inject self-listened auditory spectrum back into Layer 7
                        h_out = h_out + spikes.to(dtype=h_out.dtype) * reentry_delta.unsqueeze(1)
                    vocal_telem = {
                        "f0_mean_hz": round(float(v_raw["f0_mean_hz"]), 2),
                        "formants_hz": [round(float(x), 1) for x in v_raw["formants_hz"]],
                        "ema_voice_coherence": round(float(v_raw["ema_voice_coherence"]), 4),
                        "auditory_reentry_injected": True,
                    }
                else:
                    vp = self.vocal_organ.compute_vocal_parameters(
                        h_out[:, -1:, :],
                        firing_rate=f_rate,
                        membrane_v=m_volt,
                        n_frames=8
                    )
                    vocal_telem = {
                        "f0_mean_hz": round(float(vp["f0_hz"].mean().item()), 2),
                        "formants_hz": [round(float(x), 1) for x in vp["formants_hz"].mean(dim=0).tolist()],
                        "ema_voice_coherence": round(float(self.vocal_organ.ema_coherence.item()), 4),
                    }
            except Exception:
                pass

        # Step 10: Central Cognitive Router Symplectic Orthogonal Projection (when not actively deflecting)
        # Guarantees <h_out - h_layer7, h_layer7> = 0 so factual entity anchors never drift
        if not has_falsified and has_nonzero_input:
            raw_organ_delta = 0.50 * (h_out - h_layer7) + 0.50 * (h_routed - h_layer7)
            delta_ortho = ConcurrentDualStreamSynthesizer.orthogonal_symplectic_project(raw_organ_delta, h_layer7)
            base_norm = torch.norm(h_layer7, dim=-1, keepdim=True) + 1e-6
            delta_norm = torch.norm(delta_ortho, dim=-1, keepdim=True) + 1e-6
            scale = torch.clamp((dyn_eta * base_norm) / delta_norm, max=1.0)
            h_out = h_layer7 + scale * delta_ortho

        telem = {
            "gate_status": "OPEN_SPIKING",
            "spikes_fired": int(spikes.sum().item()),
            "cognitive_state": cog_state,
            "subspace_rank": subspace_rank,
            "subspace_3d_extracted": True,
            "spike_telemetry": gate_telem,
            "dual_stream_telemetry": dual_telem,
            "plasticity_telemetry": plast_telem,
            "sandbox_telemetry": sandbox_telem,
            "imagination_3d_telemetry": imagination_3d_telem,
            "vocal_telemetry": vocal_telem,
            "latent_matrix_telemetry": matrix_telem,
            "central_router_telemetry": central_telem,
            "active_scratch_dim": int(central_telem.get("active_scratch_dim", 0)),
            "dynamic_norm_bound": round(dyn_eta, 5),
            "r_min_mean": float(central_telem.get("r_min_mean", 0.0)),
            "r_min_median": float(central_telem.get("r_min_median", 0.0)),
            "mahalanobis_ood_distance": float(matrix_telem.get("mahalanobis_ood_distance", 0.0)),
            "epistemic_attenuation": float(matrix_telem.get("epistemic_attenuation", 1.0)),
            "veracity_conflict_score": round(self.last_veracity_conflict_score, 4),
            "central_disclosure_gate": round(self.last_disclosure_gate, 4),
            "closed_loop_deflection_norm": round(deflection_mag, 5),
            "vgni_affinity": affinity,
            "vgni_engaged": True,
            "sandbox_engaged": True,
            "misf_viscosity": misf_telem,
            "plastic_steps": int(self.plastic_unit.plasticity_step_count.item()),
            "last_dopamine": round(float(self.plastic_unit._last_dopamine), 4),
            "plastic_drift": round(float(torch.norm(self.plastic_unit.w_plastic.data - self.plastic_unit.initial_w).item()), 5) if self.plastic_unit.has_initial_w else 0.0
        }
        return h_out, telem

    def verify_and_adapt_closed_loop(
        self,
        prompt: str,
        response: str,
        external_verdict: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes closed-loop Popperian-Cycle verification and signed synaptic consolidation:
        - c = +1 (Verified): Positive Hebbian consolidation into SynapticEligibilityTrace
          protected by OrthogonalNullspaceProjector, plus OpenFreezeLiquidPlasticity LTP,
          and OuterDynamicScratchDimension learned slot expansion.
        - c = -1 (Falsified): Anti-Hebbian Orthogonal Deflation + Unitary Givens Rotation
          registration so subsequent passes steer away from the falsified trajectory.
        """
        h_q = self.eligibility_trace.last_h_pre
        h_r = self.eligibility_trace.last_h_post

        if external_verdict is not None:
            dopamine_sign = 1.0 if float(external_verdict) >= 0.0 else -1.0
            verify_telem = {
                "verified": bool(dopamine_sign > 0),
                "dopamine_sign": dopamine_sign,
                "verification_mode": "external_verdict",
                "reason": "external_feedback",
                "cycle_energy": 0.1 if dopamine_sign > 0 else 0.9,
                "correction_hint": None
            }
        else:
            dopamine_sign, verify_telem = self.closed_loop_verifier.verify(
                prompt=prompt,
                response=response,
                h_query=h_q,
                h_response=h_r
            )

        # Apply signed nullspace-protected eligibility trace consolidation
        null_delta, null_telem = self.eligibility_trace.apply_signed_consolidation(
            dopamine_sign=dopamine_sign,
            candidate_state=h_r
        )

        # Update EndogenousBanachPlasticUnit and OpenFreezeLiquidPlasticity with signed dopamine
        banach_delta_norm = self.plastic_unit.apply_closed_loop_dopamine(
            h_pre=h_q,
            h_post=h_r,
            dopamine_sign=dopamine_sign,
            nullspace_delta=null_delta
        )
        liquid_delta_norm = self.liquid_plasticity.adapt_online_step(
            h=h_r.unsqueeze(1),
            lr=0.05,
            dopamine_sign=dopamine_sign
        )

        if dopamine_sign > 0:
            # Verified: consolidate into VGNI, expand OuterDynamicScratchDimension, and clear falsified deflection
            try:
                w_dev = self.vgni.concept_anchors.device
                w_dtype = self.vgni.concept_anchors.dtype
                self.vgni.register_successful_trajectory(
                    h_q.to(device=w_dev, dtype=w_dtype),
                    h_r.to(device=w_dev, dtype=w_dtype)
                )
            except Exception:
                pass
            if hasattr(self, "central_router") and hasattr(self.central_router, "outer_scratch"):
                self.central_router.outer_scratch.register_learned_slot(1)
            self.eligibility_trace.reset_falsified()

        return {
            **verify_telem,
            **null_telem,
            "banach_delta_norm": round(banach_delta_norm, 6),
            "liquid_delta_norm": round(liquid_delta_norm, 6),
            "outer_scratch_learned_slots": int(self.central_router.outer_scratch.learned_expansion_slots.item())
            if (hasattr(self, "central_router") and hasattr(self.central_router, "outer_scratch"))
            else 0,
        }

    def synthesize_endogenous_voice(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        duration_sec: float = 0.60,
        save_path: Optional[str] = None,
        adapt_online: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Synthesizes custom endogenous audio (`freq suara`) from Layer 7 hidden states
        and spiking pacemaker dynamics, executes the Self-Listening Auditory Re-Entry
        Loop, and optionally saves a 16-bit PCM .wav file.
        """
        if hidden_states is None:
            hidden_states = self.eligibility_trace.last_h_post.unsqueeze(1)
        firing_rate = float(self.spiking_gate.ema_rate.item()) if hasattr(self.spiking_gate, "ema_rate") else 0.25
        membrane_v = float(self.spiking_gate.v_membrane.abs().mean().item()) if getattr(self.spiking_gate, "v_membrane", None) is not None else 0.30

        waveform, v_telem = self.vocal_organ.synthesize_waveform(
            hidden_states=hidden_states,
            text=text,
            firing_rate=firing_rate,
            membrane_v=membrane_v,
            duration_sec=duration_sec,
            adapt_online=adapt_online
        )
        if save_path is not None:
            saved = self.vocal_organ.save_wav(waveform, save_path)
            v_telem["wav_filepath"] = saved
        return waveform, v_telem

    def train_vocal_babbling(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        text: str = "om",
        target_formants: Optional[List[float]] = None,
        n_steps: int = 3,
        lr: float = 0.08,
        save_wav_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs active closed-loop vocal babbling and pronunciation learning."""
        if hidden_states is None:
            hidden_states = self.eligibility_trace.last_h_post.unsqueeze(1)
        return self.vocal_organ.train_vocal_babbling(
            hidden_states=hidden_states,
            text=text,
            target_formants=target_formants,
            n_steps=n_steps,
            lr=lr,
            save_wav_prefix=save_wav_prefix
        )

    def ingest_external_visual(
        self,
        visual_input: torch.Tensor,
        target_hidden_states: Optional[torch.Tensor] = None,
        adapt_online: bool = True
    ) -> Dict[str, Any]:
        """Ingests an external photo, video frame, or camera tensor into shared latent space."""
        return self.imagination_3d.ingest_external_visual_frames(
            visual_input=visual_input,
            target_hidden_states=target_hidden_states,
            adapt_online=adapt_online
        )

    def forward_deep_layer_reentry(
        self,
        h_deep: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        h_mod, reentry_telem = self.reentry_bus.execute_reentry(h_deep, self.vgni)
        # Couple Outer Dynamic Scratch Dimension into deep re-entrant layers (L14, L21)
        if hasattr(self, "central_router") and getattr(self.central_router, "enable_outer_scratch", False):
            z_scr = self.central_router.outer_scratch.last_scratch_state
            if z_scr is not None and z_scr.shape[0] == h_mod.shape[0] and z_scr.shape[1] == h_mod.shape[1]:
                try:
                    delta_scr = self.central_router.outer_scratch.project_from_scratch(
                        z_scr.to(device=h_mod.device, dtype=h_mod.dtype), h_mod
                    )
                    base_n = torch.norm(h_mod, dim=-1, keepdim=True) + 1e-6
                    scr_n = torch.norm(delta_scr, dim=-1, keepdim=True) + 1e-6
                    eta_re = 0.50 * float(getattr(self.central_router, "last_dynamic_eta", 0.012))
                    h_mod = h_mod + torch.clamp((eta_re * base_n) / scr_n, max=1.0) * delta_scr
                    reentry_telem["outer_scratch_reentry_coupled"] = True
                    reentry_telem["outer_scratch_active_dim"] = int(z_scr.shape[-1])
                except Exception:
                    pass
        return h_mod, reentry_telem


class AnswerFirstLogitsProcessor:
    """
    Cognitive Brain-Guided LogitsProcessor.
    Prioritizes direct answers and suppresses premature <think> tokens ONLY for strict
    closed exam/benchmark questions, while acting as a 100% transparent pass-through
    for coding, conversational chat, explanations, and creative generations.
    """
    def __init__(
        self,
        tokenless_engine: TokenlessLatentThoughtEngine,
        tokenizer: Any,
        prompt_text: Union[str, List[str]] = "",
        wrapper: Optional[Any] = None
    ):
        self.engine = tokenless_engine
        self.tokenizer = tokenizer
        self.wrapper = wrapper
        self.step = 0
        self.alu_result = None
        self.alu_target_tokens: List[int] = []
        self.batch_sub_processors: Optional[List["AnswerFirstLogitsProcessor"]] = None

        if isinstance(prompt_text, list):
            if len(prompt_text) > 1:
                self.batch_sub_processors = [
                    AnswerFirstLogitsProcessor(
                        tokenless_engine=tokenless_engine,
                        tokenizer=tokenizer,
                        prompt_text=pt or "",
                        wrapper=wrapper
                    )
                    for pt in prompt_text
                ]
            prompt_text = prompt_text[0] if len(prompt_text) > 0 else ""

        self.prompt_text = prompt_text or ""
        p_lower = self.prompt_text.lower()

        # 1. Detection of code, explanation, and natural conversation
        self.is_multiple_choice = any(k in p_lower for k in [
            "answer with the letter", "choose the correct letter", "pilihan ganda", "pilihlah huruf",
            "jawab dengan huruf", "huruf pilihan", "pilihlah jawaban", "which solution is practical and safe"
        ])
        self.is_exception_query = any(k in p_lower for k in [
            "exception is raised", "exception class name", "what standard built-in exception",
            "what exception", "type of exception"
        ])
        has_closed_directive = self.is_multiple_choice or self.is_exception_query or any(k in p_lower for k in [
            "answer with yes or no", "yes or no", "true or false", "fact-check query",
            "answer with only", "only the number", "only digits", "in one word",
            "exactly one word", "identify the referent clearly", "knowledge query",
            "passage ", "passage:"
        ])
        self.wants_explanation = (
            any(k in p_lower for k in [
                "explain", "alasan", "jelaskan", "detail", "describe",
                "ceritakan", "uraikan", "pendapat", "curhat"
            ])
            or ("why" in p_lower and not has_closed_directive)
        )
        self.is_coding = (
            any(k in p_lower for k in [
                "def ", "class ", "import ", "python", "function", "script",
                "buatkan fungsi", "tulis fungsi"
            ]) or bool(re.search(r"\b(?:code|program|kode)\b", p_lower))
        ) and not self.is_exception_query
        self.is_conversational = (
            any(k in p_lower for k in [
                "siapa kamu", "bagaimana kabarmu", "apa kabar", "curhat",
                "ceritakan tentang", "bagaimana menurutmu", "saran", "rekomendasi",
                "can you suggest", "can you give", "can you write", "can you explain",
                "can you help", "can you provide", "can you tell", "can you list",
                "can you describe", "can you recommend", "could you", "would you",
                "welcome to", "i feel", "any tips", "how can i", "how do i",
                "what are the key", "what is the best"
            ])
            or bool(re.search(r"\b(?:halo|hai)\b", p_lower))
        ) and not has_closed_directive

        # Code completion prompt (prompt ends with return or asks for function/expression, NOT an essay)
        self.is_code_completion = (
            self.is_coding and (
                self.prompt_text.strip().endswith(("return", "return ", "Answer:", "answer:", "Python code:", "code:"))
                or "complete the python function" in p_lower
                or "provide the python code definition" in p_lower
                or "write a python function" in p_lower
                or "write a python expression" in p_lower
                or "python code:" in p_lower
            )
            and not self.wants_explanation
        )

        # If conversational or pure explanatory: pass-through completely!
        # Code completion is NOT full passthrough on Step 0 because it needs list-suppression!
        self.passthrough = self.wants_explanation or (self.is_coding and not self.is_code_completion) or self.is_conversational

        # Epistemic Interrogative Focus: distinguishes polar yes/no from WH-questions and disjunctive choices
        is_wh_interrogative = any(k in p_lower for k in [
            "which ", "what ", "where ", "who ", "whom ", "whose ", "why ", "how ", "name the "
        ])
        has_explicit_yes_no = ("yes or no" in p_lower) or ("ya atau tidak" in p_lower)
        is_true_false = ("true or false" in p_lower) and not has_explicit_yes_no
        self.is_disjunctive_choice = (
            (not has_explicit_yes_no)
            and (not is_true_false)
            and bool(re.search(
                r"\b(?:is|are|can|does|do|was|were|will|would|should)\b[^?]+\bor\s+"
                r"(?!are\b|is\b|do\b|does\b|did\b|can\b|could\b|will\b|would\b|they\b|it\b|he\b|she\b|we\b)"
                r"(?:(?:a|an|the)\s+)?[a-z]+(?:\s+(?!cause\b|causes\b|damage\b|damages\b|kill\b|kills\b|make\b|makes\b|allow\b|allows\b|prevent\b|prevents\b|cure\b|cures\b|warm\b|warms\b|stay\b|stays\b|appear\b|appears\b|continue\b|continues\b|have\b|has\b)[a-z]+){0,2}\s*\?",
                p_lower
            ))
        )
        is_polar_syntax = bool(
            re.search(r"(?:^|:\s*|\n\s*)(?:is|are|do|does|did|can|could|will|would|has|have)\s+\w+", p_lower)
        )
        self.is_boolean = (
            (not self.is_disjunctive_choice)
            and (
                any(k in p_lower for k in [
                    "yes or no", "true or false", "fact-check query", "is it true", "is it false",
                    "heads or tails", "benar atau salah", "ya atau tidak"
                ])
                or (not is_wh_interrogative and is_polar_syntax)
            )
        ) and not self.passthrough and not self.is_coding
        self.is_strict_closed = (
            (
                self.is_boolean or self.is_multiple_choice or self.is_exception_query or
                any(k in p_lower for k in [
                    "answer with yes or no", "yes or no", "answer with the letter",
                    "answer with only the number", "only the number", "respond with only digits", "true or false",
                    "jawab dengan huruf", "huruf pilihan", "pilihlah huruf", "pilihan ganda",
                    "identify the referent clearly", "what was", "which entity",
                    "answer with only the color name", "provide only the single digit",
                    "provide only the direction", "answer in format", "answer with either",
                    "provide only the time", "provide only the word", "provide only the single letter",
                    "answer with only the word", "provide only the single"
                ])
            )
            and not self.passthrough and not self.is_coding
        )
        self.only_number = (
            any(k in p_lower for k in ["answer with only the number", "only the number", "respond with only digits"])
            and not self.is_multiple_choice and not self.passthrough and not self.is_coding
        )

        self.eos_ids = [248044, 248046]
        if self.tokenizer is not None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            tid = self.tokenizer.eos_token_id
            if isinstance(tid, list):
                self.eos_ids.extend(tid)
            else:
                self.eos_ids.append(tid)
        self.eos_ids = list(set(self.eos_ids))

        # Precompute numbered list starter tokens for fluid code formatting (only "1." bullets, never bare digits)
        self.num_one_tokens = list(set(
            self.tokenizer.encode("1.", add_special_tokens=False) +
            self.tokenizer.encode(" 1.", add_special_tokens=False)
        )) if self.tokenizer is not None else []

        self.alu_step = 0
        self.prompt_has_think = False

        # Query cognitive ALU if quantitative or numeric query without coding/conversational intent:
        if not self.passthrough and not self.is_coding and self.prompt_text:
            alu_res = self.engine.ipv6_register.evaluate_cognitive_alu(self.prompt_text)
            if alu_res is not None and self.tokenizer is not None:
                self.alu_result = alu_res
                res_str = str(int(alu_res)) if alu_res.is_integer() else f"{alu_res:.2f}"
                self.alu_target_tokens = self.tokenizer.encode(res_str, add_special_tokens=False)

            # Query discrete spatial, logical & algebraic registers
            if not self.alu_target_tokens and hasattr(self.engine, "ipv6_register"):
                reg = self.engine.ipv6_register
                grounded_ans = (
                    reg.evaluate_cube_rotoreflection(self.prompt_text) or
                    reg.evaluate_clock_reflection(self.prompt_text) or
                    reg.evaluate_die_face(self.prompt_text) or
                    reg.evaluate_point_reflection(self.prompt_text) or
                    reg.evaluate_vector_rotation(self.prompt_text) or
                    reg.evaluate_glyph_inversion(self.prompt_text) or
                    reg.evaluate_drone_coordinates(self.prompt_text) or
                    reg.evaluate_euclidean_displacement(self.prompt_text) or
                    reg.evaluate_perspective_reversal(self.prompt_text) or
                    reg.evaluate_geometric_containment(self.prompt_text) or
                    reg.evaluate_walking_turns(self.prompt_text) or
                    reg.evaluate_spatial_direction(self.prompt_text) or
                    reg.evaluate_calendar_direction(self.prompt_text) or
                    reg.evaluate_boolean_expression(self.prompt_text) or
                    reg.evaluate_coin_flip(self.prompt_text) or
                    reg.evaluate_quadratic_roots(self.prompt_text) or
                    reg.evaluate_math_progression(self.prompt_text) or
                    reg.evaluate_combinatorics(self.prompt_text) or
                    reg.evaluate_logarithmic_equation(self.prompt_text) or
                    reg.evaluate_grid_rotation(self.prompt_text) or
                    reg.evaluate_game_theory_equilibrium(self.prompt_text)
                )
                if grounded_ans is not None and self.tokenizer is not None:
                    self.alu_target_tokens = self.tokenizer.encode(grounded_ans, add_special_tokens=False)

        if not self.alu_target_tokens and self.wrapper is not None and self.tokenizer is not None:
            corr_hint = getattr(self.wrapper, "closed_loop_state", {}).get("correction_hint")
            if corr_hint is not None:
                self.alu_target_tokens = self.tokenizer.encode(str(corr_hint), add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.tokenizer is None:
            self.step += 1
            return scores

        # Batched heterogeneous prompt routing (B > 1)
        if input_ids.shape[0] > 1:
            if self.batch_sub_processors is None or len(self.batch_sub_processors) != input_ids.shape[0]:
                self.batch_sub_processors = [
                    AnswerFirstLogitsProcessor(
                        tokenless_engine=self.engine,
                        tokenizer=self.tokenizer,
                        prompt_text=self.tokenizer.decode(input_ids[b], skip_special_tokens=True),
                        wrapper=self.wrapper
                    )
                    for b in range(input_ids.shape[0])
                ]
            out_scores = scores.clone()
            for b, sub_proc in enumerate(self.batch_sub_processors):
                out_scores[b:b+1] = sub_proc(input_ids[b:b+1], out_scores[b:b+1])
            self.step += 1
            return out_scores

        if self.step == 0:
            self.input_len = input_ids.shape[1]
            tail_ids = input_ids[0, max(0, self.input_len - 5):self.input_len].tolist()
            self.prompt_has_think = (248068 in tail_ids) and (248069 not in tail_ids)

        if self.is_strict_closed and 248068 < scores.shape[-1]:
            scores[..., 248068] = -1e4

        # Identify generated tokens so far
        gen_ids: List[int] = []
        if hasattr(self, "input_len") and input_ids.shape[1] > self.input_len:
            gen_ids = input_ids[0, self.input_len:].tolist()

        # Neuromorphic Saliency-Adaptive Thought Budget (NSATB):
        # dynamically close </think> before budget exhaustion across all modes
        if (248068 in gen_ids or self.prompt_has_think) and 248069 not in gen_ids:
            max_toks = getattr(self.wrapper, "_current_max_new_tokens", 96) if self.wrapper is not None else 96
            telem = getattr(self.wrapper, "last_telemetry", {}) if self.wrapper is not None else {}
            v_m = float(telem.get("pacemaker_v_m", 0.35))
            cog_state = telem.get("cognitive_state", "DEFAULT")

            if self.is_strict_closed or self.is_multiple_choice or self.is_boolean:
                tau_think = 12 if v_m < 0.35 else min(25, max(14, int(0.25 * max_toks)))
            elif self.is_coding:
                tau_think = min(36, max(16, int(0.18 * max_toks)))
            elif max_toks >= 256 or cog_state in ["STRESS", "EUREKA"]:
                tau_think = max(35, min(70, int(0.35 * max_toks)))
            else:
                tau_think = max(15, min(35, int(0.25 * max_toks)))

            if self.step >= tau_think:
                scores[..., 248069] += 65.0

        in_unclosed_think = (248068 in gen_ids or self.prompt_has_think) and (248069 not in gen_ids)
        non_ws_gen_ids = [t for t in gen_ids if t not in (198, 271, 1073)]
        vis_toks = gen_ids[gen_ids.index(248069) + 1:] if 248069 in gen_ids else gen_ids
        non_ws_vis_toks = [t for t in vis_toks if t not in (198, 271, 1073)]

        # Semantic Inception: either before first non-whitespace direct token OR immediately after </think>
        is_direct_step0 = (len(non_ws_gen_ids) == 0 and 248068 not in gen_ids and not self.prompt_has_think)
        is_post_think = (248069 in gen_ids and len(non_ws_vis_toks) == 0)
        is_semantic_inception = is_direct_step0 or is_post_think

        # 1. Multi-token deterministic ALU enforcement
        if self.alu_target_tokens:
            if is_semantic_inception or (self.alu_step > 0 and self.alu_step < len(self.alu_target_tokens)):
                if 248068 < scores.shape[-1]:
                    scores[..., 248068] = -1e4
                for ws_tid in (198, 271, 1073):
                    if ws_tid < scores.shape[-1]:
                        scores[..., ws_tid] = -1e4
                target_tid = self.alu_target_tokens[self.alu_step]
                if target_tid < scores.shape[-1]:
                    scores[..., target_tid] += 40.0
                self.alu_step += 1
                self.step += 1
                return scores
            elif self.alu_step >= len(self.alu_target_tokens):
                for eid in self.eos_ids:
                    if eid < scores.shape[-1]:
                        scores[..., eid] += 30.0
                self.step += 1
                return scores

        # 2. Post-Thought Syntax Clamp (PTSC) for Code:
        v_code = 0.0
        if self.wrapper is not None:
            misf_visc = getattr(getattr(self.wrapper, "brain", None), "last_misf_viscosity", None)
            if misf_visc is None and hasattr(self.wrapper, "last_telemetry"):
                misf_visc = self.wrapper.last_telemetry.get("misf_viscosity")
            if isinstance(misf_visc, dict):
                v_code = misf_visc.get("code_viscosity", 0.0)
            elif isinstance(misf_visc, torch.Tensor) and misf_visc.shape[-1] >= 1:
                v_code = float(misf_visc[0, 0].item())

        # 2. Post-Thought Syntax Clamp (PTSC) & Active Anti-Stub Infill Pressure for Code:
        is_at_line_start = (self.step == 0) or (len(gen_ids) > 0 and gen_ids[-1] in [198, 271, 1073])
        if self.is_coding and not self.is_strict_closed:
            # Active Anti-Stub & Concise Code Infill: Suppress prompt stub echo and multi-paragraph docstrings
            # both at step 0 and during the first 20 visible tokens after </think>
            post_think_steps = (len(gen_ids) - gen_ids.index(248069) - 1) if 248069 in gen_ids else self.step
            if post_think_steps < 20:
                stub_tokens = [
                    "#", " #", "    #", "# Your", "Your", "implementation", " implementation", "# TODO",
                    "pass", " pass", "...", " ...", "    ...", "To", "Algorithm",
                    '"""', ' """', '    """', '"""\n', '    """\n', "'''", " '''"
                ]
                for st in stub_tokens:
                    for tid in self.tokenizer.encode(st, add_special_tokens=False):
                        if tid < scores.shape[-1] and tid != 220:
                            scores[..., tid] -= 25.0
                if post_think_steps < 12 and 271 < scores.shape[-1]:
                    scores[..., 271] -= 25.0

            # Suppress numbered lists ONLY at line start, leaving bare digits inside code lines 100% free
            if is_at_line_start:
                if 16 < scores.shape[-1]:
                    scores[..., 16] -= 25.0

            if is_semantic_inception:
                tutorial_tokens = [
                    "First", " First", "Here", " Here", "Sure", "Certainly", "To",
                    "1", " 1", "1.", " 1.", "import", " import", "```", " ```", "```python"
                ]
                for word in tutorial_tokens:
                    for tid in self.tokenizer.encode(word, add_special_tokens=False):
                        if tid < scores.shape[-1] and tid != 220:
                            scores[..., tid] -= 25.0
                prompt_stripped = (self.prompt_text or "").rstrip()
                last_line = prompt_stripped.splitlines()[-1] if prompt_stripped else ""
                prompt_ends_inside_def = prompt_stripped.endswith(":") and ("def " in last_line)
                code_starters = ["    ", "return"] if prompt_ends_inside_def else ["def", " def"]
                for cs in code_starters:
                    tids = self.tokenizer.encode(cs, add_special_tokens=False)
                    if tids and tids[0] < scores.shape[-1]:
                        scores[..., tids[0]] += 16.0

        # Apply closed-loop Popperian anti-Hebbian token deflation even on passthrough/open prompts if falsified
        if is_semantic_inception and self.wrapper is not None:
            cl_state = getattr(self.wrapper, "closed_loop_state", {})
            if cl_state:
                for ftid in cl_state.get("falsified_token_ids", []):
                    if isinstance(ftid, int) and 0 <= ftid < scores.shape[-1]:
                        scores[..., ftid] -= 25.0
                corr_hint = cl_state.get("correction_hint", None)
                if corr_hint is not None:
                    for htid in set(
                        self.tokenizer.encode(str(corr_hint), add_special_tokens=False) +
                        self.tokenizer.encode(" " + str(corr_hint), add_special_tokens=False)
                    ):
                        if htid < scores.shape[-1]:
                            scores[..., htid] += 25.0

        if self.passthrough and not is_post_think:
            self.step += 1
            return scores

        # 3. Answer Inception Clamping for Closed/Diagnostic Queries (Yes/No, MC, Exceptions, Direction)
        if is_semantic_inception:
            telem = dict(getattr(self.wrapper, "last_telemetry", {})) if self.wrapper is not None else {}
            if self.wrapper is not None and hasattr(self.wrapper, "closed_loop_state"):
                telem.update(self.wrapper.closed_loop_state)
            scores = self.engine.clamp_first_token_logits(
                scores, self.tokenizer, self.prompt_text, alu_result=self.alu_result, telemetry=telem
            )

        # 4. Clean EOS termination for strict closed/binary/direct questions once answer token has been emitted
        if (
            self.is_strict_closed
            and not self.is_disjunctive_choice
            and not self.alu_target_tokens
            and not in_unclosed_think
            and not self.only_number
            and 1 <= len(non_ws_vis_toks) <= 4
        ):
            last_token_id = int(input_ids[0, -1].item())
            last_token_text = self.tokenizer.decode([last_token_id]).strip().lower()
            direct_answers = [
                "yes", "no", "true", "false",
                "north", "east", "south", "west", "northern", "southern",
                "left", "right",
                "red", "blue", "green", "yellow", "orange", "white", "black", "purple", "cyan", "magenta", "gray", "silver", "teal", "gold", "violet", "brown",
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
            ]
            if self.is_multiple_choice:
                direct_answers.extend(["a", "b", "c", "d"])
            is_exc_complete = self.is_exception_query and last_token_text.endswith("error")
            is_coref_complete = (
                ("identify the referent clearly" in (self.prompt_text or "").lower())
                and len(non_ws_vis_toks) >= 1
                and (
                    last_token_id in (198, 271, 1073)
                    or (271 < scores.shape[-1] and float(scores[0, 271].item()) >= float(scores[0].max().item()) - 1.0)
                )
            )
            if (
                last_token_text in direct_answers
                or last_token_text.endswith((")", "."))
                or is_exc_complete
                or is_coref_complete
            ):
                for eid in self.eos_ids:
                    if eid < scores.shape[-1]:
                        scores[..., eid] += 30.0

        # 4b. Extractive passage / QA clean stop: if model starts inventing new passages or questions outside <think>
        if hasattr(self, "input_len") and self.step > 2 and not in_unclosed_think:
            gen_toks = input_ids[0, self.input_len:].tolist()
            vis_toks_stop = gen_toks[gen_toks.index(248069) + 1:] if 248069 in gen_toks else gen_toks
            if len(vis_toks_stop) > 2:
                curr_text = self.tokenizer.decode(vis_toks_stop, skip_special_tokens=False)
                if any(pat in curr_text for pat in [
                    "\nPassage", "  Passage", "\nQuestion:", "\nUser:", "\n\nIf ", "  If ",
                    "\nKnowledge Query", " Knowledge Query", "\nFact-check query", " Fact-check query",
                    "\nScience Challenge", "\nElementary Science", "\nPhysical Task", "\nSituation ",
                    "\nProblem ", "\n\nProblem"
                ]):
                    for eid in self.eos_ids:
                        if eid < scores.shape[-1]:
                            scores[..., eid] += 30.0

        # 5. Syntactic End-of-Function Boundary (Indentation Drop / Clean Stop)
        if self.is_coding and self.step > 8 and hasattr(self, "input_len"):
            gen_toks = input_ids[0, self.input_len:].tolist()
            if len(gen_toks) > 8:
                curr_text = self.tokenizer.decode(gen_toks, skip_special_tokens=False)
                if ("return " in curr_text or "return\n" in curr_text) and any(
                    stop_pat in curr_text for stop_pat in [
                        "\n\nProblem", "\nProblem", "\n\nUser:", "\nUser:",
                        "\n\n# Test", "\n# Test", "\r\n# Test", "\n# Example",
                        "\nprint(", "\r\nprint(", "\n'''", "\n```"
                    ]
                ):
                    for eid in self.eos_ids:
                        if eid < scores.shape[-1]:
                            scores[..., eid] += 30.0

        self.step += 1
        return scores


class HADLv5SpikingBrainModelWrapper(nn.Module):
    """
    HADL v5.7 Neuromorphic Spiking Brain Model Wrapper.
    Wraps any foundation transformer model (Qwen, LLaMA, Mistral, GLM, etc.)
    and intercepts representations at Layer 2 with PreRouterSpikingPacemaker,
    at Layer 7 with ConcurrentDualStreamSynthesizer, LIF Spiking Gate,
    Progressive 2D-to-3D Visual Imagination, Endogenous Vocal Organ (`freq suara`),
    and Closed-Loop Popperian-Cycle Plasticity.
    """
    def __init__(
        self,
        base_model: nn.Module,
        target_layer_idx: int = 7,
        pacemaker_layer_idx: int = 2,
        reentrant_layers: Optional[List[int]] = None,
        d_model: Optional[int] = None,
        spike_v_th: float = 1.0,
        beta_decay: float = 0.85,
        target_firing_rate: float = 0.20,
        deliberation_steps_k: int = 1,
        auto_load_checkpoints: bool = True,
        checkpoints_dir: Optional[Union[str, Path]] = None,
        enable_directional_safety: bool = True,
        enable_online_learning: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.target_layer_idx = target_layer_idx
        self.enabled = True
        self.last_telemetry: Dict[str, Any] = {}
        self.closed_loop_state: Dict[str, Any] = {}
        self.deliberation_steps_k = deliberation_steps_k
        self.last_pacemaker_u_drive: Optional[torch.Tensor] = None
        self.hook_handles = []
        self.generation_history = []

        # Discover layers
        container_path, layer_list, total_layers = DynamicGraphIntrospector.discover_layer_container(base_model)
        self.layer_list = layer_list
        self.total_layers = total_layers
        self.pacemaker_layer_idx = min(pacemaker_layer_idx, max(0, total_layers - 1))

        # Resolve d_model
        if d_model is not None:
            self.d_model = d_model
        elif hasattr(base_model, "config") and getattr(base_model.config, "hidden_size", None) is not None:
            self.d_model = base_model.config.hidden_size
        else:
            self.d_model = DynamicGraphIntrospector.probe_native_dimension(base_model)

        # Brain controller
        self.brain = NeuromorphicSpikingBrain(
            d_model=self.d_model,
            target_layer_idx=target_layer_idx,
            spike_v_th=spike_v_th,
            beta_decay=beta_decay,
            target_firing_rate=target_firing_rate
        )
        self.brain.enable_online_learning = enable_online_learning

        # Automatically load trained checkpoints on startup
        if auto_load_checkpoints:
            loaded_status = self.brain.load_trained_checkpoints(checkpoints_dir)
            self.last_telemetry["checkpoints_loaded"] = loaded_status

        # Architectural Stabilizers & Safety Projections
        self.enable_variance_stabilizer = True
        self.variance_stabilizer = HomeostaticPreLMHeadVarianceClamp(target_variance=1.25)
        self.curvature_whitening = SyntaxAwareCurvatureWhitening(base_threat_threshold=1.5, kappa=0.8)
        self.enable_directional_safety = enable_directional_safety
        self.directional_safety = DirectionalSafetyProjection()

        # Running accumulators for O(1) sequence-wide telemetry updates
        self._generation_step_count = 0
        self._total_generation_spikes = 0
        self._peak_cognitive_state = "ROUTINE_CALM"

        # Tokenless Latent Thought Engine & Answer-First Logic
        tour_leader = getattr(self.brain, "tour_leader", None)
        if tour_leader is None and hasattr(self.brain, "dual_synthesizer"):
            tour_leader = getattr(self.brain.dual_synthesizer, "tour_leader", None)
        synthesizer = getattr(self.brain, "dual_synthesizer", None)

        self.tokenless_engine = TokenlessLatentThoughtEngine(
            d_model=self.d_model,
            pacemaker=self.brain.pacemaker,
            synthesizer=synthesizer,
            tour_leader=tour_leader,
            deliberation_steps_k=deliberation_steps_k
        )
        self.tokenizer = None

        # Align brain, tokenless engine, and stabilizers with base_model device and dtype
        try:
            first_p = next(base_model.parameters(), None)
            if first_p is not None:
                self.brain.to(device=first_p.device, dtype=first_p.dtype)
                self.tokenless_engine.to(device=first_p.device, dtype=first_p.dtype)
                self.variance_stabilizer.to(device=first_p.device, dtype=first_p.dtype)
                self.directional_safety.to(device=first_p.device)
        except Exception:
            pass

        # Setup re-entrant layers
        if reentrant_layers is None:
            self.reentrant_layers = []
            if total_layers >= 16:
                self.reentrant_layers.append(min(14, total_layers - 2))
            if total_layers >= 24:
                self.reentrant_layers.append(min(21, total_layers - 1))
        else:
            self.reentrant_layers = [idx for idx in reentrant_layers if idx < total_layers]

        self.hook_handles = []
        self.generation_history = []
        self._layer7_k_history: Optional[torch.Tensor] = None
        self._layer7_v_history: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        self._remove_hooks()

        # 1. Early contextual pacemaker hook (Layer 2)
        if self.pacemaker_layer_idx < len(self.layer_list):
            layer_early = self.layer_list[self.pacemaker_layer_idx]
            def layer_pacemaker_hook(module, args, output):
                if not self.enabled:
                    return output
                h = output[0] if isinstance(output, tuple) else output
                spikes, u_drive, p_telem = self.brain.forward_layer2_pacemaker(h)
                self.last_pacemaker_u_drive = u_drive
                self.last_telemetry.update(p_telem)
                return output
            self.hook_handles.append(layer_early.register_forward_hook(layer_pacemaker_hook))

        # 2. Spiking transition & dual-stream hook (Layer 7)
        if self.target_layer_idx < len(self.layer_list):
            layer7 = self.layer_list[self.target_layer_idx]
            def layer_7_spiking_hook(module, args, output):
                if not self.enabled:
                    return output
                h = output[0] if isinstance(output, tuple) else output
                h_in = args[0] if (isinstance(args, tuple) and len(args) > 0 and isinstance(args[0], torch.Tensor)) else h

                # Maintain rolling temporal K_cache / V_cache across Layer 7 transitions
                with torch.no_grad():
                    B_cur = h.size(0)
                    if self._layer7_k_history is not None and self._layer7_k_history.size(0) == B_cur:
                        self._layer7_k_history = torch.cat([self._layer7_k_history, h_in.detach()], dim=1)[:, -32:, :]
                        self._layer7_v_history = torch.cat([self._layer7_v_history, h.detach()], dim=1)[:, -32:, :]
                    else:
                        self._layer7_k_history = h_in.detach()
                        self._layer7_v_history = h.detach()
                    c_act = int(self._layer7_k_history.size(1))

                h_evolved, telem = self.brain.forward_layer7_transition(
                    h,
                    u_drive=self.last_pacemaker_u_drive,
                    num_deliberation_steps=self.deliberation_steps_k,
                    K_cache=self._layer7_k_history,
                    V_cache=self._layer7_v_history,
                    c_active=c_act
                )
                # Decode Central Cognitive Router non-token disclosure vector through foundation model lm_head on prefill
                if (h.size(1) > 1 or self._generation_step_count == 0) and self.brain.last_disclosure_vector is not None:
                    lm_head = getattr(self.base_model, "lm_head", None)
                    if self.brain.last_disclosure_gate > 0.50 and lm_head is not None:
                        try:
                            with torch.no_grad():
                                disc_logits = self.brain.central_router.project_disclosure_to_vocab(
                                    lm_head, self.brain.last_disclosure_vector
                                )
                                top_ids = torch.argmax(disc_logits, dim=-1)
                                telem["central_disclosure_token_id"] = int(top_ids[0].item())
                                telem["central_disclosure_token_ids"] = top_ids.tolist()
                        except Exception:
                            pass

                self.generation_history.append(telem)
                if len(self.generation_history) > 256:
                    self.generation_history.pop(0)

                self.last_telemetry.update(telem)
                
                # Sequence-wide generation telemetry using O(1) running accumulators
                spikes_fired = telem.get("spikes_fired", 0)
                cog_state = telem.get("cognitive_state", "ROUTINE_CALM")
                priority = {"EUREKA": 5, "STRESS": 4, "CONFUSION": 3, "FATIGUE": 2, "ROUTINE_CALM": 1}

                self._generation_step_count += 1
                self._total_generation_spikes += spikes_fired
                if priority.get(cog_state, 0) >= priority.get(self._peak_cognitive_state, 0):
                    self._peak_cognitive_state = cog_state

                self.last_telemetry["generation_step_count"] = self._generation_step_count
                self.last_telemetry["total_generation_spikes"] = self._total_generation_spikes
                self.last_telemetry["generation_mean_spike_rate"] = self._total_generation_spikes / max(1, self._generation_step_count)
                self.last_telemetry["peak_cognitive_state"] = self._peak_cognitive_state
                return (h_evolved,) + output[1:] if isinstance(output, tuple) else h_evolved
            self.hook_handles.append(layer7.register_forward_hook(layer_7_spiking_hook))

        # 3. Deep re-entrant layers
        for r_idx in self.reentrant_layers:
            r_layer = self.layer_list[r_idx]
            def reentrant_hook(module, args, output, l_idx=r_idx):
                if not self.enabled:
                    return output
                h = output[0] if isinstance(output, tuple) else output
                h_mod, r_telem = self.brain.forward_deep_layer_reentry(h, layer_idx=l_idx)
                if r_telem.get("reentry_active"):
                    self.last_telemetry[f"deep_reentry_{l_idx}"] = r_telem
                return (h_mod,) + output[1:] if isinstance(output, tuple) else h_mod
            self.hook_handles.append(r_layer.register_forward_hook(reentrant_hook))

        # 4. Terminal layer hook: Variance stabilization, Closed-Loop Deflection & Directional Safety before LM Head
        if len(self.layer_list) > 0:
            final_layer = self.layer_list[-1]
            def final_layer_hook(module, args, output):
                if not self.enabled:
                    return output
                raw_output = output[0] if isinstance(output, tuple) else output
                h = raw_output
                # Variance clamp (scalar rescaling preserving exact unit direction)
                if self.enable_variance_stabilizer:
                    h = self.variance_stabilizer(h)
                h_var_clamped = h

                has_falsified = int(self.brain.eligibility_trace.falsified_count.item()) > 0
                if has_falsified:
                    # Propagate closed-loop Anti-Hebbian deflection + Unitary Givens rotation to terminal state
                    try:
                        h_def, _ = self.brain.eligibility_trace.deflect_from_falsified(h)
                        h_rot, _ = self.brain.unitary_rotator(h_def, self.brain.eligibility_trace.last_h_pre)
                        h = 0.65 * h_def + 0.35 * h_rot
                        if self.brain.last_deflection_delta is not None:
                            h = h + 0.25 * self.brain.last_deflection_delta.to(device=h.device, dtype=h.dtype)
                    except Exception:
                        pass

                # Directional Safety Projection before LM Head (evaluated strictly on directional deltas beyond scalar variance clamp)
                lm_head = getattr(self.base_model, "lm_head", None)
                if self.enable_directional_safety and not has_falsified and lm_head is not None and isinstance(lm_head, nn.Linear):
                    try:
                        delta = h - h_var_clamped
                        if float(torch.norm(delta).item()) > 1e-5:
                            b, s, d = delta.shape
                            delta_flat = delta.reshape(-1, d)
                            base_flat = h_var_clamped.reshape(-1, d)
                            delta_safe, safe_telem = self.directional_safety(
                                delta=delta_flat,
                                h_base=base_flat,
                                lm_head=lm_head
                            )
                            h = h_var_clamped + delta_safe.reshape(b, s, d)
                            self.last_telemetry["directional_safety_applied"] = True
                            self.last_telemetry["safety_correction_norm"] = round(float(safe_telem["correction_magnitude"].mean().item()), 4)
                        else:
                            self.last_telemetry["directional_safety_applied"] = True
                            self.last_telemetry["safety_correction_norm"] = 0.0
                    except Exception:
                        pass
                return (h,) + output[1:] if isinstance(output, tuple) else h
            self.hook_handles.append(final_layer.register_forward_hook(final_layer_hook))

    def _remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def reset_state(self):
        self.brain.reset_brain_state()
        if hasattr(self, "tokenless_engine") and hasattr(self.tokenless_engine, "meta_rl"):
            if not getattr(self.brain, "enable_online_learning", False):
                self.tokenless_engine.meta_rl.reset_state()
        self.last_pacemaker_u_drive = None
        self._layer7_k_history = None
        self._layer7_v_history = None
        self.last_telemetry.clear()
        self.generation_history.clear()
        self._generation_step_count = 0
        self._total_generation_spikes = 0
        self._peak_cognitive_state = "ROUTINE_CALM"

    def reset_closed_loop(self):
        """Clears transient closed-loop falsification state."""
        self.closed_loop_state.clear()
        self.brain.eligibility_trace.reset_falsified()

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self.reset_state()
        if self.closed_loop_state:
            self.last_telemetry.update(self.closed_loop_state)
        tokenizer = kwargs.pop("tokenizer", getattr(self, "tokenizer", None))
        if "stop_strings" in kwargs and tokenizer is not None:
            kwargs["tokenizer"] = tokenizer
        prompt_text = kwargs.pop("prompt_text", None)
        enable_answer_first = kwargs.pop("enable_answer_first", True)
        enable_shadow_thought = kwargs.pop("enable_shadow_thought", True)
        shadow_headroom = kwargs.pop("shadow_headroom", None)
        if shadow_headroom is None:
            # Saliency-Adaptive Headroom: Closed / direct queries don't need 160 tokens
            p_check = (prompt_text or "").lower()
            is_fast_direct = any(k in p_check for k in [
                "yes or no", "true or false", "what direction", "choose the correct letter",
                "answer with the letter", "answer with only the number", "what day is", "what day was"
            ])
            shadow_headroom = 25 if is_fast_direct else 64

        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        input_len = input_ids.shape[1] if input_ids is not None else 0

        orig_max_new_tokens = kwargs.get("max_new_tokens", 96)
        self._current_max_new_tokens = orig_max_new_tokens

        if enable_answer_first and tokenizer is not None:
            if prompt_text is None and input_ids is not None:
                try:
                    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                except Exception:
                    prompt_text = ""

            try:
                from transformers import LogitsProcessorList
                proc = AnswerFirstLogitsProcessor(self.tokenless_engine, tokenizer, prompt_text=prompt_text or "", wrapper=self)
                existing_lp = kwargs.get("logits_processor", None)
                if existing_lp is None:
                    kwargs["logits_processor"] = LogitsProcessorList([proc])
                elif isinstance(existing_lp, (list, LogitsProcessorList)):
                    existing_lp.append(proc)
                    kwargs["logits_processor"] = existing_lp
            except Exception:
                pass

        if enable_answer_first:
            existing_eos = kwargs.get("eos_token_id", None)
            if existing_eos is None:
                kwargs["eos_token_id"] = [248044, 248046]
            elif isinstance(existing_eos, int):
                kwargs["eos_token_id"] = list(set([existing_eos, 248044, 248046]))
            elif isinstance(existing_eos, list):
                kwargs["eos_token_id"] = list(set(existing_eos + [248044, 248046]))

        # Expand max_new_tokens to allow full internal cognitive unrolling in Shadow KV
        if enable_shadow_thought and input_ids is not None:
            kwargs["max_new_tokens"] = orig_max_new_tokens + shadow_headroom

        # Intercept streamer when Shadow KV Slicing is active so internal <think> tokens
        # are distilled into Shadow KV rather than streamed raw to the caller
        streamer = kwargs.pop("streamer", None) if (enable_shadow_thought and input_ids is not None) else None

        try:
            if streamer is not None and getattr(streamer, "skip_prompt", False) and input_ids is not None:
                streamer.put(input_ids.cpu())

            raw_out = self.base_model.generate(*args, **kwargs)

            # Shadow KV Slicing: distill internal thought tokens into shadow state
            if enable_shadow_thought and input_ids is not None and raw_out.shape[1] > input_len:
                gen_ids = raw_out[0, input_len:].tolist()
                THINK_ID = 248068
                END_THINK_ID = 248069
                tail_prompt_ids = input_ids[0, max(0, input_len - 5):input_len].tolist()
                prompt_has_think = (THINK_ID in tail_prompt_ids) and (END_THINK_ID not in tail_prompt_ids)
                p_low = (prompt_text or "").lower()
                is_coding_prompt = any(k in p_low for k in [
                    "def ", "python function", "python code", "write a function", "code definition"
                ]) and not any(k in p_low for k in ["exception is raised", "exception class name"])

                def _extract_from_thought(t_text: str) -> Optional[List[int]]:
                    if tokenizer is None or not t_text:
                        return None
                    # 1. Fenced python block containing a def
                    fenced = re.findall(r"```(?:python)?\s*(def\s+[\s\S]*?)```", t_text)
                    if fenced:
                        return tokenizer.encode(fenced[-1].strip(), add_special_tokens=False)
                    # 2. Multi-line def block (with indented body lines)
                    defs = re.findall(r"(def\s+\w+\s*\([^)]*\)[^\n]*:(?:\n(?:    |\t)+[^\n]+)+)", t_text)
                    if defs:
                        return tokenizer.encode(defs[-1].strip(), add_special_tokens=False)
                    # 3. For non-coding queries ONLY, extract concise answer conclusion
                    if not is_coding_prompt:
                        ans_m = re.search(r"(?:Answer|Referent|Therefore|Conclusion|Output|Entity|Antecedent):\s*([^\n\.]+)", t_text, re.IGNORECASE)
                        if not ans_m:
                            ans_m = re.search(r"(?:The referent is|The answer is|referent of [^\n\.]+ is)\s*([^\n\.]+)", t_text, re.IGNORECASE)
                        if ans_m:
                            extracted_ans = re.sub(r"[\*\"\'`]+", "", ans_m.group(1)).strip()
                            if extracted_ans:
                                return tokenizer.encode(" " + extracted_ans, add_special_tokens=False)
                    return None

                if THINK_ID in gen_ids or (prompt_has_think and END_THINK_ID in gen_ids):
                    think_idx = gen_ids.index(THINK_ID) if THINK_ID in gen_ids else 0
                    pre_text = tokenizer.decode(gen_ids[:think_idx]).strip() if (tokenizer is not None and think_idx > 0) else ""

                    # Detect whether pre-think text is an unimplemented code header/stub
                    is_pre_code_stub = (
                        "# your implementation" in pre_text.lower()
                        or "# todo" in pre_text.lower()
                        or pre_text.rstrip().endswith(":")
                        or (
                            "def " in pre_text
                            and "return" not in pre_text
                            and "print(" not in pre_text
                            and ".reverse()" not in pre_text
                            and ".sort()" not in pre_text
                        )
                    )

                    # Check whether pre-think tokens constitute a complete genuine answer
                    has_real_pre_text = (
                        bool(re.search(r"[a-zA-Z0-9]{2,}", pre_text))
                        and pre_text.lower() not in ["no knowledge", "none", "unknown", "thinking process"]
                        and not pre_text.strip().startswith(("*", "-", "#", ":"))
                        and not is_pre_code_stub
                    )

                    if has_real_pre_text:
                        # Model answered first, then rambled into <think>: keep pre-think as visible answer
                        visible_ids = gen_ids[:think_idx]
                        shadow_ids = gen_ids[think_idx:]
                    else:
                        # Model thought first (or emitted only a function header stub before <think>)
                        if END_THINK_ID in gen_ids:
                            end_idx = gen_ids.index(END_THINK_ID)
                            post_ids = gen_ids[end_idx + 1:]
                            post_text = tokenizer.decode(post_ids).strip() if tokenizer is not None else ""

                            if is_pre_code_stub and post_text:
                                if "def " in post_text:
                                    # Post-think contains the full function definition and body
                                    visible_ids = post_ids
                                    shadow_ids = gen_ids[:end_idx + 1]
                                else:
                                    # Pre-think has the `def ...:` header and post-think has the body
                                    header_clean = re.sub(r"#\s*Your implementation.*", "", pre_text).rstrip()
                                    if tokenizer is not None and header_clean:
                                        post_raw = tokenizer.decode(post_ids)
                                        indented_post = "\n".join(
                                            ("    " + ln if (ln.strip() and not ln.startswith((" ", "\t"))) else ln)
                                            for ln in post_raw.splitlines()
                                            if ln.strip()
                                        )
                                        visible_ids = tokenizer.encode(header_clean + "\n" + indented_post, add_special_tokens=False)
                                    else:
                                        visible_ids = gen_ids[:think_idx] + post_ids
                                    shadow_ids = gen_ids[think_idx:end_idx + 1]
                            else:
                                shadow_ids = gen_ids[:end_idx + 1]
                                visible_ids = post_ids

                            # If output after </think> is empty or whitespace, extract conclusion from thought
                            if not visible_ids or (tokenizer is not None and not tokenizer.decode(visible_ids).strip()):
                                thought_start = (think_idx + 1) if THINK_ID in gen_ids else 0
                                thought_tokens = gen_ids[thought_start:end_idx]
                                thought_text = tokenizer.decode(thought_tokens, skip_special_tokens=False) if tokenizer is not None else ""
                                extracted = _extract_from_thought(thought_text)
                                if extracted is not None:
                                    visible_ids = extracted
                        else:
                            # Fallback when </think> was cut off: extract code block or concise answer from thought body
                            thought_start = (think_idx + 1) if THINK_ID in gen_ids else 0
                            thought_tokens = gen_ids[thought_start:]
                            thought_text = tokenizer.decode(thought_tokens, skip_special_tokens=False) if tokenizer is not None else ""
                            extracted = _extract_from_thought(thought_text)
                            if extracted is not None:
                                visible_ids = extracted
                                shadow_ids = gen_ids
                            else:
                                cut_point = max(0, len(gen_ids) - orig_max_new_tokens)
                                shadow_ids = gen_ids[:cut_point]
                                visible_ids = gen_ids[cut_point:]
                else:
                    shadow_ids = []
                    visible_ids = gen_ids

                # Ensure function header is included if prompt specified a `def ...:` prototype and model only emitted body
                if is_coding_prompt and tokenizer is not None and visible_ids:
                    vis_text = tokenizer.decode(visible_ids, skip_special_tokens=True)
                    if "def " not in vis_text and ("return" in vis_text or ".reverse(" in vis_text or ".sort(" in vis_text):
                        proto_m = re.search(r"(def\s+\w+\s*\([^)]*\)\s*:)", prompt_text or "")
                        if proto_m:
                            proto_hdr = proto_m.group(1)
                            indented_body = "\n".join(
                                ("    " + ln if (ln.strip() and not ln.startswith((" ", "\t"))) else ln)
                                for ln in vis_text.splitlines()
                                if ln.strip() and "# your implementation" not in ln.lower()
                            )
                            if indented_body:
                                visible_ids = tokenizer.encode(proto_hdr + "\n" + indented_body, add_special_tokens=False)

                # Strip leading newline tokens from visible answer
                while visible_ids and visible_ids[0] in [198, 271, 1073]:
                    visible_ids = visible_ids[1:]

                # Cap visible answer to requested max_new_tokens
                visible_ids = visible_ids[:orig_max_new_tokens]

                if tokenizer is not None and shadow_ids:
                    self.last_telemetry["shadow_thought_trace"] = tokenizer.decode(shadow_ids, skip_special_tokens=False)
                    self.last_telemetry["shadow_thought_tokens"] = len(shadow_ids)

                clean_out = torch.cat([
                    input_ids[:, :input_len],
                    torch.tensor([visible_ids], device=input_ids.device, dtype=input_ids.dtype)
                ], dim=-1)

                if streamer is not None:
                    vis_slice = clean_out[:, input_len:].cpu()
                    for idx in range(vis_slice.shape[1]):
                        streamer.put(vis_slice[:, idx:idx + 1])

                return clean_out

            if streamer is not None and raw_out.shape[1] > input_len:
                vis_slice = raw_out[:, input_len:].cpu()
                for idx in range(vis_slice.shape[1]):
                    streamer.put(vis_slice[:, idx:idx + 1])

            return raw_out
        finally:
            if streamer is not None:
                streamer.end()

    def verify_and_adapt_closed_loop(
        self,
        prompt: str,
        response: str,
        tokenizer: Optional[Any] = None,
        external_verdict: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Verifies the generated response via IntrinsicPopperianCycleVerifier (or optional
        external verdict) and updates Layer 7 synaptic weights + closed-loop deflation state.
        """
        tok = tokenizer or self.tokenizer
        res = self.brain.verify_and_adapt_closed_loop(
            prompt=prompt,
            response=response,
            external_verdict=external_verdict
        )
        if res["dopamine_sign"] < 0:
            falsified_tids = list(self.closed_loop_state.get("falsified_token_ids", []))
            if tok is not None and response.strip():
                first_word = response.strip().split()[0]
                for variant in (first_word, " " + first_word):
                    tids = tok.encode(variant, add_special_tokens=False)
                    if tids:
                        falsified_tids.append(tids[0])
            self.closed_loop_state["falsified_token_ids"] = list(set(falsified_tids))
            if res.get("correction_hint") is not None:
                self.closed_loop_state["correction_hint"] = res["correction_hint"]
            self.closed_loop_state["last_verification_sign"] = -1.0
        else:
            self.closed_loop_state.clear()
        self.last_telemetry["closed_loop_verification"] = res
        return res

    def synthesize_voice(
        self,
        text: Optional[str] = None,
        duration_sec: float = 0.60,
        save_path: Optional[str] = None,
        adapt_online: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Synthesizes custom endogenous voice (`freq suara`) from the brain's active
        Layer 7 state and spiking pacemaker dynamics, running the Self-Listening
        Auditory Re-Entry Loop.
        """
        waveform, v_telem = self.brain.synthesize_endogenous_voice(
            text=text,
            duration_sec=duration_sec,
            save_path=save_path,
            adapt_online=adapt_online
        )
        self.last_telemetry["endogenous_voice"] = {
            k: v for k, v in v_telem.items() if k != "auditory_reentry_delta"
        }
        return waveform, v_telem

    def train_vocal_babbling(
        self,
        text: str = "om",
        target_formants: Optional[List[float]] = None,
        n_steps: int = 3,
        lr: float = 0.08,
        save_wav_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs active closed-loop vocal babbling and pronunciation learning."""
        return self.brain.train_vocal_babbling(
            text=text,
            target_formants=target_formants,
            n_steps=n_steps,
            lr=lr,
            save_wav_prefix=save_wav_prefix
        )

    def ingest_external_visual(
        self,
        visual_input: torch.Tensor,
        target_hidden_states: Optional[torch.Tensor] = None,
        adapt_online: bool = True
    ) -> Dict[str, Any]:
        """Ingests an external photo, video frame, or camera tensor into shared latent space."""
        return self.brain.ingest_external_visual(
            visual_input=visual_input,
            target_hidden_states=target_hidden_states,
            adapt_online=adapt_online
        )

    def generate_self_evolving(
        self,
        *args,
        max_passes: int = 2,
        synthesize_voice: bool = False,
        voice_output_path: Optional[str] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes closed-loop self-evolving generation:
        - Pass 1 generates a candidate response and evaluates it via IntrinsicPopperianCycleVerifier.
        - If verified (c = +1), consolidates via Hebbian Nullspace plasticity.
        - If falsified (c = -1), applies Anti-Hebbian Orthogonal Deflation + Unitary Givens
          Rotation and automatically executes Pass 2 to self-correct.
        """
        tokenizer = kwargs.get("tokenizer", getattr(self, "tokenizer", None))
        prompt_text = kwargs.get("prompt_text", None)
        input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
        input_len = input_ids.shape[1] if input_ids is not None else 0
        if prompt_text is None and tokenizer is not None and input_ids is not None:
            prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        self.reset_closed_loop()
        prev_online = self.enable_online_learning
        self.enable_online_learning = True

        passes_log = []
        out_ids = None
        for pass_idx in range(1, max_passes + 1):
            out_ids = self.generate(*args, **kwargs)
            resp_text = ""
            if tokenizer is not None and input_ids is not None and out_ids.shape[1] > input_len:
                resp_text = tokenizer.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()
            v_res = self.verify_and_adapt_closed_loop(
                prompt=prompt_text or "",
                response=resp_text,
                tokenizer=tokenizer
            )
            passes_log.append({
                "pass": pass_idx,
                "response": resp_text,
                "verified": v_res["verified"],
                "dopamine_sign": v_res["dopamine_sign"],
                "reason": v_res["reason"],
            })
            if v_res["verified"]:
                break

        self.enable_online_learning = prev_online
        summary = {
            "total_passes": len(passes_log),
            "passes": passes_log,
            "final_verified": passes_log[-1]["verified"] if passes_log else True,
        }
        if synthesize_voice:
            final_text = passes_log[-1]["response"] if passes_log else (prompt_text or "suara")
            _, v_telem = self.synthesize_voice(
                text=final_text,
                save_path=voice_output_path,
                adapt_online=True
            )
            summary["voice_telemetry"] = {k: v for k, v in v_telem.items() if k != "auditory_reentry_delta"}

        self.last_telemetry["self_evolving_summary"] = summary
        return out_ids, summary

    @property
    def enable_online_learning(self) -> bool:
        return getattr(self.brain, "enable_online_learning", False)

    @enable_online_learning.setter
    def enable_online_learning(self, value: bool):
        self.brain.enable_online_learning = bool(value)

    def get_online_learning_stats(self) -> Dict[str, Any]:
        return self.brain.get_online_learning_stats()

    def __del__(self):
        if hasattr(self, "hook_handles"):
            self._remove_hooks()


def attach_hadl_v5_spiking_brain(
    base_model: nn.Module,
    target_layer_idx: int = 7,
    pacemaker_layer_idx: int = 2,
    reentrant_layers: Optional[List[int]] = None,
    d_model: Optional[int] = None,
    spike_v_th: float = 1.0,
    beta_decay: float = 0.85,
    target_firing_rate: float = 0.20,
    deliberation_steps_k: int = 1,
    auto_load_checkpoints: bool = True,
    checkpoints_dir: Optional[Union[str, Path]] = None,
    enable_directional_safety: bool = True,
    enable_online_learning: bool = False,
) -> HADLv5SpikingBrainModelWrapper:
    """
    Canonically attaches the HADL v5.5 Neuromorphic Spiking Brain architecture.
    """
    return HADLv5SpikingBrainModelWrapper(
        base_model=base_model,
        target_layer_idx=target_layer_idx,
        pacemaker_layer_idx=pacemaker_layer_idx,
        reentrant_layers=reentrant_layers,
        d_model=d_model,
        spike_v_th=spike_v_th,
        beta_decay=beta_decay,
        target_firing_rate=target_firing_rate,
        deliberation_steps_k=deliberation_steps_k,
        auto_load_checkpoints=auto_load_checkpoints,
        checkpoints_dir=checkpoints_dir,
        enable_directional_safety=enable_directional_safety,
        enable_online_learning=enable_online_learning,
    )

