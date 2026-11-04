"""
Pre-Router Spiking Pacemaker, Concurrent Dual-Stream Synthesizer & Tour Leader Attractor
========================================================================================
HADL v5.5 Neuromorphic Cognitive Architecture

Components:
1. ModelPersonalityAdapter:
   - Dimension-aware physical scaling laws for membrane time constant tau_m(D),
     threshold V_th(D), leak factor beta(D), and bottleneck dimension D_c(D)
     across model architectures (D=2048, 2304, 4096, etc.).

2. PreRouterSpikingPacemaker:
   - Hooked at early contextual layers (e.g. Layer 2).
   - Fractional leaky integrate-and-fire membrane dynamics emitting action
     potential s(t) in {0, 1} and continuous postsynaptic driving vector u_drive(t).

3. TourLeaderAttractor:
   - Continuous vector field attractor: v_tour = -mu_tour * nabla Phi_logic(h).
   - Guides the model toward the Logic Base Unit (LBU) manifold and eliminates
     cohomological defect without rigid prompt templates.

4. ConcurrentDualStreamSynthesizer:
   - System 1 (fast reflexive stream, <5 ms feedforward base pass).
   - System 2 (latent contractive Banach step in compressed space D_c, K <= 3).
   - Orthogonal symplectic projection fusion:
     h_fused = h_S1 + tanh(u_drive) * P_perp(h_S1) Delta h_S2 + v_tour
     guaranteeing <P_perp(h_S1) Delta h_S2, h_S1> = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..vgni_lbu_cloud_engine import VirtualWorldGeneralIntelligence, LogicBaseUnit
from ..verification import DirectionalSafetyProjection


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


@dataclass
class PersonalityProfile:
    """Dimension-aware physical and mathematical parameters for model personality."""
    d_model: int
    tau_m: float
    beta: float
    v_th: float
    d_bottleneck: int
    eta_plastic: float
    gamma_contraction: float


class ModelPersonalityAdapter:
    """
    Model Personality Adapter.
    
    Calibrates neuromorphic parameters according to physical and mathematical
    scaling laws across model dimensions (D=2048 for Qwen, D=2304 for Gemma,
    D=4096 for LLaMA/Mistral).
    
    Scaling Laws:
    - Normalization scale: Representation norm ||h|| ~ sqrt(D) by Central Limit Theorem.
    - Membrane time constant: tau_m(D) = tau_0 * sqrt(D / D_0)
    - Membrane leak factor: beta(D) = exp(-Delta t / tau_m(D)) = beta_0^(sqrt(D_0 / D))
    - Action potential threshold: V_th(D) = V_0 * sqrt(D / D_0)
    - Deliberation bottleneck dimension: D_c(D) = min(1024, max(256, D // 2))
    - Plastic learning rate: eta_plastic(D) = eta_0 * (D_0 / D)
    """
    D_REF: int = 2048
    TAU_REF: float = 20.0     # ms
    BETA_REF: float = 0.85
    V_TH_REF: float = 1.0
    ETA_REF: float = 0.05
    GAMMA_REF: float = 0.85

    @classmethod
    def calibrate(
        cls,
        d_model: int,
        base_v_th: Optional[float] = None,
        base_beta: Optional[float] = None
    ) -> PersonalityProfile:
        ratio = d_model / cls.D_REF
        sqrt_ratio = math.sqrt(ratio)

        tau_m = cls.TAU_REF * sqrt_ratio
        
        # Leak factor beta: beta = beta_ref^(1 / sqrt_ratio)
        beta_calib = (cls.BETA_REF if base_beta is None else base_beta) ** (1.0 / max(0.1, sqrt_ratio))
        beta_calib = max(0.50, min(0.98, beta_calib))

        # Threshold scales with sqrt(D / D_0)
        v_th_calib = (cls.V_TH_REF if base_v_th is None else base_v_th) * sqrt_ratio

        # Bottleneck dimension for Banach contraction
        if d_model >= 512:
            d_bottleneck = min(1024, max(256, (d_model // 2 // 64) * 64))
        else:
            d_bottleneck = min(d_model, max(64, (d_model // 2 // 64) * 64))

        # Plastic learning rate scales inversely with dimension
        eta_plastic = cls.ETA_REF / max(0.5, ratio)

        return PersonalityProfile(
            d_model=d_model,
            tau_m=round(tau_m, 2),
            beta=round(beta_calib, 4),
            v_th=round(v_th_calib, 4),
            d_bottleneck=d_bottleneck,
            eta_plastic=round(eta_plastic, 6),
            gamma_contraction=cls.GAMMA_REF,
        )


class PreRouterSpikingPacemaker(nn.Module):
    """
    Pre-Router Spiking Pacemaker (Layer 2 Early Hook).
    
    Evaluates early syntactic-semantic saliency BEFORE deep contextual routing.
    Emits:
    - s(t): Binary action potential (via FastSigmoidSurrogate)
    - u_drive(t): Continuous postsynaptic driving signal for routing and fusion
    
    Dynamics:
        I_early(t) = W_sensor h_early(t)
        V_m(t) = beta * V_m(t-1) + (1 - beta) * I_early(t)
        s(t) = Theta(V_m(t) - V_th(t))
        u_drive(t) = tanh(W_drive h_early(t) + s(t) * w_s)
        V_th(t) = V_th_0 + alpha_homeo * (bar_s - r_target)
    """
    def __init__(
        self,
        d_model: int = 2048,
        personality: Optional[PersonalityProfile] = None,
        target_firing_rate: float = 0.20,
        homeostatic_alpha: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.profile = personality or ModelPersonalityAdapter.calibrate(d_model)

        self.beta = self.profile.beta
        self.base_v_th = self.profile.v_th
        self.target_rate = target_firing_rate
        self.homeostatic_alpha = homeostatic_alpha

        # Sensor projection: compresses hidden state into scalar current
        self.norm = nn.LayerNorm(d_model)
        self.w_sensor = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.w_sensor.weight, std=1.0 / math.sqrt(d_model))

        # Continuous driving projection: maps early representation to driving field
        self.w_drive = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.w_drive.weight, std=0.02)
        nn.init.zeros_(self.w_drive.bias)
        self.w_spike_coupling = nn.Parameter(torch.tensor(0.50))

        # Internal state buffers
        self.register_buffer("v_membrane", torch.zeros(1, 1, 1))
        self.register_buffer("v_th", torch.tensor(self.base_v_th))
        self.register_buffer("ema_rate", torch.tensor(target_firing_rate))
        self.register_buffer("total_spikes", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_steps", torch.tensor(0, dtype=torch.long))

    def reset_state(self):
        """Resets membrane potential and homeostatic buffers."""
        with torch.no_grad():
            self.v_membrane = torch.zeros(1, 1, 1, device=self.v_th.device, dtype=self.v_th.dtype)
            self.v_th = torch.tensor(self.base_v_th, device=self.v_th.device, dtype=self.v_th.dtype)

    def forward(
        self,
        h_early: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Processes early layer activations (Layer 2).
        
        Args:
            h_early: [B, S, D] hidden states from early layer
            
        Returns:
            spikes: [B, S, 1] action potentials in {0, 1}
            u_drive: [B, S, 1] continuous postsynaptic driving signal in [-1, 1]
            telemetry: diagnostic metadata
        """
        B, S, D = h_early.shape
        target_dtype = self.w_sensor.weight.dtype
        target_device = h_early.device
        h_norm = self.norm(h_early.to(dtype=target_dtype))

        # 1. Early sensory input current with biological amplitude scaling
        h_mag = torch.norm(h_early.to(dtype=target_dtype), dim=-1, keepdim=True) / math.sqrt(D)
        i_early = torch.tanh(h_mag) * self.w_sensor(h_norm)  # [B, S, 1]

        # 2. Integrate leaky membrane dynamics
        if self.v_membrane.device != target_device or self.v_membrane.shape != (B, 1, 1):
            self.v_membrane = torch.zeros(B, 1, 1, device=target_device, dtype=target_dtype)

        v_m_seq = []
        spikes_list = []
        v_cur = self.v_membrane[:, :, 0]  # [B, 1]

        v_th_val = self.v_th.to(device=target_device, dtype=target_dtype)

        for t in range(S):
            i_t = i_early[:, t, :]  # [B, 1]
            # Leaky integration with biological non-negative resting boundary
            v_next = torch.clamp(self.beta * v_cur + (1.0 - self.beta) * i_t, min=0.0)
            # Spike emission via surrogate
            s_t = spike_fn(v_next, v_th_val, gamma=4.0)
            # Refractory reset
            v_cur = torch.clamp(v_next * (1.0 - s_t), min=0.0)
            v_m_seq.append(v_next.unsqueeze(1))
            spikes_list.append(s_t.unsqueeze(1))

        self.v_membrane = v_cur.unsqueeze(1).detach()
        v_membrane_all = torch.cat(v_m_seq, dim=1)  # [B, S, 1]
        spikes = torch.cat(spikes_list, dim=1)      # [B, S, 1]

        # 3. Continuous postsynaptic driving signal u_drive
        raw_drive = self.w_drive(h_norm) + self.w_spike_coupling * spikes
        u_drive = torch.tanh(raw_drive).to(dtype=h_early.dtype)

        # 4. Homeostatic threshold adaptation
        firing_rate = float(spikes.mean().item())
        if self.training:
            with torch.no_grad():
                new_ema = 0.95 * float(self.ema_rate.item()) + 0.05 * firing_rate
                self.ema_rate = torch.tensor(new_ema, device=self.ema_rate.device, dtype=self.ema_rate.dtype)
                rate_error = new_ema - self.target_rate
                new_v_th = max(0.20, min(3.0, float(self.v_th.item()) + self.homeostatic_alpha * rate_error))
                self.v_th = torch.tensor(new_v_th, device=self.v_th.device, dtype=self.v_th.dtype)

        with torch.no_grad():
            self.total_steps = self.total_steps + S
            self.total_spikes = self.total_spikes + int(spikes.sum().item())

        v_mean = float(v_membrane_all.mean().item())
        v_max = float(v_membrane_all.max().item())

        telemetry = {
            "pacemaker_firing_rate": round(firing_rate, 4),
            "pacemaker_v_m_mean": round(v_mean, 4),
            "pacemaker_v_m_max": round(v_max, 4),
            "pacemaker_v_th": round(float(self.v_th.item()), 4),
            "pacemaker_spikes_fired": int(spikes.sum().item()),
            "pacemaker_u_drive_mean": round(float(u_drive.mean().item()), 4),
        }
        return spikes, u_drive, telemetry


class TourLeaderAttractor(nn.Module):
    """
    Tour Leader Attractor (Continuous Vector Field Guidance).
    
    Replaces rigid prompt templates (<think>, 'Target Deliverable:', etc.) with a
    continuous potential field attractor:
        v_tour = -mu_tour * nabla Phi_logic(h)
    
    Extrinsic Logic Potential:
        Phi_logic(h) = 1/2 * dist^2(h, M_logic) + alpha * ||delta^0(h)||^2
    
    Properties:
    - If h is logically consistent and on the manifold: Phi_logic ~ 0 => v_tour = 0.
    - If a contradiction or defect emerges: -nabla Phi_logic restores h non-invasively
      back toward the closest logical manifold subspace.
    - Zero template overhead: Model reasons in its own natural voice and personality!
    """
    def __init__(
        self,
        d_model: int = 2048,
        mu_tour: float = 0.15,
        alpha_defect: float = 0.25
    ):
        super().__init__()
        self.d_model = d_model
        self.mu_tour = mu_tour
        self.alpha_defect = alpha_defect

        self.norm = nn.LayerNorm(d_model)

        # Logic manifold subspace projection (Householder basis or low-rank manifold)
        self.w_manifold = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.w_manifold.weight)

    def forward(
        self,
        h: torch.Tensor,
        lbu: Optional[LogicBaseUnit] = None,
        concept_anchors: Optional[torch.Tensor] = None,
        viscosity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes non-invasive restoring drift v_tour with continuous task-elastic coupling.
        
        Args:
            h: [B, S, D] hidden representations
            lbu: optional LogicBaseUnit instance with verified Householder reflectors
            concept_anchors: optional [N, D] concept anchors from VGNI
            viscosity: optional [B, 4] or [B, S, 4] continuous hydraulic viscosity tensor
                       for (code, math, logic, discourse) from MISF
            
        Returns:
            v_tour: [B, S, D] restorative drift vector
            telemetry: potential energy and restoring force norm
        """
        B, S, D = h.shape
        target_dtype = self.w_manifold.weight.dtype
        h_cast = h.to(dtype=target_dtype)
        h_norm = self.norm(h_cast)

        # 1. Continuous Task-Elastic Coupling (Fermi-Dirac Statistical Physics Modulation)
        if viscosity is not None:
            if viscosity.dim() == 2:
                visc_exp = viscosity.unsqueeze(1)  # [B, 1, 4]
            else:
                visc_exp = viscosity
            v_code = visc_exp[..., 0:1]
            v_formal = visc_exp[..., 1:2] + visc_exp[..., 2:3]
            # Fermi-Dirac transition factor: activates strongly for formal STEM/logic (v_formal > 0.55)
            fermi_factor = torch.sigmoid(12.0 * (v_formal - 0.55))
            code_damping = torch.clamp(1.0 - 2.0 * v_code, min=0.0)
            effective_mu = self.mu_tour * fermi_factor * code_damping
        else:
            effective_mu = self.mu_tour

        # 2. Project onto logic manifold M_logic
        if concept_anchors is not None and concept_anchors.numel() > 0:
            anchors = concept_anchors.to(device=h.device, dtype=target_dtype)
            if anchors.dim() == 2 and anchors.shape[0] <= anchors.shape[1]:
                try:
                    Q, _ = torch.linalg.qr(anchors.t(), mode="reduced")
                    h_proj = torch.matmul(torch.matmul(h_cast, Q), Q.t())
                except Exception:
                    anchors_u = F.normalize(anchors, dim=-1)
                    proj_coeffs = torch.matmul(h_cast, anchors_u.t())
                    h_proj = torch.matmul(proj_coeffs, anchors_u)
            else:
                anchors_u = F.normalize(anchors, dim=-1)
                proj_coeffs = torch.matmul(h_cast, anchors_u.t())
                h_proj = torch.matmul(proj_coeffs, anchors_u)
        elif lbu is not None and hasattr(lbu, "u_logic"):
            try:
                Q, _ = torch.linalg.qr(lbu.u_logic.to(device=h.device, dtype=target_dtype).t(), mode="reduced")
                h_proj = torch.matmul(torch.matmul(h_cast, Q), Q.t())
            except Exception:
                u_norm = F.normalize(lbu.u_logic.to(device=h.device, dtype=target_dtype), dim=-1)
                proj_coeffs = torch.matmul(h_cast, u_norm.t())
                h_proj = torch.matmul(proj_coeffs, u_norm)
        else:
            h_proj = self.w_manifold(h_cast)

        # Manifold distance gradient: nabla (1/2 ||h - h_proj||^2) = h - h_proj
        # Restoring force: -nabla = h_proj - h
        drift_manifold = h_proj - h_cast

        # 3. Sheaf 0-cocycle defect ||delta^0(h)||^2: Obstruction to global consistency
        defect_norm = torch.norm(drift_manifold, dim=-1, keepdim=True)

        # Soft deadband: if potential energy is below threshold, tour leader stays completely passive
        deadband = 0.10
        active_mask = F.relu(defect_norm - deadband) / (defect_norm + 1e-8)

        v_tour = effective_mu * active_mask * torch.tanh(drift_manifold)
        v_tour = v_tour.to(dtype=h.dtype)

        mean_force = float(torch.norm(v_tour, dim=-1).mean().item())
        mean_defect = float(defect_norm.mean().item())

        telemetry = {
            "tour_restoring_force": round(mean_force, 4),
            "tour_sheaf_defect": round(mean_defect, 4),
            "tour_active": mean_force > 0.01,
            "effective_mu_tour": round(float(effective_mu.mean().item()) if isinstance(effective_mu, torch.Tensor) else effective_mu, 4)
        }
        return v_tour, telemetry


class ConcurrentDualStreamSynthesizer(nn.Module):
    """
    Concurrent Dual-Stream Synthesizer (Parallel System 1 & System 2).
    
    Architecture:
    - System 1 (Reflexive Stream): Fast base feedforward pass (<5 ms).
    - System 2 (Deliberative Stream): Inline contractive Banach step in compressed space
      (D_c = 1024 or dimension-aware bottleneck, K <= 3 contraction steps).
    - Symplectic Orthogonal Fusion:
        h_fused = h_S1 + tanh(u_drive) * P_perp(h_S1) Delta h_S2 + v_tour
      where P_perp(h_S1) is the orthogonal symplectic projector:
        P_perp(h_S1) Delta = Delta - (<Delta, h_S1> / (||h_S1||^2 + eps)) * h_S1
      guaranteeing: <P_perp(h_S1) Delta, h_S1> = 0.
    """
    def __init__(
        self,
        d_model: int = 2048,
        personality: Optional[PersonalityProfile] = None,
        max_deliberation_steps: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        self.profile = personality or ModelPersonalityAdapter.calibrate(d_model)
        self.d_bottleneck = self.profile.d_bottleneck
        self.max_steps = max_deliberation_steps

        # System 2 Deliberative Contraction Kernels
        self.s2_down = nn.Linear(d_model, self.d_bottleneck, bias=False)
        self.s2_contract = nn.Linear(self.d_bottleneck, self.d_bottleneck, bias=False)
        self.s2_up = nn.Linear(self.d_bottleneck, d_model, bias=False)
        self.s2_norm = nn.LayerNorm(d_model)

        # Strict contractive initialization: Lipschitz constant <= gamma < 1.0
        nn.init.orthogonal_(self.s2_down.weight)
        nn.init.orthogonal_(self.s2_contract.weight)
        nn.init.orthogonal_(self.s2_up.weight)

        self.gamma_contraction = nn.Parameter(torch.tensor(self.profile.gamma_contraction))
        self.s2_scale = nn.Parameter(torch.tensor(0.20))

        # Tour Leader Attractor
        self.tour_leader = TourLeaderAttractor(d_model=d_model)

    @staticmethod
    def orthogonal_symplectic_project(delta: torch.Tensor, h_base: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Projects delta onto the orthogonal complement of h_base:
            P_perp(h_base) delta = delta - (<delta, h_base> / (||h_base||^2 + eps)) * h_base
        
        Guarantees:
            <P_perp(h_base) delta, h_base> = 0
            <h_base + P_perp(h_base) delta, h_base> = ||h_base||^2
        """
        # delta: [B, S, D], h_base: [B, S, D]
        inner = torch.sum(delta * h_base, dim=-1, keepdim=True)            # [B, S, 1]
        norm_sq = torch.sum(h_base * h_base, dim=-1, keepdim=True) + eps  # [B, S, 1]
        projection = (inner / norm_sq) * h_base                            # [B, S, D]
        delta_perp = delta - projection                                    # [B, S, D]
        return delta_perp

    def forward_system2_contraction(
        self,
        h_s1: torch.Tensor,
        num_steps: int = 1
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Executes inline System 2 contractive Banach fixed-point steps in compressed space.
        
        Equation per iteration:
            z_{k+1} = gamma * tanh(W_contract z_k + z_0)
        conditioning the Banach contraction on input anchor z_0.
        """
        target_dtype = self.s2_down.weight.dtype
        h_norm = self.s2_norm(h_s1.to(dtype=target_dtype))

        # Down-project to compressed bottleneck D_c
        z = self.s2_down(h_norm)  # [B, S, D_c]
        z_0 = z
        gamma = self.gamma_contraction.clamp(min=0.50, max=0.95)

        # Contractive Banach iterations conditioned on input anchor z_0
        steps = max(1, min(self.max_steps, num_steps))
        for _ in range(steps):
            z = gamma * torch.tanh(self.s2_contract(z) + z_0)

        # Up-project back to model dimension D
        delta_s2 = self.s2_scale * self.s2_up(z)
        delta_norm = float(torch.norm(delta_s2, dim=-1).mean().item())
        return delta_s2.to(dtype=h_s1.dtype), delta_norm, steps

    def forward(
        self,
        h_s1: torch.Tensor,
        u_drive: torch.Tensor,
        num_deliberation_steps: int = 1,
        lbu: Optional[LogicBaseUnit] = None,
        concept_anchors: Optional[torch.Tensor] = None,
        viscosity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Concurrently synthesizes System 1 and System 2 with symplectic fusion.
        
        Args:
            h_s1: [B, S, D] base feedforward representations (System 1 fast reflex)
            u_drive: [B, S, 1] continuous driving signal from pre-router pacemaker
            num_deliberation_steps: K iterations for System 2 contraction (1 to 3)
            lbu: optional LogicBaseUnit instance
            concept_anchors: optional concept anchor bank
            viscosity: optional continuous hydraulic viscosity tensor from MISF
            
        Returns:
            h_fused: [B, S, D] fused representations preserving base fluency
            telemetry: dual-stream telemetry
        """
        # Step 1: System 2 inline Banach contraction
        delta_s2, s2_delta_norm, actual_steps = self.forward_system2_contraction(h_s1, num_steps=num_deliberation_steps)

        # Step 2: Orthogonal symplectic projection
        # Ensures Delta h_S2 is perpendicular to h_S1, preventing degradation of base accuracy
        delta_perp = self.orthogonal_symplectic_project(delta_s2, h_s1)

        # Step 3: Modulate by pacemaker driving signal with biological non-negative rectification
        drive_weight = torch.clamp(torch.tanh(u_drive.to(dtype=h_s1.dtype)), min=0.0)  # [B, S, 1]
        modulated_s2 = drive_weight * delta_perp

        # Step 4: Tour Leader Attractor guidance with task-elastic fluid coupling
        v_tour, tour_telem = self.tour_leader(h_s1, lbu=lbu, concept_anchors=concept_anchors, viscosity=viscosity)

        # Step 5: Master Symplectic Fusion
        h_fused = h_s1 + modulated_s2 + v_tour

        # Verify inner product preservation: <h_fused - v_tour, h_s1> == ||h_s1||^2
        inner_check = torch.sum((h_fused - v_tour) * h_s1, dim=-1).mean().item()
        base_norm_sq = torch.sum(h_s1 * h_s1, dim=-1).mean().item()

        telemetry = {
            "s2_delta_norm": round(s2_delta_norm, 4),
            "s2_drive_weight": round(float(drive_weight.mean().item()), 4),
            "tour_restoring_force": tour_telem.get("tour_restoring_force", 0.0),
            "tour_active": tour_telem.get("tour_active", False),
            "effective_mu_tour": tour_telem.get("effective_mu_tour", self.tour_leader.mu_tour),
            "inner_product_preservation_ratio": round(inner_check / max(1e-6, base_norm_sq), 6),
            "deliberation_steps_k": actual_steps,
        }
        return h_fused, telemetry


class LatentThoughtMatrixEngine(nn.Module):
    """
    Non-Token Multi-Part Latent Thought Matrix Engine (HADL v5.8).

    Lifts a 1D hidden representation h(t) in R^{B x S x D} into a structured
    2D Latent Thought Matrix M(t) in R^{B x S x R x D} (R = 4 cognitive sub-parts):
      - M_0: Invariant Factual & Syntactic Anchor (h_base)
      - M_1: Orthogonal Wild / Lateral Exploration Subspace (P_perp(M_0) Delta_wild)
      - M_2: Learned Epistemic Veracity & Counter-Myth Subspace (P_perp(M_0) Delta_veracity)
             with sequence-pooled proposition readout s_veracity in [-1, +1]
      - M_3: Banach Symbolic / Structural Deduction Subspace (P_perp(M_0) Delta_struct)
    """
    def __init__(
        self,
        d_model: int = 2048,
        num_subparts: int = 4,
        bottleneck_dim: int = 128,
        veracity_dim: int = 128
    ):
        super().__init__()
        self.d_model = d_model
        self.num_subparts = num_subparts
        self.bottleneck_dim = min(bottleneck_dim, max(32, d_model // 4))
        self.veracity_dim = min(veracity_dim, max(32, d_model // 4))

        self.norm = nn.LayerNorm(d_model)

        # Subpart 1: Wild / Lateral Exploration Subspace
        self.wild_down = nn.Linear(d_model, self.bottleneck_dim, bias=False)
        self.wild_up = nn.Linear(self.bottleneck_dim, d_model, bias=False)

        # Subpart 2: Learned Epistemic Veracity & Counter-Myth Subspace
        self.veracity_token_attn = nn.Linear(d_model, 1, bias=True)
        self.veracity_down = nn.Linear(d_model, self.veracity_dim, bias=True)
        self.veracity_mid = nn.Linear(self.veracity_dim, self.veracity_dim, bias=True)
        self.veracity_up = nn.Linear(self.veracity_dim, d_model, bias=False)
        self.veracity_head = nn.Linear(self.veracity_dim, 1, bias=True)

        # Subpart 3: Banach Symbolic / Structural Subspace
        self.struct_down = nn.Linear(d_model, self.bottleneck_dim, bias=False)
        self.struct_contract = nn.Linear(self.bottleneck_dim, self.bottleneck_dim, bias=False)
        self.struct_up = nn.Linear(self.bottleneck_dim, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for layer in [
                self.wild_down, self.wild_up,
                self.veracity_down, self.veracity_mid, self.veracity_up,
                self.struct_down, self.struct_contract, self.struct_up
            ]:
                w = torch.empty(layer.weight.shape, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(w)
                if layer in [self.wild_up, self.veracity_up, self.struct_up]:
                    w.mul_(0.05)
                elif layer is self.struct_contract:
                    w.mul_(0.80)
                layer.weight.data.copy_(w.to(device=layer.weight.device, dtype=layer.weight.dtype))
                if getattr(layer, "bias", None) is not None:
                    nn.init.zeros_(layer.bias)

            nn.init.normal_(self.veracity_token_attn.weight, std=0.01)
            nn.init.zeros_(self.veracity_token_attn.bias)
            nn.init.zeros_(self.veracity_head.weight)
            nn.init.zeros_(self.veracity_head.bias)

    def compute_pooled_proposition(self, h_norm: torch.Tensor) -> torch.Tensor:
        """
        Computes a sequence-level semantic proposition vector in R^{B x D}
        combining mean pooling, attention-weighted content pooling, and terminal token state.
        """
        B, S, D = h_norm.shape
        if S == 1:
            return h_norm[:, 0, :]
        h_mean = h_norm.mean(dim=1)
        h_last = h_norm[:, -1, :]
        attn_logits = self.veracity_token_attn(h_norm)  # [B, S, 1]
        attn_weights = F.softmax(attn_logits, dim=1)    # [B, S, 1]
        h_attn = torch.sum(attn_weights * h_norm, dim=1)
        return 0.45 * h_mean + 0.35 * h_attn + 0.20 * h_last

    def evaluate_veracity_subspace(self, h_base: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates the Epistemic Veracity & Counter-Myth Subspace on h_base.

        Returns:
            m2_ortho: [B, S, D] orthogonal veracity state
            veracity_score: [B, 1] continuous veracity conflict score in [-1, +1]
            z_prop: [B, D_v] latent veracity bottleneck representation
        """
        target_dtype = self.veracity_down.weight.dtype
        h_norm = self.norm(h_base.to(dtype=target_dtype))
        h_prop = self.compute_pooled_proposition(h_norm)  # [B, D]

        z_prop = torch.tanh(self.veracity_mid(F.silu(self.veracity_down(h_prop))))  # [B, D_v]
        raw_logit = self.veracity_head(z_prop)  # [B, 1]
        veracity_score = torch.tanh(raw_logit)  # [B, 1] in [-1, +1]

        z_seq = torch.tanh(self.veracity_mid(F.silu(self.veracity_down(h_norm))))   # [B, S, D_v]
        delta_v = self.veracity_up(z_seq).to(dtype=h_base.dtype)
        m2_ortho = ConcurrentDualStreamSynthesizer.orthogonal_symplectic_project(delta_v, h_base)
        return m2_ortho, veracity_score, z_prop

    def forward(
        self,
        h_base: torch.Tensor,
        delta_wild_ext: Optional[torch.Tensor] = None,
        delta_struct_ext: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Constructs the 4-subpart Non-Token Latent Thought Matrix M in R^{B x S x 4 x D}.
        """
        target_dtype = self.wild_down.weight.dtype
        h_norm = self.norm(h_base.to(dtype=target_dtype))

        # Subpart 0: Invariant Factual / Syntactic Anchor
        m0 = h_base

        # Subpart 1: Orthogonal Wild / Lateral Exploration
        raw_wild = self.wild_up(F.silu(self.wild_down(h_norm))).to(dtype=h_base.dtype)
        if delta_wild_ext is not None:
            raw_wild = raw_wild + 0.50 * delta_wild_ext.to(dtype=h_base.dtype)
        m1 = ConcurrentDualStreamSynthesizer.orthogonal_symplectic_project(raw_wild, m0)

        # Subpart 2: Learned Epistemic Veracity & Counter-Myth Subspace
        m2, veracity_score, _ = self.evaluate_veracity_subspace(h_base)

        # Subpart 3: Banach Symbolic / Structural Subspace
        z0 = self.struct_down(h_norm)
        z_s = torch.tanh(self.struct_contract(z0) + z0)
        raw_struct = self.struct_up(z_s).to(dtype=h_base.dtype)
        if delta_struct_ext is not None:
            raw_struct = raw_struct + 0.50 * delta_struct_ext.to(dtype=h_base.dtype)
        m3 = ConcurrentDualStreamSynthesizer.orthogonal_symplectic_project(raw_struct, m0)

        # Stack into 2D Latent Matrix per token: [B, S, 4, D]
        M = torch.stack([m0, m1, m2, m3], dim=2)

        v_score_val = float(veracity_score.mean().item())
        telem = {
            "latent_matrix_subparts": self.num_subparts,
            "veracity_conflict_score": round(v_score_val, 4),
            "epistemic_refutation_active": bool(v_score_val > 0.35),
            "subpart_norms": [
                round(float(torch.norm(m0, dim=-1).mean().item()), 4),
                round(float(torch.norm(m1, dim=-1).mean().item()), 4),
                round(float(torch.norm(m2, dim=-1).mean().item()), 4),
                round(float(torch.norm(m3, dim=-1).mean().item()), 4),
            ],
        }
        self.last_veracity_scores = veracity_score
        return M, telem


class CentralCognitiveRouter(nn.Module):
    """
    Recurrent Toroidal Central Matrix Router ("Otak Berputar" — HADL v5.8).

    Connects all R=4 sub-parts of the Non-Token Latent Thought Matrix M in R^{B x S x R x D}
    and circulates information across them over C recurrent micro-cycles before synthesizing:
      1. Invariant-preserving Layer-7 residual representation:
           h_routed = M_0 + g_central * P_perp(M_0)(sum_{r=1}^{R-1} w_r M_r^{(C)})
         guaranteeing <h_routed - M_0, M_0> = 0 (100% inner-product preservation on factual anchors).
      2. Non-Token Cognitive Disclosure Direction d_disclosure in S^{D-1} subset R^D and
         Disclosure Gate g_disc in (0, 1), which interface with the foundation model's
         frozen peripheral language codec (lm_head) without hardcoded keyword lists.
    """
    def __init__(
        self,
        d_model: int = 2048,
        num_subparts: int = 4,
        router_dim: int = 64,
        disclosure_dim: int = 256,
        num_cycles: int = 2,
        max_ortho_ratio: float = 0.015
    ):
        super().__init__()
        self.d_model = d_model
        self.num_subparts = num_subparts
        self.router_dim = min(router_dim, max(16, d_model // 8))
        self.disclosure_dim = min(disclosure_dim, max(32, d_model // 4))
        self.num_cycles = num_cycles
        self.max_ortho_ratio = max_ortho_ratio

        self.q_proj = nn.Linear(d_model, self.router_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.router_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.router_dim, bias=False)
        self.out_proj = nn.Linear(self.router_dim, d_model, bias=False)

        # Learnable toroidal circulation bias encouraging cyclic interaction (0->1->2->3->0)
        torus_init = torch.zeros(num_subparts, num_subparts)
        for r in range(num_subparts):
            torus_init[r, (r + 1) % num_subparts] = 0.35
            torus_init[r, r] = 0.50
        self.torus_bias = nn.Parameter(torus_init)

        # Dynamic subpart gate controller: maps [R * router_dim] + [u_drive, veracity_score] -> R weights
        self.gate_controller = nn.Linear(num_subparts * self.router_dim + 2, num_subparts, bias=True)
        self.cycle_step_scale = nn.Parameter(torch.tensor(0.20))

        # Non-Token Cognitive Disclosure Projector (Brain -> Peripheral Language Codec)
        self.subpart_norm = nn.LayerNorm(d_model)
        self.disclosure_token_attn = nn.Linear(d_model, 1, bias=True)
        self.disclosure_down = nn.Linear(d_model, self.disclosure_dim, bias=True)
        self.disclosure_mid = nn.Linear(self.disclosure_dim, self.disclosure_dim, bias=True)
        self.disclosure_up = nn.Linear(self.disclosure_dim, d_model, bias=False)
        self.disclosure_gate_head = nn.Linear(self.disclosure_dim, 1, bias=True)
        self.disclosure_scale = nn.Parameter(torch.tensor(16.0))

        # Live differentiable tensors from the most recent forward pass
        self.last_gate_logits: Optional[torch.Tensor] = None
        self.last_subpart_weights: Optional[torch.Tensor] = None
        self.last_disclosure_vector: Optional[torch.Tensor] = None
        self.last_disclosure_gate: Optional[torch.Tensor] = None
        self.last_disclosure_gate_logit: Optional[torch.Tensor] = None
        self.last_circulated_matrix: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for layer in [
                self.q_proj, self.k_proj, self.v_proj, self.out_proj,
                self.disclosure_down, self.disclosure_mid, self.disclosure_up
            ]:
                w = torch.empty(layer.weight.shape, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(w)
                if layer is self.out_proj:
                    w.mul_(0.05)
                layer.weight.data.copy_(w.to(device=layer.weight.device, dtype=layer.weight.dtype))
                if getattr(layer, "bias", None) is not None:
                    nn.init.zeros_(layer.bias)

            nn.init.normal_(self.gate_controller.weight, std=0.02)
            # Bias slightly favors M_0 (Anchor) by default for stability
            self.gate_controller.bias.data.copy_(
                torch.tensor([0.8, 0.0, 0.1, 0.1], dtype=self.gate_controller.bias.dtype)
            )
            nn.init.normal_(self.disclosure_token_attn.weight, std=0.01)
            nn.init.zeros_(self.disclosure_token_attn.bias)
            nn.init.normal_(self.disclosure_gate_head.weight, std=0.01)
            nn.init.constant_(self.disclosure_gate_head.bias, -2.0)

    def project_disclosure_to_vocab(
        self,
        lm_head_or_weight: Union[nn.Module, torch.Tensor],
        d_disclosure: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Projects the non-token cognitive disclosure direction d_disclosure in S^{D-1}
        through the foundation model's frozen peripheral language codec (lm_head).
        """
        vec = d_disclosure if d_disclosure is not None else self.last_disclosure_vector
        if vec is None:
            raise RuntimeError("No disclosure vector available; run CentralCognitiveRouter.forward first.")
        w_lm = lm_head_or_weight.weight if isinstance(lm_head_or_weight, nn.Module) else lm_head_or_weight
        scale = torch.clamp(self.disclosure_scale, min=1.0, max=60.0).to(device=w_lm.device, dtype=w_lm.dtype)
        vec_norm = F.normalize(vec.to(device=w_lm.device, dtype=w_lm.dtype), p=2, dim=-1)
        return scale * F.linear(vec_norm, w_lm)

    def forward(
        self,
        M: torch.Tensor,
        u_drive: Optional[torch.Tensor] = None,
        veracity_conflict_score: Union[float, torch.Tensor] = 0.0,
        num_cycles: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes recurrent toroidal routing across the R sub-parts of Latent Matrix M.

        Args:
            M: [B, S, R, D] Latent Thought Matrix where M[:, :, 0, :] is the base anchor
            u_drive: optional [B, S, 1] pacemaker driving signal
            veracity_conflict_score: scalar float or [B, 1] tensor in [-1, +1] from Epistemic Veracity Subspace
            num_cycles: optional override for recurrent rotation cycles C

        Returns:
            h_routed: [B, S, D] invariant-preserving routed hidden state
            telemetry: dictionary of Central Cognitive Router metrics
        """
        B, S, R, D = M.shape
        cycles = num_cycles if num_cycles is not None else self.num_cycles
        target_dtype = self.q_proj.weight.dtype

        m0 = M[:, :, 0, :]  # [B, S, D] strictly invariant anchor
        M_curr = M.to(dtype=target_dtype)
        scale_factor = 1.0 / math.sqrt(self.router_dim)
        alpha = torch.clamp(self.cycle_step_scale, min=0.05, max=0.50).to(dtype=target_dtype)
        b_torus = self.torus_bias.to(device=M.device, dtype=target_dtype).view(1, 1, R, R)

        # Recurrent Toroidal Circulation ("Otak Berputar")
        for _ in range(max(1, cycles)):
            Q = self.q_proj(M_curr)  # [B, S, R, d_r]
            K = self.k_proj(M_curr)  # [B, S, R, d_r]
            V = self.v_proj(M_curr)  # [B, S, R, d_r]

            scores = torch.matmul(Q, K.transpose(-1, -2)) * scale_factor + b_torus  # [B, S, R, R]
            attn_matrix = F.softmax(scores, dim=-1)
            mixed_v = torch.matmul(attn_matrix, V)  # [B, S, R, d_r]
            delta_M = self.out_proj(mixed_v)        # [B, S, R, D]

            # Keep M_0 invariant; update M_1..M_{R-1} and re-project onto P_perp(M_0)
            m0_t = m0.to(dtype=target_dtype)
            updated_subparts = [m0_t]
            for r in range(1, R):
                m_r_new = M_curr[:, :, r, :] + alpha * delta_M[:, :, r, :]
                m_r_ortho = ConcurrentDualStreamSynthesizer.orthogonal_symplectic_project(m_r_new, m0_t)
                updated_subparts.append(m_r_ortho)
            M_curr = torch.stack(updated_subparts, dim=2)

        # Sequence-level proposition pooling across all R circulated sub-parts
        M_norm = self.subpart_norm(M_curr)  # [B, S, R, D]
        if S == 1:
            M_pool = M_norm[:, 0, :, :]  # [B, R, D]
        else:
            attn_logits = self.disclosure_token_attn(M_norm[:, :, 0, :])  # [B, S, 1]
            attn_w = F.softmax(attn_logits, dim=1).unsqueeze(2)           # [B, S, 1, 1]
            M_mean = M_norm.mean(dim=1)                                   # [B, R, D]
            M_attn = torch.sum(attn_w * M_norm, dim=1)                    # [B, R, D]
            M_last = M_norm[:, -1, :, :]                                  # [B, R, D]
            M_pool = 0.45 * M_mean + 0.35 * M_attn + 0.20 * M_last        # [B, R, D]

        # Compute dynamic sub-part routing weights w in Delta^{R-1}
        q_pool = self.q_proj(M_pool)  # [B, R, d_r]
        q_flat = q_pool.reshape(B, R * self.router_dim)
        u_mean = (
            u_drive.mean(dim=1).to(device=M.device, dtype=target_dtype)
            if u_drive is not None else torch.zeros(B, 1, device=M.device, dtype=target_dtype)
        )
        if u_mean.dim() == 1:
            u_mean = u_mean.unsqueeze(-1)
        if isinstance(veracity_conflict_score, torch.Tensor):
            v_tensor = veracity_conflict_score.to(device=M.device, dtype=target_dtype).reshape(B, 1)
            v_scalar = float(v_tensor.detach().mean().item())
        else:
            v_scalar = float(veracity_conflict_score)
            v_tensor = torch.full((B, 1), v_scalar, device=M.device, dtype=target_dtype)

        gate_in = torch.cat([q_flat, u_mean, v_tensor], dim=-1)
        gate_logits = self.gate_controller(gate_in)  # [B, R]

        # Elevate veracity subpart weight dynamically when veracity conflict is strong
        if R >= 3:
            v_boost = torch.where(
                v_tensor.abs() > 0.25,
                1.5 * v_tensor.abs(),
                torch.zeros_like(v_tensor)
            ).squeeze(-1)
            gate_logits = gate_logits.clone()
            gate_logits[:, 2] = gate_logits[:, 2] + v_boost

        subpart_weights = F.softmax(gate_logits, dim=-1)  # [B, R]
        w_exp = subpart_weights.view(B, 1, R, 1).to(dtype=M.dtype)

        # Synthesize Non-Token Cognitive Disclosure Vector in S^{D-1} and Disclosure Gate
        exploratory_mean = M_pool[:, 1:, :].mean(dim=1)  # [B, D]
        c_thought = torch.sum(subpart_weights.unsqueeze(-1) * M_pool, dim=1) + 0.25 * exploratory_mean
        z_disc = torch.tanh(self.disclosure_mid(F.silu(self.disclosure_down(c_thought))))  # [B, d_disc]
        d_raw = self.disclosure_up(z_disc)  # [B, D]
        d_disclosure = F.normalize(d_raw, p=2, dim=-1)  # [B, D] on unit hypersphere S^{D-1}
        disc_gate_logit = self.disclosure_gate_head(z_disc)  # [B, 1]
        disc_gate = torch.sigmoid(disc_gate_logit)           # [B, 1]

        self.last_gate_logits = gate_logits
        self.last_subpart_weights = subpart_weights
        self.last_disclosure_vector = d_disclosure
        self.last_disclosure_gate = disc_gate
        self.last_disclosure_gate_logit = disc_gate_logit
        self.last_circulated_matrix = M_curr

        # Combine orthogonal exploratory sub-parts (r = 1..R-1) for Layer 7 residual stream
        M_final = M_curr.to(dtype=M.dtype)
        delta_combined = (
            w_exp[:, :, 1, :] * M_final[:, :, 1, :]
            + w_exp[:, :, 2, :] * M_final[:, :, 2, :]
            + w_exp[:, :, 3, :] * M_final[:, :, 3, :]
        )
        delta_ortho = ConcurrentDualStreamSynthesizer.orthogonal_symplectic_project(delta_combined, m0)

        # Homeostatic relative norm bound so factual embeddings never drift
        base_norm = torch.norm(m0, dim=-1, keepdim=True) + 1e-6
        delta_norm = torch.norm(delta_ortho, dim=-1, keepdim=True) + 1e-6
        bound_scale = torch.clamp((self.max_ortho_ratio * base_norm) / delta_norm, max=1.0)
        delta_safe_ortho = bound_scale * delta_ortho

        h_routed = m0 + delta_safe_ortho

        # Verify strict orthogonality: <h_routed - m0, m0> == 0
        ortho_err = float(torch.abs(torch.sum(delta_safe_ortho * m0, dim=-1)).mean().item())
        base_norm_sq = float(torch.sum(m0 * m0, dim=-1).mean().item()) + 1e-6
        rel_ortho_err = ortho_err / base_norm_sq

        w_mean = subpart_weights.mean(dim=0)
        entropy = -float(torch.sum(w_mean * torch.log(w_mean + 1e-8)).item())
        dominant_idx = int(torch.argmax(w_mean).item())
        subpart_names = ["anchor_factual", "wild_exploration", "epistemic_veracity", "banach_structural"]

        telem = {
            "central_router_cycles": max(1, cycles),
            "central_router_dominant": subpart_names[dominant_idx % len(subpart_names)],
            "central_router_weights": [round(float(x.item()), 4) for x in w_mean],
            "central_router_entropy": round(entropy, 4),
            "veracity_conflict_score": round(v_scalar, 4),
            "central_disclosure_gate": round(float(disc_gate.detach().mean().item()), 4),
            "orthogonal_inner_product_error": round(rel_ortho_err, 8),
            "inner_product_preservation_ratio": round(1.0 - rel_ortho_err, 6),
        }
        return h_routed, telem


__all__ = [
    "FastSigmoidSurrogate",
    "spike_fn",
    "PersonalityProfile",
    "ModelPersonalityAdapter",
    "PreRouterSpikingPacemaker",
    "TourLeaderAttractor",
    "ConcurrentDualStreamSynthesizer",
    "DirectionalSafetyProjection",
    "LatentThoughtMatrixEngine",
    "CentralCognitiveRouter",
]

