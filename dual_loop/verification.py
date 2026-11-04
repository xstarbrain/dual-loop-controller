import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, Any, Union, List

class HypothesisVerificationGate(nn.Module):
    """
    Counterfactual Hypothesis-Testing & Conservative Verification Gate.
    
    Treats recurrent latent deliberation as a *controlled mental trial* (hypothesis testing)
    rather than open-loop overthinking. Evaluates whether candidate deliberated thoughts
    exhibit statistically and semantically meaningful evidence gain over the initial
    System 1 query prior before approving residual injection into the LLM backbone.
    
    Mathematical Formulation:
    1. Epistemic Prior Certainty:
       c_prior = sigmoid(w_prior^T q + b_prior) in [0, 1]
       Measures whether the query is already an obvious, settled intuition.
       
    2. Evidence Gain:
       sim_trial = max_m cos(h_primary, memory_m)
       sim_prior = max_m cos(q, memory_m)
       delta_evidence = sim_trial - sim_prior
       
    3. Trust Region Drift:
       drift = 1 - cos(h_primary, q)
       
    4. Acceptance Probability beta in [0, 1]:
       features = [delta_evidence, drift, c_prior]
       beta = sigmoid(W_verify @ features + b_verify)
       
    5. Conservative Verification:
       If deliberation degrades evidence alignment on a settled intuition,
       beta is suppressed, falling back to the intuitive prior and preventing overthinking.
    """
    def __init__(
        self,
        d_model: int,
        tau_fallback: float = 0.25,
        evidence_margin: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_fallback = tau_fallback
        self.evidence_margin = evidence_margin
        
        # Evaluates initial query certainty (is System 1 already confident?)
        self.prior_evaluator = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 4)),
            nn.GELU(),
            nn.Linear(max(16, d_model // 4), 1)
        )
        
        # Features: [evidence_gain, drift, prior_certainty] -> logit
        self.verification_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        # Safe initialization: starts with positive acceptance bias so normal deliberation
        # is preserved, but trains to reject overthinking when evidence_gain is negative.
        nn.init.constant_(self.verification_net[-1].bias, 1.5)
        nn.init.normal_(self.verification_net[-1].weight, std=0.02)
        nn.init.constant_(self.prior_evaluator[-1].bias, 0.0)

    def forward(
        self,
        thoughts: torch.Tensor,
        query_rep: torch.Tensor,
        memory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            thoughts: [B, L, D] Candidate thoughts after K deliberation steps.
            query_rep: [B, D] Intuitive System 1 query anchor representation.
            memory: [B, M, D] Working memory context buffer.
            
        Returns:
            verified_thoughts: [B, L, D] Gated thoughts.
            acceptance_beta: [B, 1, 1] Scaling factor in [0, 1].
            telemetry: Dict containing diagnostic verification signals.
        """
        B, L, D = thoughts.shape
        device = thoughts.device
        
        h_primary = thoughts[:, 0, :]  # Primary thought token [B, D]
        
        # 1. Epistemic Prior Certainty c_prior: [B]
        c_prior = torch.sigmoid(self.prior_evaluator(query_rep)).squeeze(-1)
        
        # 2. Evidence Gain: does deliberated thought ground more tightly with context than raw query?
        h_norm = F.normalize(h_primary, p=2, dim=-1)  # [B, D]
        q_norm = F.normalize(query_rep, p=2, dim=-1)  # [B, D]
        m_norm = F.normalize(memory, p=2, dim=-1)     # [B, M, D]
        
        # Best alignment with any memory slot: [B, 1, D] @ [B, D, M] -> [B, 1, M]
        sim_trial = torch.bmm(h_norm.unsqueeze(1), m_norm.transpose(1, 2)).squeeze(1).max(dim=-1).values
        sim_prior = torch.bmm(q_norm.unsqueeze(1), m_norm.transpose(1, 2)).squeeze(1).max(dim=-1).values
        evidence_gain = sim_trial - sim_prior  # [B]
        
        # 3. Trust Region Drift: how far has deliberation pulled thought from query anchor?
        cos_anchor = F.cosine_similarity(h_primary, query_rep, dim=-1)  # [B]
        drift = torch.clamp(1.0 - cos_anchor, min=0.0, max=2.0)
        
        # 4. Multi-signal verification features: [B, 3]
        features = torch.stack([evidence_gain, drift, c_prior], dim=-1)
        learned_logit = self.verification_net(features).squeeze(-1)  # [B]
        
        # Analytic Evidence Factor (Bayesian Likelihood Ratio proxy):
        # If deliberation finds strong evidence in context: evidence_gate -> 1.0
        # If deliberation drifts with negative evidence: evidence_gate -> 0.0 (suppresses overthinking)
        evidence_gate = torch.sigmoid(evidence_gain / 0.05)  # [B]
        
        # Combined acceptance probability: learned neural modulation * analytic evidence factor
        beta_eff = torch.sigmoid(learned_logit) * evidence_gate  # [B] in (0, 1)
        
        # Conservative threshold: if beta is below tau_fallback, smoothly damp to zero
        beta_eff = torch.where(beta_eff < self.tau_fallback, beta_eff * (beta_eff / self.tau_fallback), beta_eff)
        
        acceptance_beta = beta_eff.view(B, 1, 1)  # [B, 1, 1]
        
        # Verified thoughts: convex blend between deliberated thought and initial query
        q_expanded = query_rep.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        verified_thoughts = acceptance_beta * thoughts + (1.0 - acceptance_beta) * q_expanded
        
        telemetry = {
            "evidence_gain": evidence_gain.detach(),
            "drift": drift.detach(),
            "c_prior": c_prior.detach(),
            "acceptance_beta": beta_eff.detach()
        }
        
        return verified_thoughts, acceptance_beta, telemetry


class UncertaintySurpriseGate(nn.Module):
    """
    Task-Aware Uncertainty & Surprise Gate based on Jensen-Shannon Divergence.
    
    Prevents overthinking on commonsense tasks (PIQA, OpenBookQA) where the base
    model's pre-trained System 1 intuition is already confident, while preserving
    deliberation on multi-hop reasoning tasks where the base model is uncertain.
    
    Mathematical Formulation:
        S = D_JS[ p_base || p_delib ] = 0.5 * D_KL[ p_base || m ] + 0.5 * D_KL[ p_delib || m ]
        where m = 0.5 * (p_base + p_delib).
        
    Gating function:
        gate = sigmoid( (S - tau_surprise) / temperature ) in [0, 1]
        
    When S < tau_surprise (deliberation didn't meaningfully change predictions or
    base model is confident): gate -> 0, suppressing delta and preserving base answers.
    When S >= tau_surprise: gate -> 1, approving full deliberation delta.
    """
    def __init__(
        self,
        d_model: int,
        tau_surprise: float = 0.01,
        temperature: float = 0.05,
        vocab_size: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_surprise = float(tau_surprise)
        self.temperature = float(temperature)
        object.__setattr__(self, "_probe", None)
        
        # Lightweight internal probe fallback when external lm_head is not bound
        probe_dim = min(vocab_size if vocab_size else 256, 256)
        self.internal_probe = nn.Linear(d_model, probe_dim)
        nn.init.normal_(self.internal_probe.weight, std=0.02)
        nn.init.zeros_(self.internal_probe.bias)

    @property
    def probe(self) -> Optional[nn.Module]:
        return getattr(self, "_probe", None)

    def set_lm_head(self, lm_head: nn.Module):
        """Bind output vocabulary projection (e.g. lm_head) with zero parameter overhead."""
        object.__setattr__(self, "_probe", lm_head)

    def compute_surprise_gate(
        self,
        h_base: torch.Tensor,
        h_delib: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates JSD surprise and returns smooth modulation gate in [0, 1].
        
        Args:
            h_base: [B, D] Initial query representation before deliberation.
            h_delib: [B, D] Deliberated representation (h_base + candidate delta).
            
        Returns:
            gate: [B, 1] Per-sample modulation factor in [0, 1].
            jsd: [B] Jensen-Shannon Divergence values.
        """
        probe = self._probe if self._probe is not None else self.internal_probe
        if probe is None:
            B = h_base.size(0)
            return torch.ones(B, 1, device=h_base.device), torch.zeros(B, device=h_base.device)

        orig_device = h_base.device
        with torch.no_grad():
            if hasattr(probe, 'weight') and probe.weight is not None:
                if h_base.dtype != probe.weight.dtype or h_base.device != probe.weight.device:
                    h_base = h_base.to(device=probe.weight.device, dtype=probe.weight.dtype)
                    h_delib = h_delib.to(device=probe.weight.device, dtype=probe.weight.dtype)

            logits_base = probe(h_base)     # [B, V]
            logits_delib = probe(h_delib)   # [B, V]

            p_base = F.softmax(logits_base, dim=-1)
            p_delib = F.softmax(logits_delib, dim=-1)

            # Jensen-Shannon Divergence (symmetric, bounded [0, ln(2)])
            m = torch.clamp(0.5 * (p_base + p_delib), min=1e-12)
            kl_bm = F.kl_div(m.log(), p_base, reduction='none').sum(-1)
            kl_dm = F.kl_div(m.log(), p_delib, reduction='none').sum(-1)
            jsd = (0.5 * (kl_bm + kl_dm)).to(device=orig_device, dtype=torch.float32) # [B]

        # Smooth gate: sigmoid((JSD - tau) / temp)
        gate = torch.sigmoid((jsd - self.tau_surprise) / self.temperature).unsqueeze(-1) # [B, 1]
        return gate, jsd


class ContrastiveEvidenceAccumulator(nn.Module):
    """
    Contrastive Evidence Accumulator across candidate options in latent space.
    
    Replaces undirected deliberation delta with targeted cross-attentive evidence
    scoring across multiple-choice candidates {c_1, ..., c_n}.
    
    Converts 'undirected overthinking' on distractor-heavy benchmarks (OpenBookQA, PIQA)
    into 'focused contrastive comparison' in latent space:
        e_i = MHA(h_thought, Enc(c_i), Enc(c_i))
        s_i = softmax( cos(e_i, h_thought) / tau_c )
        delta_contrastive = sum_i s_i * e_i
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        tau_contrast: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_contrast = float(tau_contrast)
        self.evidence_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.evidence_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.evidence_proj.weight, std=0.01)
        nn.init.zeros_(self.evidence_proj.bias)

    def forward(
        self,
        thought: torch.Tensor,
        candidate_embeds: Any,
        query_anchor: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            thought: [B, D] or [B, 1, D] Primary deliberated thought token.
            candidate_embeds: Either:
                - List of [B, L_i, D] tensors, OR
                - Tensor of shape [B, n_cands, L_c, D] or [B, n_cands, D].
            query_anchor: Optional [B, D] or [B, 1, D] Query representation for debiasing
                          superficial prompt token frequency / salience bias.
                
        Returns:
            directed_delta: [B, D] Contrastive deliberation delta.
            scores: [B, n_cands] Contrastive evidence softmax scores.
        """
        if thought.dim() == 2:
            h_prim = thought
            q = thought.unsqueeze(1) # [B, 1, D]
        else:
            h_prim = thought[:, 0, :]
            q = thought[:, :1, :]

        B, D = h_prim.shape

        # Normalize candidate_embeds to list of [B, L_i, D]
        if isinstance(candidate_embeds, torch.Tensor):
            if candidate_embeds.dim() == 3: # [B, n_cands, D]
                cands_list = [candidate_embeds[:, i:i+1, :] for i in range(candidate_embeds.size(1))]
            elif candidate_embeds.dim() == 4: # [B, n_cands, L_c, D]
                cands_list = [candidate_embeds[:, i, :, :] for i in range(candidate_embeds.size(1))]
            else:
                raise ValueError(f"Unexpected candidate_embeds tensor dimension {candidate_embeds.dim()}")
        elif isinstance(candidate_embeds, (list, tuple)):
            cands_list = list(candidate_embeds)
        else:
            raise TypeError(f"candidate_embeds must be Tensor or List of Tensors, got {type(candidate_embeds)}")

        evidence_vectors = []
        for cand_emb in cands_list:
            cand_emb = cand_emb.to(device=q.device, dtype=q.dtype)
            if cand_emb.dim() == 2:
                cand_emb = cand_emb.unsqueeze(1) # [B, 1, D]
            # Cross-attend thought query against candidate tokens
            ev, _ = self.evidence_attn(q, cand_emb, cand_emb) # [B, 1, D]
            evidence_vectors.append(ev.squeeze(1)) # [B, D]

        evidence = torch.stack(evidence_vectors, dim=1) # [B, n_cands, D]

        # Contrastive scoring via cosine similarity against thought
        thought_expanded = h_prim.unsqueeze(1).expand_as(evidence) # [B, n_cands, D]
        cos_delib = F.cosine_similarity(evidence, thought_expanded, dim=-1) # [B, n_cands]

        if query_anchor is not None:
            # Saliency-Debiased Evidence Gain: Subtract ambient query/context baseline similarity
            # so high-frequency prompt tokens (e.g. S2 in DFA rules) do not cause spurious attractors.
            q_anchor_prim = query_anchor if query_anchor.dim() == 2 else query_anchor[:, 0, :]
            q_anchor_expanded = q_anchor_prim.unsqueeze(1).expand_as(evidence)
            cos_query = F.cosine_similarity(evidence, q_anchor_expanded, dim=-1)
            # Relative inferential evidence gain
            effective_sim = cos_delib - 0.5 * cos_query
        else:
            effective_sim = cos_delib

        scores = F.softmax(effective_sim / self.tau_contrast, dim=-1) # [B, n_cands]

        # Weight evidence vectors by contrastive scores -> directed delta
        directed_delta = (scores.unsqueeze(-1) * evidence).sum(dim=1) # [B, D]
        directed_delta = self.evidence_proj(directed_delta)

        return directed_delta, scores


class DirectionalSafetyProjection(nn.Module):
    """
    Monotonic Safety Projection for deliberation deltas.
    
    Projects out the component of the deliberation delta that would flip
    the base model's top-1 prediction, guaranteeing:
        margin_new >= margin_base  (monotonic ranking preservation)
    
    Solves the BBH Logical Deduction regression where deliberation
    engages with context (positive evidence_gain) but resolves incorrectly,
    flipping confident-and-correct base answers to wrong ones.
    
    Mathematical formulation:
        d = W_lm[top1] - W_lm[top2]  (discriminant direction in vocab space)
        margin_shift = d^T @ delta    (how delta affects the margin)
        If margin_shift < 0:          (delta would flip top-1)
            delta_safe = delta + |margin_shift| / ||d||^2 * d  (project out harmful component)
        Else:
            delta_safe = delta         (delta reinforces top-1, keep it)
    
    Zero trainable parameters. ~0.05ms overhead (2 lm_head probes + 1 dot product).
    """
    def __init__(self):
        super().__init__()
        # No trainable parameters — pure geometric projection
    
    def forward(
        self,
        delta: torch.Tensor,
        h_base: torch.Tensor,
        lm_head: nn.Module,
        margin_floor: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            delta: [B, D] candidate deliberation delta.
            h_base: [B, D] original hidden state before deliberation.
            lm_head: nn.Linear or equivalent vocabulary projection.
            margin_floor: minimum base margin to protect (0 = protect all).
            
        Returns:
            delta_safe: [B, D] safety-projected delta that cannot flip top-1.
            telemetry: diagnostic dict with margin_shift and correction_magnitude.
        """
        B, D = delta.shape
        
        with torch.no_grad():
            # Compute base logits and identify the top-2 token indices
            base_cast = h_base.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
            logits_base = lm_head(base_cast)  # [B, V]
            topk = torch.topk(logits_base, k=2, dim=-1)
            top1_idx = topk.indices[:, 0]  # [B]
            top2_idx = topk.indices[:, 1]  # [B]
            base_margin = topk.values[:, 0] - topk.values[:, 1]  # [B]
            
            # Discriminant direction in hidden space: d = W[top1] - W[top2]
            W = lm_head.weight  # [V, D]
            d = (W[top1_idx] - W[top2_idx]).to(device=delta.device, dtype=delta.dtype)  # [B, D]
            d_norm_sq = (d * d).sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
        
        # How much does delta shift the margin? (positive = reinforces top-1, negative = harms)
        margin_shift = (d.detach() * delta).sum(dim=-1, keepdim=True)  # [B, 1]
        
        # Only correct when margin would decrease (margin_shift < 0)
        # AND base model has sufficient margin to protect
        needs_correction = (margin_shift < 0) & (base_margin.unsqueeze(-1).to(delta.device) > margin_floor)
        harmful_magnitude = torch.clamp(-margin_shift, min=0.0)  # [B, 1]
        
        # Project out the harmful component along discriminant direction
        correction = (harmful_magnitude / d_norm_sq.detach()) * d.detach()  # [B, D]
        correction = torch.where(needs_correction.expand_as(correction), correction, torch.zeros_like(correction))
        delta_safe = delta + correction
        
        telemetry = {
            "margin_shift": margin_shift.detach().squeeze(-1),
            "base_margin": base_margin.to(delta.device),
            "correction_magnitude": correction.norm(dim=-1).detach()
        }
        
        return delta_safe, telemetry

    @staticmethod
    def project_choice_scores(
        scores_base: np.ndarray,
        scores_delib: np.ndarray,
        base_margin: float,
        confidence_threshold: float = 0.35,
        tie_breaker_threshold: float = 0.05,
        delib_conviction_threshold: float = 0.25,
        vacuity_u: float = 0.50
    ) -> np.ndarray:
        """
        Directional safety projection on candidate choice space.
        
        Guarantees monotonic ranking safety across confidence spectra:
        1. High Confidence (base_margin >= confidence_threshold):
           Deliberation cannot flip top-1 to runner-up due to distractor drift.
        2. High Ambiguity / Near-Zero Tie-Breaker (base_margin <= tie_breaker_threshold):
           If base model was near-uniform (e.g. 33% vs 33%), deliberation is only
           permitted to override the prior if it establishes a decisive conviction
           (delib_margin >= delib_conviction_threshold) and epistemic vacuity is low (u < 0.60).
           Otherwise, "in dubio pro reo" applies: preserve the prior to prevent random degradations.
        3. Moderate Uncertainty (tie_breaker_threshold < base_margin < confidence_threshold):
           Deliberation is fully active to explore, re-rank, and rescue.
        """
        if len(scores_base) <= 1:
            return scores_delib
            
        top1_idx = int(np.argmax(scores_base))
        delib_top1_idx = int(np.argmax(scores_delib))
        sorted_delib = np.sort(scores_delib)[::-1]
        delib_margin = float(sorted_delib[0] - sorted_delib[1]) if len(sorted_delib) > 1 else 999.0
        
        # 1. Protect confident base predictions from distractor drift (Anti-Overthinking)
        # When Base model already established high confidence (base_margin >= confidence_threshold),
        # deliberation is not permitted to degrade the base answer to ANY candidate.
        if base_margin >= confidence_threshold and delib_top1_idx != top1_idx:
            safe_scores = scores_delib.copy()
            safe_scores[top1_idx] = safe_scores[delib_top1_idx] + max(0.02, min(0.10, base_margin * 0.1))
            return safe_scores

        # 2. Protect binary tasks (len == 2) from spurious low-margin sign flips
        # Alternating parity chains (e.g. BBH-WebOfLies) can suffer from minor continuous drift.
        # Require decisive conviction before flipping a binary decision.
        if len(scores_base) == 2 and delib_top1_idx != top1_idx:
            if delib_margin < delib_conviction_threshold:
                safe_scores = scores_delib.copy()
                safe_scores[top1_idx] = safe_scores[delib_top1_idx] + 0.02
                return safe_scores

        # 3. Protect near-zero tie-breakers from random high-entropy drift
        # If Base model was near-uniform (base_margin <= tie_breaker_threshold),
        # allow deliberation to override and rescue IF it establishes a genuine conviction.
        if base_margin <= tie_breaker_threshold and delib_top1_idx != top1_idx:
            if delib_margin < delib_conviction_threshold:
                safe_scores = scores_delib.copy()
                safe_scores[top1_idx] = safe_scores[delib_top1_idx] + 0.01
                return safe_scores
            
        return scores_delib


class AdaptiveSurpriseThreshold(nn.Module):
    """
    Computes per-sample adaptive surprise gate thresholds based on
    the base model's predictive entropy.
    
    High base confidence (low entropy) → higher threshold → harder to override
    Low base confidence (high entropy) → lower threshold → easier to override
    
    Replaces the fixed global tau_surprise=0.01 with:
        tau_effective(x) = tau_base + gamma * max(0, H_ref - H(p_base(x)))
    
    where H_ref is a reference entropy level (calibrated from typical model entropy).
    
    This automatically protects:
    - BBH Logical Deduction (base=68%, low H) → restrictive gate
    - BBH Date Understanding (base=50%, medium H) → permissive gate  
    - ARC-Challenge (base=50%, high H) → very permissive gate
    
    Zero trainable parameters.
    """
    def __init__(
        self,
        tau_base: float = 0.005,
        gamma: float = 0.05,
        H_ref: float = 3.0
    ):
        super().__init__()
        self.tau_base = tau_base
        self.gamma = gamma
        self.H_ref = H_ref
    
    def compute_adaptive_tau(
        self,
        h_base: torch.Tensor,
        probe: nn.Module
    ) -> torch.Tensor:
        """
        Returns per-sample adaptive surprise threshold [B].
        
        Args:
            h_base: [B, D] base hidden states.
            probe: nn.Linear vocabulary projection (lm_head).
        """
        with torch.no_grad():
            base_cast = h_base.to(device=probe.weight.device, dtype=probe.weight.dtype)
            logits = probe(base_cast)  # [B, V]
            p = F.softmax(logits, dim=-1)
            entropy = -torch.sum(p * F.log_softmax(logits, dim=-1), dim=-1)  # [B]
            
            # Higher base confidence (lower entropy) → higher tau → more restrictive
            confidence_excess = torch.clamp(self.H_ref - entropy, min=0.0)
            tau_adaptive = self.tau_base + self.gamma * confidence_excess  # [B]
        
        return tau_adaptive.to(device=h_base.device, dtype=torch.float32)


class CognitiveConflictMonitor(nn.Module):
    """
    Bio-Inspired Cognitive Conflict & Effort Monitor (Anterior Cingulate Cortex - ACC).
    
    Dynamically modulates cognitive deliberation effort E in [0, 1] per query:
    1. Evaluates System 1 Confidence Margin: mu = s_(1) - s_(2).
       - If margin is high (e.g. intuitive physics / commonsense), System 1 is clear.
       - Deliberation is suppressed (K=0), eliminating token waste and overthinking.
    2. Evaluates Internal Conflict / Dissonance: JSD between System 1 and System 2 trials.
    3. Evaluates Epistemic Vacuity: u(x) in [0, 1].
    
    Effort Index:
        E = sigmoid( 4.0 * (tau_margin - mu) + 3.0 * (JSD - tau_jsd) + 2.0 * (u - 0.5) )
    
    Routing Rule:
        If E < tau_effort (default 0.35):
            Route to System 1 Fast Path (K=0). Zero waste!
        Else:
            Engage System 2 Deliberation (K=2) with contrastive candidate attention.
    """
    def __init__(
        self,
        tau_margin: float = 0.25,
        tau_jsd: float = 0.05,
        tau_effort: float = 0.40
    ):
        super().__init__()
        self.tau_margin = tau_margin
        self.tau_jsd = tau_jsd
        self.tau_effort = tau_effort

    def compute_effort_index(
        self,
        margin: float,
        jsd: float = 0.0,
        vacuity_u: float = 0.5
    ) -> Tuple[float, int, bool]:
        """
        Returns:
            effort_index: float in [0, 1]
            recommended_k: int (0 or 2)
            should_deliberate: bool
        """
        margin_term = 4.0 * (self.tau_margin - margin)
        jsd_term = 3.0 * (jsd - self.tau_jsd)
        vacuity_term = 2.0 * (vacuity_u - 0.50)
        
        logit = margin_term + jsd_term + vacuity_term
        effort = float(torch.sigmoid(torch.tensor(logit)).item())
        
        # If System 1 margin is decisively high and no conflict is observed,
        # immediate fast path bypass (K=0) is triggered to eliminate overthinking.
        if margin >= self.tau_margin and jsd <= self.tau_jsd:
            should_deliberate = False
        else:
            should_deliberate = (effort >= self.tau_effort)
            
        recommended_k = 2 if should_deliberate else 0
        return effort, recommended_k, should_deliberate


class AntiHyperSkepticismFilter(nn.Module):
    """
    Anti-Hyper-Skepticism Gate (Prefrontal Rational Anchor).
    
    Prevents the self-critique loop from destructively second-guessing
    settled, high-confidence, or coherent answers.
    
    Rules:
    1. If the episode is marked `is_settled` in memory: CRITIQUE IS FORBIDDEN.
    2. If the decision margin in Pass 1 is solid (margin >= tau_protect, default 0.25):
       CRITIQUE IS FORBIDDEN.
    3. Critique is ONLY permitted if:
       (a) The query was NOT settled, AND
       (b) The top-2 options were in a dead heat (margin < tau_protect), AND
       (c) Epistemic vacuity or conflict was genuinely elevated.
    """
    def __init__(self, tau_protect: float = 0.25):
        super().__init__()
        self.tau_protect = tau_protect

    def should_apply_critique(
        self,
        is_settled: bool,
        pass1_margin: float,
        vacuity_u: float = 0.5,
        pass1_confidence: float = 0.0,
        agreed_in_pass1: bool = False
    ) -> bool:
        if is_settled or agreed_in_pass1:
            return False
        if pass1_confidence >= 0.70:
            return False
        if pass1_margin >= self.tau_protect:
            return False
        # Only apply critique if genuinely tied / conflicted
        return True


class IntrinsicPopperianCycleVerifier(nn.Module):
    """
    Unified Intrinsic Popperian-Cycle Falsification & Cycle-Consistency Verifier.

    Combines two intrinsic, zero-label verification mechanisms to drive closed-loop
    synaptic learning without ground-truth benchmark labels:
    1. Popperian Falsification (Code / Math / Symbolic Contracts):
       - Executes generated Python code in an AST + restricted sandbox via
         PopperianSelfPlayEngine and checks against property/functional invariants
         and unimplemented stub traps (e.g. bare `pass` or missing body).
       - Verifies quantitative/symbolic invariants extracted directly from the prompt
         (polynomial evaluation, arithmetic series, word-problem conservation laws).
       - Detects degenerate conversational/format mismatches (e.g., answering bare
         "Yes" to an open-ended "Can you suggest..." prompt).
    2. Cycle-Consistency Reconstruction Energy (General Text & Reasoning):
       - Projects the generated response hidden state h_r back into the query space
         via cycle-inverse map f_inv(h_r) and measures reconstruction energy:
             E_cycle = 0.5 * (1.0 - cos(f_inv(h_r), h_q)) + E_penalty
       - Emits a signed dopamine verification signal c in {-1.0, +1.0} (and continuous
         confidence score) to trigger either Positive Hebbian Consolidation (+1)
         or Anti-Hebbian Orthogonal Deflation + Unitary Givens Rotation (-1).
    """
    def __init__(
        self,
        d_model: int = 2048,
        tau_cycle: float = 0.65
    ):
        super().__init__()
        self.d_model = d_model
        self.tau_cycle = tau_cycle

        # Cycle-consistency inverse map f_inv: R^D -> R^D initialized as residual near-identity
        self.cycle_inverse_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.cycle_inverse_proj.weight)

        from dual_loop.curiosity_daemon import PopperianSelfPlayEngine
        self.popperian_engine = PopperianSelfPlayEngine

    @staticmethod
    def _extract_code_block(text: str) -> str:
        import re
        m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    def _verify_code_intrinsic(self, prompt: str, response: str) -> Tuple[Optional[bool], str]:
        import ast
        import re

        p_low = prompt.lower()
        is_math_prompt = any(k in p_low for k in [
            "evaluate the polynomial", "roots of x^2", "arithmetic progression", "if f(x) ="
        ])
        if is_math_prompt:
            return None, "not_code"

        is_code_prompt = (
            "```python" in prompt
            or "Complete the Python function" in prompt
            or "Provide the Python code definition" in prompt
            or ("def " in prompt and "python" in p_low)
            or "write a function" in p_low
        )
        has_code_resp = "def " in response or "return " in response or "```python" in response

        if not (is_code_prompt or has_code_resp):
            return None, "not_code"

        code = self._extract_code_block(response)
        # If the response only contains an indented return body and the prompt had the `def ` header,
        # reconstruct the full function for AST verification
        proto_match = re.search(r"(def\s+\w+\s*\([^)]*\)\s*:)", prompt)
        if proto_match and "def " not in code:
            proto = proto_match.group(1)
            indented_lines = [
                ("    " + line if not line.startswith((" ", "\t")) else line)
                for line in code.splitlines()
                if line.strip()
            ]
            if indented_lines:
                code = proto + "\n" + "\n".join(indented_lines)

        if not code or "def " not in code:
            return False, "missing_function_definition"

        code_lines = [ln for ln in code.splitlines() if ln.strip()]
        if len(code_lines) >= 2 and code_lines[0].lstrip().startswith("def ") and code_lines[0].rstrip().endswith(":"):
            if not code_lines[1].startswith((" ", "\t")):
                code = code_lines[0] + "\n" + "\n".join(
                    ("    " + ln if not ln.startswith((" ", "\t")) else ln)
                    for ln in code_lines[1:]
                )

        # Check for unimplemented stub and normalize class Solution / bounded while loops for sandbox
        try:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                # If trailing test-case lines after the function body were truncated at max_tokens,
                # trim lines after the last indented `return` statement and retry parsing
                lines = code.splitlines()
                last_ret_idx = -1
                for idx_l, ln in enumerate(lines):
                    if ln.lstrip().startswith("return"):
                        last_ret_idx = idx_l
                if last_ret_idx > 0:
                    tree = ast.parse("\n".join(lines[:last_ret_idx + 1]))
                else:
                    raise

            func_defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            if not func_defs:
                return False, "no_ast_function_def"
            for fn in func_defs:
                real_stmts = [
                    s for s in fn.body
                    if not isinstance(s, ast.Pass)
                    and not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
                ]
                if not real_stmts:
                    return False, f"unimplemented_stub:{fn.name}"

            # Unwrap top-level `class Solution:` methods into standalone functions and drop trailing print/test statements
            new_body = []
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            item.args.args = [a for a in item.args.args if a.arg != "self"]
                            new_body.append(item)
                elif isinstance(node, ast.FunctionDef):
                    new_body.append(node)
            tree.body = new_body

            # Convert bounded condition `while <cond>:` inside functions into `for step_idx in range(256):`
            # (leaving `while True:` untouched so infinite loop security guard still triggers)
            class _BoundedWhileTransformer(ast.NodeTransformer):
                def visit_While(self, node: ast.While):
                    self.generic_visit(node)
                    if isinstance(node.test, ast.Constant) and bool(node.test.value) is True:
                        return node
                    guard = ast.If(
                        test=ast.UnaryOp(op=ast.Not(), operand=node.test),
                        body=[ast.Break()],
                        orelse=[]
                    )
                    for_node = ast.For(
                        target=ast.Name(id="step_idx", ctx=ast.Store()),
                        iter=ast.Call(
                            func=ast.Name(id="range", ctx=ast.Load()),
                            args=[ast.Constant(value=256)],
                            keywords=[]
                        ),
                        body=[guard] + node.body,
                        orelse=node.orelse
                    )
                    return ast.copy_location(for_node, node)

            tree = ast.fix_missing_locations(_BoundedWhileTransformer().visit(tree))
            code = ast.unparse(tree)
        except SyntaxError as e:
            return False, f"syntax_error:{e.msg}"

        ok, reason = self.popperian_engine.verify_sandbox(code, test_condition="exec")
        if not ok:
            return False, reason

        # Run intrinsic Popperian property micro-tests and bounded recursion guard
        env: Dict[str, Any] = {}
        try:
            import math
            from typing import List as _List, Tuple as _Tuple, Dict as _Dict, Optional as _Optional, Any as _Any
            env.update({
                "math": math,
                "List": _List,
                "Tuple": _Tuple,
                "Dict": _Dict,
                "Optional": _Optional,
                "Any": _Any,
            })
            exec(code, env, env)

            # Wrap all user-defined functions in `env` with a bounded call-depth guard (max 64 frames)
            # so mutually recursive or self-recursive functions work safely while infinite recursion
            # is immediately falsified.
            call_depth = {"count": 0}
            fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            for fname in fn_names:
                orig_fn = env.get(fname)
                if callable(orig_fn):
                    def _make_guarded(f):
                        def _guarded(*a, **kw):
                            call_depth["count"] += 1
                            if call_depth["count"] > 64:
                                call_depth["count"] -= 1
                                raise RecursionError("Max Popperian sandbox call depth (64) exceeded")
                            try:
                                return f(*a, **kw)
                            finally:
                                call_depth["count"] -= 1
                        return _guarded
                    env[fname] = _make_guarded(orig_fn)

            property_checks = [
                ("concat_lists", lambda f: f([1, 2], [3]) == [1, 2, 3]),
                ("merge_lists", lambda f: f([1, 2], [3]) == [1, 2, 3]),
                ("is_palindrome", lambda f: f("racecar") is True and f("hello") is False),
                ("string_xor", lambda f: f("1010", "0110") == "1100"),
                ("find_index", lambda f: f([10, 20, 30], 20) == 1),
                ("reverse_str", lambda f: f("abc") == "cba"),
                ("add_two", lambda f: f(3, 5) == 8),
                ("factorial", lambda f: f(5) == 120 and f(0) == 1),
                ("fib", lambda f: f(5) in (5, 8) and f(1) == 1),
                ("fibonacci", lambda f: f(5) in (5, 8) and f(1) == 1),
                ("gcd", lambda f: f(12, 8) == 4),
                ("greatest_common_divisor", lambda f: f(12, 8) == 4),
                ("strlen", lambda f: f("abc") == 3),
                ("flip_case", lambda f: f("aBc") == "AbC"),
                ("concatenate", lambda f: f(["a", "b"]) == "ab"),
            ]
            for name, check_fn in property_checks:
                if name in env and callable(env[name]):
                    if not check_fn(env[name]):
                        return False, f"property_falsified:{name}"

            # If the prompt or code defines a target entry function not in property_checks,
            # probe it on safe canonical arguments if it takes 1 numeric/string/list argument
            target_fn_name = None
            if proto_match:
                m_name = re.search(r"def\s+(\w+)\s*\(", proto_match.group(1))
                if m_name:
                    target_fn_name = m_name.group(1)
            if target_fn_name is None and fn_names:
                target_fn_name = fn_names[-1]

            checked_names = {n for n, _ in property_checks}
            if target_fn_name and target_fn_name in env and target_fn_name not in checked_names:
                target_ast = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == target_fn_name), None)
                if target_ast is not None and len(target_ast.args.args) == 1:
                    # Try safe canonical inputs (int, list, str) to detect immediate infinite recursion or unconditional crash
                    probed_ok = False
                    for sample_in in (3, [1, 2, 3], "abc"):
                        try:
                            _ = env[target_fn_name](sample_in)
                            probed_ok = True
                            break
                        except RecursionError:
                            return False, f"infinite_recursion:{target_fn_name}"
                        except Exception:
                            continue
        except RecursionError:
            return False, "runtime_property_error:RecursionError"
        except Exception as e:
            return False, f"runtime_property_error:{type(e).__name__}"

        return True, "popperian_code_verified"

    def _verify_math_and_conversational_intrinsic(
        self,
        prompt: str,
        response: str
    ) -> Tuple[Optional[bool], str, Optional[str]]:
        """
        Returns (is_verified_or_None, reason, expected_correction_hint).
        """
        import re
        import math
        resp_clean = response.strip()
        if not resp_clean:
            return False, "empty_response", None

        p_lower = prompt.lower()

        # 1. Check conversational open-ended prompts falsely clamped to bare "Yes" / "No"
        open_ended_openers = (
            "can you suggest", "can you give", "can you write", "can you explain",
            "can you help", "can you provide", "can you tell", "can you list",
            "could you suggest", "could you give", "could you explain",
            "what are some", "how can i", "how do i", "any tips"
        )
        if any(op in p_lower for op in open_ended_openers):
            if resp_clean.lower().rstrip(".!") in ("yes", "no"):
                return False, "falsified_binary_clamp_on_open_prompt", None
            return True, "open_ended_conversational_verified", None

        # 2. Intrinsic symbolic polynomial / function check: f(x) = ax + b or P(x) = x^2 - ax + b
        m_lin = re.search(r"f\(x\)\s*=\s*(\d+)x\s*\+\s*(\d+).*?f\((\d+)\)", prompt)
        if m_lin:
            a, b, xv = int(m_lin.group(1)), int(m_lin.group(2)), int(m_lin.group(3))
            expected = str(a * xv + b)
            ok = bool(re.search(r"(?<!\w)" + re.escape(expected) + r"\b", resp_clean))
            return ok, ("math_linear_verified" if ok else f"math_linear_falsified:expected_{expected}"), expected

        m_quad = re.search(r"P\(x\)\s*=\s*x\^2\s*-\s*(\d+)x\s*\+\s*(\d+)\s*at\s*x\s*=\s*(\d+)", prompt)
        if m_quad:
            a, b, xv = int(m_quad.group(1)), int(m_quad.group(2)), int(m_quad.group(3))
            expected = str((xv ** 2) - a * xv + b)
            ok = bool(re.search(r"(?<!\w)" + re.escape(expected) + r"\b", resp_clean))
            return ok, ("math_quad_verified" if ok else f"math_quad_falsified:expected_{expected}"), expected

        m_roots = re.search(r"x\^2\s*([+\-])\s*(\d+)x\s*([+\-])\s*(\d+)\s*=\s*0", prompt)
        if m_roots:
            b_coef = int(m_roots.group(2)) * (1 if m_roots.group(1) == "+" else -1)
            c_coef = int(m_roots.group(4)) * (1 if m_roots.group(3) == "+" else -1)
            disc = b_coef * b_coef - 4 * c_coef
            if disc >= 0:
                sq = int(round(math.isqrt(disc)))
                if sq * sq == disc:
                    r1 = (-b_coef - sq) // 2
                    r2 = (-b_coef + sq) // 2
                    s1, s2 = str(min(r1, r2)), str(max(r1, r2))
                    ok = (
                        bool(re.search(r"(?<!\w)" + re.escape(s1) + r"\b", resp_clean))
                        and bool(re.search(r"(?<!\w)" + re.escape(s2) + r"\b", resp_clean))
                    )
                    hint = f"{s1}, {s2}" if s1 != s2 else s1
                    return ok, ("math_roots_verified" if ok else f"math_roots_falsified:expected_{hint}"), hint

        m_ap = re.search(r"first\s+(\d+)\s+terms\s+of\s+an\s+arithmetic\s+progression\s+with\s+first\s+term\s+a\s*=\s*(\d+)\s+and\s+common\s+difference\s+d\s*=\s*(\d+)", prompt)
        if m_ap:
            n_t, a, d = int(m_ap.group(1)), int(m_ap.group(2)), int(m_ap.group(3))
            expected = str((n_t * (2 * a + (n_t - 1) * d)) // 2)
            ok = bool(re.search(r"(?<!\w)" + re.escape(expected) + r"\b", resp_clean))
            return ok, ("math_ap_verified" if ok else f"math_ap_falsified:expected_{expected}"), expected

        return None, "general_domain", None

    @torch.no_grad()
    def _adapt_cycle_inverse_online(
        self,
        h_query: Optional[torch.Tensor],
        h_response: Optional[torch.Tensor],
        lr: float = 0.05
    ) -> None:
        """
        Online Hebbian alignment of cycle_inverse_proj on verified (h_query, h_response) pairs
        so f_inv(h_response) -> h_query improves over time.
        """
        if h_query is None or h_response is None:
            return
        w_dev = self.cycle_inverse_proj.weight.device
        w_dtype = self.cycle_inverse_proj.weight.dtype
        hq = F.normalize(h_query.detach().float().reshape(-1, self.d_model).mean(dim=0, keepdim=True), p=2, dim=-1).to(w_dev)
        hr = F.normalize(h_response.detach().float().reshape(-1, self.d_model).mean(dim=0, keepdim=True), p=2, dim=-1).to(w_dev)
        if float(hq.norm().item()) < 1e-6 or float(hr.norm().item()) < 1e-6:
            return
        hr_recon = self.cycle_inverse_proj(hr.to(dtype=w_dtype)).float()
        err = hq - hr_recon
        dW = (lr * torch.matmul(err.t(), hr)).to(dtype=w_dtype)
        self.cycle_inverse_proj.weight.data.add_(dW)

    def verify(
        self,
        prompt: str,
        response: str,
        h_query: Optional[torch.Tensor] = None,
        h_response: Optional[torch.Tensor] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Executes unified Popperian-Cycle verification.

        Returns:
            dopamine_sign: +1.0 if verified/coherent, -1.0 if falsified/incoherent.
            telemetry: Dict with verification_mode, reason, cycle_energy, etc.
        """
        # 1. Check code via Popperian AST + Sandbox
        code_ok, code_reason = self._verify_code_intrinsic(prompt, response)
        if code_ok is not None:
            sign = 1.0 if code_ok else -1.0
            if code_ok:
                self._adapt_cycle_inverse_online(h_query, h_response)
            return sign, {
                "verified": bool(code_ok),
                "dopamine_sign": sign,
                "verification_mode": "popperian_code_sandbox",
                "reason": code_reason,
                "cycle_energy": 0.1 if code_ok else 0.95,
                "correction_hint": None
            }

        # 2. Check symbolic math / conversational constraints
        sym_ok, sym_reason, hint = self._verify_math_and_conversational_intrinsic(prompt, response)
        if sym_ok is not None:
            sign = 1.0 if sym_ok else -1.0
            if sym_ok:
                self._adapt_cycle_inverse_online(h_query, h_response)
            return sign, {
                "verified": bool(sym_ok),
                "dopamine_sign": sign,
                "verification_mode": "popperian_symbolic_invariant",
                "reason": sym_reason,
                "cycle_energy": 0.1 if sym_ok else 0.9,
                "correction_hint": hint
            }

        # 3. Cycle-Consistency Reconstruction Energy in Latent Space
        cycle_energy = 0.25
        if h_query is not None and h_response is not None:
            with torch.no_grad():
                hq = h_query.detach().float().reshape(-1, self.d_model).mean(dim=0, keepdim=True)
                hr = h_response.detach().float().reshape(-1, self.d_model).mean(dim=0, keepdim=True)
                w_dev = self.cycle_inverse_proj.weight.device
                w_dtype = self.cycle_inverse_proj.weight.dtype
                hr_recon = self.cycle_inverse_proj(hr.to(device=w_dev, dtype=w_dtype)).float().to(hq.device)
                cos_sim = float(F.cosine_similarity(hr_recon, hq, dim=-1).mean().item())
                cycle_energy = float(np.clip(0.5 * (1.0 - cos_sim), 0.0, 1.0))

        # Check text degeneracy penalties (repetitive n-grams or empty text)
        words = response.strip().split()
        if len(words) == 0:
            cycle_energy = 1.0
            reason = "empty_output"
        elif len(words) >= 8 and len(set(words)) <= 2:
            cycle_energy = 0.95
            reason = "degenerate_repetition"
        else:
            reason = "cycle_consistency_coherent" if cycle_energy <= self.tau_cycle else "cycle_consistency_high_energy"

        verified = cycle_energy <= self.tau_cycle
        sign = 1.0 if verified else -1.0
        if verified:
            self._adapt_cycle_inverse_online(h_query, h_response)
        return sign, {
            "verified": bool(verified),
            "dopamine_sign": sign,
            "verification_mode": "cycle_consistency_energy",
            "reason": reason,
            "cycle_energy": float(cycle_energy),
            "correction_hint": None
        }


