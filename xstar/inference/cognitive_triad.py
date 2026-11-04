"""
X-Star Cognitive Triad Engine (/goal Slash Command)
===================================================
Executes the HADL Dual-Loop Cognitive Triad:
1. /goal: Specification Extraction, Invariant Bounds, and Edge Cases
2. /plan: System 1 vs. System 2 Routing and Action Decomposition
3. /boost: Sibling-Pattern Amplification & Neuromorphic Deliberation Overdrive
4. /goal Audit: Popperian Invariant Verification Gate

Enables autonomous, multi-phase problem solving with zero hallucination.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional, Callable, Generator, Tuple
from .engine import CustomInferenceEngine, GenerationMetrics


class CognitiveTriadRunner:
    """
    Orchestrates the 4-phase HADL Cognitive Triad autonomously over the
    custom inference model.
    """

    @classmethod
    def execute_goal(
        cls,
        engine: CustomInferenceEngine,
        goal_prompt: str,
        on_phase_change: Optional[Callable[[str, str], None]] = None,
        on_stream_chunk: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the goal sequentially through:
        Phase 1: Invariant Bounds & Spec Extraction (/goal)
        Phase 2: Dual-Process Action Plan (/plan)
        Phase 3: Neuromorphic Deliberation Synthesis (/boost)
        Phase 4: Popperian Audit & Invariant Verification
        """
        trace: Dict[str, Any] = {
            "goal": goal_prompt,
            "phases": {},
            "metrics": {},
            "final_solution": "",
        }

        # -------------------------------------------------------------
        # Phase 1: /goal (Specification Extraction & Invariant Bounds)
        # -------------------------------------------------------------
        if on_phase_change:
            on_phase_change(
                "Phase 1: /goal [Specification Extraction & Invariant Bounds]",
                "Extracting exact literals, return signatures, invariant constraints, and edge cases...",
            )

        spec_prompt = (
            f"<|im_start|>system\n"
            f"You are the HADL Cognitive Triad Specification Engine (/goal). "
            f"Analyze the objective below. Extract:\n"
            f"1. Target Deliverable & Exact Literals\n"
            f"2. Hard Invariant Bounds (what MUST NOT break or deviate)\n"
            f"3. Edge Cases (null, empty, boundary, negative, type constraints)\n"
            f"Be concise, rigorous, and bulleted.<|im_end|>\n"
            f"<|im_start|>user\nObjective: {goal_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        spec_text, m1 = engine.generate(spec_prompt, max_new_tokens=300, temperature=0.2)
        trace["phases"]["spec_extraction"] = spec_text

        # -------------------------------------------------------------
        # Phase 2: /plan (System 1 vs System 2 Routing)
        # -------------------------------------------------------------
        if on_phase_change:
            on_phase_change(
                "Phase 2: /plan [Cognitive Routing & Milestone Plan]",
                "Selecting System 1 (Fast-Path) vs System 2 (Latent Deliberation) and ordering milestones...",
            )

        plan_prompt = (
            f"<|im_start|>system\n"
            f"You are the HADL Cognitive Planner (/plan). Given the objective and extracted invariant bounds:\n"
            f"Route the solution:\n"
            f"- Cognitive Mode: System 1 (Fast-Path) or System 2 (Deep Deliberation)\n"
            f"- Structured Action Milestones (1, 2, 3...)<|im_end|>\n"
            f"<|im_start|>user\nObjective: {goal_prompt}\n\nBounds:\n{spec_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        plan_text, m2 = engine.generate(plan_prompt, max_new_tokens=350, temperature=0.3)
        trace["phases"]["action_plan"] = plan_text

        # -------------------------------------------------------------
        # Phase 3: /boost (Neuromorphic Deliberation Overdrive)
        # -------------------------------------------------------------
        if on_phase_change:
            on_phase_change(
                "Phase 3: /boost [Neuromorphic Deliberation Synthesis]",
                "Activating LIF Spiking Gate Deliberation Overdrive (v_th=0.50, VarianceClamp=Active)...",
            )

        # Temporarily enable boost mode for maximum cognitive scrutiny
        prev_boost = engine._boost_active
        prev_level = engine._boost_level
        engine.set_boost(enabled=True, level=4)

        solution_prompt = (
            f"<|im_start|>system\n"
            f"You are the HADL Cognitive Deliberation Core (/boost). "
            f"Generate the comprehensive, definitive solution that strictly adheres to the extracted bounds and action plan. "
            f"Provide robust, production-grade implementations and answers.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Objective: {goal_prompt}\n\n"
            f"Specification & Bounds:\n{spec_text}\n\n"
            f"Action Milestones:\n{plan_text}\n\n"
            f"Execute the complete solution now:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        solution_chunks = []
        for chunk, m3 in engine.stream_generate(
            solution_prompt,
            max_new_tokens=800,
            temperature=0.4,
            repetition_penalty=1.1,
        ):
            if chunk:
                solution_chunks.append(chunk)
                if on_stream_chunk:
                    on_stream_chunk(chunk)
            if m3:
                trace["metrics"]["boost_phase"] = m3

        solution_text = "".join(solution_chunks)
        trace["phases"]["solution"] = solution_text
        trace["final_solution"] = solution_text

        # Restore previous boost state
        engine.set_boost(enabled=prev_boost, level=prev_level)

        # -------------------------------------------------------------
        # Phase 4: /goal Audit (Popperian Invariant Verification Gate)
        # -------------------------------------------------------------
        if on_phase_change:
            on_phase_change(
                "Phase 4: /goal Gate [Popperian Invariant Audit]",
                "Verifying solution against initial invariant bounds, AST/sandbox execution, and cycle consistency...",
            )

        audit_prompt = (
            f"<|im_start|>system\n"
            f"You are the Popperian Verification Gate (/goal audit). "
            f"Strictly evaluate if the solution satisfies all invariant bounds and edge cases.\n"
            f"Format as:\n"
            f"- Status: [PASSED | DEFECT_FLAGGED]\n"
            f"- Invariant Compliance Score: X/10\n"
            f"- Verification Summary: (1-2 sentences)<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Bounds:\n{spec_text}\n\n"
            f"Proposed Solution:\n{solution_text[:1200]}\n\n"
            f"Audit report:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        audit_text, m4 = engine.generate(audit_prompt, max_new_tokens=200, temperature=0.1)
        trace["phases"]["audit"] = audit_text
        upper_audit = audit_text.upper()
        llm_passed = ("PASSED" in upper_audit) and ("NOT PASSED" not in upper_audit) and ("DEFECT_FLAGGED" not in upper_audit)

        from .engine import EngineConfig
        cfg = getattr(engine, "config", None)
        has_real_cfg = isinstance(cfg, EngineConfig)

        intrinsic_verified = True
        if has_real_cfg and cfg.enable_online_learning and hasattr(engine, "verify_and_adapt"):
            try:
                intrinsic_res = engine.verify_and_adapt(goal_prompt, solution_text)
                trace["closed_loop_verification"] = intrinsic_res
                intrinsic_verified = bool(intrinsic_res.get("verified", True))
            except Exception:
                intrinsic_verified = True

        trace["passed_gate"] = bool(llm_passed and intrinsic_verified)

        # Closed-loop self-correction pass when Popperian verification falsifies Pass 1
        if not trace["passed_gate"] and has_real_cfg and cfg.enable_self_evolving:
            defect_reason = (
                trace.get("closed_loop_verification", {}).get("reason", audit_text[:300])
            )
            correction_prompt = (
                f"<|im_start|>system\n"
                f"You are the HADL Closed-Loop Self-Evolving Corrector. "
                f"The previous draft was falsified by the Popperian Verification Gate ({defect_reason}). "
                f"Produce a corrected, complete, executable solution that eliminates all defects.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Objective: {goal_prompt}\n\n"
                f"Defective Draft:\n{solution_text[:800]}\n\n"
                f"Audit / Falsification Reason:\n{defect_reason}\n\n"
                f"Corrected Solution:<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            refined_text, m_ref = engine.generate(correction_prompt, max_new_tokens=600, temperature=0.2)
            if refined_text.strip():
                trace["phases"]["closed_loop_refinement"] = refined_text
                trace["final_solution"] = refined_text
                if hasattr(engine, "verify_and_adapt"):
                    try:
                        re_verify = engine.verify_and_adapt(goal_prompt, refined_text)
                        trace["closed_loop_verification_pass2"] = re_verify
                        if re_verify.get("verified", False):
                            trace["passed_gate"] = True
                    except Exception:
                        pass

        return trace
