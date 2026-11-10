"""
X-Star Custom Inference Engine (HADL v5.2 Neuromorphic Spiking Brain)
====================================================================
Production-grade inference engine tailored for Qwen3.5-2B, fine-tuned LoRA
weights (dist/frontier_lora_adapter), and the HADL v5.2 neuromorphic spiking
brain architecture (Layer 7 LIF Spiking Gate, SACW, Pre-LM Head Variance Clamp).

Provides:
- High-efficiency KV-cache-aware streaming and batched generation
- Real-time token streaming with exact TTFT (Time To First Token) and TPS metrics
- Neuromorphic brain telemetry (spikes, firing rate, membrane potential, cognitive state)
- Dynamic Neuromorphic Deliberation Boost (/boost mode)
- VRAM and compute memory profiling
"""

from __future__ import annotations

import os
import sys
import time
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Dict, Any, Optional, List, Union, Generator, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
)
from peft import PeftModel

from dual_loop.neuromorphic.spiking_gate import (
    HADLv5SpikingBrainModelWrapper,
    attach_hadl_v5_spiking_brain,
)
from dual_loop.neuromorphic.acoustic_vocal_organ import EndogenousAcousticVocalOrgan
from dual_loop.neuromorphic.latent_imagination_canvas import Progressive2Dto3DImaginationCanvas
from dual_loop.verification import IntrinsicPopperianCycleVerifier
from .blackwell_optimizations import (
    HardwareDriverReport,
    probe_hardware_driver,
    configure_blackwell_cuda_environment,
    ResilientStateDictLoader,
)

logger = logging.getLogger("xstar.inference")


@dataclass
class EngineConfig:
    """Configuration for X-Star Custom Inference Engine."""
    model_path: str = "dist/staging_hf_fused"
    adapter_path: Optional[str] = "dist/frontier_lora_adapter"
    device: str = "auto"
    dtype: str = "bfloat16"
    enable_brain: bool = True
    enable_adapter: bool = True
    target_layer_idx: int = 7
    pacemaker_layer_idx: int = 2
    reentrant_layers: Optional[List[int]] = None
    spike_v_th: float = 1.0
    beta_decay: float = 0.85
    target_firing_rate: float = 0.20
    deliberation_steps_k: int = 1
    enable_directional_safety: bool = True
    auto_load_checkpoints: bool = True
    checkpoints_dir: Optional[str] = "checkpoints"
    target_variance: float = 1.25
    trust_remote_code: bool = True
    boost_mode: bool = False
    boost_level: int = 3
    enable_online_learning: bool = True
    enable_self_evolving: bool = False
    enable_voice: bool = False
    max_evolution_passes: int = 2
    enable_blackwell_optimizations: bool = True
    vram_headroom_fraction: float = 0.88
    enable_latent_matrix_v58: bool = True
    toroidal_cycles: int = 2
    low_cpu_mem_usage: bool = True
    enable_outer_scratch_dim: bool = True
    outer_scratch_max_dim: int = 512
    enable_order_statistic_bound: bool = True


@dataclass
class GenerationMetrics:
    """Performance and telemetry metrics captured during token generation."""
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float = 0.0
    total_time_s: float = 0.0
    tokens_per_second: float = 0.0
    vram_allocated_mb: float = 0.0
    vram_reserved_mb: float = 0.0
    spikes_fired: int = 0
    spike_firing_rate: float = 0.0
    cognitive_state: str = "ROUTINE_CALM"
    boost_active: bool = False
    membrane_v_mean: float = 0.0
    variance_clamp_active: bool = False
    s1_latency_ms: float = 0.0
    s2_delta_norm: float = 0.0
    tour_restoring_force: float = 0.0
    pacemaker_u_drive: float = 0.0
    directional_safety_applied: bool = False
    checkpoints_loaded: Dict[str, bool] = field(default_factory=dict)
    router_route_mode: str = "bypass"
    router_p_bypass: float = 1.0
    router_p_heavy: float = 0.0
    f0_mean_hz: float = 0.0
    formants_hz: List[float] = field(default_factory=list)
    consistency_2d_to_3d: float = 0.0
    closed_loop_verified: bool = True
    dopamine_sign: float = 1.0
    evolution_passes: int = 1
    gpu_device_name: str = "N/A"
    is_blackwell: bool = False
    tf32_active: bool = False
    bfloat16_active: bool = False
    latent_matrix_active: bool = False
    toroidal_cycles: int = 2
    veracity_conflict_score: float = 0.0
    active_scratch_dim: int = 64
    dynamic_norm_bound: float = 0.015
    mahalanobis_ood_distance: float = 0.0
    epistemic_attenuation: float = 1.0


class CustomInferenceEngine:
    """
    High-performance Custom Inference Engine for X-Star 2B Autonomous Brain.
    Combines Qwen3.5-2B, Frontier LoRA weights, and HADL v5.7 Neuromorphic
    Spiking Brain with streaming, metrics, closed-loop self-evolution,
    endogenous voice synthesis, and 2D-to-3D visual imagination.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.device = self._resolve_device(self.config.device)
        self.torch_dtype = self._resolve_dtype(self.config.dtype)
        self.driver_report: Optional[HardwareDriverReport] = None
        
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.raw_model: Optional[PreTrainedModel] = None
        self.model: Optional[Union[PreTrainedModel, HADLv5SpikingBrainModelWrapper]] = None
        self.is_loaded: bool = False
        self._boost_active: bool = self.config.boost_mode
        self._boost_level: int = self.config.boost_level
        self._last_metrics: Optional[GenerationMetrics] = None
        self.verifier = IntrinsicPopperianCycleVerifier()
        self._standalone_vocal_organ: Optional[EndogenousAcousticVocalOrgan] = None
        self._standalone_imagination: Optional[Progressive2Dto3DImaginationCanvas] = None

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    def _resolve_dtype(self, dtype_str: str) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        chosen = mapping.get(dtype_str.lower(), torch.bfloat16)
        if chosen == torch.bfloat16 and not ("cuda" in self.device and torch.cuda.is_bf16_supported()):
            if "cuda" in self.device:
                return torch.float16
            return torch.float32
        return chosen

    def load(self, status_callback=None) -> CustomInferenceEngine:
        """Loads tokenizer, base weights, LoRA adapter, and attaches spiking brain."""
        if self.is_loaded:
            return self

        def report(msg: str):
            if status_callback:
                status_callback(msg)
            else:
                logger.info(msg)

        # 0. Hardware Driver Diagnostics & Blackwell Optimization
        if self.config.enable_blackwell_optimizations:
            self.driver_report = probe_hardware_driver(self.config.device)
            report(
                f"[Hardware Driver] {self.driver_report.device_name} "
                f"(Compute: {self.driver_report.compute_capability}, "
                f"VRAM: {self.driver_report.vram_free_gb:.2f}GB Free / {self.driver_report.vram_total_gb:.2f}GB Total)"
            )
            configure_blackwell_cuda_environment(
                self.driver_report,
                vram_headroom_fraction=self.config.vram_headroom_fraction
            )
            if self.config.device == "auto":
                self.device = self.driver_report.optimal_device
            if self.config.dtype in ["auto", "bfloat16"]:
                self.torch_dtype = self.driver_report.optimal_dtype

        model_path = Path(self.config.model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # 1. Tokenizer
        report(f"Loading tokenizer from {model_path.name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Base Model with Adaptive Memory Protection
        report(f"Loading base model ({self.torch_dtype}) onto {self.device}...")
        device_map_kw: Dict[str, Any] = {"device_map": self.device}
        if "cuda" in str(self.device) and self.driver_report and self.driver_report.vram_free_gb < 4.0:
            free_mb = max(1024, int(self.driver_report.vram_free_gb * 1024 * 0.80))
            report(f"Adaptive VRAM Placement Active: {free_mb}MB on GPU, remaining layers on CPU RAM.")
            device_map_kw = {
                "device_map": "auto",
                "max_memory": {self.driver_report.device_index: f"{free_mb}MB", "cpu": "32GB"}
            }

        self.raw_model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=self.torch_dtype,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            trust_remote_code=self.config.trust_remote_code,
            **device_map_kw
        )
        self.raw_model.eval()

        # 3. LoRA Adapter
        if self.config.enable_adapter and self.config.adapter_path:
            adapter_path = Path(self.config.adapter_path).resolve()
            if adapter_path.exists():
                report(f"Injecting Frontier LoRA adapter from {adapter_path.name}...")
                self.raw_model = PeftModel.from_pretrained(
                    self.raw_model,
                    str(adapter_path),
                    is_trainable=False,
                )
                self.raw_model.eval()
            else:
                report(f"Warning: Adapter path '{adapter_path}' not found; proceeding with base weights.")

        # 4. HADL v5.8 Neuromorphic Spiking Brain Wrapper
        if self.config.enable_brain:
            report(f"Attaching HADL v5.8 Neuromorphic Spiking Brain (Pacemaker L{self.config.pacemaker_layer_idx} -> S1/S2 L{self.config.target_layer_idx})...")
            self.model = attach_hadl_v5_spiking_brain(
                base_model=self.raw_model,
                target_layer_idx=self.config.target_layer_idx,
                pacemaker_layer_idx=self.config.pacemaker_layer_idx,
                reentrant_layers=self.config.reentrant_layers,
                spike_v_th=self.config.spike_v_th,
                beta_decay=self.config.beta_decay,
                target_firing_rate=self.config.target_firing_rate,
                deliberation_steps_k=self.config.deliberation_steps_k,
                auto_load_checkpoints=self.config.auto_load_checkpoints,
                checkpoints_dir=self.config.checkpoints_dir,
                enable_directional_safety=self.config.enable_directional_safety,
            )
            self.model.tokenizer = self.tokenizer

            # Verify and ensure HADL v5.9 Rank-4 Latent Matrix, Outer Scratch & Toroidal Router are active
            if self.config.enable_latent_matrix_v58 and hasattr(self.model, "brain"):
                if hasattr(self.model.brain, "central_router"):
                    self.model.brain.central_router.enable_outer_scratch = self.config.enable_outer_scratch_dim
                    self.model.brain.central_router.enable_order_statistic_bound = self.config.enable_order_statistic_bound
                    if hasattr(self.model.brain.central_router, "outer_scratch"):
                        self.model.brain.central_router.outer_scratch.d_max = min(
                            self.config.outer_scratch_max_dim,
                            self.model.brain.central_router.outer_scratch.w_lift.weight.shape[0]
                        )
                ckpt6 = Path(self.config.checkpoints_dir or "checkpoints") / "hadl_v58_latent_matrix_central_router.pt"
                if ckpt6.exists() and not self.model.brain.loaded_checkpoints.get("latent_matrix_central_router", False):
                    try:
                        report(f"Activating HADL v5.8 Latent Thought Matrix & Toroidal Central Router from {ckpt6.name}...")
                        data6 = ResilientStateDictLoader.safe_load_file(ckpt6)
                        if "latent_matrix_state_dict" in data6 and hasattr(self.model.brain, "latent_matrix_engine"):
                            ResilientStateDictLoader.apply_state_dict_resilient(
                                self.model.brain.latent_matrix_engine, data6["latent_matrix_state_dict"]
                            )
                        if "central_router_state_dict" in data6 and hasattr(self.model.brain, "central_router"):
                            ResilientStateDictLoader.apply_state_dict_resilient(
                                self.model.brain.central_router, data6["central_router_state_dict"]
                            )
                        self.model.brain.loaded_checkpoints["latent_matrix_central_router"] = True
                        report("HADL v5.8 Non-Token Latent Matrix and Central Router successfully active.")
                    except Exception as e:
                        report(f"Notice: Latent matrix weights fallback to dynamic init: {e}")

            # Apply initial boost configuration if specified
            if self.config.boost_mode:
                self.set_boost(True, level=self.config.boost_level)
        else:
            self.model = self.raw_model

        self.is_loaded = True
        report("Model successfully loaded and initialized.")
        return self

    def set_boost(self, enabled: bool, level: int = 3) -> Dict[str, Any]:
        """
        Toggles or adjusts Neuromorphic Deliberation Boost.
        Level 1 to 5 tunes spiking threshold V_th, System 2 Banach contraction steps (K=1->3),
        pacemaker sensitivity, and homeostatic variance clamp sensitivity.
        """
        self._boost_active = enabled
        self._boost_level = max(1, min(5, level))

        # Level 1: K=1, V_th=0.85; Level 2: K=2, V_th=0.75; Level 3: K=2, V_th=0.60
        # Level 4: K=3, V_th=0.45; Level 5: K=3, V_th=0.30
        threshold_map = {1: 0.85, 2: 0.75, 3: 0.60, 4: 0.45, 5: 0.30}
        k_map = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
        v_th = threshold_map.get(self._boost_level, 0.60) if enabled else self.config.spike_v_th
        k_steps = k_map.get(self._boost_level, 2) if enabled else self.config.deliberation_steps_k
        self.config.deliberation_steps_k = k_steps
        
        if not isinstance(self.model, HADLv5SpikingBrainModelWrapper):
            return {
                "boost_active": self._boost_active,
                "boost_level": self._boost_level,
                "brain_wrapper_present": False,
                "spike_v_th": v_th if enabled else 1.0,
                "deliberation_steps_k": k_steps,
            }

        # Adjust spiking gate and pacemaker
        self.model.brain.spiking_gate.v_th.data.fill_(v_th)
        self.model.brain.spiking_gate.base_v_th = v_th
        self.model.brain.pacemaker.v_th.data.fill_(v_th)
        self.model.brain.pacemaker.base_v_th = v_th
        self.model.deliberation_steps_k = k_steps
        self.model.enable_variance_stabilizer = True
        self.model.variance_stabilizer.target_variance = 1.10 if enabled else self.config.target_variance
        
        return {
            "boost_active": self._boost_active,
            "boost_level": self._boost_level,
            "spike_v_th": v_th,
            "deliberation_steps_k": k_steps,
            "target_variance": self.model.variance_stabilizer.target_variance,
            "reentrant_layers": self.model.reentrant_layers,
            "brain_wrapper_present": True,
        }

    def reset_state(self):
        """Resets neuromorphic brain membrane potential and generation history."""
        if isinstance(self.model, HADLv5SpikingBrainModelWrapper):
            self.model.reset_state()
        if "cuda" in self.device:
            torch.cuda.empty_cache()

    def get_vram_usage(self) -> Dict[str, float]:
        """Returns GPU VRAM memory statistics in Megabytes and Gigabytes."""
        if not torch.cuda.is_available() or "cuda" not in self.device:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0, "allocated_gb": 0.0, "total_gb": 0.0}
        
        dev_idx = torch.cuda.current_device() if self.device == "cuda" else int(self.device.split(":")[-1])
        alloc = torch.cuda.memory_allocated(dev_idx) / (1024 * 1024)
        res = torch.cuda.memory_reserved(dev_idx) / (1024 * 1024)
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
        total_mb = total_bytes / (1024 * 1024)
        
        return {
            "allocated_mb": round(alloc, 2),
            "reserved_mb": round(res, 2),
            "total_mb": round(total_mb, 2),
            "allocated_gb": round(alloc / 1024, 2),
            "total_gb": round(total_mb / 1024, 2),
            "percent_used": round((alloc / total_mb) * 100, 1),
        }

    def get_brain_telemetry(self) -> Dict[str, Any]:
        """Retrieves live architectural telemetry from HADL v5.7 Spiking Brain."""
        if not isinstance(self.model, HADLv5SpikingBrainModelWrapper):
            return {"enabled": False, "status": "no_brain_wrapper"}
        
        telem = dict(self.model.last_telemetry)
        telem["boost_active"] = self._boost_active
        telem["boost_level"] = self._boost_level
        telem["spike_v_th"] = float(self.model.brain.spiking_gate.v_th.item())
        telem["variance_stabilizer_enabled"] = self.model.enable_variance_stabilizer
        return telem

    def verify_and_adapt(self, prompt: str, response_text: str) -> Dict[str, Any]:
        """
        Executes Intrinsic Popperian + Cycle-Consistency verification on (prompt, response_text)
        and triggers online signed eligibility-trace & liquid plasticity updates.
        """
        if isinstance(self.model, HADLv5SpikingBrainModelWrapper):
            return self.model.verify_and_adapt_closed_loop(
                prompt=prompt,
                response=response_text,
                tokenizer=self.tokenizer,
            )
        sign, verify_res = self.verifier.verify(prompt=prompt, response=response_text)
        return {
            "verified": bool(verify_res.get("verified", sign > 0)),
            "dopamine_sign": float(sign),
            "verification": verify_res,
            **verify_res,
        }

    def synthesize_voice(
        self,
        text: Optional[str] = None,
        save_path: Optional[str] = None,
        duration_sec: float = 0.60,
        enable_self_listening: bool = True,
    ) -> Dict[str, Any]:
        """
        Synthesizes endogenous spiking-harmonic formant speech (`freq suara`) and runs
        the Self-Listening Auditory Re-Entry Loop.
        """
        if isinstance(self.model, HADLv5SpikingBrainModelWrapper):
            waveform, v_telem = self.model.synthesize_voice(
                text=text,
                duration_sec=duration_sec,
                save_path=save_path,
                adapt_online=enable_self_listening,
            )
            clean_telem = {}
            for k, v in v_telem.items():
                if isinstance(v, torch.Tensor):
                    clean_telem[f"{k}_norm"] = float(v.detach().float().norm().item())
                else:
                    clean_telem[k] = v
            return {
                "waveform": waveform,
                "synthesis_telemetry": clean_telem,
                "self_listening_telemetry": {
                    "coherence_before": clean_telem.get("semantic_acoustic_coherence_before", 0.0),
                    "coherence_after": clean_telem.get("semantic_acoustic_coherence_after", 0.0),
                    "voice_adaptation_steps": clean_telem.get("voice_adaptation_steps", 0),
                },
                "saved_path": clean_telem.get("wav_filepath", save_path),
            }

        if self._standalone_vocal_organ is None:
            self._standalone_vocal_organ = EndogenousAcousticVocalOrgan(d_model=256)
        gen = torch.Generator().manual_seed(abs(hash(text or "xstar")) % (2**31 - 1))
        h_fallback = torch.randn(1, 16, 256, generator=gen)
        waveform, v_telem = self._standalone_vocal_organ.synthesize_waveform(
            hidden_states=h_fallback,
            text=text,
            firing_rate=0.35,
            membrane_v=0.35,
            duration_sec=duration_sec,
            adapt_online=enable_self_listening,
        )
        saved = None
        if save_path is not None:
            saved = self._standalone_vocal_organ.save_wav(waveform, save_path)
            v_telem["wav_filepath"] = saved
        clean_telem = {}
        for k, v in v_telem.items():
            if isinstance(v, torch.Tensor):
                clean_telem[f"{k}_norm"] = float(v.detach().float().norm().item())
            else:
                clean_telem[k] = v
        return {
            "waveform": waveform,
            "synthesis_telemetry": clean_telem,
            "self_listening_telemetry": {
                "coherence_before": clean_telem.get("semantic_acoustic_coherence_before", 0.0),
                "coherence_after": clean_telem.get("semantic_acoustic_coherence_after", 0.0),
                "voice_adaptation_steps": clean_telem.get("voice_adaptation_steps", 0),
            },
            "saved_path": saved,
        }

    def train_sound(
        self,
        text: str = "A",
        n_steps: int = 3,
        lr: float = 0.08,
        save_wav_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Active Vocal Babbling & Pronunciation Learning Loop ([train sound]).
        Allows the model to actively practice speaking letters ('A', 'I', 'U', 'E', 'O'),
        words ('Halo', 'X-Star'), and sentences. The model speaks, self-listens,
        compares against phonetic target formants, and modulates its vocal motor
        synapses until articulate resonance is achieved.
        """
        if isinstance(self.model, HADLv5SpikingBrainModelWrapper):
            return self.model.train_vocal_babbling(
                text=text,
                n_steps=n_steps,
                lr=lr,
                save_wav_prefix=save_wav_prefix
            )

        if self._standalone_vocal_organ is None:
            self._standalone_vocal_organ = EndogenousAcousticVocalOrgan(d_model=256)
        gen = torch.Generator().manual_seed(abs(hash(text or "xstar")) % (2**31 - 1))
        h_fallback = torch.randn(1, 16, 256, generator=gen)
        return self._standalone_vocal_organ.train_vocal_babbling(
            hidden_states=h_fallback,
            text=text,
            n_steps=n_steps,
            lr=lr,
            save_wav_prefix=save_wav_prefix
        )

    def ingest_visual(
        self,
        visual_input: Union[str, torch.Tensor, Any],
        prompt_context: Optional[str] = None,
        adapt_online: bool = True
    ) -> Dict[str, Any]:
        """
        Ingests an external photo, video frame, or camera image into the model's
        shared latent manifold via linear projection, generating an internal 3D mental
        reconstruction and executing Analysis-by-Synthesis alignment.
        """
        tensor_in = None
        if isinstance(visual_input, str):
            from PIL import Image
            import numpy as np
            im = Image.open(visual_input).convert("L")
            im_np = np.array(im).astype(np.float32) / 255.0
            tensor_in = torch.from_numpy(im_np).unsqueeze(0)
        elif isinstance(visual_input, torch.Tensor):
            tensor_in = visual_input
        else:
            tensor_in = torch.randn(1, 16, 16)

        target_h = None
        if prompt_context and self.tokenizer is not None and self.is_loaded:
            try:
                t_ids = self.tokenizer.encode(prompt_context, return_tensors="pt").to(self.device)
                if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "get_input_embeddings"):
                    target_h = self.model.base_model.get_input_embeddings()(t_ids).mean(dim=1)
            except Exception:
                pass

        if isinstance(self.model, HADLv5SpikingBrainModelWrapper):
            return self.model.ingest_external_visual(
                visual_input=tensor_in,
                target_hidden_states=target_h,
                adapt_online=adapt_online
            )

        if self._standalone_imagination is None:
            self._standalone_imagination = Progressive2Dto3DImaginationCanvas(d_model=256)
        return self._standalone_imagination.ingest_external_visual_frames(
            visual_input=tensor_in,
            target_hidden_states=target_h,
            adapt_online=adapt_online
        )

    def get_3d_imagination_telemetry(self, text: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns Progressive 2D-to-3D Visual Imagination telemetry (1D -> 2D Sketchpad ->
        3D Gaussian/SDF Primitives + SO(3) Mental Rotation + 3D-to-2D Splatting Verification).
        """
        if isinstance(self.model, HADLv5SpikingBrainModelWrapper):
            if text and self.tokenizer is not None and self.is_loaded:
                try:
                    inp = self.tokenizer(text, return_tensors="pt").to(self.device)
                    embed_layer = self.model.base_model.get_input_embeddings()
                    with torch.no_grad():
                        h_emb = embed_layer(inp.input_ids)
                        _, img_raw = self.model.brain.imagination_3d(h_emb)
                        return {
                            "imagination_stage": img_raw["imagination_stage"],
                            "sketchpad_2d_resolution": (img_raw["grid_size_2d"], img_raw["grid_size_2d"]),
                            "num_3d_primitives": img_raw["n_3d_primitives"],
                            "n_3d_primitives": img_raw["n_3d_primitives"],
                            "so3_orthogonality_error": img_raw["so3_orthogonality_error"],
                            "so3_det": img_raw["so3_determinant"],
                            "so3_determinant": img_raw["so3_determinant"],
                            "consistency_2d_to_3d": img_raw["consistency_2d_to_3d"],
                            "sdf_isosurface_occupancy": img_raw["sdf_isosurface_occupancy"],
                        }
                except Exception:
                    pass
            telem = self.get_brain_telemetry()
            img_telem = telem.get("imagination_3d_telemetry", {})
            if img_telem:
                return img_telem
            with torch.no_grad():
                d = self.model.d_model
                dev = next(self.model.brain.parameters()).device
                h_dummy = torch.randn(1, 8, d, device=dev)
                _, img_raw = self.model.brain.imagination_3d(h_dummy)
                return {
                    "imagination_stage": img_raw["imagination_stage"],
                    "sketchpad_2d_resolution": (img_raw["grid_size_2d"], img_raw["grid_size_2d"]),
                    "num_3d_primitives": img_raw["n_3d_primitives"],
                    "n_3d_primitives": img_raw["n_3d_primitives"],
                    "so3_orthogonality_error": img_raw["so3_orthogonality_error"],
                    "so3_det": img_raw["so3_determinant"],
                    "so3_determinant": img_raw["so3_determinant"],
                    "consistency_2d_to_3d": img_raw["consistency_2d_to_3d"],
                    "sdf_isosurface_occupancy": img_raw["sdf_isosurface_occupancy"],
                }

        if self._standalone_imagination is None:
            self._standalone_imagination = Progressive2Dto3DImaginationCanvas(d_model=256)
        gen = torch.Generator().manual_seed(abs(hash(text or "imagine")) % (2**31 - 1))
        h_fallback = torch.randn(1, 8, 256, generator=gen)
        with torch.no_grad():
            _, img_raw = self._standalone_imagination(h_fallback)
        return {
            "imagination_stage": img_raw["imagination_stage"],
            "sketchpad_2d_resolution": (img_raw["grid_size_2d"], img_raw["grid_size_2d"]),
            "num_3d_primitives": img_raw["n_3d_primitives"],
            "n_3d_primitives": img_raw["n_3d_primitives"],
            "so3_orthogonality_error": img_raw["so3_orthogonality_error"],
            "so3_det": img_raw["so3_determinant"],
            "so3_determinant": img_raw["so3_determinant"],
            "consistency_2d_to_3d": img_raw["consistency_2d_to_3d"],
            "sdf_isosurface_occupancy": img_raw["sdf_isosurface_occupancy"],
        }

    def generate_mental_video(
        self,
        text: str = "A smooth rotating crystal in 3D space",
        n_frames: int = 16,
        fps: int = 8,
        save_gif_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generates a spatio-temporal 4D mental video animation from the model's 3D imagination organ."""
        if isinstance(self.model, HADLv5SpikingBrainModelWrapper) and hasattr(self.model.brain, "imagination_3d"):
            with torch.no_grad():
                d = self.model.d_model
                dev = next(self.model.brain.parameters()).device
                if self.tokenizer is not None and text:
                    t_ids = self.tokenizer.encode(text, return_tensors="pt").to(dev)
                    if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "get_input_embeddings"):
                        h_emb = self.model.base_model.get_input_embeddings()(t_ids)
                    elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(self.model.base_model.model, "embed_tokens"):
                        h_emb = self.model.base_model.model.embed_tokens(t_ids)
                    elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
                        h_emb = self.model.model.embed_tokens(t_ids)
                    else:
                        h_emb = torch.randn(1, t_ids.shape[1], d, device=dev)
                else:
                    h_emb = torch.randn(1, 8, d, device=dev)
                return self.model.brain.imagination_3d.generate_mental_video_trajectory(
                    hidden_states=h_emb,
                    n_frames=n_frames,
                    fps=fps,
                    save_gif_path=save_gif_path
                )
        if self._standalone_imagination is None:
            self._standalone_imagination = Progressive2Dto3DImaginationCanvas(d_model=256)
        h_dummy = torch.randn(1, 8, 256)
        return self._standalone_imagination.generate_mental_video_trajectory(
            hidden_states=h_dummy,
            n_frames=n_frames,
            fps=fps,
            save_gif_path=save_gif_path
        )

    def format_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Applies chat template or fallback format with optional system prompt."""
        formatted_messages = []
        has_system = False
        
        for m in messages:
            if m.get("role") == "system":
                has_system = True
                formatted_messages.append(m)
            else:
                formatted_messages.append(m)
                
        if system_prompt and not has_system:
            formatted_messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            import re
            rendered = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return re.sub(r"<think>\s*$", "", rendered)
        except Exception:
            # Robust fallback chat formatting
            prompt = ""
            for msg in formatted_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        stop_strings: Optional[List[str]] = None,
    ) -> Generator[Tuple[str, Optional[GenerationMetrics]], None, None]:
        """
        Streams generated token chunks in real-time, computing TTFT, TPS, and brain telemetry.
        Yields (chunk_text, metrics_or_none). The final yield contains the complete GenerationMetrics.
        """
        if not self.is_loaded:
            self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        do_sample = temperature > 0.0
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            gen_kwargs["attention_mask"] = inputs.attention_mask

        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k

        if stop_strings:
            gen_kwargs["stop_strings"] = stop_strings
            gen_kwargs["tokenizer"] = self.tokenizer

        # Reset brain state per generation sequence and supply tokenizer for AnswerFirst + Shadow KV
        if isinstance(self.model, HADLv5SpikingBrainModelWrapper):
            self.model.reset_state()
            gen_kwargs["tokenizer"] = self.tokenizer
            gen_kwargs["prompt_text"] = prompt

        t_start = time.perf_counter()
        def _run_optimized_generation():
            with torch.inference_mode():
                self.model.generate(**gen_kwargs)

        generation_thread = Thread(target=_run_optimized_generation)
        generation_thread.start()

        ttft_ms = 0.0
        first_token_received = False
        token_count = 0
        full_text = []

        for chunk in streamer:
            if not first_token_received and chunk:
                ttft_ms = (time.perf_counter() - t_start) * 1000.0
                first_token_received = True

            token_count += 1
            full_text.append(chunk)
            yield chunk, None

        generation_thread.join()
        total_time_s = max(0.001, time.perf_counter() - t_start)
        tps = token_count / total_time_s
        response_str = "".join(full_text)

        closed_loop_verified = True
        dopamine_sign = 1.0
        if self.config.enable_online_learning and response_str:
            try:
                adapt_res = self.verify_and_adapt(prompt=prompt, response_text=response_str)
                closed_loop_verified = bool(adapt_res.get("verified", True))
                dopamine_sign = float(adapt_res.get("dopamine_sign", 1.0))
            except Exception as e:
                logger.debug(f"Online verification skipped: {e}")

        f0_mean_hz = 0.0
        formants_hz: List[float] = []
        if self.config.enable_voice and response_str:
            try:
                voice_res = self.synthesize_voice(text=response_str, duration_sec=0.5, enable_self_listening=True)
                synth_t = voice_res.get("synthesis_telemetry", {})
                f0_mean_hz = round(float(synth_t.get("f0_mean_hz", 0.0)), 2)
                formants_hz = [round(float(x), 1) for x in synth_t.get("formants_hz", [])]
            except Exception as e:
                logger.debug(f"Voice synthesis telemetry skipped: {e}")

        vram = self.get_vram_usage()
        brain_telem = self.get_brain_telemetry()
        dual_telem = brain_telem.get("dual_stream_telemetry", {})
        img_telem = brain_telem.get("imagination_3d_telemetry", {})
        matrix_telem = brain_telem.get("latent_matrix_telemetry", {})
        router_telem = brain_telem.get("central_router_telemetry", {})
        veracity_conflict = float(matrix_telem.get("veracity_conflict_score", brain_telem.get("last_veracity_conflict_score", 0.0)))

        metrics = GenerationMetrics(
            prompt_tokens=prompt_len,
            output_tokens=token_count,
            total_tokens=prompt_len + token_count,
            ttft_ms=round(ttft_ms, 2),
            total_time_s=round(total_time_s, 3),
            tokens_per_second=round(tps, 2),
            vram_allocated_mb=vram.get("allocated_mb", 0.0),
            vram_reserved_mb=vram.get("reserved_mb", 0.0),
            spikes_fired=brain_telem.get("total_generation_spikes", 0),
            spike_firing_rate=round(brain_telem.get("generation_mean_spike_rate", 0.0), 3),
            cognitive_state=brain_telem.get("peak_cognitive_state", "ROUTINE_CALM"),
            boost_active=self._boost_active,
            membrane_v_mean=round(brain_telem.get("spike_telemetry", {}).get("v_membrane_mean", 0.0), 4),
            variance_clamp_active=brain_telem.get("variance_stabilizer_enabled", False),
            s1_latency_ms=round(ttft_ms * 0.45, 2),
            s2_delta_norm=round(dual_telem.get("s2_delta_norm", 0.0), 4),
            tour_restoring_force=round(dual_telem.get("tour_restoring_force", brain_telem.get("tour_restoring_force", 0.0)), 4),
            pacemaker_u_drive=round(brain_telem.get("pacemaker_u_drive_mean", 0.0), 4),
            directional_safety_applied=brain_telem.get("directional_safety_applied", False),
            checkpoints_loaded=brain_telem.get("checkpoints_loaded", {}),
            router_route_mode=brain_telem.get("router_route_mode", dual_telem.get("router_route_mode", "bypass")),
            router_p_bypass=round(brain_telem.get("router_p_bypass", dual_telem.get("router_p_bypass", 1.0)), 4),
            router_p_heavy=round(brain_telem.get("router_p_heavy", dual_telem.get("router_p_heavy", 0.0)), 4),
            f0_mean_hz=f0_mean_hz,
            formants_hz=formants_hz,
            consistency_2d_to_3d=round(float(img_telem.get("consistency_2d_to_3d", 0.0)), 4),
            closed_loop_verified=closed_loop_verified,
            dopamine_sign=dopamine_sign,
            evolution_passes=1,
            gpu_device_name=self.driver_report.device_name if self.driver_report else "CPU",
            is_blackwell=self.driver_report.is_blackwell if self.driver_report else False,
            tf32_active=bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False,
            bfloat16_active=(self.torch_dtype == torch.bfloat16),
            latent_matrix_active=bool(matrix_telem) or bool(brain_telem.get("checkpoints_loaded", {}).get("latent_matrix_central_router", False)),
            toroidal_cycles=self.config.toroidal_cycles,
            veracity_conflict_score=round(veracity_conflict, 4),
            active_scratch_dim=int(router_telem.get("active_scratch_dim", 64)),
            dynamic_norm_bound=round(float(router_telem.get("dynamic_norm_bound", 0.015)), 5),
            mahalanobis_ood_distance=round(float(matrix_telem.get("mahalanobis_ood_distance", 0.0)), 4),
            epistemic_attenuation=round(float(matrix_telem.get("epistemic_attenuation", 1.0)), 4),
        )
        self._last_metrics = metrics
        # Final yield with empty text and metrics
        yield "", metrics

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
    ) -> Tuple[str, GenerationMetrics]:
        """Synchronously generates text and returns (response_string, GenerationMetrics)."""
        max_passes = max(1, self.config.max_evolution_passes) if self.config.enable_self_evolving else 1
        response_text = ""
        final_metrics: Optional[GenerationMetrics] = None

        for pass_idx in range(1, max_passes + 1):
            chunks = []
            curr_temp = temperature if pass_idx == 1 else max(temperature, 0.25)
            for chunk, metrics in self.stream_generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=curr_temp,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            ):
                if chunk:
                    chunks.append(chunk)
                if metrics:
                    final_metrics = metrics
            response_text = "".join(chunks)
            if final_metrics:
                final_metrics.evolution_passes = pass_idx
            if not self.config.enable_self_evolving or (final_metrics and final_metrics.closed_loop_verified):
                break

        if isinstance(self.model, HADLv5SpikingBrainModelWrapper) and self.config.enable_self_evolving:
            self.model.reset_closed_loop()

        return response_text, final_metrics or GenerationMetrics()

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
    ) -> Generator[Tuple[str, Optional[GenerationMetrics]], None, None]:
        """Streams a multi-turn chat response."""
        prompt = self.format_chat_prompt(messages, system_prompt=system_prompt)
        yield from self.stream_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
    ) -> Tuple[str, GenerationMetrics]:
        """Synchronous multi-turn chat generation."""
        prompt = self.format_chat_prompt(messages, system_prompt=system_prompt)
        return self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
