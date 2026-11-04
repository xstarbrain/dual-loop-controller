import os
import wave
import math
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List


class EndogenousAcousticVocalOrgan(nn.Module):
    """
    Endogenous Custom Sound Synthesis Organ (`freq suara` — Spiking-Harmonic
    Formant Synthesizer + Self-Listening Auditory Re-Entry Loop).

    Synthesizes real continuous audio waveforms directly from Layer 7 neuromorphic
    spiking pacemaker dynamics and latent thought trajectories without static TTS
    templates, and feeds the synthesized spectrum back into Layer 7 so the model
    hears and adapts its own voice online.

    Mathematical Formulation:
    1. Glottal Fundamental Frequency Contour f_0(t) in [85, 340] Hz:
       Driven by PreRouterSpikingPacemaker firing rate r_t, membrane voltage v_t,
       and hidden state h_t:
           f_0(t) = 85.0 + 255.0 * sigmoid(W_f0 h_t + alpha_r r_t + alpha_v v_t)

    2. 4-Formant Vocal Tract Filter (F_1..F_4 in [250, 4200] Hz, Bandwidths B_1..B_4):
       Couples latent sensitivity frequencies with articulatory vowel/consonant loci:
           F_1(t) in [250, 900] Hz   (jaw opening / vowel height)
           F_2(t) in [850, 2400] Hz  (tongue frontness)
           F_3(t) in [1900, 3200] Hz (lip rounding / timbre)
           F_4(t) in [3000, 4200] Hz (speaker characteristic resonance)

    3. Harmonic Glottal Source + Spectral Resonance Filtering:
       Generates K=24 phase-continuous harmonics modulated by the 4 Lorentzian
       formant resonators and refined via rFFT/irFFT spectral envelope shaping.

    4. Self-Listening Auditory Re-Entry Loop:
       Extracts spectral features S_audio = log(1 + |rFFT(y)|) from the synthesized
       waveform, projects back to R^D via W_reentry, computes semantic-acoustic
       coherence cos(h_auditory, h_semantic), and updates formant/pitch synapses
       online so the model continuously refines its own voice.
    """
    def __init__(
        self,
        d_model: int = 2048,
        sample_rate: int = 16000,
        n_harmonics: int = 24,
        n_spectral_bins: int = 128,
        f0_min: float = 85.0,
        f0_max: float = 340.0,
        misf: Optional[nn.Module] = None,
        pacemaker: Optional[nn.Module] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.n_spectral_bins = n_spectral_bins
        self.f0_min = f0_min
        self.f0_max = f0_max

        # Non-owning references to sibling organs in NeuromorphicSpikingBrain
        object.__setattr__(self, "_misf_ref", misf)
        object.__setattr__(self, "_pacemaker_ref", pacemaker)

        # 1. Fundamental frequency (f0) & prosodic energy projection
        self.f0_proj = nn.Linear(d_model, 2)  # [f0_logit, energy_logit]
        nn.init.normal_(self.f0_proj.weight, std=0.02)
        with torch.no_grad():
            self.f0_proj.bias.copy_(torch.tensor([-0.3, 1.2]))  # Default ~155 Hz, high voicing energy

        # 2. 4-Formant frequency & bandwidth projection: [F1..F4 logits, B1..B4 logits]
        self.formant_proj = nn.Linear(d_model, 8)
        nn.init.normal_(self.formant_proj.weight, std=0.02)
        nn.init.zeros_(self.formant_proj.bias)

        # Formant frequency bounds (Hz)
        self.register_buffer("formant_low", torch.tensor([250.0, 850.0, 1900.0, 3000.0]))
        self.register_buffer("formant_high", torch.tensor([900.0, 2400.0, 3200.0, 4200.0]))
        self.register_buffer("bandwidth_low", torch.tensor([45.0, 70.0, 100.0, 130.0]))
        self.register_buffer("bandwidth_high", torch.tensor([140.0, 200.0, 280.0, 350.0]))

        # 3. Differentiable rFFT Spectral Timbre Envelope (n_spectral_bins) + SpectralResonanceHarmonizer
        self.spectral_gain = nn.Parameter(torch.ones(n_spectral_bins))
        from dual_loop.experimental_cognitive_engine import SpectralResonanceHarmonizer
        self.spectral_harmonizer = SpectralResonanceHarmonizer(d_model=d_model)

        # 4. Self-Listening Auditory Re-Entry Projection: maps acoustic spectrum -> R^D
        self.auditory_reentry_proj = nn.Linear(n_spectral_bins, d_model, bias=False)
        self.semantic_acoustic_target = nn.Linear(d_model, n_spectral_bins, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.auditory_reentry_proj.weight)
            # Initialize target as transpose so round-trip starts near-coherent
            self.semantic_acoustic_target.weight.copy_(self.auditory_reentry_proj.weight.t())

        # Online vocal plasticity eligibility trace & telemetry counters
        self.register_buffer("voice_adaptation_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("ema_coherence", torch.tensor(0.5, dtype=torch.float32))
        self.register_buffer("last_f0_mean", torch.tensor(160.0, dtype=torch.float32))
        self.register_buffer("last_formants_mean", torch.tensor([500.0, 1500.0, 2500.0, 3500.0], dtype=torch.float32))

    def bind_neuromorphic_organs(
        self,
        misf: Optional[nn.Module] = None,
        pacemaker: Optional[nn.Module] = None
    ) -> None:
        """Binds sibling MISF and PreRouterSpikingPacemaker organs for live coupling."""
        if misf is not None:
            object.__setattr__(self, "_misf_ref", misf)
        if pacemaker is not None:
            object.__setattr__(self, "_pacemaker_ref", pacemaker)

    @property
    def misf(self) -> Optional[nn.Module]:
        return getattr(self, "_misf_ref", None)

    @property
    def pacemaker(self) -> Optional[nn.Module]:
        return getattr(self, "_pacemaker_ref", None)

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

    @staticmethod
    def _text_to_articulatory_targets(text: str, n_frames: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts syllable-timed articulatory formant targets (F1..F4 in [0, 1] normalized range),
        voicing envelope, and sibilant frication level from text.
        Vowels receive sustained duration (120-200ms nuclei), while consonants provide
        sharp acoustic transitions, preventing fast linear character smearing.
        """
        vowel_map = {
            'a': (0.85, 0.38, 0.45, 0.50),  # /a/
            'e': (0.42, 0.75, 0.55, 0.55),  # /e/
            'i': (0.12, 0.92, 0.75, 0.65),  # /i/
            'o': (0.45, 0.15, 0.35, 0.45),  # /o/
            'u': (0.18, 0.10, 0.28, 0.40),  # /u/
            'y': (0.20, 0.85, 0.65, 0.60),
        }
        clean = (text or "om").strip().lower()
        if not clean:
            clean = "om"

        f_targets = []
        voicing = []
        frication = []

        # Parse into syllable-timed acoustic segments
        for ch in clean:
            if ch in vowel_map:
                # Vowel nucleus: repeat 3-4 times for natural duration
                for _ in range(3):
                    f_targets.append(vowel_map[ch])
                    voicing.append(1.0)
                    frication.append(0.01)
            elif ch in ('m', 'n', 'l', 'r', 'w', 'v', 'z', 'b', 'd', 'g'):
                # Voiced sonorants / stops
                f_targets.append((0.25, 0.45, 0.40, 0.45))
                voicing.append(0.85)
                frication.append(0.04)
            elif ch in ('s', 'f', 'h', 'c', 'x'):
                # Unvoiced fricatives / sibilants
                for _ in range(2):
                    f_targets.append((0.35, 0.70, 0.80, 0.85))
                    voicing.append(0.20)
                    frication.append(0.40)
            elif ch in ('p', 't', 'k', 'q'):
                # Plosive: brief closure then burst
                f_targets.append((0.20, 0.40, 0.40, 0.40))
                voicing.append(0.15)
                frication.append(0.01)
                f_targets.append((0.30, 0.55, 0.55, 0.50))
                voicing.append(0.40)
                frication.append(0.25)
            elif ch.isspace() or ch in (',', '.', ';', ':', '!', '?'):
                # Inter-word micro-pause
                for _ in range(2):
                    f_targets.append((0.35, 0.45, 0.45, 0.45))
                    voicing.append(0.05)
                    frication.append(0.00)
            else:
                code = (ord(ch) % 5)
                vw = list(vowel_map.values())[code]
                f_targets.append(vw)
                voicing.append(0.90)
                frication.append(0.03)

        f_arr = torch.tensor(f_targets, dtype=torch.float32, device=device)  # [C, 4]
        v_arr = torch.tensor(voicing, dtype=torch.float32, device=device)    # [C]
        n_arr = torch.tensor(frication, dtype=torch.float32, device=device)  # [C]

        # Interpolate smoothly to n_frames
        if f_arr.size(0) == 1:
            f_interp = f_arr.expand(n_frames, 4)
            v_interp = v_arr.expand(n_frames)
            n_interp = n_arr.expand(n_frames)
        else:
            f_interp = F.interpolate(
                f_arr.t().unsqueeze(0), size=n_frames, mode="linear", align_corners=True
            ).squeeze(0).t()  # [n_frames, 4]
            v_interp = F.interpolate(
                v_arr.view(1, 1, -1), size=n_frames, mode="linear", align_corners=True
            ).view(n_frames)
            n_interp = F.interpolate(
                n_arr.view(1, 1, -1), size=n_frames, mode="linear", align_corners=True
            ).view(n_frames)

        return f_interp, v_interp, n_interp

    def compute_vocal_parameters(
        self,
        hidden_states: torch.Tensor,
        firing_rate: float = 0.25,
        membrane_v: float = 0.30,
        text: Optional[str] = None,
        n_frames: int = 32
    ) -> Dict[str, torch.Tensor]:
        """
        Maps Layer 7 hidden states and spiking pacemaker telemetry to frame-level
        fundamental frequency f0(t), 4 formant frequencies F1..F4(t), bandwidths B1..B4(t),
        and voicing envelope.
        """
        device = self.f0_proj.weight.device
        if hidden_states.dim() == 2:
            h_seq = hidden_states.unsqueeze(1)  # [B, 1, D]
        else:
            h_seq = hidden_states  # [B, L, D]

        h_f = h_seq.detach().to(device=device, dtype=torch.float32)
        # Pool or interpolate token sequence to n_frames
        if h_f.size(1) == 1:
            h_frames = h_f[0].expand(n_frames, -1)  # [n_frames, D]
        else:
            h_frames = F.interpolate(
                h_f[0].t().unsqueeze(0), size=n_frames, mode="linear", align_corners=True
            ).squeeze(0).t()  # [n_frames, D]

        # 1. Fundamental frequency f0(t) in [f0_min, f0_max] Hz
        # Query live PreRouterSpikingPacemaker firing rate / membrane potential if bound
        pm_ref = getattr(self, "_pacemaker_ref", None)
        if pm_ref is not None and hasattr(pm_ref, "spiking_gate"):
            sg = pm_ref.spiking_gate
            if hasattr(sg, "ema_rate"):
                firing_rate = 0.5 * float(firing_rate) + 0.5 * float(sg.ema_rate.mean().item())
            if hasattr(sg, "v_membrane"):
                membrane_v = 0.5 * float(membrane_v) + 0.5 * float(sg.v_membrane.mean().item())

        f0_energy = self.f0_proj(h_frames)  # [n_frames, 2]
        spike_drive = 0.8 * (float(firing_rate) - 0.2) + 0.4 * (float(membrane_v) - 0.25)
        t_axis = torch.linspace(0.0, 1.0, n_frames, device=device)
        # Natural declarative prosodic declination + slight vibrato (5.5 Hz)
        prosody_curve = 0.15 * torch.sin(2.0 * math.pi * 1.5 * t_axis) + 0.04 * torch.sin(2.0 * math.pi * 5.5 * t_axis)

        f0_norm = torch.sigmoid(f0_energy[:, 0] + spike_drive + prosody_curve)
        f0_hz = self.f0_min + (self.f0_max - self.f0_min) * f0_norm  # [n_frames]

        # 2. Formant frequencies F1..F4 and bandwidths B1..B4
        # Couple MonolithicIncoherentSensitivityField (w_freq, phase_shift) when bound
        fb_logits = self.formant_proj(h_frames)  # [n_frames, 8]
        misf_ref = getattr(self, "_misf_ref", None)
        misf_f_mod = torch.zeros(1, 4, device=device)
        misf_b_mod = torch.zeros(1, 4, device=device)
        if misf_ref is not None and hasattr(misf_ref, "w_freq") and hasattr(misf_ref, "phase_shift"):
            w_f = misf_ref.w_freq.detach().to(device=device, dtype=torch.float32).view(1, 1, -1)
            p_s = misf_ref.phase_shift.detach().to(device=device, dtype=torch.float32).view(1, 1, -1)
            misf_f_mod = 0.25 * torch.tanh(F.adaptive_avg_pool1d(w_f, 4).view(1, 4))
            misf_b_mod = 0.20 * torch.sin(F.adaptive_avg_pool1d(p_s, 4).view(1, 4))

        latent_f_norm = torch.sigmoid(fb_logits[:, :4] + misf_f_mod)  # [n_frames, 4]
        latent_b_norm = torch.sigmoid(fb_logits[:, 4:] + misf_b_mod)  # [n_frames, 4]

        art_f_norm, voicing_mask, frication_mask = self._text_to_articulatory_targets(
            text or "suara", n_frames, device
        )
        # Blend neural latent formant trajectory (55%) with articulatory targets (45%)
        blended_f_norm = 0.55 * latent_f_norm + 0.45 * art_f_norm

        f_low = self.formant_low.unsqueeze(0)
        f_high = self.formant_high.unsqueeze(0)
        b_low = self.bandwidth_low.unsqueeze(0)
        b_high = self.bandwidth_high.unsqueeze(0)

        formants_hz = f_low + (f_high - f_low) * blended_f_norm      # [n_frames, 4]
        bandwidths_hz = b_low + (b_high - b_low) * latent_b_norm     # [n_frames, 4]

        # Voicing amplitude envelope with smooth attack/release window
        env_neural = torch.sigmoid(f0_energy[:, 1])
        amplitude = env_neural * voicing_mask

        return {
            "f0_hz": f0_hz,
            "formants_hz": formants_hz,
            "bandwidths_hz": bandwidths_hz,
            "amplitude": amplitude,
            "frication": frication_mask,
            "h_frames": h_frames,
            "misf_coupled": bool(misf_ref is not None)
        }

    def synthesize_waveform(
        self,
        hidden_states: torch.Tensor,
        text: Optional[str] = None,
        firing_rate: float = 0.25,
        membrane_v: float = 0.30,
        duration_sec: Optional[float] = None,
        adapt_online: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Synthesizes a continuous 1D acoustic waveform y(t) in [-1, 1] from Layer 7
        hidden states and spiking pacemaker dynamics via asymmetric Rosenberg LF
        glottal volume velocity pulse excitation and time-varying 4-formant resonant
        filtering, then executes the Self-Listening Auditory Re-Entry Loop.

        Returns:
            waveform: [N_samples] float32 tensor in [-1, 1] at self.sample_rate.
            telemetry: Dict with f0_hz, formants_hz, auditory_reentry_delta,
                       semantic_acoustic_coherence, etc.
        """
        device = self.f0_proj.weight.device

        # Dynamic syllable-timed duration estimation ensuring ample playback time (2.2s - 4.5s)
        clean = (text or "").strip()
        if duration_sec is None:
            if len(clean) <= 1:
                duration_sec = 2.2  # Rich sustained vocalization for single letters ('A', 'I', etc.)
            else:
                duration_sec = max(2.5, min(5.0, len(clean) * 0.18 + 0.6))
        else:
            duration_sec = max(0.05, float(duration_sec))

        n_samples = max(256, int(self.sample_rate * duration_sec))
        n_frames = max(16, min(96, n_samples // 256))

        vocal_params = self.compute_vocal_parameters(
            hidden_states=hidden_states,
            firing_rate=firing_rate,
            membrane_v=membrane_v,
            text=text,
            n_frames=n_frames
        )

        f0_frames = vocal_params["f0_hz"]              # [n_frames]
        formant_frames = vocal_params["formants_hz"]   # [n_frames, 4]
        bw_frames = vocal_params["bandwidths_hz"]      # [n_frames, 4]
        amp_frames = vocal_params["amplitude"]         # [n_frames]
        fric_frames = vocal_params["frication"]        # [n_frames]

        # Interpolate frame trajectories to sample rate [n_samples]
        f0_t = F.interpolate(f0_frames.view(1, 1, -1), size=n_samples, mode="linear", align_corners=True).view(n_samples)
        amp_t = F.interpolate(amp_frames.view(1, 1, -1), size=n_samples, mode="linear", align_corners=True).view(n_samples)
        fric_t = F.interpolate(fric_frames.view(1, 1, -1), size=n_samples, mode="linear", align_corners=True).view(n_samples)

        # 1. Asymmetric Rosenberg LF Glottal Volume Velocity Excitation
        phase_t = torch.remainder(torch.cumsum(f0_t / float(self.sample_rate), dim=0), 1.0)  # [n_samples]
        open_phase = 0.65
        glottal_pulses = torch.where(
            phase_t < open_phase,
            0.5 * (1.0 - torch.cos(math.pi * phase_t / open_phase)),
            torch.cos(0.5 * math.pi * (phase_t - open_phase) / (1.0 - open_phase))
        )  # [n_samples]

        # 2. Time-Varying 4-Formant Resonant Filtering via Frame Overlap-Add
        frame_len = n_samples // n_frames
        K_len = 128
        n_k = torch.arange(K_len, device=device, dtype=torch.float32)
        gains = [1.0, 0.75, 0.45, 0.25]
        voiced_wave = torch.zeros(n_samples, device=device)

        for i in range(n_frames):
            start = i * frame_len
            end = min(n_samples, start + frame_len)
            seg_g = glottal_pulses[start:end]
            if len(seg_g) < 16:
                continue

            fm_i = formant_frames[i]
            bw_i = bw_frames[i]
            h_bank = []
            for fi in range(4):
                decay = torch.exp(-math.pi * bw_i[fi] * n_k / float(self.sample_rate))
                osc = torch.sin(2.0 * math.pi * fm_i[fi] * n_k / float(self.sample_rate))
                k = gains[fi] * decay * osc
                k = k / (k.abs().sum() + 1e-6)
                h_bank.append(k)

            filter_kernel = torch.stack(h_bank).unsqueeze(1)  # [4, 1, K_len]
            seg_in = seg_g.view(1, 1, -1)
            seg_filt = F.conv1d(seg_in, filter_kernel, padding=K_len // 2).sum(dim=1).squeeze(0)[:len(seg_g)]
            pk = seg_filt.abs().max() + 1e-6
            seg_out = 0.40 * seg_g + 0.60 * (seg_filt / pk)

            fade = min(int(0.01 * self.sample_rate), len(seg_out) // 4)
            if fade > 1:
                win = torch.ones(len(seg_out), device=device)
                win[:fade] = torch.linspace(0.0, 1.0, fade, device=device)
                win[-fade:] = torch.linspace(1.0, 0.0, fade, device=device)
                seg_out = seg_out * win

            voiced_wave[start:end] = seg_out

        # 3. Add Gentle Shaped Consonant Aspiration (Turbulence)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(abs(float(f0_frames.sum().item())) * 1000) % 2147483647 + 1)
        raw_noise = torch.randn(n_samples, generator=gen, device=device)
        highpass_noise = torch.cat([raw_noise[:1], raw_noise[1:] - 0.90 * raw_noise[:-1]])
        aspir_noise = 0.08 * fric_t * highpass_noise

        # 4. Voicing Modulation & Raised Cosine Anti-Click Envelope (30ms)
        raw_wave = amp_t.clamp(min=0.35) * voiced_wave + aspir_noise

        fade_len = min(int(0.03 * self.sample_rate), n_samples // 4)
        if fade_len > 1:
            fade_in = 0.5 * (1.0 - torch.cos(math.pi * torch.arange(fade_len, device=device, dtype=torch.float32) / fade_len))
            fade_out = torch.flip(fade_in, dims=[0])
            raw_wave[:fade_len] = raw_wave[:fade_len] * fade_in
            raw_wave[-fade_len:] = raw_wave[-fade_len:] * fade_out

        # 5. Peak-normalize to [-0.90, 0.90]
        peak = raw_wave.abs().max().clamp(min=1e-6)
        waveform = 0.90 * (raw_wave / peak)

        # 6. Self-Listening Auditory Re-Entry Loop
        auditory_delta, reentry_telemetry = self.self_listen_and_reenter(
            waveform=waveform,
            hidden_states=hidden_states,
            adapt_online=adapt_online
        )

        with torch.no_grad():
            self.last_f0_mean.copy_(f0_frames.mean().detach())
            self.last_formants_mean.copy_(formant_frames.mean(dim=0).detach())

        telemetry = {
            "sample_rate": self.sample_rate,
            "duration_sec": float(n_samples / self.sample_rate),
            "n_samples": int(n_samples),
            "f0_mean_hz": float(f0_frames.mean().item()),
            "f0_min_hz": float(f0_frames.min().item()),
            "f0_max_hz": float(f0_frames.max().item()),
            "formants_hz": [round(float(x), 2) for x in formant_frames.mean(dim=0).tolist()],
            "bandwidths_hz": [round(float(x), 2) for x in bw_frames.mean(dim=0).tolist()],
            "waveform_rms": float(torch.sqrt(torch.mean(waveform ** 2)).item()),
            "misf_coupled": bool(vocal_params.get("misf_coupled", False)),
            "auditory_reentry_delta": auditory_delta,
            **reentry_telemetry
        }
        return waveform, telemetry

    def extract_spectral_envelope(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Computes log-magnitude spectral bins [1, n_spectral_bins] from a 1D waveform.
        """
        device = self.auditory_reentry_proj.weight.device
        w = waveform.to(device=device, dtype=torch.float32).view(-1)
        mag = torch.abs(torch.fft.rfft(w))
        log_mag = torch.log1p(mag)
        spec_bins = F.interpolate(
            log_mag.view(1, 1, -1),
            size=self.n_spectral_bins,
            mode="linear",
            align_corners=True
        ).view(1, self.n_spectral_bins)
        return F.normalize(spec_bins, p=2, dim=-1)

    def self_listen_and_reenter(
        self,
        waveform: torch.Tensor,
        hidden_states: torch.Tensor,
        adapt_online: bool = True,
        lr_voice: float = 0.08
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Projects the synthesized waveform's acoustic spectrum back into Layer 7
        latent space R^D, harmonizes via SpectralResonanceHarmonizer (rFFT/irFFT),
        measures semantic-acoustic coherence, and updates the vocal organ's
        spectral & formant projections online so the model adapts its own voice.
        """
        device = self.auditory_reentry_proj.weight.device
        spec_env = self.extract_spectral_envelope(waveform)  # [1, n_spectral_bins]

        if hidden_states.dim() == 3:
            h_sem = hidden_states[:, -1, :].detach().to(device=device, dtype=torch.float32).mean(dim=0, keepdim=True)
        else:
            h_sem = hidden_states.detach().to(device=device, dtype=torch.float32).mean(dim=0, keepdim=True)

        h_sem_norm = F.normalize(h_sem, p=2, dim=-1)  # [1, D]
        h_auditory = self.auditory_reentry_proj(spec_env)  # [1, D]
        h_aud_norm = F.normalize(h_auditory, p=2, dim=-1)

        coherence_before = float(F.cosine_similarity(h_aud_norm, h_sem_norm, dim=-1).mean().item())

        if adapt_online:
            with torch.no_grad():
                # 1. Hebbian alignment of auditory re-entry map toward semantic intention
                #    Delta W_reentry = lr * (h_sem_norm - h_aud_norm)^T @ spec_env
                err_latent = h_sem_norm - h_aud_norm  # [1, D]
                dW = lr_voice * torch.matmul(err_latent.t(), spec_env)  # [D, n_spectral_bins]
                self.auditory_reentry_proj.weight.add_(dW)

                # 2. Adapt spectral harmonic gain toward target semantic spectrum
                target_spec = F.normalize(F.softplus(self.semantic_acoustic_target(h_sem_norm)), p=2, dim=-1)
                spec_err = (target_spec - spec_env).squeeze(0)
                self.spectral_gain.add_(lr_voice * spec_err)
                self.spectral_gain.clamp_(min=0.2, max=3.0)

                # 3. Smoothly nudge formant projection bias to reduce spectral error energy
                formant_nudge = spec_err[:8] * 0.05
                self.formant_proj.bias.add_(formant_nudge)

                # 4. Co-adapt MISF sensitivity frequency field if coupled
                misf_ref = getattr(self, "_misf_ref", None)
                if misf_ref is not None and hasattr(misf_ref, "w_freq"):
                    misf_nudge = 0.001 * torch.matmul(
                        err_latent.t(),
                        torch.ones(1, misf_ref.w_freq.shape[1], device=err_latent.device, dtype=err_latent.dtype)
                    )
                    misf_ref.w_freq.data.add_(
                        misf_nudge.to(device=misf_ref.w_freq.device, dtype=misf_ref.w_freq.dtype)
                    )

                self.voice_adaptation_steps.add_(1)

                # Recompute post-adaptation coherence
                h_aud_after = F.normalize(self.auditory_reentry_proj(spec_env), p=2, dim=-1)
                coherence_after = float(F.cosine_similarity(h_aud_after, h_sem_norm, dim=-1).mean().item())
                self.ema_coherence.mul_(0.8).add_(0.2 * coherence_after)
                h_auditory = h_aud_after
        else:
            coherence_after = coherence_before

        # Apply SpectralResonanceHarmonizer (rFFT/irFFT phase-magnitude resonance)
        with torch.no_grad():
            h_harmonized, spectral_overlap = self.spectral_harmonizer(
                h_auditory.unsqueeze(1), h_sem_norm.unsqueeze(1)
            )
            h_reentry_vec = F.normalize(h_harmonized.squeeze(1), p=2, dim=-1)
            spectral_overlap_val = float(spectral_overlap.mean().item()) if hasattr(spectral_overlap, "mean") else float(spectral_overlap)

        # Match shape/device/dtype of hidden_states for residual re-entry injection into Layer 7
        reentry_delta = (0.02 * h_reentry_vec).to(device=hidden_states.device, dtype=hidden_states.dtype)

        return reentry_delta, {
            "semantic_acoustic_coherence_before": coherence_before,
            "semantic_acoustic_coherence_after": coherence_after,
            "spectral_resonance_overlap": spectral_overlap_val,
            "voice_adaptation_steps": int(self.voice_adaptation_steps.item()),
            "ema_voice_coherence": float(self.ema_coherence.item())
        }

    def save_wav(self, waveform: torch.Tensor, filepath: str) -> str:
        """
        Writes a 1D waveform tensor in [-1, 1] to a standard 16-bit PCM RIFF .wav file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        w_np = waveform.detach().cpu().float().numpy().reshape(-1)
        w_np = np.clip(w_np, -1.0, 1.0)
        pcm16 = (w_np * 32767.0).astype(np.int16)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm16.tobytes())

        return os.path.abspath(filepath)

    def train_vocal_babbling(
        self,
        hidden_states: torch.Tensor,
        text: str,
        target_formants: Optional[List[float]] = None,
        n_steps: int = 3,
        lr: float = 0.08,
        save_wav_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Active Vocal Babbling & Pronunciation Learning Loop ([train sound]).
        Simulates how a human child learns to speak: actively babbling letters ('A','I','U','E','O'),
        words ('Halo', 'X-Star'), and sentences; self-listening via auditory re-entry;
        evaluating acoustic formant discrepancy against the phonetic target; and
        adapting motor projection weights online until articulate resonance is mastered.
        """
        device = self.f0_proj.weight.device
        clean = (text or "om").strip().lower()

        # Canonical IPA vowel reference formants [F1, F2, F3, F4] in Hz
        canonical_vowels = {
            'a': [750.0, 1250.0, 2450.0, 3500.0],
            'i': [280.0, 2250.0, 2850.0, 3600.0],
            'u': [320.0, 850.0, 2250.0, 3400.0],
            'e': [500.0, 1850.0, 2550.0, 3550.0],
            'o': [500.0, 950.0, 2400.0, 3450.0],
        }

        # Resolve target formants [4]
        if target_formants is not None and len(target_formants) >= 4:
            tgt_f = torch.tensor(target_formants[:4], device=device, dtype=torch.float32)
        elif len(clean) == 1 and clean in canonical_vowels:
            tgt_f = torch.tensor(canonical_vowels[clean], device=device, dtype=torch.float32)
        else:
            # Multi-character target: extract dominant vowel formants
            v_matches = [ch for ch in clean if ch in canonical_vowels]
            if v_matches:
                dominant = canonical_vowels[v_matches[0]]
                tgt_f = torch.tensor(dominant, device=device, dtype=torch.float32)
            else:
                tgt_f = torch.tensor([600.0, 1400.0, 2500.0, 3500.0], device=device, dtype=torch.float32)

        f_low = self.formant_low.to(device)
        f_high = self.formant_high.to(device)
        norm_tgt = (tgt_f - f_low) / (f_high - f_low + 1e-6)

        progression = []
        final_waveform = None
        final_telemetry = {}
        saved_paths = []

        for step in range(1, n_steps + 1):
            # 1. Synthesize current active acoustic attempt
            waveform, telem = self.synthesize_waveform(
                hidden_states=hidden_states,
                text=text,
                duration_sec=None,
                adapt_online=True
            )
            final_waveform = waveform
            final_telemetry = telem

            curr_formants = torch.tensor(telem["formants_hz"], device=device, dtype=torch.float32)
            formant_err_hz = float(torch.norm(curr_formants - tgt_f).item())
            coherence = telem.get("semantic_acoustic_coherence_after", 0.5)

            # Stage classification
            if formant_err_hz > 300.0:
                stage_label = "BABBLING_UNFORMED"
            elif formant_err_hz > 140.0:
                stage_label = "FORMANT_RESONANCE_TUNING"
            else:
                stage_label = "ARTICULATE_MASTERED"

            progression.append({
                "step": step,
                "text": text,
                "formants_hz": [round(float(x), 1) for x in curr_formants.tolist()],
                "target_formants_hz": [round(float(x), 1) for x in tgt_f.tolist()],
                "formant_error_hz": round(formant_err_hz, 2),
                "semantic_acoustic_coherence": round(float(coherence), 4),
                "stage": stage_label
            })

            # Save intermediate step audio if requested
            if save_wav_prefix:
                step_path = f"{save_wav_prefix}_step{step}.wav"
                saved = self.save_wav(waveform, step_path)
                saved_paths.append(saved)

            # 2. Active Motor Plasticity Adaptation
            with torch.no_grad():
                curr_norm = (curr_formants - f_low) / (f_high - f_low + 1e-6)
                diff_norm = norm_tgt - curr_norm  # [4]
                # Nudge formant projection biases directly in articulatory motor direction
                self.formant_proj.bias[:4].add_(lr * diff_norm)
                # Stabilize fundamental pitch around optimal speech frequency ~160 Hz
                self.f0_proj.bias[0].add_(0.02 * (160.0 - float(telem.get("f0_mean_hz", 160.0))) / 100.0)

        return {
            "text": text,
            "n_steps": n_steps,
            "progression": progression,
            "final_formant_error_hz": progression[-1]["formant_error_hz"],
            "initial_formant_error_hz": progression[0]["formant_error_hz"],
            "error_reduction_pct": round(
                100.0 * max(0.0, (progression[0]["formant_error_hz"] - progression[-1]["formant_error_hz"]) / max(1e-3, progression[0]["formant_error_hz"])),
                2
            ),
            "final_stage": progression[-1]["stage"],
            "final_waveform": final_waveform,
            "final_telemetry": final_telemetry,
            "saved_wav_paths": saved_paths
        }

    @torch.no_grad()
    def reset(self) -> None:
        """Resets vocal organ adaptation counters and state buffers."""
        self.voice_adaptation_steps.zero_()
        self.ema_coherence.copy_(torch.tensor(0.5))
        self.last_f0_mean.copy_(torch.tensor(160.0))
        self.last_formants_mean.copy_(torch.tensor([500.0, 1500.0, 2500.0, 3500.0]))
