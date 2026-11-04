"""
X-Star Custom CLI Application (vLLM / llama-cli / LM Studio Experience)
========================================================================
Interactive terminal experience tailored for Qwen3.5-2B, Frontier LoRA weights,
and HADL v5.2 Neuromorphic Spiking Brain.

Features:
- Rich terminal UI with model architecture & VRAM telemetry banner
- Real-time token streaming with TTFT (ms), TPS, and Spiking Brain telemetry footer
- Interactive slash commands:
    /help               Display commands and shortcuts
    /browser <q|url>    Live web browsing and factual search grounding
    /goal <objective>   Autonomous HADL Cognitive Triad execution (/goal -> /plan -> /boost)
    /boost [on|off|1-5] Neuromorphic Deliberation Overdrive (LIF gate threshold tuning)
    /stats              Cumulative session telemetry and VRAM profiling
    /params [key=val]   Dynamic hyperparameter inspection and tuning
    /system [prompt]    Inspect or update system prompt
    /clear or /reset    Reset conversation context and spiking brain state
    /history            View full session transcript
    /save [filepath]    Export transcript (JSON/Markdown)
    /load [filepath]    Import previous transcript
    /model              Display deep architectural telemetry
    /exit or /quit      Graceful exit and GPU memory release
- One-shot execution mode with streaming (--prompt / -p)
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.rule import Rule
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

from .engine import CustomInferenceEngine, EngineConfig, GenerationMetrics
from .web_browser import WebBrowser
from .cognitive_triad import CognitiveTriadRunner


class InteractiveCLI:
    """Professional Interactive CLI interface for X-Star Autonomous Brain."""

    def __init__(self, engine: CustomInferenceEngine, default_system: Optional[str] = None):
        self.engine = engine
        self.system_prompt = default_system or (
            "You are X-Star, an elite autonomous AI brain architecture powered by "
            "Qwen3.5-2B with the HADL v5.2 neuromorphic spiking brain and Frontier LoRA weights. "
            "You reason systematically with high precision, grounded logic, and clear formatting."
        )
        self.messages: List[Dict[str, str]] = []
        self.console = Console(legacy_windows=False) if _HAS_RICH else None
        
        # Session Cumulative Stats
        self.session_turns = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_total_time = 0.0
        self.session_ttft_list: List[float] = []
        self.session_tps_list: List[float] = []
        self.session_spikes = 0

        # Default Generation Hyperparameters
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 40
        self.max_tokens = 512
        self.repetition_penalty = 1.1

    def print_banner(self):
        """Displays rich startup banner with hardware, VRAM, and model status."""
        vram = self.engine.get_vram_usage()
        brain = self.engine.get_brain_telemetry()
        gpu_name = "NVIDIA GPU"
        import torch
        if torch.cuda.is_available() and "cuda" in self.engine.device:
            dev_idx = torch.cuda.current_device() if self.engine.device == "cuda" else int(self.engine.device.split(":")[-1])
            gpu_name = torch.cuda.get_device_name(dev_idx)

        vram_str = f"{vram.get('allocated_gb', 0.0):.2f} GB / {vram.get('total_gb', 0.0):.2f} GB ({vram.get('percent_used', 0.0)}%)" if vram.get('total_gb') else "N/A (CPU)"

        if _HAS_RICH:
            table = Table.grid(padding=(0, 2))
            table.add_column(style="bold cyan", justify="right")
            table.add_column(style="bold white")
            table.add_row("Base Model", f"Qwen3.5-2B ({self.engine.torch_dtype})")
            table.add_row("LoRA Adapter", "Frontier LoRA (Rank 16, Alpha 32)")
            table.add_row("Cognitive Brain", "HADL v5.2 Neuromorphic Spiking Gate (Layer 7 LIF + SACW)")
            table.add_row("Variance Clamp", f"Active (Target: {self.engine.config.target_variance})")
            table.add_row("Compute Device", f"{self.engine.device} [{gpu_name}]")
            table.add_row("VRAM Allocated", vram_str)
            table.add_row("Boost Mode", f"[{'bold green' if self.engine._boost_active else 'dim'}]"
                                        f"{'ACTIVE (Level ' + str(self.engine._boost_level) + ')' if self.engine._boost_active else 'OFF'}[/]")

            panel = Panel(
                table,
                title="[bold gold1]* X-STAR 2B AUTONOMOUS BRAIN | HADL v5.2 INFERENCE *[/bold gold1]",
                subtitle="[dim]Type [bold cyan]/help[/bold cyan] for slash commands | [bold cyan]/exit[/bold cyan] to quit[/dim]",
                border_style="bright_blue",
                padding=(1, 2),
            )
            self.console.print(panel)
        else:
            print("=" * 76)
            print("   * X-STAR 2B AUTONOMOUS BRAIN | HADL v5.2 NEUROMORPHIC INFERENCE *")
            print("=" * 76)
            print(f" Base Model     : Qwen3.5-2B ({self.engine.torch_dtype})")
            print(" LoRA Adapter   : Frontier LoRA (Rank 16, Alpha 32)")
            print(" Cognitive Brain: HADL v5.2 Neuromorphic Spiking Gate (Layer 7 LIF + SACW)")
            print(f" Compute Device : {self.engine.device} [{gpu_name}]")
            print(f" VRAM Allocated : {vram_str}")
            print(f" Boost Mode     : {'ACTIVE (Level ' + str(self.engine._boost_level) + ')' if self.engine._boost_active else 'OFF'}")
            print("-" * 76)
            print(" Commands: /help, /browser <q|url>, /goal <task>, /boost, /stats, /exit")
            print("=" * 76)

    def print_footer(self, m: GenerationMetrics):
        """Displays professional llama-cli / LM Studio style performance telemetry footer."""
        self.session_turns += 1
        self.session_output_tokens += m.output_tokens
        self.session_prompt_tokens += m.prompt_tokens
        self.session_total_time += m.total_time_s
        if m.ttft_ms > 0:
            self.session_ttft_list.append(m.ttft_ms)
        if m.tokens_per_second > 0:
            self.session_tps_list.append(m.tokens_per_second)
        self.session_spikes += m.spikes_fired

        boost_label = f"ON (L{self.engine._boost_level})" if m.boost_active else "OFF"
        vram_str = f"{m.vram_allocated_mb / 1024:.2f} GB"

        if _HAS_RICH:
            text = Text()
            text.append("[*] ", style="bold gold1")
            text.append(f"{m.output_tokens} tokens", style="bold white")
            text.append(" | ", style="dim")
            text.append(f"{m.tokens_per_second:.1f} tok/s", style="bold cyan")
            text.append(" | ", style="dim")
            text.append(f"TTFT: {m.ttft_ms:.0f}ms", style="bold magenta")
            text.append(" | ", style="dim")
            text.append(f"VRAM: {vram_str}", style="bold green")
            text.append(" | ", style="dim")
            text.append(f"Spikes: {m.spikes_fired}", style="bold yellow")
            text.append(" | ", style="dim")
            text.append(f"State: {m.cognitive_state}", style="bold blue")
            text.append(" | ", style="dim")
            text.append(f"Boost: {boost_label}", style="bold green" if m.boost_active else "dim white")

            self.console.print(Panel(text, border_style="dim blue", padding=(0, 1)))
        else:
            print(f"\n[ {m.output_tokens} tokens | {m.tokens_per_second:.1f} tok/s | TTFT: {m.ttft_ms:.0f}ms | VRAM: {vram_str} | Spikes: {m.spikes_fired} | State: {m.cognitive_state} | Boost: {boost_label} ]\n")

    def handle_help(self):
        """Displays available interactive slash commands."""
        commands = [
            ("/browser <query>", "Search web using DuckDuckGo + Wikipedia and ground context"),
            ("/browser <url>", "Fetch webpage, extract clean text, and ground context"),
            ("/goal <task>", "Autonomous Cognitive Triad (/goal -> /plan -> /boost -> /goal audit)"),
            ("/plan <task>", "Fast architectural route inspection and milestone decomposition"),
            ("/teamwork-preview <q>", "Live dual-channel streaming: System 1 (reflex) vs System 2 (guided latent)"),
            ("/voice [text|path]", "Synthesize endogenous spiking-harmonic speech (freq suara) + self-listening loop"),
            ("/imagine [prompt]", "Inspect Progressive 2D-to-3D Visual Imagination & SO(3) mental rotation"),
            ("/boost [on|off|1-5]", "Toggle or tune Neuromorphic Spiking Deliberation Overdrive (K=1->3)"),
            ("/boost status", "Inspect live LIF threshold, Banach steps, membrane voltage & clamp parameters"),
            ("/stats", "Display cumulative session generation metrics, average TPS, and VRAM"),
            ("/params [key=val]", "Inspect or set parameters (temperature, top_p, top_k, max_tokens)"),
            ("/system [prompt]", "Inspect or set custom system prompt"),
            ("/clear, /reset", "Clear conversation history and reset neuromorphic brain state"),
            ("/history", "Display current session conversation history"),
            ("/save [filepath]", "Export conversation transcript as JSON or Markdown"),
            ("/load <filepath>", "Load conversation transcript from file"),
            ("/model", "Display deep model architecture, layer hooks, and parameter count"),
            ("/exit, /quit, /q", "Terminate session, flush memory, and release GPU resources"),
        ]

        if _HAS_RICH:
            table = Table(title="[bold gold1]X-STAR INTERACTIVE COMMANDS[/bold gold1]", border_style="cyan")
            table.add_column("Command", style="bold cyan", width=22)
            table.add_column("Description", style="white")
            for cmd, desc in commands:
                table.add_row(cmd, desc)
            self.console.print(table)
        else:
            print("\n--- X-STAR INTERACTIVE COMMANDS ---")
            for cmd, desc in commands:
                print(f"  {cmd:<24} : {desc}")
            print("-----------------------------------\n")

    def handle_browser(self, arg: str):
        """Executes /browser command for web search or direct URL reading."""
        query_or_url = arg.strip()
        if not query_or_url:
            print("Usage: /browser <search query or URL>")
            return

        if WebBrowser.is_url(query_or_url):
            if _HAS_RICH:
                self.console.print(f"[*] [cyan]Browsing URL:[/cyan] {query_or_url}...")
            else:
                print(f"[*] Browsing URL: {query_or_url}...")

            page_data = WebBrowser.fetch_url(query_or_url)
            if page_data.get("status") == "error":
                if _HAS_RICH:
                    self.console.print(f"[bold red][!] {page_data.get('content')}[/bold red]")
                else:
                    print(f"[!] {page_data.get('content')}")
                return

            if _HAS_RICH:
                self.console.print(Panel(
                    f"[bold white]{page_data['title']}[/bold white]\n[dim]{query_or_url}[/dim]\n\n"
                    f"{page_data['content'][:500]}...",
                    title="[bold green]Webpage Content Extracted[/bold green]",
                    border_style="green"
                ))
            else:
                print(f"\n[+] Title: {page_data['title']}")
                print(f"    Content preview: {page_data['content'][:300]}...\n")

            context_block = WebBrowser.format_url_context(page_data)
            prompt = (
                f"{context_block}\n\n"
                f"Please read the above browsed webpage and provide a clear, comprehensive analysis "
                f"and summary of its core findings."
            )
            self.execute_turn(prompt)

        else:
            if _HAS_RICH:
                self.console.print(f"[*] [cyan]Searching web for:[/cyan] '{query_or_url}'...")
            else:
                print(f"[*] Searching web for: '{query_or_url}'...")

            results = WebBrowser.search(query_or_url, max_results=4)
            if not results:
                if _HAS_RICH:
                    self.console.print("[yellow]No relevant external results found.[/yellow]")
                else:
                    print("[!] No relevant external results found.")
                return

            if _HAS_RICH:
                table = Table(title=f"Web Search Results: '{query_or_url}'", border_style="cyan")
                table.add_column("#", width=3, style="bold cyan")
                table.add_column("Title / Source", style="bold white", width=28)
                table.add_column("Snippet", style="white")
                for i, r in enumerate(results, start=1):
                    table.add_row(str(i), f"{r['title']}\n[dim]{r.get('source', '')}[/dim]", r['snippet'])
                self.console.print(table)
            else:
                print(f"\n--- Web Search Results for '{query_or_url}' ---")
                for i, r in enumerate(results, start=1):
                    print(f"[{i}] {r['title']} ({r.get('source')})")
                    print(f"    {r['snippet']}")
                print("------------------------------------------\n")

            context_block = WebBrowser.format_search_context(query_or_url, results)
            prompt = (
                f"{context_block}\n\n"
                f"Based on the real-time web browsing search evidence above, answer the query: '{query_or_url}' "
                f"accurately with factual citations."
            )
            self.execute_turn(prompt)

    def handle_goal(self, goal_text: str):
        """Executes /goal command running the 4-phase Autonomous Cognitive Triad."""
        goal_text = goal_text.strip()
        if not goal_text:
            print("Usage: /goal <task description or objective>")
            return

        if _HAS_RICH:
            self.console.print(Panel(
                f"[bold white]{goal_text}[/bold white]",
                title="[bold gold1][COGNITIVE TRIAD] (/goal -> /plan -> /boost -> /goal)[/bold gold1]",
                border_style="gold1",
            ))
        else:
            print("\n" + "=" * 70)
            print(f"[COGNITIVE TRIAD]: {goal_text}")
            print("=" * 70)

        def on_phase(phase_name: str, desc: str):
            if _HAS_RICH:
                self.console.print(f"\n[bold cyan]> {phase_name}[/bold cyan]")
                self.console.print(f"[dim]  {desc}[/dim]")
            else:
                print(f"\n> {phase_name}\n  {desc}")

        def on_chunk(chunk: str):
            sys.stdout.write(chunk)
            sys.stdout.flush()

        trace = CognitiveTriadRunner.execute_goal(
            engine=self.engine,
            goal_prompt=goal_text,
            on_phase_change=on_phase,
            on_stream_chunk=on_chunk,
        )

        sys.stdout.write("\n")
        
        # Display audit gate result
        passed = trace.get("passed_gate", False)
        audit_text = trace.get("phases", {}).get("audit", "")
        if _HAS_RICH:
            self.console.print(Panel(
                Text(audit_text),
                title=f"[bold {'green' if passed else 'yellow'}]Cognitive Triad Verification Gate: {'PASSED [OK]' if passed else 'VERIFIED WITH COMMENTS'}[/bold {'green' if passed else 'yellow'}]",
                border_style="green" if passed else "yellow",
            ))
        else:
            print("\n" + "-" * 70)
            print(f"Verification Gate Audit: {'PASSED' if passed else 'COMPLETED'}")
            print(audit_text)
            print("-" * 70 + "\n")

        # Save to conversation memory
        self.messages.append({"role": "user", "content": f"/goal {goal_text}"})
        self.messages.append({"role": "assistant", "content": trace.get("final_solution", "")})

        triad_metrics = trace.get("metrics", {}).get("boost_phase")
        if triad_metrics:
            self.print_footer(triad_metrics)

    def handle_plan(self, plan_text: str):
        """Executes /plan: Fast architectural route inspection and milestone decomposition without 4-phase prompt overhead."""
        plan_text = plan_text.strip()
        if not plan_text:
            print("Usage: /plan <task or problem description>")
            return

        # Fast architectural inspection
        brain_telem = self.engine.get_brain_telemetry()
        u_drive = brain_telem.get("pacemaker_u_drive_mean", 0.0)
        v_th = brain_telem.get("spike_v_th", 1.0)
        from dual_loop.neuromorphic import HADLv5SpikingBrainModelWrapper
        k_steps = self.engine.model.deliberation_steps_k if isinstance(self.engine.model, HADLv5SpikingBrainModelWrapper) else 1

        route_rec = "Direct System 1 Reflex (<5ms)" if abs(u_drive) < 0.20 else f"System 2 Latent Deliberation (K={k_steps} Banach steps)"

        if _HAS_RICH:
            table = Table.grid(padding=(0, 2))
            table.add_column(style="bold cyan", justify="right")
            table.add_column(style="bold white")
            table.add_row("Architectural Route", route_rec)
            table.add_row("Pacemaker Driving Signal u_drive", f"{u_drive:.4f}")
            table.add_row("LIF Threshold V_th", f"{v_th:.2f}")
            table.add_row("System 2 Contraction Steps (K)", str(k_steps))

            panel = Panel(
                table,
                title=f"[bold gold1][ARCHITECTURAL ROUTE INSPECTION] '{plan_text}'[/bold gold1]",
                border_style="gold1",
                padding=(1, 2)
            )
            self.console.print(panel)
        else:
            print(f"\n[ARCHITECTURAL ROUTE INSPECTION]: '{plan_text}'")
            print(f"  Route: {route_rec} | u_drive: {u_drive:.4f} | V_th: {v_th:.2f} | K: {k_steps}")

        # Stream fast milestone decomposition without heavy 4-phase overhead
        plan_prompt = (
            f"Provide a concise, direct architectural milestone decomposition for the following task without boilerplate:\n"
            f"Task: {plan_text}\n\n"
            f"Requirements:\n"
            f"1. Concise execution milestones (1-3 bullets).\n"
            f"2. Key logical invariants to uphold.\n"
            f"3. Concrete immediate next step."
        )
        self.execute_turn(plan_prompt)

    def handle_teamwork_preview(self, query: str):
        """
        Executes /teamwork-preview: Live dual-channel inspection of System 1 (intuitive fast reflex)
        vs System 2 (contractive latent deliberation & tour leader restoring force).
        """
        query = query.strip()
        if not query:
            print("Usage: /teamwork-preview <prompt or query>")
            return

        if _HAS_RICH:
            self.console.print(Panel(
                f"[bold white]{query}[/bold white]",
                title="[bold cyan][TEAMWORK PREVIEW: CONCURRENT DUAL-STREAM][/bold cyan]",
                subtitle="[dim]System 1 (Reflexive Stream, <5ms)  <-->  System 2 (Latent Banach + Tour Leader)[/dim]",
                border_style="cyan",
            ))
        else:
            print("\n" + "=" * 70)
            print(f"[TEAMWORK PREVIEW: CONCURRENT DUAL-STREAM]: {query}")
            print("System 1 (Reflexive Stream) <--> System 2 (Latent Banach + Tour Leader)")
            print("=" * 70)

        # Generate response with dual-stream telemetry
        self.messages.append({"role": "user", "content": query})

        if _HAS_RICH:
            self.console.print("\n[bold green]Concurrent Stream > [/bold green]", end="")
        else:
            print("\nConcurrent Stream > ", end="", flush=True)

        full_chunks = []
        last_metrics: Optional[GenerationMetrics] = None
        for chunk, metrics in self.engine.stream_chat(
            messages=self.messages,
            system_prompt=self.system_prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        ):
            if chunk:
                full_chunks.append(chunk)
                if _HAS_RICH:
                    self.console.print(chunk, end="", markup=False)
                else:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            if metrics:
                last_metrics = metrics

        print()
        response_text = "".join(full_chunks)
        if response_text:
            self.messages.append({"role": "assistant", "content": response_text})

        # Dual-Channel Telemetry Breakdown Panel
        s1_lat = last_metrics.s1_latency_ms if last_metrics else 0.0
        s2_norm = last_metrics.s2_delta_norm if last_metrics else 0.0
        tour_f = last_metrics.tour_restoring_force if last_metrics else 0.0
        u_dr = last_metrics.pacemaker_u_drive if last_metrics else 0.0
        sec_applied = last_metrics.directional_safety_applied if last_metrics else False

        if _HAS_RICH:
            table = Table(title="Concurrent Dual-Stream Synthesis Diagnostics", border_style="magenta")
            table.add_column("Stream Component", style="bold cyan")
            table.add_column("Latency / Dynamic Force", style="bold white")
            table.add_column("Role & Symplectic Invariant", style="green")

            table.add_row(
                "System 1 (Reflexive Stream)",
                f"{s1_lat:.1f} ms",
                "Unconstrained base forward trajectory; fluency & zero-overhead"
            )
            table.add_row(
                "System 2 (Banach Deliberation)",
                f"Delta Norm = {s2_norm:.4f} (u_drive = {u_dr:.4f})",
                "Orthogonal symplectic projection: <P_perp Delta, h_S1> = 0"
            )
            table.add_row(
                "Tour Leader Attractor",
                f"Restoring Force = {tour_f:.4f}",
                "Continuous vector field -mu * nabla Phi_logic (No rigid templates)"
            )
            table.add_row(
                "Directional Safety Hook",
                f"{'ACTIVE (Top-1 Rank Preserved)' if sec_applied else 'MONITORED (Floor Intact)'}",
                "Guarantees monotonic ranking margin_new >= margin_base"
            )
            self.console.print(table)
        else:
            print("\n--- Dual-Stream Diagnostics ---")
            print(f"  System 1 Latency        : {s1_lat:.1f} ms")
            print(f"  System 2 Delta Norm     : {s2_norm:.4f} (u_drive = {u_dr:.4f})")
            print(f"  Tour Leader Restoring   : {tour_f:.4f}")
            print(f"  Directional Safety      : {'ACTIVE' if sec_applied else 'MONITORED'}")
            print("-------------------------------\n")

        if last_metrics:
            self.print_footer(last_metrics)

    def handle_boost(self, arg: str):
        """Executes /boost command for neuromorphic deliberation overdrive."""
        parts = arg.strip().lower().split()
        sub = parts[0] if parts else ""

        if sub in ("status", "info"):
            telem = self.engine.get_brain_telemetry()
            vram = self.engine.get_vram_usage()
            from dual_loop.neuromorphic import HADLv5SpikingBrainModelWrapper
            k_steps = self.engine.model.deliberation_steps_k if isinstance(self.engine.model, HADLv5SpikingBrainModelWrapper) else 1
            ckpts = telem.get("checkpoints_loaded", {})
            ckpt_str = ", ".join([f"{k}: {'OK' if v else 'MISSING'}" for k, v in ckpts.items()]) if ckpts else "N/A"

            if _HAS_RICH:
                table = Table(title="Neuromorphic Brain & Boost Telemetry", border_style="magenta")
                table.add_column("Parameter", style="bold cyan")
                table.add_column("Value", style="bold white")
                table.add_row("Boost Mode", "[bold green]ACTIVE[/bold green]" if telem.get("boost_active") else "[dim]OFF[/dim]")
                table.add_row("Boost Level", str(telem.get("boost_level", 3)))
                table.add_row("System 2 Contraction Steps (K)", str(k_steps))
                table.add_row("Spike Threshold (V_th)", f"{telem.get('spike_v_th', 1.0):.2f}")
                table.add_row("Pacemaker Driving Signal", f"{telem.get('pacemaker_u_drive_mean', 0.0):.4f}")
                table.add_row("Pre-Router Mode", str(telem.get("router_route_mode", "bypass")).upper())
                table.add_row("Peak Cognitive State", telem.get("peak_cognitive_state", "ROUTINE_CALM"))
                table.add_row("Variance Stabilizer", "[green]Active[/green]" if telem.get("variance_stabilizer_enabled") else "Inactive")
                table.add_row("Loaded Checkpoints", ckpt_str)
                table.add_row("VRAM Allocated", f"{vram.get('allocated_mb', 0.0):.1f} MB ({vram.get('percent_used', 0.0)}%)")
                self.console.print(table)
            else:
                print("\n--- Neuromorphic Brain & Boost Telemetry ---")
                print(f"  Boost Mode         : {'ACTIVE' if telem.get('boost_active') else 'OFF'}")
                print(f"  Boost Level        : {telem.get('boost_level', 3)}")
                print(f"  System 2 Banach K  : {k_steps}")
                print(f"  Spike Threshold    : {telem.get('spike_v_th', 1.0):.2f}")
                print(f"  Pacemaker u_drive  : {telem.get('pacemaker_u_drive_mean', 0.0):.4f}")
                print(f"  Peak Cognitive State: {telem.get('peak_cognitive_state', 'ROUTINE_CALM')}")
                print(f"  Variance Stabilizer : {'Active' if telem.get('variance_stabilizer_enabled') else 'Inactive'}")
                print(f"  Loaded Checkpoints : {ckpt_str}")
                print("--------------------------------------------\n")
            return

        if sub in ("on", "1", "2", "3", "4", "5", ""):
            level = int(sub) if sub.isdigit() else (int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 3)
            res = self.engine.set_boost(enabled=True, level=level)
            msg = (
                f"[BOOST ACTIVATED] Neuromorphic Deliberation Overdrive: Level {res['boost_level']} "
                f"(K = {res.get('deliberation_steps_k', 2)} Banach steps, V_th = {res.get('spike_v_th', 0.60):.2f}, "
                f"VarianceClamp = {res.get('target_variance', 1.10):.2f})"
            )
            if _HAS_RICH:
                self.console.print(f"[bold green]{msg}[/bold green]")
            else:
                print(msg)
        elif sub in ("off", "disable", "stop"):
            res = self.engine.set_boost(enabled=False)
            msg = f"[BOOST DEACTIVATED] Restored baseline neuromorphic parameters (K = 1, V_th = {res.get('spike_v_th', 1.0):.2f})."
            if _HAS_RICH:
                self.console.print(f"[bold yellow]{msg}[/bold yellow]")
            else:
                print(msg)
        else:
            print("Usage: /boost [on | off | status | 1-5]")

    def handle_stats(self):
        """Displays cumulative session statistics."""
        vram = self.engine.get_vram_usage()
        avg_ttft = sum(self.session_ttft_list) / max(1, len(self.session_ttft_list))
        avg_tps = sum(self.session_tps_list) / max(1, len(self.session_tps_list))

        if _HAS_RICH:
            table = Table(title="X-Star Session Cumulative Telemetry", border_style="gold1")
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", style="bold white")
            table.add_row("Conversation Turns", str(self.session_turns))
            table.add_row("Prompt Tokens Processed", f"{self.session_prompt_tokens:,}")
            table.add_row("Generated Tokens Output", f"{self.session_output_tokens:,}")
            table.add_row("Total Processed Tokens", f"{self.session_prompt_tokens + self.session_output_tokens:,}")
            table.add_row("Average Generation TPS", f"{avg_tps:.2f} tok/s")
            table.add_row("Average TTFT", f"{avg_ttft:.1f} ms")
            table.add_row("Cumulative Spikes Fired", str(self.session_spikes))
            table.add_row("Current GPU VRAM Allocated", f"{vram.get('allocated_gb', 0.0):.2f} GB ({vram.get('percent_used', 0.0)}%)")
            table.add_row("Total Dedicated VRAM", f"{vram.get('total_gb', 0.0):.2f} GB")
            self.console.print(table)
        else:
            print("\n--- X-STAR SESSION CUMULATIVE TELEMETRY ---")
            print(f"  Conversation Turns      : {self.session_turns}")
            print(f"  Prompt Tokens           : {self.session_prompt_tokens:,}")
            print(f"  Generated Tokens        : {self.session_output_tokens:,}")
            print(f"  Average TPS             : {avg_tps:.2f} tok/s")
            print(f"  Average TTFT            : {avg_ttft:.1f} ms")
            print(f"  Cumulative Spikes       : {self.session_spikes}")
            print(f"  VRAM Allocated          : {vram.get('allocated_gb', 0.0):.2f} GB")
            print("-------------------------------------------\n")

    def handle_params(self, arg: str):
        """Views or modifies sampling parameters."""
        if not arg.strip():
            print(
                f"Current Parameters:\n"
                f"  temperature        = {self.temperature}\n"
                f"  top_p              = {self.top_p}\n"
                f"  top_k              = {self.top_k}\n"
                f"  max_tokens         = {self.max_tokens}\n"
                f"  repetition_penalty = {self.repetition_penalty}\n"
                f"To change: /params temperature=0.5 max_tokens=1024"
            )
            return

        import re
        norm_arg = re.sub(r"\s*=\s*", "=", arg.strip())
        tokens = norm_arg.split()

        items = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if "=" in tok:
                k, v = tok.split("=", 1)
                items.append((k, v))
                i += 1
            elif i + 1 < len(tokens):
                items.append((tok, tokens[i + 1]))
                i += 2
            else:
                items.append((tok, ""))
                i += 1

        for k, v in items:
            k = k.strip().lower()
            v = v.strip()
            if not v:
                print(f"[!] Missing value for parameter: {k}")
                continue
            try:
                if k in ("temp", "temperature"):
                    self.temperature = float(v)
                    print(f"[OK] temperature set to {self.temperature}")
                elif k in ("top_p", "topp"):
                    self.top_p = float(v)
                    print(f"[OK] top_p set to {self.top_p}")
                elif k in ("top_k", "topk"):
                    self.top_k = int(v)
                    print(f"[OK] top_k set to {self.top_k}")
                elif k in ("max_tokens", "max_new_tokens", "tokens"):
                    self.max_tokens = int(v)
                    print(f"[OK] max_tokens set to {self.max_tokens}")
                elif k in ("rep", "repetition_penalty"):
                    self.repetition_penalty = float(v)
                    print(f"[OK] repetition_penalty set to {self.repetition_penalty}")
                else:
                    print(f"[!] Unknown parameter: {k}")
            except ValueError as e:
                print(f"[!] Invalid value for {k}: {v} ({e})")

    def handle_system(self, arg: str):
        """Views or modifies system prompt."""
        new_prompt = arg.strip()
        if not new_prompt:
            print(f"\n[System Prompt]:\n{self.system_prompt}\n")
        else:
            self.system_prompt = new_prompt
            print("[OK] System prompt updated for subsequent turns.")

    def handle_clear(self):
        """Clears conversation history and resets brain state."""
        self.messages.clear()
        self.engine.reset_state()
        print("[OK] Conversation history cleared and spiking brain state reset.")

    def handle_history(self):
        """Displays session history."""
        if not self.messages:
            print("[History is empty.]")
            return
        print("\n--- CONVERSATION HISTORY ---")
        for i, m in enumerate(self.messages, start=1):
            role = m['role'].upper()
            content = m['content']
            print(f"[{i}] {role}:\n{content}\n")
        print("----------------------------\n")

    def handle_save(self, path_str: str):
        """Exports session history to file (JSON or Markdown)."""
        if not path_str.strip():
            path_str = f"xstar_chat_{int(time.time())}.json"
        p = Path(path_str.strip())
        if p.suffix.lower() == ".md":
            lines = [
                "# X-Star Conversation Transcript",
                f"**Exported At:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
                f"**System Prompt:** {self.system_prompt}\n",
                "---",
            ]
            for m in self.messages:
                role = m.get("role", "unknown").capitalize()
                content = m.get("content", "")
                lines.append(f"\n### {role}\n\n{content}\n")
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        else:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({
                    "system_prompt": self.system_prompt,
                    "messages": self.messages,
                    "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, indent=2, ensure_ascii=False)
        print(f"[OK] Session saved to {p.resolve()}")

    def handle_load(self, path_str: str):
        """Imports session history from file."""
        if not path_str.strip():
            print("Usage: /load <filepath>")
            return
        p = Path(path_str.strip())
        if not p.exists():
            print(f"[!] File not found: {p}")
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.messages = data.get("messages", [])
                if "system_prompt" in data:
                    self.system_prompt = data["system_prompt"]
            print(f"[OK] Loaded {len(self.messages)} messages from {p}")
        except Exception as e:
            print(f"[!] Failed to load {p}: {e}")

    def handle_voice(self, arg: str):
        """Synthesizes endogenous spiking-harmonic formant speech (`freq suara`) and runs self-listening loop."""
        text = arg.strip()
        if not text and self.messages:
            for m in reversed(self.messages):
                if m.get("role") == "assistant" and m.get("content"):
                    text = m["content"][:200]
                    break
        if not text:
            text = "Hello, I am X-Star speaking with my endogenous spiking harmonic vocal organ."

        out_wav = Path(f"xstar_voice_{int(time.time())}.wav")
        res = self.engine.synthesize_voice(
            text=text,
            save_path=str(out_wav),
            duration_sec=1.5,
            enable_self_listening=True,
        )
        synth_t = res.get("synthesis_telemetry", {})
        listen_t = res.get("self_listening_telemetry", {})
        f0_hz = synth_t.get("f0_mean_hz", 0.0)
        formants = synth_t.get("formants_hz", [])
        coherence = listen_t.get("coherence_after", listen_t.get("coherence_before", 0.0))
        saved_file = res.get("saved_path", str(out_wav))
        print(
            f"[VOICE SYNTHESIZED] f0={f0_hz:.1f} Hz | Formants F1..F4={[round(f, 1) for f in formants]} Hz | "
            f"Self-Listening Coherence={coherence:.4f} | Saved: {saved_file}"
        )

    def handle_imagine(self, arg: str):
        """Inspects Progressive 2D-to-3D Visual Imagination & SO(3) mental rotation."""
        prompt_text = arg.strip() or "3D geometric manifold imagination"
        telem = self.engine.get_3d_imagination_telemetry(text=prompt_text)
        print(
            f"[2D->3D IMAGINATION] 2D Sketchpad={telem.get('sketchpad_2d_resolution', (16, 16))} | "
            f"3D Primitives={telem.get('num_3d_primitives', 8)} | "
            f"SO(3) Orthogonality Err={telem.get('so3_orthogonality_error', 0.0):.2e} | "
            f"det(R)={telem.get('so3_det', 1.0):.4f} | "
            f"2D<->3D Splat Consistency={telem.get('consistency_2d_to_3d', 0.0):.4f}"
        )

    def execute_turn(self, user_content: str):
        """Executes a single conversational generation turn with streaming."""
        self.messages.append({"role": "user", "content": user_content})

        if _HAS_RICH:
            self.console.print("\n[bold green]X-Star > [/bold green]", end="")
        else:
            print("\nX-Star > ", end="", flush=True)

        full_chunks = []
        last_metrics: Optional[GenerationMetrics] = None
        in_think_block = False

        try:
            for chunk, metrics in self.engine.stream_chat(
                messages=self.messages,
                system_prompt=self.system_prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
            ):
                if chunk:
                    full_chunks.append(chunk)
                    # Visual styling for <think> tags if present
                    if "<think>" in chunk:
                        in_think_block = True
                        chunk = chunk.replace("<think>", "[Thinking trace...]\n" if not _HAS_RICH else "")
                    if "</think>" in chunk:
                        in_think_block = False
                        chunk = chunk.replace("</think>", "\n" if not _HAS_RICH else "")

                    if _HAS_RICH:
                        style = "dim italic" if in_think_block else None
                        self.console.print(chunk, end="", style=style, markup=False)
                    else:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()

                if metrics:
                    last_metrics = metrics
        except KeyboardInterrupt:
            print("\n[Generation interrupted by user]")

        print()
        response_text = "".join(full_chunks)
        if response_text:
            self.messages.append({"role": "assistant", "content": response_text})

        if last_metrics:
            self.print_footer(last_metrics)

    def run_repl(self):
        """Starts the interactive multi-turn REPL loop."""
        self.print_banner()

        while True:
            try:
                if _HAS_RICH:
                    self.console.print("[bold cyan]User > [/bold cyan]", end="")
                    user_input = input("").strip()
                else:
                    user_input = input("User > ").strip()

                if not user_input:
                    continue

                # Multiline support (ending with \ or wrapped in """)
                if user_input.endswith("\\"):
                    lines = [user_input[:-1]]
                    while True:
                        if _HAS_RICH:
                            self.console.print("[dim cyan]... [/dim cyan]", end="")
                            sub_line = input("")
                        else:
                            sub_line = input("... ")
                        if sub_line.endswith("\\"):
                            lines.append(sub_line[:-1])
                        else:
                            lines.append(sub_line)
                            break
                    user_input = "\n".join(lines).strip()
                elif user_input.startswith('"""') and (len(user_input) == 3 or not user_input.endswith('"""')):
                    lines = [user_input[3:]]
                    while True:
                        if _HAS_RICH:
                            self.console.print("[dim cyan]... [/dim cyan]", end="")
                            sub_line = input("")
                        else:
                            sub_line = input("... ")
                        if sub_line.endswith('"""'):
                            lines.append(sub_line[:-3])
                            break
                        lines.append(sub_line)
                    user_input = "\n".join(lines).strip()

                if not user_input:
                    continue

                # Handle Slash Commands
                if user_input.startswith("/"):
                    cmd_parts = user_input[1:].split(maxsplit=1)
                    cmd = cmd_parts[0].lower()
                    arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                    if cmd in ("exit", "quit", "q"):
                        print("\nSession terminated. Cleaning up VRAM...")
                        self.engine.reset_state()
                        print("Goodbye!")
                        break
                    elif cmd in ("help", "h", "?"):
                        self.handle_help()
                    elif cmd in ("browser", "browse"):
                        self.handle_browser(arg)
                    elif cmd in ("goal",):
                        self.handle_goal(arg)
                    elif cmd in ("plan",):
                        self.handle_plan(arg)
                    elif cmd in ("teamwork-preview", "teamwork", "team"):
                        self.handle_teamwork_preview(arg)
                    elif cmd in ("voice", "speak", "suara"):
                        self.handle_voice(arg)
                    elif cmd in ("imagine", "3d"):
                        self.handle_imagine(arg)
                    elif cmd in ("boost", "overdrive"):
                        self.handle_boost(arg)
                    elif cmd == "stats":
                        self.handle_stats()
                    elif cmd in ("params", "param", "config"):
                        self.handle_params(arg)
                    elif cmd == "system":
                        self.handle_system(arg)
                    elif cmd in ("clear", "reset"):
                        self.handle_clear()
                    elif cmd == "history":
                        self.handle_history()
                    elif cmd == "save":
                        self.handle_save(arg)
                    elif cmd == "load":
                        self.handle_load(arg)
                    elif cmd == "model":
                        telem = self.engine.get_brain_telemetry()
                        print(json.dumps(telem, indent=2))
                    else:
                        print(f"Unknown command: /{cmd}. Type /help for available commands.")
                    continue

                # Normal conversational generation
                self.execute_turn(user_input)

            except KeyboardInterrupt:
                print("\n[Session interrupted by user. Type /exit to quit.]")
            except Exception as e:
                print(f"\n[ERROR]: {e}")


def main():
    parser = argparse.ArgumentParser(
        prog="xstar-infer",
        description="X-Star Custom Inference Suite (HADL v5.2 Neuromorphic Spiking Brain)",
    )
    parser.add_argument("--model", type=str, default="dist/staging_hf_fused", help="Path to base model directory")
    parser.add_argument("--adapter", type=str, default="dist/frontier_lora_adapter", help="Path to LoRA adapter weights")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda:0, cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model precision")
    parser.add_argument("--no-brain", action="store_true", help="Disable HADL v5.2 Neuromorphic Spiking Brain hooks")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA adapter weights")
    parser.add_argument("--boost", action="store_true", help="Initialize with Neuromorphic Deliberation Boost enabled")
    parser.add_argument("--boost-level", type=int, default=3, choices=[1, 2, 3, 4, 5], help="Deliberation boost intensity (1-5)")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="One-shot prompt to generate and exit")
    parser.add_argument("--system", "-s", type=str, default=None, help="Custom system prompt")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--max-tokens", "-n", type=int, default=512, help="Maximum generation tokens")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--serve", action="store_true", help="Start OpenAI-compatible REST API server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host interface")
    parser.add_argument("--port", type=int, default=8000, help="Server port number")
    parser.add_argument("--api-key", type=str, default=None, help="Server API key for authentication")

    args = parser.parse_args()

    # Serve mode
    if args.serve:
        from .server import start_server
        start_server(
            model_path=args.model,
            adapter_path=None if args.no_lora else args.adapter,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            device=args.device,
            dtype=args.dtype,
        )
        return

    # Initialize Engine
    cfg = EngineConfig(
        model_path=args.model,
        adapter_path=None if args.no_lora else args.adapter,
        device=args.device,
        dtype=args.dtype,
        enable_brain=not args.no_brain,
        enable_adapter=not args.no_lora,
        boost_mode=args.boost,
        boost_level=args.boost_level,
    )
    engine = CustomInferenceEngine(cfg)

    # One-shot CLI Mode
    if args.prompt:
        engine.load()
        cli = InteractiveCLI(engine, default_system=args.system)
        cli.temperature = args.temperature
        cli.top_p = args.top_p
        cli.top_k = args.top_k
        cli.max_tokens = args.max_tokens
        cli.repetition_penalty = args.repetition_penalty

        cli.execute_turn(args.prompt)
        return

    # Interactive REPL Mode
    engine.load()
    cli = InteractiveCLI(engine, default_system=args.system)
    cli.temperature = args.temperature
    cli.top_p = args.top_p
    cli.top_k = args.top_k
    cli.max_tokens = args.max_tokens
    cli.repetition_penalty = args.repetition_penalty
    cli.run_repl()


if __name__ == "__main__":
    main()
