"""
X-Star CLI: Command-Line Interface for X-Star Autonomous Brain
==============================================================
Commands:
  xstar chat                  Interactive multi-turn terminal conversation
  xstar run <prompt>          One-shot generation command
  xstar info                  Architecture stats, memory bank & GPU telemetry
  xstar bench [--suite ...]   Execute empirical benchmark suite
  xstar serve [--port ...]    Launch OpenAI-compatible FastAPI server
"""

import os
import sys
import argparse
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def cmd_run(args):
    import xstar
    print(f"[*] Loading X-Star Model from: {args.model or 'default'}...")
    model = xstar.load(
        model_path=args.model,
        device=args.device,
        trust_remote_code=getattr(args, "trust_remote_code", False)
    )
    
    t0 = time.perf_counter()
    response = model.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        strip_thinking=not args.raw
    )
    elapsed = time.perf_counter() - t0
    
    print("\n--- [X-STAR RESPONSE] ---")
    print(response)
    print("------------------------")
    print(f"[Finished in {elapsed:.2f}s]\n")


def cmd_chat(args):
    import xstar
    print("=" * 70)
    print("X-STAR 2B AUTONOMOUS BRAIN: INTERACTIVE CHAT SESSION")
    print(f"Device: {args.device} | Type 'exit' or 'quit' to terminate.")
    print("=" * 70)

    model = xstar.load(
        model_path=args.model,
        device=args.device,
        trust_remote_code=getattr(args, "trust_remote_code", False)
    )
    messages = []

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Session terminated. Goodbye!")
                break

            messages.append({"role": "user", "content": user_input})
            t0 = time.perf_counter()
            reply = model.chat(
                messages,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            elapsed = time.perf_counter() - t0

            print(f"\nX-Star > {reply}")
            print(f"(Latency: {elapsed:.2f}s)")
            messages.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break


def cmd_info(args):
    import xstar
    model = xstar.load(
        model_path=args.model,
        device=args.device,
        trust_remote_code=getattr(args, "trust_remote_code", False)
    )
    info = model.get_info()

    print("\n" + "=" * 65)
    print("X-STAR 2B AUTONOMOUS BRAIN: ARCHITECTURAL TELEMETRY")
    print("=" * 65)
    print(f"Package Version          : v{info['version']}")
    print(f"Model Path               : {info['model_path']}")
    print(f"Compute Device           : {info['device']} ({info['gpu_name']})")
    print(f"Tensor Dtype             : {info['dtype']}")
    print(f"Total Parameters         : {info['total_parameters']:,} (~2.04B)")
    print(f"Standalone HF Architecture: {info['is_standalone_hf']}")
    if "vgni_memory_slots_filled" in info:
        print(f"VGNI Epistemic Memory    : {info['vgni_memory_slots_filled']} / {info['vgni_max_capacity']} slots occupied")
    print("Core Capabilities:")
    print("  * Deep Layer 11 Manifold Interception & VGNI Gating")
    print("  * Commutative Dual-Composition LBU Verification (f(g) vs g(f))")
    print("  * Endogenous Banach Continuous Plasticity & Homeostatic Restitution")
    print("  * cuBLAS-Accelerated Polynomial Affordance Stack")
    print("=" * 65 + "\n")


def cmd_bench(args):
    import subprocess
    project_root = Path(__file__).resolve().parent.parent

    if args.suite == "1000q":
        script = project_root / "scripts" / "benchmark_1000q_20parts_real.py"
    else:
        script = project_root / "scripts" / "benchmark_200q_20parts_real.py"

    if not script.exists():
        print(f"[!] Benchmark script not found at: {script}")
        return

    print(f"[*] Launching {args.suite} benchmark suite...")
    subprocess.run([sys.executable, str(script)], cwd=str(project_root))


def cmd_serve(args):
    # SEC-02 hardening: Enforce authentication when binding to non-loopback interfaces
    is_loopback = args.host in ("127.0.0.1", "localhost", "::1")
    resolved_api_key = (
        getattr(args, "api_key", None)
        or os.environ.get("XSTAR_API_KEY", "").strip()
        or os.environ.get("DUAL_LOOP_API_KEY", "").strip()
        or None
    )
    if not is_loopback and not resolved_api_key:
        raise ValueError(
            f"Binding to non-loopback host '{args.host}' requires an API key for authentication. "
            "Please provide --api-key or set XSTAR_API_KEY / DUAL_LOOP_API_KEY environment variable."
        )

    try:
        from fastapi import FastAPI, HTTPException, Security, Depends
        from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
        from pydantic import BaseModel
        import uvicorn
        import xstar
    except ImportError:
        print("[!] FastAPI and Uvicorn required for serve mode. Install with: pip install fastapi uvicorn")
        return

    app = FastAPI(title="X-Star Autonomous Brain API", version=xstar.__version__)
    security = HTTPBearer(auto_error=False) if resolved_api_key else None

    def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security) if security else None):
        if resolved_api_key:
            if not credentials or credentials.credentials != resolved_api_key:
                raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return True

    model = xstar.load(
        model_path=args.model,
        device=args.device,
        trust_remote_code=getattr(args, "trust_remote_code", False)
    )

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 256
        temperature: float = 0.0

    dependencies = [Depends(verify_api_key)] if resolved_api_key else []

    @app.post("/v1/completions", dependencies=dependencies)
    def completions(req: GenerateRequest):
        t0 = time.perf_counter()
        text = model.generate(req.prompt, max_new_tokens=req.max_tokens, temperature=req.temperature)
        latency = time.perf_counter() - t0
        return {
            "choices": [{"text": text}],
            "model": "xstar-2b-autonomous-brain",
            "latency_seconds": round(latency, 4)
        }

    print(f"[*] Starting X-Star REST API server on http://{args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_infer(args):
    """Launches custom inference session or one-shot prompt with /browser, /goal, and /boost."""
    from xstar.inference.engine import CustomInferenceEngine, EngineConfig
    from xstar.inference.cli_app import InteractiveCLI

    model_dir = args.model or "dist/staging_hf_fused"
    adapter_dir = None if getattr(args, "no_lora", False) else getattr(args, "adapter", "dist/frontier_lora_adapter")

    config = EngineConfig(
        model_path=model_dir,
        adapter_path=adapter_dir,
        device=args.device,
        dtype=getattr(args, "dtype", "bfloat16"),
        enable_brain=not getattr(args, "no_brain", False),
        enable_adapter=not getattr(args, "no_lora", False),
        boost_mode=getattr(args, "boost", False),
        boost_level=getattr(args, "boost_level", 3),
    )
    engine = CustomInferenceEngine(config)
    engine.load()

    cli = InteractiveCLI(engine, default_system=getattr(args, "system", None))
    cli.temperature = getattr(args, "temperature", 0.7)
    cli.top_p = getattr(args, "top_p", 0.9)
    cli.top_k = getattr(args, "top_k", 40)
    cli.max_tokens = getattr(args, "max_tokens", 512)
    cli.repetition_penalty = getattr(args, "repetition_penalty", 1.1)

    if getattr(args, "prompt", None):
        cli.execute_turn(args.prompt)
    else:
        cli.run_repl()


def main():
    parser = argparse.ArgumentParser(description="X-Star Autonomous Brain CLI")
    parser.add_argument("--model", type=str, default=None, help="Path to X-Star model directory")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda:0, cpu)")
    parser.add_argument("--trust-remote-code", action="store_true", default=False, help="Allow executing remote code when loading custom models (default: False)")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: infer (NEW: Custom Inference CLI with /browser, /goal, /boost)
    p_infer = subparsers.add_parser("infer", help="Professional CLI inference with streaming, /browser, /goal, and /boost")
    p_infer.add_argument("--prompt", "-p", type=str, default=None, help="One-shot prompt to generate and exit")
    p_infer.add_argument("--adapter", type=str, default="dist/frontier_lora_adapter", help="Path to LoRA adapter weights")
    p_infer.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model precision")
    p_infer.add_argument("--no-brain", action="store_true", help="Disable HADL v5.2 Neuromorphic Spiking Brain hooks")
    p_infer.add_argument("--no-lora", action="store_true", help="Disable LoRA adapter weights")
    p_infer.add_argument("--boost", action="store_true", help="Initialize with Neuromorphic Deliberation Boost enabled")
    p_infer.add_argument("--boost-level", type=int, default=3, choices=[1, 2, 3, 4, 5], help="Deliberation boost intensity (1-5)")
    p_infer.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    p_infer.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    p_infer.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    p_infer.add_argument("--max-tokens", "-n", type=int, default=512, help="Maximum generation tokens")
    p_infer.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    p_infer.add_argument("--system", "-s", type=str, default=None, help="Custom system prompt")

    # Command: run
    p_run = subparsers.add_parser("run", help="Run single-shot prompt inference")
    p_run.add_argument("prompt", type=str, help="Input prompt text")
    p_run.add_argument("--max-tokens", type=int, default=256, help="Maximum generation tokens")
    p_run.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p_run.add_argument("--raw", action="store_true", help="Preserve raw thinking tags")

    # Command: chat
    p_chat = subparsers.add_parser("chat", help="Start interactive multi-turn chat")
    p_chat.add_argument("--max-tokens", type=int, default=256, help="Maximum generation tokens")
    p_chat.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    # Command: info
    subparsers.add_parser("info", help="Display architectural and hardware telemetry")

    # Command: bench
    p_bench = subparsers.add_parser("bench", help="Run benchmark audit suite")
    p_bench.add_argument("--suite", choices=["200q", "1000q"], default="200q", help="Benchmark suite scale")

    # Command: serve
    p_serve = subparsers.add_parser("serve", help="Launch OpenAI-compatible REST API server")
    p_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    p_serve.add_argument("--api-key", type=str, default=None, help="API key for authentication (required for non-loopback)")

    args = parser.parse_args()

    if args.command == "infer":
        cmd_infer(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "bench":
        cmd_bench(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
