"""
X-Star OpenAI-Compatible Inference Server
=========================================
High-performance REST API powered by CustomInferenceEngine and FastAPI.
Provides drop-in compatibility for OpenAI SDK, LM Studio, Open WebUI, and curl,
with full Server-Sent Events (SSE) streaming and HADL v5.2 Neuromorphic Brain telemetry.
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import logging
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .engine import CustomInferenceEngine, EngineConfig
from .web_browser import WebBrowser

logger = logging.getLogger("xstar.server")


# Pydantic Request Models at module level
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "xstar-2b-autonomous-brain"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=512, alias="max_new_tokens")
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    repetition_penalty: Optional[float] = 1.1
    stream: Optional[bool] = False
    boost: Optional[bool] = None

class CompletionRequest(BaseModel):
    model: Optional[str] = "xstar-2b-autonomous-brain"
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    repetition_penalty: Optional[float] = 1.1
    stream: Optional[bool] = False

class BoostRequest(BaseModel):
    enabled: bool
    level: Optional[int] = 3

class BrowseRequest(BaseModel):
    query_or_url: str
    max_results: Optional[int] = 4

class VoiceRequest(BaseModel):
    text: Optional[str] = "Hello from X-Star endogenous vocal organ."
    duration_sec: Optional[float] = 1.5
    enable_self_listening: Optional[bool] = True
    save_path: Optional[str] = None

class ImagineRequest(BaseModel):
    prompt: Optional[str] = "3D geometric manifold imagination"


def create_app(
    engine: CustomInferenceEngine,
    api_key: Optional[str] = None,
):
    from fastapi import FastAPI, HTTPException, Security, Depends, status, Body
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse

    app = FastAPI(
        title="X-Star 2B Autonomous Brain OpenAI-Compatible API",
        version="5.7.0",
        description="Production-grade inference API for Qwen3.5-2B with HADL v5.7 Self-Evolving Multimodal Spiking Brain",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if api_key:
        security = HTTPBearer(auto_error=False)

        def auth_dependency(creds: Optional[HTTPAuthorizationCredentials] = Security(security)):
            if not creds or creds.credentials != api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing API key authorization bearer.",
                )
            return True
    else:
        def auth_dependency():
            return True

    @app.get("/v1/models", dependencies=[Depends(auth_dependency)])
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "xstar-2b-autonomous-brain",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "xstar-research",
                    "root": "qwen3_5_2b_hadl_v5_7",
                    "permission": [],
                }
            ],
        }

    @app.get("/v1/telemetry", dependencies=[Depends(auth_dependency)])
    def get_telemetry():
        return {
            "vram": engine.get_vram_usage(),
            "brain": engine.get_brain_telemetry(),
            "device": engine.device,
            "dtype": str(engine.torch_dtype),
        }

    @app.post("/v1/boost", dependencies=[Depends(auth_dependency)])
    def set_boost(req: BoostRequest = Body(...)):
        res = engine.set_boost(enabled=req.enabled, level=req.level or 3)
        return res

    @app.post("/v1/browser", dependencies=[Depends(auth_dependency)])
    def browse_web(req: BrowseRequest = Body(...)):
        if WebBrowser.is_url(req.query_or_url):
            data = WebBrowser.fetch_url(req.query_or_url)
            return {"type": "url", "result": data}
        else:
            results = WebBrowser.search(req.query_or_url, max_results=req.max_results or 4)
            return {"type": "search", "results": results}

    @app.post("/v1/voice", dependencies=[Depends(auth_dependency)])
    def synthesize_voice(req: VoiceRequest = Body(...)):
        res = engine.synthesize_voice(
            text=req.text,
            save_path=req.save_path,
            duration_sec=req.duration_sec or 1.5,
            enable_self_listening=bool(req.enable_self_listening),
        )
        return {
            "synthesis_telemetry": res.get("synthesis_telemetry", {}),
            "self_listening_telemetry": res.get("self_listening_telemetry", {}),
            "saved_path": res.get("saved_path"),
        }

    @app.post("/v1/imagine", dependencies=[Depends(auth_dependency)])
    def imagine_3d(req: ImagineRequest = Body(...)):
        telem = engine.get_3d_imagination_telemetry(text=req.prompt)
        return {"imagination_3d_telemetry": telem}

    @app.post("/v1/chat/completions", dependencies=[Depends(auth_dependency)])
    def chat_completions(req: ChatCompletionRequest = Body(...)):
        if req.boost is not None:
            engine.set_boost(enabled=req.boost)

        raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if req.stream:
            def event_generator():
                for chunk, metrics in engine.stream_chat(
                    messages=raw_messages,
                    max_new_tokens=req.max_tokens or 512,
                    temperature=req.temperature or 0.7,
                    top_p=req.top_p or 0.9,
                    top_k=req.top_k or 40,
                    repetition_penalty=req.repetition_penalty or 1.1,
                ):
                    if chunk:
                        payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "xstar-2b-autonomous-brain",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    if metrics:
                        done_payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "xstar-2b-autonomous-brain",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": metrics.prompt_tokens,
                                "completion_tokens": metrics.output_tokens,
                                "total_tokens": metrics.total_tokens,
                            },
                            "hadl_telemetry": {
                                "ttft_ms": metrics.ttft_ms,
                                "tokens_per_second": metrics.tokens_per_second,
                                "spikes_fired": metrics.spikes_fired,
                                "cognitive_state": metrics.cognitive_state,
                                "vram_mb": metrics.vram_allocated_mb,
                                "closed_loop_verified": metrics.closed_loop_verified,
                                "dopamine_sign": metrics.dopamine_sign,
                                "f0_mean_hz": metrics.f0_mean_hz,
                                "formants_hz": metrics.formants_hz,
                                "consistency_2d_to_3d": metrics.consistency_2d_to_3d,
                            },
                        }
                        yield f"data: {json.dumps(done_payload)}\n\n"
                        yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Synchronous execution
        content, metrics = engine.chat(
            messages=raw_messages,
            max_new_tokens=req.max_tokens or 512,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
            top_k=req.top_k or 40,
            repetition_penalty=req.repetition_penalty or 1.1,
        )

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": "xstar-2b-autonomous-brain",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.output_tokens,
                "total_tokens": metrics.total_tokens,
            },
            "hadl_telemetry": {
                "ttft_ms": metrics.ttft_ms,
                "tokens_per_second": metrics.tokens_per_second,
                "spikes_fired": metrics.spikes_fired,
                "cognitive_state": metrics.cognitive_state,
                "vram_mb": metrics.vram_allocated_mb,
                "closed_loop_verified": metrics.closed_loop_verified,
                "dopamine_sign": metrics.dopamine_sign,
                "f0_mean_hz": metrics.f0_mean_hz,
                "formants_hz": metrics.formants_hz,
                "consistency_2d_to_3d": metrics.consistency_2d_to_3d,
            },
        }

    @app.post("/v1/completions", dependencies=[Depends(auth_dependency)])
    def text_completions(req: CompletionRequest = Body(...)):
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if req.stream:
            def event_generator():
                for chunk, metrics in engine.stream_generate(
                    prompt=req.prompt,
                    max_new_tokens=req.max_tokens or 512,
                    temperature=req.temperature or 0.7,
                    top_p=req.top_p or 0.9,
                    top_k=req.top_k or 40,
                    repetition_penalty=req.repetition_penalty or 1.1,
                ):
                    if chunk:
                        payload = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": created,
                            "model": req.model or "xstar-2b-autonomous-brain",
                            "choices": [
                                {
                                    "text": chunk,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    if metrics:
                        done_payload = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": created,
                            "model": req.model or "xstar-2b-autonomous-brain",
                            "choices": [
                                {
                                    "text": "",
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": metrics.prompt_tokens,
                                "completion_tokens": metrics.output_tokens,
                                "total_tokens": metrics.total_tokens,
                            },
                            "hadl_telemetry": {
                                "ttft_ms": metrics.ttft_ms,
                                "tokens_per_second": metrics.tokens_per_second,
                                "spikes_fired": metrics.spikes_fired,
                                "cognitive_state": metrics.cognitive_state,
                                "vram_mb": metrics.vram_allocated_mb,
                            },
                        }
                        yield f"data: {json.dumps(done_payload)}\n\n"
                        yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        content, metrics = engine.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens or 512,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
            top_k=req.top_k or 40,
            repetition_penalty=req.repetition_penalty or 1.1,
        )

        return {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": req.model or "xstar-2b-autonomous-brain",
            "choices": [
                {
                    "text": content,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.output_tokens,
                "total_tokens": metrics.total_tokens,
            },
            "hadl_telemetry": {
                "ttft_ms": metrics.ttft_ms,
                "tokens_per_second": metrics.tokens_per_second,
                "spikes_fired": metrics.spikes_fired,
                "cognitive_state": metrics.cognitive_state,
                "vram_mb": metrics.vram_allocated_mb,
            },
        }

    return app


def start_server(
    model_path: str = "dist/staging_hf_fused",
    adapter_path: Optional[str] = "dist/frontier_lora_adapter",
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: Optional[str] = None,
    device: str = "auto",
    dtype: str = "bfloat16",
):
    """Initializes CustomInferenceEngine and launches the FastAPI server via Uvicorn."""
    import uvicorn

    config = EngineConfig(
        model_path=model_path,
        adapter_path=adapter_path,
        device=device,
        dtype=dtype,
    )
    engine = CustomInferenceEngine(config)
    engine.load()

    app = create_app(engine, api_key=api_key)
    print(f"\n[*] Starting X-Star OpenAI-compatible REST server on http://{host}:{port}...")
    print(f"[*] Compatible with LM Studio, Open WebUI, and OpenAI Python SDK.\n")
    uvicorn.run(app, host=host, port=port)
