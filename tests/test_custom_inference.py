"""
Unit and Integration Tests for X-Star Custom Inference Engine & CLI Suite
========================================================================
Validates:
1. EngineConfig, GenerationMetrics, and CustomInferenceEngine initialization.
2. Neuromorphic Deliberation Boost (/boost) parameter adjustments & telemetry.
3. WebBrowser search, URL cleaning, and context formatting (/browser).
4. CognitiveTriadRunner 4-phase goal execution (/goal -> /plan -> /boost -> /goal audit).
5. InteractiveCLI command routing, parameter tuning, transcript saving & loading.
6. OpenAI-compatible REST API endpoints (/v1/models, /v1/chat/completions, /v1/telemetry, /v1/boost, /v1/browser).
"""

import os
import json
import tempfile
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from xstar.inference import (
    CustomInferenceEngine,
    EngineConfig,
    GenerationMetrics,
    WebBrowser,
    CognitiveTriadRunner,
    InteractiveCLI,
    create_app,
)
from dual_loop.neuromorphic.spiking_gate import attach_hadl_v5_spiking_brain


class DummyTransformer(nn.Module):
    """Minimal dummy causal LM for fast deterministic unit testing."""
    def __init__(self, vocab_size=100, hidden_size=64, num_layers=8):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.config.num_hidden_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return MagicMock(logits=logits)

    def generate(self, input_ids, max_new_tokens=5, **kwargs):
        # Deterministic generation of incremental tokens
        curr = input_ids
        for _ in range(max_new_tokens):
            next_token = torch.tensor([[42]], device=input_ids.device)
            curr = torch.cat([curr, next_token], dim=-1)
        return curr


def test_engine_config_defaults():
    cfg = EngineConfig()
    assert cfg.model_path == "dist/staging_hf_fused"
    assert cfg.adapter_path == "dist/frontier_lora_adapter"
    assert cfg.device == "auto"
    assert cfg.dtype == "bfloat16"
    assert cfg.enable_brain is True
    assert cfg.target_layer_idx == 7
    assert cfg.boost_mode is False


def test_custom_inference_engine_boost_toggle():
    dummy_model = DummyTransformer(hidden_size=64, num_layers=10)
    wrapped = attach_hadl_v5_spiking_brain(dummy_model, target_layer_idx=2)
    
    cfg = EngineConfig(enable_brain=True, device="cpu", dtype="float32")
    engine = CustomInferenceEngine(cfg)
    engine.model = wrapped
    engine.raw_model = dummy_model
    engine.is_loaded = True

    # Test initial state
    assert engine._boost_active is False
    telem_init = engine.get_brain_telemetry()
    assert telem_init["boost_active"] is False

    # Activate boost level 3
    res_boost = engine.set_boost(enabled=True, level=3)
    assert res_boost["boost_active"] is True
    assert res_boost["boost_level"] == 3
    assert res_boost["spike_v_th"] == 0.60
    assert engine.model.brain.spiking_gate.v_th.item() == pytest.approx(0.60, rel=1e-4)

    # Activate boost level 5 (maximum overdrive)
    res_boost_5 = engine.set_boost(enabled=True, level=5)
    assert res_boost_5["boost_level"] == 5
    assert res_boost_5["spike_v_th"] == 0.30
    assert engine.model.brain.spiking_gate.v_th.item() == pytest.approx(0.30, rel=1e-4)

    # Deactivate boost
    res_off = engine.set_boost(enabled=False)
    assert res_off["boost_active"] is False
    assert res_off["spike_v_th"] == 1.0
    assert engine.model.brain.spiking_gate.v_th.item() == pytest.approx(1.0, rel=1e-4)


def test_web_browser_url_cleaning_and_detection():
    assert WebBrowser.is_url("https://example.com/test") is True
    assert WebBrowser.is_url("http://localhost:8000") is True
    assert WebBrowser.is_url("Not a url at all") is False
    assert WebBrowser.is_url("quantum computing") is False

    html_doc = """
    <html>
        <head><title>Test Page</title><script>console.log('secret');</script></head>
        <body>
            <h1>Welcome to X-Star</h1>
            <p>Neuromorphic <b>spiking</b> intelligence architecture.</p>
            <style>body { color: red; }</style>
        </body>
    </html>
    """
    cleaned = WebBrowser.clean_html(html_doc)
    assert "console.log" not in cleaned
    assert "color: red" not in cleaned
    assert "Welcome to X-Star" in cleaned
    assert "Neuromorphic spiking intelligence architecture." in cleaned


def test_web_browser_search_and_context_formatting():
    results = [
        {"title": "Quantum supremacy", "snippet": "Computational demonstration using qubits.", "url": "https://en.wikipedia.org/wiki/Quantum_supremacy", "source": "Wikipedia"},
        {"title": "Neuromorphic engineering", "snippet": "Analog, digital, and mixed-mode VLSI systems.", "url": "https://en.wikipedia.org/wiki/Neuromorphic_engineering", "source": "Wikipedia"}
    ]
    ctx = WebBrowser.format_search_context("quantum neuromorphic", results)
    assert "=== WEB BROWSING EVIDENCE: 'quantum neuromorphic' ===" in ctx
    assert "Quantum supremacy" in ctx
    assert "Computational demonstration using qubits." in ctx
    assert "=== END WEB BROWSING EVIDENCE ===" in ctx

    page_data = {"title": "Paper", "url": "https://arxiv.org/abs/123", "content": "Abstract of study."}
    page_ctx = WebBrowser.format_url_context(page_data)
    assert "=== BROWSED WEBPAGE: Paper ===" in page_ctx
    assert "https://arxiv.org/abs/123" in page_ctx


def test_cognitive_triad_runner_mocked():
    mock_engine = MagicMock(spec=CustomInferenceEngine)
    mock_engine._boost_active = False
    mock_engine._boost_level = 3
    mock_engine.set_boost.return_value = {"boost_active": True, "boost_level": 4}

    # Phase 1, 2, 4 use generate
    mock_engine.generate.side_effect = [
        ("1. Exact Literals: int, list\n2. Invariant: Sort order must be ascending\n3. Edge cases: Empty list", GenerationMetrics()),
        ("Mode: System 2\nMilestones:\n1. Inspect list\n2. Apply quicksort", GenerationMetrics()),
        ("Status: PASSED\nCompliance: 10/10\nVerified successfully.", GenerationMetrics()),
    ]
    # Phase 3 uses stream_generate
    mock_engine.stream_generate.return_value = iter([
        ("def sort_asc(items):\n", None),
        ("    return sorted(items)\n", GenerationMetrics(output_tokens=15, tokens_per_second=40.0)),
    ])

    phases_recorded = []
    def on_phase(p, d):
        phases_recorded.append(p)

    chunks_recorded = []
    def on_chunk(c):
        chunks_recorded.append(c)

    trace = CognitiveTriadRunner.execute_goal(
        engine=mock_engine,
        goal_prompt="Sort a list in ascending order",
        on_phase_change=on_phase,
        on_stream_chunk=on_chunk,
    )

    assert len(phases_recorded) == 4
    assert "Phase 1: /goal" in phases_recorded[0]
    assert "Phase 2: /plan" in phases_recorded[1]
    assert "Phase 3: /boost" in phases_recorded[2]
    assert "Phase 4: /goal Gate" in phases_recorded[3]

    assert "def sort_asc(items):" in trace["final_solution"]
    assert trace["passed_gate"] is True
    assert "spec_extraction" in trace["phases"]
    assert "action_plan" in trace["phases"]
    assert "audit" in trace["phases"]


def test_interactive_cli_slash_commands():
    mock_engine = MagicMock(spec=CustomInferenceEngine)
    mock_engine.device = "cpu"
    mock_engine.torch_dtype = torch.float32
    mock_engine._boost_active = False
    mock_engine._boost_level = 3
    mock_engine.config = EngineConfig()
    mock_engine.get_vram_usage.return_value = {"allocated_gb": 0.5, "total_gb": 8.0, "percent_used": 6.2}
    mock_engine.get_brain_telemetry.return_value = {"boost_active": False, "peak_cognitive_state": "ROUTINE_CALM"}

    cli = InteractiveCLI(mock_engine)

    # Test /params
    cli.handle_params("temperature=0.2 max_tokens=1024 top_p=0.85")
    assert cli.temperature == 0.2
    assert cli.max_tokens == 1024
    assert cli.top_p == 0.85

    # Test /system
    cli.handle_system("New system instructions.")
    assert cli.system_prompt == "New system instructions."

    # Test /boost
    mock_engine.set_boost.return_value = {"boost_active": True, "boost_level": 4, "spike_v_th": 0.45}
    cli.handle_boost("4")
    mock_engine.set_boost.assert_called_with(enabled=True, level=4)

    # Test /save and /load
    cli.messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        temp_path = tf.name

    try:
        cli.handle_save(temp_path)
        assert os.path.exists(temp_path)

        # Clear and reload
        cli.handle_clear()
        assert len(cli.messages) == 0

        cli.handle_load(temp_path)
        assert len(cli.messages) == 2
        assert cli.messages[0]["content"] == "hi"
        assert cli.messages[1]["content"] == "hello"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_openai_compatible_server_endpoints():
    from fastapi.testclient import TestClient

    mock_engine = MagicMock(spec=CustomInferenceEngine)
    mock_engine.device = "cuda:0"
    mock_engine.torch_dtype = torch.bfloat16
    mock_engine._boost_active = False
    mock_engine._boost_level = 3
    mock_engine.get_vram_usage.return_value = {"allocated_mb": 3800.0, "allocated_gb": 3.71, "percent_used": 46.4}
    mock_engine.get_brain_telemetry.return_value = {
        "total_generation_spikes": 2,
        "generation_mean_spike_rate": 0.05,
        "peak_cognitive_state": "ROUTINE_CALM",
        "boost_active": False,
    }
    mock_engine.chat.return_value = (
        "Quantum computing harnesses superposition and entanglement.",
        GenerationMetrics(
            prompt_tokens=10,
            output_tokens=8,
            total_tokens=18,
            ttft_ms=120.0,
            tokens_per_second=35.0,
            spikes_fired=2,
            cognitive_state="ROUTINE_CALM",
        ),
    )
    mock_engine.set_boost.return_value = {"boost_active": True, "boost_level": 3, "spike_v_th": 0.60}

    app = create_app(mock_engine, api_key="secret-xstar-key")
    client = TestClient(app)

    # 1. Test unauthorized access
    unauth_resp = client.get("/v1/models")
    assert unauth_resp.status_code == 401

    auth_headers = {"Authorization": "Bearer secret-xstar-key"}

    # 2. Test /v1/models
    models_resp = client.get("/v1/models", headers=auth_headers)
    assert models_resp.status_code == 200
    m_data = models_resp.json()
    assert m_data["object"] == "list"
    assert m_data["data"][0]["id"] == "xstar-2b-autonomous-brain"

    # 3. Test /v1/telemetry
    telem_resp = client.get("/v1/telemetry", headers=auth_headers)
    assert telem_resp.status_code == 200
    t_data = telem_resp.json()
    assert "vram" in t_data
    assert "brain" in t_data

    # 4. Test /v1/boost
    boost_resp = client.post("/v1/boost", headers=auth_headers, json={"enabled": True, "level": 3})
    assert boost_resp.status_code == 200
    assert boost_resp.json()["boost_active"] is True

    # 5. Test /v1/chat/completions (non-streaming)
    chat_resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "messages": [{"role": "user", "content": "What is quantum computing?"}],
            "temperature": 0.7,
            "stream": False,
        },
    )
    assert chat_resp.status_code == 200
    c_data = chat_resp.json()
    assert c_data["choices"][0]["message"]["content"] == "Quantum computing harnesses superposition and entanglement."
    assert c_data["usage"]["total_tokens"] == 18
    assert c_data["hadl_telemetry"]["spikes_fired"] == 2

    # 6. Test /v1/chat/completions (streaming SSE)
    mock_engine.stream_chat.return_value = iter([
        ("Superposition ", None),
        ("enables parallelism.", GenerationMetrics(prompt_tokens=5, output_tokens=4, total_tokens=9, tokens_per_second=38.0)),
    ])
    stream_resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "messages": [{"role": "user", "content": "Explain in brief."}],
            "stream": True,
        },
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    stream_content = stream_resp.text
    assert "data: " in stream_content
    assert "Superposition " in stream_content
    assert "[DONE]" in stream_content

    # 7. Test /v1/completions (sync text completion)
    mock_engine.generate.return_value = (
        "Text completion result.",
        GenerationMetrics(prompt_tokens=4, output_tokens=4, total_tokens=8, tokens_per_second=30.0),
    )
    comp_resp = client.post(
        "/v1/completions",
        headers=auth_headers,
        json={"prompt": "Complete this", "stream": False},
    )
    assert comp_resp.status_code == 200
    comp_data = comp_resp.json()
    assert comp_data["object"] == "text_completion"
    assert comp_data["choices"][0]["text"] == "Text completion result."

    # 8. Test /v1/completions (streaming SSE text completion)
    mock_engine.stream_generate.return_value = iter([
        ("Completed ", None),
        ("stream.", GenerationMetrics(prompt_tokens=4, output_tokens=2, total_tokens=6)),
    ])
    stream_comp_resp = client.post(
        "/v1/completions",
        headers=auth_headers,
        json={"prompt": "Complete streaming", "stream": True},
    )
    assert stream_comp_resp.status_code == 200
    assert "text/event-stream" in stream_comp_resp.headers["content-type"]
    assert "Completed " in stream_comp_resp.text
    assert "[DONE]" in stream_comp_resp.text


def test_server_without_api_key():
    from fastapi.testclient import TestClient

    mock_engine = MagicMock(spec=CustomInferenceEngine)
    mock_engine.device = "cpu"
    mock_engine.torch_dtype = torch.float32
    mock_engine.generate.return_value = ("No auth output", GenerationMetrics())

    # Verify create_app does not crash with ForwardRef error when api_key=None
    app = create_app(mock_engine, api_key=None)
    client = TestClient(app)

    models_resp = client.get("/v1/models")
    assert models_resp.status_code == 200

    comp_resp = client.post("/v1/completions", json={"prompt": "Hello"})
    assert comp_resp.status_code == 200
    assert comp_resp.json()["choices"][0]["text"] == "No auth output"


def test_cognitive_triad_gate_hardening_against_false_positives():
    mock_engine = MagicMock(spec=CustomInferenceEngine)
    mock_engine._boost_active = False
    mock_engine._boost_level = 3
    mock_engine.set_boost.return_value = {"boost_active": True, "boost_level": 4}

    # Case A: Audit says NOT PASSED -> must evaluate to passed_gate = False
    mock_engine.generate.side_effect = [
        ("Bounds", GenerationMetrics()),
        ("Plan", GenerationMetrics()),
        ("Status: NOT PASSED\nCompliance: 2/10\nDefects found.", GenerationMetrics()),
    ]
    mock_engine.stream_generate.return_value = iter([("solution", GenerationMetrics())])

    trace_fail = CognitiveTriadRunner.execute_goal(mock_engine, "Test fail goal")
    assert trace_fail["passed_gate"] is False

    # Case B: Audit says DEFECT_FLAGGED -> must evaluate to passed_gate = False
    mock_engine.generate.side_effect = [
        ("Bounds", GenerationMetrics()),
        ("Plan", GenerationMetrics()),
        ("Status: DEFECT_FLAGGED (PASSED preliminary, but failed invariant)", GenerationMetrics()),
    ]
    mock_engine.stream_generate.return_value = iter([("solution", GenerationMetrics())])

    trace_defect = CognitiveTriadRunner.execute_goal(mock_engine, "Test defect goal")
    assert trace_defect["passed_gate"] is False


def test_browser_grounding_injects_into_messages_and_stream_chat():
    mock_engine = MagicMock(spec=CustomInferenceEngine)
    mock_engine.stream_chat.return_value = iter([("Grounded response", GenerationMetrics(output_tokens=5))])

    cli = InteractiveCLI(mock_engine)

    with patch("xstar.inference.web_browser.WebBrowser.search") as mock_search:
        mock_search.return_value = [
            {"title": "Spiking Gate", "snippet": "LIF Neuromorphic unit.", "url": "https://example.com/lif", "source": "Web"}
        ]
        cli.handle_browser("spiking gate mechanism")

    # Verify stream_chat received messages containing the web search context
    assert mock_engine.stream_chat.called
    call_args = mock_engine.stream_chat.call_args[1]
    messages = call_args["messages"]
    assert len(messages) >= 1
    user_msg = messages[0]
    assert user_msg["role"] == "user"
    assert "Spiking Gate" in user_msg["content"]
    assert "spiking gate mechanism" in user_msg["content"]
    # And assistant message was saved
    assert any(m["role"] == "assistant" for m in cli.messages)


def test_interactive_cli_params_spacing_and_markdown_save():
    mock_engine = MagicMock(spec=CustomInferenceEngine)
    cli = InteractiveCLI(mock_engine)

    # Test spaces around equals
    cli.handle_params("temperature = 0.35 max_tokens = 768")
    assert cli.temperature == 0.35
    assert cli.max_tokens == 768

    # Test key value without equals
    cli.handle_params("top_p 0.8 top_k 25")
    assert cli.top_p == 0.8
    assert cli.top_k == 25

    # Test markdown save
    cli.messages = [
        {"role": "user", "content": "What is HADL?"},
        {"role": "assistant", "content": "HADL is Hybrid Asymmetric Dual Loop."},
    ]
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
        md_path = tf.name

    try:
        cli.handle_save(md_path)
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "# X-Star Conversation Transcript" in content
        assert "### User" in content
        assert "What is HADL?" in content
        assert "### Assistant" in content
    finally:
        if os.path.exists(md_path):
            os.remove(md_path)


def test_engine_stream_generate_stop_strings_and_attention_mask():
    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_tok.return_value = MagicMock(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
    )
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 2

    engine = CustomInferenceEngine(EngineConfig(enable_brain=False, device="cpu", dtype="float32"))
    engine.model = mock_model
    engine.tokenizer = mock_tok
    engine.is_loaded = True

    def fake_generate(**kwargs):
        streamer = kwargs.get("streamer")
        if streamer:
            streamer.end()

    mock_model.generate.side_effect = fake_generate

    # Run stream_generate with stop_strings
    gen_iter = engine.stream_generate("test prompt", stop_strings=["\n\n", "User:"])
    # Consume generator
    list(gen_iter)

    assert mock_model.generate.called
    kwargs = mock_model.generate.call_args[1]
    assert "attention_mask" in kwargs
    assert "stop_strings" in kwargs
    assert kwargs["stop_strings"] == ["\n\n", "User:"]
    assert "tokenizer" in kwargs
