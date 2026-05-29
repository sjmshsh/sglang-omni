# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace

import torch

from sglang_omni.models.moss_tts.audio_codec import load_audio_to_24k
from sglang_omni.models.moss_tts.model import MossTTSDelayModel
from sglang_omni.models.moss_tts.request_builders import (
    apply_de_delay_pattern,
    apply_delay_pattern,
    build_moss_tts_state,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.openai_api import build_speech_generate_request
from sglang_omni.serve.protocol import CreateSpeechRequest


def _tiny_wav_data_uri() -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 16)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def test_load_audio_to_24k_accepts_data_uri_reference() -> None:
    wav, sr = load_audio_to_24k(_tiny_wav_data_uri())

    assert sr == 24000
    assert wav.dtype == torch.float32
    assert wav.numel() == 16


def test_moss_tts_state_accepts_audio_data_and_token_count() -> None:
    payload = StagePayload(
        request_id="moss-1",
        request=OmniRequest(
            inputs={
                "text": "[pause 0.5s] ni3 hao3 /h@'loU/",
                "references": [{"audio_data": _tiny_wav_data_uri(), "text": "ref"}],
            },
            params={"temperature": 0},
            metadata={"tts_params": {"token_count": 125, "language": "English"}},
        ),
        data={},
    )

    state = build_moss_tts_state(payload)

    assert state.reference_audio.startswith("data:audio/wav;base64,")
    assert state.tokens == 125
    assert state.language == "English"
    assert state.text == "[pause 0.5s] ni3 hao3 /h@'loU/"
    assert state.text_temperature == 0
    assert state.audio_temperature == 0


def test_openai_speech_request_passes_moss_audio_data_reference() -> None:
    audio_data = _tiny_wav_data_uri()
    request = CreateSpeechRequest(
        input="hello",
        audio_data=audio_data,
        ref_text="reference transcript",
    )

    generate_request = build_speech_generate_request(request, "default-model")

    assert generate_request.prompt == {
        "text": "hello",
        "references": [
            {
                "audio_data": audio_data,
                "text": "reference transcript",
            }
        ],
    }
    assert generate_request.metadata["tts_params"]["audio_data"] == audio_data
    assert generate_request.metadata["tts_params"]["ref_text"] == "reference transcript"


def test_delay_pattern_round_trip() -> None:
    codes = torch.tensor([[1, 10, 100], [2, 20, 200]], dtype=torch.long)
    delayed = apply_delay_pattern(codes, pad_code=1024)

    assert delayed.tolist() == [
        [1, 1024, 1024],
        [2, 10, 1024],
        [1024, 20, 100],
        [1024, 1024, 200],
    ]
    assert apply_de_delay_pattern(delayed).tolist() == codes.tolist()


def test_decode_input_buffer_preallocates_graph_capacity() -> None:
    model = MossTTSDelayModel.__new__(MossTTSDelayModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(channels=3)
    model.register_buffer(
        "_decode_input_ids",
        torch.empty(0, 3, dtype=torch.long),
        persistent=False,
    )
    model._decode_input_capacity = 8

    model.prepare_decode_inputs(torch.ones(1, 3, dtype=torch.long))
    first_ptr = model._decode_input_ids.data_ptr()

    assert model._decode_input_ids.shape == (8, 3)

    model.prepare_decode_inputs(torch.ones(4, 3, dtype=torch.long) * 2)

    assert model._decode_input_ids.data_ptr() == first_ptr
    assert model._decode_input_ids[:4].tolist() == [[2, 2, 2]] * 4
    assert model._decode_input_batch_size == 4
    assert model._decode_input_staged is True


def test_moss_tts_engine_enables_cuda_graph_and_compile_by_default(monkeypatch) -> None:
    from sglang_omni.models.moss_tts import stages

    captured: dict[str, object] = {}

    monkeypatch.setattr(stages, "register_moss_tts_hf_config", lambda: None)
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        stages,
        "_load_config",
        lambda _checkpoint_dir: SimpleNamespace(
            language_config=SimpleNamespace(max_position_embeddings=40960)
        ),
    )

    def fake_build_sglang_server_args(checkpoint_dir, context_length, **overrides):
        server_args = SimpleNamespace(
            disable_cuda_graph=overrides["disable_cuda_graph"],
            disable_overlap_schedule=overrides["disable_overlap_schedule"],
            enable_torch_compile=overrides["enable_torch_compile"],
            torch_compile_max_bs=overrides["torch_compile_max_bs"],
        )
        captured["checkpoint_dir"] = checkpoint_dir
        captured["context_length"] = context_length
        captured["overrides"] = overrides
        captured["server_args"] = server_args
        return server_args

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        captured["gpu_id"] = gpu_id
        captured["infrastructure_kwargs"] = kwargs
        model_worker = SimpleNamespace(
            model_runner=SimpleNamespace(model=SimpleNamespace())
        )
        return (
            model_worker,
            object(),
            object(),
            object(),
            object(),
            object(),
            SimpleNamespace(),
        )

    class FakeOutputProcessor:
        def __init__(self, **kwargs) -> None:
            captured["output_processor_kwargs"] = kwargs

    class FakeModelRunner:
        def __init__(self, model_worker, output_proc) -> None:
            captured["model_runner_args"] = (model_worker, output_proc)

    class FakeScheduler:
        def __init__(self, **kwargs) -> None:
            captured["scheduler_kwargs"] = kwargs
            self.server_args = kwargs["server_args"]

    monkeypatch.setattr(
        stages, "build_sglang_server_args", fake_build_sglang_server_args
    )
    monkeypatch.setattr(
        stages, "create_sglang_infrastructure", fake_create_sglang_infrastructure
    )
    monkeypatch.setattr(stages, "SGLangOutputProcessor", FakeOutputProcessor)
    monkeypatch.setattr(stages, "MossTTSModelRunner", FakeModelRunner)
    monkeypatch.setattr(
        stages,
        "make_moss_tts_scheduler_adapters",
        lambda *args, **kwargs: (lambda payload: payload, lambda data: data),
    )
    monkeypatch.setattr(stages, "OmniScheduler", FakeScheduler)

    scheduler = stages.create_sglang_tts_engine_executor("moss-model", device="cuda:0")

    overrides = captured["overrides"]
    assert isinstance(overrides, dict)
    assert captured["checkpoint_dir"] == "moss-model"
    assert captured["context_length"] == 40960
    assert captured["gpu_id"] == 0
    assert overrides["disable_cuda_graph"] is False
    assert overrides["disable_overlap_schedule"] is True
    assert overrides["enable_torch_compile"] is True
    assert overrides["cuda_graph_max_bs"] == 16
    assert overrides["torch_compile_max_bs"] == 16
    assert captured["infrastructure_kwargs"] == {
        "model_arch_override": "MossTTSDelayModel"
    }
    assert scheduler.server_args.disable_cuda_graph is False
    assert scheduler.server_args.enable_torch_compile is True
