# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS pipeline."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang_omni.models.moss_tts.audio_codec import (
    DEFAULT_MOSS_AUDIO_TOKENIZER,
    MossAudioTokenizerCodec,
    resolve_checkpoint,
)
from sglang_omni.models.moss_tts.hf_config import (
    MossTTSDelayConfig,
    register_moss_tts_hf_config,
)
from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.request_builders import (
    MossTTSPromptBuilder,
    build_moss_tts_state,
    extract_moss_tts_audio_segments,
    make_moss_tts_scheduler_adapters,
    to_codes_TN,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_MAX_REF_AUDIO_SEC = 120


def load_state(payload: StagePayload) -> MossTTSState:
    return MossTTSState.from_dict(payload.data)


def store_state(payload: StagePayload, state: MossTTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _load_config(checkpoint_dir: str) -> MossTTSDelayConfig:
    register_moss_tts_hf_config()
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=False)
    if not isinstance(config, MossTTSDelayConfig):
        config = MossTTSDelayConfig(**config.to_dict())
    return config


def _build_usage(state: MossTTSState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage = {
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "total_tokens": state.prompt_tokens + state.completion_tokens,
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
) -> ThreadedSimpleScheduler:
    del model_path

    def _preprocess(payload: StagePayload) -> StagePayload:
        state = build_moss_tts_state(payload)
        payload.data = state.to_dict()
        return payload

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_audio_encoder_executor(
    model_path: str,
    *,
    codec_path: str = DEFAULT_MOSS_AUDIO_TOKENIZER,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    register_moss_tts_hf_config()
    checkpoint_dir = resolve_checkpoint(model_path)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    config = _load_config(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    prompt_builder = MossTTSPromptBuilder(tokenizer, config)
    codec = MossAudioTokenizerCodec.from_pretrained(
        codec_path,
        device=device,
        dtype=dtype,
    )

    def _encode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        state.n_vq = int(config.n_vq)
        state.audio_vocab_size = int(config.audio_vocab_size)
        state.audio_pad_code = int(config.audio_pad_code)
        state.sample_rate = int(config.sampling_rate)

        reference_codes: list[torch.Tensor] = []
        if state.reference_codes is not None:
            codes = to_codes_TN(state.reference_codes, config.n_vq)
            if codes is not None:
                reference_codes.append(codes)
        elif state.reference_audio is not None:
            codes = codec.encode_reference(state.reference_audio, n_vq=config.n_vq)
            if codes.shape[0] > _MAX_REF_AUDIO_SEC * 13:
                raise ValueError(
                    f"reference_audio is too long ({codes.shape[0]} codec frames); "
                    f"cap at about {_MAX_REF_AUDIO_SEC}s."
                )
            reference_codes.append(codes)

        state.prompt_token_ids = prompt_builder.build_prompt_ids(
            state,
            reference_codes,
        )
        state.reference_audio = None
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(
        _encode,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_new_tokens: int | None = 2048,
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    register_moss_tts_hf_config()
    checkpoint_dir = resolve_checkpoint(model_path)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    config = _load_config(checkpoint_dir)
    context_length = int(
        getattr(config.language_config, "max_position_embeddings", 40960)
    )

    overrides: dict[str, Any] = {
        "dtype": dtype,
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": True,
        "mem_fraction_static": 0.85,
        "max_prefill_tokens": min(context_length, 16384),
        "max_running_requests": 16,
        "sampling_backend": "pytorch",
        "torch_compile_max_bs": 16,
        "trust_remote_code": False,
        "cuda_graph_max_bs": 16,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=context_length,
        **overrides,
    )

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        model_arch_override="MossTTSDelayModel",
    )

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    request_builder, result_adapter = make_moss_tts_scheduler_adapters(
        config,
        max_new_tokens_cap=max_new_tokens,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=MossTTSModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    codec_path: str = DEFAULT_MOSS_AUDIO_TOKENIZER,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
    max_batch_size: int = 4,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    del model_path
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    codec = MossAudioTokenizerCodec.from_pretrained(
        codec_path,
        device=device,
        dtype=dtype,
    )
    sample_rate = int(codec.sample_rate)

    def _prepare_vocoder_item(
        payload: StagePayload,
    ) -> tuple[MossTTSState, list[torch.Tensor]]:
        state = load_state(payload)
        if not state.output_codes:
            return state, []
        rows = torch.tensor(state.output_codes, dtype=torch.long)
        segments = extract_moss_tts_audio_segments(
            rows,
            n_vq=state.n_vq,
            audio_pad_code=state.audio_pad_code,
        )
        return state, segments

    def _store_result(
        payload: StagePayload,
        state: MossTTSState,
        waveform: torch.Tensor | None,
    ) -> StagePayload:
        payload = store_state(payload, state)
        if waveform is None:
            waveform = torch.empty(0, dtype=torch.float32)
        payload.data.update(audio_waveform_payload(waveform, source_hint="MOSS-TTS"))
        payload.data["sample_rate"] = sample_rate
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _vocode(payload: StagePayload) -> StagePayload:
        state, segments = _prepare_vocoder_item(payload)
        if not segments:
            return _store_result(payload, state, None)
        wavs = codec.decode_batch(segments)
        waveform = torch.cat(wavs, dim=-1) if len(wavs) > 1 else wavs[0]
        return _store_result(payload, state, waveform)

    def _vocode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        items = [_prepare_vocoder_item(payload) for payload in payloads]
        flat_segments: list[torch.Tensor] = []
        spans: list[tuple[int, int]] = []
        for _, segments in items:
            start = len(flat_segments)
            flat_segments.extend(segments)
            spans.append((start, len(flat_segments)))

        decoded = codec.decode_batch(flat_segments) if flat_segments else []
        outputs: list[StagePayload] = []
        for payload, (state, _segments), (start, end) in zip(payloads, items, spans):
            wavs = decoded[start:end]
            waveform = None
            if wavs:
                waveform = torch.cat(wavs, dim=-1) if len(wavs) > 1 else wavs[0]
            outputs.append(_store_result(payload, state, waveform))
        return outputs

    return SimpleScheduler(
        _vocode,
        batch_compute_fn=_vocode_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


__all__ = [
    "create_audio_encoder_executor",
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
