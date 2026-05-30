# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Delay pipeline."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

from sglang_omni.models.moss_tts.codec import split_moss_audio_segments
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.request_builders import (
    cleanup_prepared_moss_tts_request,
    make_moss_tts_scheduler_adapters,
    preprocess_moss_tts_payload,
    set_moss_tts_preprocessing_context,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_MOSS_TTS_INSTALL_HINT = (
    "MOSS-TTS support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer."
)


def load_state(payload: StagePayload) -> MossTTSState:
    return MossTTSState.from_dict(payload.data)


def store_state(payload: StagePayload, state: MossTTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def _resolve_stage_device(device: str, gpu_id: int | None) -> tuple[str, int]:
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    resolved_gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    return device, resolved_gpu_id


def _patch_moss_transformers_processor_compat() -> None:
    """Patch small Transformers API drifts used by MOSS remote processor code."""
    import transformers.configuration_utils as configuration_utils
    from transformers import PreTrainedModel, processing_utils

    if not hasattr(configuration_utils, "PreTrainedConfig"):
        configuration_utils.PreTrainedConfig = configuration_utils.PretrainedConfig

    auto_mapping = getattr(processing_utils, "AUTO_TO_BASE_CLASS_MAPPING", None)
    if isinstance(auto_mapping, dict):
        auto_mapping.setdefault("AutoModel", "PreTrainedModel")
        if not hasattr(processing_utils, "MODALITY_TO_BASE_CLASS_MAPPING"):
            processing_utils.MODALITY_TO_BASE_CLASS_MAPPING = auto_mapping

    # MOSS-Audio-Tokenizer is loaded through AutoModel and is a PreTrainedModel.
    # Transformers 4.57 otherwise rejects it in ProcessorMixin's optional
    # audio_tokenizer branch before the model can be moved to the vocoder stage.
    if hasattr(processing_utils, "PreTrainedAudioTokenizerBase"):
        processing_utils.PreTrainedAudioTokenizerBase = PreTrainedModel


def _load_moss_processor_class(checkpoint_dir: str) -> type:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    processor_config_path = os.path.join(checkpoint_dir, "processor_config.json")
    with open(processor_config_path, encoding="utf-8") as f:
        processor_config = json.load(f)

    class_ref = (processor_config.get("auto_map") or {}).get("AutoProcessor")
    if not class_ref:
        raise RuntimeError("MOSS-TTS processor_config.json lacks AutoProcessor map")

    processor_cls = get_class_from_dynamic_module(class_ref, checkpoint_dir)
    if list(getattr(processor_cls, "attributes", [])) == [
        "feature_extractor",
        "tokenizer",
    ]:
        processor_cls.attributes = ["tokenizer"]
    return processor_cls


def _normalize_moss_processor_config(processor: Any) -> None:
    model_config = getattr(processor, "model_config", None)
    if model_config is None:
        return
    for attr, default in (
        ("audio_start_token_id", 151652),
        ("audio_end_token_id", 151653),
        ("audio_assistant_gen_slot_token_id", 151656),
        ("audio_assistant_delay_slot_token_id", 151662),
        ("audio_pad_code", 1024),
        ("im_start_token_id", 151644),
        ("im_end_token_id", 151645),
        ("pad_token_id", 151643),
    ):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)


def _load_moss_processor(
    model_path: str,
    *,
    device: str = "cpu",
    dtype: str | torch.dtype = "float32",
) -> Any:
    try:
        _patch_moss_transformers_processor_compat()
    except ImportError as exc:
        raise RuntimeError(_MOSS_TTS_INSTALL_HINT) from exc

    checkpoint_dir = _resolve_checkpoint(model_path)
    logger.info("Loading MOSS-TTS processor from %s on %s", checkpoint_dir, device)
    try:
        processor_cls = _load_moss_processor_class(checkpoint_dir)
        processor = processor_cls.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(_MOSS_TTS_INSTALL_HINT) from exc

    _normalize_moss_processor_config(processor)
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is not None:
        if hasattr(audio_tokenizer, "eval"):
            audio_tokenizer.eval()
        if hasattr(audio_tokenizer, "to"):
            kwargs: dict[str, Any] = {"device": device}
            if device != "cpu":
                kwargs["dtype"] = _torch_dtype(dtype)
            audio_tokenizer.to(**kwargs)
    return processor


def _build_usage(state: MossTTSState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage = {
        "prompt_tokens": int(state.prompt_tokens),
        "completion_tokens": int(state.completion_tokens),
        "total_tokens": int(state.prompt_tokens + state.completion_tokens),
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


def _processor_sample_rate(processor: Any, fallback: int = 24000) -> int:
    model_config = getattr(processor, "model_config", None)
    audio_config = getattr(getattr(processor, "audio_tokenizer", None), "config", None)
    return int(
        getattr(model_config, "sampling_rate", 0)
        or getattr(audio_config, "sampling_rate", 0)
        or fallback
        or 24000
    )


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    processor = _load_moss_processor(model_path, device="cpu", dtype="float32")
    set_moss_tts_preprocessing_context(processor=processor)
    return SimpleScheduler(
        preprocess_moss_tts_payload,
        abort_callback=cleanup_prepared_moss_tts_request,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
) -> Any:
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    _, gpu_id = _resolve_stage_device(device, gpu_id)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=8192,
        dtype=dtype,
        cuda_graph_bs=[1, 2, 4, 8, 16],
        cuda_graph_max_bs=16,
        disable_cuda_graph=False,
        disable_overlap_schedule=True,
        enable_torch_compile=True,
        mem_fraction_static=0.70,
        max_prefill_tokens=8192,
        max_running_requests=16,
        sampling_backend="pytorch",
        torch_compile_max_bs=16,
        trust_remote_code=True,
    )

    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph:
        server_args.disable_cuda_graph = True

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
        model_arch_override="MossTTSDelaySGLangModel",
    )

    if want_cuda_graph:
        server_args.disable_cuda_graph = False

    model = model_worker.model_runner.model
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_moss_tts_scheduler_adapters(model=model)

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
        abort_callback=cleanup_prepared_moss_tts_request,
    )


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    device, _ = _resolve_stage_device(device, gpu_id)
    processor = _load_moss_processor(model_path, device=device, dtype=dtype)
    audio_pad_code = int(
        getattr(
            getattr(processor, "model_config", None),
            "audio_pad_code",
            1024,
        )
    )

    def _prepare_vocoder_item(
        payload: StagePayload,
    ) -> tuple[MossTTSState, torch.Tensor]:
        state = load_state(payload)
        if state.delayed_audio_codes is None:
            raise RuntimeError("MOSS-TTS vocoder requires delayed_audio_codes")
        delayed_codes = torch.as_tensor(state.delayed_audio_codes, dtype=torch.long)
        if delayed_codes.numel() == 0:
            raise RuntimeError("MOSS-TTS generated no delayed audio codes")
        return state, delayed_codes

    def _extract_audio_segments(
        state: MossTTSState,
        delayed_codes: torch.Tensor,
    ) -> list[torch.Tensor]:
        delayed_codes = delayed_codes.to(device=device, dtype=torch.long)
        return split_moss_audio_segments(
            delayed_codes,
            audio_pad_code=audio_pad_code,
        )

    def _decode_waveforms(segments: list[torch.Tensor]) -> list[torch.Tensor]:
        for segment in segments:
            if ((segment < 0) | (segment >= audio_pad_code)).any():
                raise RuntimeError(
                    "MOSS-TTS vocoder received an incomplete audio code segment; "
                    "refusing to decode pad/out-of-range code ids"
                )
        try:
            decoded = processor.decode_audio_codes(segments)
        except (RuntimeError, TypeError, ValueError):
            decoded = []
            for segment in segments:
                decoded.extend(processor.decode_audio_codes([segment]))
        if not decoded:
            raise RuntimeError("MOSS-TTS vocoder decoded no audio segments")
        return [
            torch.as_tensor(wav).detach().reshape(-1).to("cpu") for wav in decoded
        ]

    def _trim_assistant_prefix_audio(
        waveforms: list[torch.Tensor],
        segments: list[torch.Tensor],
        assistant_start_length: int,
    ) -> list[torch.Tensor]:
        if assistant_start_length <= 0 or not waveforms or not segments:
            return waveforms
        first_codes_length = int(segments[0].shape[0])
        if first_codes_length <= 0:
            return waveforms
        trim_ratio = max(
            0.0,
            min(float(assistant_start_length) / float(first_codes_length), 1.0),
        )
        if trim_ratio >= 1.0:
            return waveforms[1:]
        if trim_ratio <= 0.0:
            return waveforms
        trim_samples = int(waveforms[0].shape[-1] * trim_ratio)
        waveforms[0] = waveforms[0][trim_samples:]
        return waveforms

    def _decode_segments(
        segments: list[torch.Tensor],
        *,
        assistant_start_length: int,
    ) -> torch.Tensor:
        waveforms = _decode_waveforms(segments)
        waveforms = _trim_assistant_prefix_audio(
            waveforms,
            segments,
            assistant_start_length,
        )
        if not waveforms:
            raise RuntimeError("MOSS-TTS vocoder decoded no audio after trimming")
        return torch.cat(waveforms, dim=0)

    def _decode_audio(
        state: MossTTSState,
        delayed_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        segments = _extract_audio_segments(state, delayed_codes)
        waveform = _decode_segments(
            segments,
            assistant_start_length=int(state.assistant_start_length),
        )
        return waveform, _processor_sample_rate(processor, state.sample_rate)

    def _store_vocoder_result(
        payload: StagePayload,
        state: MossTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        audio_payload = audio_waveform_payload(wav, source_hint="MOSS-TTS")
        state.delayed_audio_codes = None
        state.sample_rate = int(sample_rate)
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _vocode(payload: StagePayload) -> StagePayload:
        state, delayed_codes = _prepare_vocoder_item(payload)
        wav, sample_rate = _decode_audio(state, delayed_codes)
        return _store_vocoder_result(payload, state, wav, sample_rate)

    def _vocode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        prepared = [_prepare_vocoder_item(payload) for payload in payloads]
        segment_groups: list[tuple[int, int]] = []
        all_segments: list[torch.Tensor] = []
        for state, delayed_codes in prepared:
            start = len(all_segments)
            segments = _extract_audio_segments(state, delayed_codes)
            all_segments.extend(segments)
            segment_groups.append((start, len(all_segments)))

        decoded = _decode_waveforms(all_segments) if all_segments else []
        if len(decoded) != len(all_segments):
            raise RuntimeError("MOSS-TTS vocoder decoded an unexpected segment count")

        results = []
        for payload, (state, _), (start, end) in zip(
            payloads, prepared, segment_groups
        ):
            if start == end:
                raise RuntimeError("MOSS-TTS vocoder decoded no audio segments")
            waveforms = _trim_assistant_prefix_audio(
                decoded[start:end],
                all_segments[start:end],
                int(state.assistant_start_length),
            )
            if not waveforms:
                raise RuntimeError("MOSS-TTS vocoder decoded no audio after trimming")
            waveform = torch.cat(waveforms, dim=0)
            sample_rate = _processor_sample_rate(processor, state.sample_rate)
            results.append(
                _store_vocoder_result(payload, state, waveform, sample_rate)
            )
        return results

    return SimpleScheduler(
        _vocode,
        batch_compute_fn=_vocode_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
