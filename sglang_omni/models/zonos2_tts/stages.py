# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the ZONOS2 TTS pipeline.

Three stages:
1. Preprocessing: text normalization, byte tokenization, speaker embedding
2. TTS Engine: MoE AR backbone generating multi-codebook DAC codes
3. Vocoder: DAC decode -> PCM @ 44.1kHz
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.models.zonos2_tts.hf_config import (
    ensure_zonos2_hf_layout,
    load_zonos2_params,
    register_zonos2_hf_config,
)
from sglang_omni.models.zonos2_tts.payload_types import (
    ZONOS2_SAMPLE_RATE,
    Zonos2TTSState,
)
from sglang_omni.models.zonos2_tts.request_builders import (
    cleanup_prepared_zonos2_request,
    make_zonos2_scheduler_adapters,
    preprocess_zonos2_tts_payload,
    set_zonos2_preprocessing_context,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_ZONOS2_INSTALL_HINT = (
    "ZONOS2 TTS requires the 'dac' package for audio decoding. "
    "Install with: pip install descript-audio-codec"
)


def load_state(payload: StagePayload) -> Zonos2TTSState:
    return Zonos2TTSState.from_dict(payload.data)


def store_state(payload: StagePayload, state: Zonos2TTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def _load_zonos2_model_config(checkpoint_dir: str) -> Any:
    """Load ZONOS2 model config from checkpoint directory."""
    import json

    params_path = os.path.join(checkpoint_dir, "params.json")
    if os.path.isfile(params_path):
        return SimpleNamespace(**load_zonos2_params(checkpoint_dir))

    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
        return SimpleNamespace(**config_data)

    logger.warning("No config.json or params.json found in %s", checkpoint_dir)
    return None


def _load_speaker_model(device: str = "cpu") -> Any:
    """Load the speaker embedding model (Qwen3-voice-embedding).

    This is optional and only needed for voice cloning.
    Returns None if the model cannot be loaded.
    """
    try:
        from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding

        model = Qwen3SpeakerEmbedding(device=device)
        model.eval()
        logger.info("Loaded speaker embedding model on %s", device)
        return model
    except ImportError:
        logger.info(
            "zonos2 package is not importable; voice-clone speaker embedding "
            "extraction is disabled unless a speaker_embedding is supplied"
        )
        return None
    except Exception as exc:
        logger.info(
            "Speaker embedding model not available (voice cloning disabled): %s", exc
        )
        return None


def _build_usage(state: Zonos2TTSState) -> dict[str, Any] | None:
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


# ============================================================================
# Stage 1: Preprocessing
# ============================================================================


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str = "cpu",
    max_concurrency: int = 8,
    load_speaker_model: bool = False,
) -> SimpleScheduler:
    """Create the preprocessing stage executor.

    Loads model config and optionally the speaker embedding model.
    """
    checkpoint_dir = _resolve_checkpoint(model_path)
    model_config = _load_zonos2_model_config(checkpoint_dir)

    speaker_model = None
    if load_speaker_model:
        speaker_device = device if device != "cpu" else "cpu"
        speaker_model = _load_speaker_model(speaker_device)

    set_zonos2_preprocessing_context(
        model_config=model_config,
        speaker_model=speaker_model,
    )

    return SimpleScheduler(
        preprocess_zonos2_tts_payload,
        abort_callback=cleanup_prepared_zonos2_request,
        max_concurrency=max_concurrency,
    )


# ============================================================================
# Stage 2: TTS Engine (MoE AR backbone)
# ============================================================================


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    """Create the SGLang-based TTS engine executor.

    Uses OmniScheduler with CUDA graphs for efficient batched inference.
    """
    from sglang_omni.models.zonos2_tts.model_runner import Zonos2TTSModelRunner
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    register_zonos2_hf_config()
    checkpoint_dir = _resolve_checkpoint(model_path)
    sglang_checkpoint_dir = ensure_zonos2_hf_layout(checkpoint_dir)
    zonos2_config = _load_zonos2_model_config(checkpoint_dir)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    context_length = int(getattr(zonos2_config, "max_seqlen", 6144) or 6144)
    cuda_graph_max_bs = 256

    overrides: dict[str, Any] = {
        "dtype": dtype,
        "cuda_graph_bs": [1, 2, 4, *range(8, cuda_graph_max_bs + 1, 8)],
        "cuda_graph_max_bs": cuda_graph_max_bs,
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": False,
        "max_prefill_tokens": 8192,
        "max_running_requests": 256,
        "sampling_backend": "pytorch",
        "torch_compile_max_bs": 32,
        "trust_remote_code": True,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        sglang_checkpoint_dir,
        context_length=context_length,
        **overrides,
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
        model_arch_override="Zonos2SGLangModel",
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
    request_builder, result_adapter = make_zonos2_scheduler_adapters(model=model)

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=Zonos2TTSModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=cleanup_prepared_zonos2_request,
    )


# ============================================================================
# Stage 3: Vocoder (DAC decode -> PCM @ 44.1kHz)
# ============================================================================


def shear_up(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Remove delay pattern: column j shifted up by j rows.

    This is the inverse of shear() - it removes the delay pattern applied
    during generation to align all codebook outputs for DAC decoding.
    """
    H, W = x.shape[-2:]
    out = x.new_full(x.shape, pad_id)
    for j in range(W):
        if H > j:
            out[..., : H - j, j] = x[..., j:, j]
    return out


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    """Create the DAC vocoder stage executor.

    Decodes multi-codebook audio codes to PCM waveform at 44.1kHz.
    """
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"

    # Lazy-load DAC model
    _dac_model = None

    def _get_dac():
        nonlocal _dac_model
        if _dac_model is None:
            try:
                import dac as dac_module

                _dac_model = (
                    dac_module.DAC.load(
                        dac_module.utils.download(model_type="44khz")
                    )
                    .eval()
                    .to(device)
                )
                logger.info("Loaded DAC 44kHz vocoder on %s", device)
            except ImportError as exc:
                raise RuntimeError(_ZONOS2_INSTALL_HINT) from exc
        return _dac_model

    # Pre-load DAC at startup
    _get_dac()

    def _vocode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        audio_codes = state.audio_codes
        if audio_codes is None:
            raise RuntimeError("ZONOS2 vocoder requires audio_codes")

        if not isinstance(audio_codes, torch.Tensor):
            audio_codes = torch.as_tensor(audio_codes, dtype=torch.long)

        if audio_codes.numel() == 0:
            raise RuntimeError("ZONOS2 generated no audio codes")

        # Remove delay pattern
        n_codebooks = int(state.n_codebooks)
        audio_pad_id = int(state.audio_pad_id)
        codes = audio_codes.to(dtype=torch.long)

        # Apply shear_up to remove delay
        codes = shear_up(codes, audio_pad_id)

        # Trim to EOS frame if detected
        if state.eos_frame is not None and state.eos_frame >= 0:
            codes = codes[: max(0, state.eos_frame)]

        if codes.numel() == 0:
            # Empty audio - return silence
            wav = torch.zeros(1, dtype=torch.float32)
        else:
            # Clamp to valid codebook range
            codes = torch.clamp(codes, max=int(state.codebook_size) - 1)

            # DAC expects (batch, codebooks, seq_len)
            codes = codes.unsqueeze(0).permute(0, 2, 1).contiguous()
            codes = codes.to(device=device, dtype=torch.long)

            dac = _get_dac()
            with torch.no_grad(), torch.inference_mode():
                z = dac.quantizer.from_codes(codes)[0]
                wav = dac.decode(z).float().squeeze(0).squeeze(0).cpu()

        # Build output payload
        audio_payload = audio_waveform_payload(wav, source_hint="ZONOS2")
        state.audio_codes = None  # Free memory
        state.sample_rate = ZONOS2_SAMPLE_RATE
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = ZONOS2_SAMPLE_RATE
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _vocode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        return [_vocode(payload) for payload in payloads]

    return SimpleScheduler(
        _vocode,
        batch_compute_fn=_vocode_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
