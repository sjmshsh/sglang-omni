# SPDX-License-Identifier: Apache-2.0
"""MOSS-Audio-Tokenizer facade used by MOSS-TTS stages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.preprocessing.base import _is_url
from sglang_omni.preprocessing.resource_connector import global_http_connection

DEFAULT_MOSS_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer"


def resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    return snapshot_download(checkpoint)


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, str(dtype))


def _loudness_normalize(
    wav: torch.Tensor,
    *,
    target_dbfs: float = -20.0,
    gain_range: tuple[float, float] = (-3.0, 3.0),
) -> torch.Tensor:
    wav = wav.to(torch.float32)
    if wav.numel() == 0:
        return wav
    current_dbfs = 10.0 * torch.log10(torch.mean(wav**2) + 1e-9)
    gain = float(target_dbfs - current_dbfs)
    gain = max(gain_range[0], min(gain, gain_range[1]))
    return wav * (10.0 ** (gain / 20.0))


def load_audio_to_24k(source: Any) -> tuple[torch.Tensor, int]:
    """Load audio input as mono 24 kHz float tensor [T]."""

    if isinstance(source, torch.Tensor):
        wav = source.detach().to(torch.float32).cpu()
        if wav.ndim > 1:
            wav = wav.reshape(-1, wav.shape[-1]).mean(dim=0)
        return wav.reshape(-1), 24000
    if isinstance(source, np.ndarray):
        wav = torch.from_numpy(np.asarray(source, dtype=np.float32))
        if wav.ndim > 1:
            wav = wav.reshape(-1, wav.shape[-1]).mean(dim=0)
        return wav.reshape(-1), 24000
    if isinstance(source, (list, tuple)):
        return torch.tensor(source, dtype=torch.float32).reshape(-1), 24000

    io = AudioMediaIO(target_sr=24000)

    def _load_path_or_url(src: str | Path) -> tuple[torch.Tensor, int]:
        if isinstance(src, str) and _is_url(src):
            response = global_http_connection.get_sync_client().get(src)
            response.raise_for_status()
            audio, sr = io.load_bytes(response.content)
        else:
            audio, sr = io.load_file(Path(src))
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(-1)
        return wav, int(sr)

    if isinstance(source, (str, Path)):
        return _load_path_or_url(source)
    if not isinstance(source, dict):
        raise TypeError(f"Unsupported audio reference type: {type(source).__name__}")

    nested = source.get("audio")
    if nested is not None and nested is not source:
        return load_audio_to_24k(nested)
    if "audio_path" in source or "path" in source or "ref_audio" in source:
        return _load_path_or_url(
            source.get("audio_path") or source.get("path") or source["ref_audio"]
        )
    if "bytes" in source:
        raw = source["bytes"]
        if isinstance(raw, str):
            import base64

            raw = base64.b64decode(raw)
        audio, sr = io.load_bytes(raw)
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(-1)
        return wav, int(sr)
    data = source.get("base64") or source.get("data")
    if data is None:
        raise ValueError(
            "audio reference dict must include path, bytes, base64, or data"
        )
    audio, sr = io.load_base64(source.get("media_type", "audio/wav"), data)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(-1), int(sr)


class MossAudioTokenizerCodec:
    """Frozen wrapper around the official MOSS-Audio-Tokenizer."""

    SAMPLE_RATE = 24000

    def __init__(self, model: torch.nn.Module, *, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.sample_rate = int(getattr(model, "sampling_rate", self.SAMPLE_RATE))

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path = DEFAULT_MOSS_AUDIO_TOKENIZER,
        *,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype = torch.float32,
    ) -> "MossAudioTokenizerCodec":
        device = torch.device(device)
        torch_dtype = _torch_dtype(dtype)
        model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).eval()
        model = model.to(device=device)
        for param in model.parameters():
            param.requires_grad_(False)
        return cls(model, device=device)

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    @torch.no_grad()
    def encode_reference(self, source: Any, *, n_vq: int = 32) -> torch.Tensor:
        wav, _sample_rate = load_audio_to_24k(source)
        wav = _loudness_normalize(wav).to(self._model_device())
        if hasattr(self.model, "batch_encode"):
            enc = self.model.batch_encode([wav], num_quantizers=n_vq)
            audio_codes = enc.audio_codes
            audio_lengths = enc.audio_codes_lengths
        else:
            input_values = wav.view(1, 1, -1)
            padding_mask = torch.ones(
                (1, wav.shape[-1]), device=wav.device, dtype=torch.bool
            )
            enc = self.model.encode(
                input_values,
                padding_mask=padding_mask,
                num_quantizers=n_vq,
                return_dict=True,
            )
            audio_codes = enc.audio_codes
            audio_lengths = enc.audio_codes_lengths
        if audio_codes is None or audio_lengths is None:
            raise RuntimeError("MOSS-Audio-Tokenizer encode returned empty outputs")
        length = int(audio_lengths[0].item())
        return (
            audio_codes[:n_vq, 0, :length]
            .transpose(0, 1)
            .contiguous()
            .to(torch.long)
            .cpu()
        )

    @torch.no_grad()
    def decode(self, codes_TN: torch.Tensor) -> torch.Tensor:
        if codes_TN.ndim != 2:
            raise ValueError(f"codes must be [T, N], got {tuple(codes_TN.shape)}")
        if codes_TN.numel() == 0:
            return torch.empty(0, dtype=torch.float32)
        device = self._model_device()
        audio_codes = (
            codes_TN.transpose(0, 1)
            .unsqueeze(1)
            .contiguous()
            .to(device=device, dtype=torch.long)
        )
        padding_mask = torch.ones(
            (1, codes_TN.shape[0]), device=device, dtype=torch.bool
        )
        dec = self.model.decode(
            audio_codes,
            padding_mask=padding_mask,
            return_dict=True,
            chunk_duration=8,
        )
        audio = dec.audio
        lengths = dec.audio_lengths
        if audio is None or lengths is None:
            raise RuntimeError("MOSS-Audio-Tokenizer decode returned empty outputs")
        return audio[0, 0, : int(lengths[0].item())].to(torch.float32).cpu()

    @torch.no_grad()
    def decode_batch(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        if not codes_list:
            return []
        if len(codes_list) == 1:
            return [self.decode(codes_list[0])]

        device = self._model_device()
        n_vq = int(codes_list[0].shape[1])
        max_t = max(int(codes.shape[0]) for codes in codes_list)
        audio_codes = torch.zeros(
            (n_vq, len(codes_list), max_t),
            device=device,
            dtype=torch.long,
        )
        padding_mask = torch.zeros(
            (len(codes_list), max_t),
            device=device,
            dtype=torch.bool,
        )
        for idx, codes in enumerate(codes_list):
            if codes.ndim != 2 or int(codes.shape[1]) != n_vq:
                raise ValueError(
                    "All MOSS audio-code tensors must be [T, N] with the same N"
                )
            t = int(codes.shape[0])
            audio_codes[:, idx, :t] = codes.transpose(0, 1).to(
                device=device,
                dtype=torch.long,
            )
            padding_mask[idx, :t] = True

        dec = self.model.decode(
            audio_codes,
            padding_mask=padding_mask,
            return_dict=True,
            chunk_duration=8,
        )
        audio = dec.audio
        lengths = dec.audio_lengths
        if audio is None or lengths is None:
            raise RuntimeError("MOSS-Audio-Tokenizer decode returned empty outputs")
        return [
            audio[idx, 0, : int(lengths[idx].item())].to(torch.float32).cpu()
            for idx in range(len(codes_list))
        ]


__all__ = [
    "DEFAULT_MOSS_AUDIO_TOKENIZER",
    "MossAudioTokenizerCodec",
    "load_audio_to_24k",
    "resolve_checkpoint",
]
