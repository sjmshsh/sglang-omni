# SPDX-License-Identifier: Apache-2.0
"""Per-request state for the MOSS-TTS pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MossTTSState:
    # request / prompt
    text: str = ""
    reference_audio: Any | None = None
    reference_text: str | None = None
    reference_codes: list[list[int]] | None = None
    prompt_token_ids: list[list[int]] = field(default_factory=list)

    instruction: str | None = None
    tokens: int | None = None
    quality: str | None = None
    sound_event: str | None = None
    ambient_sound: str | None = None
    language: str | None = None

    n_vq: int = 32
    audio_vocab_size: int = 1024
    audio_pad_code: int = 1024
    sample_rate: int = 24000

    # generation params
    max_new_tokens: int = 2048
    text_temperature: float = 1.5
    text_top_p: float = 1.0
    text_top_k: int = 50
    audio_temperature: float = 1.7
    audio_top_p: float = 0.8
    audio_top_k: int = 25
    repetition_penalty: float = 1.0
    audio_repetition_penalty: float = 1.0
    seed: int | None = None

    # tts_engine output
    output_codes: list[list[int]] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "prompt_token_ids": self.prompt_token_ids,
            "n_vq": self.n_vq,
            "audio_vocab_size": self.audio_vocab_size,
            "audio_pad_code": self.audio_pad_code,
            "sample_rate": self.sample_rate,
            "max_new_tokens": self.max_new_tokens,
            "text_temperature": self.text_temperature,
            "text_top_p": self.text_top_p,
            "text_top_k": self.text_top_k,
            "audio_temperature": self.audio_temperature,
            "audio_top_p": self.audio_top_p,
            "audio_top_k": self.audio_top_k,
            "repetition_penalty": self.repetition_penalty,
            "audio_repetition_penalty": self.audio_repetition_penalty,
        }
        for key in (
            "reference_audio",
            "reference_text",
            "reference_codes",
            "instruction",
            "tokens",
            "quality",
            "sound_event",
            "ambient_sound",
            "language",
            "seed",
            "output_codes",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        for key in ("prompt_tokens", "completion_tokens", "engine_time_s"):
            value = getattr(self, key)
            if value:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MossTTSState":
        return cls(
            text=str(data.get("text") or ""),
            reference_audio=data.get("reference_audio"),
            reference_text=data.get("reference_text"),
            reference_codes=data.get("reference_codes"),
            prompt_token_ids=[list(row) for row in data.get("prompt_token_ids", [])],
            instruction=data.get("instruction"),
            tokens=data.get("tokens"),
            quality=data.get("quality"),
            sound_event=data.get("sound_event"),
            ambient_sound=data.get("ambient_sound"),
            language=data.get("language"),
            n_vq=int(data.get("n_vq", 32)),
            audio_vocab_size=int(data.get("audio_vocab_size", 1024)),
            audio_pad_code=int(data.get("audio_pad_code", 1024)),
            sample_rate=int(data.get("sample_rate", 24000)),
            max_new_tokens=int(data.get("max_new_tokens", 2048)),
            text_temperature=float(data.get("text_temperature", 1.5)),
            text_top_p=float(data.get("text_top_p", 1.0)),
            text_top_k=int(data.get("text_top_k", 50)),
            audio_temperature=float(data.get("audio_temperature", 1.7)),
            audio_top_p=float(data.get("audio_top_p", 0.8)),
            audio_top_k=int(data.get("audio_top_k", 25)),
            repetition_penalty=float(data.get("repetition_penalty", 1.0)),
            audio_repetition_penalty=float(
                data.get("audio_repetition_penalty", 1.0)
            ),
            seed=data.get("seed"),
            output_codes=data.get("output_codes"),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            engine_time_s=float(data.get("engine_time_s", 0.0)),
        )


__all__ = ["MossTTSState"]

