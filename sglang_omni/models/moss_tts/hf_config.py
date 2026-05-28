# SPDX-License-Identifier: Apache-2.0
"""Transformers config shim for OpenMOSS MOSS-TTS checkpoints."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3 import Qwen3Config


def _build_qwen3_config(raw: Any) -> Qwen3Config:
    if isinstance(raw, Qwen3Config):
        return raw
    if isinstance(raw, PretrainedConfig):
        return Qwen3Config(**raw.to_dict())
    return Qwen3Config(**dict(raw or {}))


class MossTTSDelayConfig(PretrainedConfig):
    """Config for the MOSS-TTS delay-pattern Qwen3 backbone."""

    model_type = "moss_tts_delay"
    is_composition = True
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        language_config: dict[str, Any] | PretrainedConfig | None = None,
        initializer_range: float = 0.02,
        n_vq: int = 32,
        pad_token_id: int = 151643,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        audio_vocab_size: int = 1024,
        audio_user_slot_token_id: int = 151654,
        audio_assistant_gen_slot_token_id: int = 151656,
        audio_assistant_delay_slot_token_id: int = 151662,
        audio_start_token_id: int = 151652,
        audio_end_token_id: int = 151653,
        audio_pad_code: int = 1024,
        sampling_rate: int = 24000,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=kwargs.pop("bos_token_id", None),
            eos_token_id=kwargs.pop("eos_token_id", im_end_token_id),
            **kwargs,
        )
        self.language_config = _build_qwen3_config(language_config)
        self.initializer_range = initializer_range
        self.n_vq = int(n_vq)
        self.audio_vocab_size = int(audio_vocab_size)
        self.audio_user_slot_token_id = int(audio_user_slot_token_id)
        self.audio_assistant_gen_slot_token_id = int(
            audio_assistant_gen_slot_token_id
        )
        self.audio_assistant_delay_slot_token_id = int(
            audio_assistant_delay_slot_token_id
        )
        self.audio_start_token_id = int(audio_start_token_id)
        self.audio_end_token_id = int(audio_end_token_id)
        self.audio_pad_code = int(audio_pad_code)
        self.sampling_rate = int(sampling_rate)
        self.im_start_token_id = int(im_start_token_id)
        self.im_end_token_id = int(im_end_token_id)

        self.hidden_size = int(self.language_config.hidden_size)
        self.vocab_size = int(self.language_config.vocab_size)
        self.channels = self.n_vq + 1
        self.vocab_size_list = [self.vocab_size] + [
            self.audio_vocab_size + 1
        ] * self.n_vq
        self.pad_token = [int(self.pad_token_id)] + [self.audio_pad_code] * self.n_vq
        self.speech_pad_token_id = self.audio_pad_code

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        del decoder
        return self.language_config

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["language_config"] = self.language_config.to_dict()
        output["n_vq"] = self.n_vq
        output["audio_vocab_size"] = self.audio_vocab_size
        output["audio_user_slot_token_id"] = self.audio_user_slot_token_id
        output["audio_assistant_gen_slot_token_id"] = (
            self.audio_assistant_gen_slot_token_id
        )
        output["audio_assistant_delay_slot_token_id"] = (
            self.audio_assistant_delay_slot_token_id
        )
        output["audio_start_token_id"] = self.audio_start_token_id
        output["audio_end_token_id"] = self.audio_end_token_id
        output["audio_pad_code"] = self.audio_pad_code
        output["sampling_rate"] = self.sampling_rate
        output["im_start_token_id"] = self.im_start_token_id
        output["im_end_token_id"] = self.im_end_token_id
        output["channels"] = self.channels
        output["vocab_size_list"] = list(self.vocab_size_list)
        output["pad_token"] = list(self.pad_token)
        return output


def register_moss_tts_hf_config() -> None:
    """Register local config so HF remote-code import is not required."""

    try:
        AutoConfig.register(MossTTSDelayConfig.model_type, MossTTSDelayConfig)
    except ValueError:
        pass


__all__ = ["MossTTSDelayConfig", "register_moss_tts_hf_config"]

