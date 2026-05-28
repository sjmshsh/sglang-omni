# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.moss_tts"


class MossTTSPipelineConfig(PipelineConfig):
    """4-stage MOSS-TTS pipeline.

    preprocessing -> audio_encoder -> tts_engine -> vocoder
    """

    architecture: ClassVar[str] = "MossTTSDelayModel"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossTTSDelay",
        "MossTTSDelayWithCodec",
        "MossTTSDdelayWithCodec",
    )

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="audio_encoder",
        ),
        StageConfig(
            name="audio_encoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_audio_encoder_executor",
            factory_args={"device": "cuda"},
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"device": "cuda", "max_new_tokens": 2048},
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"device": "cuda"},
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = MossTTSPipelineConfig
