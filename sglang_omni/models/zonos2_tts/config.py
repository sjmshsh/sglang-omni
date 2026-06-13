# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for ZONOS2 TTS."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.zonos2_tts"


def _stages(*, codec_device: str = "cuda:0") -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={
                "device": "cpu",
                "load_speaker_model": False,
            },
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"gpu_id": 0, "dtype": "bfloat16"},
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"device": codec_device},
            gpu=0,
            terminal=True,
        ),
    ]


class Zonos2TTSPipelineConfig(PipelineConfig):
    """ZONOS2 TTS pipeline: preprocessing -> MoE AR engine -> DAC vocoder."""

    architecture: ClassVar[str] = "Zonos2ForCausalLM"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "Zonos2SGLangModel",
        "Zonos2TTS",
        "Zonos2ForConditionalGeneration",
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0")
    )


EntryClass = Zonos2TTSPipelineConfig
