# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.models.zonos2_tts.stages import (
    _resolve_codec_device,
    _slice_batched_dac_waveforms,
)
from sglang_omni.models.zonos2_tts.config import Zonos2TTSPipelineConfig


def test_slice_batched_dac_waveforms_removes_padding_tail() -> None:
    wavs = torch.arange(2 * 100, dtype=torch.float32).view(2, 100)

    trimmed = _slice_batched_dac_waveforms(wavs, [10, 5])

    assert trimmed[0].shape == (100,)
    assert trimmed[1].shape == (50,)
    torch.testing.assert_close(trimmed[0], wavs[0])
    torch.testing.assert_close(trimmed[1], wavs[1, :50])


def test_slice_batched_dac_waveforms_accepts_dac_channel_dim() -> None:
    wavs = torch.arange(2 * 1 * 80, dtype=torch.float32).view(2, 1, 80)

    trimmed = _slice_batched_dac_waveforms(wavs, [4, 2])

    assert [tuple(wav.shape) for wav in trimmed] == [(80,), (40,)]
    torch.testing.assert_close(trimmed[1], wavs[1, 0, :40])


def test_resolve_codec_device_prefers_launcher_gpu_id() -> None:
    assert _resolve_codec_device("cuda:1", 0) == "cuda:0"
    assert _resolve_codec_device("cuda:0", 1) == "cuda:1"


def test_default_pipeline_uses_async_talker_and_second_gpu_vocoder() -> None:
    config = Zonos2TTSPipelineConfig(model_path="Zyphra/ZONOS2")
    stages = {stage.name: stage for stage in config.stages}

    assert stages["tts_engine"].factory_args["enable_async_decode"] is True
    assert stages["tts_engine"].gpu == 0
    assert stages["vocoder"].process == "vocoder"
    assert stages["vocoder"].factory_args["device"] == "cuda:1"
    assert stages["vocoder"].gpu == 1
