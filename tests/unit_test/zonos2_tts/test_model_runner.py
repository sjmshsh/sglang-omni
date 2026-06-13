# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import collections
from types import SimpleNamespace

import torch

from sglang_omni.models.zonos2_tts.model_runner import Zonos2TTSModelRunner


def test_sampling_filters_are_per_row() -> None:
    scores = torch.tensor(
        [
            [10.0, 9.0, 8.0],
            [10.0, 9.0, 8.0],
        ]
    )
    filtered = Zonos2TTSModelRunner._apply_top_k_scores(
        scores,
        torch.tensor([1, 2]),
    )

    assert torch.isneginf(filtered[0, 1])
    assert filtered[1, 1].item() == 9.0

    probs = torch.tensor(
        [
            [0.60, 0.30, 0.10],
            [0.60, 0.30, 0.10],
        ]
    )
    top_p = Zonos2TTSModelRunner._apply_top_p_rows(
        probs,
        torch.tensor([0.50, 0.95]),
    )
    min_p = Zonos2TTSModelRunner._apply_min_p_rows(
        probs,
        torch.tensor([0.50, 0.0]),
    )

    assert top_p[0, 1].item() == 0.0
    assert top_p[1, 1].item() > 0.0
    assert min_p[0, 2].item() == 0.0
    assert min_p[1, 2].item() > 0.0


def test_prefill_replays_generated_rows_after_retraction() -> None:
    runner = object.__new__(Zonos2TTSModelRunner)
    runner.model = SimpleNamespace(
        hidden_size=2,
        dtype=torch.float32,
        _prepare_multi_modal_inputs=lambda rows: rows[:, :2].to(torch.float32),
    )
    data = SimpleNamespace(
        req=SimpleNamespace(extend_input_len=2, prefix_indices=[0], rid="req"),
        prompt_rows=torch.tensor([[10, 11], [12, 13]], dtype=torch.long),
        output_rows=[torch.tensor([14, 15], dtype=torch.long)],
        pending_feedback_queue=collections.deque([torch.ones(2)]),
        speaker_token_position=-1,
        speaker_embedding=None,
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(2, dtype=torch.long))

    embeds = runner._build_prefill_input_embeds(
        forward_batch,
        [SimpleNamespace(data=data)],
    )

    assert embeds.tolist() == [[12.0, 13.0], [14.0, 15.0]]
    assert list(data.pending_feedback_queue) == []
