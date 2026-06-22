# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.utils.radix_hash import (
    RADIX_HASH_SPACE,
    gpu_radix_row_hash,
)


def test_moss_generated_row_hash_keeps_im_end_and_hashes_audio_end() -> None:
    slot_id = 151656
    audio_end_id = 151653
    im_end_id = 151645
    rows = torch.tensor(
        [
            [slot_id, 1, 2],
            [slot_id, 1, 3],
            [audio_end_id, 1024, 1024],
            [im_end_id, 1024, 1024],
        ],
        dtype=torch.long,
    )

    keys = gpu_radix_row_hash(rows, rows[:, 0], im_end_id)

    assert int(keys[3]) == im_end_id
    assert int(keys[2]) != audio_end_id
    assert int(keys[0]) != int(keys[1])
    assert 0 <= int(keys[0]) < RADIX_HASH_SPACE
    assert 0 <= int(keys[1]) < RADIX_HASH_SPACE
    assert 0 <= int(keys[2]) < RADIX_HASH_SPACE
