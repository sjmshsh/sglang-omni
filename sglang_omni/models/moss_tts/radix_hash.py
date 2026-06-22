# SPDX-License-Identifier: Apache-2.0
"""Capture-safe GPU radix-key hash for MOSS-TTS generated frames."""

from __future__ import annotations

import torch

# <|endoftext|> = 151643 opens the special/control id band. Generated radix
# keys fold strictly below it; models keep their own stop id raw so existing
# eos/vocab-boundary detection still fires.
RADIX_HASH_SPACE = 151643

# Polynomial-hash constants.
#
# _MOD is the Mersenne prime 2**31 - 1. With the accumulator and every channel
# value reduced below _MOD (< 2**31) and _BASE < _MOD, each Horner step
# ``acc * _BASE + v`` stays below 2**31 * 2**31 = 2**62, comfortably inside
# signed int64 (max 2**63 - 1). So the int64 ops never overflow and the result
# is bit-reproducible on CPU and GPU -- no implementation-defined wraparound.
#
# _BASE is a large prime well below _MOD. Folding each channel in as a power of
# _BASE (Horner) makes the hash order-sensitive (a channel permutation changes
# the key) and spreads neighbours (a single-channel +/-1 changes the key by a
# power of _BASE mod _MOD). Both constants are arbitrary fixed primes chosen
# only for these size/spread properties; the generated-row key space is private
# to the radix cache, so the exact values carry no on-disk/ABI contract.
_MOD = 2147483647  # 2**31 - 1, Mersenne prime M31
_BASE = 1000000007  # 1e9 + 7, prime, < _MOD


def poly_row_hash(rows: torch.Tensor) -> torch.Tensor:
    """Fixed-coefficient polynomial hash of each row, in ``[0, _MOD)``.

    ``rows`` is ``[B, C]`` integer. Returns ``[B]`` int64 on ``rows.device``.
    Pure elementwise int64 torch ops (mul / add / remainder) over a static
    channel count -- no host sync, CUDA-graph capturable.
    """
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2-D [B, C], got shape {tuple(rows.shape)}")
    work = rows.to(torch.int64)
    acc = torch.zeros(work.shape[0], dtype=torch.int64, device=work.device)
    # Static trip count (one frame = a fixed number of channels): the loop
    # unrolls into a fixed op sequence at capture time.
    for channel in range(work.shape[1]):
        # Reduce defensively in case a caller passes a raw id >= _MOD.
        value = torch.remainder(work[:, channel], _MOD)
        acc = torch.remainder(acc * _BASE + value, _MOD)
    return acc


def gpu_radix_row_hash(
    rows: torch.Tensor,
    next_text: torch.Tensor,
    end_id: int,
    *,
    hash_space: int = RADIX_HASH_SPACE,
) -> torch.Tensor:
    """Capture-safe radix token ids for a batch of generated rows.

    ``rows`` is ``[B, C]`` int64 (text channel + RVQ codes); ``next_text`` is
    ``[B]``. Continuing rows get a key in ``[0, hash_space)``; rows whose text
    channel equals ``end_id`` keep that raw id so existing eos detection fires.
    device/dtype follow ``rows``.
    """
    folded = torch.remainder(poly_row_hash(rows), hash_space)
    return torch.where(next_text == end_id, next_text.to(torch.int64), folded)
