# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 TTS model runner for OmniScheduler.

Handles multi-codebook sampling with:
- Per-codebook repetition penalty
- Top-k / top-p / min-p sampling
- EOS detection with frame alignment (delayed codebook pattern)
- Speaker embedding injection
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.zonos2_tts.radix_hash import gpu_radix_row_hash
from sglang_omni.scheduling.types import RequestOutput


class Zonos2TTSModelRunner(ModelRunner):
    """Samples ZONOS2 multi-codebook audio codes and manages EOS detection."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._pending_rows: torch.Tensor | None = None
        self._pending_embeds: torch.Tensor | None = None

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Inject 2D prompt embeddings for prefill."""
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch, requests
        )
        return None

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        """Write decode-step input embeddings from the feedback queue."""
        del is_lookahead, schedule_batch
        self._write_decode_input_embedding(forward_batch, requests)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_step(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_step(result, forward_batch, schedule_batch, requests)

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        """Build input embeddings for prefill from 2D prompt rows."""
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            rows = data.prompt_rows
            if rows is None:
                raise RuntimeError("ZONOS2 prefill requires prompt_rows")
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            current_rows = rows[prefix_len : prefix_len + req_len]
            embeds = self.model._prepare_multi_modal_inputs(
                current_rows.to(device=forward_batch.input_ids.device)
            )
            speaker_position = int(getattr(data, "speaker_token_position", -1))
            speaker_embedding = getattr(data, "speaker_embedding", None)
            if (
                speaker_embedding is not None
                and speaker_position >= prefix_len
                and speaker_position < prefix_len + req_len
                and hasattr(self.model, "project_speaker_embedding")
            ):
                local_idx = speaker_position - prefix_len
                embeds[local_idx] = self.model.project_speaker_embedding(
                    speaker_embedding
                ).to(device=embeds.device, dtype=embeds.dtype)
            pieces.append(embeds)
        if not pieces:
            return torch.empty(
                (0, self.model.hidden_size),
                device=forward_batch.input_ids.device,
                dtype=self.model.dtype,
            )
        result = torch.cat(pieces, dim=0)
        return result.to(device=forward_batch.input_ids.device, dtype=self.model.dtype)

    def _write_decode_input_embedding(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        """Write the next decode step's input embedding from feedback queue."""
        batch_size = len(requests)
        if batch_size == 0:
            return
        embedding = self.model._decode_input_embedding
        weight = embedding.weight

        rows = []
        for sched_req in requests:
            queue = sched_req.data.pending_feedback_queue
            if not queue:
                rows.append(torch.zeros(self.model.hidden_size, device=weight.device))
                continue
            if hasattr(queue, "popleft"):
                rows.append(queue.popleft())
            else:
                rows.append(queue.pop(0))

        stacked = torch.stack(rows, dim=0).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            weight[:batch_size].copy_(stacked)

        row_ids = torch.arange(
            batch_size, dtype=torch.long, device=forward_batch.input_ids.device
        )
        forward_batch.input_ids[:batch_size].copy_(row_ids)

    def _collect_step(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Sample multi-codebook tokens and prepare next-step embeddings."""
        hidden_states = self._get_hidden_states(result)
        if hidden_states is None or not requests:
            return

        # Compute multi-codebook logits
        logits = self.model.compute_multi_codebook_logits(hidden_states)
        # logits: [batch_size, n_codebooks, audio_vocab]

        datas = [sched_req.data for sched_req in requests]
        sampled_codes = self._sample_codebooks(logits, datas)
        # sampled_codes: [batch_size, n_codebooks]

        # Build full frame: audio codes + text padding
        batch_size = sampled_codes.shape[0]
        text_pad = int(self.model.text_vocab) if self.model.text_vocab is not None else 0
        frame_width = self.model.frame_width
        rows = torch.full(
            (batch_size, frame_width),
            self.model.config.audio_pad_id,
            dtype=torch.long,
            device=sampled_codes.device,
        )
        rows[:, : self.model.n_codebooks] = sampled_codes
        if self.model.text_vocab is not None:
            rows[:, self.model.n_codebooks] = text_pad

        # Detect EOS: any codebook emitting eoa_id
        eoa_id = int(self.model.config.eoa_id)

        # ZONOS2 uses delayed EOS (countdown pattern): when any codebook emits
        # eoa_id, we still need n_codebooks + 1 more steps to align the delayed
        # codebook outputs. So we do NOT signal EOS to SGLang here. Instead,
        # all frames get a hash-based radix key, and EOS is handled manually
        # in post_process_outputs when the countdown finishes.
        # Pass eoa_mask=False for all frames so gpu_radix_row_hash always
        # returns a hash value, never the raw eoa_id.
        no_eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=rows.device)

        # Use GPU radix row hash for radix-cache-safe token ids.
        # This hashes the full multi-channel row so that a radix match implies
        # identical audio content, enabling prefix sharing across requests.
        next_token_ids = gpu_radix_row_hash(rows, no_eos_mask, eoa_id)
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids

        # Compute embeddings for next decode step
        embeds = self.model._prepare_multi_modal_inputs(rows)

        self._pending_rows = rows
        self._pending_embeds = embeds.detach()

    def _get_hidden_states(self, result: Any) -> torch.Tensor | None:
        """Extract hidden states from model output."""
        logits_output = result.logits_output
        hidden_states = getattr(logits_output, "hidden_states", None)
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim == 3:
                hidden_states = hidden_states[:, -1, :]
            return hidden_states
        return None

    def _sample_codebooks(
        self,
        logits: torch.Tensor,
        datas: list,
    ) -> torch.Tensor:
        """Sample from multi-codebook logits with per-request parameters.

        Args:
            logits: [batch_size, n_codebooks, audio_vocab]
            datas: Per-request data objects with sampling params

        Returns:
            sampled: [batch_size, n_codebooks] token ids
        """
        batch_size, n_codebooks, vocab_size = logits.shape
        device = logits.device

        # Gather per-request sampling parameters
        temperatures = torch.tensor(
            [float(d.temperature) for d in datas], dtype=torch.float32, device=device
        )
        top_ks = torch.tensor(
            [int(d.top_k) for d in datas], dtype=torch.long, device=device
        )
        top_ps = torch.tensor(
            [float(d.top_p) for d in datas], dtype=torch.float32, device=device
        )
        min_ps = torch.tensor(
            [float(d.min_p) for d in datas], dtype=torch.float32, device=device
        )
        rep_penalties = torch.tensor(
            [float(d.repetition_penalty) for d in datas],
            dtype=torch.float32,
            device=device,
        )

        # Apply repetition penalty
        if bool((rep_penalties > 1.0).any()):
            self._apply_repetition_penalty(logits, datas, rep_penalties)

        # Mask out audio_pad_id and beyond (only valid codes are 0..codebook_size-1 + eoa)
        # eoa_id (1024) is valid, audio_pad_id (1025) is not
        codebook_size = int(self.model.codebook_size)
        audio_pad_id = int(self.model.config.audio_pad_id)
        if audio_pad_id < vocab_size:
            logits[:, :, audio_pad_id:] = float("-inf")

        # Temperature scaling
        safe_temp = temperatures.clamp(min=1e-8).view(batch_size, 1, 1)
        scaled_logits = logits / safe_temp

        # Flatten for sampling: [batch * n_codebooks, vocab]
        flat_logits = scaled_logits.view(batch_size * n_codebooks, vocab_size)

        # Apply top-k (use min across batch for efficiency)
        min_top_k = int(top_ks.min().item())
        if 0 < min_top_k < vocab_size:
            values, _ = torch.topk(flat_logits, min_top_k, dim=-1)
            kth = values[..., -1].unsqueeze(-1)
            flat_logits = flat_logits.masked_fill(flat_logits < kth, float("-inf"))

        # Softmax to probabilities
        probs = F.softmax(flat_logits, dim=-1)

        # Apply top-p
        max_top_p = float(top_ps.max().item())
        if 0.0 < max_top_p < 1.0:
            probs = self._apply_top_p(probs, max_top_p)

        # Apply min-p
        max_min_p = float(min_ps.max().item())
        if max_min_p > 0.0:
            probs = self._apply_min_p(probs, max_min_p)

        # Handle all-zero rows (fallback to greedy)
        invalid = probs.sum(dim=-1) <= 0
        if bool(invalid.any()):
            greedy = flat_logits.argmax(dim=-1)
            fallback = torch.zeros_like(probs)
            fallback.scatter_(-1, greedy.unsqueeze(-1), 1.0)
            probs = torch.where(invalid.unsqueeze(-1), fallback, probs)

        # Sample. Match Zyphra's per-request generator semantics when a seed is
        # supplied, while still using the batch-level filtering above.
        greedy_mask = temperatures <= 0
        if bool(greedy_mask.all()):
            sampled = flat_logits.argmax(dim=-1)
        elif any(getattr(d, "seed", None) is not None for d in datas):
            sampled = torch.empty(
                batch_size * n_codebooks,
                dtype=torch.long,
                device=device,
            )
            for req_idx, data in enumerate(datas):
                start = req_idx * n_codebooks
                end = start + n_codebooks
                if bool(greedy_mask[req_idx]):
                    sampled[start:end] = flat_logits[start:end].argmax(dim=-1)
                    continue
                generator = self._get_sampling_generator(data, device)
                sampled[start:end] = torch.multinomial(
                    probs[start:end],
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1)
        else:
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # Override greedy for zero-temperature requests
            if bool(greedy_mask.any()):
                greedy_ids = greedy_mask.nonzero(as_tuple=False).squeeze(-1)
                for idx in greedy_ids:
                    start = int(idx) * n_codebooks
                    end = start + n_codebooks
                    sampled[start:end] = flat_logits[start:end].argmax(dim=-1)

        return sampled.view(batch_size, n_codebooks).to(torch.long)

    @staticmethod
    def _get_sampling_generator(data: Any, device: torch.device) -> torch.Generator | None:
        seed = getattr(data, "seed", None)
        if seed is None:
            return None
        generator = getattr(data, "_zonos2_sampling_generator", None)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            setattr(data, "_zonos2_sampling_generator", generator)
        return generator

    @staticmethod
    def _apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering."""
        if p <= 0.0 or p >= 1.0:
            return probs
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        probs = probs.scatter(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return probs

    @staticmethod
    def _apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
        """Apply min-p filtering."""
        if min_p <= 0.0:
            return probs
        top_probs, _ = probs.max(dim=-1, keepdim=True)
        mask = probs < (min_p * top_probs)
        probs = probs.masked_fill(mask, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return probs

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        datas: list,
        penalties: torch.Tensor,
    ) -> None:
        """Apply per-codebook repetition penalty in-place."""
        batch_size, n_codebooks, vocab_size = logits.shape
        device = logits.device

        for i, data in enumerate(datas):
            penalty = float(penalties[i])
            if penalty <= 1.0:
                continue
            rep_window = int(getattr(data, "repetition_window", 50))
            rep_codebooks = int(getattr(data, "repetition_codebooks", n_codebooks))
            if rep_window <= 0 or rep_codebooks == 0:
                continue
            if rep_codebooks < 0:
                rep_codebooks = n_codebooks

            # Gather recent output history
            output_rows = getattr(data, "output_rows", None)
            if not output_rows:
                continue
            history = torch.stack(output_rows[-rep_window:], dim=0).to(
                device=device, dtype=torch.long
            )
            # history: [window, frame_width], take first n_codebooks columns
            history = history[:, :n_codebooks]

            for cb in range(min(rep_codebooks, n_codebooks)):
                tokens = torch.unique(history[:, cb])
                tokens = tokens[(tokens >= 0) & (tokens < vocab_size)]
                if tokens.numel() == 0:
                    continue
                scores = logits[i, cb, tokens]
                logits[i, cb, tokens] = torch.where(
                    scores > 0, scores / penalty, scores * penalty
                )

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Store output rows and feedback embeddings for next decode step.

        Also handles ZONOS2's delayed EOS detection: when the countdown
        finishes, we manually terminate the request via req.to_finish since
        SGLang's automatic stop_token_ids detection is disabled (ZONOS2 needs
        additional frames after the first eoa_id to align the delayed codebook
        pattern).
        """
        del result
        rows = self._pending_rows
        embeds = self._pending_embeds
        self._pending_rows = None
        self._pending_embeds = None
        if rows is None or embeds is None:
            return

        eoa_id = int(self.model.config.eoa_id)
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs.get(sched_req.request_id)
            if req_output is None:
                continue
            data = sched_req.data

            # Store output row
            row = rows[row_idx].detach().clone()
            data.output_rows.append(row)

            # Check EOS: any codebook emitting eoa_id triggers countdown
            audio_codes = row[: self.model.n_codebooks].tolist()
            finished = data.check_eos(audio_codes)

            if finished:
                # Manually terminate the request. SGLang's automatic EOS is
                # disabled for ZONOS2 because of the delayed codebook pattern.
                req = data.req
                if req is not None:
                    from sglang.srt.managers.schedule_batch import (
                        FINISH_MATCHED_TOKEN,
                    )

                    req.finished_reason = FINISH_MATCHED_TOKEN(eoa_id)
            else:
                # Queue embedding for next decode step
                data.pending_feedback_queue.append(embeds[row_idx].detach().clone())
