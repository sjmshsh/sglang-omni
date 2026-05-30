# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Delay model runner for OmniScheduler."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.request_builders import _INF_DELAY
from sglang_omni.scheduling.types import RequestOutput


class MossTTSModelRunner(ModelRunner):
    """Samples MOSS-TTS text/audio channels and maintains delay-pattern state."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._pending_rows: torch.Tensor | None = None
        self._pending_embeds: torch.Tensor | None = None
        self._audio_text_token_ids: dict[tuple[torch.device, bool], torch.Tensor] = {}

    def prepare_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch,
            requests,
        )
        return None

    def prepare_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del schedule_batch
        self._write_decode_input_embedding(forward_batch, requests)
        return None

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_moss_step(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_moss_step(result, forward_batch, schedule_batch, requests)

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            rows = data.prompt_rows
            if rows is None:
                raise RuntimeError("MOSS-TTS prefill requires prompt_rows")
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            current_rows = rows[prefix_len : prefix_len + req_len]
            embeds = self.model._prepare_multi_modal_inputs(
                current_rows.to(device=forward_batch.input_ids.device)
            )
            pieces.append(embeds)
        if not pieces:
            return torch.empty(
                (0, self.model.hidden_size),
                device=forward_batch.input_ids.device,
                dtype=self.model.dtype,
            )
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=self.model.dtype,
        )

    def _write_decode_input_embedding(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        embedding = self.model._decode_input_embedding
        weight = embedding.weight
        graph_batch_size = int(getattr(forward_batch, "batch_size", batch_size))
        if graph_batch_size < batch_size:
            raise ValueError(
                f"forward_batch.batch_size ({graph_batch_size}) < "
                f"len(requests) ({batch_size})"
            )
        if graph_batch_size > int(weight.shape[0]):
            raise ValueError(
                "MOSS-TTS decode embedding table is smaller than the CUDA graph "
                f"batch size ({weight.shape[0]} < {graph_batch_size})"
            )

        rows = []
        for sched_req in requests:
            queue = sched_req.data.pending_feedback_queue
            if not queue:
                raise RuntimeError(
                    "MOSS-TTS decode is missing feedback embedding for active "
                    f"request {getattr(sched_req, 'request_id', '<unknown>')}"
                )
            if hasattr(queue, "popleft"):
                rows.append(queue.popleft())
            else:
                rows.append(queue.pop(0))
        if graph_batch_size > batch_size:
            pad = weight.new_zeros(
                (graph_batch_size - batch_size, self.model.hidden_size)
            )
            rows.extend([row for row in pad])

        stacked = torch.stack(rows, dim=0).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            weight[:graph_batch_size].copy_(stacked)

        row_ids = torch.arange(
            graph_batch_size,
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        forward_batch.input_ids[:graph_batch_size].copy_(row_ids)

    def _collect_moss_step(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        channel_logits = self._channel_logits_from_result(result, forward_batch)
        n_vq = len(channel_logits) - 1
        if n_vq <= 0:
            raise RuntimeError("MOSS-TTS requires at least one audio codebook head")

        device = channel_logits[0].device
        data_items = [sched_req.data for sched_req in requests]
        text_token_batch = self._sample_text_tokens_batch(
            channel_logits[0],
            data_items=data_items,
            n_vq=n_vq,
        )
        active_masks = [
            self._sampling_audio_mask(data, n_vq=n_vq) for data in data_items
        ]

        audio_token_batch = self._sample_audio_tokens_batch(
            channel_logits,
            data_items=data_items,
            active_masks=active_masks,
            n_vq=n_vq,
        )

        rows = torch.empty(
            (len(data_items), n_vq + 1),
            dtype=torch.long,
            device=device,
        )
        rows[:, 0] = text_token_batch
        rows[:, 1:] = audio_token_batch

        next_token_ids = text_token_batch.to(dtype=torch.long)
        text_tokens = next_token_ids.detach().cpu().tolist()
        for data, text_token in zip(data_items, text_tokens):
            self._update_delay_state(data, int(text_token), n_vq=n_vq)

        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids
        self._pending_rows = rows
        self._pending_embeds = self.model._prepare_multi_modal_inputs(
            self._pending_rows.to(device=self.model.device)
        ).detach()

    def _channel_logits_from_result(
        self,
        result: Any,
        forward_batch: Any,
    ) -> list[torch.Tensor]:
        logits_output = result.logits_output
        customized = getattr(logits_output, "customized_info", None)
        if isinstance(customized, dict):
            values = customized.get("moss_tts_channel_logits")
            if isinstance(values, list) and values:
                return values
        hidden_states = getattr(logits_output, "hidden_states", None)
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim == 3:
                hidden_states = hidden_states[:, -1, :]
            return self.model.compute_channel_logits(hidden_states, forward_batch)
        raise RuntimeError("MOSS-TTS model output did not include channel logits")

    def _sample_next_row(
        self,
        channel_logits: list[torch.Tensor],
        *,
        row_idx: int,
        data: Any,
        n_vq: int,
    ) -> tuple[int, torch.Tensor]:
        cfg = self.model.config
        device = channel_logits[0].device
        audio_tokens = torch.full(
            (n_vq,),
            int(cfg.audio_pad_code),
            dtype=torch.long,
            device=device,
        )

        text_token = self._next_text_token(
            channel_logits[0][row_idx],
            data=data,
            n_vq=n_vq,
        )
        active = self._sampling_audio_mask(data, n_vq=n_vq)
        rep_penalty = float(data.audio_repetition_penalty)
        for vq_idx in range(n_vq):
            if not active[vq_idx]:
                continue
            logits = channel_logits[vq_idx + 1][row_idx].clone()
            logits[int(cfg.audio_pad_code)] = float("-inf")
            audio_tokens[vq_idx] = self._sample_logits(
                logits,
                temperature=float(data.audio_temperature),
                top_p=float(data.audio_top_p),
                top_k=int(data.audio_top_k),
                repetition_penalty=rep_penalty,
                prev_tokens=(
                    self._previous_audio_tokens(data, vq_idx)
                    if rep_penalty != 1.0
                    else None
                ),
            )

        self._update_delay_state(data, int(text_token), n_vq=n_vq)
        return int(text_token), audio_tokens

    def _sample_audio_tokens_batch(
        self,
        channel_logits: list[torch.Tensor],
        *,
        data_items: list[Any],
        active_masks: list[list[bool]],
        n_vq: int,
    ) -> torch.Tensor:
        cfg = self.model.config
        device = channel_logits[0].device
        batch_size = len(data_items)
        audio_tokens = torch.full(
            (batch_size, n_vq),
            int(cfg.audio_pad_code),
            dtype=torch.long,
            device=device,
        )
        active_pairs = [
            (row_idx, vq_idx)
            for row_idx, mask in enumerate(active_masks)
            for vq_idx, active in enumerate(mask)
            if active
        ]
        if not active_pairs:
            return audio_tokens

        fused_audio_logits = getattr(channel_logits, "audio_logits", None)
        if isinstance(fused_audio_logits, torch.Tensor):
            audio_logits = fused_audio_logits[:batch_size]
        else:
            audio_logits = torch.stack(
                [channel_logits[vq_idx + 1][:batch_size] for vq_idx in range(n_vq)],
                dim=1,
            )
        pair_rows = torch.tensor(
            [row_idx for row_idx, _ in active_pairs],
            dtype=torch.long,
            device=device,
        )
        pair_vqs = torch.tensor(
            [vq_idx for _, vq_idx in active_pairs],
            dtype=torch.long,
            device=device,
        )
        active_logits = audio_logits[pair_rows, pair_vqs].clone()
        active_logits[:, int(cfg.audio_pad_code)] = float("-inf")

        samples = torch.empty(len(active_pairs), dtype=torch.long, device=device)
        grouped: dict[tuple[float, float, int], list[int]] = {}
        fallback_indices: list[int] = []
        for sample_idx, (row_idx, _) in enumerate(active_pairs):
            data = data_items[row_idx]
            if float(data.audio_repetition_penalty) != 1.0:
                fallback_indices.append(sample_idx)
                continue
            key = (
                float(data.audio_temperature),
                float(data.audio_top_p),
                int(data.audio_top_k),
            )
            grouped.setdefault(key, []).append(sample_idx)

        for (temperature, top_p, top_k), sample_indices in grouped.items():
            idx = torch.tensor(sample_indices, dtype=torch.long, device=device)
            samples.index_copy_(
                0,
                idx,
                self._sample_logits_batch(
                    active_logits.index_select(0, idx),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ),
            )

        for sample_idx in fallback_indices:
            row_idx, vq_idx = active_pairs[sample_idx]
            data = data_items[row_idx]
            samples[sample_idx] = self._sample_logits(
                active_logits[sample_idx],
                temperature=float(data.audio_temperature),
                top_p=float(data.audio_top_p),
                top_k=int(data.audio_top_k),
                repetition_penalty=float(data.audio_repetition_penalty),
                prev_tokens=self._previous_audio_tokens(data, vq_idx),
            )

        audio_tokens[pair_rows, pair_vqs] = samples
        return audio_tokens

    def _sample_text_tokens_batch(
        self,
        logits: torch.Tensor,
        *,
        data_items: list[Any],
        n_vq: int,
    ) -> torch.Tensor:
        cfg = self.model.config
        batch_size = len(data_items)
        device = logits.device
        tokens = torch.empty(batch_size, dtype=torch.long, device=device)
        if batch_size == 0:
            return tokens

        forced_delay_rows: list[int] = []
        forced_end_rows: list[int] = []
        audio_rows_by_delay: dict[bool, list[int]] = {False: [], True: []}
        text_rows: list[int] = []
        for row_idx, data in enumerate(data_items):
            delayed_length = self._delayed_length_value(data.delayed_length)
            if delayed_length is not None and delayed_length < n_vq:
                forced_delay_rows.append(row_idx)
            elif delayed_length == n_vq:
                forced_end_rows.append(row_idx)
            elif bool(data.is_audio):
                audio_rows_by_delay[int(data.generation_steps) != 0].append(row_idx)
            else:
                text_rows.append(row_idx)

        self._fill_token_rows(
            tokens,
            forced_delay_rows,
            int(cfg.audio_assistant_delay_slot_token_id),
        )
        self._fill_token_rows(tokens, forced_end_rows, int(cfg.audio_end_token_id))
        for include_delay, rows in audio_rows_by_delay.items():
            self._sample_audio_mode_text_tokens_batch(
                logits,
                tokens=tokens,
                data_items=data_items,
                rows=rows,
                include_delay=include_delay,
            )
        self._sample_non_audio_text_tokens_batch(
            logits,
            tokens=tokens,
            data_items=data_items,
            rows=text_rows,
            n_vq=n_vq,
        )
        return tokens

    @staticmethod
    def _fill_token_rows(
        tokens: torch.Tensor,
        rows: list[int],
        token_id: int,
    ) -> None:
        if not rows:
            return
        row_tensor = torch.tensor(rows, dtype=torch.long, device=tokens.device)
        tokens.index_fill_(0, row_tensor, int(token_id))

    def _sample_audio_mode_text_tokens_batch(
        self,
        logits: torch.Tensor,
        *,
        tokens: torch.Tensor,
        data_items: list[Any],
        rows: list[int],
        include_delay: bool,
    ) -> None:
        if not rows:
            return
        row_tensor = torch.tensor(rows, dtype=torch.long, device=logits.device)
        candidate_ids = self._audio_mode_text_token_ids(
            logits,
            include_delay=include_delay,
        )
        if candidate_ids.numel() == 0:
            tokens.index_fill_(0, row_tensor, 0)
            return
        if candidate_ids.numel() == 1:
            tokens.index_copy_(0, row_tensor, candidate_ids.expand(len(rows)))
            return

        candidate_logits = logits.index_select(0, row_tensor).index_select(
            1,
            candidate_ids,
        )
        sampled_rank = torch.empty(len(rows), dtype=torch.long, device=logits.device)
        grouped: dict[tuple[float, float, int], list[int]] = {}
        for local_idx, row_idx in enumerate(rows):
            data = data_items[row_idx]
            key = (
                float(data.text_temperature),
                float(data.text_top_p),
                int(data.text_top_k),
            )
            grouped.setdefault(key, []).append(local_idx)

        for (temperature, top_p, top_k), sample_indices in grouped.items():
            idx = torch.tensor(sample_indices, dtype=torch.long, device=logits.device)
            sampled_rank.index_copy_(
                0,
                idx,
                self._sample_logits_batch(
                    candidate_logits.index_select(0, idx),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ),
            )

        sampled = candidate_ids.index_select(0, sampled_rank)
        tokens.index_copy_(0, row_tensor, sampled)

    def _sample_non_audio_text_tokens_batch(
        self,
        logits: torch.Tensor,
        *,
        tokens: torch.Tensor,
        data_items: list[Any],
        rows: list[int],
        n_vq: int,
    ) -> None:
        if not rows:
            return

        cfg = self.model.config
        row_tensor = torch.tensor(rows, dtype=torch.long, device=logits.device)
        masked = logits.index_select(0, row_tensor).clone()
        excluded = [
            int(cfg.pad_token_id),
            int(cfg.audio_assistant_gen_slot_token_id),
            int(cfg.audio_assistant_delay_slot_token_id),
            int(cfg.audio_end_token_id),
        ]
        valid_excluded = [
            token_id for token_id in excluded if 0 <= token_id < masked.shape[-1]
        ]
        if valid_excluded:
            excluded_tensor = torch.tensor(
                valid_excluded,
                dtype=torch.long,
                device=logits.device,
            )
            masked[:, excluded_tensor] = float("-inf")

        generation_steps = torch.tensor(
            [int(data_items[row_idx].generation_steps) for row_idx in rows],
            dtype=torch.long,
            device=logits.device,
        )
        delay_token_id = int(cfg.audio_assistant_delay_slot_token_id)
        if 0 <= delay_token_id < masked.shape[-1]:
            masked[generation_steps == 0, delay_token_id] = float("-inf")
        im_end_token_id = int(cfg.im_end_token_id)
        if 0 <= im_end_token_id < masked.shape[-1]:
            masked[generation_steps <= n_vq, im_end_token_id] = float("-inf")

        samples = torch.empty(len(rows), dtype=torch.long, device=logits.device)
        grouped: dict[tuple[float, float, int], list[int]] = {}
        for local_idx, row_idx in enumerate(rows):
            data = data_items[row_idx]
            key = (
                float(data.text_temperature),
                float(data.text_top_p),
                int(data.text_top_k),
            )
            grouped.setdefault(key, []).append(local_idx)

        for (temperature, top_p, top_k), sample_indices in grouped.items():
            idx = torch.tensor(sample_indices, dtype=torch.long, device=logits.device)
            samples.index_copy_(
                0,
                idx,
                self._sample_logits_batch(
                    masked.index_select(0, idx),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ),
            )
        tokens.index_copy_(0, row_tensor, samples)

    def _next_text_token(self, logits: torch.Tensor, *, data: Any, n_vq: int) -> int:
        cfg = self.model.config
        delayed_length = self._delayed_length_value(data.delayed_length)
        if delayed_length is not None and delayed_length < n_vq:
            return int(cfg.audio_assistant_delay_slot_token_id)
        if delayed_length == n_vq:
            data.is_audio = False
            return int(cfg.audio_end_token_id)

        if bool(data.is_audio):
            return self._sample_audio_mode_text_token(logits, data=data)

        masked = logits.clone()
        for token_id in (
            int(cfg.pad_token_id),
            int(cfg.audio_assistant_gen_slot_token_id),
            int(cfg.audio_assistant_delay_slot_token_id),
            int(cfg.audio_end_token_id),
        ):
            if 0 <= token_id < masked.shape[-1]:
                masked[token_id] = float("-inf")
        if int(data.generation_steps) == 0:
            token_id = int(cfg.audio_assistant_delay_slot_token_id)
            if 0 <= token_id < masked.shape[-1]:
                masked[token_id] = float("-inf")
        if int(data.generation_steps) <= n_vq:
            token_id = int(cfg.im_end_token_id)
            if 0 <= token_id < masked.shape[-1]:
                masked[token_id] = float("-inf")

        return int(
            self._sample_logits(
                masked,
                temperature=float(data.text_temperature),
                top_p=float(data.text_top_p),
                top_k=int(data.text_top_k),
            ).item()
        )

    def _sample_audio_mode_text_token(self, logits: torch.Tensor, *, data: Any) -> int:
        # Upstream MossTTSDelay.generate() globally forbids delay-slot at
        # time_step == 0, even when the prompt already ends in audio_start and
        # is_audio is true. Missing this lets some requests immediately enter
        # the flush path and creates the high-WER tail seen in full-set runs.
        candidate_ids = self._audio_mode_text_token_ids(
            logits,
            include_delay=int(data.generation_steps) != 0,
        )
        if candidate_ids.numel() == 0:
            return 0
        if candidate_ids.numel() == 1:
            return int(candidate_ids[0].item())
        candidate_logits = torch.index_select(logits, 0, candidate_ids)
        selected = self._sample_logits(
            candidate_logits,
            temperature=float(data.text_temperature),
            top_p=float(data.text_top_p),
            top_k=int(data.text_top_k),
        )
        return int(candidate_ids[selected].item())

    def _audio_mode_text_token_ids(
        self,
        logits: torch.Tensor,
        *,
        include_delay: bool,
    ) -> torch.Tensor:
        device = logits.device
        cache = getattr(self, "_audio_text_token_ids", None)
        if cache is None:
            cache = {}
            self._audio_text_token_ids = cache
        cached = cache.get((device, bool(include_delay)))
        if cached is not None and cached.shape[0] > 0:
            return cached

        cfg = self.model.config
        vocab_size = int(logits.shape[-1])
        token_ids = [int(cfg.audio_assistant_gen_slot_token_id)]
        if include_delay:
            token_ids.append(int(cfg.audio_assistant_delay_slot_token_id))
        valid = [token_id for token_id in token_ids if 0 <= token_id < vocab_size]
        cached = torch.tensor(valid, dtype=torch.long, device=device)
        cache[(device, bool(include_delay))] = cached
        return cached

    @staticmethod
    def _delayed_length_value(delayed_length: int) -> int | None:
        delayed = int(delayed_length)
        return None if delayed == _INF_DELAY else delayed

    @staticmethod
    def _sampling_audio_mask(
        data: Any,
        *,
        n_vq: int,
    ) -> list[bool]:
        delayed = MossTTSModelRunner._delayed_length_value(data.delayed_length)
        audio_length = int(data.audio_length)
        return [
            audio_length > vq_idx and (delayed is None or vq_idx > delayed - 1)
            for vq_idx in range(n_vq)
        ]

    def _previous_audio_tokens(self, data: Any, vq_idx: int) -> torch.Tensor | None:
        parts = []
        if data.prompt_rows is not None and data.prompt_rows.numel() > 0:
            parts.append(data.prompt_rows[:, vq_idx + 1])
        if data.output_rows:
            parts.append(torch.stack(data.output_rows, dim=0)[:, vq_idx + 1])
        if not parts:
            return None
        return torch.cat(
            [part.to(dtype=torch.long, device=self.model.device) for part in parts]
        )

    @staticmethod
    def _sample_logits(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float = 1.0,
        prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = logits.to(dtype=torch.float32).clone()
        if prev_tokens is not None and repetition_penalty != 1.0:
            valid = prev_tokens[(prev_tokens >= 0) & (prev_tokens < scores.shape[-1])]
            if valid.numel() > 0:
                unique = torch.unique(valid)
                prior = scores[unique]
                penalty = torch.tensor(
                    float(repetition_penalty),
                    dtype=prior.dtype,
                    device=prior.device,
                )
                scores[unique] = torch.where(
                    prior > 0,
                    prior / penalty,
                    prior * penalty,
                )

        if not torch.isfinite(scores).any():
            return torch.zeros((), dtype=torch.long, device=logits.device)

        if temperature <= 0:
            return torch.argmax(scores, dim=-1).to(dtype=torch.long)

        scores = scores / float(temperature)
        if top_k is not None and int(top_k) > 0 and int(top_k) < scores.shape[-1]:
            kth = torch.topk(scores, int(top_k)).values[-1]
            scores[scores < kth] = float("-inf")
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            sorted_scores, sorted_idx = torch.sort(scores, descending=True)
            probs = torch.softmax(sorted_scores, dim=-1)
            remove = torch.cumsum(probs, dim=-1) > float(top_p)
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            scores[sorted_idx[remove]] = float("-inf")
        probs = torch.softmax(scores, dim=-1)
        if not torch.isfinite(probs).all():
            finite = torch.where(
                torch.isfinite(scores),
                scores,
                torch.full_like(scores, float("-inf")),
            )
            return torch.argmax(finite, dim=-1).to(dtype=torch.long)
        try:
            return torch.multinomial(probs, 1).squeeze(0).to(dtype=torch.long)
        except RuntimeError:
            return torch.argmax(scores, dim=-1).to(dtype=torch.long)

    @staticmethod
    def _sample_logits_batch(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("batched MOSS-TTS sampling expects rank-2 logits")
        if logits.shape[0] == 0:
            return torch.empty((0,), dtype=torch.long, device=logits.device)

        scores = logits.to(dtype=torch.float32).clone()
        finite_rows = torch.isfinite(scores).any(dim=-1)
        safe_scores = torch.where(
            torch.isfinite(scores),
            scores,
            torch.full_like(scores, float("-inf")),
        )
        fallback = torch.argmax(safe_scores, dim=-1).to(dtype=torch.long)
        fallback = torch.where(
            finite_rows,
            fallback,
            torch.zeros_like(fallback),
        )

        if temperature <= 0:
            return fallback

        scores = scores / float(temperature)
        vocab_size = int(scores.shape[-1])
        use_top_k = top_k is not None and int(top_k) > 0 and int(top_k) < vocab_size
        use_top_p = top_p is not None and 0.0 < float(top_p) < 1.0

        if use_top_k:
            work_scores, work_indices = torch.topk(scores, int(top_k), dim=-1)
        elif use_top_p:
            work_scores, work_indices = torch.sort(scores, dim=-1, descending=True)
        else:
            probs = torch.softmax(scores, dim=-1)
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            return MossTTSModelRunner._multinomial_or_argmax(
                probs,
                fallback=fallback,
                finite_rows=finite_rows,
            )

        if use_top_p:
            sorted_probs = torch.softmax(work_scores, dim=-1)
            remove = torch.cumsum(sorted_probs, dim=-1) > float(top_p)
            if remove.shape[-1] > 1:
                remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            work_scores = work_scores.masked_fill(remove, float("-inf"))

        probs = torch.softmax(work_scores, dim=-1)
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        sampled_rank = MossTTSModelRunner._multinomial_or_argmax(
            probs,
            fallback=None,
            finite_rows=finite_rows,
        )
        sampled = work_indices.gather(1, sampled_rank.unsqueeze(1)).squeeze(1)
        row_sums = probs.sum(dim=-1)
        good_rows = finite_rows & torch.isfinite(row_sums) & (row_sums > 0)
        return torch.where(good_rows, sampled.to(torch.long), fallback)

    @staticmethod
    def _multinomial_or_argmax(
        probs: torch.Tensor,
        *,
        fallback: torch.Tensor | None,
        finite_rows: torch.Tensor,
    ) -> torch.Tensor:
        row_sums = probs.sum(dim=-1)
        good_rows = finite_rows & torch.isfinite(row_sums) & (row_sums > 0)
        if fallback is None:
            fallback = torch.zeros(probs.shape[0], dtype=torch.long, device=probs.device)
        safe_probs = torch.where(
            good_rows.unsqueeze(1),
            probs,
            torch.zeros_like(probs),
        )
        safe_probs[:, 0] = torch.where(
            good_rows,
            safe_probs[:, 0],
            torch.ones_like(safe_probs[:, 0]),
        )
        try:
            sampled = torch.multinomial(safe_probs, 1).squeeze(1)
        except RuntimeError:
            sampled = torch.argmax(safe_probs, dim=-1).to(dtype=torch.long)
        return torch.where(good_rows, sampled.to(dtype=torch.long), fallback)

    def _update_delay_state(self, data: Any, text_token: int, *, n_vq: int) -> None:
        cfg = self.model.config
        if text_token in (
            int(cfg.audio_start_token_id),
            int(cfg.audio_assistant_gen_slot_token_id),
            int(cfg.audio_assistant_delay_slot_token_id),
        ):
            data.audio_length = int(data.audio_length) + 1
        if text_token == int(cfg.audio_end_token_id):
            data.audio_length = 0
            data.is_audio = False
        if text_token == int(cfg.audio_start_token_id):
            data.is_audio = True
        if text_token == int(cfg.im_end_token_id):
            data.is_audio = False

        delayed = self._delayed_length_value(data.delayed_length)
        if delayed is None and text_token == int(
            cfg.audio_assistant_delay_slot_token_id
        ):
            delayed = 0
        if delayed is not None:
            delayed += 1
            if delayed > n_vq:
                delayed = None
        data.delayed_length = _INF_DELAY if delayed is None else int(delayed)

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        rows = self._pending_rows
        embeds = self._pending_embeds
        self._pending_rows = None
        self._pending_embeds = None
        if rows is None or embeds is None:
            return

        eos_id = int(self.model.config.im_end_token_id)
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == eos_id:
                continue
            sched_req.data.output_rows.append(rows[row_idx].detach().clone())
            sched_req.data.pending_feedback_queue.append(
                embeds[row_idx].detach().clone()
            )
