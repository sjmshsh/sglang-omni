# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS model runner with delay-pattern sampling."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.request_builders import (
    MOSS_TTS_DELAY_INF,
    MossTTSSGLangRequestData,
)


class MossTTSModelRunner(ModelRunner):
    """Runs MOSS-TTS AR steps and stores generated multi-channel rows."""

    def prepare_prefill(self, forward_batch: Any, schedule_batch: Any, requests: list):
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch,
            requests,
        )
        return None

    def prepare_decode(self, forward_batch: Any, schedule_batch: Any, requests: list):
        del schedule_batch
        if not requests:
            return None
        batch_size = int(getattr(forward_batch, "batch_size", len(requests)))
        rows = []
        for row_idx, sched_req in enumerate(requests):
            data: MossTTSSGLangRequestData = sched_req.data
            if data.last_input_ids is None:
                row = self._pad_decode_row(data, int(forward_batch.input_ids[row_idx]))
            else:
                row = data.last_input_ids
            rows.append(row)
        if batch_size > len(rows):
            pad_data: MossTTSSGLangRequestData = requests[0].data
            for row_idx in range(len(rows), batch_size):
                text_id = (
                    int(forward_batch.input_ids[row_idx])
                    if row_idx < int(forward_batch.input_ids.numel())
                    else pad_data.pad_token_id
                )
                rows.append(self._pad_decode_row(pad_data, text_id))
        stacked = torch.stack(rows, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=torch.long,
        )
        self.model.prepare_decode_inputs(stacked)
        return None

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data: MossTTSSGLangRequestData = sched_req.data
            req = data.req
            if data.prompt_channel_ids is None:
                raise RuntimeError("MOSS-TTS prefill requires prompt_channel_ids")
            prefix_len = len(req.prefix_indices)
            end = prefix_len + int(req.extend_input_len)
            rows = data.prompt_channel_ids[prefix_len:end].to(
                device=forward_batch.input_ids.device,
                dtype=torch.long,
            )
            pieces.append(self.model._prepare_multi_modal_inputs(rows))
        if not pieces:
            return torch.empty(
                0,
                self.model.config.hidden_size,
                device=forward_batch.input_ids.device,
                dtype=next(self.model.parameters()).dtype,
            )
        return torch.cat(pieces, dim=0).to(dtype=next(self.model.parameters()).dtype)

    @staticmethod
    def _pad_decode_row(
        data: MossTTSSGLangRequestData,
        text_token_id: int,
    ) -> torch.Tensor:
        return torch.tensor(
            [text_token_id] + [data.audio_pad_code] * data.n_vq,
            dtype=torch.long,
        )

    def _sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        del forward_batch, schedule_batch
        text_logits = logits_output.next_token_logits
        audio_logits = getattr(logits_output, "moss_tts_audio_logits", None)
        if audio_logits is None:
            raise RuntimeError("MOSS-TTS model did not return audio logits")

        next_text_tokens: list[torch.Tensor] = []
        for row_idx, sched_req in enumerate(requests):
            data: MossTTSSGLangRequestData = sched_req.data
            text_token, audio_tokens = self._sample_one_row(
                data,
                text_logits[row_idx],
                audio_logits[row_idx],
            )
            generated_row = torch.cat([text_token.view(1), audio_tokens], dim=0)
            data.last_input_ids = generated_row.detach().to(torch.long)
            data.output_rows.append(generated_row.detach().cpu().to(torch.long))
            data.sampling_step += 1
            if int(text_token.item()) == data.im_end_token_id:
                data.req.finished_reason = FINISH_MATCHED_TOKEN(data.im_end_token_id)
            next_text_tokens.append(text_token)

        return torch.stack(next_text_tokens, dim=0).to(torch.long)

    def _sample_one_row(
        self,
        data: MossTTSSGLangRequestData,
        text_logits: torch.Tensor,
        audio_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = text_logits.device
        text_token = torch.tensor(data.pad_token_id, device=device, dtype=torch.long)

        if not data.is_stopping and data.delayed_length < data.n_vq:
            text_token.fill_(data.audio_assistant_delay_slot_token_id)
        elif not data.is_stopping and data.delayed_length == data.n_vq:
            text_token.fill_(data.audio_end_token_id)
            data.is_audio = False
        elif not data.is_stopping:
            logits = self._prepare_text_logits(data, text_logits)
            text_token = self._sample_logits(
                logits,
                temperature=data.text_temperature,
                top_p=data.text_top_p,
                top_k=data.text_top_k,
                seed=data.sampling_seed,
                step=data.sampling_step,
                offset=0,
            )

        text_id = int(text_token.item())
        if text_id == data.audio_start_token_id:
            data.is_audio = True
        if text_id == data.im_end_token_id:
            data.is_stopping = True

        audio_tokens = torch.full(
            (data.n_vq,),
            data.audio_pad_code,
            device=device,
            dtype=torch.long,
        )
        sampling_audio_mask = self._audio_sampling_mask(data, device=device)
        for ch in torch.nonzero(sampling_audio_mask, as_tuple=False).flatten().tolist():
            logits = audio_logits[ch].clone()
            logits[data.audio_pad_code] = float("-inf")
            self._apply_repetition_penalty_(
                logits,
                self._audio_history(data, int(ch), device),
                data.audio_repetition_penalty,
            )
            audio_tokens[ch] = self._sample_logits(
                logits,
                temperature=data.audio_temperature,
                top_p=data.audio_top_p,
                top_k=data.audio_top_k,
                seed=data.sampling_seed,
                step=data.sampling_step,
                offset=1 + int(ch),
            )

        if text_id in (
            data.audio_start_token_id,
            data.audio_assistant_gen_slot_token_id,
            data.audio_assistant_delay_slot_token_id,
        ):
            data.audio_length += 1
        if text_id == data.audio_end_token_id:
            data.audio_length = 0
        if (
            data.delayed_length == MOSS_TTS_DELAY_INF
            and text_id == data.audio_assistant_delay_slot_token_id
        ):
            data.delayed_length = 0
        if data.delayed_length != MOSS_TTS_DELAY_INF:
            data.delayed_length += 1
        if data.delayed_length > data.n_vq:
            data.delayed_length = MOSS_TTS_DELAY_INF

        return text_token, audio_tokens

    def _prepare_text_logits(
        self,
        data: MossTTSSGLangRequestData,
        text_logits: torch.Tensor,
    ) -> torch.Tensor:
        logits = text_logits.clone()
        if not data.is_audio:
            for token_id in (
                data.pad_token_id,
                data.audio_assistant_gen_slot_token_id,
                data.audio_assistant_delay_slot_token_id,
                data.audio_end_token_id,
            ):
                if 0 <= token_id < logits.shape[-1]:
                    logits[token_id] = float("-inf")
        else:
            mask = torch.full_like(logits, float("-inf"))
            for token_id in (
                data.audio_assistant_gen_slot_token_id,
                data.audio_assistant_delay_slot_token_id,
            ):
                if 0 <= token_id < logits.shape[-1]:
                    mask[token_id] = logits[token_id]
            logits = mask
        if data.sampling_step == 0:
            logits[data.audio_assistant_delay_slot_token_id] = float("-inf")
        if (
            data.sampling_step <= data.n_vq
            and 0 <= data.im_end_token_id < logits.shape[-1]
        ):
            logits[data.im_end_token_id] = float("-inf")
        self._apply_repetition_penalty_(
            logits,
            self._text_history(data, logits.device),
            data.text_repetition_penalty,
        )
        return logits

    @staticmethod
    def _audio_sampling_mask(
        data: MossTTSSGLangRequestData,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(data.n_vq, device=device, dtype=torch.long)
        pre_audio_mask = int(data.audio_length) > positions
        if data.delayed_length == MOSS_TTS_DELAY_INF:
            post_audio_mask = torch.ones_like(pre_audio_mask, dtype=torch.bool)
        else:
            post_audio_mask = positions > int(data.delayed_length) - 1
        return pre_audio_mask & post_audio_mask

    @staticmethod
    def _sample_logits(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
        step: int,
        offset: int,
    ) -> torch.Tensor:
        if float(temperature) <= 0.0:
            return torch.argmax(logits, dim=-1)

        scores = logits.to(torch.float32) / float(temperature)
        if int(top_k) > 0 and int(top_k) < scores.shape[-1]:
            values, indices = torch.topk(scores, int(top_k), dim=-1)
            filtered = torch.full_like(scores, float("-inf"))
            filtered.scatter_(dim=-1, index=indices, src=values)
            scores = filtered
        if float(top_p) < 1.0:
            scores = _apply_top_p(scores, float(top_p))
        probs = F.softmax(scores, dim=-1)
        if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
            return torch.argmax(logits, dim=-1)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(
                (int(seed) + step * 1_000_003 + offset * 9_176) & 0x7FFFFFFF
            )
        return torch.multinomial(probs, num_samples=1, generator=generator).view(())

    @staticmethod
    def _apply_repetition_penalty_(
        logits: torch.Tensor,
        prev_tokens: torch.Tensor | None,
        penalty: float,
    ) -> None:
        if prev_tokens is None or float(penalty) == 1.0 or prev_tokens.numel() == 0:
            return
        vocab = logits.shape[-1]
        tokens = torch.unique(prev_tokens[(prev_tokens >= 0) & (prev_tokens < vocab)])
        if tokens.numel() == 0:
            return
        scores = logits[tokens].to(torch.float32)
        adjusted = torch.where(
            scores > 0,
            scores / float(penalty),
            scores * float(penalty),
        )
        logits[tokens] = adjusted.to(logits.dtype)

    @staticmethod
    def _text_history(
        data: MossTTSSGLangRequestData,
        device: torch.device,
    ) -> torch.Tensor | None:
        values = list(getattr(data.req, "origin_input_ids", []) or [])
        values.extend(getattr(data.req, "output_ids", []) or [])
        if not values:
            return None
        return torch.tensor(values, dtype=torch.long, device=device)

    @staticmethod
    def _audio_history(
        data: MossTTSSGLangRequestData,
        channel: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        values: list[int] = []
        if data.prompt_channel_ids is not None:
            values.extend(data.prompt_channel_ids[:, channel + 1].tolist())
        for row in data.output_rows:
            values.append(int(row[channel + 1].item()))
        if not values:
            return None
        tokens = torch.tensor(values, dtype=torch.long, device=device)
        return tokens[tokens != int(data.audio_pad_code)]


def _apply_top_p(scores: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = F.softmax(scores, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove_indices = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        dim=-1,
        index=sorted_indices,
        src=remove,
    )
    filtered = scores.clone()
    filtered[remove_indices] = float("-inf")
    return filtered


__all__ = ["MossTTSModelRunner"]
