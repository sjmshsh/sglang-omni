# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MOSS-TTS delay-pattern model."""

from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass, fields
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix

from sglang_omni.models.moss_tts.hf_config import MossTTSDelayConfig

logger = logging.getLogger(__name__)


@dataclass
class MossTTSLogitsOutput(LogitsProcessorOutput):
    moss_tts_audio_logits: torch.Tensor | None = None


class MossTTSDelayModel(nn.Module):
    """MOSS-TTS Qwen3 backbone with multi-head RVQ prediction."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: MossTTSDelayConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        self.model = Qwen3Model(
            config=config.language_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        self.emb_ext = nn.ModuleList()
        if self.pp_group.is_first_rank:
            for idx in range(config.n_vq):
                self.emb_ext.append(
                    VocabParallelEmbedding(
                        config.audio_vocab_size + 1,
                        config.hidden_size,
                        quant_config=quant_config,
                        prefix=add_prefix(f"emb_ext.{idx}", prefix),
                    )
                )
        else:
            for _ in range(config.n_vq):
                self.emb_ext.append(PPMissingLayer())

        self.lm_heads = nn.ModuleList()
        if self.pp_group.is_last_rank:
            self.lm_heads.append(
                ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=add_prefix("lm_heads.0", prefix),
                )
            )
            for idx in range(config.n_vq):
                self.lm_heads.append(
                    ParallelLMHead(
                        config.audio_vocab_size + 1,
                        config.hidden_size,
                        prefix=add_prefix(f"lm_heads.{idx + 1}", prefix),
                    )
                )
        else:
            for _ in range(config.channels):
                self.lm_heads.append(PPMissingLayer())

        self.logits_processors = nn.ModuleList(
            [
                self._make_logits_processor(config, idx)
                for idx in range(config.channels)
            ]
        )
        self._pad_token_per_channel = [
            int(config.pad_token_id),
            *[int(config.audio_pad_code)] * int(config.n_vq),
        ]
        self.register_buffer(
            "_decode_input_ids",
            torch.empty(0, config.channels, dtype=torch.long),
            persistent=False,
        )

    @staticmethod
    def _make_logits_processor(
        config: MossTTSDelayConfig,
        channel: int,
    ) -> LogitsProcessor:
        """Build a channel-specific logits processor for current SGLang APIs."""

        channel_config = copy(config)
        channel_config.vocab_size = int(config.vocab_size_list[channel])
        return LogitsProcessor(channel_config)

    @staticmethod
    def _audio_logits_metadata(forward_batch: ForwardBatch) -> LogitsMetadata:
        metadata = LogitsMetadata.from_forward_batch(forward_batch)
        metadata.next_token_logits_buffer = None
        metadata.extend_return_logprob = False
        metadata.extend_return_top_logprob = False
        metadata.extend_token_ids_logprob = False
        metadata.extend_input_logprob_token_ids_gpu = None
        metadata.top_logprobs_nums = None
        metadata.token_ids_logprobs = None
        return metadata

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def prepare_decode_inputs(self, input_ids_BC: torch.Tensor) -> None:
        """Stage multi-channel decode rows in a stable buffer for CUDA Graph."""

        if input_ids_BC.ndim != 2 or input_ids_BC.shape[1] != self.config.channels:
            raise ValueError(
                f"decode inputs must be [B, {self.config.channels}], got "
                f"{tuple(input_ids_BC.shape)}"
            )
        batch_size = int(input_ids_BC.shape[0])
        if self._decode_input_ids.shape[0] < batch_size:
            new_size = max(batch_size, max(16, self._decode_input_ids.shape[0] * 2))
            self._decode_input_ids = torch.empty(
                new_size,
                self.config.channels,
                dtype=torch.long,
                device=input_ids_BC.device,
            )
        self._decode_input_ids[:batch_size].copy_(input_ids_BC)

    def _coerce_channel_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim == 2:
            if input_ids.shape[1] != self.config.channels:
                raise ValueError(
                    f"expected {self.config.channels} channels, got "
                    f"{input_ids.shape[1]}"
                )
            return input_ids
        if input_ids.ndim != 1:
            raise ValueError(f"input_ids must be 1-D or 2-D, got {input_ids.ndim}-D")

        total = int(input_ids.shape[0])
        channels = int(self.config.channels)
        if total % channels == 0 and total > 0:
            return input_ids.view(total // channels, channels)

        rows = torch.empty(
            total,
            channels,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        for idx, pad_id in enumerate(self._pad_token_per_channel):
            rows[:, idx].fill_(int(pad_id))
        rows[:, 0] = input_ids
        return rows

    def _prepare_multi_modal_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = self._coerce_channel_ids(input_ids)
        inputs_embeds = self.model.embed_tokens(rows[:, 0])
        for idx, emb in enumerate(self.emb_ext):
            inputs_embeds = inputs_embeds + emb(rows[:, idx + 1])
        return inputs_embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ):
        del kwargs
        if input_embeds is None:
            if input_ids is None:
                raise ValueError("MOSS-TTS forward requires input_ids or input_embeds")
            is_decode = bool(getattr(forward_batch.forward_mode, "is_decode")())
            if is_decode:
                if self._decode_input_ids.shape[0] < input_ids.shape[0]:
                    rows = torch.empty(
                        input_ids.shape[0],
                        self.config.channels,
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                    for idx, pad_id in enumerate(self._pad_token_per_channel):
                        rows[:, idx].fill_(int(pad_id))
                    rows[:, 0] = input_ids
                    self.prepare_decode_inputs(rows)
                input_embeds = self._prepare_multi_modal_inputs(
                    self._decode_input_ids[: input_ids.shape[0]]
                )
            elif self.pp_group.is_first_rank:
                input_embeds = self._prepare_multi_modal_inputs(input_ids)
            else:
                input_embeds = None

        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if not self.pp_group.is_last_rank:
            return hidden_states

        text_out = self.logits_processors[0](
            None,
            hidden_states=hidden_states,
            lm_head=self.lm_heads[0],
            logits_metadata=forward_batch,
        )
        audio_logits = []
        audio_logits_metadata = self._audio_logits_metadata(forward_batch)
        for idx in range(1, self.config.channels):
            out = self.logits_processors[idx](
                None,
                hidden_states=hidden_states,
                lm_head=self.lm_heads[idx],
                logits_metadata=audio_logits_metadata,
            )
            logits = out.next_token_logits
            logits[..., self.config.audio_pad_code] = float("-inf")
            audio_logits.append(logits)
        output_kwargs = {
            field.name: getattr(text_out, field.name)
            for field in fields(LogitsProcessorOutput)
        }
        return MossTTSLogitsOutput(
            **output_kwargs,
            moss_tts_audio_logits=torch.stack(audio_logits, dim=1),
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for raw_name, loaded_weight in weights:
            name = raw_name
            if name.startswith("language_model."):
                name = "model." + name[len("language_model.") :]
            if name.startswith("codec_model.") or name.startswith("audio_tokenizer."):
                continue
            if (
                "rotary_emb.inv_freq" in name
                or "rotary_emb.cos_cached" in name
                or "rotary_emb.sin_cached" in name
                or "projector" in name
            ):
                continue

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if name.startswith("emb_ext.") and name.endswith(".weight"):
                if name in params_dict:
                    self._load_param(params_dict[name], loaded_weight)
                    loaded.add(name)
                continue

            if name.startswith("lm_heads.") and name.endswith(".weight"):
                if name in params_dict:
                    self._load_param(params_dict[name], loaded_weight)
                    loaded.add(name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    break
                param = params_dict.get(mapped)
                if param is None:
                    break
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded.add(mapped)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict.get(name)
                if param is None:
                    logger.debug("MOSS-TTS checkpoint tensor %s was not used", raw_name)
                    continue
                self._load_param(param, loaded_weight)
                loaded.add(name)
        return loaded

    @staticmethod
    def _load_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


__all__ = ["MossTTSDelayModel"]
