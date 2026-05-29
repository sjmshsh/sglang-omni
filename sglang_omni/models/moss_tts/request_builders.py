# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for MOSS-TTS."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.moss_tts.hf_config import MossTTSDelayConfig
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.text_normalizer import normalize_tts_text
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

AUDIO_PLACEHOLDER = "<|audio|>"
MOSS_TTS_DELAY_INF = 2**62

_TOKEN_RE = re.compile(r"^\$\{token:(\d+)\}")
_INSTRUCTION_RE = re.compile(r"^\$\{instruction:(.*?)\}")
_AMBIENT_RE = re.compile(r"^\$\{ambient_sound:(.*?)\}")


@dataclass
class MossTTSSGLangRequestData(SGLangARRequestData):
    """Scheduler-owned per-request MOSS-TTS state."""

    enforce_request_limits: bool = True
    prompt_channel_ids: torch.Tensor | None = None
    assistant_prefix_rows: torch.Tensor | None = None
    assistant_start_length: int = 0
    output_rows: list[torch.Tensor] = field(default_factory=list)
    last_input_ids: torch.Tensor | None = None

    is_audio: bool = False
    is_stopping: bool = False
    audio_length: int = 0
    delayed_length: int = MOSS_TTS_DELAY_INF

    n_vq: int = 32
    audio_vocab_size: int = 1024
    audio_pad_code: int = 1024
    pad_token_id: int = 151643
    im_end_token_id: int = 151645
    audio_start_token_id: int = 151652
    audio_end_token_id: int = 151653
    audio_assistant_gen_slot_token_id: int = 151656
    audio_assistant_delay_slot_token_id: int = 151662

    text_temperature: float = 1.5
    text_top_p: float = 1.0
    text_top_k: int = 50
    audio_temperature: float = 1.7
    audio_top_p: float = 0.8
    audio_top_k: int = 25
    text_repetition_penalty: float = 1.0
    audio_repetition_penalty: float = 1.0
    sampling_seed: int | None = None
    sampling_step: int = 0
    engine_start_s: float = 0.0


def apply_delay_pattern(codes_TN: torch.Tensor, pad_code: int) -> torch.Tensor:
    """Raw RVQ codes [T, N] -> delayed codes [T + N - 1, N]."""

    if codes_TN.ndim != 2:
        raise ValueError(f"codes must be [T, N], got {tuple(codes_TN.shape)}")
    t, n = codes_TN.shape
    out = torch.full(
        (t + n - 1, n),
        int(pad_code),
        device=codes_TN.device,
        dtype=codes_TN.dtype,
    )
    for idx in range(n):
        out[idx : idx + t, idx] = codes_TN[:, idx]
    return out


def apply_de_delay_pattern(delay_codes: torch.Tensor) -> torch.Tensor:
    """Delayed RVQ codes [L, N] -> raw codes [L - N + 1, N]."""

    if delay_codes.ndim != 2:
        raise ValueError(f"delay_codes must be [L, N], got {tuple(delay_codes.shape)}")
    length, n = delay_codes.shape
    t = length - n + 1
    if t <= 0:
        return delay_codes.new_empty((0, n))
    out = delay_codes.new_zeros((t, n))
    for idx in range(n):
        out[:, idx] = delay_codes[idx : idx + t, idx]
    return out


def to_codes_TN(raw: Any, n_vq: int) -> torch.Tensor | None:
    if raw is None:
        return None
    codes = raw if isinstance(raw, torch.Tensor) else torch.tensor(raw)
    if codes.numel() == 0:
        return None
    if codes.ndim != 2:
        raise ValueError(
            f"reference_codes must be [T, {n_vq}], got {tuple(codes.shape)}"
        )
    if codes.shape[1] != n_vq and codes.shape[0] == n_vq:
        codes = codes.transpose(0, 1).contiguous()
    if codes.shape[1] != n_vq:
        raise ValueError(
            f"reference_codes must be [T, {n_vq}], got {tuple(codes.shape)}"
        )
    return codes.to(torch.long).cpu()


def normalize_moss_tts_inputs(inputs: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(inputs, str):
        return inputs, []
    if not isinstance(inputs, dict):
        return str(inputs or ""), []
    text = (
        inputs.get("input")
        if inputs.get("input") is not None
        else inputs.get("text")
    )
    refs = inputs.get("references") or []
    if refs and not isinstance(refs, list):
        raise ValueError("MOSS-TTS references must be a list")
    return str(text or ""), [dict(ref) for ref in refs if isinstance(ref, dict)]


def build_moss_tts_state(payload: StagePayload) -> MossTTSState:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references = normalize_moss_tts_inputs(inputs)
    if not isinstance(inputs, dict):
        inputs = {"text": text}

    tokens, instruction, ambient_sound, text = _parse_inline_controls(text)
    first_ref = references[0] if references else {}

    reference_audio = (
        inputs.get("reference_audio")
        or inputs.get("ref_audio")
        or inputs.get("audio")
        or tts_params.get("ref_audio")
        or first_ref.get("audio_path")
        or first_ref.get("path")
        or first_ref.get("ref_audio")
        or first_ref.get("audio")
    )
    reference_text = (
        inputs.get("reference_text")
        or inputs.get("ref_text")
        or tts_params.get("ref_text")
        or first_ref.get("text")
    )
    reference_codes = (
        inputs.get("reference_codes")
        or inputs.get("vq_codes")
        or first_ref.get("reference_codes")
        or first_ref.get("vq_codes")
        or first_ref.get("codes")
    )

    tokens = _pick_int("tokens", tokens, inputs, params, tts_params)
    token_count = _pick_int("token_count", None, inputs, params, tts_params)
    duration_tokens = _pick_int("duration_tokens", None, inputs, params, tts_params)
    if tokens is None:
        tokens = token_count if token_count is not None else duration_tokens

    common_temperature = _pick_float("temperature", None, params, tts_params)
    common_top_p = _pick_float("top_p", None, params, tts_params)
    common_top_k = _pick_int("top_k", None, params, tts_params)
    repetition_penalty = _pick_float(
        "repetition_penalty", 1.0, params, tts_params
    )
    max_new_tokens = _pick_int("max_new_tokens", 2048, params, tts_params)
    text_top_k = _pick_int(
        "text_top_k",
        50 if common_top_k is None else common_top_k,
        params,
        tts_params,
    )
    audio_top_k = _pick_int(
        "audio_top_k",
        25 if common_top_k is None else common_top_k,
        params,
        tts_params,
    )

    return MossTTSState(
        text=text,
        reference_audio=reference_audio,
        reference_text=str(reference_text) if reference_text is not None else None,
        reference_codes=(
            to_codes_TN(reference_codes, 32).tolist()
            if reference_codes is not None
            else None
        ),
        instruction=_pick_text("instruction", instruction, inputs, params, tts_params)
        or _pick_text("instructions", None, inputs, params, tts_params),
        tokens=tokens,
        quality=_pick_text("quality", None, inputs, params, tts_params),
        sound_event=_pick_text("sound_event", None, inputs, params, tts_params),
        ambient_sound=_pick_text(
            "ambient_sound", ambient_sound, inputs, params, tts_params
        ),
        language=_pick_text("language", None, inputs, params, tts_params),
        max_new_tokens=max_new_tokens if max_new_tokens is not None else 2048,
        text_temperature=_pick_float(
            "text_temperature",
            1.5 if common_temperature is None else common_temperature,
            inputs,
            params,
            tts_params,
        ),
        text_top_p=_pick_float(
            "text_top_p",
            1.0 if common_top_p is None else common_top_p,
            params,
            tts_params,
        ),
        text_top_k=text_top_k if text_top_k is not None else 50,
        audio_temperature=_pick_float(
            "audio_temperature",
            1.7 if common_temperature is None else common_temperature,
            inputs,
            params,
            tts_params,
        ),
        audio_top_p=_pick_float(
            "audio_top_p",
            0.8 if common_top_p is None else common_top_p,
            params,
            tts_params,
        ),
        audio_top_k=audio_top_k if audio_top_k is not None else 25,
        repetition_penalty=float(repetition_penalty or 1.0),
        audio_repetition_penalty=_pick_float(
            "audio_repetition_penalty",
            float(repetition_penalty or 1.0),
            params,
            tts_params,
        ),
        seed=_pick_int("seed", None, params, tts_params),
    )


def _parse_inline_controls(text: str) -> tuple[int | None, str | None, str | None, str]:
    tokens = None
    instruction = None
    ambient_sound = None
    text = str(text or "")
    match = _TOKEN_RE.match(text)
    if match:
        tokens = int(match.group(1))
        text = _TOKEN_RE.sub("", text, count=1)
    match = _INSTRUCTION_RE.match(text)
    if match:
        instruction = match.group(1)
        text = _INSTRUCTION_RE.sub("", text, count=1)
    match = _AMBIENT_RE.match(text)
    if match:
        ambient_sound = match.group(1)
        text = _AMBIENT_RE.sub("", text, count=1)
    return tokens, instruction, ambient_sound, text


def _pick_text(name: str, default: str | None, *sources: dict[str, Any]) -> str | None:
    for source in sources:
        if name in source and source[name] not in (None, ""):
            return str(source[name])
    return default


def _pick_float(
    name: str,
    default: float | None,
    *sources: dict[str, Any],
) -> float | None:
    for source in sources:
        if name in source and source[name] is not None:
            return float(source[name])
    return float(default) if default is not None else None


def _pick_int(name: str, default: int | None, *sources: dict[str, Any]) -> int | None:
    for source in sources:
        if name in source and source[name] is not None:
            return int(source[name])
    return default


class MossTTSPromptBuilder:
    """Build the upstream MOSS-TTS multi-channel prompt tensor."""

    _TEMPLATE = """<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""

    def __init__(self, tokenizer: Any, config: MossTTSDelayConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.audio_user_slot_token = self._id_to_token(config.audio_user_slot_token_id)
        self.audio_assistant_gen_slot_token = self._id_to_token(
            config.audio_assistant_gen_slot_token_id
        )
        self.audio_assistant_delay_slot_token = self._id_to_token(
            config.audio_assistant_delay_slot_token_id
        )
        self.audio_start_token = self._id_to_token(config.audio_start_token_id)
        self.audio_end_token = self._id_to_token(config.audio_end_token_id)

    def build_prompt_ids(
        self,
        state: MossTTSState,
        reference_codes_list: list[torch.Tensor] | None = None,
    ) -> list[list[int]]:
        reference_codes_list = reference_codes_list or []
        content = self._build_user_content(state, len(reference_codes_list))
        content = self._apply_chat_template(content)
        rows = self._get_unified_codes("user", content, reference_codes_list)
        return rows.to(torch.long).cpu().tolist()

    def _id_to_token(self, token_id: int) -> str:
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        if isinstance(token, list):
            return str(token[0]) if token else ""
        return str(token)

    def _build_user_content(self, state: MossTTSState, num_references: int) -> str:
        if num_references <= 0:
            reference = "None"
        else:
            reference = "\n".join(
                f"[S{idx + 1}]:\n{AUDIO_PLACEHOLDER}" for idx in range(num_references)
            )
        text = _normalize_prompt_text(state.text)
        return (
            self._TEMPLATE.replace("{reference}", reference)
            .replace("{instruction}", str(state.instruction))
            .replace("{tokens}", str(state.tokens))
            .replace("{quality}", str(state.quality))
            .replace("{sound_event}", str(state.sound_event))
            .replace("{ambient_sound}", str(state.ambient_sound))
            .replace("{language}", str(state.language))
            .replace("{text}", text)
        )

    def _apply_chat_template(self, content: str) -> str:
        message = [{"role": "user", "content": content}]
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                rendered = apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                if isinstance(rendered, str):
                    return rendered
            except TypeError:
                pass
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def _replace_audio_placeholders(
        content: str,
        lengths: list[int],
        n_vq: int,
        gen_slot_token: str,
        delay_slot_token: str,
        audio_start_token: str,
        audio_end_token: str,
    ) -> str:
        if n_vq < 1:
            raise ValueError(f"n_vq must be >= 1, got {n_vq}")
        num_placeholders = content.count(AUDIO_PLACEHOLDER)
        if num_placeholders != len(lengths):
            raise ValueError(
                f"Number of {AUDIO_PLACEHOLDER} ({num_placeholders}) "
                f"does not match lengths ({len(lengths)})"
            )

        def build_audio_block(length: int) -> str:
            if length < 0:
                raise ValueError(f"length must be >= 0, got {length}")
            if length == 0:
                return f"{audio_start_token}{audio_end_token}"
            return (
                f"{audio_start_token}"
                f"{gen_slot_token * int(length)}"
                f"{delay_slot_token * (n_vq - 1)}"
                f"{audio_end_token}"
            )

        lengths_iter = iter(lengths)
        return re.sub(
            re.escape(AUDIO_PLACEHOLDER),
            lambda _match: build_audio_block(next(lengths_iter)),
            content,
        )

    @staticmethod
    def _merge_consecutive_audio_placeholders(
        content: str,
        audio_codes_list: list[torch.Tensor],
    ) -> tuple[str, list[torch.Tensor]]:
        matches = list(re.finditer(re.escape(AUDIO_PLACEHOLDER), content))
        if len(matches) <= 1:
            return content, audio_codes_list
        if len(matches) != len(audio_codes_list):
            raise ValueError("Audio placeholders do not match tokenizer output")

        new_audio_codes_list: list[torch.Tensor] = []
        new_parts: list[str] = []
        last_pos = 0
        idx = 0
        while idx < len(matches):
            end_idx = idx
            while (
                end_idx + 1 < len(matches)
                and content[matches[end_idx].end() : matches[end_idx + 1].start()]
                .strip()
                == ""
            ):
                end_idx += 1

            new_parts.append(content[last_pos : matches[idx].start()])
            new_parts.append(AUDIO_PLACEHOLDER)
            last_pos = matches[end_idx].end()

            if end_idx == idx:
                new_audio_codes_list.append(audio_codes_list[idx])
            else:
                new_audio_codes_list.append(
                    torch.cat(audio_codes_list[idx : end_idx + 1], dim=0)
                )
            idx = end_idx + 1

        new_parts.append(content[last_pos:])
        return "".join(new_parts), new_audio_codes_list

    def _get_unified_codes(
        self,
        role: str,
        content: str,
        audio_codes_list: list[torch.Tensor],
    ) -> torch.Tensor:
        if role == "user":
            gen_slot = delay_slot = self.audio_user_slot_token
        else:
            gen_slot = self.audio_assistant_gen_slot_token
            delay_slot = self.audio_assistant_delay_slot_token
        n_vq = audio_codes_list[0].shape[1] if audio_codes_list else self.config.n_vq
        if len(audio_codes_list) > 1 and AUDIO_PLACEHOLDER in content:
            content, audio_codes_list = self._merge_consecutive_audio_placeholders(
                content,
                audio_codes_list,
            )
        content = self._replace_audio_placeholders(
            content=content,
            lengths=[int(c.shape[0]) for c in audio_codes_list],
            n_vq=n_vq,
            gen_slot_token=gen_slot,
            delay_slot_token=delay_slot,
            audio_start_token=self.audio_start_token,
            audio_end_token=self.audio_end_token,
        )
        text_codes = torch.tensor(self.tokenizer.encode(content), dtype=torch.long)
        if not audio_codes_list:
            audio_codes = torch.full(
                (text_codes.shape[0], n_vq),
                int(self.config.audio_pad_code),
                dtype=torch.long,
            )
            return torch.cat([text_codes.unsqueeze(1), audio_codes], dim=1)

        audio_start = torch.where(text_codes == self.config.audio_start_token_id)[0]
        audio_end = torch.where(text_codes == self.config.audio_end_token_id)[0]
        if len(audio_start) != len(audio_codes_list) or len(audio_end) != len(
            audio_codes_list
        ):
            raise ValueError("Audio placeholders do not match tokenizer output")

        pieces: list[torch.Tensor] = []
        prefix_idx = 0
        for start_t, end_t, codes in zip(audio_start, audio_end, audio_codes_list):
            start = int(start_t.item())
            end = int(end_t.item())
            delayed = apply_delay_pattern(
                codes.to(torch.long),
                self.config.audio_pad_code,
            )
            pad = torch.full(
                (start - prefix_idx + 1, n_vq),
                int(self.config.audio_pad_code),
                dtype=torch.long,
            )
            pieces.extend([pad, delayed])
            prefix_idx = end

        last_end = int(audio_end[-1].item())
        tail = torch.full(
            (text_codes.shape[0] - last_end, n_vq),
            int(self.config.audio_pad_code),
            dtype=torch.long,
        )
        pieces.append(tail)
        delayed_audio = torch.cat(pieces, dim=0)
        if text_codes.shape[0] != delayed_audio.shape[0]:
            text_codes = text_codes[: delayed_audio.shape[0]]
        return torch.cat([text_codes.unsqueeze(1), delayed_audio], dim=1)


def _normalize_prompt_text(text: str) -> str:
    return normalize_tts_text(str(text or ""))


def _reference_fingerprint(prompt_rows: torch.Tensor) -> str | None:
    if prompt_rows.numel() == 0 or prompt_rows.shape[1] <= 1:
        return None
    audio_rows = prompt_rows[:, 1:].contiguous()
    if not audio_rows.numel():
        return None
    digest = hashlib.blake2b(
        audio_rows.cpu().numpy().tobytes(),
        digest_size=16,
    )
    return digest.hexdigest()


def build_sglang_moss_tts_request(
    state: MossTTSState,
    config: MossTTSDelayConfig,
    *,
    request_id: str = "",
) -> MossTTSSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    if not state.prompt_token_ids:
        raise RuntimeError("MOSS-TTS prompt_token_ids are missing before tts_engine")

    prompt_rows = torch.tensor(state.prompt_token_ids, dtype=torch.long)
    if prompt_rows.ndim != 2 or prompt_rows.shape[1] != config.channels:
        raise ValueError(
            f"MOSS-TTS prompt must be [T, {config.channels}], got "
            f"{tuple(prompt_rows.shape)}"
        )
    origin_input_ids = prompt_rows[:, 0].tolist()
    im_start_idx = torch.where(prompt_rows[:, 0] == int(config.im_start_token_id))[0]
    assistant_start_idx = (
        int(im_start_idx[-1].item()) + 3
        if im_start_idx.numel() > 0
        else int(prompt_rows.shape[0])
    )
    assistant_start_idx = max(0, min(assistant_start_idx, int(prompt_rows.shape[0])))
    assistant_prefix_rows = prompt_rows[assistant_start_idx:].clone()

    sampling_params = SamplingParams(
        max_new_tokens=int(state.max_new_tokens),
        temperature=1.0,
        stop_token_ids=[int(config.im_end_token_id)],
    )
    sampling_params.normalize(tokenizer=None)
    req = Req(
        rid=request_id,
        origin_input_text=state.text,
        origin_input_ids=origin_input_ids,
        sampling_params=sampling_params,
        vocab_size=int(config.vocab_size),
        eos_token_ids={int(config.im_end_token_id)},
        extra_key=_reference_fingerprint(prompt_rows),
    )
    req._codec_suppress_tokens = None
    req._input_embeds_are_projected = False

    last_text = int(prompt_rows[-1, 0].item())
    audio_starts = torch.where(prompt_rows[:, 0] == config.audio_start_token_id)[0]
    is_continuation = last_text in (
        int(config.audio_start_token_id),
        int(config.audio_assistant_gen_slot_token_id),
    )
    audio_length = 0
    if is_continuation and audio_starts.numel() > 0:
        audio_length = int(prompt_rows.shape[0] - int(audio_starts[-1].item()))

    return MossTTSSGLangRequestData(
        input_ids=torch.tensor(origin_input_ids, dtype=torch.long),
        req=req,
        prompt_channel_ids=prompt_rows,
        assistant_prefix_rows=assistant_prefix_rows,
        assistant_start_length=int(assistant_prefix_rows.shape[0]),
        last_input_ids=prompt_rows[-1].clone(),
        is_audio=bool(is_continuation),
        audio_length=audio_length,
        n_vq=int(config.n_vq),
        audio_vocab_size=int(config.audio_vocab_size),
        audio_pad_code=int(config.audio_pad_code),
        pad_token_id=int(config.pad_token_id),
        im_end_token_id=int(config.im_end_token_id),
        audio_start_token_id=int(config.audio_start_token_id),
        audio_end_token_id=int(config.audio_end_token_id),
        audio_assistant_gen_slot_token_id=int(
            config.audio_assistant_gen_slot_token_id
        ),
        audio_assistant_delay_slot_token_id=int(
            config.audio_assistant_delay_slot_token_id
        ),
        max_new_tokens=int(state.max_new_tokens),
        text_temperature=float(state.text_temperature),
        text_top_p=float(state.text_top_p),
        text_top_k=int(state.text_top_k),
        audio_temperature=float(state.audio_temperature),
        audio_top_p=float(state.audio_top_p),
        audio_top_k=int(state.audio_top_k),
        text_repetition_penalty=float(state.repetition_penalty),
        audio_repetition_penalty=float(state.audio_repetition_penalty),
        sampling_seed=int(state.seed) if state.seed is not None else None,
    )


def apply_moss_tts_result(state: MossTTSState, data: MossTTSSGLangRequestData) -> None:
    if data.output_rows:
        generated_rows = torch.stack(data.output_rows, dim=0).to(torch.long)
        rows = generated_rows
        if data.assistant_prefix_rows is not None and data.assistant_prefix_rows.numel():
            rows = torch.cat(
                [data.assistant_prefix_rows.to(torch.long), generated_rows],
                dim=0,
            )
            state.decode_start_length = int(data.assistant_start_length)
        else:
            state.decode_start_length = 0
        state.output_codes = rows.tolist()
        state.completion_tokens = int(generated_rows.shape[0])
    else:
        state.output_codes = None
        state.decode_start_length = 0
    state.prompt_tokens = (
        int(data.prompt_channel_ids.shape[0])
        if data.prompt_channel_ids is not None
        else 0
    )


_MossRequestBuilder = Callable[[StagePayload], MossTTSSGLangRequestData]
_MossResultAdapter = Callable[[MossTTSSGLangRequestData], StagePayload]


def make_moss_tts_scheduler_adapters(
    config: MossTTSDelayConfig,
    *,
    max_new_tokens_cap: int | None = None,
) -> tuple[_MossRequestBuilder, _MossResultAdapter]:
    def request_builder(payload: StagePayload) -> MossTTSSGLangRequestData:
        state = MossTTSState.from_dict(payload.data)
        if max_new_tokens_cap is not None:
            state.max_new_tokens = min(int(state.max_new_tokens), max_new_tokens_cap)
        data = build_sglang_moss_tts_request(
            state,
            config,
            request_id=payload.request_id,
        )
        data.engine_start_s = time.perf_counter()
        data.stage_payload = payload
        return data

    def result_adapter(data: MossTTSSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = MossTTSState.from_dict(payload.data)
        apply_moss_tts_result(state, data)
        if data.engine_start_s:
            state.engine_time_s = time.perf_counter() - data.engine_start_s
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


def extract_moss_tts_audio_segments(
    output_rows: torch.Tensor,
    *,
    n_vq: int,
    audio_pad_code: int,
) -> list[torch.Tensor]:
    """Extract de-delayed non-pad audio segments from assistant MOSS rows."""

    if output_rows is None or output_rows.numel() == 0:
        return []
    rows = output_rows.to(torch.long)
    if rows.ndim != 2 or rows.shape[1] != n_vq + 1 or rows.shape[0] < n_vq:
        return []
    audio_codes = apply_de_delay_pattern(rows[:, 1:])
    if audio_codes.numel() == 0:
        return []

    all_pad = (audio_codes == int(audio_pad_code)).all(dim=1)
    non_pad_idx = torch.nonzero(~all_pad, as_tuple=False).flatten()
    if non_pad_idx.numel() == 0:
        return []

    breaks = torch.where(non_pad_idx[1:] != non_pad_idx[:-1] + 1)[0] + 1
    chunks = (
        torch.tensor_split(non_pad_idx, breaks.cpu().tolist())
        if breaks.numel()
        else [non_pad_idx]
    )
    segments: list[torch.Tensor] = []
    for idx in chunks:
        segment = audio_codes[idx]
        if segment.numel() == 0:
            continue
        segments.append(segment.cpu())
    return segments


__all__ = [
    "AUDIO_PLACEHOLDER",
    "MOSS_TTS_DELAY_INF",
    "MossTTSPromptBuilder",
    "MossTTSSGLangRequestData",
    "apply_delay_pattern",
    "apply_de_delay_pattern",
    "apply_moss_tts_result",
    "build_moss_tts_state",
    "build_sglang_moss_tts_request",
    "extract_moss_tts_audio_segments",
    "make_moss_tts_scheduler_adapters",
    "to_codes_TN",
]
