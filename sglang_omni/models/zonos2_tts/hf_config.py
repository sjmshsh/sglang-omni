# SPDX-License-Identifier: Apache-2.0
"""Hugging Face config adapter for Zyphra ZONOS2 checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from transformers import AutoConfig, PretrainedConfig


class Zonos2Config(PretrainedConfig):
    """Minimal HF config wrapper around ZONOS2's native ``params.json``."""

    model_type = "zonos2"

    def __init__(self, **kwargs: Any) -> None:
        params = dict(kwargs)
        if "dim" in params and "hidden_size" not in params:
            params["hidden_size"] = int(params["dim"])
        if "n_layers" in params and "num_hidden_layers" not in params:
            params["num_hidden_layers"] = int(params["n_layers"])

        hidden_size = int(params.get("hidden_size", 2048))
        head_dim = int(params.get("head_dim", 128))
        n_heads = params.get("n_heads")
        if n_heads is None:
            n_heads = hidden_size // head_dim
        params.setdefault("num_attention_heads", int(n_heads))
        params.setdefault("num_key_value_heads", int(params.get("n_kv_heads", n_heads)))

        if "intermediate_size" not in params:
            multiplier = float(params.get("ffn_dim_multiplier", 4.0))
            multiple_of = int(params.get("multiple_of", 256))
            raw_intermediate = int(multiplier * hidden_size)
            params["intermediate_size"] = (
                multiple_of
                * ((raw_intermediate + multiple_of - 1) // multiple_of)
            )

        params.setdefault("rms_norm_eps", float(params.get("norm_eps", 1e-5)))
        params.setdefault(
            "max_position_embeddings", int(params.get("max_seqlen", 4096))
        )
        params.setdefault("rope_theta", float(params.get("rope_theta", 10000.0)))

        codebook_size = int(params.get("codebook_size", 1024))
        n_codebooks = int(params.get("n_codebooks", 9))
        text_vocab = params.get("text_vocab")
        vocab_size = n_codebooks * (codebook_size + 2)
        if text_vocab is not None:
            vocab_size += int(text_vocab) + 1
        params.setdefault("vocab_size", vocab_size)

        params.setdefault("architectures", ["Zonos2SGLangModel"])
        super().__init__(**params)


def register_zonos2_hf_config() -> None:
    """Register the local ZONOS2 config class with Transformers."""

    try:
        AutoConfig.register(Zonos2Config.model_type, Zonos2Config)
    except ValueError:
        pass


def load_zonos2_params(checkpoint_dir: str | os.PathLike[str]) -> dict[str, Any]:
    params_path = Path(checkpoint_dir) / "params.json"
    if not params_path.is_file():
        raise FileNotFoundError(f"ZONOS2 params.json not found under {checkpoint_dir}")
    with params_path.open(encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise ValueError(f"ZONOS2 params.json must contain an object: {params_path}")
    return params


def build_zonos2_hf_config_dict(params: dict[str, Any]) -> dict[str, Any]:
    cfg = Zonos2Config(**params)
    data = cfg.to_dict()
    data["model_type"] = Zonos2Config.model_type
    data["architectures"] = ["Zonos2SGLangModel"]
    return data


def ensure_zonos2_hf_layout(checkpoint_dir: str | os.PathLike[str]) -> str:
    """Return a temp directory with HF-style config and weight entrypoints.

    Zyphra publishes ``params.json`` + ``model.pth``. SGLang expects a
    Transformers-readable ``config.json`` and a standard weight filename, so we
    create a deterministic temp view with symlinks back to the original files.
    """

    checkpoint = Path(checkpoint_dir).resolve()
    params = load_zonos2_params(checkpoint)
    digest = hashlib.blake2b(str(checkpoint).encode("utf-8"), digest_size=12).hexdigest()
    layout_dir = Path(tempfile.gettempdir()) / "sglang_omni_zonos2" / digest
    layout_dir.mkdir(parents=True, exist_ok=True)

    config_path = layout_dir / "config.json"
    config_data = build_zonos2_hf_config_dict(params)
    current = None
    if config_path.is_file():
        try:
            current = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = None
    if current != config_data:
        config_path.write_text(
            json.dumps(config_data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    params_link = layout_dir / "params.json"
    _ensure_link(params_link, checkpoint / "params.json")

    weight_path = checkpoint / "model.pth"
    if not weight_path.is_file():
        raise FileNotFoundError(f"ZONOS2 model.pth not found under {checkpoint}")
    _ensure_link(layout_dir / "pytorch_model.bin", weight_path)
    return str(layout_dir)


def _ensure_link(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        try:
            if link.resolve() == target.resolve():
                return
        except FileNotFoundError:
            pass
        link.unlink()
    try:
        link.symlink_to(target)
    except OSError:
        os.link(target, link)
