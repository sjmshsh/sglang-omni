# SPDX-License-Identifier: Apache-2.0
"""Summarize SeedTTS TTS benchmark runs.

This utility reads the files emitted by ``benchmark_tts_seedtts.py``:

* ``speed_results.json``
* ``wer_results.json``
* ``similarity_results.json`` (optional)

and prints the compact report table used for TTS acceptance/debugging.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSpec:
    label: str
    split: str
    result_dir: Path


@dataclass(frozen=True)
class ReportRow:
    label: str
    split: str
    error_metric: str
    evaluated: int
    total_samples: int
    failed: int
    skipped: int
    speaker_sim: float | None
    throughput_qps: float | None
    rtf_mean: float | None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _summary(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path} does not contain a summary object")
    return summary


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _metric_name(split: str) -> str:
    return "CER" if split.lower() == "zh" else "WER"


def build_row(spec: RunSpec) -> ReportRow:
    speed_path = spec.result_dir / "speed_results.json"
    wer_path = spec.result_dir / "wer_results.json"
    similarity_path = spec.result_dir / "similarity_results.json"
    if not speed_path.exists():
        raise FileNotFoundError(f"Missing {speed_path}")
    if not wer_path.exists():
        raise FileNotFoundError(f"Missing {wer_path}")

    speed = _summary(speed_path)
    wer = _summary(wer_path)
    similarity = _summary(similarity_path) if similarity_path.exists() else {}

    split = spec.split.upper()
    error = _as_float(wer.get("wer_corpus")) or 0.0
    metric = _metric_name(spec.split)
    return ReportRow(
        label=spec.label,
        split=split,
        error_metric=f"{metric} {error * 100.0:.2f}%",
        evaluated=_as_int(wer.get("evaluated")),
        total_samples=_as_int(wer.get("total_samples")),
        failed=_as_int(speed.get("failed_requests")),
        skipped=_as_int(wer.get("skipped")),
        speaker_sim=_as_float(similarity.get("speaker_similarity_mean")),
        throughput_qps=_as_float(speed.get("throughput_qps")),
        rtf_mean=_as_float(speed.get("rtf_mean")),
    )


def _format_float(value: float | None, digits: int) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def format_markdown(rows: list[ReportRow]) -> str:
    lines = [
        "| Model / config | Split | WER / CER | Speaker SIM | Samples | Failed / skipped | Throughput | rtf_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.label} | "
            f"{row.split} | "
            f"{row.error_metric} | "
            f"{_format_float(row.speaker_sim, 2)} | "
            f"{row.evaluated}/{row.total_samples} | "
            f"{row.failed} / {row.skipped} | "
            f"{_format_float(row.throughput_qps, 3)} req/s | "
            f"{_format_float(row.rtf_mean, 2)} |"
        )
    return "\n".join(lines)


def format_tsv(rows: list[ReportRow]) -> str:
    lines = [
        "\t".join(
            [
                "Model / config",
                "Split",
                "WER / CER",
                "Speaker SIM",
                "Samples",
                "Failed / skipped",
                "Throughput",
                "rtf_mean",
            ]
        )
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.label,
                    row.split,
                    row.error_metric,
                    _format_float(row.speaker_sim, 2),
                    f"{row.evaluated}/{row.total_samples}",
                    f"{row.failed} / {row.skipped}",
                    f"{_format_float(row.throughput_qps, 3)} req/s",
                    _format_float(row.rtf_mean, 2),
                ]
            )
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SeedTTS TTS speed/WER result directories."
    )
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("LABEL", "SPLIT", "RESULT_DIR"),
        required=True,
        help=(
            "One result row. SPLIT is en or zh. RESULT_DIR must contain "
            "speed_results.json and wer_results.json. If similarity_results.json "
            "exists, Speaker SIM is included. Repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "tsv"],
        default="markdown",
        help="Output table format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write. The table is always also printed.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specs = [
        RunSpec(label=label, split=split, result_dir=Path(result_dir))
        for label, split, result_dir in args.run
    ]
    rows = [build_row(spec) for spec in specs]
    table = format_tsv(rows) if args.format == "tsv" else format_markdown(rows)
    print(table)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n")


if __name__ == "__main__":
    main()
