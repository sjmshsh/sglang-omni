# MOSS-TTS SeedTTS Eval Runbook

This runbook is for validating `OpenMOSS-Team/MOSS-TTS-v1.5` on the
Seed-TTS-eval EN/ZH subsets with SGLang Omni.

Use these commands from the container repo root:

git checkout feature/support_moss_tts

# 1. 安装 uv（若还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# 2. 创建虚拟环境并安装项目
uv venv .venv -p python3.11
source .venv/bin/activate

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip uv
uv pip install -e .
uv pip install funasr zhconv zhon openai-whisper s3prl soundfile scipy

apt-get update
apt-get install -y libnuma1
apt-get install -y libibverbs1 ffmpeg

```bash
cd /workspace/sglang-omni
source .venv/bin/activate
export HF_HOME=/workspace
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
```

huggingface-cli download OpenMOSS-Team/MOSS-TTS-v1.5
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer
huggingface-cli download zhaochenyang20/seed-tts-eval-arrow
huggingface-cli download openai/whisper-large-v3

python - <<'PY'
from funasr import AutoModel
AutoModel(model="paraformer-zh")
PY

Do not pass `--temperature 0`, `--top-p 1`, `--top-k -1`, or
`--repetition-penalty 1` for this benchmark unless you are explicitly testing
greedy decoding. MOSS-TTS should use its model-owned default sampling values.

Use `--token-count auto` for MOSS-TTS SeedTTS runs. It forwards an estimated
duration token count to MOSS-TTS and reduces short clipping / long tail output.

## 1. Preflight

Run this after code changes:

```bash
python -m py_compile \
  sglang_omni/models/moss_tts/model_runner.py \
  sglang_omni/models/moss_tts/codec.py \
  sglang_omni/models/moss_tts/stages.py \
  benchmarks/tasks/tts.py \
  benchmarks/eval/benchmark_tts_seedtts.py \
  benchmarks/eval/report_tts_seedtts.py \
  tests/unit_test/moss_tts/test_pipeline.py

python -m pytest tests/unit_test/moss_tts/test_pipeline.py -q
```

## 2. Start Server

Terminal 1:

```bash
cd /workspace/sglang-omni
source .venv/bin/activate
export HF_HOME=/workspace
export CUDA_VISIBLE_DEVICES=0

python -m sglang_omni.cli serve \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --config examples/configs/moss_tts.yaml \
  --host 0.0.0.0 \
  --port 8000
```

Wait until the server is ready. After any code change, stop and restart this
server before re-running benchmarks.

## 3. Smoke50

Terminal 2:

```bash
cd /workspace/sglang-omni
source .venv/bin/activate
export HF_HOME=/workspace
export CUDA_VISIBLE_DEVICES=0
```

If you have a second GPU and want ASR / similarity to avoid sharing with the
server, use this in Terminal 2 instead:

```bash
export CUDA_VISIBLE_DEVICES=1
```

### EN Smoke50

Generate audio and compute WER:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model OpenMOSS-Team/MOSS-TTS-v1.5 \
  --port 8000 \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --lang en \
  --max-samples 50 \
  --max-new-tokens 4096 \
  --token-count auto \
  --warmup 0 \
  --max-concurrency 1 \
  --ref-format references \
  --device cuda:0 \
  --output-dir results/moss_tts_en_smoke50
```

Compute speaker similarity:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --similarity-only \
  --model OpenMOSS-Team/MOSS-TTS-v1.5 \
  --port 8000 \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --lang en \
  --max-samples 50 \
  --ref-format references \
  --device cuda:0 \
  --output-dir results/moss_tts_en_smoke50
```

### ZH Smoke50

Generate audio and compute CER:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model OpenMOSS-Team/MOSS-TTS-v1.5 \
  --port 8000 \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --lang zh \
  --max-samples 50 \
  --max-new-tokens 4096 \
  --token-count auto \
  --warmup 0 \
  --max-concurrency 1 \
  --ref-format references \
  --device cuda:0 \
  --output-dir results/moss_tts_zh_smoke50
```

Compute speaker similarity:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --similarity-only \
  --model OpenMOSS-Team/MOSS-TTS-v1.5 \
  --port 8000 \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --lang zh \
  --max-samples 50 \
  --ref-format references \
  --device cuda:0 \
  --output-dir results/moss_tts_zh_smoke50
```

### Smoke50 Report

```bash
python -m benchmarks.eval.report_tts_seedtts \
  --run "MOSS-TTS-v1.5 smoke50" en results/moss_tts_en_smoke50 \
  --run "MOSS-TTS-v1.5 smoke50" zh results/moss_tts_zh_smoke50 \
  --output results/moss_tts_smoke50_report.md

cat results/moss_tts_smoke50_report.md
```

## 4. Full Acceptance Run

Run this only after Smoke50 has acceptable quality and no failures.

### EN Full

Generate audio and compute WER:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model OpenMOSS-Team/MOSS-TTS-v1.5 \
  --port 8000 \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --lang en \
  --max-new-tokens 4096 \
  --token-count auto \
  --warmup 0 \
  --max-concurrency 16 \
  --ref-format references \
  --device cuda:0 \
  --output-dir results/moss_tts_en_full
```

Compute speaker similarity:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --similarity-only \
  --model OpenMOSS-Team/MOSS-TTS-v1.5 \
  --port 8000 \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --lang en \
  --ref-format references \
  --device cuda:0 \
  --output-dir results/moss_tts_en_full
```

### ZH Full

Generate audio and compute CER:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model OpenMOSS-Team/MOSS-TTS-v1.5 \
  --port 8000 \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --lang zh \
  --max-new-tokens 4096 \
  --token-count auto \
  --warmup 0 \
  --max-concurrency 16 \
  --ref-format references \
  --device cuda:0 \
  --output-dir results/moss_tts_zh_full
```

Compute speaker similarity:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --similarity-only \
  --model OpenMOSS-Team/MOSS-TTS-v1.5 \
  --port 8000 \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --lang zh \
  --ref-format references \
  --device cuda:0 \
  --output-dir results/moss_tts_zh_full
```

### Full Report

```bash
python -m benchmarks.eval.report_tts_seedtts \
  --run "MOSS-TTS-v1.5 compile + CUDA graph" en results/moss_tts_en_full \
  --run "MOSS-TTS-v1.5 compile + CUDA graph" zh results/moss_tts_zh_full \
  --output results/moss_tts_full_report.md

cat results/moss_tts_full_report.md
```

## 5. Inspect Clipping / Failures

Use this if audio sounds clipped, too short, too long, or if failed/skipped is
non-zero:

If the server prints a CUDA `device-side assert triggered` from the MOSS audio
tokenizer decoder, stop and restart the server before any further requests.
CUDA device asserts poison the current process; later stack traces may point at
unrelated scheduler code.

```bash
python - <<'PY'
import csv
from pathlib import Path

for result_dir in [
    Path("results/moss_tts_en_smoke50"),
    Path("results/moss_tts_zh_smoke50"),
    Path("results/moss_tts_en_full"),
    Path("results/moss_tts_zh_full"),
]:
    csv_path = result_dir / "results.csv"
    if not csv_path.exists():
        continue
    print(f"\n== {result_dir} ==")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows[:10]:
        print(row)
PY
```

Check output JSON summaries:

```bash
python - <<'PY'
import json
from pathlib import Path

for result_dir in [
    Path("results/moss_tts_en_smoke50"),
    Path("results/moss_tts_zh_smoke50"),
]:
    print(f"\n== {result_dir} ==")
    for name in ["speed_results.json", "wer_results.json", "similarity_results.json"]:
        path = result_dir / name
        if path.exists():
            data = json.load(open(path))
            print(name, data.get("summary"))
PY
```

## 6. Expected Report Columns

The report should include:

```text
Model / config
Split
WER / CER
Speaker SIM
Samples
Failed / skipped
Throughput
rtf_mean
```

For acceptance, report full EN and ZH subsets:

```text
EN: WER and Speaker SIM
ZH: CER and Speaker SIM
```
