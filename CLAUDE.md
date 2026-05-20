# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python package for marking CM clips in MPEG2‑TS videos using multiple methods (subtitles, logo detection via NCC, clip info, speech recognition, ensemble).

- **Logo detection**: Uses normalized cross-correlation (NCC) on raw pixel templates instead of edge-based AND comparison. The logo template is stored as a full-frame grayscale mean image (1440×1080 PNG), with the logo region auto-detected at mark time via edge-density scanning.
- **CM classification**: `_auto_by_method` chooses `_groundtruth > _ensemble > speech > logo > subtitles`. Falls back to `logo` when subtitles have no signal (all 0.0 or 0.5).
- **Subtitle correction**: After whisper STT, `.generated.srt` is corrected via LLM (three-layer detection: single-entry plausibility → local coherence → global consistency) to `.corrected.srt`. Gaps from faster-whisper VAD bugs are then fixed in-place (see `fix_srt_gaps.py`). The speech module uses `.corrected.srt` for classification.
- **Speech-to-text**: Google Web Speech API replaced by local faster-whisper (small model, int8, ~320MB RAM). Consecutive no-subtitle clips merged for context before transcription.
- Entry point: `tsmarker.marker:main`, console script `tsmarker`

- Requires Python ≥3.13

## Common Development Commands

```bash
uv pip install -e .
uv run pytest tests/
```

Tests rely on sample files at `C:\Samples`.

## CLI Commands

All commands via the `tsmarker` console script:

```bash
# Mark clips by method(s) + generate EDL
tsmarker mark --method subtitles --method clipinfo --method logo --method speech -i video.mkv -x index.ptsmap -m output.markermap --edl output.edl

# Cut using auto-detected method
tsmarker cut --input video.mkv --index index.ptsmap --marker output.markermap --output clips/

# Ground truth from manual clip adjustment
tsmarker groundtruth --input video.mkv --index index.ptsmap --marker output.markermap --clips clips/

# Get program clip groups
tsmarker get-program-clips --marker output.markermap --index index.ptsmap

# Logo extraction and crop detection
tsmarker extract-logo -i video.mkv -x index.ptsmap -o logo.png
tsmarker crop-detect -i logo.png

# Subtitle preparation (loads existing ASS, does speech-to-text)
tsmarker prepare-subtitles -i video.mkv -x index.ptsmap

# Correct whisper-generated subtitles using LLM with YAML metadata
tsmarker correct-srt -i video.mkv

# Fix subtitle end-time gaps from faster-whisper VAD bugs (modifies in-place)
tsmarker fix-srt-gaps -s video.corrected.srt

# Ensemble training pipeline
tsmarker ensemble-dataset -i search_folder/ -o dataset.csv
tsmarker ensemble-train -i dataset.csv -o model.pkl
tsmarker ensemble-predict --model model.pkl --index index.ptsmap --marker output.markermap
```

## Architecture

### Core Modules
- `marker.py` — CLI entry point and dispatch for all subcommands
- `common.py` — `MarkerMap` class for reading/writing `.markermap` files; `get_program_clips()` grouping logic; EDL generation (Kodi MPlayer format: seconds, space-separated, action=3 for CM, adjacent CMs merged)
- `subtitles.py` — Subtitle-based CM detection (reads embedded ASS from MKV via ffmpeg; skips empty-text events from ARIB control data)
- `fix_srt_gaps.py` — Post-process `.corrected.srt` to fix end-time gaps caused by faster-whisper VAD bugs (k=0.8s merge threshold, m=0.5s extension)
- `clipinfo.py` — Clip position/duration marking
- `logo.py` — Logo-based CM detection
- `correct_srt.py` — LLM-based SRT correction (three-layer detection), integrated into speech pipeline after whisper STT
- `speech/` — Speech-to-text (whisper) + LLM-based marking; `whisper_stt.py` SRT generation, `text_extractor.py` subtitle prep, `llm_client.py` OpenAI client, `prompt_engine.py` YAML context prompts
- `ensemble.py` — Ensemble model training and prediction
- `groundtruth.py` — Manual verification ground truth marking
- `pipeline.py` — `ExtractLogoPipeline`, `CropDetectPipeline`, `ExtractMeanImage` logo/crop utilities

## Dependencies & External Tools
- **LLM API** — speech marking uses OpenAI-compatible API (`OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL` env vars)
- **ffmpeg/ffprobe** — must be in PATH
- Python: rich, scikit-learn, opencv-python, pandas, pysubs2, faster-whisper, openai, requests, PyYAML, python-dotenv
