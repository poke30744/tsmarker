# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python package for marking CM clips in MPEG2‑TS videos using multiple methods (subtitles, logo detection, clip info, speech recognition, ensemble).
- Entry point: `tsmarker.marker:main`, console script `tsmarker`
- Dependency: `tscutter` (uses `PtsMap`, `SplitVideo`; also calls CLI for list-clips/select-clips/probe)
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
# Mark clips by method(s)
tsmarker mark --method subtitles clipinfo logo speech -i video.ts -x index.ptsmap -m output.markermap

# Cut using auto-detected method (_groundtruth > _ensemble > subtitles)
tsmarker cut --input video.ts --index index.ptsmap --marker output.markermap --output clips/

# Ground truth from manual clip adjustment (stdout JSON)
tsmarker groundtruth --input video.ts --index index.ptsmap --marker output.markermap --clips clips/

# Extract TS byte ranges (stdout pipe or files)
tsmarker extract-clips -i video.ts -x index.ptsmap -c '[[0.0,100.5]]'

# Get program clip groups
tsmarker get-program-clips --marker output.markermap --index index.ptsmap

# Logo extraction and crop detection
tsmarker extract-logo -i video.ts -x index.ptsmap -o logo.png
tsmarker crop-detect -i logo.png

# Subtitle preparation
tsmarker prepare-subtitles -i video.ts -x index.ptsmap

# Ensemble training pipeline
tsmarker ensemble-dataset -i search_folder/ -o dataset.csv
tsmarker ensemble-train -i dataset.csv -o model.pkl
tsmarker ensemble-predict --model model.pkl --index index.ptsmap --marker output.markermap
```

## Architecture

### Core Modules
- `marker.py` — CLI entry point and dispatch for all subcommands
- `common.py` — `MarkerMap` class for reading/writing `.markermap` files; `get_program_clips()` grouping logic
- `clip_utils.py` — `extract_clips_stdout()` / `extract_clips_to_dir()` for TS byte range extraction
- `subtitles.py` — Subtitle-based CM detection using Caption2AssC
- `clipinfo.py` — Clip position/duration marking
- `logo.py` — Logo-based CM detection
- `speech/` — Speech recognition and LLM-based marking
- `ensemble.py` — Ensemble model training and prediction
- `groundtruth.py` — Manual verification ground truth marking
- `pipeline.py` — `ExtractLogoPipeline`, `CropDetectPipeline` logo/crop utilities

## Dependencies & External Tools
- **Caption2AssC** — required for subtitle extraction, must be in PATH
- **LLM API** — speech marking uses OpenAI-compatible API (`OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL` env vars)
- **tscutter CLI** — needed for `list-clips`, `select-clips`, `probe` subcommands
- **ffmpeg/ffprobe** — must be in PATH
- Python: scikit-learn, opencv-python, pandas, pysubs2, SpeechRecognition, openai, requests, PyYAML, python-dotenv
