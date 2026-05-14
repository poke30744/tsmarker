# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python package for marking CM clips in MPEG2‑TS videos using multiple methods (subtitles, logo detection via NCC, clip info, speech recognition, ensemble).

- **Logo detection**: Uses normalized cross-correlation (NCC) on raw pixel templates instead of edge-based AND comparison. The logo template is stored as a full-frame grayscale mean image (1440×1080 PNG), with the logo region auto-detected at mark time via edge-density scanning.
- **CM classification**: `_auto_by_method` chooses `_groundtruth > _ensemble > speech > logo > subtitles`. Falls back to `logo` when subtitles have no signal (all 0.0 or 0.5).
- **Subtitle extraction**: subtitles are read from embedded ASS in MKV via ffmpeg. No standalone `.ass.original` file needed.
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

# Ensemble training pipeline
tsmarker ensemble-dataset -i search_folder/ -o dataset.csv
tsmarker ensemble-train -i dataset.csv -o model.pkl
tsmarker ensemble-predict --model model.pkl --index index.ptsmap --marker output.markermap
```

## Architecture

### Core Modules
- `marker.py` — CLI entry point and dispatch for all subcommands
- `common.py` — `MarkerMap` class for reading/writing `.markermap` files; `get_program_clips()` grouping logic; EDL generation (Kodi MPlayer format: seconds, space-separated, action=3 for CM, adjacent CMs merged)
- `subtitles.py` — Subtitle-based CM detection (reads embedded ASS from MKV via ffmpeg)
- `subtitles.py` — Subtitle-based CM detection (from ASS file)
- `clipinfo.py` — Clip position/duration marking
- `logo.py` — Logo-based CM detection
- `speech/` — Speech recognition and LLM-based marking
- `ensemble.py` — Ensemble model training and prediction
- `groundtruth.py` — Manual verification ground truth marking
- `pipeline.py` — `ExtractLogoPipeline`, `CropDetectPipeline`, `ExtractMeanImage` logo/crop utilities

## Dependencies & External Tools
- **LLM API** — speech marking uses OpenAI-compatible API (`OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL` env vars)
- **ffmpeg/ffprobe** — must be in PATH
- Python: rich, scikit-learn, opencv-python, pandas, pysubs2, SpeechRecognition, openai, requests, PyYAML, python-dotenv
