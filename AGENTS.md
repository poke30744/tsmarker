# tsmarker Agent Instructions

High-signal, repo-specific guidance for OpenCode agents working in this repository.

## Overview
- Python package for marking CM clips in MPEG2‑TS videos using multiple methods (subtitles, logo detection, clip info, speech recognition).
- Primary dependency: `tscutter` (external package).
- Requires Python >=3.9 (see `pyproject.toml`).
- Uses `uv` for dependency management and packaging.

## Commands
- **Main CLI** (via `tsmarker` console script):
  ```bash
  tsmarker mark --method subtitles --input video.ts
  tsmarker cut --method subtitles --input video.ts
  tsmarker groundtruth --input video.ts
  tsmarker merge --input video1.ts video2.ts
  ```
- **Module entry points** (run with `python -m`):
  ```bash
  python -m tsmarker.ensemble dataset --input folder --output dataset.csv
  python -m tsmarker.ensemble train --input dataset.csv --output model.pkl
  python -m tsmarker.ensemble predict --model model.pkl --input video.markermap
  python -m tsmarker.subtitles --input video.ts
  python -m tsmarker.pipeline logo --input video.ts
  python -m tsmarker.pipeline cropdetect --input video.ts
  python -m tsmarker.speech dataset --input folder --output json
  python -m tsmarker.speech mark --input folder --url http://...
  ```
- **Building distribution** (Jenkins flow):
  ```bash
  uv build
  ```
- **Development installation**:
  ```bash
  uv pip install -e .
  ```

## Testing
- Uses `pytest` (configured in `.vscode/settings.json`).
- Tests rely on sample files located at `C:\Samples` (see `tests/__init__.py`). If samples are unavailable, adjust the `samplesDir` path or skip tests.
- Run all tests: `pytest tests/`.
- Single test: `pytest tests/test_common.py::test_MarkerMap_Properties`.

## Dependencies & External Tools
- **Caption2AssC.cmd** – required for subtitle extraction (Windows‑only; called by `subtitles.py`).
- **Speech recognition** – expects a running HTTP server at `http://127.0.0.1:5000/api/mark/speech` (see `marker.py:28`).
- **tscutter** – must be installed separately (not part of this repo).

## Ignored Artifacts
- `dataset.csv`, `ensemble_model.pkl`, `speech.json`, `speech.keras` are generated files and are ignored by `.gitignore`. Do not commit them.

## Conventions
- No specific linting or formatting config; follow existing code style.
- Commit messages are straightforward (no special template).
- Jenkins builds with uv Python 3.9 in a Docker container (see `Jenkinsfile`).
