import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pysubs2

from .text_extractor import _extract_ass_from_mkv, _text_for_clip

logger = logging.getLogger("tsmarker.speech.whisper_stt")

REPO_ID = "ggerganov/whisper.cpp"
MODEL_FILE = "ggml-medium-q5_0.bin"
PROGRESS_RE = re.compile(r"whisper_print_progress_callback:\s*progress\s*=\s*(\d+)%")
REPEAT_THRESHOLD = 5  # consecutive identical entries triggers re-recognition
LONG_ENTRY_S = 12.0   # single entry exceeding this duration is flagged


def _find_whisper_cli():
    path = shutil.which("whisper-cli")
    if path is None:
        raise FileNotFoundError("whisper-cli not found on PATH")
    return path


def _get_model_path():
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)


def _run_whisper(whisper_bin: str, model_path: str, wav_path: Path,
                srt_path: Path, extra_flags: list[str] | None = None,
                progress=None, tid: str = "", base_n: float = 0.0):
    """Run whisper-cli, return parsed SRT events (timestamps relative to WAV start)."""
    args = [
        whisper_bin, "-m", model_path, "-f", str(wav_path),
        "-l", "ja", "-osrt", "-of", str(srt_path), "-pp",
    ]
    if extra_flags:
        args.extend(extra_flags)

    proc = subprocess.Popen(args, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    for line in proc.stderr:
        m = PROGRESS_RE.search(line)
        if m and progress is not None:
            pct = int(m.group(1))
            if pct > 0:
                progress.update(tid, base_n + pct / 100)
    proc.wait()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args)

    srt_file = Path(str(srt_path) + ".srt")
    if srt_file.exists():
        return pysubs2.load(srt_file, encoding="utf-8")
    return None


def generate_subtitles(video_path: Path, pts_map, progress=None) -> Path:
    """Generate SRT subtitles for clips without original subtitles using whisper.cpp.

    Runs whisper-cli per-clip, detects repetitive hallucination loops,
    re-recognizes affected segments with --no-fallback, and writes .generated.srt.
    """
    output_path = pts_map.path.with_suffix(".generated.srt")
    clips = pts_map.Clips()

    text_list = [""] * len(clips)
    ass = _extract_ass_from_mkv(video_path)
    if ass is not None:
        text_list = [_text_for_clip(ass, clip) for clip in clips]

    no_sub_clips = [(i, clips[i]) for i in range(len(clips)) if text_list[i] == ""]
    if not no_sub_clips:
        logger.info("All clips have original subtitles, no STT needed")
        return output_path

    whisper_bin = _find_whisper_cli()
    model_path = _get_model_path()

    subs = pysubs2.SSAFile()

    tid = "whisper_stt"
    total = len(no_sub_clips)
    if progress is not None:
        progress.add_task(tid, total, "Whisper STT")

    for ci, (clip_idx, (clip_start, clip_end)) in enumerate(no_sub_clips):
        dur = clip_end - clip_start
        logger.info(
            f"Whisper STT clip {clip_idx}: "
            f"[{clip_start:.1f}s - {clip_end:.1f}s] ({dur:.0f}s) [{ci + 1}/{total}]"
        )

        with tempfile.TemporaryDirectory(prefix="whisper_stt_") as tmp:
            wav_path = Path(tmp) / "audio.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
                 "-ss", str(clip_start), "-to", str(clip_end),
                 "-i", str(video_path),
                 "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(wav_path)],
                check=True, timeout=60)

            srt_subs = _run_whisper(
                whisper_bin, model_path, wav_path, Path(tmp) / "output",
                progress=progress, tid=tid, base_n=float(ci))

            if srt_subs is not None:
                for event in srt_subs.events:
                    event.start += int(clip_start * 1000)
                    event.end += int(clip_start * 1000)
                    subs.events.append(event)

        if progress is not None:
            progress.update(tid, ci + 1)

    # Detect and fix repetitive hallucination loops
    subs.events.sort(key=lambda e: e.start)
    subs.events = _reprocess_repetitions(
        subs.events, video_path, whisper_bin, model_path, progress, tid)

    subs.events = _cleanup_events(subs.events)
    subs.save(str(output_path))
    logger.info(f"Generated {len(subs.events)} subtitle events to {output_path}")

    if progress is not None:
        progress.done(tid)

    return output_path


def _find_repetitive_runs(events):
    """Find runs needing re-recognition: ≥5 consecutive identical texts,
    or single entries exceeding LONG_ENTRY_S."""
    runs = []
    in_run = set()  # indices already covered by a run
    i = 0
    while i < len(events):
        j = i + 1
        while j < len(events) and _norm_text(events[j].text) == _norm_text(events[i].text):
            j += 1
        if j - i >= REPEAT_THRESHOLD:
            runs.append((i, j))
            in_run.update(range(i, j))
        i = j
    # Flag abnormally long single entries not already in runs
    for i, e in enumerate(events):
        if i not in in_run:
            dur = (e.end - e.start) / 1000
            if dur > LONG_ENTRY_S:
                runs.append((i, i + 1))
    runs.sort(key=lambda r: r[0])
    return runs


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _reprocess_repetitions(events, video_path, whisper_bin, model_path, progress, tid):
    """Re-recognize segments with repetitive hallucinations using --no-fallback."""
    runs = _find_repetitive_runs(events)
    if not runs:
        return events

    logger.info(f"Found {len(runs)} repetitive run(s), re-recognizing with --no-fallback")
    fixed = 0

    # Process runs in reverse order so indices stay valid
    for start_idx, end_idx in reversed(runs):
        seg_start = events[start_idx].start / 1000  # ms → seconds
        seg_end = events[end_idx - 1].end / 1000
        dur = seg_end - seg_start
        count = end_idx - start_idx
        text = events[start_idx].text[:60]
        logger.info(f"  [{seg_start:.1f}s - {seg_end:.1f}s] ×{count} \"{text}\"")

        with tempfile.TemporaryDirectory(prefix="whisper_retry_") as tmp:
            wav_path = Path(tmp) / "audio.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
                 "-ss", str(seg_start), "-to", str(seg_end),
                 "-i", str(video_path),
                 "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(wav_path)],
                check=True, timeout=60)

            srt_subs = _run_whisper(
                whisper_bin, model_path, wav_path, Path(tmp) / "output",
                extra_flags=["--no-fallback"])

            if srt_subs is not None and len(srt_subs.events) > 0:
                retry_texts = [_norm_text(e.text) for e in srt_subs.events]

                # Check if re-recognition is still repetitive
                retry_runs = _find_repetitive_runs(srt_subs.events)
                if retry_runs:
                    logger.warning(
                        f"  Re-recognition still has {len(retry_runs)} run(s), "
                        f"falling back to dedup"
                    )
                    # Dedup: keep only the first of each repetitive run
                    keep = []
                    for e in srt_subs.events:
                        offset = int(seg_start * 1000)
                        e.start += offset
                        e.end += offset
                        keep.append(e)
                    # Remove exact duplicates
                    seen = set()
                    deduped = []
                    for e in keep:
                        key = (_norm_text(e.text),)
                        if key not in seen:
                            seen.add(key)
                            deduped.append(e)
                    events[start_idx:end_idx] = deduped
                else:
                    # Success: offset and replace
                    for e in srt_subs.events:
                        offset = int(seg_start * 1000)
                        e.start += offset
                        e.end += offset
                    events[start_idx:end_idx] = srt_subs.events
                    fixed += 1
            else:
                # Whisper produced nothing — dedup original
                events[start_idx:end_idx] = [events[start_idx]]
                logger.warning("  Re-recognition produced no output, deduping")

    logger.info(f"Re-recognition fixed {fixed}/{len(runs)} run(s)")
    return events


def _cleanup_events(events):
    """Remove duplicate events and excessively short ones."""
    import re
    seen = set()
    cleaned = []
    for e in events:
        dur = e.end - e.start
        if dur < 300:  # skip <0.3s fragments
            continue
        text = re.sub(r"\s+", "", e.text)
        if not text:
            continue
        key = (e.start, e.end, text)
        if key not in seen:
            seen.add(key)
            cleaned.append(e)
    return cleaned
