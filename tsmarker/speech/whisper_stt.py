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


def _find_whisper_cli():
    path = shutil.which("whisper-cli")
    if path is None:
        raise FileNotFoundError("whisper-cli not found on PATH")
    return path


def _get_model_path():
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)


def generate_subtitles(video_path: Path, pts_map, progress=None) -> Path:
    """Generate SRT subtitles for clips without original subtitles using whisper.cpp.

    Extracts embedded ASS from MKV to find clips without original text, runs
    whisper-cli on each clip individually, and writes .generated.srt.
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

            srt_path = Path(tmp) / "output"
            args = [
                whisper_bin,
                "-m", model_path,
                "-f", str(wav_path),
                "-l", "ja",
                "-osrt",
                "-of", str(srt_path),
                "-pp",
            ]

            proc = subprocess.Popen(
                args,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            for line in proc.stderr:
                m = PROGRESS_RE.search(line)
                if m:
                    pct = int(m.group(1))
                    if progress is not None and pct > 0:
                        progress.update(tid, ci + pct / 100)
            proc.wait()

            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, args)

            srt_file = Path(str(srt_path) + ".srt")
            if srt_file.exists():
                srt_subs = pysubs2.load(srt_file, encoding="utf-8")
                for event in srt_subs.events:
                    event.start += int(clip_start * 1000)
                    event.end += int(clip_start * 1000)
                    subs.events.append(event)

        if progress is not None:
            progress.update(tid, ci + 1)

    subs.events.sort(key=lambda e: e.start)
    subs.events = _cleanup_events(subs.events)
    subs.save(str(output_path))
    logger.info(f"Generated {len(subs.events)} subtitle events to {output_path}")

    if progress is not None:
        progress.done(tid)

    return output_path


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
