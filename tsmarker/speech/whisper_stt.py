import logging
import tempfile
import subprocess
from pathlib import Path

from .text_extractor import _extract_ass_from_mkv, _text_for_clip

logger = logging.getLogger("tsmarker.speech.whisper_stt")

_whisper_model = None


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            "small", device="cpu", compute_type="int8",
            download_root=str(Path(__file__).resolve().parent.parent.parent / ".models"),
        )
    return _whisper_model


def generate_ass_subtitles(video_path: Path, pts_map, progress=None) -> Path:
    """Generate ASS subtitles for clips without original subtitles using Whisper.

    Extracts embedded ASS from MKV to find clips without original text, groups
    consecutive ones, runs faster-whisper on merged audio, and writes
    .generated.ass with PTS-based timestamps.
    """
    import pysubs2

    output_path = pts_map.path.with_suffix(".generated.srt")
    clips = pts_map.Clips()

    # Find clips without original subtitle text
    text_list = [""] * len(clips)
    ass = _extract_ass_from_mkv(video_path)
    if ass is not None:
        text_list = [_text_for_clip(ass, clip) for clip in clips]

    # Group consecutive no-subtitle clips
    groups = _group_no_subtitle_clips(clips, text_list)
    if not groups:
        logger.info("All clips have original subtitles, no STT needed")
        return output_path

    model = _get_model()

    # Collect all generated subtitle events
    subs = pysubs2.SSAFile()

    tid = "whisper_stt"
    if progress is not None:
        progress.add_task(tid, len(groups), "Whisper STT")

    for gi, group in enumerate(groups):
        merged_start = group["merged_start"]
        merged_end = group["merged_end"]
        dur = merged_end - merged_start
        logger.info(f"Whisper STT: [{merged_start:.1f}s - {merged_end:.1f}s] "
                     f"({dur:.0f}s, {len(group['clip_indices'])} clips)")

        with tempfile.TemporaryDirectory(prefix="whisper_stt_") as tmp:
            wav_path = Path(tmp) / "audio.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
                 "-ss", str(merged_start), "-to", str(merged_end),
                 "-i", str(video_path),
                 "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(wav_path)],
                check=True, timeout=60)

            segments, _info = model.transcribe(
                str(wav_path), language="ja", word_timestamps=True,
                vad_filter=True)

            for seg in segments:
                # Convert to PTS timestamps (same domain as .ass.original)
                event = pysubs2.SSAEvent()
                event.start = int((merged_start + seg.start) * 1000)  # ms
                event.end = int((merged_start + seg.end) * 1000)
                event.text = seg.text.strip()
                subs.events.append(event)

        if progress is not None:
            progress.update(tid, gi + 1)

    # Sort events by start time
    subs.events.sort(key=lambda e: e.start)

    # Remove duplicate/short events
    subs.events = _cleanup_events(subs.events)

    # Write ASS file
    subs.save(str(output_path))
    logger.info(f"Generated {len(subs.events)} subtitle events to {output_path}")

    if progress is not None:
        progress.done(tid)

    return output_path


def _group_no_subtitle_clips(clips, text_list):
    """Group consecutive clips that have no original subtitle text."""
    groups = []
    i = 0
    while i < len(clips):
        if text_list[i] == "":
            j = i
            while j < len(clips) and text_list[j] == "":
                j += 1
            start = clips[i][0]
            end = clips[j - 1][1]
            groups.append({
                "merged_start": start,
                "merged_end": end,
                "clip_indices": list(range(i, j)),
            })
            i = j
        else:
            i += 1
    return groups


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
