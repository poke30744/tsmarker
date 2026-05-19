import logging
import re
from pathlib import Path
import subprocess
import tempfile

import pysubs2

from ..ptsmap import PtsMap
from .dataset import ExtractSubtitlesText as OriginalExtractSubtitlesText

logger = logging.getLogger("tsmarker.speech.text_extractor")

# Reuse functions from dataset.py
ExtractSubtitlesText = OriginalExtractSubtitlesText


def _extract_ass_from_mkv(video_path: Path) -> pysubs2.SSAFile | None:
    """Extract embedded ASS from MKV to a temp file, return parsed subtitles."""
    with tempfile.NamedTemporaryFile(suffix='.ass', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(['ffmpeg', '-y', '-hide_banner',
                        '-i', str(video_path), '-map', '0:s:0', '-c:s', 'copy',
                        str(tmp_path)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if tmp_path.stat().st_size > 0:
            return pysubs2.load(tmp_path, encoding='utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    finally:
        tmp_path.unlink(missing_ok=True)
    return None


def _text_for_clip(ass: pysubs2.SSAFile, clip: tuple[float, float]) -> str:
    """Extract concatenated text from parsed ASS events overlapping a clip."""
    texts = []
    start_ms, end_ms = clip[0] * 1000, clip[1] * 1000
    for event in ass.events:
        if event.start < end_ms and start_ms < event.end:
            text = re.sub(r'\{.*?\}', '', event.text)
            text = text.replace(r'\N', '')
            texts.append(text)
    return ' '.join(texts)


def PrepareSubtitles(videoPath: Path, ptsMap: PtsMap, progress=None):
    """Generate speech-to-text SRT subtitles for clips without original subtitles.

    Uses faster-whisper to generate .generated.srt. Original subtitles are
    read from the MKV on demand (no .ass.original persisted).
    """

    generatedSubtitlesPath = ptsMap.path.with_suffix(".generated.srt")

    if not generatedSubtitlesPath.exists():
        from .whisper_stt import generate_ass_subtitles
        generate_ass_subtitles(videoPath, ptsMap, progress=progress)

    return generatedSubtitlesPath


def LoadClipTexts(
    videoPath: Path,
    ptsMap: PtsMap,
    subtitlesPath: Path,
) -> list[str]:
    """Load all clip texts from original (MKV-embedded) and STT subtitles.

    subtitlesPath should point to .corrected.srt (preferred) or .generated.srt.
    """
    clips = ptsMap.Clips()
    textList = [""] * len(clips)

    # Try original subtitles from MKV
    ass = _extract_ass_from_mkv(videoPath)
    if ass is not None:
        for i, clip in enumerate(clips):
            textList[i] = _text_for_clip(ass, clip)

    # Supplement from STT subtitles
    if subtitlesPath.exists():
        for i in range(len(clips)):
            if textList[i] == "":
                text = ExtractSubtitlesText(subtitlesPath, clips[i])
                if text:
                    textList[i] = text

    return textList
