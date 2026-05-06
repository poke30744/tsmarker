import logging
from pathlib import Path
import shutil
import tempfile
import json
import time
import speech_recognition as sr

from tscutter.ffmpeg import InputFile
from tscutter.common import PtsMap
from ..subtitles import Extract
from .dataset import ExtractSubtitlesText as OriginalExtractSubtitlesText

logger = logging.getLogger("tsmarker.speech.text_extractor")

# Reuse functions from dataset.py
ExtractSubtitlesText = OriginalExtractSubtitlesText


def ExtractAudioText(videoPath: Path, clip: tuple[float, float]) -> str:
    """Extract text from audio (speech recognition)"""
    recognizer = sr.Recognizer()
    inputFile = InputFile(videoPath)
    with tempfile.TemporaryDirectory(prefix="ExtractAudioText_") as tmpFolder:
        inputFile.ExtractStream(
            output=Path(tmpFolder),
            ss=clip[0],
            to=clip[1],
            toWav=True,
            videoTracks=[],
            audioTracks=[0],
        )
        audioFilename = Path(tmpFolder) / "audio_0.wav"
        try:
            with sr.AudioFile(str(audioFilename)) as source:
                audio = recognizer.record(source)
        except ValueError:
            return ""
    last_error = None
    for attempt in range(3):
        try:
            text = recognizer.recognize_google(audio, language="ja-JP")
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Speech recognition failed after 3 retries: {last_error}"
    )


def PrepareSubtitles(videoPath: Path, ptsMap: PtsMap, progress=None):
    """Prepare subtitle files: extract original subtitles and generate speech-to-text."""

    originalSubtitlesPath = ptsMap.path.with_suffix(".ass.original")
    generatedSubtitlesPath = ptsMap.path.with_suffix(".assgen")

    if originalSubtitlesPath.exists() and generatedSubtitlesPath.exists():
        return originalSubtitlesPath, generatedSubtitlesPath

    if not originalSubtitlesPath.exists():
        with tempfile.TemporaryDirectory(prefix="ExtractSubtitles_") as tmpFolder:
            for sub in Extract(videoPath, Path(tmpFolder)):
                if sub.suffix == ".ass":
                    shutil.copy(sub, originalSubtitlesPath)
                    break

    clips = ptsMap.Clips()
    textList = (
        [ExtractSubtitlesText(originalSubtitlesPath, clip) for clip in clips]
        if originalSubtitlesPath.exists()
        else [""] * len(clips)
    )

    generatedSubtitles = {}
    tid = "speech_to_text"
    if progress is not None:
        progress.add_task(tid, len(clips), "Speech-to-text")
    for i, clip in enumerate(clips):
        if textList[i] == "":
            textList[i] = ExtractAudioText(videoPath, clips[i])
            generatedSubtitles[str(clips[i])] = textList[i]
        if progress is not None:
            progress.update(tid, i + 1)
    if progress is not None:
        progress.done(tid)

    with generatedSubtitlesPath.open("w") as f:
        json.dump(generatedSubtitles, f, ensure_ascii=False, indent=True)

    return originalSubtitlesPath, generatedSubtitlesPath


def LoadClipTexts(
    videoPath: Path,
    ptsMap: PtsMap,
    originalSubtitlesPath: Path,
    generatedSubtitlesPath: Path,
) -> list[str]:
    """
    Load all clip texts

    Args:
        videoPath: Video file path
        ptsMap: PTS map
        originalSubtitlesPath: Original subtitle file path
        generatedSubtitlesPath: Generated subtitle file path

    Returns:
        Text list, each element corresponds to a clip
    """
    clips = ptsMap.Clips()

    # Log file status
    logger.info(f"Subtitle file status: original={originalSubtitlesPath.exists()}, generated={generatedSubtitlesPath.exists()}")
    logger.info(f"Processing {len(clips)} clips")

    # Extract from original subtitles
    textList = (
        [ExtractSubtitlesText(originalSubtitlesPath, clip) for clip in clips]
        if originalSubtitlesPath.exists()
        else [""] * len(clips)
    )

    # Track sources
    source_list = ["none"] * len(clips)
    for i, text in enumerate(textList):
        if text:
            source_list[i] = "original"

    # Count original subtitle results
    original_count = sum(1 for source in source_list if source == "original")
    logger.info(f"Original subtitles provide {original_count}/{len(clips)} clip texts")

    # Supplement from generated subtitles
    generated_count = 0
    if generatedSubtitlesPath.exists():
        with generatedSubtitlesPath.open() as f:
            generatedSubtitles = json.load(f)

        for i in range(len(clips)):
            if textList[i] == "":
                clip_key = str(clips[i])
                if clip_key in generatedSubtitles:
                    generated_text = generatedSubtitles[clip_key]
                    if generated_text:  # Only record non-empty text
                        textList[i] = generated_text
                        source_list[i] = "generated"
                        generated_count += 1
                        logger.info(f"Clip {i} using generated subtitle: {generated_text[:100]}...")
                    else:
                        logger.info(f"Clip {i} generated subtitle is empty, skipping")

    if generated_count > 0:
        logger.info(f"Generated subtitles supplement {generated_count}/{len(clips)} clip texts")

    # Final statistics
    total_with_text = sum(1 for text in textList if text)
    logger.info(f"Finally {total_with_text}/{len(clips)} clips have text content")

    # Log source statistics
    source_summary = {}
    for source in source_list:
        source_summary[source] = source_summary.get(source, 0) + 1
    logger.info(f"Subtitle source statistics: {source_summary}")

    # Log some clip text examples (up to 5)
    recorded = 0
    for i, (text, source) in enumerate(zip(textList, source_list)):
        if text and recorded < 5:
            logger.info(f"Clip {i} example ({source}): {text[:100]}...")
            recorded += 1

    return textList
