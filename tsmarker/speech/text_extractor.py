import logging
from pathlib import Path
import shutil
import tempfile
import json
from tqdm import tqdm
import speech_recognition as sr

from tscutter.ffmpeg import InputFile
from tscutter.common import PtsMap
from ..subtitles import Extract
from .dataset import ExtractSubtitlesText as OriginalExtractSubtitlesText

logger = logging.getLogger("tsmarker.speech.text_extractor")

# 复用dataset.py中的函数
ExtractSubtitlesText = OriginalExtractSubtitlesText


def ExtractAudioText(videoPath: Path, clip: tuple[float, float]) -> str:
    """从音频提取文本（语音识别）"""
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
            quiet=True,
        )
        audioFilename = Path(tmpFolder) / "audio_0.wav"
        try:
            with sr.AudioFile(str(audioFilename)) as source:
                audio = recognizer.record(source)
        except ValueError:
            return ""
    try:
        text = recognizer.recognize_google(audio, language="ja-JP")
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""


def PrepareSubtitles(videoPath: Path, ptsMap: PtsMap, quiet: bool = False):
    """
    准备字幕文件，包括提取原始字幕和生成语音转写字幕

    返回：
        originalSubtitlesPath: 原始字幕文件路径
        generatedSubtitlesPath: 生成的字幕文件路径
    """
    originalSubtitlesPath = ptsMap.path.with_suffix(".ass.original")
    generatedSubtitlesPath = ptsMap.path.with_suffix(".assgen")

    # 如果文件已存在，直接返回
    if originalSubtitlesPath.exists() and generatedSubtitlesPath.exists():
        return originalSubtitlesPath, generatedSubtitlesPath

    # 提取原始字幕
    if not originalSubtitlesPath.exists():
        with tempfile.TemporaryDirectory(prefix="ExtractSubtitles_") as tmpFolder:
            for sub in Extract(videoPath, Path(tmpFolder)):
                if sub.suffix == ".ass":
                    shutil.copy(sub, originalSubtitlesPath)
                    break

    # 提取每个clip的文本
    clips = ptsMap.Clips()
    textList = (
        [ExtractSubtitlesText(originalSubtitlesPath, clip) for clip in clips]
        if originalSubtitlesPath.exists()
        else [""] * len(clips)
    )

    # 对于没有字幕的clip，从音频提取
    generatedSubtitles = {}
    for i in tqdm(range(len(clips)), disable=quiet):
        if textList[i] == "":
            textList[i] = ExtractAudioText(videoPath, clips[i])
            generatedSubtitles[str(clips[i])] = textList[i]

    # 保存生成的字幕
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
    加载所有clip的文本

    Args:
        videoPath: 视频文件路径
        ptsMap: PTS映射
        originalSubtitlesPath: 原始字幕文件路径
        generatedSubtitlesPath: 生成的字幕文件路径

    Returns:
        文本列表，每个元素对应一个clip
    """
    clips = ptsMap.Clips()

    # 从原始字幕提取
    textList = (
        [ExtractSubtitlesText(originalSubtitlesPath, clip) for clip in clips]
        if originalSubtitlesPath.exists()
        else [""] * len(clips)
    )

    # 从生成的字幕补充
    if generatedSubtitlesPath.exists():
        with generatedSubtitlesPath.open() as f:
            generatedSubtitles = json.load(f)

        for i in range(len(clips)):
            if textList[i] == "":
                clip_key = str(clips[i])
                if clip_key in generatedSubtitles:
                    textList[i] = generatedSubtitles[clip_key]

    return textList
