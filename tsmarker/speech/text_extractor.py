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

    # 记录文件状态
    logger.info(f"字幕文件状态: original={originalSubtitlesPath.exists()}, generated={generatedSubtitlesPath.exists()}")
    logger.info(f"处理 {len(clips)} 个clip")

    # 从原始字幕提取
    textList = (
        [ExtractSubtitlesText(originalSubtitlesPath, clip) for clip in clips]
        if originalSubtitlesPath.exists()
        else [""] * len(clips)
    )

    # 跟踪来源
    source_list = ["none"] * len(clips)
    for i, text in enumerate(textList):
        if text:
            source_list[i] = "original"

    # 统计原始字幕结果
    original_count = sum(1 for source in source_list if source == "original")
    logger.info(f"原始字幕提供 {original_count}/{len(clips)} 个clip文本")

    # 从生成的字幕补充
    generated_count = 0
    if generatedSubtitlesPath.exists():
        with generatedSubtitlesPath.open() as f:
            generatedSubtitles = json.load(f)

        for i in range(len(clips)):
            if textList[i] == "":
                clip_key = str(clips[i])
                if clip_key in generatedSubtitles:
                    generated_text = generatedSubtitles[clip_key]
                    if generated_text:  # 只记录非空文本
                        textList[i] = generated_text
                        source_list[i] = "generated"
                        generated_count += 1
                        logger.debug(f"Clip {i} 使用生成字幕: {generated_text[:100]}...")
                    else:
                        logger.debug(f"Clip {i} 生成字幕为空，跳过")

    if generated_count > 0:
        logger.info(f"生成字幕补充 {generated_count}/{len(clips)} 个clip文本")

    # 最终统计
    total_with_text = sum(1 for text in textList if text)
    logger.info(f"最终 {total_with_text}/{len(clips)} 个clip有文本内容")

    # 记录来源统计
    source_summary = {}
    for source in source_list:
        source_summary[source] = source_summary.get(source, 0) + 1
    logger.info(f"字幕来源统计: {source_summary}")

    # 记录部分clip文本示例（最多5个）
    recorded = 0
    for i, (text, source) in enumerate(zip(textList, source_list)):
        if text and recorded < 5:
            logger.debug(f"Clip {i} 示例 ({source}): {text[:100]}...")
            recorded += 1

    return textList
