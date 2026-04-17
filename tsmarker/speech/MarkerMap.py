import logging
import os
from pathlib import Path
import shutil
import tempfile, json
from tqdm import tqdm
import speech_recognition as sr
from .. import common
from .dataset import ExtractSubtitlesText
from ..subtitles import Extract
from tscutter.ffmpeg import InputFile
from tscutter.common import PtsMap
from .llm_client import OpenAIClient
from .prompt_engine import PromptEngine
from .text_extractor import PrepareSubtitles as PrepareSubtitlesNew, LoadClipTexts

logger = logging.getLogger('tsmarker.speech')

def ExtractAudioText(videoPath: Path, clip: tuple[float, float]) -> str:
    recognizer = sr.Recognizer()
    inputFile = InputFile(videoPath)
    with tempfile.TemporaryDirectory(prefix='ExtractAudioText_') as tmpFolder:
        inputFile.ExtractStream(output=Path(tmpFolder), ss=clip[0], to=clip[1], toWav=True, videoTracks=[], audioTracks=[0], quiet=True)
        audioFilename = Path(tmpFolder) / 'audio_0.wav'
        try:
            with sr.AudioFile(str(audioFilename)) as source:
                audio = recognizer.record(source)
        except ValueError: 
            return ''
    try:
        text = recognizer.recognize_google(audio, language='ja-JP')
        return text
    except sr.UnknownValueError:
        return ''
    except sr.RequestError:
        return ''

def PrepareSubtitles(videoPath: Path, ptsMap: PtsMap, quiet: bool = False):
    originalSubtitlesPath = ptsMap.path.with_suffix('.ass.original')
    with tempfile.TemporaryDirectory(prefix='ExtractSubtitles_') as tmpFolder:
        for sub in Extract(videoPath, Path(tmpFolder)):
            if sub.suffix == '.ass':
                shutil.copy(sub, originalSubtitlesPath)
                break
    clips = ptsMap.Clips()
    textList = [ ExtractSubtitlesText(originalSubtitlesPath, clip) for clip in clips ] if originalSubtitlesPath.exists() else [ '' ] * len(clips)
    # for clips without subtitles, extract it from audio
    generatedSubtitlesPath = ptsMap.path.with_suffix('.assgen')
    generatedSubtitles = {}
    for i in tqdm(range(len(clips)), disable=quiet):
        if textList[i] == '':
            textList[i] = ExtractAudioText(videoPath, clips[i])
            generatedSubtitles[str(clips[i])] = textList[i]
    # save generated subtitles
    with generatedSubtitlesPath.open('w') as f:
        json.dump(generatedSubtitles, f, ensure_ascii=False, indent=True)

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, apiUrl: str, quiet=False) -> None:
        """
        使用LLM标记所有clip

        Args:
            videoPath: 视频文件路径（必须存在）
            apiUrl: 被忽略的参数（保持接口兼容性）
            quiet: 静默模式
        """

        # 验证视频文件路径
        if not videoPath or not videoPath.exists():
            raise FileNotFoundError(f"视频文件不存在: {videoPath}")

        # 准备字幕文件
        originalSubtitlesPath = self.path.with_suffix(".ass.original")
        generatedSubtitlesPath = self.path.with_suffix(".assgen")
        if not originalSubtitlesPath.exists() or not generatedSubtitlesPath.exists():
            PrepareSubtitlesNew(videoPath, self.ptsMap, quiet=quiet)

        # 加载所有clip文本
        clips = self.Clips()
        textList = LoadClipTexts(
            videoPath,
            self.ptsMap,
            originalSubtitlesPath,
            generatedSubtitlesPath,
        )

        # 检查是否有文本内容
        if not any(textList):
            logger.warning("所有clip都没有文本内容，跳过标记")
            return

        try:
            # 初始化LLM客户端
            llm_client = OpenAIClient()

            # 初始化提示引擎
            prompt_engine = PromptEngine(videoPath)
            program_info = prompt_engine.get_program_info()

            # 获取提示模板
            system_prompt = prompt_engine.get_system_prompt()
            user_prompt_template = prompt_engine.get_user_prompt_template()

            # 批量分类
            logger.info(f"使用LLM分析{len(textList)}个clip...")
            probabilities = llm_client.classify_batch(
                texts=textList,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                **program_info,
            )

            # 标记每个clip
            for i, clip in enumerate(clips):
                prob = probabilities[i] if i < len(probabilities) else 0.5
                self.Mark(clip, "speech", float(prob))

            self.Save()
            logger.info(f"成功标记{len(clips)}个clip")

        except Exception as e:
            logger.error(f"LLM标记失败: {str(e)}")
            raise

def ReMarkAll(markermapFolder: Path, apiUrl: str, quiet=False):
    files = []
    for markerPath in tqdm(markermapFolder.glob('**/*.markermap'), desc='searching *.markermap ...', disable=quiet):
        indexPath = markerPath.with_suffix('.ptsmap')
        originalSubtitlesPath = markerPath.with_suffix('.ass.original')
        generatedSubtitlesPath = markerPath.with_suffix('.assgen')
        if indexPath.exists() and originalSubtitlesPath.exists() and generatedSubtitlesPath.exists():
            files.append(markerPath)
    logger.info(f"Will re-mark {len(files)} files")
    for markerPath in tqdm(files, desc="re-marking ...", disable=quiet):
        indexPath = markerPath.with_suffix(".ptsmap")
        markermap = MarkerMap(markerPath, PtsMap(indexPath))

        # 推断视频文件路径
        # 尝试encoded目录下的.mp4文件（可能在子目录中）
        video_path = markerPath.parent.parent / f"{markerPath.stem}.mp4"
        if not video_path.exists():
            # 尝试raw目录下的.m2ts文件
            # encoded下有子目录时：parent.parent.parent.parent为TestFiles目录
            raw_dir = markerPath.parent.parent.parent.parent / "raw"
            video_path = raw_dir / f"{markerPath.stem}.m2ts"

        if video_path.exists():
            markermap.MarkAll(video_path, apiUrl, quiet=quiet)
        else:
            logger.warning(f"找不到视频文件，跳过标记: {markerPath.stem}")
            continue