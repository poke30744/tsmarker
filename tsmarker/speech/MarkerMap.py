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
    def MarkAll(self, videoPath: Path, quiet=False) -> None:
        """
        Mark all clips using LLM

        Args:
            videoPath: Video file path (must exist)
            quiet: Silent mode
        """

        # Validate video file path
        if not videoPath or not videoPath.exists():
            raise FileNotFoundError(f"Video file does not exist: {videoPath}")

        # Prepare subtitle files
        originalSubtitlesPath = self.path.with_suffix(".ass.original")
        generatedSubtitlesPath = self.path.with_suffix(".assgen")
        if not originalSubtitlesPath.exists() or not generatedSubtitlesPath.exists():
            PrepareSubtitlesNew(videoPath, self.ptsMap, quiet=quiet)

        # Load all clip texts
        clips = self.Clips()
        textList = LoadClipTexts(
            videoPath,
            self.ptsMap,
            originalSubtitlesPath,
            generatedSubtitlesPath,
        )

        # Check if there is any text content
        if not any(textList):
            logger.warning("All clips have no text content, skipping marking")
            return

        try:
            # Initialize LLM client
            llm_client = OpenAIClient()

            # Initialize prompt engine
            prompt_engine = PromptEngine(videoPath, self.path)
            program_info = prompt_engine.get_program_info()

            # Get prompt templates
            system_prompt = prompt_engine.get_system_prompt()
            user_prompt_template = prompt_engine.get_user_prompt_template()

            # Filter empty text clips
            non_empty_indices = [i for i, text in enumerate(textList) if text]
            non_empty_texts = [textList[i] for i in non_empty_indices]

            if not non_empty_texts:
                logger.warning("All clips have no text content, skipping marking")
                return

            logger.info(f"Using LLM to analyze {len(non_empty_texts)} clips with text (out of {len(textList)} total clips)...")

            # Only process clips with text
            non_empty_probabilities = llm_client.classify_batch(
                texts=non_empty_texts,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                **program_info,
            )

            # Build complete probability list, empty text clips use default value 0.5
            probabilities = [0.5] * len(textList)  # Default value: uncertain
            for idx, prob in zip(non_empty_indices, non_empty_probabilities):
                probabilities[idx] = prob

            # Mark each clip
            for i, clip in enumerate(clips):
                prob = probabilities[i]
                self.Mark(clip, "speech", float(prob))

            self.Save()
            logger.info(f"Successfully marked {len(clips)} clips ({len(non_empty_texts)} with text, {len(textList)-len(non_empty_texts)} using default value 0.5)")

        except Exception as e:
            logger.error(f"LLM marking failed: {str(e)}")
            raise

