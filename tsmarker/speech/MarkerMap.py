import logging
from pathlib import Path
from .. import common
from .llm_client import OpenAIClient
from .prompt_engine import PromptEngine
from .text_extractor import PrepareSubtitles as PrepareSubtitlesNew, LoadClipTexts

logger = logging.getLogger('tsmarker.speech')

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, progress=None) -> None:
        """Mark all clips using LLM speech analysis."""

        if not videoPath or not videoPath.exists():
            raise FileNotFoundError(f"Video file does not exist: {videoPath}")

        generatedSubtitlesPath = self.path.with_suffix(".generated.srt")
        if not generatedSubtitlesPath.exists():
            PrepareSubtitlesNew(videoPath, self.ptsMap, progress=progress)

        correctedPath = self.path.with_suffix(".corrected.srt")
        if not correctedPath.exists() and generatedSubtitlesPath.exists():
            from ..correct_srt import correct_srt
            correct_srt(videoPath, generatedSubtitlesPath, correctedPath)
            from ..fix_srt_gaps import fix_srt_gaps
            fix_srt_gaps(correctedPath)

        srtPath = correctedPath if correctedPath.exists() else generatedSubtitlesPath
        clips = self.Clips()
        textList = LoadClipTexts(videoPath, self.ptsMap, srtPath)

        if not any(textList):
            logger.warning("All clips have no text content, skipping marking")
            return

        try:
            llm_client = OpenAIClient()
            prompt_engine = PromptEngine(videoPath, self.path)
            program_info = prompt_engine.get_program_info()
            system_prompt = prompt_engine.get_system_prompt()
            user_prompt_template = prompt_engine.get_user_prompt_template()

            non_empty_indices = [i for i, text in enumerate(textList) if text and text.strip()]
            non_empty_texts = [textList[i] for i in non_empty_indices]

            if not non_empty_texts:
                logger.warning("All clips have no text content, skipping marking")
                return

            non_empty_probabilities = llm_client.classify_batch(
                texts=non_empty_texts,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                progress=progress,
                **program_info,
            )

            probabilities = [0.5] * len(textList)
            for idx, prob in zip(non_empty_indices, non_empty_probabilities):
                probabilities[idx] = prob

            for i, clip in enumerate(clips):
                self.Mark(clip, "speech", float(probabilities[i]))

            self.Save()
            logger.info(f"Successfully marked {len(clips)} clips ({len(non_empty_texts)} with text)")

        except Exception as e:
            logger.error(f"LLM marking failed: {str(e)}")
            raise

