import logging
from pathlib import Path
from .. import common
from .llm_client import OpenAIClient
from .prompt_engine import PromptEngine
from .text_extractor import LoadClipTexts

logger = logging.getLogger('tsmarker.speech')

class MarkerMap(common.MarkerMap):
    def MarkAll(self, videoPath: Path, progress=None) -> None:
        """Mark all clips using LLM speech analysis."""

        if not videoPath or not videoPath.exists():
            raise FileNotFoundError(f"Video file does not exist: {videoPath}")

        originalSubtitlesPath = self.path.with_suffix(".ass.original")
        generatedSubtitlesPath = self.path.with_suffix(".assgen")

        clips = self.Clips()
        textList = LoadClipTexts(
            videoPath, self.ptsMap, originalSubtitlesPath, generatedSubtitlesPath)

        try:
            llm_client = OpenAIClient()
            prompt_engine = PromptEngine(videoPath, self.path)
            program_info = prompt_engine.get_program_info()
            system_prompt = prompt_engine.get_system_prompt()
            user_prompt_template = prompt_engine.get_user_prompt_template()

            non_empty_indices = [i for i, text in enumerate(textList) if text and text.strip()]
            non_empty_texts = [textList[i] for i in non_empty_indices]

            if not non_empty_texts:
                logger.warning("All clips have no text content, using default 0.5")

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

