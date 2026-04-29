import logging
from pathlib import Path
from .. import common
from .llm_client import OpenAIClient
from .prompt_engine import PromptEngine
from .text_extractor import PrepareSubtitles as PrepareSubtitlesNew, LoadClipTexts

logger = logging.getLogger('tsmarker.speech')

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

            # Filter empty/whitespace-only text clips
            non_empty_indices = [i for i, text in enumerate(textList) if text and text.strip()]
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

