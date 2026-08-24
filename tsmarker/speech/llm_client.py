import os
import logging
from typing import List, Optional
from openai import OpenAI

logger = logging.getLogger("tsmarker.speech.llm")


class OpenAIClient:
    """Simplified OpenAI client, only supports chat completion API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize OpenAI client

        Args:
            api_key: OpenAI API key, read from environment variable if not provided
            base_url: API base URL, read from environment variable or use default if not provided
            model: Model name, read from environment variable or use default if not provided
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        logger.info(
            f"OpenAI client initialized with model: {self.model}, base_url: {self.base_url}"
        )

    def classify_batch(
        self,
        texts: List[str],
        system_prompt: str,
        user_prompt_template: str,
        progress=None,
        **prompt_kwargs,
    ) -> List[float]:
        if not texts:
            return []
        return self._classify_iterative(texts, system_prompt, user_prompt_template, progress=progress, **prompt_kwargs)


    def _classify_iterative(
        self,
        texts: List[str],
        system_prompt: str,
        user_prompt_template: str,
        progress=None,
        **prompt_kwargs,
    ) -> List[float]:
        probabilities = []
        total = len(texts)

        tid = "speech_llm"
        if progress is not None:
            progress.add_task(tid, total, "LLM speech analysis")

        # Format system prompt (if it contains placeholders)
        if "{" in system_prompt:
            formatted_system_prompt = system_prompt.format(**prompt_kwargs)
        else:
            formatted_system_prompt = system_prompt  # Generic prompt, no formatting needed

        # Initialize message history, only contains system prompt
        messages = [
            {"role": "system", "content": formatted_system_prompt}
        ]

        for idx, text in enumerate(texts, start=1):
            # Use clip text, no numbering prefix needed
            prompt_kwargs["clip_texts_formatted"] = text

            # Build user prompt
            user_prompt = user_prompt_template.format(**prompt_kwargs)

            # Add user message to history
            messages.append({"role": "user", "content": user_prompt})

            # Log request (log system prompt first time, then only user prompts)
            log_system = (idx == 1)
            self._log_request(formatted_system_prompt, user_prompt, 1, idx, total, log_system=log_system)

            max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
            for attempt in range(max_retries + 1):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=float(os.getenv("SPEECH_TEMPERATURE", "0.1")),
                    max_tokens=int(os.getenv("SPEECH_MAX_TOKENS", "500")),
                    timeout=float(os.getenv("OPENAI_TIMEOUT", "30.0")),
                    extra_body={"thinking": {"type": "disabled"}},
                )

                content = response.choices[0].message.content
                if content:
                    break
                if attempt < max_retries:
                    logger.warning(f"Empty response from API for clip {idx}/{total}, retrying ({attempt + 1}/{max_retries})...")
            else:
                raise ValueError("Empty response from OpenAI API")

            # Log response
            self._log_response(content)

            # Parse single response
            prob = self._parse_single_response(content)
            probabilities.append(prob)

            # Add assistant response to history for complete context in next clip
            messages.append({"role": "assistant", "content": content})

            logger.info(f"Processing clip {idx}/{total}: probability={prob}")
            if progress is not None:
                progress.update(tid, idx)

        if progress is not None:
            progress.done(tid)
        return probabilities

    def _log_request(self, system_prompt: str, user_prompt: str, clip_count: int, current_idx: int = None, total: int = None, log_system: bool = True):
        """Log request"""
        if current_idx is not None and total is not None:
            logger.info(f"Requesting clip {current_idx}/{total} (total {clip_count} clips)")
        else:
            logger.info(f"Requesting {clip_count} clips")
        # Only log system prompt (if first time) and user prompt
        if log_system:
            logger.info(f"System prompt: {system_prompt}")
        logger.info(f"User prompt: {user_prompt}")

    def _log_response(self, response_text: str):
        """Log response"""
        logger.info(f"Response: {response_text}")

    def _parse_single_response(self, response_text: str) -> float:
        """
        Parse single clip response, extract probability value

        Supported formats (probability may optionally be wrapped in brackets):
        - AD: 0.95 reason
        - AD: [0.95] reason
        - 1. AD: [0.95] reason
        """
        import re
        lines = response_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Match formats: number. AD: probability reason or AD: probability reason
            # Probability may be wrapped in brackets, e.g. AD: [0.95] reason
            patterns = [
                r"^\d+\.\s*AD:\s*\[?([0-9]*\.?[0-9]+)\]?",  # 1. AD: [0.95]
                r"AD:\s*\[?([0-9]*\.?[0-9]+)\]?",           # AD: [0.95]
            ]
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        prob = float(match.group(1))
                        # Ensure probability is in 0-1 range
                        prob = max(0.0, min(1.0, prob))
                        return prob
                    except ValueError:
                        raise ValueError(f"Cannot parse probability value: {line}")
        raise ValueError(f"Cannot parse LLM response: {response_text[:200]}...")

