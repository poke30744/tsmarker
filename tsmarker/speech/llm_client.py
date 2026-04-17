import os
import logging
from typing import List, Optional
from openai import OpenAI

logger = logging.getLogger("tsmarker.speech.llm")


class OpenAIClient:
    """简化版OpenAI客户端，仅支持聊天完成接口"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化OpenAI客户端

        Args:
            api_key: OpenAI API密钥，如未提供则从环境变量读取
            base_url: API基础URL，如未提供则从环境变量读取或使用默认值
            model: 模型名称，如未提供则从环境变量读取或使用默认值
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
        **prompt_kwargs,
    ) -> List[float]:
        """
        批量分类文本，返回每个文本为广告的概率

        Args:
            texts: 文本列表
            system_prompt: 系统提示
            user_prompt_template: 用户提示模板
            **prompt_kwargs: 提示模板参数

        Returns:
            概率列表，范围0.0-1.0
        """
        if not texts:
            return []

        # 格式化clip文本
        clip_texts_formatted = self._format_clip_texts(texts)
        prompt_kwargs["clip_texts_formatted"] = clip_texts_formatted

        # 构建用户提示
        user_prompt = user_prompt_template.format(**prompt_kwargs)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=float(os.getenv("SPEECH_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("SPEECH_MAX_TOKENS", "2000")),
                timeout=float(os.getenv("OPENAI_TIMEOUT", "30.0")),
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI API")

            # 解析响应
            probabilities = self._parse_response(content, len(texts))
            return probabilities

        except Exception as e:
            logger.error(f"OpenAI API调用失败: {str(e)}")
            raise

    def _format_clip_texts(self, texts: List[str]) -> str:
        """格式化clip文本列表为提示字符串"""
        formatted = []
        for i, text in enumerate(texts, 1):
            # 如果文本太长，适当截断
            if len(text) > 500:
                text = text[:497] + "..."
            formatted.append(f"{i}. {text}")
        return "\n\n".join(formatted)

    def _parse_response(self, response_text: str, expected_count: int) -> List[float]:
        """
        解析LLM响应，提取概率值

        支持格式：
        1. AD: 0.95 理由
        2. AD: 0.30 理由
        或
        1. AD: 0.95 理由
        2. AD: 0.20 理由
        """
        probabilities = []

        # 按行分割
        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 匹配格式：序号. AD: 概率 理由
            # 支持多种格式变体
            import re

            # 模式1: "1. AD: 0.95 理由"
            pattern1 = r"^\d+\.\s*AD:\s*([0-9]*\.?[0-9]+)"
            # 模式2: "1. AD: 0.95 理由" (无空格)
            pattern2 = r"^\d+\.\s*AD:\s*([0-9]*\.?[0-9]+)"
            # 模式3: "AD: 0.95 理由" (可能无序)
            pattern3 = r"AD:\s*([0-9]*\.?[0-9]+)"

            match = None
            for pattern in [pattern1, pattern2, pattern3]:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    break

            if match:
                try:
                    prob = float(match.group(1))
                    # 确保概率在0-1范围内
                    prob = max(0.0, min(1.0, prob))
                    probabilities.append(prob)
                except ValueError:
                    logger.warning(f"无法解析概率值: {line}")
                    probabilities.append(0.5)  # 默认值
            else:
                logger.warning(f"无法解析行: {line}")
                probabilities.append(0.5)  # 默认值

        # 如果解析出的数量不足，用默认值填充
        if len(probabilities) < expected_count:
            logger.warning(
                f"解析出的概率数量({len(probabilities)})少于期望值({expected_count})，用默认值填充"
            )
            probabilities.extend([0.5] * (expected_count - len(probabilities)))

        # 如果解析出的数量过多，截断
        if len(probabilities) > expected_count:
            probabilities = probabilities[:expected_count]

        return probabilities
