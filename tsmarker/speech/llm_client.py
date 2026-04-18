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
        分类文本，返回每个文本为广告的概率（使用对话模式，每次处理一个clip）

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
        return self._classify_iterative(texts, system_prompt, user_prompt_template, **prompt_kwargs)


    def _classify_iterative(
        self,
        texts: List[str],
        system_prompt: str,
        user_prompt_template: str,
        **prompt_kwargs,
    ) -> List[float]:
        """
        迭代分类文本（使用对话模式，系统提示只发送一次）
        """
        probabilities = []
        total = len(texts)

        # 格式化系统提示（如果包含占位符）
        if "{" in system_prompt:
            formatted_system_prompt = system_prompt.format(**prompt_kwargs)
        else:
            formatted_system_prompt = system_prompt  # 通用提示，不需要格式化

        # 初始化消息历史，只包含系统提示
        messages = [
            {"role": "system", "content": formatted_system_prompt}
        ]

        for idx, text in enumerate(texts, start=1):
            # 格式化单个clip文本
            single_text_formatted = f"1. {text}"
            prompt_kwargs["clip_texts_formatted"] = single_text_formatted

            # 构建用户提示
            user_prompt = user_prompt_template.format(**prompt_kwargs)

            # 添加用户消息到历史
            messages.append({"role": "user", "content": user_prompt})

            # 记录请求（第一次记录系统提示，之后只记录用户提示）
            log_system = (idx == 1)
            self._log_request(formatted_system_prompt, user_prompt, 1, idx, total, log_system=log_system)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=float(os.getenv("SPEECH_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("SPEECH_MAX_TOKENS", "500")),
                timeout=float(os.getenv("OPENAI_TIMEOUT", "30.0")),
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI API")

            # 记录响应
            self._log_response(content)

            # 解析单个响应
            prob = self._parse_single_response(content)
            probabilities.append(prob)

            # 添加助手响应到历史，以便下一个clip有完整上下文
            messages.append({"role": "assistant", "content": content})

            logger.info(f"处理clip {idx}/{total}: 概率={prob}")

        return probabilities

    def _log_request(self, system_prompt: str, user_prompt: str, clip_count: int, current_idx: int = None, total: int = None, log_system: bool = True):
        """记录请求日志"""
        if current_idx is not None and total is not None:
            logger.info(f"请求 clip {current_idx}/{total} (共{clip_count}个clip)")
        else:
            logger.info(f"请求 {clip_count}个clip")
        # 只记录系统提示（如果是第一次）和用户提示
        if log_system:
            logger.info(f"系统提示: {system_prompt}")
        logger.info(f"用户提示: {user_prompt}")

    def _log_response(self, response_text: str):
        """记录响应日志"""
        logger.info(f"响应: {response_text}")

    def _parse_single_response(self, response_text: str) -> float:
        """
        解析单个clip的响应，提取概率值

        支持格式：
        1. AD: 0.95 理由
        2. AD: 0.30 理由
        或
        1. AD: 0.95 理由
        """
        import re
        lines = response_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 匹配格式：序号. AD: 概率 理由 或 AD: 概率 理由
            patterns = [
                r"^\d+\.\s*AD:\s*([0-9]*\.?[0-9]+)",  # 1. AD: 0.95
                r"AD:\s*([0-9]*\.?[0-9]+)",           # AD: 0.95
            ]
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        prob = float(match.group(1))
                        # 确保概率在0-1范围内
                        prob = max(0.0, min(1.0, prob))
                        return prob
                    except ValueError:
                        raise ValueError(f"无法解析概率值: {line}")
        raise ValueError(f"无法解析LLM响应: {response_text[:200]}...")

