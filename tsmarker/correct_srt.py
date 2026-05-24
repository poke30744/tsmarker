import logging
import os
import re
from pathlib import Path
from datetime import datetime

from openai import OpenAI

logger = logging.getLogger("tsmarker.correct_srt")

WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]


def _load_yaml(video_path: Path) -> dict | None:
    """Find and load the YAML metadata file for a video."""
    yaml_path = video_path.parent / f"{video_path.stem}.yaml"
    if not yaml_path.exists():
        logger.warning(f"YAML not found: {yaml_path}")
        return None
    import yaml
    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        logger.warning(f"Invalid YAML: {yaml_path}")
        return None
    logger.info(f"Loaded YAML: {yaml_path}")
    return data


def _load_srt(srt_path: Path) -> str:
    """Read the full .generated.srt content."""
    with srt_path.open(encoding="utf-8") as f:
        content = f.read()
    logger.info(f"Loaded SRT: {srt_path} ({len(content)} chars)")
    return content


def _count_srt_entries(srt_content: str) -> int:
    """Count SRT entries by counting the index lines (positive integers on their own line)."""
    return len(re.findall(r"^\d+\s*$", srt_content, re.MULTILINE))


def _build_system_prompt(yaml_data: dict) -> str:
    """Build system prompt from YAML metadata."""

    # Program info
    name = yaml_data.get("name", "不明番組")
    description = yaml_data.get("description", "")
    channel = yaml_data.get("serviceId_desc", "不明チャンネル")
    duration = yaml_data.get("duration_desc", "不明")

    # Broadcast time
    start_at = yaml_data.get("startAt")
    broadcast_time = "不明"
    weekday = "不明"
    if start_at:
        try:
            dt = datetime.fromtimestamp(start_at / 1000)
            broadcast_time = yaml_data.get("startAt_desc", dt.strftime("%Y-%m-%d %H:%M"))
            weekday = WEEKDAY_JA[dt.weekday()]
        except Exception:
            pass

    # Extended info
    extended = yaml_data.get("extended")
    extended_text = ""
    if extended and isinstance(extended, dict):
        parts = []
        for k, v in extended.items():
            if v:
                parts.append(f"{k}: {v}")
        extended_text = "\n".join(parts)

    # Cast
    cast = yaml_data.get("出演者", "")

    prompt = f"""你是日语语音识别后处理专家。你的任务是对一份字幕文件进行全面的可疑词检测和修正。

## 错误来源
原始字幕由 **faster-whisper small** 模型（int8 量化，CPU 推理）生成。这是一个参数量很小的低精度模型，对日语同音/近音词的识别错误率**非常高**。你必须假设每条字幕都可能包含错误，不要信任任何文字的表面形式。

错误模式：
- **同音异字**：模型将正确词语转写为发音相同或相近但意义完全不同的汉字/假名（如「ローエングラム侯」→「廊園グラム港」、「ラインハルト」→「ライン・ハルト」）
- **专有名词错误**：人名、地名、组织名被映射为发音相近的无意义词组
- **词语边界错误**：分词错误导致词组被错误拆分或合并
- **缺乏语义上下文**：模型逐帧识别，不考虑跨句、跨段落的语义连贯性

## 节目信息
- 节目名：{name}
- 描述：{description}
- 频道：{channel}
- 播出时间：{broadcast_time}（{weekday}）
- 时长：{duration}
- 出演者：{cast if cast else "不明"}

{extended_text}

以上信息中的专有名词（人名、地名、组织名、作品名）是**权威写法**。字幕中出现的任何发音相近的变体都应统一修正为这些权威写法。

## 内容类型
字幕可能包含：节目正文（对话/旁白）、商业广告、节目预告。三种内容都需要修正，不要跳过或删除任何条目。

## 修正方法论

你需要逐条审视每一条字幕，对每一个词语执行以下三步检测：

### 第一层：单条合理性
- 这条字幕中的每个词语，在日语中是否是一个**真实存在的词组**？
- 如果不是（如「有経無経」「白者」），这个发音对应什么**正确的同音词**？
- 如果是，这个词语在**当前句子的语境**中是否通顺？

### 第二层：局部连贯性
- 这条字幕和**上一条、下一条**是否构成连贯的对话或叙述？
- 如果不连贯，中间哪个词最有可能是同音错误导致了断裂？
- 修正这个词后，前后逻辑是否恢复？

### 第三层：全局一致性
- 同一个人名/地名/组织名在全文中的写法是否一致？
- 节目的剧情背景（参考上方节目信息）是否支持这条字幕想要表达的意思？
- 如果不支持，最可能的正确文本是什么？

## 修正守则
- 每一处修正都应该是"发音相同或相近 + 上下文合理"的最佳匹配，不是随意的润色
- 保持原始的语气、敬语、口语化表达不变
- 日语中真实存在的词组不必修正，即使觉得"可以有更好的说法"
- 广告中的产品名、品牌名、电话号码可能看起来奇怪但可能是真实存在的，谨慎处理

## 输出格式
输出完整的标准 SRT 文件（序号、时间轴、修正后的文本），保持原始序号和时间轴完全不变。不要省略任何条目。不要添加说明、注释或分析文字。"""

    return prompt


def _build_user_prompt(srt_content: str) -> str:
    """Build user prompt containing the raw SRT content."""
    return f"请修正以下字幕文件：\n\n{srt_content}"


def _parse_corrected_srt(response_text: str) -> str:
    """Extract SRT content from LLM response (may be wrapped in markdown code blocks)."""
    # Try to extract from markdown code block
    m = re.search(r"```(?:srt)?\s*\n(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # If no code block, assume the whole response is SRT
    # Strip leading/trailing markers
    text = response_text.strip()
    # If it starts with a number and newline, it's likely raw SRT
    if re.match(r"^\d+\s*\n", text):
        return text
    # Otherwise, try to find where SRT starts
    m = re.search(r"^1\s*\n\d{2}:", text, re.MULTILINE)
    if m:
        return text[m.start():].strip()
    return text


def _validate_srt(original: str, corrected: str) -> list[str]:
    """Validate corrected SRT and return warnings."""
    warnings = []
    orig_count = _count_srt_entries(original)
    corr_count = _count_srt_entries(corrected)

    if corr_count == 0:
        warnings.append("Corrected SRT has no parseable entries")

    if orig_count > 0 and corr_count > 0:
        diff_pct = abs(corr_count - orig_count) / orig_count * 100
        if diff_pct > 5:
            warnings.append(
                f"Entry count changed from {orig_count} to {corr_count} ({diff_pct:.1f}% difference)"
            )
        elif corr_count != orig_count:
            warnings.append(
                f"Entry count changed from {orig_count} to {corr_count}"
            )

    return warnings


BATCH_SIZE = 200
OVERLAP = 6


def _parse_srt_entries(srt_content: str) -> list[tuple[int, str]]:
    """Parse SRT content into list of (index, entry_text) tuples.

    Each entry_text is the raw SRT block including index, timestamp, and text lines.
    """
    entries = []
    # Split on blank lines between entries
    blocks = re.split(r"\n\s*\n", srt_content.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = re.match(r"^(\d+)\s*\n", block)
        if m:
            idx = int(m.group(1))
            entries.append((idx, block))
    return entries


def _format_srt_entries(entries: list[str]) -> str:
    """Format SRT entry strings into a complete SRT, renumbering from 1."""
    result = []
    for i, entry in enumerate(entries, start=1):
        # Replace the index line with new index
        renumbered = re.sub(r"^\d+\s*\n", f"{i}\n", entry, count=1)
        result.append(renumbered)
    return "\n\n".join(result)


def _call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call LLM with retry on empty response, matching speech module pattern."""
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=32 * 1024,
            timeout=300.0,
        )
        if response is None:
            if attempt < max_retries:
                logger.warning(f"Null response from API, retrying ({attempt + 1}/{max_retries})...")
                continue
            raise ValueError("Null response from OpenAI API")
        content = response.choices[0].message.content
        if content:
            return content
        if attempt < max_retries:
            logger.warning(f"Empty response from API, retrying ({attempt + 1}/{max_retries})...")
    raise ValueError("Empty response from OpenAI API")


def correct_srt(
    video_path: Path,
    generated_srt_path: Path,
    output_path: Path,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    progress=None,
) -> Path:
    """Correct a .generated.srt using LLM with YAML metadata as context.

    Args:
        video_path: Path to the video file (for finding YAML).
        generated_srt_path: Path to the generated SRT to correct.
        output_path: Where to write the corrected SRT.
        model: Override OPENAI_MODEL env var.
        base_url: Override OPENAI_API_BASE env var.
        api_key: Override OPENAI_API_KEY env var.

    Returns:
        The output_path.
    """
    # Load inputs
    yaml_data = _load_yaml(video_path)
    if yaml_data is None:
        raise FileNotFoundError(f"No .yaml found for {video_path}")

    srt_content = _load_srt(generated_srt_path)
    all_entries = _parse_srt_entries(srt_content)
    orig_count = len(all_entries)
    logger.info(f"Original SRT: {orig_count} entries, {len(srt_content)} chars")

    # Build system prompt (same for all batches — cache hits)
    system_prompt = _build_system_prompt(yaml_data)
    logger.info(f"System prompt: {len(system_prompt)} chars")

    # Setup LLM client
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = model or os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Process in batches
    corrected_entries = []  # list of (orig_index, entry_text)
    num_batches = (orig_count + BATCH_SIZE - 1) // BATCH_SIZE

    tid = "correct_srt"
    if progress is not None:
        progress.add_task(tid, num_batches, "LLM SRT correction")

    for bi in range(num_batches):
        start = bi * BATCH_SIZE
        end = min(start + BATCH_SIZE, orig_count)
        batch = all_entries[start:end]
        batch_texts = [entry_text for _idx, entry_text in batch]

        # Build user prompt for this batch
        if bi == 0:
            user_prompt = _build_user_prompt(_format_srt_entries(batch_texts))
        else:
            # Include overlap from previous batch as read-only context
            overlap_start = max(0, len(corrected_entries) - OVERLAP)
            overlap_entries = [e for _, e in corrected_entries[overlap_start:]]
            overlap_srt = _format_srt_entries(overlap_entries)
            batch_srt = _format_srt_entries(batch_texts)
            user_prompt = (
                f"以下是上文的最后 {len(overlap_entries)} 条（已修正，仅供参考上下文）：\n\n"
                f"{overlap_srt}\n\n"
                f"以下是需要修正的部分（第{bi + 1}块，共{num_batches}块）：\n\n"
                f"{batch_srt}"
            )

        logger.info(
            f"Batch {bi + 1}/{num_batches}: entries {start + 1}-{end}, "
            f"user_prompt={len(user_prompt)} chars"
        )

        content = _call_llm(client, model, system_prompt, user_prompt)
        logger.info(f"Batch {bi + 1}/{num_batches}: response={len(content)} chars")

        if progress is not None:
            progress.update(tid, bi + 1)

        # Parse corrected batch
        corrected_text = _parse_corrected_srt(content)
        corrected_batch = _parse_srt_entries(corrected_text)
        logger.info(f"Batch {bi + 1}/{num_batches}: corrected={len(corrected_batch)} entries")

        # Warn if entry count changed within batch
        if len(corrected_batch) != len(batch):
            logger.warning(
                f"Batch {bi + 1}/{num_batches}: count changed "
                f"({len(batch)} → {len(corrected_batch)})"
            )

        # Map back to original indices
        for i, (_renumbered_idx, entry_text) in enumerate(corrected_batch):
            if i < len(batch):
                orig_idx = batch[i][0]
                corrected_entries.append((orig_idx, entry_text))

    # Reconstruct full SRT with original numbering
    corrected_entries.sort(key=lambda x: x[0])
    final_entries = []
    for orig_idx, entry_text in corrected_entries:
        # Restore original index
        restored = re.sub(r"^\d+\s*\n", f"{orig_idx}\n", entry_text, count=1)
        final_entries.append(restored)

    corrected = "\n\n".join(final_entries)

    corr_count = len(final_entries)
    logger.info(f"Corrected SRT: {corr_count} entries, {len(corrected)} chars")

    if corr_count != orig_count:
        logger.warning(f"Total count changed: {orig_count} → {corr_count}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(corrected + "\n")

    logger.info(f"Wrote corrected SRT to {output_path}")

    if progress is not None:
        progress.done(tid)

    return output_path
