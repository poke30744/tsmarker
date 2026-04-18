import os
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("tsmarker.speech.prompt")

# Day of week mapping (Japanese)
WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]

# Default values
DEFAULT_YAML_VALUES = {
    "name": "不明番組",
    "description": "",
    "genre_lv1": "未分類",
    "genre_lv2": "未分類",
    "genre_en": "unknown",
    "channel_name": "不明チャンネル",
    "broadcast_time": "時間不明",
    "duration_desc": "長さ不明",
    "service_id": 0,
}


class PromptEngine:
    """Prompt engine for building LLM prompts and processing YAML metadata"""

    def __init__(self, video_path: Path, markermap_path: Optional[Path] = None):
        self.video_path = video_path
        self.markermap_path = markermap_path
        self.yaml_data = self._load_yaml_data()

    def _load_yaml_data(self) -> Dict[str, Any]:
        """Load YAML file data"""
        video_path = self.video_path
        video_name = video_path.stem

        if not self.markermap_path:
            raise ValueError("markermap_path must be provided to determine YAML file location")

        # YAML is in the parent directory of markermap directory, with same name as video file
        yaml_path = self.markermap_path.parent.parent / f"{video_name}.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(
                f"YAML file does not exist: {yaml_path}\n"
                f"Video file: {video_path}\n"
                f"Expected YAML location: {yaml_path}"
            )

        logger.info(f"Loading YAML file: {yaml_path}")
        return self._load_yaml_file(yaml_path)

    def _load_yaml_file(self, yaml_path: Path) -> Dict[str, Any]:
        """Load a single YAML file"""
        try:
            with yaml_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                logger.error(f"Invalid YAML file format: {yaml_path}")
                raise ValueError(f"Invalid YAML file format: {yaml_path}")

            logger.info(f"Successfully loaded YAML file: {yaml_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load YAML file {yaml_path}: {str(e)}")
            raise


    def get_program_info(self) -> Dict[str, Any]:
        """Get program information, including all YAML fields"""
        info = DEFAULT_YAML_VALUES.copy()

        # Basic fields
        info["program_name"] = self.yaml_data.get("name", self.video_path.stem)
        info["program_description"] = self.yaml_data.get("description", "")
        info["service_id"] = self.yaml_data.get("serviceId", 0)

        # Channel name
        channel_name = self.yaml_data.get("serviceId_desc")
        info["channel_name"] = channel_name or DEFAULT_YAML_VALUES["channel_name"]

        # Broadcast time processing
        start_at = self.yaml_data.get("startAt")
        if start_at:
            try:
                dt = datetime.fromtimestamp(start_at / 1000)
                info["broadcast_time"] = self.yaml_data.get(
                    "startAt_desc", dt.strftime("%Y-%m-%d %H:%M")
                )
                info["weekday"] = WEEKDAY_JA[dt.weekday()]  # Japanese day of week
                info["hour"] = dt.hour
            except Exception as e:
                logger.warning(f"Failed to parse broadcast time: {str(e)}")
                info["broadcast_time"] = DEFAULT_YAML_VALUES["broadcast_time"]
                info["weekday"] = "Unknown"
                info["hour"] = 0
        else:
            info["broadcast_time"] = DEFAULT_YAML_VALUES["broadcast_time"]
            info["weekday"] = "Unknown"
            info["hour"] = 0

        # Program duration
        info["duration_desc"] = self.yaml_data.get(
            "duration_desc", DEFAULT_YAML_VALUES["duration_desc"]
        )

        # Program genre
        genres = self.yaml_data.get("genres", [])
        if genres and isinstance(genres, list) and len(genres) > 0:
            genre = genres[0]
            if isinstance(genre, dict):
                info["genre_lv1"] = genre.get("lv1", DEFAULT_YAML_VALUES["genre_lv1"])
                info["genre_lv2"] = genre.get("lv2", DEFAULT_YAML_VALUES["genre_lv2"])
                info["genre_en"] = genre.get("un1", DEFAULT_YAML_VALUES["genre_en"])

        # Video filename
        info["video_filename"] = self.video_path.name

        # Extended description (if available)
        extended = self.yaml_data.get("extended")
        if extended and isinstance(extended, dict):
            # Merge extended description
            extended_text = " ".join(str(v) for v in extended.values() if v)
            if extended_text:
                info["program_description"] += f"\n\n{extended_text}"

        # Truncate overly long description
        if len(info["program_description"]) > 500:
            info["program_description"] = info["program_description"][:497] + "..."

        return info

    def get_system_prompt(self) -> str:
        """Get system prompt (contains all video-shared information)"""
        return """你是广告识别专家，专门分析日本电视节目的广告片段。请根据节目上下文和文本内容，判断每个视频片段是否为广告。

节目上下文分析：
1. **节目性质**：{program_name}
   - 类型：{genre_lv1} > {genre_lv2} ({genre_en})
   - 描述：{program_description}
   - 总时长：{duration_desc}

2. **播出信息**：
   - 频道：{channel_name} (ID: {service_id})
   - 播出时间：{broadcast_time} ({weekday} {hour}:00)
   - 视频文件：{video_filename}

3. **广告特征参考**：
   - 商业广告：推销产品/服务，包含品牌、价格、优惠、购买呼吁
   - 节目预告：宣传其他节目，包含播出时间、频道信息
   - 赞助商信息：节目赞助商展示，可能包含logo和简短宣传
   - 公益广告：非商业宣传，社会公益内容

4. **非广告特征**：
   - 节目正片：与{genre_lv1}类型相关的内容
   - 新闻播报：事实报道，无推销性质
   - 访谈对话：专家、嘉宾谈话内容
   - 过渡片段：节目段落切换，无实质内容

请综合考虑节目类型、播出时间和内容特征进行判断。

输出格式：
AD: [概率值] [简短理由]
- 概率范围：0.0-1.0
- 0.0 = 确定是广告 (CM)
- 1.0 = 确定不是广告 (节目正片)
- 中间值表示可能性"""

    def get_user_prompt_template(self) -> str:
        """Get user prompt template (only contains current clip text)"""
        return "{clip_texts_formatted}"
