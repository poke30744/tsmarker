import os
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("tsmarker.speech.prompt")

# 星期几映射（日语）
WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]

# 默认值
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
    """提示引擎，负责构建LLM提示和处理YAML元数据"""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.yaml_data = self._load_yaml_data()
        self.channels_map = self._load_channels_map()

    def _load_yaml_data(self) -> Dict[str, Any]:
        """加载YAML文件数据"""
        yaml_path = self.video_path.with_suffix(".yaml")

        # 如果视频文件在raw目录，尝试在encoded目录查找YAML文件
        if not yaml_path.exists():
            # 检查路径中是否包含"raw"
            video_str = str(self.video_path)
            if "raw" in video_str:
                # 尝试替换"raw"为"encoded"来查找YAML文件
                encoded_path = video_str.replace("raw", "encoded")
                encoded_yaml_path = Path(encoded_path).with_suffix(".yaml")

                # 如果encoded目录下有YAML文件，使用它
                if encoded_yaml_path.exists():
                    yaml_path = encoded_yaml_path
                    logger.info(f"在encoded目录找到YAML文件: {yaml_path}")
                else:
                    # 尝试在encoded目录下递归查找同名YAML文件
                    encoded_root = Path(video_str[:video_str.find("raw")]) / "encoded"
                    if encoded_root.exists():
                        video_name = self.video_path.stem
                        # 由于文件名包含特殊字符，使用简单的遍历查找
                        for yaml_file in encoded_root.rglob("*.yaml"):
                            if yaml_file.stem == video_name:
                                yaml_path = yaml_file
                                logger.info(f"递归查找到YAML文件: {yaml_path}")
                                break

        if not yaml_path.exists():
            logger.warning(f"YAML文件不存在: {yaml_path}")
            return {}

        try:
            with yaml_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                logger.warning(f"YAML文件格式无效: {yaml_path}")
                return {}

            return data

        except Exception as e:
            logger.error(f"加载YAML文件失败 {yaml_path}: {str(e)}")
            return {}

    def _load_channels_map(self) -> Dict[int, str]:
        """加载channels.yml映射表"""
        channels_map = {}

        # 尝试多个可能的位置
        possible_paths = [
            Path(__file__).parent.parent.parent
            / "tstriage"
            / "tstriage"
            / "channels.yml",
            Path(__file__).parent.parent.parent / "tstriage" / "channels.yml",
            Path("channels.yml"),
        ]

        for path in possible_paths:
            if path.exists():
                try:
                    with path.open(encoding="utf-8") as f:
                        channels = yaml.safe_load(f)

                    if isinstance(channels, list):
                        for channel in channels:
                            if (
                                isinstance(channel, dict)
                                and "serviceId" in channel
                                and "name" in channel
                            ):
                                channels_map[channel["serviceId"]] = channel["name"]
                        logger.info(f"从 {path} 加载了 {len(channels_map)} 个频道映射")
                        break
                except Exception as e:
                    logger.warning(f"加载channels.yml失败 {path}: {str(e)}")

        return channels_map

    def get_program_info(self) -> Dict[str, Any]:
        """获取节目信息，包含所有YAML字段"""
        info = DEFAULT_YAML_VALUES.copy()

        # 基础字段
        info["program_name"] = self.yaml_data.get("name", self.video_path.stem)
        info["program_description"] = self.yaml_data.get("description", "")
        info["service_id"] = self.yaml_data.get("serviceId", 0)

        # 频道名称
        channel_name = self.yaml_data.get("serviceId_desc")
        if not channel_name and info["service_id"] in self.channels_map:
            channel_name = self.channels_map[info["service_id"]]
        info["channel_name"] = channel_name or DEFAULT_YAML_VALUES["channel_name"]

        # 播出时间处理
        start_at = self.yaml_data.get("startAt")
        if start_at:
            try:
                dt = datetime.fromtimestamp(start_at / 1000)
                info["broadcast_time"] = self.yaml_data.get(
                    "startAt_desc", dt.strftime("%Y-%m-%d %H:%M")
                )
                info["weekday"] = WEEKDAY_JA[dt.weekday()]  # 日语星期
                info["hour"] = dt.hour
            except Exception as e:
                logger.warning(f"解析播出时间失败: {str(e)}")
                info["broadcast_time"] = DEFAULT_YAML_VALUES["broadcast_time"]
                info["weekday"] = "不明"
                info["hour"] = 0
        else:
            info["broadcast_time"] = DEFAULT_YAML_VALUES["broadcast_time"]
            info["weekday"] = "不明"
            info["hour"] = 0

        # 节目时长
        info["duration_desc"] = self.yaml_data.get(
            "duration_desc", DEFAULT_YAML_VALUES["duration_desc"]
        )

        # 节目类型
        genres = self.yaml_data.get("genres", [])
        if genres and isinstance(genres, list) and len(genres) > 0:
            genre = genres[0]
            if isinstance(genre, dict):
                info["genre_lv1"] = genre.get("lv1", DEFAULT_YAML_VALUES["genre_lv1"])
                info["genre_lv2"] = genre.get("lv2", DEFAULT_YAML_VALUES["genre_lv2"])
                info["genre_en"] = genre.get("un1", DEFAULT_YAML_VALUES["genre_en"])

        # 视频文件名
        info["video_filename"] = self.video_path.name

        # 扩展描述（如有）
        extended = self.yaml_data.get("extended")
        if extended and isinstance(extended, dict):
            # 合并扩展描述
            extended_text = " ".join(str(v) for v in extended.values() if v)
            if extended_text:
                info["program_description"] += f"\n\n{extended_text}"

        # 截断过长的描述
        if len(info["program_description"]) > 500:
            info["program_description"] = info["program_description"][:497] + "..."

        return info

    def get_system_prompt(self) -> str:
        """获取系统提示"""
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

请综合考虑节目类型、播出时间和内容特征进行判断。"""

    def get_user_prompt_template(self) -> str:
        """获取用户提示模板"""
        return """请分析以下视频片段（来自节目"{program_name}"）：

{clip_texts_formatted}

请为每个片段判断是否为广告，考虑以下因素：
1. 与节目主题的相关性（{genre_lv1}类型节目）
2. 播出时段特征（{broadcast_time}，{hour}:00）
3. 文本中的商业元素（品牌、价格、购买呼吁等）
4. 节目上下文（{program_description}）

回答格式（每个片段一行）：
[序号]. AD: [概率0.0-1.0] [简短理由]

示例：
1. AD: 0.95 明确推销产品，包含价格和购买链接
2. AD: 0.20 节目正片内容，专家访谈关于{genre_lv2}
3. AD: 0.80 节目预告，宣传明天同频道节目
4. AD: 0.60 可能为赞助商信息，包含品牌名称但无直接推销

注意：概率范围0.0-1.0，0.0=绝对不是广告，1.0=确定是广告"""
