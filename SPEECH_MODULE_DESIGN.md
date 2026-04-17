# Speech模块LLM化设计文档 (marker.speech)

## 项目目标
替换现有的 `marker.speech` 模块，将基于BERT的HTTP API判断改为基于LLM的方式，保持输出格式完全不变。

## 1. 现状分析

### 当前speech模块流程
1. **文本提取**：
   - 从字幕文件(.ass.original)提取文本
   - 若无字幕，从音频提取语音转文本
   - 保存到 `.assgen` JSON文件

2. **BERT推理**：
   - 发送文本列表到HTTP API: `http://127.0.0.1:5000/api/mark/speech`
   - API返回预测概率列表 `Y`
   - 每个clip标记为: `self.Mark(clips[i], 'speech', float(Y[i][0]))`

3. **输出格式**：
   - 在 `.markermap` 文件中添加 `'speech'` 方法标记
   - 每个clip有对应的概率值(0.0-1.0)

## 2. 新设计：简化版LLM Speech模块

### 架构原则
1. **接口兼容**：完全保持现有 `MarkerMap.MarkAll()` 接口
2. **输出一致**：标记格式、文件名、概率范围不变
3. **简化配置**：仅使用环境变量，无配置文件
4. **批量优化**：一次提交所有clip文本（默认）

### 模块结构
```
tsmarker/
├── speech/                          # 改造后模块目录
│   ├── __init__.py                 # 导出 MarkerMap 类
│   ├── MarkerMap.py                # 主类（LLM版本）
│   ├── llm_client.py               # OpenAI客户端
│   ├── prompt_engine.py            # 文本分类提示工程
│   └── text_extractor.py           # 文本提取（复用现有逻辑）
```

## 3. 充分利用YAML文件信息

### YAML文件完整结构（基于tstriage/tscutter分析）
每个视频文件对应一个 `.yaml` 文件，包含完整的节目元数据，源自EPG（电子节目指南）：

```yaml
# 基本节目信息
name: "ＮＨＫスペシャル　富士山大噴火　迫る"灰色の悪夢"　火山灰・溶岩流の脅威[字]"
description: "富士山の噴火シナリオを検証。火山灰が首都圏を襲う"灰色の悪夢"や、溶岩流の脅威をCGで再現。"
serviceId: 1024  # NHK総合１・東京
serviceId_desc: "ＮＨＫ総合１・東京"  # 频道名称
startAt: 1744491600000  # 时间戳(毫秒)，2026-04-12 21:00:00
startAt_desc: "2026-04-12 21:00 (Sun)"  # 格式化时间
duration: 3600000  # 时长(毫秒)，60分钟
duration_desc: "60 mins"  # 格式化时长

# 扩展描述（可能包含详细内容）
extended:
  text1: "詳細な噴火シナリオ分析..."
  text2: "専門家インタビュー..."

# 节目类型（多层次分类）
genres:
  - lv1: "ドキュメンタリー・教養"  # 一级分类：纪录片/教育
    lv2: "ドキュメンタリー"         # 二级分类：纪录片
    un1: "documentary"              # 英文一级
    un2: "documentary"              # 英文二级

# 其他可能字段
video: [type: 'mpeg2', resolution: '1920x1080']
audio: [type: 'aac']
```

### 在提示中充分利用所有可用信息
将完整节目元数据加入LLM提示，提供丰富上下文：

1. **节目内容信息**：
   - `name`：完整节目名称，包含主题和字幕信息（如[字]表示字幕）
   - `description`：节目简介，描述主要内容
   - `extended`：扩展描述，可能包含详细情节或专家访谈

2. **频道与播出信息**：
   - `serviceId_desc`：播出频道（如"ＮＨＫ総合１・東京"）
   - `startAt_desc`：具体播出时间（年月日 时分 星期）
   - 可计算：`start_hour`（21点）、`start_weekday`（0=周日）

3. **节目类型信息**：
   - `genres[0].lv1/lv2`：日文分类（如"ドキュメンタリー・教養"/"ドキュメンタリー"）
   - `genres[0].un1/un2`：英文分类（如"documentary"）
   - 类型影响广告特征：娱乐节目vs新闻纪录片广告模式不同

4. **技术特征**：
   - `duration_desc`：节目总时长（如"60 mins"）
   - 长节目通常有规律广告插播，短节目广告模式不同

### 上下文价值
- **NHK纪录片**：通常广告较少，可能有赞助商信息
- **晚间黄金时段**（21:00）：广告价值高，可能包含品牌广告
- **周末播出**：广告内容可能与工作日不同
- **教育类节目**：可能包含出版社、教育机构广告

## 4. 环境变量配置（简化版）

### 必需环境变量
```bash
# OpenAI API配置
OPENAI_API_KEY=sk-...                # API密钥
OPENAI_API_BASE=https://api.openai.com/v1  # API网关地址（可自定义）
OPENAI_MODEL=gpt-4o-mini            # 模型名称（默认）
```

### 可选环境变量
```bash
# 批量处理配置
SPEECH_BATCH_ALL=true               # 一次提交所有clip（默认true）
SPEECH_MAX_TOKENS=2000              # 最大响应token数
SPEECH_TEMPERATURE=0.1              # 温度参数（低值确保一致性）

# 调试选项
SPEECH_DEBUG=false                  # 启用调试日志
```

## 5. 执行流程

### 单视频处理流程
```
1. 加载视频和索引文件
2. 加载对应YAML文件获取节目元数据
3. 提取每个clip的文本（字幕优先，音频备用）
4. 构建提示：包含节目元数据 + 所有clip文本
5. 单次调用OpenAI API，获取所有clip的判断结果
6. 解析响应，提取概率值
7. 调用 self.Mark(clip, 'speech', probability)
8. 保存.markermap文件
```

### 提示模板设计（充分利用YAML信息）
```python
SYSTEM_PROMPT = """
你是广告识别专家，专门分析日本电视节目的广告片段。请根据节目上下文和文本内容，判断每个视频片段是否为广告。

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
"""

USER_PROMPT_TEMPLATE = """
请分析以下视频片段（来自节目"{program_name}"）：

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

注意：概率范围0.0-1.0，0.0=绝对不是广告，1.0=确定是广告
"""
```

### 变量替换说明
- `{program_name}`: YAML中的`name`字段
- `{program_description}`: YAML中的`description`字段（前200字符）
- `{channel_name}`: `serviceId_desc` 或通过serviceId查channels.yml
- `{service_id}`: `serviceId` 字段
- `{broadcast_time}`: `startAt_desc` 格式化时间
- `{weekday}`: 从`startAt`计算的星期几（月火水木金土日）
- `{hour}`: 从`startAt`计算的小时（0-23）
- `{duration_desc}`: `duration_desc` 字段
- `{genre_lv1}`: `genres[0].lv1` 一级分类
- `{genre_lv2}`: `genres[0].lv2` 二级分类  
- `{genre_en}`: `genres[0].un1` 英文分类
- `{video_filename}`: 视频文件名（不含路径）
- `{clip_texts_formatted}`: 格式化后的clip文本列表

## 6. 错误处理简化

### 错误策略
1. **API失败**：直接抛出异常，由上层调用者处理
2. **网络超时**：使用默认超时（30秒）
3. **格式解析失败**：记录错误，跳过该clip标记
4. **YAML文件缺失或字段不全**：优雅降级处理

### YAML文件处理策略
1. **文件缺失**：跳过YAML信息，仅使用clip文本进行判断
2. **字段缺失**：使用默认值或占位符：
   - `name`缺失：使用视频文件名
   - `description`缺失：使用空字符串
   - `genres`缺失：使用"未分類/unknown"
   - `serviceId_desc`缺失：根据`serviceId`查询channels.yml或使用"不明チャンネル"
   - `startAt_desc`缺失：从`startAt`计算或使用"時間不明"

3. **channels.yml集成**：
   - 尝试加载`../tstriage/tstriage/channels.yml`
   - 根据`serviceId`查找频道名称
   - 文件缺失时使用`serviceId`作为频道标识

### 无重试逻辑
- 假设OpenAI API稳定性足够
- 避免复杂重试逻辑增加代码复杂度
- 失败时快速反馈，便于问题排查

### 默认值定义
```python
DEFAULT_YAML_VALUES = {
    'name': '不明番組',
    'description': '',
    'genre_lv1': '未分類',
    'genre_lv2': '未分類', 
    'genre_en': 'unknown',
    'channel_name': '不明チャンネル',
    'broadcast_time': '時間不明',
    'duration_desc': '長さ不明',
}
```

## 7. 实现步骤

### 第一阶段：基础框架（0.5天）
1. **创建文件结构**：
   - `llm_client.py`：OpenAI客户端（简单封装）
   - `prompt_engine.py`：提示模板管理
   - `text_extractor.py`：复用现有文本提取逻辑

2. **改造MarkerMap.py**：
   - 移除BERT HTTP调用
   - 集成OpenAI客户端
   - 添加YAML文件加载逻辑

### 第二阶段：集成测试（0.5天）
1. **单元测试**：文本提取、提示构建、响应解析
2. **集成测试**：完整标记流程
3. **对比测试**：与原BERT结果对比验证

### 第三阶段：部署迁移（0.5天）
1. **更新依赖**：`pyproject.toml` 添加openai库
2. **文档更新**：环境变量配置说明
3. **验证兼容性**：确保现有工作流不受影响

## 8. 依赖更新

### 新增依赖
```toml
[dependencies]
openai = ">=1.0.0"          # OpenAI官方库
```

### 保持依赖
```toml
# 保留现有依赖
speech-recognition          # 用于音频转文本备用
```

## 9. CLI使用示例

### 基本使用（接口完全不变）
```bash
# 原有方式（自动切换到LLM版本）
tsmarker mark --method speech --input video.ts

# 需要设置环境变量
export OPENAI_API_KEY=sk-...
tsmarker mark --method speech --input video.ts
```

### 环境变量示例
```bash
# 使用自定义API网关（如Azure OpenAI）
export OPENAI_API_KEY=your-key
export OPENAI_API_BASE=https://your-resource.openai.azure.com/openai/deployments/gpt-4
export OPENAI_MODEL=gpt-4

# 调试模式
export SPEECH_DEBUG=true
tsmarker mark --method speech --input video.ts
```

## 10. 向后兼容性

### 完全兼容
1. **文件格式**：`.markermap`、`.ass.original`、`.assgen` 格式不变
2. **标记格式**：`'speech'` 方法标记，概率值范围0.0-1.0
3. **CLI接口**：`--method speech` 参数不变
4. **模块导入**：`from tsmarker.speech import MarkerMap` 不变

### 行为变化
1. **推理后端**：从BERT HTTP API改为OpenAI API
2. **配置方式**：从硬编码URL改为环境变量
3. **性能特征**：一次API调用处理所有clip

## 11. 性能预期

### 优势
1. **单次API调用**：大幅减少网络开销
2. **LLM强大理解**：优于BERT的文本分类能力
3. **节目上下文**：利用YAML元数据提高准确率

### 注意事项
1. **token成本**：所有clip文本一次发送，需注意长度限制
   - 策略：clip文本总长度超过模型限制时，智能截断（保留重要clip）
   - 估算：平均每个clip文本100-500字符，50个clip约25K字符
   - 模型限制：gpt-4o-mini支持128K tokens，足够多数情况

2. **失败影响**：单次调用失败影响整个视频处理
   - 策略：详细错误日志，便于问题排查
   - 备选：可考虑分批处理，但按用户要求优先单次批量

3. **响应解析**：需要可靠解析批量响应格式
   - 策略：多种格式匹配，正则表达式解析
   - 验证：响应行数应与clip数量一致

4. **YAML信息整合**：需要正确处理字段缺失和格式变化
   - 策略：健壮的YAML解析，默认值填充
   - 集成：channels.yml文件路径处理（相对路径查找）

## 12. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| OpenAI API成本 | 监控token使用，可选更小模型（gpt-4o-mini），优化提示减少冗余 |
| API响应超时 | 设置合理超时（30秒），快速失败 |
| 批量处理失败 | 记录详细错误信息，便于排查 |
| YAML文件缺失/字段不全 | 优雅降级，使用默认值，集成channels.yml查询 |
| YAML解析错误 | 捕获yaml.load异常，提供详细错误信息 |
| 提示token超限 | 动态截断clip文本，优先保留重要clip |
| 响应格式解析失败 | 多格式尝试，正则表达式匹配，记录原始响应 |
| 向后兼容破坏 | 完整接口测试，确保现有脚本可用 |

## 13. 简化设计总结

### 核心简化点
1. **无配置文件**：仅用环境变量，减少配置复杂度
2. **无重试逻辑**：假设API稳定，失败快速反馈
3. **单次批量处理**：一次提交所有clip，最大化性能
4. **仅OpenAI**：专注主流API，减少适配代码
5. **利用YAML**：复用现有节目元数据，提高准确率

### 开发重点
1. **提示工程**：优化批量分类提示模板
2. **响应解析**：可靠解析批量响应格式
3. **错误处理**：简洁明了的错误反馈
4. **测试验证**：确保与原BERT结果一致

---

**文档版本**：2.1  
**最后更新**：2026-04-15  
**作者**：opencode (DeepSeek-Reasoner)  
**状态**：简化设计完成，YAML信息充分利用  
**主要变更**：
- 移除配置文件，仅使用环境变量
- 默认一次提交所有clip文本
- 取消重试逻辑，简化错误处理
- 仅支持OpenAI API格式
- **深度集成YAML节目元数据**：充分利用name、description、genres、channel、播出时间等所有字段
- **channels.yml集成**：根据serviceId查询频道名称
- **优雅降级处理**：YAML字段缺失时使用合理默认值
- **上下文优化提示**：根据节目类型、播出时段等优化广告判断