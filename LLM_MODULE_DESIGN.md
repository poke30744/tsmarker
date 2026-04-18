# LLM 模块设计文档 (marker.ensemble - 重构版)

## 项目目标
重构现有的 `marker.ensemble` 模块，将基于机器学习的集成学习改为基于规则文件和 LLM 推理的混合引擎来标记广告 clip。保持模块名称和 CLI 接口不变以确保兼容性。

## 1. 架构决策

### 保留现有模块名称，重构内部实现
**理由**：
- 保持向后兼容性：现有脚本使用 `--method ensemble` 无需修改
- 复用现有模块结构和导入方式
- 简化迁移：直接替换 `ensemble.py` 文件即可
- 继承现有 `MarkerMap` 模式，与其它模块保持一致

### 采用 LLM Agent + 工具调用架构
**核心设计变更**：
1. **LLM 作为决策主体**：而非程序执行规则，LLM 根据规则描述进行推理判断
2. **动态工具调用**：LLM 可请求调用特定工具获取额外信息
3. **贪心处理策略**：按 clip 长度优先级处理，最长 clip 优先
4. **结果缓存**：单clip处理但缓存结果供相似clip参考，减少重复思考

## 2. 模块结构（重构 ensemble.py）

```
tsmarker/
├── ensemble.py                    # 重构后的主模块文件
│   ├── 类定义
│   │   ├── MarkerMap(common.MarkerMap) - 主类，保持接口不变
│   │   ├── AgentEngine - LLM Agent 执行引擎
│   │   ├── ToolSet - 工具集管理
│   │   ├── RuleManager - 规则加载与管理
│   │   └── CacheManager - 结果缓存管理
│   ├── 工具实现
│   │   ├── DurationTool - 时长相关特征
│   │   ├── LogoTool - logo 检测分数
│   │   ├── SubtitlesTool - 字幕文本提取
│   │   ├── PositionTool - 位置信息
│   │   └── ContextTool - 节目上下文信息
│   └── LLM 集成
│       ├── LLMClient - 统一客户端（OpenAI/Ollama）
│       ├── PromptBuilder - Agent 提示工程
│       └── ResponseParser - 响应解析
└── marker.py                    # 仅需添加新参数支持，方法分支已存在
```

## 3. 规则文件格式 (YAML) - LLM 可理解的规则描述

规则文件包含给 LLM 的规则描述和可用的工具映射。LLM 根据这些规则进行推理，并可调用工具获取额外信息。

```yaml
version: 1.1
description: "广告识别规则集 - 供 LLM Agent 使用"

# 可用工具定义
tools:
  - name: get_duration
    description: "获取 clip 的时长信息（秒）"
    returns: "duration: float, duration_prev: float, duration_next: float"
    
  - name: get_position
    description: "获取 clip 在视频中的位置信息"
    returns: "position: float (0.0-1.0), is_start: bool, is_end: bool"
    
  - name: get_logo_score
    description: "获取 logo 检测分数（0.0-1.0）"
    returns: "logo_score: float"
    
  - name: get_subtitles
    description: "获取字幕文本内容（如果有）"
    returns: "subtitles_text: string, has_subtitles: bool"
    
  - name: get_program_context
    description: "获取节目上下文信息（从 YAML 文件）"
    returns: "program_name: string, program_description: string, genres: list, channel: string"

# 广告识别规则（供 LLM 参考）
rules:
  - id: duration_pattern
    description: "广告 clip 通常有标准时长：5秒或15秒（允许±1秒误差）"
    examples: "5秒广告、15秒广告、14-16秒可能是广告"
    
  - id: combined_duration
    description: "如果 clip 自身时长加上前一个或后一个 clip 的时长为15秒的倍数，这两个 clip 可能都是广告"
    examples: "7秒 + 8秒 = 15秒 → 两个都是广告"
    
  - id: promotional_content
    description: "字幕内容听起来像推销、宣传或包含品牌名称、价格、购买呼吁的可能是广告"
    examples: "新产品介绍、优惠促销、品牌广告"
    
  - id: program_related
    description: "与节目名称或描述相关的内容通常是节目正片，不是广告"
    examples: "节目主题讨论、专家访谈、相关内容"
    
  - id: long_with_logo
    description: "特别长的片段（>60秒）且有明显的 logo（分数>0.5）通常是节目内容，不是广告"
    examples: "90秒的节目片段带有台标"
    
  - id: edge_short_clips
    description: "视频开头或结尾的短片段（<10秒）可能是之前节目的残余，视为广告"
    examples: "开头5秒的黑屏或片尾字幕"
    
  - id: logo_uncertainty
    description: "如果其他特征不确定但有明显 logo，可能不是广告（logo 通常出现在节目内容）"
    examples: "不确定的片段但有台标"

# 处理策略
strategy:
  clip_order: "longest_first"  # 按时长降序处理，一次处理一个clip
  confidence_threshold: 0.8    # 高置信度阈值（暂不使用）
  require_tools: false         # LLM 必须调用工具还是可直接判断
```

## 4. Agent 引擎与工具集

### Agent 引擎架构
Agent 引擎协调 LLM 决策和工具调用，实现以下功能：
1. **规则理解**：将规则描述提供给 LLM 作为决策参考
2. **工具调度**：根据 LLM 请求调用相应工具
3. **状态管理**：跟踪 clip 处理状态和已获取特征
4. **结果缓存**：缓存已处理clip的结果供相似clip参考

### 工具集设计
所有工具返回原始特征值，而非决策结果。工具设计为无状态、可重用的函数。

#### 1. DurationTool
- **功能**：获取 clip 时长相关信息
- **输入**：clip ID, markermap 数据
- **输出**：`{"duration": float, "duration_prev": float, "duration_next": float}`
- **实现**：从 markermap 读取 `duration`, `duration_prev`, `duration_next` 字段

#### 2. PositionTool
- **功能**：获取 clip 位置信息
- **输入**：clip ID, markermap 数据, 总 clip 数
- **输出**：`{"position": float, "is_start": bool, "is_end": bool, "clip_index": int}`
- **实现**：从 markermap 读取 `position` 字段，计算是否为开头/结尾

#### 3. LogoTool
- **功能**：获取 logo 检测分数
- **输入**：clip ID, markermap 数据
- **输出**：`{"logo_score": float, "has_logo": bool}`
- **实现**：从 markermap 读取 `logo` 字段，`has_logo = logo_score > 0.5`

#### 4. SubtitlesTool
- **功能**：获取字幕文本内容
- **输入**：clip ID, 视频路径, ptsmap
- **输出**：`{"subtitles_text": str, "has_subtitles": bool}`
- **实现**：从 `.ass.original` 或 `.assgen` 文件提取对应 clip 的文本

#### 5. ProgramContextTool
- **功能**：获取节目上下文信息
- **输入**：视频路径
- **输出**：`{"program_name": str, "program_description": str, "genres": list, "channel": str, "broadcast_time": str}`
- **实现**：从对应的 `.yaml` 文件加载节目元数据

#### 6. NeighborTool
- **功能**：获取相邻 clip 的信息摘要
- **输入**：clip ID, markermap 数据, 相邻范围
- **输出**：`{"prev_clip_summary": dict, "next_clip_summary": dict}`
- **实现**：提取前一个和后一个 clip 的关键特征

### Agent 执行流程
```python
# 伪代码示例
class AgentEngine:
    def process_clip(self, clip_id, initial_features=None):
        # 1. 构建初始上下文
        context = {
            "clip_id": clip_id,
            "available_tools": self.tool_descriptions,
            "rules": self.rules,
            "features": initial_features or {}
        }
        
        # 2. LLM 决策循环
        max_steps = 3  # 最大工具调用次数
        for step in range(max_steps):
            # LLM 决定：判断或调用工具
            llm_response = self.llm.decide(context)
            
            if llm_response.action == "judge":
                # LLM 直接给出判断
                probability = llm_response.probability
                reason = llm_response.reason
                break
            elif llm_response.action == "call_tool":
                # 调用工具获取更多信息
                tool_name = llm_response.tool_name
                tool_args = llm_response.tool_args
                result = self.tools[tool_name].execute(**tool_args)
                context["features"].update(result)
            else:
                # 未知动作，默认处理
                probability = 0.5
                reason = "无法判断"
                break
        
        # 3. 返回最终结果
        return {
            "clip_id": clip_id,
            "probability": probability,
            "reason": reason,
            "tools_used": context.get("tools_used", [])
        }
```

### 缓存与性能优化
为减少 API 调用和重复计算，实现以下优化：
1. **特征预提取**：一次性读取所有 clip 的基本特征（时长、位置等）
2. **结果缓存**：相同特征组合的结果缓存，避免重复LLM思考
3. **工具结果共享**：一个工具调用的结果可供多个 clip 使用（如节目上下文信息）
4. **单clip决策**：每次只处理一个clip，但通过缓存复用相似决策

## 5. LLM Agent 集成方案

### Agent 决策流程
```
初始化阶段：
  1. 加载规则文件和工具定义
  2. 预提取所有 clip 的基本特征（时长、位置）
  3. 按 clip 长度降序排序

处理循环（贪心策略）：
  对于每个 clip（一次只处理一个）：
    1. 构建 Agent 上下文：
       - clip 基本信息（ID、时长、位置等）
       - 可用工具描述
       - 广告识别规则
       - 已获取的特征数据
    
    2. LLM 决策：
       - 选项 A：直接给出判断（概率 + 理由）
       - 选项 B：调用工具获取更多信息
    
    3. 工具调用（如需要）：
       - 执行请求的工具
       - 更新特征数据
       - 返回给 Agent 继续决策
    
    4. 最终判断：
       - LLM 基于所有可用信息给出最终概率
       - 记录决策理由和使用的工具
    
    5. 结果缓存：
       - 缓存特征组合与结果的映射
       - 供后续相似 clip 参考
```

### API 客户端与工具调用支持
1. **OpenAI 函数调用**：利用 OpenAI 的 function calling 能力
2. **Ollama 工具调用**：通过结构化提示实现工具调用
3. **Anthropic 工具使用**：支持 Claude 的工具使用功能
4. **本地回退**：无工具调用支持的模型使用文本指令

### Agent 提示模板设计
```python
# 系统提示（定义 Agent 角色和能力）
SYSTEM_PROMPT = """
你是一个广告识别专家 Agent，负责分析视频片段是否为广告。
你有以下能力：
1. 基于广告识别规则进行推理判断
2. 调用工具获取额外信息辅助决策
3. 为每个片段输出概率值（0.0-1.0）和判断理由

广告识别规则：
{rules_summary}

可用工具：
{tools_summary}

处理策略：
- 优先使用已有信息判断，必要时调用工具
- 考虑节目上下文和播出特征
- 提供明确的判断理由
- 输出格式：概率值（0.0-1.0） + 简短理由
"""

# 用户提示（具体 clip 信息）
USER_PROMPT_TEMPLATE = """
请分析以下视频片段：

片段 {clip_index}/{total_clips}
- 片段 ID: {clip_id}
- 时长: {duration} 秒
- 位置: {position}（开头/中间/结尾）
- 节目: {program_name}
- 频道: {channel_name}
- 播出时间: {broadcast_time}

已有特征：
{existing_features}

请判断该片段是否为广告：
1. 如果需要更多信息，请指定要调用的工具
2. 如果已有足够信息，请直接给出判断

判断标准：
- 0.0: 绝对不是广告（节目正片）
- 0.5: 无法确定
- 1.0: 确定是广告
"""

# 工具调用响应格式
TOOL_RESPONSE_TEMPLATE = """
工具调用结果：
{tool_name}: {tool_result}

请基于新信息更新判断：
1. 概率值（0.0-1.0）
2. 判断理由
3. 是否需要更多工具
"""
```

### 单clip处理策略
1. **逐个处理**：每次只处理一个clip，确保决策质量和可解释性
2. **工具调用按需**：每个clip独立决定是否需要调用工具
3. **增量特征收集**：逐步收集特征，避免不必要的数据提取
4. **对话式缓存**：通过对话历史缓存提示和结果，减少重复思考

### 成本优化策略
1. **特征预筛选**：明显特征（如极长片段）直接判断，不调用 LLM
2. **工具调用最小化**：LLM 需证明工具调用的必要性
3. **结果复用**：相同特征组合的结果缓存复用
4. **置信度阈值**：高置信度结果直接采用，减少深度分析

## 6. 实现步骤与迁移计划

### 第一阶段：基础架构重构（2天）
1. **重构 `ensemble.py` 文件**：
   - 保留现有 `MarkerMap` 类接口
   - 移除机器学习训练代码，保留预测框架
   - 添加 Agent 引擎基础类

2. **实现工具集**：
   - `DurationTool`：时长特征提取
   - `PositionTool`：位置信息计算
   - `LogoTool`：logo 分数获取
   - `SubtitlesTool`：字幕文本提取
   - `ProgramContextTool`：节目上下文加载

3. **规则管理器**：
   - YAML 规则文件加载和解析
   - 规则描述格式化
   - 工具映射管理

### 第二阶段：LLM Agent 集成（2-3天）
1. **LLM 客户端增强**：
   - 基于现有 `speech` 模块的 `OpenAIClient` 扩展
   - 添加函数调用/工具调用支持
   - 多模型支持（OpenAI, Ollama, Anthropic）

2. **Agent 引擎实现**：
   - `AgentEngine` 类：协调 LLM 决策和工具调用
   - 决策循环：判断 → 工具调用 → 最终决策
   - 状态管理和上下文维护

3. **提示工程优化**：
   - Agent 系统提示模板
   - 工具调用格式设计
   - 单clip提示优化和上下文管理

### 第三阶段：处理流程与优化（1-2天）
1. **贪心处理策略**：
   - 按 clip 时长降序排序
   - 逐个处理clip，最长clip优先
   - 结果缓存和复用

2. **性能优化**：
   - 特征预提取和缓存
   - 结果缓存和复用优化
   - 工具调用合并和共享

3. **错误处理与健壮性**：
   - LLM API 错误处理
   - 工具调用失败回退
   - 部分失败时的 graceful degradation

### 第四阶段：CLI 集成与测试（1天）
1. **CLI 参数扩展**：
   ```bash
   # 保持现有接口不变
   tsmarker mark --method ensemble --input video.ts
   
   # 新增可选参数
   tsmarker mark --method ensemble --input video.ts --rules ./rules.yaml
   tsmarker mark --method ensemble --input video.ts --llm-model gpt-4
   ```

2. **向后兼容性**：
   - 确保现有脚本无需修改
   - 默认使用内置规则集

3. **测试验证**：
   - 单元测试：工具、Agent 引擎、规则管理
   - 集成测试：完整标记流程
   - 性能测试：处理时间和 API 成本
   - 兼容性测试：现有工作流验证

### 第五阶段：部署与文档（0.5天）
1. **依赖管理**：
   ```toml
   # 更新 pyproject.toml
   dependencies = [
       "openai>=1.0.0",  # 新增（如果 speech 模块未已添加）
       # 移除 scikit-learn, pandas（如不再需要）
   ]
   ```

2. **文档更新**：
   - 更新 README 和使用示例
   - 规则文件编写指南
   - 配置和调优说明

3. **性能监控**：
   - 添加处理统计（耗时、API 调用次数、工具使用频率）
   - 成本估算和优化建议

## 7. 配置管理（简化版）

### 配置原则
1. **环境变量优先**：主要配置通过环境变量设置
2. **命令行参数覆盖**：运行时参数可覆盖环境变量
3. **合理默认值**：无需配置即可基本运行
4. **规则文件必需**：广告识别规则需通过规则文件定义

### 环境变量配置
```bash
# LLM API 配置（必需）
OPENAI_API_KEY=sk-...                # OpenAI API 密钥
# 或使用 Ollama
OLLAMA_HOST=http://localhost:11434   # Ollama 服务地址

# 可选配置
OPENAI_API_BASE=https://api.openai.com/v1  # 自定义 API 网关
OPENAI_MODEL=gpt-4o-mini             # 默认模型
ENSEMBLE_RULES_PATH=./rules.yaml     # 默认规则文件路径
ENSEMBLE_MAX_TOOLS=3                 # 每个 clip 最大工具调用次数
ENSEMBLE_CACHE_ENABLED=true          # 结果缓存
```

### 规则文件配置
规则文件是必需的配置，包含广告识别规则和工具定义：
```yaml
version: 1.1
description: "广告识别规则集"

tools:
  # 工具定义...

rules:
  # 规则描述...

strategy:
  clip_order: "longest_first"
  confidence_threshold: 0.8
```

### 配置优先级（从高到低）
1. **命令行参数**：`--rules`, `--llm-model`, `--max-tools` 等
2. **环境变量**：`OPENAI_API_KEY`, `ENSEMBLE_RULES_PATH` 等
3. **规则文件中的策略设置**：`strategy` 部分
4. **程序默认值**：内置合理默认值


## 8. CLI 使用示例

```bash
# 基本使用（完全兼容现有命令）
tsmarker mark --method ensemble --input video.ts

# 使用自定义规则文件
tsmarker mark --method ensemble --input video.ts --rules ./my_rules.yaml

# 指定 LLM 模型
tsmarker mark --method ensemble --input video.ts --rules rules.yaml --llm-model gpt-4
tsmarker mark --method ensemble --input video.ts --rules rules.yaml --llm-model llama3

# 使用自定义 API 端点
tsmarker mark --method ensemble --input video.ts --rules rules.yaml \
  --llm-model gpt-4 \
  --llm-base-url https://custom.openai.example.com/v1

# 配置处理参数
tsmarker mark --method ensemble --input video.ts --rules rules.yaml \
  --max-tools 3 \
  --confidence-threshold 0.75

# 批量处理文件夹（保持现有工作流）
for f in *.ts; do
  tsmarker mark --method ensemble --input "$f" --quiet
done

# 调试模式（查看 Agent 决策过程）
tsmarker mark --method ensemble --input video.ts --rules rules.yaml --debug

# 仅使用规则启发式（跳过 LLM，降级模式）
tsmarker mark --method ensemble --input video.ts --rules rules.yaml --no-llm
```

## 9. 关键优势

1. **智能决策**：LLM Agent 能够理解复杂规则并进行推理，优于简单的规则引擎
2. **动态工具调用**：根据需要获取信息，避免不必要的数据提取和传输
3. **完全可解释**：每个决策都有明确的推理过程和工具使用记录
4. **向后兼容**：保持现有 `--method ensemble` 接口不变，无需修改现有脚本
5. **灵活配置**：规则文件使用自然语言描述，易于理解和修改
6. **成本可控**：贪心策略和单clip处理优化减少 API 调用，工具调用最小化
7. **健壮性**：优雅降级机制，LLM 失败时仍能基于规则启发式判断
8. **可扩展性**：易于添加新工具和规则，支持多 LLM 提供商
9. **性能优化**：特征预提取、结果缓存、智能缓存等多重优化
10. **无缝集成**：复用现有特征提取模块（logo、subtitles、clipinfo）

## 10. 后续扩展方向

1. **Agent 学习优化**：基于 groundtruth 数据优化 Agent 的决策策略
2. **多 Agent 协作**：不同专业 Agent 协作判断（时长专家、文本专家、视觉专家）
3. **实时反馈学习**：根据用户修正反馈调整 Agent 行为
4. **工具自动发现**：自动识别和集成新的特征提取工具
5. **离线本地模型**：集成小型本地 LLM，减少 API 依赖
6. **规则知识库**：构建和共享广告识别规则知识库
7. **性能智能监控**：实时监控成本、准确率、性能，自动调优
8. **自适应策略**：根据视频特征自动选择最优处理策略
9. **跨节目学习**：学习不同节目类型的广告模式差异
10. **可视化决策树**：将 Agent 决策过程可视化为可解释的决策树

## 11. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| **LLM API 成本高** | 贪心策略（长clip优先）、单clip处理、工具调用最小化、结果缓存 |
| **API 响应慢/失败** | 超时设置、自动重试、优雅降级到规则启发式、进度显示 |
| **工具调用过度** | 最大工具调用次数限制、工具调用必要性验证、特征预提取 |
| **Agent 决策不一致** | 确定性规则锚定、置信度阈值、多clip上下文考虑 |
| **特征提取失败** | 优雅降级（跳过相关工具）、默认值处理、错误恢复 |
| **规则理解偏差** | 明确的规则描述、示例提供、规则权重调整 |
| **处理性能问题** | 特征预提取缓存、智能缓存、异步处理、性能监控 |
| **配置复杂性** | 合理默认值、环境变量优先、简化配置选项、详细错误提示 |
| **向后兼容破坏** | 保持 CLI 接口不变、渐进式迁移、兼容性测试 |
| **模型输出格式不稳定** | 多格式响应解析、格式验证、默认值回退 |

## 12. 成功指标

1. **决策质量**：Agent能够理解复杂规则并推理，准确率（F1-score）高于简单规则引擎
2. **工具调用效率**：平均每个clip工具调用次数<2次，90%以上clip在3轮内完成决策
3. **可解释性**：每个决策都有明确的推理过程和工具使用记录，完全可追溯
4. **成本控制**：贪心策略和单clip处理减少API调用，工具调用最小化，成本可控
5. **向后兼容**：现有`--method ensemble`脚本无需修改，无缝迁移
6. **处理性能**：贪心算法优先处理长clip，智能缓存优化整体处理时间

---

**下一步行动**：
1. 确认Agent架构设计后开始实现第一阶段
2. 优先实现Agent引擎基础类和工具集（特征提取工具）
3. 实现贪心处理策略和缓存优化机制
4. 集成LLM Agent决策循环和工具调用
5. 最后进行CLI集成和ensemble模块重构
6. **注意**：speech模块将单独进行LLM化改造（见SPEECH_MODULE_DESIGN.md）

**文档版本**：3.0  
**最后更新**：2026-04-17  
**作者**：opencode (DeepSeek-Reasoner)  
**状态**：Agent架构设计完成（专注替换ensemble模块）  
**主要更新**：
- 架构从"两阶段混合引擎"改为"LLM Agent + 工具调用架构"
- 规则文件从可执行条件改为LLM可理解的描述
- 工具从直接计算结果的规则工具改为特征提取工具
- 执行流程改为贪心算法（最长clip优先）和Agent决策循环
- 支持动态工具调用和缓存优化
- **范围调整**：专注替换ensemble模块，speech模块独立LLM化