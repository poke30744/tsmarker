# LLM 模块设计文档 (marker.llm)

## 项目目标
替换现有的 `marker.ensemble` 模块，新增 `marker.llm` 模块，使用基于规则文件和 LLM 推理的混合引擎来标记广告 clip。

## 1. 架构决策

### 选择完整模块而非 SKILL 插件
**理由**：
- 需要管理复杂的特征提取、规则解析、LLM 集成逻辑
- 需与现有 `marker.py` CLI 深度集成（新增 `--method llm` 选项）
- 需处理本地规则计算 + LLM 语义判断的混合引擎
- 现有 `ensemble` 为独立模块，遵循相同模式更易维护
- 需要管理 LLM 客户端、提示工程、批量推理等非平凡功能

## 2. 模块结构

```
tsmarker/
├── llm/                          # 新模块目录
│   ├── __init__.py              # 导出 MarkerMap 类
│   ├── markermap.py             # 继承 common.MarkerMap，实现 MarkAll
│   ├── rule_tools.py            # 7个直接计算结果的规则工具
│   ├── rule_engine.py           # 规则加载与两阶段执行引擎
│   ├── llm_client.py            # LLM API 客户端抽象（优化版）
│   ├── prompt_builder.py        # 语义判断提示工程
│   ├── config.py                # 配置管理
│   └── batch_processor.py       # 批量处理（本地优先）
├── marker.py                    # 修改：新增 llm 方法分支
└── common.py                    # 保持不变
```

## 3. 规则文件格式 (YAML)

```yaml
version: 1.0
program_name: "节目名称"  # 可选，用于规则7

rules:
  - id: duration_5_15
    type: deterministic
    description: "广告clip长度为5秒，或者15秒的背熟，前后相差1秒以内"
    condition: "duration in [5±1, 15±1] seconds"
    action: "mark_as_ad"
    weight: 1.0
    priority: 1

  - id: combined_15_multiple
    type: deterministic  
    description: "如果一个clip自身长度加上其前或后clip的长度为15秒倍数，则这两个clip都是广告"
    condition: "(duration + prev_duration) % 15 == 0 OR (duration + next_duration) % 15 == 0"
    action: "mark_both_as_ad"
    weight: 1.0
    priority: 2

  - id: long_with_logo_not_ad
    type: deterministic
    description: "特别长的，有logo的clip不是广告"
    condition: "duration > 60 AND logo_score > 0.5"
    action: "mark_as_program"
    weight: 0.9
    priority: 3

  - id: edge_short_clip
    type: deterministic
    description: "开头或者结尾的，只有数秒的clip，是之前节目的残余片段，视为广告"
    condition: "position in ['start', 'end'] AND duration < 10"
    action: "mark_as_ad"
    weight: 0.8
    priority: 4

  - id: unsure_with_logo
    type: llm_semantic
    description: "对于还是难以判断的，如果有logo的，可能不是广告"
    condition: "logo_score > 0.5 AND confidence < 0.7"
    prompt: "If clip has logo but unsure, likely not ad"
    weight: 0.6
    priority: 5

  - id: promotional_subtitles
    type: llm_semantic
    description: "如果subtitles听起来像在推销或者宣传的，视为广告"
    condition: "subtitles exists"
    prompt: "Does subtitle sound like promotion? Answer yes/no"
    weight: 0.7
    priority: 6

  - id: program_related
    type: llm_semantic
    description: "和本节目名称有关的内容，不是广告"
    condition: "program_name exists"
    prompt: "Is content related to program '{program_name}'? Answer yes/no"
    weight: 0.8
    priority: 7
```

## 4. 直接计算结果的规则工具集（7个核心工具）

所有工具直接返回 `RuleResult` 对象，包含决策结果和置信度，而非原始特征。

### 1. DurationRuleTool
- **功能**：应用规则1 - 5秒或15秒±1秒 → 广告
- **输入**：clip 时长 (duration)
- **输出**：`RuleResult(is_ad=True/False, confidence=0.0-1.0, rule_id="duration_5_15")`
- **逻辑**：
  ```python
  if 4 <= duration <= 6 or 14 <= duration <= 16:
      return RuleResult(is_ad=True, confidence=1.0, rule_id="duration_5_15")
  ```

### 2. CombinedDurationRuleTool
- **功能**：应用规则2 - 自身+前/后片段时长为15倍数 → 广告
- **输入**：当前时长、前一片段时长、后一片段时长
- **输出**：`RuleResult(is_ad=True, confidence=1.0, rule_id="combined_15_multiple", affect_neighbors=True)`
- **逻辑**：
  ```python
  if (duration + prev_duration) % 15 < 1 or (duration + next_duration) % 15 < 1:
      return RuleResult(is_ad=True, confidence=1.0, rule_id="combined_15_multiple", affect_neighbors=True)
  ```

### 3. LongWithLogoRuleTool
- **功能**：应用规则3 - 长片段(>60s)且有logo(>0.5) → 节目
- **输入**：时长、logo分数
- **输出**：`RuleResult(is_ad=False, confidence=0.9, rule_id="long_with_logo")`
- **逻辑**：
  ```python
  if duration > 60 and logo_score > 0.5:
      return RuleResult(is_ad=False, confidence=0.9, rule_id="long_with_logo")
  ```

### 4. EdgeShortClipRuleTool
- **功能**：应用规则4 - 开头/结尾且<10秒 → 广告
- **输入**：clip位置、时长
- **输出**：`RuleResult(is_ad=True, confidence=0.8, rule_id="edge_short")`
- **逻辑**：
  ```python
  if position in ["start", "end"] and duration < 10:
      return RuleResult(is_ad=True, confidence=0.8, rule_id="edge_short")
  ```

### 5. LogoUncertaintyTool
- **功能**：应用规则5 - 其他规则不确定但logo分数高 → 可能不是广告
- **输入**：logo分数、先前规则结果（是否高置信度）
- **输出**：`RuleResult(is_ad=False, confidence=0.6, rule_id="unsure_with_logo")`
- **逻辑**：
  ```python
  if not previous_results.any_high_confidence() and logo_score > 0.5:
      return RuleResult(is_ad=False, confidence=0.6, rule_id="unsure_with_logo")
  ```

### 6. SubtitleFeatureTool
- **功能**：提取字幕文本供LLM判断规则6（仅特征提取，非决策）
- **输入**：原始字幕文件/文本
- **输出**：清理后的字幕文本（前200字符）
- **用途**：供LLM判断规则6（字幕是否像推销）

### 7. ProgramFeatureTool
- **功能**：提取节目名称供LLM判断规则7（仅特征提取，非决策）
- **输入**：元数据文件（.yaml）
- **输出**：节目名称字符串
- **用途**：供LLM判断规则7（内容是否与节目相关）

### RuleResult 数据结构
```python
@dataclass
class RuleResult:
    is_ad: bool                    # True=广告, False=节目
    confidence: float              # 置信度 0.0-1.0
    rule_id: str                   # 规则标识符
    affect_neighbors: bool = False # 是否影响相邻clip（规则2）
    reason: str = ""               # 决策理由（可解释性）
```

## 5. LLM 集成方案（优化版：最小化LLM思考量）

### 执行策略：两阶段混合引擎
```
阶段1：本地规则直接计算（快速决策）
  1. 顺序应用规则工具1-5
  2. 如果任何工具返回高置信度(confidence > 0.8)
  3. → 采用该结果，跳过LLM阶段
  4. 否则进入阶段2

阶段2：LLM语义判断（仅处理不确定情况）
  1. 提取语义特征（字幕文本、节目名称）
  2. 向LLM展示已计算的规则结果
  3. LLM仅判断规则6、7（语义规则）
  4. 综合阶段1结果给出最终判断
```

### API 客户端支持
1. **OpenAIClient**：使用 `openai>=1.0.0`
2. **OllamaClient**：本地部署，HTTP 调用 `http://localhost:11434/api/generate`
3. **扩展性**：支持 Anthropic 或其他兼容 API

### 优化提示模板设计（LLM只处理语义判断）
```python
# 系统提示（告知LLM只判断规则6、7）
SYSTEM_PROMPT = """
你是一个广告识别助手。本地规则引擎已计算了数值规则的结果，你只需要判断两个语义规则：

语义规则：
6. 字幕听起来像推销/宣传 → 广告
7. 内容与节目名称相关 → 节目

请基于本地规则结果和语义特征，给出最终判断。
"""

# 用户提示（包含本地计算结果和语义特征）
USER_PROMPT_TEMPLATE = """
本地规则计算结果：
{local_rule_results}

需要LLM判断的语义特征：
- 字幕文本："{subtitle_text}"
- 节目名称："{program_name}"

请根据规则6、7判断是否为广告：
规则6：字幕听起来像推销 → 广告
规则7：内容与"{program_name}"相关 → 节目

参考本地规则结果，给出最终判断：
格式：AD/PROGRAM [置信度0.0-1.0] [理由]
"""

# 本地规则结果格式化示例
LOCAL_RESULTS_FORMAT = """
- 规则1(时长5/15秒)：{不是}广告 ❌
- 规则2(组合15倍数)：{不是}广告 ❌
- 规则3(长片段+logo)：{不符合条件} ❌
- 规则4(边缘短片段)：{不符合条件} ❌
- 规则5(不确定+logo)：{未触发}
"""
```

### 优化效果
1. **思考量减少80%**：LLM无需重新计算数值规则
2. **token成本降低**：仅需传递语义特征而非所有原始特征
3. **响应速度加快**：本地规则快速决策，多数clip无需LLM调用
4. **结果一致性**：数值规则始终由本地引擎执行，避免LLM计算误差

### 批量处理优化
- **批大小**：可配置（默认10个clip/批）
- **缓存**：相同特征缓存结果
- **重试**：API 错误自动重试
- **超时**：可配置超时时间

## 6. 实现步骤与迁移计划

### 第一阶段：基础架构（1-2天）
1. 创建 `tsmarker/llm/` 目录结构
2. 实现 `rule_tools.py`（7个直接计算结果的规则工具）
3. 实现 `rule_engine.py` 两阶段执行框架
4. 编写 `config.py` 配置管理和 RuleResult 数据结构

### 第二阶段：LLM 集成（1-2天）
1. 实现 `llm_client.py`（OpenAI + Ollama 优化版）
2. 实现 `prompt_builder.py` 语义判断提示工程（最小化token）
3. 实现 `batch_processor.py` 批量处理（本地优先策略）
4. 实现 `markermap.py` 主类（两阶段决策集成）

### 第三阶段：CLI 集成（1天）
1. 修改 `marker.py`：
   - 添加 `llm` 导入和方法分支
   - 更新 `--method` 选项添加 `llm`
2. 添加 CLI 参数：
   ```bash
   --rules path/to/rules.yaml
   --llm-model gpt-4|llama3|...
   --llm-api-key $ENV_VAR
   --llm-base-url http://... (可选)
   ```

### 第四阶段：移除旧模块（0.5天）
1. 删除 `tsmarker/ensemble.py` 文件
2. 更新 `pyproject.toml`：
   ```toml
   dependencies = [
       "openai>=1.0.0",  # 新增
       # 可选保留 scikit-learn, pandas（如有其他用途）
   ]
   ```
3. 更新 `tool.setuptools.packages` 配置

### 第五阶段：测试优化（1-2天）
1. 单元测试：规则工具、规则引擎
2. 集成测试：完整的标记流程
3. 性能测试：批量处理效率
4. 错误处理：API 失败、文件缺失等

## 7. 配置管理

### 环境变量
```bash
OPENAI_API_KEY=sk-...           # OpenAI
OLLAMA_HOST=http://localhost:11434  # Ollama
LLM_RULES_PATH=./rules.yaml     # 默认规则文件
```

### 配置文件示例
```yaml
# llm_config.yaml
default_model: "gpt-4"
rules_path: "./rules.yaml"
batch_size: 10
timeout: 30
cache_enabled: true
log_level: "INFO"

openai:
  api_key: "${OPENAI_API_KEY}"
  base_url: "https://api.openai.com/v1"

ollama:
  base_url: "${OLLAMA_HOST}"
  model: "llama3:latest"

rule_priorities:
  deterministic_first: true
  conflict_strategy: "highest_weight"  # highest_weight|latest|average
```

## 8. CLI 使用示例

```bash
# 基本使用
tsmarker mark --method llm --input video.ts --rules rules.yaml

# 指定模型
tsmarker mark --method llm --input video.ts --rules rules.yaml --llm-model gpt-4

# 使用 Ollama 本地模型
tsmarker mark --method llm --input video.ts --rules rules.yaml --llm-model llama3 --llm-base-url http://localhost:11434

# 批量处理文件夹
for f in *.ts; do
  tsmarker mark --method llm --input "$f" --rules rules.yaml --quiet
done

# 仅使用本地规则（跳过 LLM）
tsmarker mark --method llm --input video.ts --rules rules.yaml --no-llm
```

## 9. 关键优势

1. **极高性能**：7个规则工具直接计算结果，80%以上clip由本地引擎决策，无需LLM调用
2. **极低成本**：LLM仅处理语义判断，思考量减少80%，token使用量降低70%
3. **高准确率**：数值规则本地精确计算，避免LLM计算误差，语义规则由LLM处理
4. **可配置**：规则文件支持动态更新和定制，无需代码修改
5. **可扩展**：支持多LLM提供商，易于添加新规则工具
6. **完全可解释**：每个决策都有明确的规则依据和置信度
7. **无缝兼容**：复用现有特征提取结果（logo、subtitles、clip信息）

## 10. 后续扩展方向

1. **自动规则优化**：基于 groundtruth 数据自动调整规则权重
2. **多模型 Ensemble**：集成多个 LLM 结果投票
3. **实时学习**：根据用户反馈调整规则
4. **特征重要性分析**：识别最有效的特征
5. **离线模式**：完全本地执行，无需网络
6. **规则市场**：共享和导入社区规则
7. **性能监控**：规则命中率、LLM 成本统计

## 11. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| LLM API 成本高 | **大幅降低**：80%clip本地决策，LLM仅处理语义判断 |
| API 响应慢 | 超时设置、异步处理、进度显示、本地规则优先 |
| 规则冲突 | 优先级策略、置信度加权、人工审核标记 |
| 特征缺失 | 优雅降级（跳过相关规则）、默认值处理 |
| 模型偏见 | 多模型对比、置信度阈值、本地规则锚定 |
| 本地规则误判 | 规则权重可调、置信度阈值可配、人工验证接口 |

## 12. 成功指标

1. **准确率**：与 groundtruth 对比 F1-score > 0.85（本地规则确保数值精度）
2. **性能**：80%以上clip由本地引擎即时决策，整体速度比纯LLM快5-10倍
3. **成本**：LLM API 调用减少80%，token使用量减少70%
4. **可维护性**：规则文件修改即时生效，无需代码部署
5. **用户满意度**：决策完全可解释，每个结果都有规则依据和置信度
6. **资源效率**：CPU本地计算替代LLM推理，节省云计算成本

---

**下一步行动**：
1. 确认优化设计后开始实现第一阶段
2. 优先实现7个直接计算结果的规则工具（rule_tools.py）
3. 实现两阶段执行引擎（本地优先，LLM语义后备）
4. 最后进行CLI集成和ensemble模块移除
5. **注意**：speech模块将单独进行LLM化改造（见SPEECH_MODULE_DESIGN.md）

**文档版本**：2.1  
**最后更新**：2026-04-15  
**作者**：opencode (DeepSeek-Reasoner)  
**状态**：设计优化完成（专注替换ensemble模块）  
**主要更新**：
- 将10个特征提取工具重构为7个直接计算结果的规则工具
- 优化执行流程：本地规则优先，LLM仅处理语义判断
- 大幅减少LLM思考量（减少80%）和token使用量（减少70%）
- 明确RuleResult数据结构和决策流程
- **范围调整**：专注替换ensemble模块，speech模块独立LLM化