---
title: 可观测性
order: 3
---
# 可观测性
LLM 应用的可观测性比传统软件更重要也更复杂。传统 Web 应用有明确的成功/失败标准（HTTP 状态码），但 LLM 应用的"失败"可能是模糊的——回答不准确、格式不符合要求、遗漏关键信息。可观测性需要从多个维度追踪 LLM 应用的行为，帮助开发者理解"到底发生了什么"并快速定位问题。

## 全链路追踪
LLM 应用的一个请求可能涉及多个步骤：接收用户输入 → 检索相关文档 → 构建 Prompt → 调用 LLM → 执行工具 → 生成最终回答。全链路追踪记录每个步骤的输入、输出、耗时和状态，可视化展示请求的完整路径。

### LangSmith
LangSmith 是 LangChain 官方的可观测性平台，提供了开箱即用的全链路追踪。每次 LLM 调用、工具调用、Prompt 变更都被记录，可以查看每一步的输入输出、执行时间和 Token 消耗。

```python
import os
os.environ["LANGSMITH_API_KEY"] = "your-key"
os.environ["LANGSMITH_PROJECT"] = "my-rag-app"

# 开启追踪后，所有 LangChain 调用自动记录到 LangSmith
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
# 这次调用会出现在 LangSmith 仪表板中
response = llm.invoke("什么是向量数据库？")
```

LangSmith 的核心功能包括：Trace 可视化（展示每步的执行时间和数据流向）、Prompt 版本管理（追踪每次 Prompt 修改和效果对比）、评估集成（直接在平台上对 Trace 进行标注和评分）、标签和过滤（按用户、场景、模型等维度筛选 Trace）。

### LangFuse
LangFuse 是开源的可观测性方案，功能与 LangSmith 类似但可以自部署。它的特色是成本追踪——详细记录每次调用的 Token 消耗和费用，按模型、用户、场景汇总成本。

```python
from langfuse import LangFuse

langfuse = LangFuse(
    public_key="pk-xxx",
    secret_key="sk-xxx",
    host="http://localhost:3000"  # 自部署地址
)

# 创建 Trace
trace = langfuse.trace(name="rag-query", user_id="user-123")

# 记录检索步骤
retrieval_span = trace.span(name="retrieval", input={"query": user_query})
docs = retriever.invoke(user_query)
retrieval_span.end(output={"docs_count": len(docs)})

# 记录 LLM 调用
generation = trace.generation(
    name="llm-response",
    model="gpt-4o",
    input=prompt,
    usage={"prompt_tokens": 100, "completion_tokens": 200}
)
generation.end(output=response)
```

LangFuse 的成本追踪特别适合多模型场景。当应用同时使用 GPT-4o、Claude、本地模型时，LangFuse 可以按模型分别统计费用，帮助识别成本优化的方向（如将部分 GPT-4o 调用切换到更便宜的模型）。

### Arize Phoenix
Arize Phoenix 专注于 LLM 应用的性能监控和故障排查。它自动构建调用链的依赖图，可视化展示请求从入口到 LLM、到向量数据库、到外部 API 的完整路径。当延迟升高或错误率上升时，Phoenix 能快速定位瓶颈节点。

## 成本追踪
Token 消耗是 LLM 应用最大的运营成本之一。精确的成本追踪需要记录每次 API 调用的 Token 用量，按模型、用户、场景、时间维度汇总。

**实时成本监控**：设置每日/每月预算上限，接近上限时发送告警。成本突增可能是配置错误（如意外使用了高价模型）或滥用攻击（恶意用户大量调用）的信号。

**成本归因**：将 Token 消耗归因到具体的功能模块或业务场景。客服问答模块消耗 40% 的 Token，代码审查模块消耗 30%，文档摘要模块消耗 20%，其他 10%。这种归因帮助确定优化的优先级。

**成本趋势**：追踪成本随时间的变化趋势。如果每次 Prompt 修改后成本显著变化，说明修改影响了输出长度或工具调用频率。

## 性能监控
LLM 应用的性能指标包括：

**端到端延迟**（P50/P95/P99）：从用户发送消息到收到完整回复的时间。P50 反映典型体验，P95/P99 反映长尾情况（复杂查询、模型负载高时的响应时间）。

**首 token 延迟**（Time to First Token, TTFT）：用户发送消息到看到第一个字的时间。这直接影响用户的感知等待时间——TTFT < 1 秒时用户感觉"很快"，> 3 秒时感觉"卡顿"。

**吞吐量**（Tokens per Second, TPS）：每秒生成的 token 数。TPS 受模型大小、推理硬件、批处理策略影响。

**错误率**：API 调用失败（超时、限流、内容审查拒绝）和工具执行失败的比例。

## 告警与故障排查
告警规则应该覆盖以下场景：

**延迟异常**：P95 延迟超过基线 2 倍。可能原因：模型提供商性能下降、网络问题、上下文过长导致推理变慢。

**错误率上升**：API 错误率超过 1%。可能原因：API Key 过期、提供商服务中断、请求格式变更。

**成本突增**：日均成本超过基线 3 倍。可能原因：用户量激增、模型路由配置错误（所有请求都走了高价模型）、攻击或滥用。

**质量下降**：用户负面反馈（点踩）比例上升。可能原因：Prompt 被意外修改、检索索引过期、模型提供商更新了默认模型版本。

故障排查的流程：收到告警 → 确认影响范围（全部用户/特定场景）→ 在可观测性平台查看相关 Trace → 定位异常环节（检索/LLM/工具）→ 对比正常和异常 Trace 的差异 → 制定修复方案。全链路追踪让这个过程从"盲人摸象"变成"有据可查"。
