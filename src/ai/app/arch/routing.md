---
title: 模型路由
order: 1
---
# 模型路由与降级
不是所有问题都需要最强的模型。"今天天气怎么样"和"分析这段代码的安全漏洞"的复杂度差了几个数量级，如果都用 GPT-4o 处理，简单问题的成本和延迟都是浪费。模型路由（Model Routing）根据请求特征选择最合适的模型，级联降级（Cascading Fallback）在模型失败时自动切换备用方案，两者共同实现成本、质量和可用性的动态平衡。

## 路由策略

### 基于规则的路由
最简单的路由方式是根据请求的特征匹配规则。用户发送的消息长度短（< 100 字）且包含简单关键词（"天气"、"时间"、"翻译"），路由到轻量模型（GPT-4o-mini、Claude Haiku）。消息长度超过阈值或包含复杂关键词（"分析"、"设计"、"比较"），路由到强模型。工具调用请求路由到支持 Function Calling 的模型。

```python
def route_request(messages, tools=None):
    last_message = messages[-1]["content"]
    complexity = estimate_complexity(last_message)

    if tools and len(tools) > 5:
        return "gpt-4o"  # 复杂工具调用需要强模型
    if complexity == "simple":
        return "gpt-4o-mini"
    elif complexity == "medium":
        return "gpt-4o"
    else:
        return "claude-opus-4-20250514"
```

基于规则的路由优点是确定性高、延迟低（不需要额外推理）、易于调试。缺点是规则难以覆盖所有场景，维护成本随业务复杂度增长。适合请求模式相对固定的应用。

### 基于模型的路由
用一个小模型（或专用分类器）判断请求的复杂度，再路由到对应模型。这比规则更灵活，能处理未预见的请求模式。具体做法是将用户消息输入路由模型，输出一个分类标签（simple/medium/complex）或直接输出目标模型名。

```python
import openai

router_prompt = """判断以下问题的复杂度，只回答 simple、medium 或 complex：
- simple: 简单事实查询、翻译、短文本生成
- medium: 需要 2-3 步推理的分析、中等长度内容生成
- complex: 多步推理、代码审查、长文档分析、需要调用多个工具
"""

def route_by_model(user_message):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 用小模型做路由，成本极低
        messages=[
            {"role": "system", "content": router_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0,
        max_tokens=10
    )
    complexity = response.choices[0].message.content.strip()
    model_map = {
        "simple": "gpt-4o-mini",
        "medium": "gpt-4o",
        "complex": "claude-opus-4-20250514"
    }
    return model_map.get(complexity, "gpt-4o")  # 默认中等模型
```

路由模型的开销很小——一次小模型调用约 50ms 和 $0.0001，但可能节省一次大模型调用的 $0.05-0.50。只要路由准确率超过 80%，就能显著降低整体成本。

### 语义路由
将用户请求向量化，与预定义的意图向量做相似度匹配，根据匹配到的意图选择模型。这种方法利用了 Embedding 的语义理解能力，"帮我查天气"和"今天外面冷不冷"会被匹配到同一个"简单查询"意图，路由到同一个模型。

语义路由的初始化成本较高（需要定义意图和对应的示例），但运行时成本极低（一次向量相似度计算），适合高并发场景。

## 级联降级
生产环境中的模型服务不是 100% 可用的。OpenAI API 可能限流或宕机，自建推理服务可能 OOM 或 GPU 故障。级联降级定义了一组有序的备用模型，主模型失败时自动切换。

```python
from langchain_openai import ChatOpenAI

# LangChain 的 fallback 机制
primary = ChatOpenAI(model="gpt-4o")
fallback1 = ChatOpenAI(model="claude-sonnet-4-20250514", base_url="https://api.anthropic.com/v1")
fallback2 = ChatOpenAI(model="gpt-4o-mini")

llm = primary.with_fallbacks([fallback1, fallback2])
response = llm.invoke("分析这段代码")  # 自动尝试 fallback
```

级联降级的关键设计决策：

**降级触发条件**不应该只在完全失败时触发。API 延迟超过阈值（如 30 秒）、返回质量明显下降（如输出过短或格式错误）、或提供商宣布服务降级，都应该触发降级。

**模型质量降级需要告知用户**。从 GPT-4o 降级到 GPT-4o-mini 时，可以在响应中附加降级标记，前端展示"当前使用轻量模型，结果可能不够精确"。这种透明性让用户理解为什么回答质量下降，而不是归咎于产品本身。

**自动恢复**。降级后不应永久使用备用模型。健康检查线程定期探测主模型的可用性，恢复后自动切回。切回时可以选择渐进式（先 10% 流量走主模型，确认无问题后逐步恢复到 100%）。

## 成本优化组合
模型路由和级联降级可以与语义缓存组合，形成多层成本优化：

1. 请求到达 → 语义缓存检查（命中则直接返回，成本 ≈ 0）
2. 缓存未命中 → 路由判断（简单 → 小模型，复杂 → 大模型）
3. 模型调用失败 → 级联降级（主模型 → 备用模型）
4. 响应返回 → 写入缓存（下次相似问题直接命中）

这种多层架构的开销主要是路由判断的延迟（50-200ms）和缓存查询的延迟（1-10ms），相比于大模型调用的数秒延迟，这个开销可以忽略不计。但整体成本可以降低 50-70%，因为大部分用户请求是简单且重复的。
