---
title: 网关与缓存
order: 2
---
# API 网关与缓存
当应用需要调用多个 LLM 提供商、管理大量 API Key、追踪成本和用量时，API 网关成为必要的基础设施。语义缓存则是 LLM 应用特有的优化——传统缓存基于精确的键匹配，语义缓存基于向量相似度匹配"意思相近"的请求，命中率大幅提升。

## LLM API 网关

### LiteLLM
LiteLLM 是最流行的开源 LLM API 网关，封装了 100+ 模型提供商的 API 差异。它的核心价值是"写一次代码，用任何模型"——应用代码只需要对接 OpenAI SDK，LiteLLM 负责翻译到不同提供商的 API 格式。

```yaml
# litellm_config.yaml
model_list:
  - model_name: smart
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: smart
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: fast
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

router_settings:
  routing_strategy: usage-based-routing-v2
  num_retries: 2
  timeout: 30
  allowed_fails: 3
  cooldown_time: 60

general_settings:
  master_key: sk-your-gateway-key  # 网关鉴权
  max_budget: 100  # 每月预算上限（美元）
  budget_duration: 30d
```

启动网关：`litellm --config litellm_config.yaml --port 4000`，应用通过 `http://localhost:4000` 调用，格式与 OpenAI API 完全一致。网关的 `routing_strategy` 支持多种策略：轮询（round-robin）、最少延迟（least-busy）、成本优先（cost-based-routing）、用量均衡（usage-based-routing）。

### 限流与配额
LLM API 的限流有两个维度：提供商侧限制和应用侧限制。提供商侧限制（OpenAI 的 RPM/TPM）由网关自动处理，通过请求队列和速率控制避免触发提供商的 429 错误。应用侧限制按用户或项目设置配额，防止单个用户耗尽共享预算。

```python
# FastAPI 限流中间件示例
from fastapi import FastAPI, Request, HTTPException
from slowapi import Limiter

app = FastAPI()
limiter = Limiter(key_func=lambda: request.headers.get("X-API-Key"))

@app.post("/v1/chat/completions")
@limiter.limit("60/minute")  # 每个 API Key 每分钟 60 次
async def chat_completions(request: Request):
    # 检查 Token 配额
    usage = get_user_usage(api_key)
    if usage["tokens_used"] > usage["tokens_limit"]:
        raise HTTPException(429, "Token 配额已用尽")
    ...
```

限流策略应该区分用户等级。免费用户限制更严格（10 RPM），付费用户放宽（100 RPM），企业用户可以协商更高配额。Token 级别的限流比请求级别更精确——一个 5000 token 的请求和 50 token 的请求不应该消耗相同的配额。

## 语义缓存

### 原理
传统缓存使用精确的键匹配，用户问"什么是机器学习"和"解释一下机器学习"是两个不同的键，无法命中缓存。语义缓存将用户请求通过 Embedding 模型转换为向量，与缓存中的向量做相似度匹配，如果相似度超过阈值（如 0.92），直接返回缓存的回答。

```
用户: "什么是机器学习？" → Embedding → [0.12, 0.85, 0.33, ...]
                                          ↓ 余弦相似度
缓存: "解释一下机器学习" → [0.11, 0.84, 0.35, ...] → 相似度 0.97 > 阈值
                                          ↓ 命中
返回: 缓存的回答（成本 ≈ 0，延迟 < 50ms）
```

### GPTCache
GPTCache 是专门为 LLM 应用设计的语义缓存库。它封装了 Embedding 计算、向量存储、相似度匹配和缓存管理的完整流程。

```python
from gptcache import Cache
from gptcache.adapter import openai
from gptcache.embedding import OpenAI as EmbedOpenAI
from gptcache.manager import CacheBase, VectorBase, ObjectBase
from gptcache.similarity_evaluation import ExactMatchEvaluation

# 初始化缓存
cache = Cache()
embedding = EmbedOpenAI(model="text-embedding-3-small")

cache_base = CacheBase("sqlite", sql_url="sqlite:///cache.db")
vector_base = VectorBase("faiss", dimension=1536)
object_base = ObjectBase("local", path="./cache_objects")

cache.init(
    pre_embedding_func=lambda data: data.strip().lower(),
    embedding_func=embedding.to_embeddings,
    data_manager=CacheBase.manager("sqlite", vector_base=VectorBase("faiss", dimension=1536)),
    similarity_evaluation=ExactMatchEvaluation(),
)

# 使用缓存的 OpenAI 调用
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "什么是机器学习"}],
    cache_obj=cache  # 命中缓存时直接返回，不调用 API
)
```

### 缓存策略
**相似度阈值**是缓存质量的关键参数。阈值太高（0.99）缓存命中率低，形同虚设；阈值太低（0.80）会把不同问题错误地匹配到同一个答案。生产环境通常设在 0.90-0.95 之间，需要根据实际数据调优。

**缓存失效**比传统缓存更复杂。缓存的内容可能因为知识更新而过时——"2024 年世界杯冠军"的答案在 2026 年就不再准确。解决方法包括：TTL 过期（设置缓存有效期，如 24 小时）、手动失效（知识库更新时清除相关缓存）、版本标记（缓存附带时间戳，超过阈值的低优先级使用）。

**缓存预热**可以在系统启动时加载高频问题及其答案，避免冷启动时的缓存空洞。高频问题可以从历史查询日志中统计获得。

**缓存分区**按业务场景分离缓存池。客服问答的缓存不应该被技术问答污染，因为两者的问题向量分布差异很大，混合存储会降低匹配精度。每个分区可以有独立的相似度阈值和 TTL 策略。

### 缓存的局限
语义缓存不适合所有场景。创意性任务（"写一首诗"）每次应该生成不同结果，缓存会导致重复输出。高度个性化的任务（"根据我的偏好推荐"）每个用户的答案不同，缓存命中率极低。实时性要求高的任务（"今天的新闻"）缓存内容可能过时。

最适合缓存的场景是：事实性问答、FAQ 类查询、标准流程说明、固定格式输出。这些场景的问题表述多样但答案相对固定，语义缓存的价值最大。实际数据中，这类请求通常占总量的 30-50%，意味着 30-50% 的 API 调用可以被缓存拦截。
