---
title: API 设计
order: 35
---
# LLM API 设计
LLM 应用的 API 层是连接前端/客户端与模型服务的桥梁。与传统的 REST API 不同，LLM API 需要处理流式传输、Token 计费、多模型代理等特殊需求。OpenAI 的 Chat Completions API 已经成为事实标准，几乎所有主流模型服务商和推理框架都兼容这一协议。

## OpenAI 兼容协议
OpenAI 的 `POST /v1/chat/completions` 接口定义了 LLM 调用的标准格式。请求体包含 `model`（模型名）、`messages`（消息数组，每条有 role 和 content）、`temperature`、`max_tokens` 等参数。响应包含 `choices`（生成结果）、`usage`（Token 消耗统计）。

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama 兼容端点
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "解释什么是向量数据库"}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
print(f"Token 用量: {response.usage}")
```

协议兼容的价值在于生态复用。vLLM、Ollama、TGI 等推理框架都实现了 OpenAI 兼容的 API，意味着用 OpenAI SDK 写的应用可以无缝切换到本地模型，只需修改 `base_url`。LangChain、LlamaIndex 等框架也基于这个协议封装了通用的 Model I/O 层。

工具调用（Function Calling）通过 `tools` 参数定义可用函数，模型返回 `tool_calls` 指示程序执行函数，程序将结果通过 `tool` role 消息返回，模型基于结果继续生成。这套机制已经被 Anthropic、Google 等厂商采纳，成为跨平台的工具调用标准。

## 流式传输
LLM 生成响应需要数秒到数十秒，如果等待完整响应再返回，用户体验很差。流式传输（Streaming）将响应拆分为多个 chunk 逐步返回，前端可以实时渲染"打字机效果"。

Server-Sent Events (SSE) 是实现流式传输的标准协议。客户端发送 `stream: true` 参数，服务端保持 HTTP 连接开放，逐个 token 发送数据块，每个 chunk 格式为 `data: {json}\n\n`，最后的 `data: [DONE]\n\n` 标记传输结束。

```python
# 服务端 SSE 实现
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import openai

app = FastAPI()

@app.post("/api/chat")
async def chat(request: ChatRequest):
    client = openai.OpenAI()
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=request.messages,
        stream=True
    )

    def generate():
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

流式传输的工程细节值得注意。Nginx 等反向代理默认会缓冲响应，需要设置 `X-Accel-Buffering: no` 禁用缓冲。超时设置要足够长（通常 60-120 秒），因为 LLM 生成长响应需要时间。工具调用的流式处理更复杂——模型可能在流式输出中间突然切换到工具调用模式，需要解析部分 JSON 来判断是否包含 `tool_calls` 字段。

## Token 计费
Token 是 LLM API 的计费单位。输入 Token 和输出 Token 的单价不同，输出通常贵 2-3 倍（因为生成比理解计算量更大）。GPT-4o 的输入约 $2.50/1M tokens，输出约 $10/1M tokens（2026 年价格持续下降中）。

计费的关键是准确统计 Token 用量。`tiktoken` 是 OpenAI 官方的 Token 计数库，支持不同模型的编码器（cl100k_base 用于 GPT-4，o200k_base 用于 GPT-4o）。对于 OpenAI 兼容的 API，响应中的 `usage` 字段提供了精确的 Token 统计。

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o")
text = "这是一个测试文本"
tokens = encoding.encode(text)
print(f"Token 数量: {len(tokens)}")
```

成本控制是 LLM 应用的运维重点。常见策略包括：限制单次请求的 max_tokens（防止无限生成）、设置用户/项目的日/月预算上限、使用更小的模型处理简单任务、缓存重复请求的结果。LiteLLM 等网关工具内置了预算管理和成本追踪功能。

## API 版本管理
LLM API 的版本管理比传统 API 更复杂，因为模型本身在不断更新。GPT-4 到 GPT-4o 到 GPT-5，每个新版本的行为可能不同，但 API 接口保持兼容。最佳实践是在请求中指定具体的模型版本（如 `gpt-4o-2024-08-06`），而不是使用别名（如 `gpt-4o`），这样即使模型提供商更新了默认指向，你的应用行为也不会变化。

模型切换是版本管理的常见场景。从 GPT-4 切换到 Claude 或本地模型时，虽然 API 协议兼容，但行为差异（输出风格、工具调用格式、安全策略）可能导致问题。渐进式切换策略是先让 5% 流量走新模型，观察质量指标（准确率、延迟、成本），确认无问题后逐步扩大。

## API 网关
多模型场景下，API 网关负责统一入口、路由请求、管理认证。LiteLLM 是最流行的开源 LLM 网关，它封装了 100+ 模型提供商的 API 差异，对外暴露统一的 OpenAI 兼容接口。

```yaml
# LiteLLM 配置示例
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: gpt-4o
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  routing_strategy: usage-based-routing-v2
  allowed_fails: 3
  cooldown_time: 60
```

网关的负载均衡策略包括轮询（round-robin）、最少延迟优先（least-latency）、成本优先（cheapest）、用量均衡（usage-based）。对于高可用场景，同一个模型名可以配置多个提供商（如 OpenAI 和 Azure OpenAI），网关在主提供商失败时自动切换到备用。

限流是 API 网关的核心功能。LLM 提供商通常有 RPM（每分钟请求数）和 TPM（每分钟 Token 数）限制，网关需要在应用侧实现请求队列和速率控制，防止突发流量触发提供商的限流。用户级别的限流则需要按 API Key 追踪用量，超出配额时返回 429 状态码。
