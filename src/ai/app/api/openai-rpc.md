---
title: OpenAI_API
order: 15
---

# OpenAI RPC 库生态
OpenAI 的 Python/JS SDK 是 LLM 应用的事实标准接口——不是因为它设计得最好，而是因为整个行业的兼容策略都围绕它展开：vLLM、SGLang、llama.cpp、Ollama、各大云厂商的托管服务全部实现 OpenAI 兼容 API。学会一个 SDK 等于学会了调用所有推理后端。这个"RPC 库"的生态值得从接口设计、多后端切换、流式协议三个维度理解。

## 接口形态：像本地函数一样调用远程模型
SDK 把 HTTP REST 调用包装为类型化方法。核心入口是 `chat.completions.create`：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen3.6-32b",
    messages=[
        {"role": "system", "content": "你是一个简洁的技术助手"},
        {"role": "user", "content": "解释什么是 KV Cache"}
    ],
    temperature=0.7,
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

接口的隐含契约：请求是**无状态**的（全部上下文在 messages 数组中显式传递——[会话状态](/ai/app/overlay/session/)由客户端维护）、响应是**结构化**的（choices/content/usage 字段有固定 schema）、模型通过字符串引用（model 字段——同一 API 服务多个模型）。

参数体系值得记忆的分类：采样参数（temperature/top_p 控制随机性）、长度参数（max_tokens 限制输出）、停止条件（stop 序列）、惩罚参数（frequency_penalty/presence_penalty 抑制重复）。这些参数在不同后端的兼容度不一——本地引擎通常全支持，云厂商的兼容层可能只支持子集。

## 多后端：base_url 的切换魔法
OpenAI SDK 设计中最有生态影响力的是 `base_url` 参数——SDK 不绑定 OpenAI 的服务器，任何实现兼容 API 的服务都能接入：

```python
# OpenAI 官方
client = OpenAI()
# 本地 vLLM
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
# 本地 Ollama（Ollama 暴露 OpenAI 兼容端点）
client = OpenAI(base_url="http://localhost:11434/v1", api_key="none")
# 云厂商兼容层（如 DeepSeek、零一万物）
client = OpenAI(base_url="https://api.deepseek.com/v1", api_key="sk-...")
```

这个设计促成了"应用一次编写、后端随时切换"的格局——开发时用本地小模型（零成本），生产切到云端（或反过来：云端开发、数据敏感场景切回本地）。切换的粒度可以到请求级——同一个应用中简单任务路由到便宜模型、复杂任务路由到强模型（级联策略的实现基础）。

兼容层的实现质量参差是实际的坑点：字段缺失（某些后端不返回 usage）、错误格式不统一（有的返回标准 error object、有的返回裸字符串）、流式 chunk 格式差异。跨后端的应用要测试每个目标后端的实际响应格式，或使用 LiteLLM 这类代理层抹平差异。

## 流式：SSE 与增量聚合
LLM 的逐 token 生成使流式输出成为标配体验。OpenAI API 的流式协议是 **SSE（Server-Sent Events）**——HTTP 长连接上以 `data: {...}\n\n` 格式推送增量：

```python
stream = client.chat.completions.create(
    model="qwen3.6-32b",
    messages=[...],
    stream=True,   # 关键参数
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

流式的数据结构从完整响应的 `message.content` 变为增量的 `delta.content`——每个 chunk 只携带新增的 token 片段。客户端聚合 delta 得到完整文本。chunk 中还携带 role（首个 chunk 标明 assistant 角色）、finish_reason（末尾 chunk 标明停止原因——stop/length/content_filter）。

SSE 相比 WebSocket 的取舍：SSE 是单向的（客户端不能在流中发送）、基于普通 HTTP（防火墙和代理友好）、自动重连（浏览器原生 EventSource）。LLM 场景只需要服务端推送，SSE 的单向性不是限制——这解释了为什么整个行业选择 SSE 而不是 WebSocket。

函数调用（Function Calling / Tool Use）是接口的进阶能力：请求中声明可用工具的 JSON Schema，模型决定调用哪个工具并返回结构化的参数（`tool_calls` 字段），客户端执行工具后把结果作为 `role: "tool"` 消息追加到 messages 再次调用——循环直到模型给出最终回答。这个循环是 [Agent](/ai/app/overlay/agent/) 架构的通信基础。

## 生态位
OpenAI SDK 之上还有两层封装。**LiteLLM**：代理层 + 统一 SDK——用同一套调用格式访问 100+ 提供商（OpenAI/Anthropic/Google/本地引擎），附带路由、限流、成本追踪、fallback 策略。**LangChain/LlamaIndex 的模型接口**：框架级的抽象——Chain 和 Agent 使用框架的 Model 接口，接口下再适配各家 SDK。选择建议：单一后端用官方 SDK 足够；多云/本地混合路由用 LiteLLM；完整应用框架用 LangChain 的抽象（但要理解框架抽象与原生 SDK 的能力差异——最新参数往往框架滞后支持）。
