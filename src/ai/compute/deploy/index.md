---
title: 部署
order: 100
---

# AI 模型部署
将 AI 模型高效地运行在实际硬件上，涉及从 GPU 选型、量化压缩、多卡并行到推理加速和服务部署的完整链路。每一层决策都直接影响模型能否跑起来、跑多快、服务多少用户。

从零开始的流程：本文下述章节覆盖选模型→选工具→参数调优→实战配置。深入各环节：[GPU 硬件选型](hardware) → [模型量化](quantization) → [多卡推理](multi-gpu) → [推理优化](optimization) → [服务部署](serving)。生产运维：[LLMOps](llmops) 负责评估、监控和成本控制。

推理引擎本身的架构和使用见[推理引擎](/ai/app/engine/)章节。

**如何把一个 AI 模型跑起来？**

假设你已经有了硬件——无论是一张 RTX 4060 还是一台二手 EPYC + 多路 GPU 服务器。接下来的问题是：下载哪个模型、用什么工具跑、参数怎么调。本文是从零到跑起一个对话模型的完整流程。

量化原理和格式选型见[模型量化](quantization)，多 GPU 策略见[多卡推理](multi-gpu)，推理加速技术见[推理优化](optimization)，GPU 硬件选型见 [GPU 硬件](hardware)。

## 选模型
模型选择取决于三个硬约束：显存容量、使用场景、质量要求。FP16 下每 1B 参数约需 2GB 显存，INT4 量化（Q4_K_M）压缩到约原来的 1/4：

- 8GB 显存：INT4 下能跑 7-8B 模型（Llama-3-8B ≈ 5GB）
- 24GB 显存：INT4 下能跑 32-34B 模型（Qwen2.5-32B ≈ 19GB）
- 48GB 显存：INT4 下能跑 70B 模型（Llama-3-70B ≈ 38GB）

当前推荐模型（2026 年 7 月）：

| 模型 | 参数量 | 显存需求 (Q4_K_M) | 适合场景 |
|------|--------|-------------------|---------|
| Qwen3.6-8B | 8B | ~5 GB | 中文对话、轻量 Agent |
| Gemma-3-12B | 12B | ~8 GB | 多语言对话、长上下文 (32K) |
| Qwen3.6-32B | 32B | ~19 GB | 代码生成、复杂推理 |
| DeepSeek-V3.1 | 685B (37B active) | ~24 GB | MoE 旗舰，综合能力最强 |
| Llama-4-70B | 70B | ~38 GB | 需多 GPU 或高端卡 |

Qwen3.6 是目前开源中文能力最强的系列。Gemma 3 在同参数级别中长上下文表现突出。DeepSeek-V3.1 的 MoE 架构每次只激活约 37B 参数，综合能力对标 GPT-5 且支持 128K 上下文。

## 选工具
三条路径，选哪条取决于技术深度和使用场景。

Ollama 路线（5 分钟跑起来）：`ollama run qwen2.5:14b` 一条命令下载并启动模型。自动管理 GGUF 版本、GPU 层数、KV Cache 类型。适合非技术用户、快速测试和日常对话。通过 Modelfile 定制 system prompt 和参数。

llama.cpp 路线（精细硬件控制）：手动指定 GPU 层数（`-ngl`）、KV Cache 量化（`--cache-type-k`）、上下文长度（`--ctx-size`）。CPU+GPU 混合推理突破显存上限。适合有动手能力、追求硬件控制的用户。

SGLang / vLLM 路线（生产环境、高频调用）：支持连续批处理，多请求并发处理。SGLang 在单 GPU 多轮对话场景的 RadixAttention 对 RAG 有额外加速，部署更简单。vLLM 对多 GPU 分布式推理支持更成熟。命令行一行启动，兼容 OpenAI API。

引擎的详细对比见[推理引擎](/ai/app/engine/)，工具生态和启动命令参考见[部署工具](eco)。

## 参数调优
Temperature（0-2，常用 0.7-1.0）控制随机性：代码生成设 < 0.3，创意写作设 0.8-1.0。Top-p（0-1，常用 0.9）从累积概率达到 p 的最小词集合中采样。上下文长度每增加一个 token 都消耗 KV Cache 显存——8K vs 32K 可能差出 4-6GB。Repeat penalty（1.1-1.2）抑制模型重复输出。

## 不同硬件的实战配置

### 消费级入门（RTX 4060 8GB）
```bash
ollama run qwen2.5:7b-q4_K_M
# 或 llama.cpp:
./llama-cli -m qwen2.5-7b-Q4_K_M.gguf --ctx-size 4096 -ngl 999 --temp 0.7
```

能做的事：日常对话、翻译、简单文档问答。不能做的事：32K+ 长文档分析、复杂多步推理。

### 消费级中高端（RTX 4090 24GB）
```bash
sglang serve Qwen/Qwen2.5-32B-Instruct-AWQ --max-total-tokens 8192
# 或 Ollama:
ollama run qwen2.5:32b-q4_K_M
```

AWQ 量化后 32B 模型约 19GB，剩余 5GB 给 KV Cache 支持 8K 上下文。能做代码生成和 review、复杂 RAG 问答、Agent 工作流。

### 本地服务器（EPYC + 多 GPU）
单路 EPYC Rome 64 核 + 128GB RDIMM + 2×RTX 4090 的典型配置：

```bash
sglang serve Qwen/Qwen2.5-72B-Instruct-AWQ --max-total-tokens 16384 --tp 2
```

TP=2 将模型拆分到两张 GPU，AWQ 量化后 72B 约 42GB。NVLink 桥接下通信延迟约 2-5 μs，PCIe 直连下约 15-30 μs——每 token 生成时间 PCIe 比 NVLink 慢约 5-10ms。多 GPU 并行策略的选择见[多卡推理](multi-gpu)。

## 验证跑通
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-32b","messages":[{"role":"user","content":"1+1等于几？"}]}'
```

返回 JSON 中有 `choices[0].message.content` 即为成功。LLM 服务启动后通常有 Chat Web UI 在 `http://localhost:8080`。

## 常见问题
模型下载慢：设置镜像 `export HF_ENDPOINT=https://hf-mirror.com`。GGUF 文件在 HuggingFace 上搜索模型名 + GGUF 即可。Ollama 会自动从加速源拉取。

OOM（显存不够）：降低模型精度（Q4_K_M → Q3_K_M）、缩短最大上下文长度（`--ctx-size 4096`）、减少 GPU 层数（`-ngl` 调小让更多层走 CPU 卸载）。

生成速度慢：检查是否全部层都在 GPU 上（Ollama 中 `ollama ps` 看 GPU 利用率，llama.cpp 中确认 `-ngl 999`）。CPU 卸载的每层会让速度掉 2-5 倍。投机解码可将 Decode 速度提升 1.5-2.5 倍，详见[推理优化](optimization)。

输出乱码或循环：调低 temperature（0.1-0.3）或调高 repeat_penalty（1.15-1.2）。Qwen 系列在 temperature 低于 0.5 时表现更好，Llama 系列适合 0.7-0.9。
