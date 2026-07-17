---
title: 本地部署
order: 25
---

# 本地模型部署

本地部署大模型的核心挑战是在消费级硬件有限的显存和算力下，实现可用的推理速度和服务质量。与云端部署可以随意扩展 GPU 不同，本地部署受限于单张显卡的显存容量（通常 8-24GB）和 PCIe 带宽。优化技术的选择直接决定了模型能否跑起来、跑多快。

## 部署工具生态

目前主流的本地推理工具分为三个流派，各有侧重。

llama.cpp 是 C++ 实现的轻量级推理引擎，专为 CPU 推理和消费级 GPU 设计。它的核心优势是 GGUF 量化格式——支持 2-bit 到 8-bit 的各种量化方案，且量化后的模型是单个文件，分发极其方便。llama.cpp 对 Apple Silicon（Metal）、AMD GPU（ROCm）、NVIDIA GPU（CUDA）都有良好支持，CPU 推理性能在同类工具中最优。它的劣势是不支持 continuous batching（每个请求串行处理），不适合多用户并发场景，但作为个人使用的本地推理工具非常合适。

Ollama 是 llama.cpp 的上层封装，提供了类似 Docker 的模型管理体验。`ollama pull llama3` 一行命令下载模型，`ollama run llama3` 启动对话，屏蔽了量化参数选择、GPU 层数配置等底层细节。Ollama 的 Modelfile 支持自定义 system prompt 和参数，适合非技术用户快速上手。底层使用的仍是 llama.cpp 的推理能力。

SGLang 是更适合本地部署的推理框架，尤其是在多轮对话和 RAG 场景中。它的核心优势是 RadixAttention——一种自动前缀缓存机制。当多个请求共享相同的前缀（如 system prompt、RAG 检索结果、工具定义）时，RadixAttention 自动识别并复用 KV Cache，避免了重复的 prefill 计算。在多轮对话中，system prompt 和早期轮次每轮都在上下文里，RadixAttention 可以带来 5 倍以上的吞吐量提升——这对本地单 GPU 场景尤为关键，因为 prefill 是单卡推理中最耗时的环节。

SGLang 的另一个优势是显存开销更低。相比 vLLM，它在同等模型和上下文长度下通常节省 10-20% 的显存，意味着同样的 24GB 卡可以容纳更大的模型或更长的上下文。显存节省来自更高效的 KV Cache 管理和更轻量的调度器。此外 SGLang 的部署极其简单——`pip install sglang && sglang serve`，不需要 Ray 或多进程管理，对本地单 GPU 场景非常友好。

vLLM 本地模式适合需要 TP/PP 的多卡用户，它对多 GPU 分布式推理的支持更成熟。但对于最常见的本地场景——单 GPU、多轮对话、RAG 应用——SGLang 在速度、显存和易用性上都有优势。

## 硬件分级与模型选择

本地可用的模型规模直接取决于显存容量。FP16 模型每 1B 参数约需 2GB 显存，但推理时还需为 KV Cache 和中间激活值预留空间（通常额外 20-30%）。

8GB 显存（RTX 3070/4060 等）：可运行 7B-8B 的 INT4 量化模型，或 3B-4B 的 FP16 模型。Llama-3-8B 的 Q4_K_M 版本约 4.5GB，留出 3.5GB 给 KV Cache 可支持 8K 上下文。这是最普遍的入门配置。

16-24GB 显存（RTX 4080/4090、A4000 等）：可运行 13B-34B 的 INT4 量化模型，或 7B-8B 的 FP16 模型。Qwen2.5-32B 的 Q4_K_M 约 18GB，勉强装入 24GB 显存。Mixtral 8x7B（MoE 模型）因稀疏激活特性，INT4 量化后约 16GB，实际推理速度接近 14B Dense 模型。

32-48GB 显存（A6000、双卡 24GB 等）：可运行 70B 的 INT4 量化模型（约 36GB），或通过 vLLM TP=2 跨两张 24GB 卡运行 70B 模型。这是能够运行"大模型"的最低门槛。

## 量化格式选型

本地部署最关键的决策是选择哪种量化格式。不同格式在质量、速度和工具链支持上各有差异。

GGUF 是 llama.cpp 的原生格式，支持 2-8 bit 多种量化变体。Q4_K_M 是社区公认的"甜点"——在质量损失极小（PPL 退化 < 0.5）的情况下将模型压缩到约原来的 1/4。Q5_K_M 质量略好但体积大 15%，Q3_K_M 体积更小但质量损失明显。GGUF 的另一优势是支持 CPU+GPU 混合推理——将部分层放在 GPU 上，其余放在 CPU 内存中——突破了纯 GPU 推理的显存上限。

AWQ（Activation-aware Weight Quantization）通过分析激活值分布来指导量化，质量优于同位宽的 GPTQ。AWQ INT4 在多数模型上 PPL 退化不到 0.3。vLLM、TGI 等服务框架对 AWQ 有原生支持，推理速度优于 GGUF（因为可以直接在 GPU 上做 INT4 矩阵乘法）。

GPTQ 是更早的 LLM 量化方案，基于 Hessian 矩阵逐列补偿量化误差。GPTQ INT4 的精度与 AWQ 接近，但量化过程更慢（需要计算 Hessian），且跨领域泛化略差。vLLM 和 TGI 同样支持 GPTQ，存量模型较多。

选型建议：本地单用户、追求省心 → Ollama（一键下载运行）。多轮对话、RAG 应用、Agent 开发 → SGLang（RadixAttention 自动缓存重复前缀，提速明显）。多 GPU 或需要 TP/PP → vLLM（分布式支持最成熟）。CPU 推理或非 NVIDIA 硬件 → llama.cpp。

## 推理加速技术

本地部署中，模型加载之后的推理速度直接影响使用体验。除了量化减小模型体积外，还有几项关键的加速技术。

投机解码用小模型辅助大模型推理。本地运行 70B 模型时，可用同系列 7B 模型作为 draft model。DFlash 等扩散式投机方法进一步将 draft 从串行改为并行，加速比可达 2-6 倍。llama.cpp 已集成投机解码支持，通过 `--draft-model` 参数指定 draft 模型文件。

KV Cache 量化是长上下文场景的关键优化。TurboQuant 将 KV Cache 压缩到 3-4 bit，旋转处理将异常值通道的能量分散到所有通道，量化误差极小。原本 32K 上下文需要约 16GB 显存放 KV Cache，4-bit 量化后仅需 4GB。相同硬件可以支持 4 倍长的上下文，或在相同上下文下节省显存加载更大的模型。

FlashAttention 通过分块计算减少显存访问，将注意力计算的显存带宽利用率从 20% 提升到 80% 以上。大多数现代推理框架（vLLM、llama.cpp、Ollama）已默认启用。对于本地硬件，FlashAttention 的收益在长上下文场景下最为显著——8K 上下文时注意力计算可能占推理时间的 30-40%。

## 实战配置参考

以本地运行 Qwen2.5-32B 为例（目标：24GB 显存，8K 上下文）：

llama.cpp 方案：下载 Q4_K_M 版本（约 18GB），全部层放在 GPU 上（`-ngl 999`），KV Cache 用 INT8。启动命令：

```bash
./llama-cli -m qwen2.5-32b-Q4_K_M.gguf \
  --ctx-size 8192 --cache-type-k q8_0 --cache-type-v q8_0 \
  --temp 0.7 --top-p 0.9 -ngl 999
```

Ollama 方案：创建 Modelfile 指定量化版本和上下文长度，然后 `ollama create && ollama run`。Ollama 自动管理 GPU 层数和 KV Cache 类型，无需手动配置。

```dockerfile
FROM qwen2.5:32b-q4_K_M
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
```

SGLang 方案（推荐，多轮对话和 RAG 场景首选）：RadixAttention 自动缓存重复前缀，多轮对话吞吐量远高于 vLLM。启动命令简洁，单 GPU 无需额外配置：

```bash
pip install sglang
sglang serve qwen2.5-32b-AWQ --max-total-tokens 8192
```

SGLang 的 prefix caching 在多轮对话中自动生效，无需手动启用。如果本地有大量并发请求（如同时运行多个 Agent），SGLang 的 RadixAttention 能显著降低 prefill 开销。

vLLM 方案（多 GPU 或需要 TP/PP 时）：vLLM 对分布式推理的支持更成熟，`--gpu-memory-utilization` 控制显存使用率：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model qwen2.5-32b-AWQ \
  --max-model-len 8192 --gpu-memory-utilization 0.90
```
