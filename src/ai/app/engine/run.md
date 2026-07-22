---
title: 跑起模型
order: 5
---

# 如何把一个 AI 模型跑起来

假设你已经有了硬件——无论是一张 RTX 4060 的消费级平台，还是一台二手 EPYC + 多路 GPU 的服务器。接下来的问题是：下载哪个模型、用什么工具跑、参数怎么调。本文给出从零到跑起一个对话模型的完整流程。

## 选模型

模型选择取决于三个硬约束：显存容量、使用场景、以及对质量的要求。

显存是唯一不能讨价还价的约束。FP16 下每 1B 参数约需 2GB 显存，加上 KV Cache 和推理框架开销通常还需 20-30%。INT4 量化（Q4_K_M）将模型压缩到约原来的 1/4。因此：

- 8GB 显存（RTX 4060）：INT4 下能跑 7-8B 模型（Llama-3-8B ≈ 5GB）
- 24GB 显存（RTX 4090）：INT4 下能跑 32-34B 模型（Qwen2.5-32B ≈ 19GB）
- 48GB 显存（A6000 / 双卡 24GB）：INT4 下能跑 70B 模型（Llama-3-70B ≈ 38GB）

当前推荐模型（2026 年 7 月）：

| 模型          | 参数量            | 显存需求 (Q4_K_M) | 适合场景                   |
| ------------- | ----------------- | ----------------- | -------------------------- |
| Qwen3.6-8B    | 8B                | ~5 GB             | 中文对话、轻量 Agent       |
| Gemma-3-12B   | 12B               | ~8 GB             | 多语言对话、长上下文 (32K) |
| Qwen3.6-32B   | 32B               | ~19 GB            | 代码生成、复杂推理         |
| DeepSeek-V3.1 | 685B (37B active) | ~24 GB            | MoE 旗舰，综合能力最强     |
| Llama-4-70B   | 70B               | ~38 GB            | 需多 GPU 或高端卡          |

Qwen3.6 是目前开源中文能力最强的系列，32B 版本在编程和数学上接近闭源模型水平。Gemma 3 是 Google 的最新开源系列，12B 版本在同参数级别中长上下文（32K）和多语言表现突出。DeepSeek-V3.1 是 MoE 架构的旗舰，671B 总参数但每次只激活约 37B，综合能力对标 GPT-5 且支持 128K 上下文。

## 选工具

三条路径，选哪条取决于你的技术深度和使用场景。

**Ollama 路线**（终端用户，5 分钟跑起来）。安装后一条命令下载并启动模型——`ollama run llama3`。Ollama 自动管理 GGUF 量化版本、GPU 层数、KV Cache 类型，不需要任何配置。适合非技术用户、快速测试和日常对话。Modelfile 可以定制 system prompt 和参数。

```
# 安装后直接跑
ollama run qwen2.5:14b

# 或在 Modelfile 中定制
FROM qwen2.5:32b-q4_K_M
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
```

**llama.cpp 路线**（有动手能力，追求硬件控制）。直接下载 GGUF 文件后通过命令行启动。需要手动指定 GPU 层数（`-ngl`）、KV Cache 量化类型（`--cache-type-k/q8_0`）、上下文长度（`--ctx-size`）等。好处是对硬件的精细控制——可以指定哪些层在 GPU 上算、哪些卸载到 CPU、KV Cache 的量化精度、线程数。

```bash
# 下载 GGUF 模型（huggingface 上的 TheBloke 等提供）
# 启动服务
./llama-server -m qwen2.5-32b-Q4_K_M.gguf \
  --ctx-size 8192 --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 0.0.0.0 --port 8080 -ngl 999
```

**vLLM / SGLang 路线**（开发者，高频调用，生产环境）。需要较新的 NVIDIA GPU（Compute Capability 7.0+）和 CUDA 环境。支持 continuous batching——多个请求可以同时处理而不需要排队。SGLang 在单 GPU 场景下的部署更简单且 RadixAttention 对多轮对话有额外加速。

```bash
pip install sglang
sglang serve qwen/Qwen2.5-14B-Instruct --max-total-tokens 8192
```

## 参数调优

模型跑起来后，几个参数直接影响体验。

**Temperature**（0-2，常用 0.7-1.0）：控制随机性。设 0 时模型总选概率最高的 token（确定性的），适合代码生成或翻译。设 1.0 时输出有创造性，适合对话和写作。

**Top-p（nucleus sampling，0-1，常用 0.9）**：模型只从累积概率达到 p 的最小 token 集合中采样。p 越小输出越保守。对代码和数学任务设 0.1-0.5 可以显著减少错误。

**上下文长度（max context）**：模型能"记住"的最大 token 数。8K 满足大多数单轮对话。32K 可以放入长文档或多轮对话历史。增加的每个 token 都会消耗额外的 KV Cache 显存——8K vs 32K 上下文可能差出 4-6GB 的显存。

**Repeat Penalty**：抑制模型重复输出。如果模型开始循环同一句话，调高重罚因子（1.1-1.2）。

## 不同硬件的实战配置

### 消费级单卡（RTX 4060 8GB）

```
ollama run qwen2.5:7b-q4_K_M
```

或用 llama.cpp：

```bash
./llama-cli -m qwen2.5-7b-Q4_K_M.gguf \
  --ctx-size 4096 -ngl 999 --temp 0.7
```

能做的事：日常对话、翻译、简单的文档问答。不能做的事：32K+ 长文档分析、复杂多步推理。

### 消费级中高端（RTX 4090 24GB）

```
sglang serve Qwen/Qwen2.5-32B-Instruct-AWQ --max-total-tokens 8192
```

或用 Ollama：

```
ollama run qwen2.5:32b-q4_K_M
```

能做的事：代码生成和 review、复杂 RAG 问答、Agent 工作流。AWQ 量化后 32B 模型约 19GB，剩余 5GB 给 KV Cache 支持 8K 上下文。

### 本地服务器 EPYC + 多 GPU

单路 EPYC Rome 64 核 + 128GB RDIMM + 2×RTX 4090 48GB 组合的典型配置：

```
sglang serve Qwen/Qwen2.5-72B-Instruct-AWQ \
  --max-total-tokens 16384 --tp 2
```

TP=2 将模型拆分到两张 GPU 上。AWQ 量化后 72B 约 42GB，两张 24GB 卡通过 PCIe Gen4 x8 互联（带宽 ~16GB/s）。张量并行每层都需要 AllReduce，但模型层间的通信量相对可控。两张卡之间用 NVLink 桥效果更好——但没有 NVLink 的 PCIe 直连也能工作，只是每 token 的延迟会增加 5-10ms。

## 量化实施方案

量化是本地推理最关键的优化手段，直接决定模型能否装进显存。不同量化方案的选择取决于精度需求和硬件特性。

GGUF 量化通过 llama.cpp 的量化工具将模型转为多种精度的 GGUF 文件。Q4_K_M 是社区公认的甜点——在 PPL 退化小于 0.5 的情况下将模型压缩到原来的约 1/4。Q5_K_M 质量略好但体积大 15%，适合对精度要求高的场景。IQ 系列（IQ2_XXS、IQ3_XXS）进一步压缩到 2-3 bit，适合内存极其紧张的设备。GGUF 的另一关键优势是 CPU+GPU 混合推理——`-ngl` 参数控制多少层放在 GPU 上，其余层走 CPU 内存。即使显存放不下完整模型，只要系统内存足够大就能跑。

AWQ 量化通过分析激活值分布指导量化，INT4 下 PPL 退化通常不到 0.3，在 vLLM/SGLang 中推理速度接近 FP16。AWQ 的量化过程本身很快（约 10-30 分钟对一个 70B 模型），且不需要 Hessian 计算，对校准数据的依赖较小。AWQ 格式的模型在 HuggingFace 上可直接用 AutoAWQ 库加载，vLLM 和 SGLang 原生支持。

GPTQ 是更早的量化方案，基于 Hessian 矩阵逐列补偿量化误差。GPTQ INT4 精度与 AWQ 接近，但量化过程慢 2-3 倍（需要计算 Hessian），且对校准数据的分布更敏感。存量模型多，生态成熟，vLLM 的 `gptq_marlin` kernel 推理速度优秀。

选型建议：单卡本地推理用 GGUF + llama.cpp/Ollama（兼容性最好）；服务端高频调用用 AWQ + SGLang/vLLM（推理速度最优）；存量 GPTQ 模型也能用但新模型优先选 AWQ。

## 多卡推理

多 GPU 推理的关键挑战是跨卡通信。不同并行策略的通信开销差异巨大，选错策略会导致 GPU 利用率大幅下降。

### 张量并行

将模型每层的权重矩阵切分到多张 GPU 上。以 70B 模型为例，TP=2 时每张卡持有每层一半的权重。前向传播时，列并行的输出自然分片在各 GPU 上，行并行需要通过 AllReduce 求和。每层至少触发一次 AllReduce——通信频率极高。因此张量并行只适合节点内通过 NVLink 互联的 GPU（带宽 600-900 GB/s）。没有 NVLink 时只通过 PCIe 做 TP 带宽严重受限——PCIe Gen4 x8 仅 16 GB/s，约为 NVLink 的 1/40。

SGLang 启动 TP 只需一行：

```bash
# 2 卡 TP：每张卡显存减半
sglang serve Qwen/Qwen3.6-72B-Instruct-AWQ --tp 2

# 4 卡 TP：能跑 FP16 的 70B 模型
sglang serve meta-llama/Llama-4-70B --tp 4
```

TP=2 的实际效果：AWQ 量化的 72B 模型约 42GB，单张 24GB 卡装不下。拆到 2×24GB 后每卡约 21GB + 每层 AllReduce 通信开销。NVLink 下通信延迟约 2-5 μs，PCIe 下约 15-30 μs——每 token 生成时间 PCIe 比 NVLink 慢约 5-10ms。

### 流水线并行

将模型按层切分——GPU 0 持有 1-20 层，GPU 1 持有 21-40 层。数据像流水线一样滚动。通信只在段边界发生一次 Send/Recv，通信量远小于 TP。PP 适合节点间带宽较低的场景。气泡问题是 PP 的最大缺点：前端 GPU 计算时后端 GPU 空闲等待。PP 通常与 TP 组合——节点内 TP（高频通信走 NVLink），节点间 PP（低频通信走 InfiniBand 或以太网）。

```bash
# 4 GPU：2 节点内 TP × 2 节点间 PP
sglang serve Qwen/Qwen3.6-72B --tp 2 --pp 2
```

### 数据并行

多 GPU 各跑一份完整模型，各自处理不同请求。这是最高效的"多卡"方案——因为完全不需要跨卡通信，延迟为零，吞吐量线性增长。前提是单卡能装下完整量化模型。适合高并发推理场景（如同时服务多个用户）且每张卡的显存都够用。

### 选型总结

单卡能装下模型时不需要 TP/PP。多张 GPU 时判断逻辑：同节点内有 NVLink → 用 TP 拆分大模型；同节点内无 NVLink 且模型太大单卡装不下 → 只能做 TP 但通信损失严重，不如降级用更小的模型 + DP 分别服务；跨节点 → 用 PP + TP 组合，PP 承载跨节点通信。3D 并行（DP + TP + PP）通常在千卡以上集群中使用——个人本地环境不需要。

## 投机解码实施方案

投机解码用小模型加速大模型推理，在本地环境中收益显著——因为本地 GPU 的 prefill 资源本来就紧张。最新方案（DFlash、DDTree）可将推理速度提升 2-8 倍。

### 经典方案：Draft Model 接龙

大模型推理时，用一个更快的小模型（draft model）先生成 $k$ 个候选 token，大模型一次 forward pass 并行验证这 $k$ 个 token。通过的就保留，不通过的就从失败点由大模型自己重新生成。当 draft model 的准确率超过 80% 时，这 $k$ 个 token 的生成时间约等于 1 次大模型 forward + $k$ 次小模型 forward——而小模型的 forward 比大模型快 5-10 倍。

llama.cpp 集成了投机解码——同系列模型可直接用 4-bit 量化版做 draft：

```bash
# Qwen3.6-32B 主模型 + Qwen3.6-1.8B draft 模型
./llama-server -m qwen3.6-32b-Q4_K_M.gguf \
  --draft-model qwen3.6-1.8b-Q4_K_M.gguf \
  --speculative-tokens 8 -ngl 999
```

SGLang 的 EAGLE 投机引擎更高效——它使用与主模型共享 embedding/lm_head 的轻量 draft head，比独立 draft model 的分布对齐度更高，验证通过率可达 90%+。

```bash
sglang serve Qwen/Qwen3.6-32B --speculative-algorithm EAGLE \
  --speculative-num-tokens 8
```

### DFlash：扩散式投机

DFlash 把 draft 阶段从"逐 token 生成"改为"一次性并行扩散"。给定上下文，一个轻量扩散模型从噪声出发，经少数几步去噪迭代，一次性生成整个候选 token 块（8-16 个 token）。这完全消除了 draft model 的串行瓶颈——原本生成 8 个 token 需要 8 次 draft forward，现在只需要 1 次扩散去噪。候选块质量可以比逐 token 生成更高（扩散模型能看到块的全局结构）。

### 适用条件

投机解码的前提是 draft model 和主模型的 token 分布非常接近，否则验证通过率太低反而拖慢推理。最理想的搭配是同系列的模型——Qwen3.6-32B 配合 Qwen3.6-1.8B draft——共享词表和训练数据，分布对齐度最高，通过率可达 85-95%。跨系列搭配（如用 Llama-7B 做 Qwen-32B 的 draft）通常效果差，通过率可能只有 50-60%，甚至比不用投机还慢。本地单 GPU 上投机解码还需要额外的显存用于 draft model——Qwen3.6-1.8B 的 Q4_K_M 版本约 1.2GB，通常可以接受。

## 验证跑通

最简验证：问一个已知答案的问题。

```
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-32b","messages":[{"role":"user","content":"1+1等于几？"}]}'
```

返回 JSON 中有 `choices[0].message.content` 即为成功。LLM 服务启动后通常会有一个 Chat Web UI 在 `http://localhost:8080`，可以在浏览器直接使用。

## 常见问题

**模型下载慢**：HuggingFace 国内直连速度慢，设置镜像 `export HF_ENDPOINT=https://hf-mirror.com`。GGUF 文件通常在 HuggingFace 上也能找到——搜索模型名 + GGUF 即可。如果用 Ollama 则不需要手动下载，`ollama pull` 会自动从镜像或加速源拉取。

**OOM（显存不够）**：降低模型精度（Q4_K_M → Q3_K_M 甚至 IQ2_XXS）、缩短最大上下文长度（`--ctx-size 4096`）、减少 GPU 层数（`-ngl` 调小让更多层走 CPU 卸载）。

**生成速度慢**：检查模型是否全部层都在 GPU 上（Ollama 中 `ollama ps` 看 GPU 利用率，llama.cpp 中确认 `-ngl 999`）。CPU 卸载的每层会让速度掉 2-5 倍。如果是 vLLM/SGLang 出现频繁的 prefill 延迟过长，减少 `max-model-len` 或增加 `--gpu-memory-utilization`。

**输出乱码或循环**：调低 temperature（0.1-0.3）或调高 repeat_penalty（1.15-1.2）。有些模型对温度敏感——Qwen 系列在 temperature 低于 0.5 时表现更好，Llama 系列适合 0.7-0.9。
