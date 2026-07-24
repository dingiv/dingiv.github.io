---
title: 推理优化
order: 30
---

# 推理优化
Transformer 大模型的一次推理请求由两个计算特性截然不同的阶段组成。理解这两个阶段是理解所有推理优化技术的起点。

## Prefill 与 Decode：两种计算范式
用户输入一个 prompt 后，模型并非"一边读一边写"，而是分两步工作。

**Prefill（预填充）阶段**——模型一次性处理输入 prompt 中的所有 token。所有 token 的注意力计算可以并行执行，GPU 的数千个 CUDA 核心同时工作，矩阵乘法填满计算单元。这个阶段的瓶颈是**算力（Compute-Bound）**——GPU 计算单元满负荷运转，显存带宽不是限制因素。Prefill 的耗时决定了**首字延迟（TTFT, Time To First Token）**——用户发出请求后多久看到第一个字。

**Decode（自回归生成）阶段**——模型逐 token 生成输出。每生成一个 token，都需要从显存中读取整个模型的权重（70B 模型约 140GB FP16），做一次完整的前向传播。但这一次前向传播只处理一个新 token，矩阵乘法的计算量远小于 Prefill 阶段，GPU 的计算单元大量空闲——时间主要消耗在从显存读取权重上。这个阶段的瓶颈是**显存带宽（Memory-Bound）**——生成速度（token/s）直接受限于显存带宽。Decode 的耗时决定了**生成延迟（TPOT, Time Per Output Token）**——每个 token 之间的间隔。

$$
\text{Prefill: } \text{瓶颈} = \text{算力 (TFLOPS)} \quad | \quad \text{Decode: } \text{瓶颈} = \text{显存带宽 (GB/s)}
$$

这一差异是所有推理优化技术的底层逻辑：解决首字慢和解决生成慢用的是完全不同的手段。混淆两个阶段的瓶颈，就会在错误的方向上浪费资源——给 Decode 加更多算力没用（GPU 已经在等显存了），给 Prefill 扩显存带宽收益也有限（GPU 计算单元才是瓶颈）。

以 RTX 3090 为例：FP16 算力 142 TFLOPS，显存带宽 936 GB/s。一个 70B 模型（FP16 约 140GB）在 Decode 阶段，每生成一个 token 需要读取约 140GB 权重数据。936 GB/s 的带宽下，理论最快生成速度为 936 ÷ 140 ≈ 6.7 token/s。即使 GPU 有无限算力，速度也不可能超过这个值——这就是 Memory-Bound 的本质。

## Prefill 优化：降低首字延迟
Prefill 是 Compute-Bound——优化方向是减少不必要的计算量、提升计算效率。

### FlashAttention：让注意力计算不再等显存
自注意力机制的理论瓶颈不在算力而在显存访问模式。标准的注意力计算需要反复读写一个 $N \times N$ 的注意力矩阵（$N$ 为序列长度），每次读写都经过显存。32K 上下文生成一个 $32000 \times 32000$ 的矩阵，即使是 FP16 也需要约 2GB 显存，而计算本身只需要几毫秒——绝大部分时间花在显存读写上。

FlashAttention 的核心思想是分块计算（Tiling）：将注意力矩阵切分为小块，每个块的计算完全在 GPU 的 SRAM（on-chip 缓存，带宽 > 10 TB/s）中完成，中间结果不回写显存。通过重新编排计算顺序，避免了显存带宽瓶颈。注意力计算的显存带宽利用率从 ~20% 提升到 80%+。FlashAttention-2 需要 sm_80（Ampere）及以上 CUDA 算力——这也是为什么 3090 是本地 AI 部署的架构起点。

8K+ 长上下文中，注意力计算可能占 Prefill 总时间的 30-40%。FA2 将其降到 10% 以下。大多数现代推理框架默认启用——不需要额外配置，只要 GPU 架构支持就自动生效。

### Prefix Caching：相同的开头只算一次
生产环境中大量请求共享相同的"开头"——system prompt（"你是一个有帮助的助手…"）、RAG 检索到的文档片段、工具调用的定义和示例。传统推理对每个请求都从头计算这些公共前缀的 KV Cache，重复消耗 Prefill 算力。

Prefix Caching 将已计算的 KV Cache 按前缀树（Trie）结构缓存。新请求到达时，如果前缀匹配缓存中的某个节点，直接复用该节点及所有祖先的 KV Cache，只计算不匹配的后缀部分。SGLang 的 RadixAttention 将这一思想做到极致——任何前缀匹配（不仅是完整 prompt 匹配）都能命中缓存，多轮对话中 system prompt 和早期轮次的 KV Cache 自动复用，prefill 计算开销降低 5 倍以上。

### Chunked Prefill：计算与访存的时间重叠
在并发服务场景中，一个长 prompt 的 Prefill 可能独占 GPU 数秒，期间所有其他请求的 Decode 被挂起——用户看到的是"转圈圈"。

Chunked Prefill 将一个长 prompt 切分为多个小块，与正在进行的 Decode 步拼装到同一个 batch 中同时提交 GPU。Prefill 的矩阵计算填满 CUDA 核心（Compute-Bound），同时 Decode 的访存利用显存带宽（Memory-Bound）——计算与访存在硬件层面重叠执行。对用户来说，其他请求的 Decode 不再被长 Prefill 阻塞。vLLM 和 SGLang 默认启用。

## Decode 优化：提升生成速度
Decode 是 Memory-Bound——优化方向是减少每个 token 需要从显存读取的数据量。

### 模型量化：读得更少就是算得更快
量化对 Decode 的加速效果往往被误解为"减少计算量"，实际上主要收益来自"减少显存读取量"。Q4_K_M 量化将权重从 16-bit 压缩到 4-bit，每个 token 需要从显存读取的权重数据量降为原来的 1/4。Memory-Bound 场景下，读取量减半→速度翻倍。回到 3090 + 70B 的例子：INT4 量化后权重约 35GB，理论 Decode 速度上限从 6.7 token/s 提升到 936 ÷ 35 ≈ 26.7 token/s。

量化方案和格式选型详见[模型量化](quantization)。Decode 优化的关键是：不是所有的量化对生成速度有同等提升——GGUF Q4_K_M 的 Decode 速度约是 FP16 的 3-4 倍，而 AWQ INT4 的 Decode 速度约是 FP16 的 2-3 倍（AWQ 在 GPU 上做 INT4 矩阵乘法需要反量化开销，GGUF 在 llama.cpp 中有更激进的 kernel 优化）。

### 投机解码：打破自回归的串行锁
Decode 的根本瓶颈不是显存带宽太小，而是自回归生成本身的串行性——必须先产生 token N，才能计算 token N+1。GPU 的并行计算能力在 Decode 阶段被浪费——几千个 CUDA 核心等着处理一个 token。

投机解码用算法换并行度。用一个极小的 draft model（1B-3B）快速串行预测 $k$ 个候选 token（draft model 很小，Memory-Bound 的瓶颈也小，生成速度极快），然后大模型一次 Prefill（并行验证）这 $k$ 个 token。通过的保留，不通过的回退。draft model 准确率超过 80% 时，$k$ 个 token 的总耗时 ≈ 1 次大模型 Prefill + $k$ 次小模型 Decode——而小模型的单次 Decode 比大模型快 5-10 倍。

投机解码不影响模型权重，精度无损。同系列模型搭配通过率最高——Qwen3.6-32B + Qwen3.6-1.8B draft 共享词表，通过率 85-95%。EAGLE 更进一步，使用与主模型共享 embedding/lm_head 的轻量 draft head，通过率可达 90%+。DFlash 将 draft 从逐 token 生成改为一次性并行扩散，完全消除了 draft 的串行瓶颈。

```bash
# llama.cpp: draft model 投机
./llama-server -m qwen3.6-32b-Q4_K_M.gguf \
  --draft-model qwen3.6-1.8b-Q4_K_M.gguf --speculative-tokens 8 -ngl 999

# SGLang: EAGLE 投机引擎
sglang serve Qwen/Qwen3.6-32B --speculative-algorithm EAGLE \
  --speculative-num-tokens 8
```

投机解码的代价是额外的显存——draft model 的权重和 KV Cache。Qwen3.6-1.8B Q4_K_M 约 1.2GB，通常可以接受。投机解码可以和量化、多卡并行叠加使用，效果累乘。

### MTP：免草稿模型的原生投机
MTP（Multi-Token Prediction）是投机解码的演进方向——将多 token 预测能力直接内置到模型架构中，完全消除对独立 draft model 的依赖。

传统投机解码需要在显存中额外驻留一个 draft model（1B-3B，约 1-2GB），且 draft model 必须与主模型共享词表。两个模型的加载和管理增加了工程复杂度——版本匹配、显存分配、框架兼容性都是隐藏成本。MTP 通过在模型最后一层添加额外的预测头来解决：主模型正常计算每个 token 的 hidden state，然后 $N$ 个并行的 MTP head 各自预测第 $t+1, t+2, \dots, t+N$ 位置的 token。一次 forward pass 同时输出当前 token + $N$ 个未来 token。

MTP head 的结构极其轻量——每个 head 通常只是 RMSNorm + 小规模线性投影，额外参数量约为主模型的 1%。DeepSeek-V3 的 MTP 进一步优化了 head 之间的因果依赖关系——前一个 head 的输出作为后一个 head 的额外输入——预测质量更高。Qwen 2.5 系列同样内置了 MTP 支持。推理框架检测到模型有 MTP head 时自动启用——不需要 `--draft-model` 参数。

MTP 相对于独立 draft model 的优势：零额外显存（MTP head 约 1% 参数增量已包含在模型权重中）、无需词表对齐（同一个模型天然一致）、框架兼容性更好（不需要管理两个模型的加载和切分）、通过率更高（MTP head 与主模型共享 hidden representation，分布对齐度优于独立小模型）。

MTP 的局限：需要模型本身支持 MTP head（Qwen 2.5、DeepSeek-V3 等少数新模型），大量旧模型没有；可预测的额外 token 数受 head 数量限制（通常 1-2 个，而独立 draft model 可预测 5-8 个）；单个 head 的预测质量通常低于同系列独立小模型（小模型有完整的多层 Transformer，MTP head 只是浅层投影）。

MTP 在多卡并行场景下的最大优势是与所有并行策略兼容——无论 PP 还是 TP，MTP 都随主模型自动生效。vLLM 中 PP + 独立 draft model 投机解码不兼容，但 PP + 原生 MTP（如果模型支持）可以同时工作，因为 MTP 不需要加载第二个模型。llama.cpp 中 MTP 和独立 draft model 都支持，可择一使用：

```bash
# llama.cpp: 原生 MTP（模型内置 MTP head）
./llama-server -m qwen3.6-32b-Q4_K_M.gguf --speculative-tokens 2

# vLLM: 原生 MTP 随 TP/PP 自动生效，无需额外参数
vllm serve Qwen/Qwen3.6-32B --tp 2
```

选型：模型支持 MTP → 优先用 MTP（零额外成本，框架兼容性完美）；模型不支持 MTP 但有同系列小模型 → 用独立 draft model（通过率 85-95%，1.5-2.5x 加速）；既无 MTP 又无合适 draft model → 无法用投机解码。MTP 和独立 draft model 不能同时使用——前者是内置方案，后者是外挂方案，解决同一个问题。

### KV Cache 压缩：GQA、MLA 与量化
Decode 阶段每个 token 不仅需要读取模型权重，还需要读取之前所有 token 的 Key 和 Value 缓存来做注意力计算。32K 上下文的 KV Cache（FP16）约 16GB——已经超过很多模型的量化权重。

GQA（Grouped Query Attention）是架构层面的压缩——多个 Query 头共享一组 KV 头。Llama-3 使用 8 组 KV 头（相比 32 个 Query 头），KV Cache 缩小 4 倍。这是现代开源模型能在 24GB 显卡上运行的架构前提。选择模型时优先挑自带 GQA 的版本——这是免费的架构红利。

MLA（Multi-Head Latent Attention，DeepSeek 提出）更激进——利用低秩矩阵分解将 KV Cache 压缩到极小潜空间，压缩比 90%+。KV Cache 从几十 GB 降到几 GB。MLA 是模型架构特性，下载支持 MLA 的模型（DeepSeek-V2/V3），部署即自动生效。

KV Cache 量化是后训练优化——将 K/V 缓存从 FP16 压缩到 4-8 bit。TurboQuant（Google Research, ICLR 2026）是目前最优方案：PolarQuant 随机旋转变换 + 标量量化，配合 QJL 1-bit 残差符号修正实现无偏注意力计算。4-bit KV Cache 后相同硬件支持 4 倍上下文。代价是 Attention 计算前需动态反量化，Decode 延迟增加 10-50%。硬件支持 FP8（Ada/Hopper 架构）时，FP8 KV Cache 零延迟但压缩比仅 2x。完整机制与选型决策见[模型量化](quantization)。

## CPU+GPU 混合推理
当显存放不下完整模型时，CPU+GPU 混合推理用系统内存补足显存缺口。代价是速度——数据经过 PCIe 总线从 CPU 内存传输到 GPU 时，瓶颈从 GPU 显存带宽（~936 GB/s）骤降到 PCIe 带宽（PCIe 4.0 x16 ≈ 31.5 GB/s，约为显存带宽的 1/30）。混合推理的核心调优目标：最大化 GPU 承载的层数、最小化 PCIe 传输、把 CPU 端推到性能极限。

不是所有框架都适合混合推理。llama.cpp 是为此场景设计的引擎，支持细粒度层切分（`-ngl`）、CPU 线程绑定、NUMA 优化和内存锁定。vLLM 和 SGLang 的 CPU offload 会引入 PagedKV 管理和额外内存开销，混合推理场景不推荐。

### 层切分策略：榨干每 MB 显存
原则是"能多放一层 GPU，绝不留给 CPU"。CPU 卸载的每一层会让 Decode 速度掉 2-5 倍——因为该层的权重数据需要从 CPU 内存经 PCIe 传到 GPU，每生成一个 token 都要经历一次。

计算单层显存开销：模型文件大小 ÷ 总层数 ≈ 单层权重。7B Q4_K_M 模型约 4.5GB、32 层，每层约 140MB。32B Q4_K_M 约 19GB、64 层，每层约 300MB。但实际分配时，不同层的权重张量大小不均——Attention 的 Q/K/V/O 投影和 MLP 的上下投影占用不同。

关键预留给 KV Cache。8K 上下文的 KV Cache（FP16）约 1.5-3GB，如果开启 KV Cache 量化（Q8_0）则减半至 0.8-1.5GB。先把 KV Cache 的显存预留出来，剩余空间全分给权重层。预留不足会导致推理中途 OOM，比一开始就少放几层更糟。

MoE 模型的特殊策略：优先把 Attention 层、Shared Expert、Embedding 全部塞进 GPU——这些是每个 token 必算的"全量"模块，放 GPU 收益最大。稀疏的 Routed Expert 放在 CPU 上，每次只激活其中一小部分，PCIe 传输开销可控。

```bash
# 第一步：测试极限——逐步增加层数直到 OOM
llama-cli -m qwen3.6-32b-Q4_K_M.gguf -c 4096 -ngl 999 -fa
# 观察 OOM 崩溃点（如第 28 层），安全值设崩溃点 -2~3 层

# 第二步：设定安全值 + KV Cache 量化
llama-cli -m qwen3.6-32b-Q4_K_M.gguf \
  -ngl 25 -c 8192 -ctk q8_0 -ctv q8_0 -fa

# 第三步：观察 kv-cache 量化后释放的显存，尝试 +1~2 层
llama-cli -m qwen3.6-32b-Q4_K_M.gguf \
  -ngl 27 -c 8192 -ctk q8_0 -ctv q8_0 -fa
```

### CPU 端调优
线程数设物理核心数（Physical Cores），禁用超线程。大模型矩阵乘法是高度密集的 ALU 运算——超线程的两个逻辑核共享 L1/L2 缓存和执行单元，互相抢占资源，实测反而降低 10-20% 推理速度。对于 Intel 13/14 代异构 CPU（P-Core + E-Core），只绑定 P-Core 的数量。

```bash
# 查询物理核心数
lscpu | grep "Core(s) per socket"
# 假设 14 个物理核心
llama-cli -m model.gguf -ngl 20 -t 14
```

内存带宽直接决定 CPU 算层的速度上限。DDR4 2400MHz 单通道约 19 GB/s，四通道约 76 GB/s——差距 4 倍。为混合推理配机时，内存通道数的重要性高于 CPU 核心数。双路服务器需要开启 NUMA 绑定（`--numa distribute`），防止 CPU 0 跨 Socket 访问 CPU 1 的内存，跨 NUMA 节点延迟比本地高 2 倍以上。

`--mlock` 强制将 CPU 端的模型权重锁定在物理内存中，禁止操作系统 swap 到硬盘。被 swap 出去的页面一旦需要访问，延迟从纳秒级变成毫秒级——直接表现为 Tokens 输出卡顿。

### 完整调优检查清单
按顺序执行：

1. 确定极限：`llama-cli -m model.gguf -c 4096 -ngl 999 -fa`，记录 OOM 崩溃层数
2. 设安全值：`-ngl` 降到崩溃点以下 2-3 层，为 KV Cache 留足余量
3. KV Cache 量化：加入 `-ctk q8_0 -ctv q8_0`，观察释放的显存，尝试加 1-2 层
4. 绑定 CPU：`-t <物理核心数> --mlock`，双路服务器加 `--numa distribute`
5. 验证吞吐：对比 `eval time`（Prefill）和 `predict time`（Decode），微调 `-t` 找最高吞吐点
6. 最终确认：`nvidia-smi -q -d PCI_EXPRESS` 确认显卡运行在正确的 PCIe 速率下（x8 或 x16）

混合推理的性能天花板很明确——CPU 卸载层数每增加 10%，Decode 速度约下降 30-50%。如果卸载层数超过总数的 50%，不如考虑更强的量化（Q3_K_M 甚至 IQ2_XXS）让更多层放进 GPU。

## 跨阶段优化

### 连续批处理：填满 GPU 的等待间隙
并发场景下，不同请求处于不同阶段——有的在 Prefill（吃算力），有的在 Decode（等显存）。连续批处理（Continuous Batching）在每个迭代动态决定本轮 batch 中包含哪些请求：新请求的 Prefill 可以立即加入，已完成的请求立即释放资源。配合 PagedAttention 的 block 级 KV Cache 管理，请求加入和退出只需分配/释放 block 指针，无内存搬移开销。高并发场景下吞吐量提升 5-10 倍。

### Prefill/Decode 分离：为两个阶段配不同的硬件
解耦部署（Disaggregated Serving）将 GPU 集群物理划分为 Prefill 节点池和 Decode 节点池。Prefill 节点配高算力 GPU（如 H100），Decode 节点配大显存带宽 GPU——各自配备最适合该阶段瓶颈的硬件。Prefill 节点算完 KV Cache 后通过 RDMA（InfiniBand/RoCE）高速推送给 Decode 节点。

解耦部署在云端大厂是标准实践，但本地单机/小集群场景不推荐——它需要 400Gbps+ InfiniBand 网络传输 GB 级 KV Cache，本地 10G/25G 网卡传输时间远超计算时间。且低并发时硬件利用率暴跌——Prefill 节点在没人提问时全程空闲。Chunked Prefill 在单机内已实现计算与访存的重叠，是解耦部署在本地场景的最优替代。

## 本地落地方阵
在本地单 GPU 或 PCIe 多卡环境下，优化技术按投入产出比排序：

选择自带 GQA/MLA 的模型（Qwen 2.5、DeepSeek-V3）——零成本获取 KV Cache 压缩。启动服务确保 FlashAttention-2 启用（现代框架默认）。Prefill 慢（首字延迟高）→ 开启 Prefix Caching（SGLang RadixAttention，多轮对话场景收益最大）。Decode 慢（生成速度低）→ 开启投机解码（1.5-2.5x，draft model 需同系列）。长上下文场景 → KV Cache 4-bit 量化（显存节省 > 权重量化收益）。高并发 API 服务 → 连续批处理（vLLM/SGLang）。显存放不下模型 → CPU+GPU 混合推理（llama.cpp，层切分+KV Cache 量化优先）。多 GPU 场景 → 量化+DP 多实例覆盖并发，或 PP 拆分大模型（见[多卡推理](multi-gpu)）。

关键认知：Prefill 瓶颈加算力，Decode 瓶颈减数据。选错方向比不作为更糟——给一台 3090 加第二张卡（算力翻倍）对 Decode 速度几乎无影响。混合推理的核心不是"CPU 能帮 GPU 算"，而是"尽量不让 CPU 算"——KV Cache 量化省出 1-2GB 显存多放 8-16 层到 GPU，收益远大于调 CPU 线程数。
