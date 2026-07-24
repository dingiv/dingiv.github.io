---
title: 多卡推理
order: 25
---

# 多卡推理
当单卡显存放不下模型时，多 GPU 并行是唯一的出路。但并行策略选错，多卡可能跑得比单卡还慢——通信开销吃掉计算收益。理解每种并行策略的通信模式和硬件前提，是做对选型的关键。

## 三种并行策略
多 GPU 推理有三种基本并行方式，按通信频率从低到高排列：

**数据并行（DP）**：每张卡持有完整模型副本，各自处理不同请求，卡间完全不需要通信。吞吐量线性增长，但单卡必须能装下完整模型。这是最简单的"多卡"方案——如果能用 DP，就不要用更复杂的策略。

**流水线并行（PP）**：将模型按层切分——GPU 0 持有 1-16 层，GPU 1 持有 17-32 层。数据像流水线一样滚动，卡间只在层边界传递一次激活值。通信量极小，PCIe 3.0/4.0 完全够用。代价是存在"气泡"——前端 GPU 计算时后端 GPU 空闲等待。

**张量并行（TP）**：将每层权重矩阵切分到多张 GPU，层内计算需要 AllReduce 同步。一个 32 层的模型，每生成一个 token，卡间就要进行 64 次通信。通信频率极高——必须有 NVLink（600 GB/s+）支撑，纯 PCIe 下 TP 通信延迟可能远超计算时间。

三者的核心差异：

| 维度 | DP | PP | TP |
|------|----|----|-----|
| 通信频率 | 无 | 极低（层边界一次） | 极高（每层 2 次） |
| 带宽要求 | 无 | PCIe 3.0 即可 | 必须 NVLink |
| 能否"凑显存" | 不能（单卡须装全量） | 能（按层切分） | 能（按矩阵切分） |
| 降低单请求延迟 | 不能 | 不能（甚至略增） | 能（多卡合力算一层） |
| 提升并发吞吐 | 能（QPS 线性增长） | 能（配合 micro-batch） | 较弱 |

## 通信硬件：NVLink vs PCIe
NVLink 是 NVIDIA 的卡间直连技术，Ampere 代（3090、A6000 等）提供 112.5 GB/s 双向带宽。PCIe 4.0 x16 单向带宽仅 31.5 GB/s，约为 NVLink 的 1/4。TP 每层触发两次 AllReduce，NVLink 下通信耗时占总时间 < 5%，PCIe 下可能 > 50%。完整的 NVLink 兼容性参考见 [GPU 硬件](hardware)。

不具备 NVLink 的卡（RTX 4090、3080 等）多卡通信只能走 PCIe，优化方向：

开启 PCIe P2P：数据直接在 PCIe 总线上跨卡传输（GPU A → PCIe Switch → GPU B），不经系统内存中转。检查 `nvidia-smi topo -m`——PIX/PXB 表示 P2P 已生效，SYS 表示仍在走 CPU 中转，需检查 BIOS ACS 设置。

NCCL 参数调优：
```bash
export NCCL_P2P_DISABLE=0        # 强制开启 P2P
export NCCL_ALGO=Tree,Ring       # 选择适合 PCIe 拓扑的算法
export NCCL_BUFFSIZE=8388608     # 增大缓冲池，减少小包传输开销
```

物理拓扑优化：确保所有 GPU 插在同一 NUMA 节点的 PCIe 插槽上，跨 NUMA 通信带宽减半、延迟翻倍。使用 PCIe Switch 扩展板可以让 GPU 间通信在 Switch 芯片内部转发，不经 CPU。

## 本地部署的并行策略
**单卡能装下模型时**——不需要 TP/PP。用多卡跑 DP 多实例，前面挂负载均衡，获得最高 QPS。这是最简单、最高效的方案。

**单卡装不下、有 NVLink 时**——用 TP=2 拆分模型。NVLink 带宽充裕，通信开销可忽略，推理延迟接近单卡的 1/2。双卡 3090 + NVLink 桥是本地跑 70B 模型的最优方案。

**单卡装不下、无 NVLink 时**——优先用 PP（流水线并行），将模型按层切分到不同卡上。PP 的通信量极小，PCIe 完全够用。避免在纯 PCIe 环境下使用高数位 TP——你会发现 GPU 绝大多数时间在挂起等待数据传输。

```bash
# SGLang: TP=2（有 NVLink 场景）
sglang serve Qwen/Qwen3.6-72B-Instruct-AWQ --tp 2

# SGLang: PP=2（无 NVLink 场景，按层切分）
sglang serve Qwen/Qwen3.6-72B-Instruct --pp 2

# SGLang: TP=2 + PP=2 混合（4 GPU）
sglang serve Qwen/Qwen3.6-72B --tp 2 --pp 2

# vLLM: DP 多实例（单卡能装下，追求高吞吐）
# 在每个 GPU 上分别启动一个服务实例，前面用 Nginx 负载均衡
```

## 专家并行
MoE（Mixture of Experts）架构的模型——如 DeepSeek-V3、Mixtral 8x7B、Qwen2.5-MoE——与传统 Dense 模型有根本区别：每个 Transformer 层的 FFN 被替换为多个并行的"专家"子网络，每个 token 只激活其中少数几个（通常 top-2）。参数量巨大（DeepSeek-V3 达 671B）但每次前向传播的计算量只对应激活参数（约 37B）。

专家并行（EP）将不同的 expert 分配到不同 GPU。每个 MoE 层的处理流程：Token 到达 → 门控网络（Router）计算每个 expert 的匹配分数 → 取 top-k expert（k=2） → 通过 All-to-All 通信将 token 分发到对应 expert 所在的 GPU → 各 GPU 计算自己负责的 expert → All-to-All 通信将结果传回 → 加权求和输出。

EP 的通信模式与 TP 有本质区别。TP 需要每层两次 AllReduce——所有 GPU 之间进行密集全量同步。EP 的 All-to-All 只发生在 token 路由时：每个 GPU 只发送/接收自己那部分 token 到激活的 expert 所在的 GPU。通信量与激活 expert 数成正比而非模型总层数——一个 token 在 60 层 MoE 模型中只触发 MoE 层（约每两层一次）的路由通信，且每次只与 $k$ 个 expert 通信。

这使 EP 对 PCIe 环境的耐受度远超 TP。本地无 NVLink 的多卡平台跑 MoE 模型时，EP 的通信开销可控——不会像 TP 那样 GPU 绝大部分时间在等待 PCIe 传输。Mixtral 8x7B 在 2 张 3090 PCIe 直连下使用 EP，推理速度接近单卡的两倍；而同样的硬件跑 Dense 70B 模型用 TP=2，吞吐量可能比单卡还低。

EP 与 TP/PP 可以混合使用。DeepSeek-V3 的训练和推理使用了 EP + TP + PP 的组合：节点内 NVLink 互联的 GPU 之间用 TP 处理 Attention 层和 Shared Expert（每个 token 必算），跨节点用 EP 分发 Routed Expert，PP 处理流水线层级切分。本地小规模部署通常只需要 EP 或 EP + 小规模 PP。

```bash
# vLLM: MoE 模型自动启用 EP
vllm serve deepseek-ai/DeepSeek-V3 --tensor-parallel-size 2

# 对于 MoE 模型，vLLM 会自动将 expert 分配到各 GPU
# 可以通过 --max-num-seqs 控制并发请求数来间接影响 EP 效率
```

MoE 模型的 CPU+GPU 混合推理同样遵循"Attention 优先放 GPU"原则——MoE 层中的 Attention 和 Shared Expert（如果存在）是每个 token 的必经之路，必须放 GPU。Routed Expert 可以部分卸载到 CPU——每个 token 只激活其中 2 个，PCIe 传输开销有限。

## 异构显存与非对称多卡
本地常见的硬件组合是一张 32G/48G 主卡配几张 16G 副卡——捡二手硬件时很难凑齐同规格。这种非对称组合能用，但用对方式和用错方式的体验天差地别。

"大卡+小卡"的核心优势是单卡独立运行能力。32G 卡可以单卡装下 32B INT4 模型完整运行——完全不需要跨卡通信，延迟最低、速度最快。同等条件下全 16G 卡必须至少两张做 PP 才能跑同规模模型。此外 PP 允许非对称层切分——60 层的模型，32G 卡分配 40 层，16G 卡分配 20 层，可以最大化总显存利用率。

但在张量并行（TP）中，木桶效应是致命问题。主流框架（vLLM、SGLang）的 TP 要求每张卡分配等量的权重切片——32G 卡被迫当 16G 用，富余的 16G 显存直接浪费。更糟的是大卡通常伴随更高的算力和带宽，AllReduce 同步时快卡必须等慢卡——32G 卡的计算单元频繁空转，整机吞吐被 16G 卡拉低。TP 还要求卡数是 2 的幂次（2、4、8），1 张 32G + 2 张 16G 的 3 卡组合很多框架无法开启 TP=3。

最佳实践不是让大卡和小卡挤在同一个并行组里，而是做任务隔离：

方案 A（推荐，省心）：32G 卡单独跑一个 vLLM/SGLang 实例服务主力 LLM；16G 卡单独跑另一个实例服务 Embedding、RAG 向量检索或轻量 Agent。各管各的，互不拖累。

方案 B（极客，llama.cpp 异构切分）：用 llama.cpp 的层映射功能，手动指定不同卡加载不同层数——32G 加载 35 层，16G 加载 15 层。前提是使用 PP 模式的层切分而非 TP，且接受单请求延迟偏高的代价。

## PP 与投机解码的框架兼容性
投机解码和流水线并行在概念上可以叠加——PP 解决显存问题，投机解码解决速度问题。但框架兼容性有差异：

vLLM 和 SGLang 目前不支持 PP 与投机解码同时开启。投机解码在这些框架中只能配合 TP 使用——所以如果你的多卡方案选择了 PP，就无法在同一实例中使用投机采样。

llama.cpp 原生支持两者叠加——模型按层切分到多卡/CPU 的同时可以挂载 draft model 做投机解码。需要为 draft model 预留 1-2GB 显存空间。

MTP（Multi-Token Prediction）是绕过这个限制的替代方案。Qwen 2.5 等模型内置了多 Token 预测头——模型自身就能一次预测多个 token，不需要额外的 draft model。主模型用 PP 拆分的同时，MTP 投机自动生效，既省显存又免去框架兼容性问题：

```bash
# llama.cpp: PP层切分 + draft model 投机（同时启用）
llama-cli -m qwen3.6-32b-Q4_K_M.gguf \
  -md qwen3.6-0.5b-Q4_K_M.gguf \  # draft model
  -ngl 20 --draft-max 16           # 主模型 GPU 层数 + 投机深度

# vLLM: TP + 投机（PP 不兼容投机解码）
vllm serve Qwen/Qwen3.6-32B --tp 2 \
  --speculative-model Qwen/Qwen3.6-0.5B
```

## 选型决策树
单卡能装下完整量化模型 → DP 多实例（零通信，最高吞吐）。单卡装不下、节点内有 NVLink → TP 优先（低延迟）+ MTP/投机解码。单卡装不下、节点内无 NVLink → PP 优先（低带宽容忍度）+ MTP（模型支持时）；如果模型不支持 MTP 则需要 llama.cpp 才能同时启用 PP + 投机解码。异构显存（大小卡混插）→ 任务隔离优先，不要塞进同一 TP 组。MoE 模型 → EP 优先（All-to-All 通信对 PCIe 友好），可叠加小规模 PP 拆分 attention+shared expert 与 routed expert。大规模集群（千卡以上） → DP + TP + PP 3D 并行——个人本地环境不需要。

EP 解决的是 MoE 模型的切分方式选择问题，MTP 解决的是 Decode 速度问题——两者解决不同维度，可以叠加使用。完整的并行策略与技术组合方案见[推理优化](optimization)。
