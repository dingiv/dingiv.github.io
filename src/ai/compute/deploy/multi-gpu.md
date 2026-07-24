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
NVLink 是 NVIDIA 的卡间直连技术，Ampere 代（3090、A6000 等）提供 112.5 GB/s 双向带宽。PCIe 4.0 x16 单向带宽仅 31.5 GB/s，约为 NVLink 的 1/4。TP 每层触发两次 AllReduce，NVLink 下通信耗时占总时间 < 5%，PCIe 下可能 > 50%。完整的 NVLink 兼容性参考见 [GPU AI 部署参考](gpu-ai#nvlink-桥接参考)。

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
对于 MoE（混合专家）模型（如 DeepSeek-V3、Mixtral），专家并行（EP）将不同的 expert 分配到不同 GPU。Token 通过门控网络路由到激活的 expert 所在 GPU——卡间通信只在路由时发生一次 All-to-All，频率远低于 TP 的逐层全同步。

EP 对 PCIe 环境友好——通信频率低、数据量可控。如果你的部署目标主要是 MoE 模型，EP 是比 TP/PP 更高效的切分方式。

## 选型决策树
单卡能装下完整量化模型 → DP 多实例（零通信，最高吞吐）。单卡装不下、节点内有 NVLink → TP 优先（低延迟）。单卡装不下、节点内无 NVLink → PP 优先（低带宽容忍度）。MoE 模型 → EP + 小规模 PP 组合。大规模集群（千卡以上） → DP + TP + PP 3D 并行——个人本地环境不需要。

多 GPU 推理中投机解码可以和任何并行策略叠加——它解决的是 Decode 速度而非显存问题，详见[推理优化](optimization)。
