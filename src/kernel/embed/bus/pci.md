# PCI
PCI 总线协议是一种广泛用于现代机器上的高速硬件协议。

二、PCIe 通道（Lane）

每个 PCIe “lane” 是一对发送线和一对接收线组成的全双工链路。
一个设备可以用多个通道聚合成更高带宽的连接，比如：

x1：1 条通道（最小）

x4：4 条通道（常见于网卡、NVMe SSD）

x8：8 条通道（部分存储或网络加速卡）

x16：16 条通道（标准显卡插槽）

每一代 PCIe 都提高带宽：

版本	每通道单向速率	x16 理论带宽（单向）
PCIe 3.0	1 GB/s	16 GB/s
PCIe 4.0	2 GB/s	32 GB/s
PCIe 5.0	4 GB/s	64 GB/s
PCIe 6.0	8 GB/s（PAM4 信号）	128 GB/s


CPU 直接提供部分 PCIe 通道（直连显卡、NVMe 等高性能设备）。

主板芯片组（PCH） 提供额外通道，连接低速外设（USB 控制器、网卡、SATA 控制器等）。

所有通道都通过 PCIe Switch 或 Root Complex 管理。


CXL（Compute Express Link）

建立在 PCIe 物理层之上，但协议不同。

支持 CPU 与 GPU/加速器/内存之间的缓存一致性（cache coherence）。

Intel、AMD、ARM、NVIDIA、Google 都在推。

它让多设备共享内存空间，从而减少 PCIe 复制数据的延迟。

NVLink / NVSwitch（NVIDIA）

专为 GPU 间通信设计。

比 PCIe 快几个数量级：NVLink 4.0 单 GPU 总带宽可达 900GB/s。

用于大型 GPU 集群，比如 DGX、HGX 系列。

Infinity Fabric（AMD）

连接 CPU、GPU、I/O 节点的高速互联。

在 EPYC、Instinct 系列中承担与 NVLink 类似的角色。

UCIe（Universal Chiplet Interconnect Express）

新标准，目标是芯粒级（chiplet）互联。

AMD、Intel、TSMC、Samsung 都在支持，未来趋势是 SoC 内部模块化互联取代外部 PCIe。


主要因为现代负载的数据需求爆炸性增长：

AI 模型训练 / 推理
GPU 不仅需要算力，还需要巨量参数与样本在 CPU ↔ GPU 之间流动。
例如训练 GPT-4 级模型时，单轮梯度同步可能涉及数百 GB 数据。PCIe 就成了喉咙。
所以 NVIDIA 改用 NVLink / NVSwitch，让 GPU 间通信达到 TB/s 级。

高性能存储与缓存系统
NVMe SSD 读写性能已接近 PCIe x4 的极限。多个 SSD 同时工作时，PCIe Root Complex 会饱和。

多设备并行（多 GPU / 多 FPGA）
PCIe 的拓扑通常是树形结构（CPU 根节点 → 多个下行设备），通信要“绕远路”，带宽被分摊，延迟增加。

内存分离架构（Memory Pooling）
当服务器尝试把内存资源集中管理（通过 CXL 等技术共享），传统 PCIe 无法提供足够低延迟和高一致性支持。

三、带宽不是唯一问题 —— 延迟也致命

PCIe 是包传输结构（packetized protocol），每次通信有握手、封装、路由，延迟比内存总线高几个数量级。
对于像 GPU 同步、远程 DMA、共享内存这类需要“纳秒级响应”的场景，这种延迟直接拖垮效率。