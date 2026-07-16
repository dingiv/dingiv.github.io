# 协议栈
Linux 网络协议栈的核心载体是 `struct sk_buff`（skb），所有网络数据在内核中都封装为 skb 进行传递。skb 是一个高度优化的包描述符，头部指针分层推进（`mac_header`、`network_header`、`transport_header`），支持 clone（零拷贝）、scatter-gather 和 GSO/GRO。理解 skb 的生命周期是理解 Linux 网络性能的关键——数据路径本质上是 skb 在不同子系统之间移动。

## NAPI 与中断缓解
在高速网络（100G 及以上）场景下，每个数据包触发一次硬件中断的传统模型会导致中断风暴——CPU 将所有时间花在处理中断上，没有余力处理实际的数据。NAPI（New API）通过"中断 + 轮询"的混合机制解决了这个问题：当数据包到达时，网卡触发一次硬中断，中断处理程序将对应的 NAPI 结构体加入软中断轮询队列，然后关闭该队列的硬中断；内核在软中断上下文中通过 `net_rx_action()` 轮询收包，一次处理多个数据包（配额默认 300 或按 `netdev_budget` 配置），处理完后重新开启硬中断。

这种机制在高负载下自动退化为纯轮询模式——硬中断被持续关闭，软中断不断被触发，CPU 全力收包。NAPI 的关键调整参数包括 `netdev_budget`（每次轮询的最大收包数）和 `netdev_budget_usecs`（每次轮询的最大时间预算），在延迟和吞吐之间取得平衡。

## GRO/GSO/TSO 的分段与零拷贝
这三种技术围绕着一个共同目标：减少协议栈处理的包数量，从而降低 CPU 开销。

TSO（TCP Segmentation Offload）将 TCP 分段工作从内核协议栈卸载到网卡硬件。应用层发送一个 64KB 的大 skb，内核协议栈只处理一次（计算一次校验和、走一次 netfilter），网卡硬件负责将其切分为 MTU 大小的小包发出。这大幅降低了 CPU 的处理次数。GSO（Generic Segmentation Offload）是 TSO 的软件版本——在网卡不支持 TSO 时，内核将分段延迟到即将交给驱动之前的最后一刻执行，在此之前协议栈始终以大数据块为单位处理，享受与 TSO 相同的协议栈减负收益。

GRO（Generic Receive Offload）是接收方向的对应优化。网卡硬件或驱动在 NAPI 收包时将属于同一条流的多个小包合并为一个大的 skb 再提交给协议栈，协议栈处理一次而非 N 次。GRO 的合并发生在数据包穿越协议栈之前，因此对 TCP/IP 层的 CPU 消耗有显著的降低效果。对于转发场景（路由器/网桥），合并后的包在转发时再通过 GSO 切分发送，形成 GRO→转发→GSO 的高效流水线。

## XDP 与 eBPF 重塑网络路径
XDP（eXpress Data Path）是 Linux 内核中最早可以介入数据包处理的挂钩点——它在网卡驱动收包后、skb 分配之前执行。XDP 程序通过 eBPF 虚拟机运行在内核态，但受限于严格的安全校验（类型安全、无界循环检测、内存边界检查），可以安全地在生产环境加载。

XDP 的核心优势在于"早"。传统数据路径要经过 skb 分配、netfilter、路由查找等环节，而 XDP 在驱动层直接操作原始数据帧。一个 XDP_DROP 判决可以在数据包进入协议栈之前将其丢弃，性能远超在 netfilter 中通过 iptables 规则丢弃——后者的数据包已经经历了完整的 skb 分配和协议栈遍历。在 DDoS 防御场景中，XDP 可以在占用极少 CPU 的情况下丢弃数百万 pps 的攻击流量。

XDP 支持三种操作模式：返回 XDP_PASS 将包交给内核协议栈正常处理（相当于放行），返回 XDP_DROP 直接丢弃（防火墙黑名单、DDoS 防御），返回 XDP_TX 或 XDP_REDIRECT 将包从同一网卡发出或重定向到其他网卡（负载均衡、转发加速）。结合 eBPF maps，XDP 程序可以在内核中维护有状态的数据结构（如连接跟踪表、黑名单集合），实现高性能的包处理逻辑，无需将数据拷贝到用户空间。

## 多队列网卡与 CPU 亲和
现代高速网卡（10G 及以上）通常配备多个硬件收发队列（Multi-Queue），每个队列有独立的中断线和 DMA 通道，可以绑定到不同的 CPU 核心。如果不做亲和性配置，来自不同队列的中断可能随机落在任意 CPU 上，导致缓存局部性极差——一个 TCP 连接的数据包被不同 CPU 处理，每次都需要跨 CPU 同步连接状态。

RSS（Receive Side Scaling）在硬件层面解决这个问题。网卡对数据包的源 IP、目的 IP、源端口、目的端口做哈希计算，根据哈希值将包分发到不同的接收队列。同一个五元组的数据包始终进入同一个队列，由同一个 CPU 处理，保证连接级别的缓存局部性。

RPS（Receive Packet Steering）是 RSS 的软件模拟。当网卡不支持 RSS 或队列数少于 CPU 核心数时，RPS 在软中断处理中将收包重新分发到目标 CPU 的 backlog 队列。RPS 的哈希计算在内核中完成，可以根据负载动态调整分发策略。RFS（Receive Flow Steering）在 RPS 的基础上进一步考虑应用层——如果某个 socket 的消费者线程运行在 CPU 3 上，RFS 会尽量将属于该 socket 的数据包导向 CPU 3，避免跨核通信的开销。

XPS（Transmit Packet Steering）是发送方向的对称机制，将发送队列与 CPU 亲和绑定，确保同一连接的发送处理也在同一个 CPU 上完成。

## TCP 栈收包路径
从驱动到用户态 socket，一个 TCP 数据包的接收路径如下：

1. 网卡 DMA 将数据帧写入内核分配的接收缓冲区，发起硬中断
2. 中断处理程序将 NAPI 加入软中断队列，关闭该队列的硬中断
3. 软中断 `net_rx_action()` 调用驱动的 NAPI poll 函数收包，分配 skb 并填充协议头指针
4. skb 经过 GRO 合并后提交到协议栈入口 `netif_receive_skb()`
5. 根据 skb->protocol（ETH_P_IP 等）分发到 IP 层 `ip_rcv()`
6. IP 层完成校验和验证、netfilter PREROUTING 钩子处理、路由查找后，调用 `ip_local_deliver()` 分发到上层协议
7. TCP 入口 `tcp_v4_rcv()` 根据四元组查找对应的 socket，将 skb 放入 socket 的接收队列
8. 唤醒阻塞在 `recv()`/`read()` 系统调用上的用户进程，将数据拷贝到用户空间

这个路径中的核心变量是 socket 的接收缓冲区大小。当用户进程读取速度不及网卡收包速度时，skb 在接收队列中堆积，最终触发丢包——这不是网络丢包，而是内核丢弃已经收到的数据。调优接收缓冲区大小（`tcp_rmem`）和应用程序的读取策略（使用较大的 `recv` 缓冲区、epoll 边缘触发模式），是提高 TCP 收包吞吐量的关键。

## 防火墙
netfilter 框架在五个关键 hook 点插入逻辑：PREROUTING、INPUT、FORWARD、OUTPUT、POSTROUTING。iptables 是历史接口，现代系统推荐使用 nftables。NAT 本质是连接跟踪（conntrack）驱动的地址重写——没有 conntrack 就没有状态 NAT，性能瓶颈往往出现在 conntrack 表的容量和查找效率上。

## 路由子系统
Linux 路由查找基于 FIB（Forwarding Information Base），采用最长前缀匹配算法，支持多表和策略路由（RPDB）。

## 邻居子系统
ARP 属于邻居子系统（neigh）。ARP 表并非简单的缓存，而是一个完整的状态机：INCOMPLETE → REACHABLE → STALE → DELAY → PROBE。理解这个状态机才能理解 ARP 抖动和丢包现象——当一条 ARP 条目进入 STALE 状态后，如果有流量需要发送，内核不会立即发起 ARP 请求，而是先进入 DELAY 状态等待一段时间，如果在此期间收到了对方的 Gratuitous ARP 或单播确认，状态回到 REACHABLE。只有 DELAY 超时后仍未确认，才进入 PROBE 状态发起 ARP 探测。
