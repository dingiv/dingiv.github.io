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

Linux 路由查找基于 FIB（Forwarding Information Base），采用最长前缀匹配算法。每个网络命名空间维护一个独立的 FIB，包含该空间内所有已知的路由条目。一条路由条目最少包含三个字段：目的前缀（如 `192.168.1.0/24`）、下一跳网关（或出接口）、以及度量值（metric）。当 IP 层的 `ip_route_output()` 或 `ip_route_input()` 被调用时，FIB 按特定的查找规则找到最优路径。

### 最长前缀匹配

最长前缀匹配（LPM, Longest Prefix Match）是 FIB 查找的核心算法。当多个路由表项与目标 IP 部分匹配时，选择匹配位数最多的那条。例如目标 `192.168.1.100` 匹配 `192.168.0.0/16` 和 `192.168.1.0/24`——/24 有 24 位匹配、/16 只有 16 位匹配，最终选择 /24 这条。

内核使用压缩前缀树（LC-trie, Level-Compressed Trie）存储 FIB 条目。相比传统二叉树，LC-trie 通过将单分支链压缩为一级节点，在查找路径深度和内存占用之间取得平衡。每跳步可以跳过多个比特位——对于稀疏分布的 IPv4 路由表（大多数前缀长度集中在 /16 ~ /24），LC-trie 的平均查找深度远小于 32。

大规模路由表的内存和查找效率是关键指标。互联网骨干路由器的 BGP 全表约 100 万条路由，在 Linux 上使用 LC-trie 的查找延迟通常在 100-300ns 量级，内存占用约 50-80 MB。相比硬件 TCAM 的 O(1) 查找，软件 LPM 的延迟在普通服务器场景下完全可接受——实际瓶颈通常在 netfilter 和 TCP 栈处理，而非路由查找本身。

### 策略路由与 RPDB

RPDB（Routing Policy Database）将单表路由扩展为多表+规则的架构。传统的"查一张路由表"在策略路由下变为"先匹配规则，再根据规则查指定表"。

```bash
# 查看策略路由规则
ip rule list
# 0:      from all lookup local      # local 表（最高优先级，本机地址）
# 32766:  from all lookup main       # main 表（默认表）
# 32767:  from all lookup default    # default 表（默认路由）
```

规则按优先级数字从小到大排序匹配。每条规则定义了选择器（`from` 源地址、`fwmark` 防火墙标记、`iif` 入接口、`tos` 服务类型）和动作（`lookup <table>` 查指定路由表、`blackhole/unreachable/prohibit` 丢弃）。源地址策略路由是最常见的应用——来自子网 A 的流量走 ISP 1，来自子网 B 的流量走 ISP 2：

```bash
# 创建两个自定义路由表
echo "100 isp1" >> /etc/iproute2/rt_tables
echo "200 isp2" >> /etc/iproute2/rt_tables

# 给各表添加默认路由
ip route add default via 10.0.1.1 table isp1
ip route add default via 10.0.2.1 table isp2

# 规则：不同源 IP 查不同的表
ip rule add from 192.168.1.0/24 table isp1
ip rule add from 192.168.2.0/24 table isp2
```

每个网络命名空间维护四张初始路由表：`local`（本机地址和广播地址，优先级最高，内核自动填充）、`main`（`ip route` 不加 `table` 参数时的默认表，ID 254）、`default`（默认路由专用，优先级最低，ID 253）、以及 `unspec`（空表，ID 0，查找失败时的兜底）。自定义表 ID 从 1 到 252，名称在 `/etc/iproute2/rt_tables` 中定义。

防火墙标记（fwmark）是策略路由中更灵活的选择器。它的价值在于可以将路由决策与流量特征关联起来——不是"来自哪个 IP"这种静态属性，而是"这个包经过了哪些 netfilter 链处理后匹配了什么规则"的动态属性。

```bash
# 将所有 TCP 443 (HTTPS) 流量标记为 10，走 VPN 路由表
iptables -t mangle -A PREROUTING -p tcp --dport 443 -j MARK --set-mark 10
ip rule add fwmark 10 table vpn

# nftables 等效
nft add rule ip mangle PREROUTING tcp dport 443 counter meta mark set 10
```

Mangle 表是 fwmark 的主要工作场所。Packet mark 是一个 32 位无符号整数，存储在 `skb->mark` 字段中，贯穿整个协议栈处理周期——从 PREROUTING 到 POSTROUTING，任何 hook 点都能读取或修改这个值。标记只在本地主机有效，不会出现在线路上。

### VRF：基于 L3 Master Device 的路由隔离

VRF（Virtual Routing and Forwarding）在 Linux 中以 L3 master device 的形式实现——创建一个虚拟设备作为多个物理接口的 master，附加到该设备的接口共享独立的路由表和网络栈决策。与网络命名空间不同，VRF 共享同一套 TCP/IP 栈（socket、连接跟踪、netfilter），只隔离路由查找。

```bash
# 创建两个 VRF，分别对应不同客户或业务
ip link add vrf-blue type vrf table 100
ip link add vrf-red type vrf table 200
ip link set eth1 master vrf-blue
ip link set eth2 master vrf-red
ip link set vrf-blue up
ip link set vrf-red up
# eth1 和 eth2 的路由查找现在各自在 table 100/200 中独立进行
```

VRF 的关键特性是"本进程绑定"。一个 socket 创建时通过 `IP_UNICAST_IF` 或 `SO_BINDTODEVICE` 绑定到特定 VRF 设备后，该 socket 的所有路由查找都在 VRF 的主表中进行。未绑定的进程走全局路由表——这使得一台主机上可以同时运行多个使用重叠 IP 地址空间的独立应用，只要它们分别绑定到不同的 VRF。VRF 在 ISP 多租户 CPE 设备中广泛应用——同一台路由器为多个客户提供 WAN 接入，每个客户的路由表完全隔离。

### 双 ISP 策略路由实战

一台服务器接入两条 ISP 线路是最常见的策略路由场景。假设 eth0 连接 ISP-A（网关 10.0.1.1），eth1 连接 ISP-B（网关 10.0.2.1），目标是"正常流量走 ISP-A，特定业务走 ISP-B，ISP-A 断网时自动切到 ISP-B"。

```bash
# 1. 创建两张独立路由表
echo "100 isp_a" >> /etc/iproute2/rt_tables
echo "200 isp_b" >> /etc/iproute2/rt_tables

# 2. 各自配置默认路由
ip route add default via 10.0.1.1 table isp_a
ip route add default via 10.0.2.1 table isp_b

# 3. 规则：主 IP 走对应线路，其余走 ISP-A
ip rule add from 10.0.1.10 table isp_a priority 100
ip rule add from 10.0.2.10 table isp_b priority 100
ip rule add from all table isp_a priority 200   # 兜底

# 4. 通过 fwmark 将特定端口流量导向 ISP-B
iptables -t mangle -A OUTPUT -p tcp --dport 8080 -j MARK --set-mark 200
ip rule add fwmark 200 table isp_b priority 50
```

"断网自动切换"依赖路由健康检测。简单的做法是用 `ip route add ... nexthop` 配置多个下一跳——内核检测到 eth0 链路断开（link down）时自动跳过该路径。更完善的方案是通过 BFD（Bidirectional Forwarding Detection）或应用层健康检查守护进程（定期 ping ISP 网关），检测到故障时动态修改路由表的默认路由指向备用下一跳。

### 路由查找调试

`ip route get` 是策略路由调试的核心工具——给定一个数据包的参数（源 IP、目的 IP、入接口、fwmark），显示内核会选择哪条路由。

```bash
# 查从 10.0.1.10 到 8.8.8.8 会走哪条路由
ip route get 8.8.8.8 from 10.0.1.10 iif eth0
# 输出示例：8.8.8.8 from 10.0.1.10 via 10.0.1.1 dev eth0 table isp_a

# 指定 fwmark 查路由
ip route get 8.8.8.8 mark 200
# 输出示例：8.8.8.8 via 10.0.2.1 dev eth1 table isp_b mark 0x200

# 查一个包经过的完整路径（包括策略规则匹配）
ip route get 8.8.8.8 from 10.0.2.10 iif eth1
# 如果返回 "RTNETLINK answers: Network is unreachable" 且 ip rule 中存在
# 对应 from 的规则指向了错误的 table——该 table 中没有到达目的地的路由
```

常见排错场景：策略路由"不生效"时，检查 `ip rule list` 的优先级排列是否在 `lookup main` 之前、`ip route show table <table>` 中是否有匹配的路由条目、以及 netfilter 的 conntrack 是否缓存了旧的路由决策。conntrack 是策略路由最隐蔽的陷阱——连接的第一个数据包触发路由查找并缓存结果，后续同连接的其他数据包直接复用缓存，即使策略路由规则或路由表发生了变化也不再生效。需要清除 conntrack 条目或重启连接才能让新规则生效。

### 路由缓存与 ECMP

Linux 3.6 之前的版本有全局的路由缓存（routing cache），但 2012 年移除了——路由缓存与连接跟踪表的缓存驱逐竞争导致性能退化，且在多租户环境中容易受缓存投毒攻击。现代 Linux 使用 per-flow 的下一跳缓存——每个 `dst_entry` 对象直接关联到对应的 `fib_nh`（下一跳），skb 的路由结果缓存在 socket 级别的 `sk->sk_dst_cache` 中，避免相同流重复执行完整的 FIB 查找。这在长连接场景（如数据库连接池、gRPC 流）中收效显著。

ECMP（Equal-Cost Multi-Path）在存在多条度量值相等的路由时，将流量均匀（或按权重）分配到各路径。内核通过 `fib_multipath_hash()` 对数据包的五元组（源 IP、目的 IP、源端口、目的端口、协议）做哈希，相同流的数据包始终走同一路径——保证 TCP 连接不出现乱序。哈希算法默认使用 L4（含端口），可通过 `sysctl net.ipv4.fib_multipath_hash_policy` 切换到 L3（仅 IP，适用于 GRE 隧道等端口不重要的场景）或定制哈希策略。

### FIB 查找路径

一个入站数据包在 IP 层的完整路由查找路径为：`ip_rcv()` → netfilter PREROUTING → `ip_rcv_finish()` → `ip_route_input()` 查 FIB。如果目的 IP 是本机地址（`RTN_LOCAL`），skb 以 `dst.input = ip_local_deliver` 进入 `ip_local_deliver()` 分发给上层 TCP/UDP。如果目的 IP 是外部地址（`RTN_UNICAST`），skb 以 `dst.output = ip_forward` 进入转发路径——netfilter FORWARD → `ip_output()` → netfilter POSTROUTING → 发送。

发送方向的路径对称：应用层 `sendmsg()` → UDP/TCP 构造 skb → `ip_route_output_flow()` 查 FIB → 如果本机直连则 `dst.output = ip_output` 直接发送，如果下一跳非直连则触发邻居子系统（ARP/NDP）查找下一跳 MAC 地址。

## 邻居子系统
ARP 属于邻居子系统（neigh）。ARP 表并非简单的缓存，而是一个完整的状态机：INCOMPLETE → REACHABLE → STALE → DELAY → PROBE。理解这个状态机才能理解 ARP 抖动和丢包现象——当一条 ARP 条目进入 STALE 状态后，如果有流量需要发送，内核不会立即发起 ARP 请求，而是先进入 DELAY 状态等待一段时间，如果在此期间收到了对方的 Gratuitous ARP 或单播确认，状态回到 REACHABLE。只有 DELAY 超时后仍未确认，才进入 PROBE 状态发起 ARP 探测。
