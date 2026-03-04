---
title: 网络实现
order: 60
---

# 网络实现
总结来说，Linux 的网络架构分成了如下层次：
1. 应用层
2. 系统调用 socket API，更多内容参考[系统调用](../syscall/)
3. 插口层/套接字层，使用请参考[套接字编程](../develop/socket)
4. 协议层/协议栈，实现[网络协议](/basic/network/protocol)各个层
5. 驱动层，更多内容参考[设备管理](../device/)
6. 物理层

## 网络接口层
在 Linux 中，通过 `net_device` 结构体抽象一个**网络接口**，或者叫**网卡**。这是一个极具误导性的命名，它不是严格的网络设备，或者说远没有看上去那么简单。该定义是一个大杂烩，它直接包揽了二层和三层的所有内容，从数据结构上不再区分二层和三层，而网络的协议和数据收发是通过协议栈来进行，网络设备不关心数据操作，只呈现属性。这一层是 Linux 网络“可塑性”的来源。

```c
struct net_device {
    const struct net_device_ops *netdev_ops;
    const struct header_ops *header_ops;
    struct netdev_queue *_tx;
    netdev_features_t gso_partial_features;
    unsigned int real_num_tx_queues;
    unsigned int gso_max_size;
    unsigned int gso_ipv4_max_size;
    u16 gso_max_segs;
    s16 num_tc;
    // ... more properties
}
```

它向上提供发送接口（ndo_start_xmit），向下绑定驱动，内部管理 TX/RX 队列，维护统计信息、MTU、特性位；

从数据路径角度看：协议栈通过 dev_queue_xmit() 将 skb 推入 net_device，驱动通过 NAPI poll 将 skb 提交给协议栈。

net_device 本质上是“包交换的接口契约”。它不处理协议逻辑，不解析 IP，不关心 TCP 状态机。它只是流量的出入口。

### eth
真实物理以太接口。

### TAP 虚拟以太网卡
TAP 网卡是一个 Linux 内部实现的**用户空间**虚拟以太网接口，同样使用一个 net_device 结构体来表示。尽管真实的物理线路上并没有物理网络接口，但是操作系统可以假装有一个，并将写入这个接口中的数据，发送到其他地方去，如应用层的应用软件，而不是调用真实物理网卡的驱动。这是一个非常强大的设计，这意味着，Linux 的用户可以定义用户空间的"物理"传输链路，这也为虚拟机的网络技术提供了一个简单而可靠的方法。

由于网络的分层架构，二层的以太网络接口其实和一层的物理链路之间不一定是强耦合的，所以下层的物理链路可以是基于普通的**双绞线**网线，也可以基于**无线信号**、**光信号**。并且，在 Linux 中，由于可以给用户创建一个二层的网络设备 tap，一个 tap 意味着一个虚拟的网络接口，其数据的传输走的是软件处理，因此数据也可以走虚拟通路进行传输。

TAP 网卡拥有前后端：
- 前端是协议栈
- 后端是用户空间的自定义的数据传输软件

我们一般只需要实现 TAP 网卡的后端。当我们创建了一个 TAP 网卡 `tap0` 之后，在 `/dev/net/tap0` 路径将会出现我们创建的网卡在 Linux VFS 上的文件抽象。

- 当我们**读**该文件的时候，相当于模拟**网卡处理从协议栈发出的数据，并向外界发送数据**
- 当我们**写**该文件的时候，相当于模拟**网卡接收到从外界发送来的数据，并通过中断通知上层协议栈进行处理**

更进一步的讲，我们需要向 TAP 设备读写和处理**以太帧**。

### TUN 虚拟三层网卡
TUN 设备与 TAP 设备类似，但它在三层（IP 层）操作，处理的是 IP 数据包而不是以太帧。TUN 设备常用于 VPN 隧道（如 OpenVPN）和 IP 隧道协议（如 IPIP、GRE）。TUN 设备的读写操作处理的是 IP 数据包，没有以太网头部。

TUN/TAP 设备的存在使得用户空间程序可以完全参与协议栈的数据流动，这是实现虚拟机网络（如 QEMU/KVM）、容器网络（如 Docker、Kubernetes）、网络功能虚拟化（NFV）的基础。TUN/TAP 的设计体现了 Linux 网络的哲学：协议栈是可插拔的。

### Bridge 网桥-虚拟交换机
网桥是一个虚拟二层网卡，也可以将其看做操作系统内部实现的虚拟交换机，使用软件模拟的交换机，性能较差（交换机转发流量的行为逻辑一般是由硬件直接实现的，因此性能极高），它让通用计算机体现出与交换机类似的行为，是容器网络和虚拟机网络的基础。虽然软件转发性能不如 ASIC 交换机，但借助 XDP、硬件 offload，可以将部分转发逻辑下沉到网卡。

网桥根据 MAC 地址转发以太帧，维护一个 MAC 地址到端口映射的转发表（FDB，Forwarding Database）。网桥可以连接多个网络接口（物理或虚拟），使这些接口处于同一个二层广播域中。网桥的实现涉及内核的 netfilter 框架，通过在数据包的路径上注册钩子函数来实现转发逻辑。

### VLAN-Virtual Local Area Network 虚拟局域网
VLAN 允许在同一个物理网络上创建多个逻辑隔离的网络。它的主要作用是：
- 隔离广播域，提高安全性
- 灵活管理网络结构
- 将一个物理网络划分为多个独立的逻辑网络
- 允许一个网卡连接多个网桥（通过 VLAN 子接口）
- 减少广播流量，提升网络性能

VLAN 在 Linux 中通过 802.1Q 实现。

VLAN 通过 VLAN Tagging 允许一个物理网卡创建多个 VLAN 子接口，每个 VLAN 接口可以加入不同的网桥，从而让一个以太网接口（eth0）能够连接多个网桥。一个物理接口可以派生出多个子接口（eth0.100）。VLAN 让广播域逻辑独立于物理拓扑。这是一种常见的网络管理方式，在虚拟化和云计算环境中非常常见。

### Bond
Bond 是**网络绑定（NIC Bonding）**创建的一个逻辑网络接口，用于将多个物理网卡（NIC）绑定在一起，提高网络的带宽、冗余性和可靠性。它是 Linux 网卡聚合（Link Aggregation）的实现，类似于 LACP（IEEE 802.3ad）。

Bond 支持多种工作模式：mode-0（balance-rr）轮询策略；mode-1（active-backup）主备策略；mode-4（802.3ad）LACP 聚合；mode-5（balance-tlb）自适应传输负载均衡；mode-6（balance-alb）自适应负载均衡。

### Macvlan
Macvlan 是另一种虚拟网络接口技术，它允许在一个物理接口上创建多个虚拟接口，每个虚拟接口有自己独立的 MAC 地址。Macvlan 适用于容器网络场景，每个容器可以拥有自己的虚拟网卡并直接连接到物理网络。但是，Macvlan 需要一个主设备，主设备

Macvlan 有几种工作模式：VEPA 模式让流量发往物理交换机并由交换机转发回同一主机上的其他 macvlan 接口；Bridge 模式允许同一物理接口上的 macvlan 接口直接通信；Private 模式隔离同一物理接口上的 macvlan 接口；Passthru 模式允许一个物理接口传递所有 MAC 地址。缺点是：主机和子接口默认不能互通。

## 协议栈
Linux 网络协议栈是实现网络层协议的组件，它基于下层的设备抽象层，同时为上层的 socket 层提供协议的抽象机制。协议栈采用分层设计，每层协议通过注册回调函数的方式参与数据包的处理流程。

所有网络数据在内核中都封装为 struct sk_buff（skb）。

skb 是一个高度优化的包描述符：

头部指针分层推进（mac_header / network_header / transport_header）

支持 clone（零拷贝）

支持 scatter-gather

支持 GSO/GRO

理解 skb 的生命周期，是理解 Linux 网络性能的关键。

数据路径本质上是 skb 在不同子系统之间移动。

### Routing 表-路由表
路由表决定数据包的转发路径，存储网络目标和下一跳信息。Linux 支持多张路由表，通过路由策略（RPDB）决定使用哪张路由表。路由查找基于最长前缀匹配原则，路由表可以配置静态路由，也可以通过动态路由协议（如 OSPF、BGP）学习路由。

### ARP 表
ARP (Address Resolution Protocol) 表将目标主机的 IP 地址映射为目标主机的 MAC 地址，是实现从三层切换到二层的机制。当主机需要发送数据包到同网段的另一个主机时，它首先查询 ARP 表，如果未找到对应的 MAC 地址则广播 ARP 请求。ARP 表项有超时时间，通常为几分钟。

### FDB 表
FDB (Forwarding Database Table) 表将目标主机的 MAC 地址映射为交换机 port 号的表，是类似于交换机的 MAC 表。对于 Linux 网桥来说，FDB 表维护了 MAC 地址到网桥端口的映射关系，用于决定如何转发以太帧。FDB 表可以通过 `bridge fdb` 命令手动管理。

### Netfilter 框架 - iptables
Netfilter 是一套 Linux 上的网络过滤框架，它在网络包收发的各个过程中添加钩子，从而拦截和过滤流量，为了网络安全而设计。在这个框架中，使用一个名叫链（Chain）的概念来表达一个钩子，"链"就像是数据包在内核中旅行时要经过的一个个"检查站"，每个检查站（链）都有一堆规则，这些规则决定数据包是放行（ACCEPT）、丢弃（DROP）、修改（NAT）还是其他操作，每个点对应数据包生命周期的不同阶段。

该框架包含多张表，定义了为数据收发的不同阶段添加钩子，满足多样的网络过滤规则。包括：filter、NAT、mangle、raw、security 等，其中 filter 和 NAT 最常用。

#### Filter 表-防火墙
该表直接决定了是否处理一个数据包，是防火墙的底层基础，包含三个钩子（链、Chain）：
- INPUT：在三层数据包的目标是本机时触发
- FORWARD：在三层数据包的经过本机进行转发时触发
- OUTPUT：在本机发出三层数据的时候触发

钩子可以指定一些规则，用于匹配数据包的特征，例如可以匹配数据包的源地址、目标地址、协议类型等。当规则被匹配的时候，进行相应的行为：
- ACCEPT：接收该包
- DROP：丢弃该包
- REJECT：明确拒绝该包

防火墙规则的匹配是按顺序进行的，一旦匹配成功就执行相应的动作并停止后续规则的检查。

#### NAT 表-网关
网关是指当一个 Linux 机器位于多个子网中时，那么它可以作为子网之间交互的通道，这个操作就是 NAT 操作。Linux 一般默认支持 NAT 的功能，但是默认关闭，必须为其配置转发规则（通过 `sysctl -w net.ipv4.ip_forward=1`），才能使得 NAT 开始工作，然后子网中的其他机器可以通过配置路由表项，告诉访问某个网段的时候使用该机器作为网关。

NAT 表直接决定了本机的 NAT 行为是否进行，是网关功能的底层实现，包含三个钩子：
- PREROUTING：数据包进入系统后的第一个处理点，用于修改目标地址（DNAT）
- POSTROUTING：数据包离开系统前的最后一个处理点，用于修改源地址（SNAT 或 MASQUERADE）
- OUTPUT：本地主机发出的数据包，用于修改目标地址（DNAT）

回调：
- DNAT：目标地址修改
- SNAT：源地址修改
- MASQUERADE：掩饰，动态修改数据包的源地址，增强版 SNAT
- REDIRECT：将数据包的目标地址重定向到本地主机的某个端口
- RETURN：自定义链
- ACCEPT
- DROP
- NETMAP
- TPROXY

### Traffic Control-QoS (Quality of Service)

流量控制模块，是 Linux 中与协议栈平行的一个网络控制子系统，用来管理网络流量的带宽、延迟和优先级。QoS 的目标是优化网络性能，确保关键流量优先传输，避免拥堵。

Linux 支持多种队列规则：pfifo_fast 是默认的队列规则；tbf（Token Bucket Filter）令牌桶过滤器，可以限制带宽；htb（Hierarchy Token Bucket）分层令牌桶，支持复杂的层次化带宽分配；fq_codel（Fair Queuing with Controlled Delay）结合了公平队列和主动队列管理，可以减少延迟和队列溢出。

## 套接字
传输层面向进程，向进程提供了编程接口 Socket，并且在各大操作系统中均被采用，拥有较为统一的心智模型。其中，TCP 和 UDP 是协议栈的客户端，其 socket 内部会调用协议栈完成基于网络的交互。上层的接口采用 socket 接口进行抽象和定义，与下层无关，并且 socket 接口完全可以选择不走下层的协议栈，例如：unix socket 和 virtual socket。

Socket 在内核中由 `socket` 结构体表示，包含协议族、类型、协议、状态、等待队列、发送和接收缓冲区等字段。Socket 的数据传输使用 `sk_buff` 结构体（简称 skb）来封装数据包，skb 是一个双向链表节点，可以串联成队列，它包含数据包的各个层级的头部信息、数据指针、长度信息等。

## TCP

TCP Socket 提供面向连接的、可靠的、字节流传输服务。TCP Socket 的状态机包括 CLOSED、LISTEN、SYN_SENT、SYN_RCVD、ESTABLISHED、FIN_WAIT_1、FIN_WAIT_2、CLOSE_WAIT、CLOSING、LAST_ACK、TIME_WAIT 等状态。

应用程序通过 `listen()` 系统调用将 Socket 进入 LISTEN 状态，等待客户端的连接请求；通过 `connect()` 系统调用主动发起连接；通过 `accept()` 系统调用接受连接请求，返回一个新的 Socket 用于与客户端通信。

Linux 为 TCP Socket 提供了丰富的配置选项，可以通过 `setsockopt()` 系统调用或 `/proc/sys/net/ipv4/` 下的 sysctl 参数进行调优。重要的参数包括 TCP_KEEPALIVE_TIME、TCP_MAX_SYN_BACKLOG、tcp_slow_start_after_idle 等。

## UDP

UDP Socket 提供无连接的、不可靠的数据报传输服务。UDP Socket 不需要建立连接，每个数据报独立发送，可能沿不同的路径到达，可能乱序、重复或丢失。UDP 的简单性使得它的开销很小，适合对延迟敏感但可以容忍数据丢失的应用。

UDP Socket 的发送和接收相对简单：发送时，数据从用户空间复制到内核，添加 UDP 头部和 IP 头部后直接发送；接收时，数据到达后经过协议栈处理，最终复制到接收缓冲区。应用程序需要自行实现可靠性机制（如重传、确认、顺序控制）。

## Unix

Unix Domain Socket（AF_UNIX）是一种特殊的 Socket，它用于同一主机上的进程间通信（IPC），不经过网络协议栈。Unix Socket 使用文件系统路径作为地址，数据完全在内核中传递，不涉及网络接口，因此性能远高于网络 Socket。

Unix Socket 支持流式和数据报两种类型，还可以传递文件描述符（通过 SCM_RIGHTS 辅助消息）。Unix Socket 在许多场景中被使用：X Window 客户端与服务器的通信、Docker 守护进程与客户端的通信、systemd 的日志收集、数据库的本地连接。

## VSock

VSock（Virtual Socket）是为虚拟化场景设计的一种 Socket 地址族（AF_VSOCK），它允许宿主机和虚拟机之间或虚拟机之间建立高效的通信通道，而不需要经过传统的网络协议栈。VSock 使用 `(CID, port)` 地址对，其中 CID（Context ID）标识虚拟机或宿主机，port 标识服务端口。

VSock 的传输机制依赖于虚拟化平台的实现（如 VMware 的 VMCI、KVM 的 virtio-vsock）。数据通过虚拟化平台的共享内存或 hypervisor 传递，避免了网络协议栈和模拟网络设备的开销，提供了比以太网更高的性能和更低的延迟。VSock 常用于虚拟机的管理代理（如 QEMU Guest Agent）、文件共享、快速数据传输等场景。

## 系统调用

## 应用层
由用户态程序实现，此处不展开。