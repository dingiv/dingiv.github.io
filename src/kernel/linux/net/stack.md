# 协议栈

sk_buff：Linux 网络的核心载体

所有网络数据在内核中都封装为 struct sk_buff（skb）。

skb 是一个高度优化的包描述符：

头部指针分层推进（mac_header / network_header / transport_header）

支持 clone（零拷贝）

支持 scatter-gather

支持 GSO/GRO

理解 skb 的生命周期，是理解 Linux 网络性能的关键。

数据路径本质上是 skb 在不同子系统之间移动。

## 防火墙

netfilter 框架

## 路由子系统

## 邻居子系统

路由与邻居子系统

Linux 路由查找基于 FIB（Forwarding Information Base）。

最长前缀匹配

支持多表

支持策略路由（RPDB）

ARP 属于邻居子系统（neigh）。

ARP 表并不是简单缓存，而是状态机：

INCOMPLETE → REACHABLE → STALE → DELAY → PROBE

理解这个状态机，才能理解 ARP 抖动和丢包现象。

Netfilter 与 nftables

Netfilter 在五个关键 hook 点插入逻辑：

PREROUTING

INPUT

FORWARD

OUTPUT

POSTROUTING

iptables 是历史接口。 现代系统推荐使用 nftables。

NAT 本质是连接跟踪（conntrack）驱动的地址重写。

没有 conntrack，就没有状态 NAT。

性能瓶颈往往出现在 conntrack 表。