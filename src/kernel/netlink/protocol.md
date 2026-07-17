---
title: 协议分层
order: 20
---

# 通信协议分层

通信系统是一个跨越电气工程、体系结构、操作系统和分布式算法的分层体系。不同链路技术——PCIe、NVLink、InfiniBand、以太网——在物理层都是高速串行差分信号加 SerDes，本质差别不在电信号，而在协议层。理解每层做什么、瓶颈在哪，才能真正定位通信问题。

## 物理层：信号与带宽

物理层关注三个核心问题：比特如何在铜线或光纤上传输、信号速率是多少、每条链路有多少 lane。PCIe 5.0 的 32 GT/s、NVLink 的专用 SerDes 通道、InfiniBand 的 NDR 编码速率，都是物理层参数。

带宽的本质是单 lane 速率 × lane 数 × 编码效率。延迟与物理距离、SerDes 编解码延迟、交换芯片转发深度相关。一个关键洞察：PCIe、CXL、NVLink 在物理层共享相同的基础技术，真正的差异在上层协议。

## 数据链路层：可靠传输与流控

这一层负责 CRC 校验、重传机制和流控。不同技术采用不同的流控策略：PCIe 使用 credit-based 流控，InfiniBand 使用基于 Virtual Lane 的流控，以太网依赖 MAC + PFC 实现无损传输。

RoCE 的复杂性正源于此——它要在以太网上模拟 InfiniBand 的无损语义，依赖 PFC 防止丢包、ECN 做拥塞标记。如果网络丢包，RDMA 语义就会崩溃。这是协议层"语义依赖物理行为"的典型案例。

## 传输语义层：消息模型 vs 内存模型

这是通信技术中最重要的分水岭。

消息传递模型（TCP/IP、MPI）通过 send/recv 原语通信，CPU 参与协议栈处理。每个数据包需要经过内核网络栈的层层处理，延迟在数十到数百微秒量级。

内存语义模型（RDMA、CXL.mem、NVLink peer memory）允许直接读写远程内存，CPU 不参与数据路径，实现 zero-copy。RDMA 的本质不是"快"，而是绕过远端 CPU 协议栈——这就是 kernel bypass。InfiniBand 的约 1 μs 延迟正源于此。

两者不是替代关系：消息模型适合控制面和不频繁的通信，内存模型适合数据面和高频通信。分布式训练中的梯度同步使用 RDMA 走内存模型，而训练脚本的启动和参数分发仍然走消息模型。

## 缓存一致性层：CXL 的突破

CXL.cache 引入了 cache coherency——CPU cache line 可以与 GPU cache line 保持一致。这是比 RDMA 更高级的语义：RDMA 是远程内存访问，CXL 是远程 cache 访问。

这一层的突破改变了系统架构的可能性：内存池化（多个主机共享一个内存池）、内存扩展（通过 CXL 连接扩展内存）、设备共享页表。本质上，CXL 在做"跨设备 NUMA"——如果你熟悉 NUMA 架构，理解 CXL 只需要把"跨 socket 内存访问"的概念扩展到"跨设备内存访问"。

## 拓扑层：交换结构与带宽放大

NVSwitch、InfiniBand Switch 的作用是构建网络拓扑——Clos、Fat-tree、Ring、Mesh——并放大 bisection bandwidth。不同拓扑有不同的取舍：环形拓扑带宽利用率高但延迟叠加，全连接拓扑延迟最低但成本指数增长。Google TPU Pod 的 ICI 使用 2D mesh，是在成本与带宽之间的工程权衡。

## 集合通信层：从硬件到框架

硬件之上是软件抽象层。NCCL、MPI、Gloo、OneCCL 等通信库实现 AllReduce、AllGather、Broadcast、ReduceScatter 等集合操作。这些操作的算法复杂度与底层拓扑密切相关：Ring AllReduce 为 $O(N)$，Tree AllReduce 为 $O(\log N)$。

## 分布式策略层：算法感知硬件

Megatron 的 3D 并行是"算法感知硬件"的典范：节点内使用 NVLink 承载高频的张量并行通信（每层都需要同步），节点间使用 InfiniBand 承载中频的流水线并行通信（每个 microbatch 同步一次），最外层使用数据并行（每个 iteration 同步一次）。不同的通信频率被映射到不同物理层，最大化通信效率。

## 通信墙

通信墙的本质是计算增长约 $O(N)$，而通信增长约 $O(N \log N)$ 或 $O(N^2)$。当 GPU 数量增加时，参数同步量线性增长，拓扑拥塞指数增长。物理层无法无限扩展——光模块功耗、交换芯片端口密度、电源限制都会成为瓶颈。下一代互联技术正在使用硅光、光互连背板、Chiplet 封装内互联向"光速墙"逼近。

真正的优化不是"换一张更快的网卡"，而是理解整条分层栈的瓶颈在哪一层。
