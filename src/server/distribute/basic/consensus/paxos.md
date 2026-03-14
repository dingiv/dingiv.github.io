---
title: Paxos 算法
order: 1
---

# Paxos 算法

Paxos 是 Leslie Lamport 于 1998 年提出的分布式共识算法，是第一个解决共识问题的算法。Paxos 理论完备但实现复杂，被公认为难以理解。

## Paxos 的背景

在 Paxos 之前，分布式共识问题没有通用的解决方案。大多数系统依赖单一领导者或超时机制，无法处理网络分区和节点故障。

Paxos 的名字来源于希腊岛屿 Paxos，岛上立法机构通过投票制定法律。Lamport 用这个类比来描述共识过程。

## Paxos 的角色

### Proposer（提议者）

Proposer 提出提案，提案包含编号和值。Proposer 的目标是让 Acceptor 接受它的提案。Proposer 可以有多个，但通常只有一个 Proposer 能够成功（称为 Leader）。

### Acceptor（接受者）

Acceptor 接受或拒绝提案。Acceptor 可以有多个，但只有多数派接受才能达成一致。Acceptor 必须持久化存储已接受的提案，否则重启后会丢失状态。

### Learner（学习者）

Learner 获取已接受的提案，学习达成一致的值。Learner 可以从 Acceptor 获取已接受的提案，也可以从其他 Learner 获取。

## Basic Paxos

Basic Paxos 是 Paxos 的基础算法，用于达成一次共识。Basic Paxos 包含两个阶段：Prepare 阶段和 Accept 阶段。

### Prepare 阶段

Proposer 选择一个提案编号 n，向多数派 Acceptor 发送 Prepare(n) 请求。

Acceptor 收到 Prepare(n) 请求后，如果 n 大于它之前响应过的所有 Prepare 请求编号，则承诺不再接受编号小于 n 的提案，并返回它之前接受过的最大编号的提案（如果有）。

如果 n 小于或等于它之前响应过的 Prepare 请求编号，则拒绝 Prepare 请求。

### Accept 阶段

如果 Proposer 收到多数派 Acceptor 的 Prepare 响应，则发送 Accept(n, value) 请求。如果响应中有已接受的提案，则选择响应中编号最大的提案的值；否则，Proposer 可以自由选择值。

Acceptor 收到 Accept(n, value) 请求后，如果 n 大于或等于它承诺过的最大编号，则接受该提案；否则拒绝。

### Learn 阶段

一旦 Proposer 收到多数派 Acceptor 的 Accept 响应，就认为提案已达成一致，可以通知 Learner。Learner 可以从 Proposer 或 Acceptor 获取已接受的提案。

## Paxos 的安全性

### 安全性保证

Paxos 保证安全性：如果一个提案被多数派接受，那么该提案的值将被所有后续提案保持不变。这意味着一旦某个值被选定，后续提案只能选择这个值。

Paxos 还保证活性：如果没有提案被接受，Proposer 可以不断增加提案编号，最终会有提案被接受。

### 活性受阻

Paxos 的活性可能受阻：如果两个 Proposer 交替提出 Prepare 请求，但都未能完成 Accept 阶段，那么系统无法达成一致。这种情况被称为活锁。

活锁的解决方案：选举 Leader，只允许 Leader 提出提案。Leader 可以是稳定的（长时间不变），也可以是动态的（故障时重新选举）。

## Multi-Paxos

Basic Paxos 只能达成一次共识，效率低（每次共识需要两轮 RPC）。Multi-Paxos 是 Basic Paxos 的优化，通过选举 Leader 减少 RPC 次数。

### Leader 选举

Multi-Paxos 选举一个稳定的 Leader，由 Leader 提出所有提案。Leader 可以通过 Basic Paxos 选举，也可以通过外部服务（如 ZooKeeper）选举。

Leader 的优势：Proposer 只有一个，可以跳过 Prepare 阶段，直接进入 Accept 阶段。这减少了 RPC 次数，提高了吞吐量。

Leader 的劣势：Leader 可能成为瓶颈。如果 Leader 故障，需要重新选举，影响可用性。

### 日志复制

Multi-Paxos 将多个提案组织成日志，每个提案对应日志中的一个条目。Leader 接收客户端请求，将请求追加到日志，然后复制到 Follower。

日志复制的流程：Leader 发送 Accept(n, log_entry) 请求给 Follower，Follower 接受后追加到本地日志。多数派接受后，Leader 认为日志条目已提交。

日志的安全性：已提交的日志条目不能修改。日志条目按顺序提交，Leader 不能跳过某个条目提交后面的条目。

## Paxos 的实现

Paxos 的实现复杂，容易出现错误。Lamport 甚至写了一篇论文《Paxos Made Simple》，但仍然难以理解。

Paxos 的实现难点：Prepare 和 Accept 阶段的协调、多数派的判断、Leader 选举、日志复制、故障恢复。

Paxos 的开源实现：Google Chubby（基于 Paxos）、Apache ZooKeeper（基于 ZAB，类似 Paxos）、etcd（基于 Raft）。

## Paxos vs Raft

Paxos 理论完备但实现复杂，Raft 简化了 Paxos，使其易于理解和实现。Paxos 允许多个 Proposer，Raft 只允许一个 Leader。Paxos 的日志可以不连续，Raft 的日志必须连续。

Paxos 适合作为理论研究的对象，Raft 适合作为工程实践的参考。

## Paxos 的应用

Google Chubby：Google 的锁服务，用于 Google 内部的分布式协调。Chubby 使用 Paxos 保证元数据的一致性。

Google Spanner：Google 的全球分布式数据库，使用 Paxos 实现数据复制。

Apache ZooKeeper：Apache 的协调服务，使用 ZAB 协议（类似 Paxos）。

Paxos 是共识算法的基石，虽然难以理解，但其思想影响了后续的共识算法设计。理解 Paxos 有助于理解其他共识算法。
