---
title: ZAB 协议
order: 3
---

# ZAB 协议

ZAB（ZooKeeper Atomic Broadcast，ZooKeeper 原子广播）是 ZooKeeper 的原子广播协议，用于保证 ZooKeeper 集群的一致性。ZAB 类似 Raft，有领导者选举和消息广播两个阶段。

## ZAB 的背景

ZooKeeper 是 Apache 的分布式协调服务，用于分布式锁、配置管理、 leader 选举。ZooKeeper 需要保证多个副本的数据一致性，因此设计了 ZAB 协议。

ZAB 的目标是保证所有副本的顺序一致性：所有节点看到相同的事务顺序。

## ZAB 的角色

### Leader（领导者）

Leader 接收客户端请求，将请求广播给 Follower。Leader 是唯一的，系统中最多有一个 Leader。

Leader 的职责：接收客户端请求、将请求广播给 Follower、收集 Follower 的确认、提交已确认的事务。

### Follower（跟随者）

Follower 响应 Leader 的请求，包括事务请求和心跳请求。Follower 可以处理读请求，但写请求需要转发给 Leader。

Follower 的职责：响应 Leader 的事务请求、响应 Leader 的心跳请求、处理客户端读请求、如果长时间未收到 Leader 心跳，则发起选举。

### Observer（观察者）

Observer 不参与选举和事务确认，只接收 Leader 的事务广播。Observer 可以扩展读能力，但不影响写性能。

Observer 的职责：接收 Leader 的事务广播、处理客户端读请求、不参与选举和投票。

## 消息广播

### 事务提交流程

客户端向 Follower 提交事务请求，Follower 将请求转发给 Leader。

Leader 为事务分配全局递增的 ZXID（ZooKeeper Transaction ID），ZXID 是 64 位，高 32 位是 epoch（纪元），低 32 位是计数器。Leader 将事务广播给 Follower。

Follower 收到事务后，将其添加到事务日志，并向 Leader 发送 ACK。

Leader 收到多数派（N/2 + 1）的 ACK 后，广播 COMMIT 消息，通知 Follower 提交事务。

Follower 收到 COMMIT 消息后，执行事务，更新内存数据库。

### 消息广播的特性

顺序性：消息按 ZXID 顺序广播，保证所有节点看到相同的事务顺序。

原子性：事务要么被所有节点执行，要么都不执行。只有多数派确认的事务才会被提交。

## 崩溃恢复

### Leader 崩溃

如果 Leader 崩溃，Follower 会发起选举，选出新的 Leader。

新 Leader 必须拥有所有已提交的事务。已提交的事务是指多数派确认的事务，ZXID 最大。

新 Leader 恢复未提交的事务。未提交的事务是指 Leader 已广播但未获得多数派确认的事务，新 Leader 会丢弃这些事务。

### Follower 崩溃

Follower 崩溃后恢复，会从 Leader 同步数据。Follower 发送其最后的 ZXID 给 Leader，Leader 发送该 ZXID 之后的所有事务。

Follower 可能落后很多，Leader 可能发送快照（Snapshot）而不是事务。快照是某个时刻的内存数据库状态。

### 数据一致性

ZAB 保证数据一致性：已提交的事务不会丢失，未提交的事务不会执行。

已提交的事务是指多数派确认的事务，新 Leader 必须拥有这些事务。未提交的事务是指未获得多数派确认的事务，新 Leader 会丢弃这些事务。

## Leader 选举

### 选举触发

Follower 如果长时间（timeout，默认 2 倍 tickTime）未收到 Leader 心跳，则认为 Leader 故障，转为 LOOKING 状态，发起选举。

### 选举流程

节点广播选举投票（投票包含自己 ZXID），收到其他节点的投票后，比较 ZXID，选择 ZXID 最大的节点作为 Leader。

ZXID 最大的节点拥有最新的事务，选择它可以保证已提交的事务不丢失。

如果节点收到多数票，则成为 Leader，广播 LEADER 消息。其他节点收到 LEADER 消息后，转为 FOLLOWING 状态。

### 选举的快速恢复

ZAB 的选举可以快速恢复，因为 ZXID 最大的节点直接成为 Leader，不需要额外的数据同步。

ZAB 的选举可以保证数据一致性，因为 ZXID 最大的节点拥有所有已提交的事务。

## ZAB vs Raft

ZAB 和 Raft 都是 Leader-based 的共识算法，有领导者选举和日志复制两个阶段。

ZAB 的 epoch 类似 Raft 的 Term，ZAB 的 ZXID 类似 Raft 的日志索引和任期号组合。

ZAB 的区别：ZAB 只支持顺序操作（ZooKeeper 是类似文件系统的 API），Raft 支持通用状态机。ZAB 的快照是完整的内存数据库状态，Raft 的快照是状态机的快照。

## ZooKeeper 的应用

分布式锁：ZooKeeper 的临时顺序节点可以实现分布式锁。客户端创建临时顺序节点，序号最小的节点获得锁。

Leader 选举：ZooKeeper 的临时节点可以实现 Leader 选举。多个客户端创建临时节点，成功创建的客户端成为 Leader，其他客户端监听该节点。

配置管理：ZooKeeper 的监听机制可以实现配置变更通知。客户端监听配置节点，配置变更时 ZooKeeper 通知客户端。

服务发现：ZooKeeper 的临时节点可以实现服务注册与发现。服务启动时注册临时节点，客户端监听服务列表。

## ZAB 的总结

ZAB 是 ZooKeeper 的原子广播协议，保证 ZooKeeper 集群的一致性。ZAB 类似 Raft，有领导者选举和消息广播两个阶段。ZAB 保证顺序一致性、数据一致性，是分布式协调的基础。

理解 ZAB 有助于理解 ZooKeeper 的原理，也有助于理解其他共识算法。
