---
title: 分布式能力
order: 2
---

# 存储与分布式架构
Milvus 生而就是为了分布式设计的。

采用存算分离架构，将存储、计算、协调三个关注点彻底解耦。这种架构使得各层可以独立扩缩容——写入量大时增加 Data Node，查询量大时增加 Query Node，存储量大时扩展对象存储集群。理解底层存储机制，有助于排查写入延迟、查询性能异常和数据一致性问题。

## 整体架构
Milvus 的组件分为四层。
+ 接口层由 Proxy 组成，负责客户端连接管理、请求路由、结果聚合和负载均衡，客户端只与 Proxy 通信，不知道内部拓扑。
+ 协调层由四个 Coordinator 组成：Root Coordinator 负责 DDL 操作（创建/删除 Collection、索引管理）、Data Coordinator 负责 Data Node 的调度和 Segment 均衡、Query Coordinator 负责 Query Node 的调度和数据加载、Index Coordinator 负责 Index Node 的调度和索引构建任务分发。所有 Coordinator 通过 etcd 选主实现高可用。
+ 执行层包括 Data Node（数据写入）、Query Node（查询执行）和 Index Node（索引构建），是无状态的，可以随意增删。
+ 存储层包括 etcd（元数据）、MinIO/S3（数据文件）、Pulsar/Kafka（消息日志），Milvus 自身不存储数据，完全依赖外部存储。

## Segment 的生命周期
Segment 是 Milvus 数据存储和查询的最小单元，理解它的生命周期是理解 Milvus 存储的关键。Segment 经历四个阶段。

Growing Segment 是内存中的可变数据结构，接收实时写入的数据。每个 Collection 在每个 Data Node 上有一个 Growing Segment，写入直接追加到内存缓冲区。Growing Segment 支持实时查询（索引通常为 FLAT），延迟最低。

当 Growing Segment 的大小达到阈值（默认 512MB）或存在时间超过阈值（默认 60 秒），Data Coordinator 会将其 Seal（封存），转化为 Sealed Segment。Sealed Segment 不再接受写入，变为不可变的。Sealed 的过程是将内存中的数据 Flush 到对象存储（MinIO/S3），生成持久化的数据文件。

Flushed Segment 存储在对象存储中，包含完整的数据和元信息。此时数据已持久化，但还没有构建向量索引（除非是 FLAT 索引）。Query Node 加载 Flushed Segment 后，使用 FLAT 做暴力搜索。

当 Flushed Segment 满足索引构建条件（大小和时间的配置阈值），Index Coordinator 将任务分配给 Index Node，构建向量索引。索引构建完成后写回对象存储，Segment 变为 Indexed Segment。Query Node 加载 Indexed Segment 时使用构建好的索引进行高效检索。

## 日志系统

Milvus 的写入链路基于日志即数据（Log-as-Data）的设计思想，类似于 Kafka 的消费模型。客户端的 Insert/Delete 请求经过 Proxy 后，先写入消息队列（Pulsar 或 Kafka），再由 Data Node 消费。

日志系统的设计有几个关键点。消息按 Collection + Shard 分 Topic，Shard 是 Collection 的物理分片，写入时可以指定 Shard Key 让相关数据落在同一 Shard，提高查询时的数据局部性。消息的顺序由消息队列保证，Data Node 按序消费，保证 Segment 内数据的顺序性。WAL 的保留策略决定了数据的安全性和存储成本：如果设置了持久化消费进度，即使 Query Node 全部宕机，重新加载后仍能从正确的位置恢复消费。

这种日志架构的好处是写入和计算完全解耦——写入只需追加日志，延迟极低；消费和索引构建是异步的，不影响写入吞吐量。代价是数据从写入到可搜索有一定延迟（取决于日志消费速度和索引构建速度）。

## 对象存储组织

Segment 在 MinIO/S3 中的存储按照紧凑的列式格式组织。一个 Flushed Segment 包含三种文件：binlog 文件存储实际数据（按列分文件，如 `image_vector.binlog`、`price.binlog`）、stat 文件存储列的统计信息（最大值、最小值、null 计数等，用于标量过滤的快速判定）、delete_log 文件存储删除记录。Indexed Segment 额外包含索引文件（如 `image_vector.index`，具体格式取决于索引类型，HNSW 的索引文件包含图的边信息和向量数据）。

这种列式存储的设计使得查询只需读取涉及的列，不需要扫描整行。例如纯向量搜索只需要读取向量列的 binlog 和索引文件，不需要加载标量列。标量过滤时先读 stat 文件做快速过滤，再读取对应列的 binlog 进行精确过滤。

## 查询执行

客户端发送 Search/Query 请求给 Proxy，Proxy 将请求按 Collection + Partition 拆分，广播给所有持有相关 Segment 的 Query Node。Query Node 对每个 Segment 独立执行检索：加载 Segment 的索引到内存（如果尚未加载），在索引中执行向量搜索和标量过滤，返回 TopK 结果。Proxy 收到所有 Query Node 的结果后，合并去重，按距离排序，返回最终的 TopK 给客户端。

Query 执行的瓶颈通常有两个：Segment 加载和搜索计算。Segment 加载是从对象存储读取索引文件到内存，大型 Segment 的加载时间可达数秒。Milvus 支持 Segment 预加载（Load Collection），在查询前将所有 Segment 加载到内存，避免查询时的冷启动延迟。搜索计算的开销取决于索引类型和数据量，HNSW 的搜索延迟通常在毫秒级。

## 一致性机制
Milvus 支持三种一致性级别。

+ Strong（强一致）保证查询能看到所有已确认写入的数据，实现方式是 Proxy 在执行查询前等待所有 Growing Segment 和最新 Flush 的数据被 Query Node 加载，延迟最高。
+ Bounded（有界延迟）允许指定一个时间窗口（如 5 秒），Proxy 只等待该时间窗口内的数据加载，在延迟和一致性之间取得平衡。
+ Eventual（最终一致）不等待任何未加载的数据，查询只反映已加载的 Segment 状态，延迟最低但可能读到旧数据。

工程实践中，大多数 AI 应用（如推荐系统、语义搜索）可以接受 Bounded 或 Eventual 一致性，因为向量检索本身就是近似计算，少量数据延迟不影响最终结果。Strong 一致性适合对数据新鲜度要求严格的场景，如实时内容审核。

## 数据均衡

Data Coordinator 和 Query Coordinator 分别负责写入和查询的均衡。写入均衡通过动态分配 Growing Segment 实现：当某个 Data Node 的 Growing Segment 达到阈值被 Seal 后，新的写入会被路由到负载较低的 Data Node。查询均衡通过在 Query Node 间重新分配 Sealed Segment 实现：当新 Query Node 加入或有 Node 宕机时，Coordinator 重新计算 Segment 的分配方案，触发 Segment 的迁移和加载。均衡过程中查询仍可正常服务，但迁移加载期间该 Segment 的查询延迟可能升高。
