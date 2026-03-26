---
title: ElasticSearch
order: 10
---

# ElasticSearch

ElasticSearch 是基于 Lucene 的分布式搜索引擎，专注于全文检索和日志分析。理解 ElasticSearch 的实现，需要深入倒排索引、分片架构、分布式协调和查询处理。

## 倒排索引

### 倒排索引结构

正排索引是文档到词的映射（文档包含哪些词），倒排索引是词到文档的映射（哪些文档包含这个词）。倒排索引包含：Term Dictionary（词项字典）、Posting List（倒排表）、Term Frequency & Position（词频和位置）。

Term Dictionary 是所有词的有序列表，支持快速查找。Posting List 是包含该词的文档 ID 列表，支持布尔查询。词频用于相关性评分，位置用于短语查询和邻近查询。

### FST 压缩

Term Dictionary 使用 FST（Finite State Transducer）压缩，FST 是一种有限状态自动机，共享前缀和后缀，极大压缩空间。FST 支持 O(len(term)) 的查找，比二分查找更快。FST 还支持映射，可以存储额外的信息（如文档频率）。

### Posting List 压缩

Posting List 是递增的整数数组，可以使用 Frame of Reference（FOR）压缩：将差分（gap）编码，然后按位存储。例如 [100, 101, 105] 压缩为 [100, 1, 4]，差分值小可以用更少的位存储。

跳表（Skip List）加速 Posting List 的交集和并集：每 n 个节点存储一个跳转指针，可以快速跳过不匹配的文档。

### 相关性评分

BM25 算法是默认的评分算法，基于词频（TF）、文档频率（IDF）、字段长度归一化。TF 越高分数越高，IDF 越高分数越高，字段长度越短分数越高。BM25 对词频有饱和效应，避免高频词主导分数。

## 分片架构

### 索引与分片

索引是文档的逻辑集合，分片是索引的物理划分。每个索引可以分成多个主分片（primary shard），每个主分片可以有多个副本分片（replica shard）。主分片数创建后不能修改，副本分片数可以动态调整。

文档路由：根据文档 ID 计算分片编号，shard = hash(_id) % number_of_primary_shards。然后路由到对应分片，由分片所在的节点处理。

### Segment

分片包含多个 Segment（段），每个段是不可变的索引文件。写入文档时，先写入内存 buffer，buffer 满或 refresh 后生成新段。段不可变，更新和删除是标记删除，添加新文档到新段。段合并（merge）将多个段合并成更大的段，清理删除的文档，减少段数量。

### Near Real-Time

refresh 默认每 1 秒执行一次，将内存 buffer 写入新段，新段对搜索可见。这是 ElasticSearch 近实时的原因。flush 将段写入磁盘，清空内存 buffer，生成 commit point。translog 记录所有操作，flush 后可以删除旧的 translog。

translog 保证持久性：每次写操作先写入 translog（fsync），然后写入内存 buffer。崩溃后重放 translog 恢复未 flush 的数据。translog 可以配置为 async（每 5 秒写一次），但可能丢失数据。

## 分布式协调

### 集群拓扑

集群由多个节点组成，节点通过 cluster.name 加入集群。节点角色：Master-eligible（可当选主节点）、Data（存储数据）、Ingest（预处理数据）、Coordinating（协调查询）。一个节点可以同时扮演多个角色。

主节点负责集群状态管理（创建删除索引、分片分配、节点加入退出）。主节点通过 cluster.state 更新集群状态，然后发布到所有节点。集群状态持久化到全局集群状态（不是每个节点都持久化）。

### 故障检测

节点定期 ping 其他节点，如果超时未响应，标记为故障。主节点故障时，从节点选举新主节点。选举使用 Raft 类似的算法：候选节点请求投票，多数节点同意则当选主节点。

脑裂（Split Brain）是两个节点都认为自己是主节点，导致集群状态不一致。防止脑裂：minimum_master_nodes 设置为 (master_eligible_nodes / 2) + 1，保证多数派才能当选主节点。ElasticSearch 7+ 默认使用投票配置，自动处理。

### 分片分配

主节点决定分片分配到哪些节点，考虑因素：磁盘空间、分片数量、节点角色、用户规则。分片可以自动迁移，平衡负载。节点退出时，分片迁移到其他节点。节点加入时，其他节点迁移分片到新节点。

## 查询处理

### 查询类型

查询分为两种：Query Context（查询上下文）和 Filter Context（过滤上下文）。Query Context 计算相关性评分，Filter Context 不计算评分（只判断是否匹配）。Filter 缓存结果，性能更好。

查询类型：match（全文查询，分词后查询）、term（精确查询，不分词）、range（范围查询）、bool（布尔查询，组合多个查询）、function_score（自定义评分）。

### 查询执行

客户端发送查询到任意节点（coordinating node），coordinating node 将查询发送到相关分片，每个分片本地执行查询，返回结果给 coordinating node，coordinating node 合并排序，返回最终结果。

分散查询（scatter-gather）：查询发送到所有分片，即使结果只有一个分片有。这是由于文档路由是 hash(_id)，查询条件可能包含其他字段，无法路由。

偏好查询（preference）：指定查询某个分片的副本，可以利用缓存。例如 _local（本地节点）、_primary（主分片）、cookie（用户一致性）。

### 聚合

聚合分为 Bucket（桶聚合，分组）、Metric（指标聚合，计算统计值）、Pipeline（管道聚合，基于其他聚合结果）、Matrix（矩阵聚合，多维度分析）。

聚合在分片本地执行，coordinating node 合并结果。某些聚合需要分散到所有分片，某些聚合可以优化。例如 cardinality（去重计数）使用 HyperLogLog 算法，可以分布式计算。

## 性能优化

### 索引设计

字段类型选择：keyword 适合精确匹配和聚合，text 适合全文检索。避免过多字段，可以使用 nested 或 object 类型。动态映射可能推断错误，建议显式定义 mapping。

### 分片策略

分片大小建议 10-50GB，过小导致分片数量多，过大会导致恢复慢。分片数量建议不超过节点数 × 2，避免分片分配开销。时间序列索引可以按时间创建索引，例如 logs-2023-01-01，定期删除旧索引。

### 查询优化

使用 filter context 代替 query context，利用缓存。避免深度分页（from + size 大于 10000），使用 scroll 或 search_after。避免 wildcard 查询前缀通配符，会扫描所有文档。避免 script 查询，性能差。

ElasticSearch 是强大的搜索引擎，适合日志分析、全文检索、指标监控。理解它的实现，有助于设计索引、优化查询和规划集群架构。
