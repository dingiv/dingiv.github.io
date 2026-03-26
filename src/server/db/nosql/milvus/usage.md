---
title: 使用实践
order: 3
---

# 使用实践

Milvus 的使用围绕 Collection 展开——Collection 类似关系数据库的表，定义 Schema、索引和分区策略后，进行数据写入和查询。与关系数据库不同，Milvus 不支持 UPDATE 操作（只支持 Insert 和 Delete，以及 Upsert 语义），不支持事务和跨 Collection 的 JOIN。理解这些差异，有助于正确设计数据模型和查询策略。

## Collection 与 Partition

Collection 是数据的逻辑容器，定义了 Schema（字段、类型、主键）和索引。Partition 是 Collection 的物理分片，数据写入时可以指定 Partition，查询时可以指定在哪些 Partition 中搜索。

Partition 的作用不仅是数据隔离，更是性能优化的手段。如果查询总是针对某个维度的子集（如按用户 ID、按时间范围），将这些数据分配到不同 Partition，查询时只搜索相关 Partition，可以大幅减少扫描的 Segment 数量。Milvus 支持在 Schema 中指定 Partition Key（如 user_id），写入时 Milvus 自动按 Partition Key 的哈希值将数据分配到不同 Partition，应用层无需手动管理。

但 Partition 数量不宜过多。每个 Partition 内部有自己的 Growing Segment 和 Sealed Segment，Partition 过多意味着 Segment 碎片化，每个 Segment 太小，索引效果差，且 Query Node 需要管理更多的 Segment 元数据。通常建议 Partition 数量不超过 1024，单个 Partition 的数据量不低于 100 万向量。

## 数据写入

Milvus 通过 gRPC 发送 Insert 请求，每次请求可以批量插入多条数据。写入的关键参数是 batch_size 和 flush_interval。batch_size 越大，网络开销越小、写入吞吐越高，但单次请求的延迟也越高，且 Data Node 的内存缓冲区需要更大。flush_interval 控制 Growing Segment 自动 Seal 的时间阈值（默认 60 秒），间隔越短数据越快可搜索，但 Segment 越小、索引越碎。

批量写入的性能优化经验：单次 Insert 建议在 1000-10000 条之间，配合 16-32 并发客户端可以达到较高的写入吞吐。写入前不需要显式创建索引——Milvus 会在 Sealed Segment 满足条件后自动触发索引构建。如果需要在写入期间就进行查询，Growing Segment 默认使用 FLAT 索引，可以满足实时性要求。

Upsert 语义通过 Insert + Delete 实现：Milvus 支持在 Insert 请求中携带已存在的主键，此时会先删除旧数据再插入新数据。频繁的 Upsert 会产生大量 delete_log，影响查询性能（需要先过滤删除记录），建议尽量避免或批量执行。

## 向量搜索

Milvus 提供了两种查询接口。Search 是向量相似度搜索，输入查询向量和 TopK，返回最相似的 K 条结果及其距离分数。Query 是标量查询，输入过滤条件，返回匹配的记录（类似关系数据库的 SELECT WHERE）。

Search 请求的核心参数是 top_k 和搜索参数（search_params）。top_k 是返回的结果数量。search_params 因索引类型而异：HNSW 需要 ef 参数（建议设为 top_k 的 2-4 倍），IVF 需要 nprobe 参数（建议 8-32），DiskANN 需要 search_list 和 max_scan 参数。这些参数控制了精度和速度的权衡，可以在运行时调整。

搜索结果中，Milvus 返回每条结果的距离分数（score）和主键（id）。score 的含义取决于创建索引时选择的 metric_type：L2 距离越小越相似，IP 和 Cosine 越大越相似。应用层通常需要将 score 转换为相似度百分比或设定阈值过滤低质量结果。

## 混合搜索

纯向量搜索只考虑语义相似度，但实际业务往往需要结合标量条件。例如电商场景中，用户搜索"红色运动鞋"，不仅需要语义匹配，还需要过滤价格区间、库存状态等条件。

Milvus 的混合搜索在 Search 接口中同时指定向量和过滤表达式（filter）。执行流程是：先对 Sealed Segment 执行标量过滤（利用 stat 文件快速跳过不满足条件的 Segment，再在满足条件的 Segment 内过滤行），再对过滤后的数据执行向量搜索。对于 Growing Segment，标量过滤和向量搜索同时进行。

过滤表达式使用简洁的表达式语法，支持比较运算（`price > 100`）、逻辑运算（`AND / OR / NOT`）、范围查询（`age IN [20, 30, 40]`）、JSON 内部字段访问（`metadata["category"] == "electronics"`）。Milvus 在 Growing Segment 上使用暴力过滤，在 Sealed Segment 上可以使用标量索引（如 Bloom Filter、Inverted Index）加速过滤，标量索引需要在 Schema 中显式创建。

## 多向量查询

某些场景需要对多个向量字段联合检索。例如广告系统中，需要同时匹配图片向量和文本向量的相似度。Milvus 支持在一个 Collection 中定义多个向量字段，每个字段独立建索引。多向量查询时，对每个向量字段分别执行 Search，然后通过 Rerank 策略合并结果。

合并策略通常使用加权分数：$final\_score = w_1 \times score_1 + w_2 \times score_2$，权重由业务决定。Milvus 的多向量查询在 Proxy 层合并，而不是在存储层，因此延迟约为各独立搜索延迟的最大值加上合并开销。

## 多租户

Milvus 支持两种多租户方案。Partition 隔离是为每个租户创建独立的 Partition，写入和查询时指定 Partition，实现数据物理隔离。这种方式简单直接，但租户数量受 Partition 数量上限限制（1024），且每个 Partition 数据量过小时索引效果差。

Partition Key 是更轻量的方案，在 Schema 中指定某个字段（如 tenant_id）为 Partition Key，Milvus 自动按该字段的哈希值分 Partition。查询时指定 Partition Key 的过滤条件，Milvus 只搜索对应的 Partition。这种方式无需手动管理 Partition，适合租户数量较多但单个租户数据量中等的场景。

## 性能调优

性能调优的关键是理解瓶颈在哪里。写入瓶颈通常出在消息队列吞吐或 Data Node 数量不足，可以通过增加 Data Node 和调大消息队列分区来缓解。查询瓶颈有几个常见原因：Segment 未预加载导致冷启动延迟——使用 Load Collection 预加载；索引选择不当导致全量扫描——参考索引选择指南更换索引类型；标量过滤条件复杂导致过滤慢——创建标量索引（Bloom Filter 或 Inverted Index）。

资源规划的经验值：假设使用 HNSW 索引，每个 float32 向量的内存开销约为 $(4 \times d + 4 \times M)$ 字节（d 为维度，M 为连接数）。1 亿条 768 维向量使用 HNSW（M=16），内存需求约为 $10^8 \times (3072 + 64) \approx 313$ GB。IVF_SQ8 的内存约为 FLAT 的 1/4。对象存储空间约为向量原始数据大小的 2-3 倍（包含索引文件、binlog、stat 等冗余）。磁盘 I/O 带宽影响 Segment 加载速度和 DiskANN 的查询性能，建议使用 SSD。
