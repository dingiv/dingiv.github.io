---
title: 性能调优实践
order: 1
---

# 性能调优实践

数据库性能调优是一个系统工程，需要从 SQL 优化、索引设计、参数配置、架构调整等多个层面入手。本节介绍实用的调优方法和工具。

## 慢查询分析

### 开启慢查询日志

慢查询日志是定位性能问题的首要工具，它记录执行时间超过阈值的 SQL 语句。MySQL 中通过 `slow_query_log` 和 `long_query_time` 参数控制。

```sql
-- 开启慢查询日志
SET GLOBAL slow_query_log = ON;
SET GLOBAL long_query_time = 1; -- 超过1秒的查询被记录
SET GLOBAL log_queries_not_using_indexes = ON; -- 记录未使用索引的查询
```

慢查询日志默认存储在数据目录，可以通过 `slow_query_log_file` 参数指定路径。日志格式包含查询时间、锁时间、返回行数、扫描行数等关键信息，这些数据是分析瓶颈的依据。

### 分析慢查询日志

pt-query-digest 是 Percona Toolkit 提供的慢查询分析工具，它能从大量日志中提取有价值的信息。工具会统计每个查询的执行次数、总执行时间、平均执行时间、返回行数、扫描行数，并按照影响排序。

分析慢查询时重点关注三个指标：执行次数、平均执行时间、扫描行数与返回行数的比值。执行次数高的查询即使单次快，总体影响也可能很大；平均执行时间长的查询需要优化；扫描行数远大于返回行数说明索引效率低，这是最典型的优化点。

### EXPLAIN 分析执行计划

EXPLAIN 显示 MySQL 如何执行查询，是诊断 SQL 性能的核心工具。输出的 `type` 字段表示访问类型，从最好到最差依次是：const、eq_ref、ref、range、index、ALL。const 表示通过主键或唯一索引定位单行，eq_ref 表示连接时使用主键或唯一索引，ref 表示非唯一索引等值查询，range 表示范围扫描，index 表示索引扫描，ALL 表示全表扫描。

`key` 字段显示实际使用的索引，`key_len` 显示使用的索引字节数，可以判断是否使用了复合索引的全部列。`rows` 字段是估算的扫描行数，越小越好。`Extra` 字段的 Using index 表示覆盖索引，Using filesort 表示需要文件排序，Using temporary 表示使用临时表，这两个 Extra 通常意味着需要优化。

## 索引诊断

### 索引使用情况分析

MySQL 的 `performance_schema` 数据库包含 `table_io_waits_summary_by_index_usage` 表，记录了每个索引的使用情况。通过查询该表可以找出从未使用的索引，这些索引占用了存储空间和写入开销，可以安全删除。

```sql
SELECT object_schema, object_name, index_name, count_star
FROM performance_schema.table_io_waits_summary_by_index_usage
WHERE index_name IS NOT NULL
AND count_star = 0
AND index_name != 'PRIMARY'
ORDER BY object_schema, object_name;
```

索引选择性是衡量索引质量的重要指标，计算公式是 `COUNT(DISTINCT column) / COUNT(*)`。选择性越接近 1，索引效率越高。对于选择性低的列（如性别、状态），索引效果很差，不适合单独建索引。

### 索引失效场景

索引列上使用函数会导致索引失效，例如 `WHERE DATE(created_at) = '2023-01-01'` 无法使用 created_at 上的索引。解决方案是将函数移到等号右侧，改写为 `WHERE created_at >= '2023-01-01' AND created_at < '2023-01-02'`。

隐式类型转换也会导致索引失效。例如列类型是 VARCHAR，查询条件用整数 `WHERE phone = 13800138000`，MySQL 会进行类型转换，导致无法使用索引。解决方案是将查询条件改为字符串形式。

LIKE 查询的前缀通配符会导致索引失效，`WHERE name LIKE '%张%'` 无法使用索引。如果业务需要模糊搜索，可以考虑全文索引或 ElasticSearch。对于后缀通配符 `WHERE name LIKE '张%'`，索引仍然有效。

### 索引设计建议

为 WHERE、JOIN、ORDER BY、GROUP BY 子句中的列创建索引，但不要为每个查询都创建索引，需要权衡查询频率和写入开销。高并发写入的表应该少建索引，读多写少的表可以多建索引。

复合索引的列顺序遵循最左前缀原则，将区分度高的列放在前面。区分度可以用 `COUNT(DISTINCT column) / COUNT(*)` 计算，值越高区分度越好。对于等值查询后跟范围查询的场景，等值列放在前面，例如索引 (status, created_at) 可以支持 `WHERE status = 1 AND created_at > '2023-01-01'`。

覆盖索引可以避免回表，极大提升查询性能。例如对于查询 `SELECT name FROM users WHERE email = ?`，索引 (email, name) 可以直接从索引获取 name，不需要回表查询完整记录。

## 参数配置优化

### 连接数配置

`max_connections` 控制最大连接数，默认值 151 对生产环境偏小。设置过小会导致连接拒绝错误，设置过大会导致内存不足。估算公式是 `max_connections = (可用内存 - Global Buffer) / 每个线程的私有内存`。每个线程的私有内存包括 sort_buffer、join_buffer、read_buffer、read_rnd_buffer 等，默认都是 256KB，可以根据实际情况调整。

连接池可以缓解连接数压力。应用层使用 HikariCP、Druid 等连接池，避免频繁创建和销毁连接。对于短连接应用，考虑使用线程池模式（MySQL Enterprise 版本或 Percona 版本支持）。

### 缓冲池配置

`innodb_buffer_pool_size` 是 InnoDB 最重要的参数，控制缓冲池大小。缓冲池缓存数据和索引页，设置为可用内存的 50%-80% 是合理范围。对于专用数据库服务器，可以设置为 70%-80%；对于与应用混合部署的服务器，设置为 50% 左右。

`innodb_buffer_pool_instances` 控制缓冲池实例数，设置为 (innodb_buffer_pool_size / 1GB) 是经验法则。多个实例可以减少锁竞争，提升并发性能。

`innodb_log_file_size` 控制 redo log 文件大小，默认 48MB 偏小。设置过小会导致频繁切换日志，设置过大会导致崩溃恢复耗时。推荐设置为 256MB-2GB，根据写入量调整。写入量大的场景可以设置更大，但要注意 `innodb_log_buffer_size` 也要相应增加，一般为 16MB-64MB。

### IO 相关配置

`innodb_io_capacity` 和 `innodb_io_capacity_max` 控制脏页刷盘速率。对于 SSD，设置为 2000-40000；对于机械硬盘，设置为 200-2000。设置过小会导致脏页积压，设置过大会影响业务 I/O。可以通过观察 `innodb_buffer_pool_wait_free` 状态判断是否合理，如果该值增长说明缓冲池不足，刷盘跟不上。

`innodb_flush_method` 控制刷盘方式。Linux 下推荐 `O_DIRECT`，避免双缓冲，减少内存拷贝。Windows 下只能使用 `unbuffered`。

`innodb_flush_log_at_trx_commit` 控制事务提交时的刷盘策略。1 表示每次提交都刷盘（最安全但最慢），2 表示每次提交写到操作系统缓存，每秒刷盘一次（折中方案），0 表示每秒写日志并刷盘（最快但不安全）。生产环境推荐设置为 1，对于可以容忍少量数据丢失的场景可以设置为 2。

## 查询优化技巧

### 分页优化

传统分页 `LIMIT offset, size` 在 offset 很大时性能很差，因为需要扫描 offset 条记录后才开始返回数据。优化方案有三种：第一种是记住上一页的最大 ID，使用 `WHERE id > last_id LIMIT size`，这要求 ID 连续且可排序；第二种是使用延迟关联，先通过覆盖索引获取 ID，再 JOIN 查询完整记录；第三种是使用覆盖索引 + 自连接，适用于复杂排序场景。

```sql
-- 传统分页（offset 大时性能差）
SELECT * FROM orders ORDER BY id LIMIT 1000000, 10;

-- 记住上一页最大 ID（推荐）
SELECT * FROM orders WHERE id > 1000000 ORDER BY id LIMIT 10;

-- 延迟关联（适用于复杂排序）
SELECT o.* FROM orders o
INNER JOIN (SELECT id FROM orders ORDER BY created_at LIMIT 1000000, 10) tmp
ON o.id = tmp.id;
```

### JOIN 优化

多表 JOIN 时，小表驱动大表是基本原则。优化器会自动选择驱动表，但统计信息不准确时可能选择错误。可以通过 STRAIGHT_JOIN 强制连接顺序，但需要确保连接条件有索引。

子查询在某些场景下可以转换为 JOIN，性能更好。MySQL 5.6+ 对子查询做了优化，但仍然建议优先使用 JOIN。对于 IN 子查询，如果外层表大、内层表小，可以考虑转换为 EXISTS。

### 批量操作

批量插入比单条插入快得多，主要优势是减少了 SQL 解析开销和网络往返次数。使用 `INSERT INTO t VALUES (...), (...), (...)` 一次插入多条记录。对于大批量导入，使用 `LOAD DATA INFILE` 比 INSERT 快 10-20 倍。

批量更新和删除同样有性能优势。一次性更新多条记录时，使用 CASE WHEN 或 JOIN 批量更新，避免逐条更新。批量删除时，使用 `WHERE id IN (...)` 比 `WHERE id = ?` 多次执行更高效。

数据库性能调优是持续的过程，建立监控体系、定期分析慢查询、根据业务特点优化参数配置，才能保持数据库的高性能运行。
