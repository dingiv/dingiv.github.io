---
title: 监控与运维
order: 3
---

# 监控与运维

数据库监控是保障稳定运行的前提。通过监控可以及时发现性能瓶颈、容量预警、异常行为，为运维决策提供数据支持。

## 核心监控指标

### 可用性指标

可用性是数据库服务最基础的指标，通常用 SLA（Service Level Agreement）衡量。数据库可用性 = (总时间 - 不可用时间) / 总时间。生产环境的可用性目标通常设定为 99.9% 或更高，意味着每年不可用时间不超过 8.76 小时。

监控数据库可用性的方法包括定期探测连接、检查服务进程状态、监控应用层错误日志。使用如 Prometheus + Alertman、Grafana、Zabbix 等监控工具，配置数据库不可用时的告警。

### 性能指标

QPS（Queries Per Second）和 TPS（Transactions Per Second）是衡量数据库负载的核心指标。QPS 表示每秒执行的查询数，TPS 表示每秒提交的事务数。对于 OLTP 系统，TPS 更能反映业务压力。监控 QPS/TPS 的趋势，可以评估数据库的负载情况和容量规划。

响应时间分为平均响应时间和 P99 响应时间。平均响应时间是所有查询的执行时间平均值，P99 响应时间是 99% 的查询的响应时间上限。P99 更能反映用户体验，因为长尾查询影响的是少数用户但体验极差。建议监控 P95、P99、P999 三个分位值，全面了解查询分布。

并发连接数反映当前活跃的连接数量，应该与 max_connections 参数对比。连接数接近上限会导致新连接被拒绝，需要及时扩容或优化连接池。连接池的配置同样重要，过大连接数会消耗大量内存，过小会导致应用等待连接。

### 资源指标

CPU 使用率需要关注长期趋势而非瞬时峰值。数据库的 CPU 使用率持续高于 80% 说明资源紧张，需要优化查询或扩容。CPU 使用率的突增可能由异常查询、备份任务、统计信息收集等引起，需要定位具体原因。

内存使用率重点关注缓冲池命中率。InnoDB 缓冲池命中率 = (1 - innodb_buffer_pool_reads / innodb_buffer_pool_read_requests) × 100%。正常值应该在 99% 以上，低于 95% 说明内存不足，需要增加缓冲池大小或优化查询减少磁盘 I/O。操作系统的内存使用率同样重要，swap 使用是危险的信号，说明物理内存不足。

磁盘 I/O 包括 IOPS、吞吐量、等待时间。监控磁盘队列长度、平均等待时间、使用率，使用率持续高于 80% 说明磁盘是瓶颈。对于 SSD，更关注 IOPS；对于机械硬盘，更关注顺序读写的吞吐量。tempdb 的增长和临时表数量也是间接的 I/O 指标，过多的磁盘临时表说明查询需要优化。

网络流量通常不是数据库的瓶颈，但需要关注异常流量。备库的复制延迟与网络带宽相关，如果复制延迟持续增长，可能是网络带宽不足或主库写入量太大。

## MySQL 监控

### 状态变量监控

MySQL 的 `SHOW STATUS` 命令提供了丰富的状态变量，是监控数据的主要来源。`Questions` 表示服务器执行的所有语句数量，`Com_select`、`Com_insert`、`Com_update`、`Com_delete` 分别表示各类操作的执行次数，通过这些变量可以计算 QPS 和读写比例。

`Slow_queries` 表示慢查询数量，如果该值持续增长，说明存在性能问题。`Connections` 表示历史连接总数，`Max_used_connections` 表示同时使用的最大连接数，这两个值可以评估连接数配置是否合理。`Threads_connected` 表示当前连接数，`Threads_running` 表示活跃的连接数，活跃连接数持续过高说明存在大量长事务或锁等待。

`Innodb_row_lock_waits` 表示行锁等待次数，`Innodb_row_lock_time` 表示行锁等待总时间。锁等待次数多说明存在锁竞争，需要检查事务隔离级别、索引设计、SQL 写法。`Handler_read_next` 表示通过索引顺序读取的行数，该值过大说明索引效率低或没有使用索引。

### Performance Schema

Performance Schema 是 MySQL 5.5+ 引入的性能监控组件，提供了底层的性能事件统计。`events_statements_summary_by_digest` 表记录了每种 SQL 语句的执行次数、执行时间、锁等待时间、扫描行数等统计信息，是分析 SQL 性能的金矿。

```sql
-- 查看执行时间最长的 10 条 SQL
SELECT digest_text, count_star, avg_timer_wait/1000000000000 as avg_sec,
  sum_timer_wait/1000000000000 as total_sec
FROM performance_schema.events_statements_summary_by_digest
ORDER BY sum_timer_wait DESC
LIMIT 10;
```

`file_summary_by_instance` 表记录了每个文件的 I/O 操作统计，可以定位热点文件。`table_io_waits_summary_by_table` 表记录了每个表的 I/O 等待统计，可以定位热点表。这些信息对性能优化有直接指导意义。

### 主从复制监控

主从复制是 MySQL 高可用的基础方案，监控复制状态至关重要。`SHOW SLAVE STATUS` 命令的输出包含复制状态的关键信息。`Seconds_Behind_Master` 表示复制延迟，正常值应该接近 0，如果持续增长说明备库同步跟不上，需要检查备库性能和网络带宽。

`Slave_IO_Running` 和 `Slave_SQL_Running` 应该都是 Yes，如果是 No 说明复制线程停止，需要查看错误信息。`Last_Error` 和 `Last_SQL_Error` 字段记录了最近的错误信息，常见的错误包括主键冲突、找不到表、连接中断等。`Relay_Log_Space` 表示中继日志大小，过大说明备库执行慢，需要排查。

对于 GTID 复制模式，监控 `Retrieved_Gtid_Set` 和 `Executed_Gtid_Set` 可以确认复制的进度。`Executed_Gtid_Set` 应该持续增长，如果停滞说明备库执行停止。

## PostgreSQL 监控

### 统计视图监控

PostgreSQL 提供了丰富的统计视图，`pg_stat_activity` 显示当前活动会话，`pg_stat_database` 显示数据库级别的统计，`pg_stat_user_tables` 显示用户表的统计，`pg_stat_user_indexes` 显示用户索引的统计。

`pg_stat_activity` 的 `state` 字段表示会话状态：active 表示正在执行查询，idle 表示空闲，idle in transaction 表示空闲但在事务中，后者是需要关注的，可能意味着应用忘记提交事务。`query_start` 字段表示查询开始时间，可以识别长查询。

```sql
-- 查找长时间运行的查询
SELECT pid, now() - query_start as duration, query
FROM pg_stat_activity
WHERE state = 'active'
AND now() - query_start > interval '5 minutes';
```

`pg_stat_database` 的 `blks_read` 和 `blks_hit` 表示磁盘读取和缓存读取次数，缓存命中率 = blks_hit / (blks_hit + blks_read)。正常值应该在 99% 以上，低于 95% 说明 shared_buffers 配置偏小或查询需要优化。

`pg_stat_user_tables` 的 `seq_scan` 和 `idx_scan` 表示顺序扫描和索引扫描次数，如果某个表的 seq_scan 很高而 idx_scan 很低，说明缺少合适的索引。`n_tup_ins`、`n_tup_upd`、`n_tup_del` 表示插入、更新、删除的行数，`n_tup_hot_upd` 表示 HOT 更新的行数，HOT 更新避免了索引维护，是 PostgreSQL 的优化特性。

### 复制监控

PostgreSQL 的流复制监控通过 `pg_stat_replication` 视图。`sync_state` 表示同步状态：sync 表示同步复制，potential 表示潜在同步复制（当同步备库故障时提升），async 表示异步复制。对于高可用要求高的场景，应该有至少一个 sync 备库。

`replay_lag` 表示备库的重放延迟，`flush_lag` 表示备库的刷盘延迟，这两个值应该保持在秒级。`sent_lsn` 和 `replay_lsn` 表示发送和重放的 LSN 位置，两者的差值表示备库待应用的 WAL 量。

## 告警策略

### 告警级别

告警分为紧急、重要、警告三个级别。紧急告警需要立即处理，例如数据库不可用、主从复制停止、磁盘空间不足 10%。重要告警需要尽快处理，例如慢查询突增、复制延迟超过 1 分钟、连接数接近上限。警告告警可以关注，例如备份失败、查询缓存命中率下降、锁等待增加。

### 告警收敛

告警收敛避免告警风暴，影响运维体验。相同告警在一段时间内只发送一次通知，或者在问题恢复前不再重复通知。告警分组将相关告警合并发送，例如同一个数据库的多个指标异常可以合并为一条告警。

### 告警通道

告警通道包括邮件、短信、钉钉、Slack、PagerDuty 等。紧急告警应该通过多种通道发送，确保及时接收。警告告警可以通过邮件或即时消息发送。告警内容应该清晰说明问题、影响范围、建议操作，避免需要登录系统查看详情。

## 日常维护

### 数据清理

数据持续增长会消耗存储空间和降低性能，定期清理过期数据是必要的维护操作。对于分区表，可以直接删除过期分区。对于普通表，使用 DELETE 删除数据后需要执行 `OPTIMIZE TABLE` 回收空间。对于时序数据，考虑使用 DROP PARTITION 或 TRUNCATE PARTITION，比 DELETE 高效得多。

```sql
-- 删除三个月前的订单（按月分区）
ALTER TABLE orders DROP PARTITION p202301;
```

定期清理 binlog 和 WAL 日志，避免占用过多磁盘空间。MySQL 的 `expire_logs_days` 参数控制 binlog 保留天数，PostgreSQL 的 `archive_cleanup_command` 控制归档 WAL 的清理。

### 统计信息收集

查询优化器依赖统计信息选择执行计划，统计信息不准确会导致错误的执行计划。对于频繁变更的表，需要定期收集统计信息。MySQL 的 `ANALYZE TABLE` 命令更新统计信息，PostgreSQL 的 `ANALYZE` 命令类似。可以配置定时任务在业务低峰期执行。

```sql
-- MySQL 更新表统计信息
ANALYZE TABLE orders;

-- PostgreSQL 更新数据库统计信息
ANALYZE;
```

统计信息收集的频率取决于数据变更频率，对于写入量大的表，每天收集一次；对于写入量小的表，每周收集一次即可。

### 索引维护

索引碎片会影响查询性能，定期重建索引可以消除碎片。MySQL 5.6+ 的 Online DDL 支持在线重建索引，不阻塞读写。`ALTER TABLE t ENGINE=InnoDB` 重建表并重建所有索引。PostgreSQL 的 `REINDEX INDEX` 重建单个索引，`REINDEX TABLE` 重建表的所有索引。

```sql
-- MySQL 在线重建表
ALTER TABLE orders ENGINE=InnoDB;

-- PostgreSQL 重建索引
REINDEX TABLE orders;
```

索引维护应该在业务低峰期执行，因为重建索引会消耗大量 I/O 和 CPU 资源。对于大表，可以考虑使用 `pt-online-schema-change` 工具在线变更表结构。

数据库监控是持续改进的基础，建立完善的监控体系、配置合理的告警策略、执行定期的维护操作，才能保持数据库的稳定运行和良好性能。
