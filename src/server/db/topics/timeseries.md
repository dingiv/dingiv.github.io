---
title: 时序数据库
order: 2
---

# 时序数据库

时序数据库是专门为时间序列数据优化的数据库，典型应用包括监控指标、IoT 传感器、金融行情、日志事件。时序数据的特点是写多读少、数据持续增长、查询通常包含时间范围。通用数据库在处理时序数据时效率不高，时序数据库通过针对时序数据的优化实现高性能。

## 时序数据特征

### 数据模型

时序数据由三个核心维度构成：时间戳、度量、标签。时间戳记录数据点的时间，度量记录数值，标签记录元数据。例如 CPU 使用率的时序数据：时间戳是 2023-01-01 00:00:00，度量是 75.5，标签是 host=server1、region=beijing。

时序数据通常有高基数标签和低基数标签。高基数标签是取值多的标签，例如请求 ID、用户 ID。低基数标签是取值少的标签，例如主机名、地区。时序数据库对标签建立索引，对度量不建立索引，因为查询通常是按标签过滤、按时间聚合。

### 查询模式

时序数据的查询模式相对固定：查询最近 N 分钟的数据、查询指定时间范围的数据、按标签分组聚合、计算趋势和同比环比。聚合查询包括求和、平均值、最大值、最小值、分位数。降采样查询将高精度数据降采样为低精度数据，减少返回数据量。

时序数据很少查询单个数据点，几乎都是范围查询和聚合查询。这使得时序数据库可以针对这种查询模式优化，例如按时间分区、预聚合、列式存储。

## InfluxDB

### 数据模型

InfluxDB 的数据模型包括 measurement、tag set、field set、timestamp。measurement 类似关系数据库的表，tag set 是标签索引，field set 是度量值不索引，timestamp 是时间戳。查询必须指定时间范围，如果不指定默认返回最近数据。

InfluxDB 的 line protocol 是写入数据的格式，格式是 `measurement,tag_set field_set timestamp`。例如 `cpu,host=server1,region=beijing usage=75.5 1672531200000000000`。这种格式紧凑高效，适合高吞吐写入。

### 存储引擎

InfluxDB 的存储引擎称为 TSM（Time-Structured Merge Tree），是 LSM 树的变体。写入操作首先写入 WAL，然后写入内存 cache。cache 满后 snapshot 为 TSM 文件，TSM 文件是不可变的。后台定期合并 TSM 文件，清理过期数据，减少文件数量。

TSM 文件按时间组织数据，同一时间点的所有度量存储在一起，时间范围查询可以顺序读取。标签单独存储在索引中，支持快速标签过滤。TSM 文件支持压缩，时间戳使用 delta 编码，浮点数使用 gorilla 压缩，压缩比可以达到 10 倍以上。

### 查询语言

InfluxDB 的查询语言是 InfluxQL，类似 SQL 但针对时序数据优化。基本查询包括 SELECT、FROM、WHERE，WHERE 必须包含时间范围。聚合查询包括 GROUP BY time()、GROUP BY tag，GROUP BY time() 实现降采样。

```sql
-- 查询最近 1 小时的平均 CPU 使用率
SELECT mean(usage) FROM cpu
WHERE time > now() - 1h
GROUP BY time(5m), host;

-- 查询指定时间范围的最大值
SELECT max(usage) FROM cpu
WHERE time > '2023-01-01' AND time < '2023-01-02'
GROUP BY region;
```

InfluxDB 支持连续查询，即预先定义的定期执行的聚合查询，将聚合结果存储到新的 measurement。连续查询实现自动降采样，减少原始数据量。

### 集群架构

InfluxDB 1.x 的集群是企业版功能，InfluxDB 2.0 放弃集群架构，转为单机 + 数据复制。InfluxDB Enterprise 支持数据分片和副本，数据按时间范围和标签哈希分片。每个分片有 3 个副本，保证高可用。

InfluxDB 2.0 引入了 Edge 和 Replication 两种新的高可用方案。Edge 是轻量级的边缘实例，数据定期同步到中心实例。Replication 是数据复制，可以配置实例之间的复制关系。

## TimescaleDB

### PostgreSQL 扩展

TimescaleDB 是 PostgreSQL 的扩展，而不是独立的数据库。这意味着 TimescaleDB 继承了 PostgreSQL 的所有功能：SQL 支持、ACID 事务、丰富的数据类型、扩展生态。TimescaleDB 的优势是可以使用 PostgreSQL 的生态工具，同时获得时序数据的优化。

TimescaleDB 将表分为 hypertable 和 chunk。hypertable 是逻辑表，对用户透明。chunk 是物理表，按时间范围分区。数据写入时自动路由到对应 chunk，查询时自动合并多个 chunk 的结果。

```sql
-- 创建 hypertable
CREATE TABLE metrics (
  time TIMESTAMPTZ NOT NULL,
  sensor_id INTEGER,
  value DOUBLE PRECISION
);
SELECT create_hypertable('metrics', 'time');

-- 插入数据
INSERT INTO metrics VALUES
  ('2023-01-01 00:00:00+00', 1, 25.3),
  ('2023-01-01 00:01:00+00', 1, 26.1);

-- 查询数据
SELECT time_bucket('5 minutes', time) AS bucket,
       sensor_id, avg(value)
FROM metrics
WHERE time > NOW() - INTERVAL '1 hour'
GROUP BY bucket, sensor_id;
```

### 压缩策略

TimescaleDB 的压缩策略是将旧的 chunk 从行存转为列存。行存适合点查询和事务，列存适合范围查询和聚合。压缩后的 chunk 占用空间更小，查询性能更好。压缩是自动的，可以配置压缩规则，例如数据超过 7 天后自动压缩。

压缩使用 Gorilla 算法，这是 Facebook 开发的时序数据压缩算法。Gorilla 利用时间序列的连续性，存储增量而非原始值，使用 XOR 压缩浮点数。压缩比可以达到 10 倍以上，且解压缩速度极快。

### 连续聚合

TimescaleDB 的连续聚合类似于 InfluxDB 的连续查询，预先定义聚合查询，定期刷新聚合结果。连续聚合使用 materialized view 实现，存储聚合结果的物理表。刷新策略包括实时刷新和延迟刷新，实时刷新立即处理新数据，延迟刷新批量处理。

连续聚合的查询性能比原始数据查询快几个数量级，因为数据量大幅减少且预聚合。例如存储秒级数据，连续聚合生成分钟级和小时级数据，查询最近 24 小时使用分钟级数据，查询更长时间使用小时级数据。

## Prometheus

### 拉取模型

Prometheus 与 InfluxDB 的最大区别是数据获取方式。InfluxDB 是推送模型，应用主动写入数据。Prometheus 是拉取模型，Prometheus 服务器定期从目标拉取数据。拉取模型的优势是简单、无状态、易于水平扩展。

Prometheus 通过配置文件定义拉取目标，包括目标地址、拉取间隔、标签。目标通过 HTTP 接口暴露指标，Prometheus 定期请求该接口获取数据。这种设计使得目标不需要知道 Prometheus 的存在，只需暴露标准的 metrics 接口。

### 数据模型

Prometheus 的数据模型是时间序列，由指标名称和标签键值对唯一标识。例如 `http_requests_total{method="POST",handler="/api"}`。指标名称反映度量内容，标签反映维度。Prometheus 支持四种指标类型：Counter（计数器，只增不减）、Gauge（仪表盘，可增可减）、Histogram（直方图，分布统计）、Summary（摘要，分布统计）。

Prometheus 的查询语言是 PromQL，支持即时查询和范围查询。即时查询返回最新的值，范围查询返回时间序列。PromQL 支持丰富的运算符和函数，包括算术运算、比较运算、逻辑运算、聚合函数。

```
# 查询 QPS
rate(http_requests_total[5m])

# 查询 P95 延迟
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# 查询 CPU 使用率
rate(process_cpu_seconds_total{job="api"}[5m]) * 100
```

### 存储架构

Prometheus 的存储是本地存储，每个时间序列单独存储。数据按 2 小时为一个 block，block 包含该时间段的所有时间序列数据。旧 block 压缩后保留，新 block 写入内存。后台定期将内存数据 flush 到磁盘，创建新 block。

Prometheus 的存储是单机的，不支持集群。水平扩展通过联邦和远程存储实现。联邦是层级式的 Prometheus 架构，下层 Prometheus 收集数据，上层 Prometheus 从下层拉取数据。远程存储是将数据写入外部存储，例如 InfluxDB、TimescaleDB、Thanos。

## 应用场景

### 监控系统

监控系统是时序数据库的主要应用，包括基础设施监控、应用性能监控、业务指标监控。时序数据库存储各种指标：CPU、内存、磁盘、网络、响应时间、错误率、订单量、收入。监控仪表盘实时展示这些指标，告警系统根据阈值触发告警。

Prometheus + Grafana 是最流行的开源监控方案。Prometheus 收集和存储指标，Grafana 可视化和告警。对于大规模监控，可以使用 Prometheus Operator 管理 Kubernetes 上的 Prometheus，使用 Thanos 或 Cortex 实现长期存储和高可用。

### IoT 数据

IoT 设备产生大量时序数据，传感器数据按秒或分钟上报。时序数据库存储这些数据，支持实时查询和历史分析。IoT 场景的特点是设备数量多、数据持续上报、查询通常是最近的设备状态。

InfluxDB 是 IoT 场景的热门选择，因为其写入性能高、查询语言灵活、支持数据保留策略。对于大规模 IoT，可以考虑使用 TimescaleDB，利用 PostgreSQL 的生态和扩展性。

### 金融行情

金融行情数据包括股票价格、期货价格、外汇汇率、加密货币价格。这些数据的特点是更新频率高（毫秒级）、需要精确查询、需要复杂分析。时序数据库存储 tick 数据，提供实时行情和历史回测。

Kdb+ 是金融行业专用的时序数据库，性能极高但价格昂贵。对于中小型机构，可以考虑 TimescaleDB 或 ClickHouse，ClickHouse 是列式数据库，适合分析和聚合查询。

时序数据库是专门优化的数据库，在监控、IoT、金融等领域有不可替代的作用。选择时序数据库需要考虑数据量、查询模式、生态工具、团队能力，不同的场景适合不同的数据库。
