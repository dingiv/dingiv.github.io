---
title: 监控指标设计
order: 2
---

# 监控指标设计

好的监控指标能够快速发现异常、定位问题、优化性能。本节介绍如何设计有效的监控指标。

## RED 方法

RED 方法是 Weaveworks 提出的监控指标设计方法，专注于请求相关的指标。

### Rate（速率）

每秒请求数，反映系统的负载。可以按服务、端点、状态码分组。Rate = 请求数 / 时间窗口。例如 http_requests_total / 60s 表示最近 60 秒的 QPS。

### Errors（错误）

错误请求数，反映系统的健康度。错误率 = 错误请求数 / 总请求数。错误可以按状态码分类（4xx 客户端错误、5xx 服务器错误）。4xx 可能是客户端问题，5xx 是服务器问题。

### Duration（持续时间）

请求的处理时间，反映系统的性能。Duration 可以用 P50（中位数）、P95（95 分位数）、P99（99 分位数）表示。P99 延迟是重要的性能指标，反映了最差的 1% 用户体验。

## USE 方法

USE 方法是 Brendan Gregg 提出的监控资源的方法，专注于资源利用率。

### Utilization（利用率）

资源的使用百分比，如 CPU 使用率、内存使用率、磁盘使用率。利用率过高（>80%）表示资源紧张，需要扩容或优化。

### Saturation（饱和度）

资源的负载程度，如 CPU 运行队列长度、磁盘 I/O 等待时间、网络连接数。饱和度高表示资源过载，请求排队等待。

### Errors（错误）

资源的错误数，如磁盘 I/O 错误、网络错误、内存页错误。错误增加表示硬件故障或配置错误。

## 指标类型

### Counter（计数器）

Counter 是只增不减的数值，适合累积事件。例如：http_requests_total（总请求数）、errors_total（总错误数）。Counter 的速率（rate）表示每秒增量，如 rate(http_requests_total[5m]) 表示最近 5 分钟的平均 QPS。

### Gauge（仪表盘）

Gauge 是可增可减的数值，适合瞬时状态。例如：memory_usage_bytes（内存使用量）、thread_count（线程数）、temperature_celsius（温度）。Gauge 的当前值反映系统状态，不需要计算速率。

### Histogram（直方图）

Histogram 是样本的分布统计，适合延迟、请求大小等。Histogram 包含：_count（样本数）、_sum（样本总和）、_bucket（桶计数）。Histogram 可以计算分位数，如 histogram_quantile(0.95, rate(http_duration_seconds_bucket[5m])) 表示 P95 延迟。

### Summary（摘要）

Summary 是客户端计算的分布统计，适合延迟。Summary 包含：_count（样本数）、_sum（样本总和）、quantile（分位数）。Summary 的分位数是客户端计算的，无法聚合多个实例的分位数。Histogram 的分位数是服务器计算的，可以聚合。

## 指标命名

### 指标名称规范

指标名用小写字母和下划线，单位用后缀表示。例如：http_requests_total（总数）、http_duration_seconds（秒）、memory_usage_bytes（字节）。避免使用缩写，用 latency 而不是 lat。

### 标签（Label）

标签是指标的维度，用于分组和过滤。例如：http_requests_total{method="GET", status="200"} 表示 GET 请求且状态码 200 的总数。标签是字符串，枚举值（如 status）比数字（如 status_code）更高效。

### 基数（Cardinality）

基数是标签的唯一值数量，高基数会导致指标爆炸。例如 user_id 有百万用户，基数是百万，会导致百万条时序数据。避免高基数标签，如 user_id、request_id。低基数标签，如 service、endpoint、status。

## 指标聚合

### 聚合操作

sum（求和）：sum(http_requests_total) 表示所有实例的总请求数。avg（平均）：avg(memory_usage_bytes) 表示所有实例的平均内存使用。max/min（最大值/最小值）：max(cpu_usage) 表示所有实例的最大 CPU 使用。

### 分组聚合

by（按标签分组）：sum(http_requests_total) by (service) 表示每个服务的总请求数。without（排除标签）：sum(http_requests_total) without (instance) 表示所有实例的总请求数（忽略 instance 标签）。

## 金金指标

Google SRE 书中提出的四个黄金指标：延迟（Duration）、流量（Traffic）、错误（Errors）、饱和度（Saturation）。

### 延迟

延迟是请求的处理时间，用 P50、P95、P99 表示。P99 延迟是最重要的性能指标，反映了最差的 1% 用户体验。监控成功请求和失败请求的延迟，失败请求可能很快（快速失败）。

### 流量

流量是每秒请求数，反映系统负载。可以按服务、端点、客户分组。流量突增可能是攻击、故障或正常增长。

### 错误

错误是失败的请求数，反映系统健康度。错误率 = 错误请求数 / 总请求数。区分客户端错误（4xx）和服务器错误（5xx），4xx 可能是客户端问题，5xx 是服务器问题。

### 饱和度

饱和度是资源的负载程度，如 CPU、内存、磁盘、网络。饱和度高表示资源过载，需要扩容或优化。CPU 运行队列长度、磁盘 I/O 等待时间、网络连接数都是饱和度指标。

## 告警规则

### 告警阈值

阈值应该基于历史数据，而不是随意设置。例如 P95 延迟的告警阈值设为历史 P95 的 2 倍。阈值应该有容忍度，避免告警风暴。

### 告警持续时间

告警持续时间（for）是条件持续多久才触发告警。避免瞬时波动触发告警。例如 rate(errors_total[5m]) > 0.01 for 10m 表示错误率超过 1% 持续 10 分钟才告警。

### 告警优先级

告警按优先级分类：P0（严重，如服务宕机）、P1（高，如错误率突增）、P2（中，如性能下降）、P3（低，如资源紧张）。不同优先级的告警有不同的响应时间、通知渠道、升级策略。
