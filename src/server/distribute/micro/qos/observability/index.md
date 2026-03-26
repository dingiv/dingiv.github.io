---
title: 可观测性
order: 8
---

# 可观测性

可观测性是系统运维的核心能力，通过 Metrics（指标）、Logging（日志）、Tracing（链路追踪）了解系统内部状态。

## 可观测性三大支柱

### Metrics（指标）

指标是数值型的测量值，反映系统的某个方面。指标类型：Counter（计数器）、Gauge（仪表盘）、Histogram（直方图）、Summary（摘要）。

指标示例：QPS、延迟（P50、P95、P99）、错误率、CPU 使用率。

### Logging（日志）

日志是离散的事件记录，包含时间戳、级别、消息、上下文。日志级别：DEBUG、INFO、WARN、ERROR、FATAL。

结构化日志（JSON）方便机器解析，文本日志方便人类阅读。

### Tracing（链路追踪）

链路追踪是请求在分布式系统中的路径，包含多个 Span（跨度）。链路追踪适合分析跨服务的调用链。

## 监控系统

### Prometheus

Prometheus 是开源的监控系统，使用 Pull 模式采集指标。PromQL 是查询语言，支持聚合、过滤、计算。

### Grafana

Grafana 是开源的可视化平台，支持多种数据源（Prometheus、ElasticSearch）。

## 日志系统

### ELK Stack

ElasticSearch（存储和搜索）、Logstash（收集和转换）、Kibana（可视化）。

### Loki

Loki 是 Grafana Labs 开发的日志系统，设计类似 Prometheus，不索引日志内容。

## 链路追踪

### OpenTelemetry

OpenTelemetry 是可观测性的开放标准，整合了 Metrics、Logging、Tracing。

### Jaeger

Jaeger 是 Uber 开源的链迹追踪系统，兼容 OpenTelemetry。

## 告警系统

### 告警规则

告警规则定义何时触发告警。Prometheus 告警规则：rate(http_requests_total[5m]) > 100。

### 告警路由

Alertmanager 路由告警到不同的接收者（邮箱、钉钉、企业微信）。

### 告警收敛

时间收敛、空间收敛、因果收敛。
