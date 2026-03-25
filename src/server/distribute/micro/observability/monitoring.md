---
title: 监控告警
order: 41
---

# 监控告警

监控告警是微服务可观测性的基础，通过收集和展示系统的运行指标，及时发现和处理异常。监控告警包括指标采集、数据存储、可视化展示、异常告警。

## 监控的核心指标

### RED 方法

RED 方法是监控黄金信号，由 Google 提出：Rate（请求速率，每秒请求数）、Errors（错误率，失败请求比例）、Duration（延迟，请求响应时间）。

这三个指标可以快速判断服务健康状态：Rate 下降可能服务故障，Errors 上升可能服务异常，Duration 上升可能性能问题。

### USE 方法

USE 方法针对资源监控：Utilization（资源使用率，如 CPU 使用率）、Saturation（资源饱和度，如 CPU 队列长度）、Errors（错误数，如 OOM 次数）。

USE 方法适用于资源层面的监控，可以提前发现资源瓶颈。

### 业务指标

除了技术指标，还需要监控业务指标：订单量、支付成功率、用户活跃度、转化率等。业务指标直接反映系统对用户的价值。

## 监控系统架构

### Prometheus

Prometheus 是 CNCF 托管的开源监控系统，采用拉模式（Pull）采集指标。

Prometheus 的架构：Prometheus Server（采集和存储指标）、Pushgateway（短周期任务推送指标）、Alertmanager（告警管理）、Exporter（指标导出器）。

Prometheus 的数据模型：时间序列，由 Metric Name（指标名）和 Label（标签）组成。如 `http_requests_total{method="GET",status="200"}`。

Prometheus 的查询语言：PromQL，支持聚合、过滤、计算。如 `rate(http_requests_total[5m])` 计算 5 分钟内的请求速率。

Prometheus 的存储：本地时序数据库（TSDB），支持高效写入和查询。默认保留 15 天，可以通过 Remote Write 长期存储。

Prometheus 的优势：简单易用、性能高、生态完善、与 Kubernetes 集成。

Prometheus 的问题：不支持集群（通过联邦实现）、不支持高可用（通过副本实现）、长期存储成本高（通过 Thanos、VictoriaMetrics 解决）。

### Grafana

Grafana 是开源的可视化平台，支持多种数据源（Prometheus、Elasticsearch、InfluxDB 等）。

Grafana 的核心：Dashboard（仪表盘，多个 Panel）、Panel（面板，单个可视化）、Query（查询，从数据源获取数据）、Alert（告警，基于规则触发）。

Grafana 的可视化类型：Graph（折线图）、Stat（单值）、Table（表格）、Heatmap（热力图）、Gauge（仪表盘）、Pie Chart（饼图）。

Grafana 的优势：美观、灵活、数据源丰富、告警集成。

Grafana 的问题：配置复杂、大量 Dashboard 难以管理。

### Exporter

Exporter 是将第三方系统的指标导出为 Prometheus 格式的组件。

常用 Exporter：Node Exporter（主机指标，如 CPU、内存、磁盘）、cAdvisor（容器指标，如 CPU、内存、网络）、Blackbox Exporter（网络探测，如 HTTP、TCP、ICMP）、MySQL Exporter（数据库指标）、Redis Exporter（缓存指标）。

## 告警系统

### 告警规则

告警规则定义告警条件和持续时间。如 `rate(http_requests_total{status="500"}[5m]) > 0.05` 表示 5 分钟内错误率超过 5% 时触发告警。

告警规则分为：Pending（告警条件满足，但未达到持续时间）、Firing（告警条件满足，已达到持续时间，触发告警）。

### 告警分组

告警分组将相关告警聚合，减少告警风暴。如按服务分组、按机房分组、按严重程度分组。

### 告警路由

告警路由将告警发送到不同的接收器。如 P0 告警发送到 PagerDuty 和电话，P1 告警发送到企业微信和邮件，P2 告警发送到企业微信。

### 告警抑制

告警抑制减少重复告警。如服务 A 告警时，抑制依赖服务 A 的服务 B 的告警（因为服务 B 的故障是由服务 A 引起的）。

### 告警静默

告警静默在已知维护期间暂停告警。如发布期间暂停告警，避免告警风暴。

## 告警最佳实践

设置合理的阈值：阈值不宜过高（漏报）或过低（误报）。根据历史数据和业务需求设置。

设置合理的持续时间：持续时间不宜过短（瞬态故障）或过长（响应慢）。根据业务容忍度设置。

分级告警：P0（核心业务故障，立即处理）、P1（重要业务故障，尽快处理）、P2（一般问题，工作时间处理）。

告警收敛：相同告警合并、相关告警聚合、依赖告警抑制。

告警确认：告警需要确认，避免告警疲劳。未确认告警升级（如 10 分钟未确认，升级到上级）。

告警复盘：定期分析告警，优化告警规则和系统稳定性。

监控告警是可观测性的基础，理解监控系统的原理和权衡，有助于构建稳定的微服务系统。
