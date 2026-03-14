---
title: 链路追踪
order: 6
---

# 链路追踪

链路追踪是微服务架构中定位问题的重要工具，通过记录一个请求经过的所有服务，可以快速定位故障和性能瓶颈。链路追踪可以帮助我们理解服务的调用关系、分析请求的延迟、定位故障的服务。

## 为什么需要链路追踪

调用链路长：微服务架构下，一个请求可能经过多个服务，调用链路长，难以定位问题。

故障定位难：服务故障时，难以确定是哪个服务的问题，需要逐个排查。

性能分析难：请求响应慢，难以确定是哪个服务的性能问题，需要逐个分析。

依赖关系复杂：服务间的依赖关系复杂，难以理解服务间的调用关系。

链路追踪的优势：可视化调用链路（可以查看请求经过的所有服务）、定位故障（快速定位故障的服务）、分析性能（分析每个服务的延迟）、理解依赖（理解服务间的依赖关系）。

## 链路追踪的核心概念

### Trace（链路）

一个 Trace 代表一个完整的请求链路，从请求进入到响应返回的全过程。一个 Trace 包含多个 Span。

### Span（跨度）

一个 Span 代表一个服务调用，记录服务调用的开始时间、结束时间、耗时、标签。Span 有父子关系，一个 Span 可以有多个子 Span。

### Trace ID（链路 ID）

Trace ID 是链路的唯一标识，所有属于同一个链路的 Span 都有相同的 Trace ID。Trace ID 需要跨服务传递，通常通过 HTTP Header 或 RPC Metadata 传递。

### Span ID（跨度 ID）

Span ID 是 Span 的唯一标识，标识一个服务调用。Parent Span ID 记录父 Span 的 ID，形成调用树。

### Annotation（注解）

Annotation 记录事件的时间戳，如 cs（Client Send，客户端发送）、sr（Server Receive，服务端接收）、ss（Server Send，服务端发送）、cr（Client Receive，客户端接收）。

### Tag（标签）

Tag 记录自定义的键值对，如 http.method、http.status_code、db.type。

### Baggage（行李）

Baggage 是跨服务传递的键值对，可以传递用户信息、调试信息。Baggage 会随着调用链传递，但会增加带宽开销。

## 链路追踪的原理

### 采样

链路追踪的采样率，记录多少比例的链路。采样率过低可能漏掉问题，采样率过高增加开销。可以根据服务重要性设置不同的采样率，核心服务采样率高，非核心服务采样率低。

### 上下文传递

Trace ID 和 Span ID 需要跨服务传递，通常通过 HTTP Header（如 X-B3-TraceId、X-B3-SpanId）或 RPC Metadata 传递。服务接收到请求后，提取 Trace ID 和 Span ID，创建子 Span，传递给下游服务。

### 数据收集

Span 数据需要上报到链路追踪系统，可以同步上报或异步上报。同步上报会影响性能，异步上报可能丢失数据。可以使用消息队列（如 Kafka）缓存 Span 数据，提高可靠性。

### 数据存储

Span 数据存储到数据库（如 Elasticsearch、Cassandra），支持按 Trace ID、Span ID、Tag 查询。Span 数据量大，需要定期清理或归档。

## 主流链路追踪系统

### Zipkin

Zipkin 是 Twitter 开源的链路追踪系统，由 Zipkin Server、Zipkin Client、Zipkin UI 组成。

Zipkin Server：收集和存储 Span 数据，提供查询 API。

Zipkin Client：嵌入在应用中，记录和上报 Span 数据。

Zipkin UI：Web 界面，可视化调用链路。

Zipkin 的特性：简单易用、与 Spring Cloud 集成、支持多种存储（内存、Cassandra、Elasticsearch）。

Zipkin 的问题：性能较差、功能相对简单、维护状态不佳。

### Jaeger

Jaeger 是 Uber 开源的链路追踪系统，兼容 Zipkin 协议。

Jaeger 的架构：Agent（收集 Span 数据）、Collector（接收和存储 Span 数据）、Query（查询 Span 数据）、UI（可视化调用链路）。

Jaeger 的特性：高性能、与 Kubernetes 集成、支持多种存储（Elasticsearch、Cassandra）、支持采样率动态配置。

Jaeger 的优势：性能高、功能完善、云原生、维护活跃。

### SkyWalking

SkyWalking 是国产开源的 APM（应用性能管理）系统，提供链路追踪、监控、告警。

SkyWalking 的特性：无侵入（字节码增强）、支持多种语言（Java、.NET、Node.js 等）、可视化调用链路、服务拓扑图、性能指标。

SkyWalking 的优势：无侵入、功能全面、中文文档完善。

SkyWalking 的问题：字节码增强可能有性能问题、学习曲线陡峭。

### OpenTelemetry

OpenTelemetry 是 CNCF 托管的可观测性项目，整合了 OpenTracing 和 OpenCensus，提供统一的链路追踪、指标、日志标准。

OpenTelemetry 的特性：统一标准（Tracing、Metrics、Logs）、多语言支持、厂商中立。

OpenTelemetry 的优势：统一标准、避免厂商锁定、生态活跃。

## 链路追踪的最佳实践

设置合理的采样率：根据服务重要性设置不同的采样率，核心服务采样率高（如 100%），非核心服务采样率低（如 10%）。

记录关键信息：记录请求方法、URL、状态码、异常信息，便于问题排查。不要记录敏感信息（如密码、Token）。

上下文传递：确保 Trace ID 和 Span ID 正确传递，避免调用链中断。异步调用（如线程池、消息队列）需要手动传递上下文。

性能监控：监控链路追踪的性能开销，避免影响业务性能。异步上报、采样上报可以降低开销。

关联日志：链路追踪和日志关联，通过 Trace ID 查看相关的日志，便于问题排查。

关联监控：链路追踪和监控关联，通过 Trace ID 查看相关的性能指标，便于性能分析。

告警设置：设置异常告警，如请求超时、错误率上升，及时发现问题。

链路追踪是微服务架构的重要工具，理解链路追踪的原理和权衡，有助于设计合适的可观测性方案。
