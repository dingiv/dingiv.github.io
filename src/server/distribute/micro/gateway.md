---
title: API 网关
order: 40
---

# API 网关

API 网关是微服务架构的统一入口，处理所有外部请求，负责路由转发、协议转换、认证授权、限流熔断等横切关注点。API 网关简化了客户端的调用，将服务治理逻辑从服务中剥离。

## 为什么需要 API 网关

客户端复杂：微服务架构下，客户端需要调用多个服务，需要处理服务发现、负载均衡、熔断降级，客户端复杂度高。

跨域问题：浏览器的同源策略限制跨域请求，需要服务支持 CORS。

认证授权：每个服务都需要认证授权，代码重复，难以维护。

限流熔断：每个服务都需要限流熔断，实现复杂。

API 网关的优势：统一入口（所有请求通过网关）、协议转换（HTTP 转 gRPC、WebSocket 等）、认证授权（集中管理认证授权）、限流熔断（集中管理限流熔断）、监控日志（集中记录访问日志）。

## API 网关的核心功能

路由转发：根据请求路径、头部、查询参数将请求转发到不同的后端服务。

负载均衡：在多个服务实例间分配请求，支持轮询、随机、权重、最少连接等策略。

协议转换：HTTP 转 gRPC、HTTP 转 WebSocket、HTTP 转 Dubbo 等。

认证授权：统一处理认证授权，支持 JWT、OAuth2、API Key 等。

限流熔断：保护后端服务不被过载请求打垮，支持令牌桶、漏桶、固定窗口等算法。

请求响应增强：添加或修改请求头、响应头，聚合多个服务的响应。

缓存：缓存后端服务的响应，减轻后端压力。

监控日志：记录访问日志，监控请求量、延迟、错误率。

灰度发布：根据请求特征路由到不同版本的服务，实现灰度发布。

## 主流 API 网关

### Zuul 1.x

Zuul 1.x 是 Netflix 开源的 API 网关，基于 Servlet 阻塞 I/O。

Zuul 的核心：Zuul Servlet（处理所有请求）、Filter（过滤器链）、Loader（动态加载 Filter）。

Zuul Filter：Pre Filter（路由前处理）、Routing Filter（路由转发）、Post Filter（响应后处理）、Error Filter（错误处理）。

Zuul 1.x 的问题：阻塞 I/O、性能差、不支持长连接。

### Zuul 2.x

Zuul 2.x 是 Netflix 开源的下一代 API 网关，基于 Netty 非阻塞 I/O。

Zuul 2.x 的特性：非阻塞 I/O、高性能、支持长连接、异步过滤器。

Zuul 2.x 的问题：与 1.x 不兼容、学习曲线陡峭、维护状态不佳。

### Spring Cloud Gateway

Spring Cloud Gateway 是 Spring Cloud 的 API 网关，基于 WebFlux 非阻塞 I/O。

Spring Cloud Gateway 的核心：Route（路由规则）、Predicate（匹配条件）、Filter（过滤器）。

Spring Cloud Gateway 的特性：非阻塞 I/O、与 Spring 生态集成、动态路由、限流（基于 Redis）。

Spring Cloud Gateway 的问题：功能相对简单、不支持 Dubbo 路由。

### Kong

Kong 是 Mashape 开源的 API 网关，基于 OpenResty（Nginx + Lua）。

Kong 的架构：Nginx（处理请求）、Lua（业务逻辑）、Cassandra（数据存储）。

Kong 的特性：高性能（基于 Nginx）、插件丰富（认证、限流、日志等）、可扩展（支持自定义插件）。

Kong 的优势：性能高、功能完善、生态活跃。

Kong 的问题：依赖 Cassandra、部署复杂、配置复杂。

### APISIX

APISIX 是 Apache 开源的 API 网关，基于 OpenResty 和 etcd。

APISIX 的架构：Nginx（处理请求）、Lua（业务逻辑）、etcd（配置存储）。

APISIX 的特性：高性能（基于 Nginx）、动态配置（基于 etcd）、插件丰富、云原生（Kubernetes Ingress）。

APISIX 的优势：性能高、配置简单、与 Kubernetes 集成、中文文档完善。

APISIX 的问题：社区相对年轻、不如 Kong 成熟。

### Traefik

Traefik 是云原生的 API 网关，自动发现服务。

Traefik 的特性：自动发现服务（Kubernetes、Consul、Etcd 等）、配置简单（基于 YAML）、自动 HTTPS（Let's Encrypt）、Metrics（Prometheus、StatsD）。

Traefik 的优势：云原生、配置简单、自动服务发现。

Traefik 的问题：性能不如 Nginx、功能相对简单。

## API 网关的设计模式

### 后端服务对于前端（BFF）

后端服务对于前端（Backend For Frontend）是为不同客户端（Web、移动端）提供不同的 API 网关。BFF 根据客户端的需求聚合后端服务的响应，提供定制化的 API。

BFF 的优势：减少客户端复杂度、优化网络请求（一次请求聚合多个服务）、隔离不同客户端的需求。

BFF 的问题：API 数量增加、维护成本增加。

### 网关集群

API 网关必须集群部署，避免单点故障。网关前可以部署负载均衡器（如 Nginx、ALB），将请求分发到网关实例。

网关需要保持无状态，便于水平扩展。状态（如限流计数）可以存储在 Redis 中。

### 网关的高可用

网关集群部署、健康检查、故障自动剔除、自动扩缩容。

网关前部署负载均衡器，实现高可用和负载均衡。

网关需要快速失败，避免阻塞请求。设置超时时间、熔断器。

## API 网关的最佳实践

路由规则：使用简洁的路由规则，避免过于复杂的正则表达式。

限流保护：保护网关和后端服务，避免被过载请求打垮。根据接口的重要性设置不同的限流策略。

超时配置：设置合理的超时时间，避免长时间等待。超时时间应大于后端服务的超时时间。

熔断降级：后端服务故障时快速失败，返回默认值或错误响应，避免雪崩。

日志记录：记录访问日志，包括请求时间、响应时间、错误码，便于问题排查。

监控告警：监控网关的请求量、延迟、错误率，异常时及时告警。

灰度发布：根据请求特征路由到不同版本的服务，实现灰度发布。可以根据 IP、Header、Cookie 等路由。

API 网关是微服务架构的入口，理解网关的原理和权衡，有助于设计合适的网关方案。
