---
title: API 网关
order: 20
---

# API 网关
代理解决的是"流量转发"——把请求从一个地址搬到另一个地址。API 网关解决的是"协议与治理"——在转发之外，理解 API 的语义并执行策略。两者边界模糊（Nginx 加几个模块也能当网关），但设计重心不同：代理优化的是连接和转发性能，网关优化的是 API 生命周期管理——路由、认证、限流、协议转换、可观测。

## 协议转换：网关的核心能力
协议转换是网关区别于普通代理的标志性能力。微服务内部通信与外部暴露的协议往往不一致——内部用 gRPC（高性能、强类型、服务发现友好），外部用 HTTP/JSON（通用、防火墙友好、前端可直接调用）。网关承担两者之间的翻译。

**HTTP → gRPC** 是最常见的方向。gRPC-Gateway（protobuf 注解生成）和 Envoy 的 grpc_json_transcoder 都做这件事：收到 HTTP/JSON 请求 → 按路由规则匹配到 gRPC 方法 → 把 JSON body 转换为 protobuf 消息 → 调用内部 gRPC 服务 → 把 protobuf 响应转回 JSON。转换的依据是 proto 定义——`google.api.http` 注解声明 URL 路径、HTTP 方法与 protobuf 字段的映射：

```protobuf
service UserService {
  rpc GetUser(GetUserRequest) returns (User) {
    option (google.api.http) = {
      get: "/v1/users/{id}"
    };
  }
}
```

这个注解让网关知道 `GET /v1/users/123` 映射到 `GetUser(id: "123")`。强类型的 proto 定义保证了转换的正确性——字段类型、嵌套结构、枚举值在编译期都有检查。

反向的 **gRPC → HTTP** 也存在——内部遗留服务只提供 HTTP API，新的 gRPC 客户端需要访问它们。这类转换更少见（通常直接让客户端走 HTTP），但网关的统一入口价值在于：客户端只需要知道一种协议。

**REST → GraphQL**、**WebSocket → gRPC streaming**、**HTTP/1.1 → HTTP/2** 这些转换组合构成了网关的协议矩阵。选型的关键约束是转换过程中的语义保真——gRPC 的 streaming、deadline 传播、metadata 在 JSON 转换中如何表达（通常转为 chunked response、HTTP timeout header、自定义 header）需要在网关层统一定义，否则客户端和服务端对语义的理解会产生分歧。

## 网关的功能版图
**路由与版本管理**：按路径、header、query 参数将请求分发到不同后端。版本路由（`/v1/*` → 旧服务、`/v2/*` → 新服务）让灰度发布和 API 演进成为运维操作而非代码改动。

**认证与授权**：集中处理 JWT 验签、API Key 校验、OAuth2 token introspection——后端服务不需要各自实现认证逻辑。这是"横切关注点集中化"的典型场景：认证逻辑写一次，所有服务受益。

**限流与熔断**：按客户端、按 API、按全局的流量控制。令牌桶/漏桶算法在网关层实现，超限请求直接返回 429 而不打到后端。熔断（连续失败后暂时拒绝流量）防止故障级联。

**可观测**：访问日志、请求链路追踪（注入 trace header）、延迟与错误率指标。网关是唯一能看到全部流量的位置——全局限流、全局审计、SLA 度量都依赖这里的观测数据。

**请求/响应改写**：header 注入（trace id）、body 转换（字段裁剪、格式统一）、mock 响应（测试环境模拟未完成的服务）。

## 与负载均衡器的分工
网关与 L4/LB（LVS、云厂商 SLB）的关系是层级分工：L4 LB 做 TCP/UDP 层的流量分发（性能极高、每秒百万连接），网关做 L7 的协议理解和策略执行（每秒万级请求但功能丰富）。典型架构是两层——外层 L4 LB（Anycast + 高可用），内层网关集群（水平扩展、无状态）。流量过 LB 的连接复用到网关，网关到后端另建连接——两层连接池各自管理。

把网关当 LB 用（单层 L7 网关直接对外）在小规模可行，但失去了 L4 层的连接级容错和 DDoS 吸收能力。把 LB 当网关用（L4 直接透传到服务）则失去了协议转换和策略集中化的全部价值。

## 主流方案
**Envoy**：CNCF 生态的核心数据面。Filter 链架构（每个过滤器处理一类关注点——认证、限流、转换可插拔）、xDS 动态配置 API（配置热更新无需重启）、原生支持 grpc_json_transcoder。Istio 服务网格的数据面就是 Envoy——这是它在大规模微服务中的事实标准地位来源。

**Kong**：基于 Nginx/OpenResty 的 API 网关，插件生态丰富（认证、限流、日志、转换各有现成插件）。管理面（Admin API + Kong Manager）适合传统部署，Kong Ingress Controller 适配 Kubernetes。优点是开箱即用，缺点是复杂转换逻辑需要写 Lua 插件。

**APISIX**：Apache 顶级项目，同样基于 Nginx/OpenResty 但架构更现代——etcd 存配置、热更新原生支持、插件用 Lua 但提供 Runner 机制支持多语言插件。国内生态活跃。

**Spring Cloud Gateway**：Java 生态的网关，基于 WebFlux 响应式模型。与 Spring Cloud 体系（服务发现、配置中心、熔断器）深度集成——Java 团队的默认选择，但 JVM 的资源占用和启动速度是运维负担。

**Traefik**：云原生的边缘路由器，自动服务发现（Kubernetes Ingress、Docker label、Consul catalog）。配置极简（标签即路由），适合中小规模和开发环境；超大规模下的性能和可定制性不如 Envoy。

## 选型视角
网关选型的第一个问题是"需要多少网关能力"。只需要路由 + TLS 终止 → Nginx 就够，不需要引入网关。需要认证 + 限流 + 协议转换 → 完整网关。第二个问题是团队生态——Java 团队用 Spring Cloud Gateway 的整合成本最低，Kubernetes 重度用户在 Envoy（Istio）和 APISIX/Kong 之间选择。第三个问题是性能需求——Envoy 的性能上限最高（C++ 数据面），OpenResty 系（Kong/APISIX）次之，Spring Cloud Gateway 最低。

协议转换的深入实践见 [gRPC](/server/network/proto/) 相关章节；网关与微服务治理（服务发现、熔断、链路追踪）的组合见[分布式](/server/distribute/)章节。
