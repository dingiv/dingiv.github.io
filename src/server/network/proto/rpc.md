---
title: RPC 框架
order: 7
---

# RPC 框架
RPC（Remote Procedure Call）允许程序像调用本地函数一样调用远程服务，屏蔽网络通信的复杂性。

## RPC 的本质

RPC 的核心思想是将远程调用伪装成本地调用。RPC 框架负责：序列化参数、发起网络请求、等待响应、反序列化结果。

## 序列化

### 文本格式

JSON 是最常见的文本格式，可读性强、支持多种语言。缺点是性能差、体积大。

### 二进制格式

Protobuf 是 Google 开发的二进制格式，性能好、体积小、有 Schema。Thrift 是 Facebook 开发的二进制格式。

## RPC 框架组件

### 客户端

Stub（桩）、序列化器、网络传输、连接池、负载均衡、超时控制、重试机制。

### 服务端

网络传输、反序列化器、服务注册、服务实现、线程池、限流控制。

## gRPC

gRPC 是 Google 开源的高性能 RPC 框架，基于 HTTP/2 和 Protobuf。

### HTTP/2 的优势

多路复用、头部压缩、二进制协议、服务端推送。

### gRPC 调用模式

一元 RPC（Unary）、服务器流 RPC（Server Streaming）、客户端流 RPC（Client Streaming）、双向流 RPC（Bidirectional Streaming）。

## Dubbo

Dubbo 是阿里巴巴开源的高性能 RPC 框架，专注于服务治理。

### Dubbo 的架构

注册中心（Registry）、协议（Protocol）、负载均衡（Load Balance）、集群容错（Cluster）。

## Service Mesh

Service Mesh 将服务治理从应用代码中剥离，下沉到基础设施层。Sidecar 代理与每个服务实例部署在一起，接管所有进出流量。

### Istio

Istio 是最流行的 Service Mesh 实现，基于 Envoy 代理。
