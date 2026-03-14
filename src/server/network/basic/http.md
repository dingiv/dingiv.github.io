---
title: HTTP 协议
order: 4
---

# HTTP 协议

HTTP（Hypertext Transfer Protocol，超文本传输协议）是应用层协议，基于请求-响应模式。HTTP 是互联网的核心协议，用于传输 HTML、图片、视频等资源。

## HTTP/0.9

HTTP/0.9 是 1991 年的版本，只有 GET 方法，只支持 HTML，协议简单。

请求格式：GET /index.html \r\n。

响应格式：HTML 内容，无响应头。

## HTTP/1.0

HTTP/1.0 是 1996 年的版本，增加多种方法、响应头、状态码。

### 方法

GET：请求资源。

POST：提交数据。

HEAD：请求响应头，不返回响应体。

### 响应头

Content-Type：响应体的类型（text/html、application/json）。

Content-Length：响应体的长度。

Date：响应时间。

Server：服务器信息。

### 状态码

200 OK：请求成功。

301 Moved Permanently：资源永久重定向。

302 Found：资源临时重定向。

404 Not Found：资源不存在。

500 Internal Server Error：服务器内部错误。

### 连接

每次请求建立新连接，请求完成后关闭连接。缺点：建立连接开销大，延迟高。

## HTTP/1.1

HTTP/1.1 是 1997 年的版本，增加持久连接、分块传输、缓存控制、更多方法。

### 持久连接

Connection: keep-alive 保持连接，多个请求复用同一个连接，减少建立连接的开销。Connection: close 关闭连接。

### 分块传输

Transfer-Encoding: chunked 分块传输，响应体大小未知时使用。每个块包含块大小和块数据，最后一个块大小为 0 表示结束。

### 缓存控制

Cache-Control：缓存控制策略，如 no-cache（不使用缓存）、max-age=3600（缓存 3600 秒）。

ETag：资源的唯一标识，用于验证缓存。

Last-Modified：资源的最后修改时间，用于验证缓存。

### 更多方法

PUT：上传资源。

DELETE：删除资源。

OPTIONS：查询支持的方法。

TRACE：回显服务器收到的请求。

CONNECT：建立隧道，用于 HTTPS 代理。

### Host 头

Host 头指定请求的主机名，虚拟主机根据 Host 头选择不同的网站。Host 头是 HTTP/1.1 的必选头。

### 范围请求

Range: bytes=0-1023 请求资源的部分内容，用于断点续传、视频分片。响应头 Content-Range: bytes 0-1023/ total 表示返回的范围。

## HTTP/2

HTTP/2 是 2015 年的版本，基于 Google 的 SPDY 协议，解决 HTTP/1.1 的性能问题。

### 二进制协议

HTTP/1.x 是文本协议，HTTP/2 是二进制协议，解析更高效。

### 头部压缩

HPACK 算法压缩头部，减少传输开销。HTTP/1.1 的头部重复多（如 Cookie），占用大量带宽。

### 多路复用

一个 TCP 连接可以同时发送多个请求，解决 HTTP/1.1 的队头阻塞问题。请求按流 ID 标识，流可以有优先级。

### 服务端推送

服务端可以主动推送资源，减少客户端请求。例如客户端请求 HTML，服务端同时推送 CSS、JS。

### 流量控制

基于流的流量控制，窗口大小针对每个流，而不是整个连接。

## HTTP/3

HTTP/3 是 2022 年的版本，基于 QUIC 协议（UDP），解决 HTTP/2 的 TCP 队头阻塞问题。

### QUIC 协议

QUIC（Quick UDP Internet Connections）是基于 UDP 的传输协议，提供可靠传输、流量控制、拥塞控制。

QUIC 的优势：连接建立快（0-RTT）、队头阻塞消除（流独立）、连接迁移（IP 变化不需要重新连接）。

### HTTP/3 的变化

传输层从 TCP 改为 QUIC，保留 HTTP/2 的二进制协议、头部压缩、多路复用。

## HTTPS

HTTPS 是 HTTP over TLS，加密 HTTP 通信，防止窃听、篡改、冒充。

### TLS 握手

客户端发送 ClientHello，包含支持的加密套件、随机数。

服务端回复 ServerHello，包含选择的加密套件、随机数、证书。

服务端发送证书，客户端验证证书。

客户端生成随机数，用服务端公钥加密，发送给服务端。双方根据三个随机数生成会话密钥。

### HTTPS 的性能

TLS 握手增加 1-2 个 RTT，连接建立慢。

TLS 加解密占用 CPU，影响吞吐量。

优化措施：Session Resumption（会话恢复，减少握手）、TLS 1.3（0-RTT 握手）、硬件加速（加速加解密）。

## HTTP 的性能优化

### 持久连接

复用 TCP 连接，减少连接建立的开销。

### 域名分片

HTTP/1.1 浏览器对同一域名的连接数有限制（如 6 个），域名分片可以增加连接数。HTTP/2 不需要域名分片。

### 资源压缩

gzip、brotli 压缩 HTML、CSS、JS，减少传输数据量。

### 缓存

使用 Cache-Control、ETag、Last-Modified 缓存资源，减少请求。

### CDN

内容分发网络（CDN）将资源缓存到边缘节点，用户从就近节点获取资源，减少延迟。

HTTP 是互联网的核心协议，理解 HTTP 的演进和性能优化，有助于构建高性能的 Web 应用。
