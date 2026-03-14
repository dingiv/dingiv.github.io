---
title: QUIC 协议
order: 6
---

# QUIC 协议

QUIC（Quick UDP Internet Connections）是 Google 开发的基于 UDP 的传输协议，旨在解决 TCP 的问题。QUIC 提供可靠传输、流量控制、拥塞控制，支持多路复用、连接迁移、0-RTT 握手。

## 为什么需要 QUIC

### TCP 的问题

队头阻塞：TCP 是字节流协议，一个报文段丢失，后续报文段无法处理，导致队头阻塞。

连接建立慢：TCP 三次握手需要 1-RTT，TLS 握手需要 2-RTT，共 3-RTT。TLS 1.2 优化后需要 2-RTT。

连接僵化：中间设备（如防火墙、NAT）记录 TCP 连接状态，IP 变化时连接中断。

协议僵化：TCP 协议在操作系统内核中，升级困难。中间设备依赖 TCP 的特性（如拥塞控制），修改 TCP 困难。

### QUIC 的解决方案

多路复用：QUIC 支持多路复用，多个流独立传输，一个流的丢包不影响其他流，解决队头阻塞。

快速握手：QUIC 握手只需要 0-RTT 或 1-RTT，比 TCP + TLS 快。

连接迁移：QUIC 连接由 Connection ID 标识，IP 变化时 Connection ID 不变，连接不中断。

用户空间实现：QUIC 在用户空间实现，协议升级不需要操作系统升级。

## QUIC 数据包

### 数据包格式

Public Header：公共头，包含 Connection ID、包编号、版本号。

Private Header：私有头，包含密钥、流 ID、偏移量。

Payload：负载，包含流控制帧、握手数据、应用数据。

### Connection ID

Connection ID 是 64 位随机数，标识 QUIC 连接。Connection ID 在连接期间不变，IP 变化时 Connection ID 不变，连接不中断。

Connection ID 可以加密，防止中间设备跟踪连接。

### 包编号

包编号是 62 位整数，标识数据包的序号。包编号单向递增，0-RTT 包的包编号从 0 开始，1-RTT 包的包编号从 1 开始。

包编号用于 ACK、去重、拥塞控制。

## QUIC 流

### 流的概念

流是 QUIC 的虚拟通道，多个流共享一个 QUIC 连接。流是单向的或双向的，每个流有独立的流 ID。

流 A 的丢包不影响流 B，解决队头阻塞。

### 流 ID

流 ID 是 62 位整数，标识流。客户端发起的流 ID 是奇数，服务端发起的流 ID 是偶数。

流 ID 可以协商，减少流 ID 冲突。

### 流控制

QUIC 支持流级和连接级的流控制。流级流控制控制单个流的数据量，连接级流控制控制整个连接的数据量。

流控制通过 MAX_DATA 帧实现，接收方告诉发送方可以接收多少字节。

## QUIC 握手

### 0-RTT 握手

客户端发送 0-RTT 包，包含应用数据，使用之前缓存的密钥加密。服务端收到后解密，处理应用数据。

0-RTT 握手减少 1 个 RTT，但存在重放攻击风险。服务端可以检测重放攻击，拒绝 0-RTT 包。

### 1-RTT 握手

客户端发送 Client Hello，包含随机数、支持的加密套件。

服务端回复 Server Hello、证书、密钥、Finished。双方计算会话密钥。

客户端验证证书，回复 Finished，完成握手。

1-RTT 握手类似于 TLS 1.3，比 TLS 1.2 快 1 个 RTT。

## QUIC 的可靠性

### ACK 确认

QUIC 使用 ACK 确认数据包的接收。ACK 包含接收到的最大包编号、ACK 延迟时间、ACK 块。

ACK 块：记录收到的包编号范围，减少 ACK 大小。

### 超时重传

发送方发送数据后启动定时器，如果超时未收到 ACK，则重传数据。超时时间根据 RTT 动态调整。

QUIC 支持探测定时器，避免虚假超时。

### 前向纠错

前向纠错（FEC）发送冗余数据，接收方可以通过冗余数据恢复丢失的数据，减少重传。

## QUIC 的拥塞控制

### 拥塞控制算法

QUIC 支持可插拔的拥塞控制算法，如 Cubic、BBR、Reno。默认使用 Cubic。

QUIC 的拥塞控制与 TCP 类似，有慢启动、拥塞避免、快速重传、快速恢复。

### RTT 测量

QUIC 测量最小 RTT 和平滑 RTT（SRTT），用于计算超时时间和拥塞窗口。

## QUIC 的安全

### 加密

QUIC 使用 TLS 1.3 加密，所有数据包（除 Initial 包）都加密。

QUIC 使用 ChaCha20-Poly1305 或 AES-GCM 加密算法。

### 认证

QUIC 使用 TLS 1.3 认证，服务端发送证书，客户端验证证书。

QUIC 支持 Client Certificate，双向认证。

### 防重放攻击

0-RTT 包存在重放攻击风险，攻击者可以重放 0-RTT 包。

解决方案：服务端记录 0-RTT 包的签名，检测重放。服务端可以拒绝 0-RTT 包，只接受 1-RTT 包。

## QUIC 的应用

### HTTP/3

HTTP/3 基于 QUIC，解决 HTTP/2 的 TCP 队头阻塞问题。HTTP/3 保留 HTTP/2 的二进制协议、头部压缩、多路复用。

### Chrome

Chrome 默认开启 QUIC，访问 Google 服务（如 YouTube、Gmail）使用 QUIC。

### gQUIC vs iQUIC

gQUIC：Google 的 QUIC 实现，加密、流控制与 HTTP/2 绑定。

iQUIC：IETF 标准化的 QUIC，传输层与 HTTP 分离，更通用。

## QUIC vs TCP

### 优势

快速握手：QUIC 0-RTT 握手，TCP 需要 1-RTT + TLS 2-RTT。

多路复用：QUIC 流独立，TCP 字节流队头阻塞。

连接迁移：QUIC IP 变化连接不中断，TCP IP 变化连接中断。

用户空间：QUIC 在用户空间实现，升级方便。TCP 在内核中，升级困难。

### 劣势

普及率：QUIC 普及率低于 TCP，部分网络设备丢弃 QUIC 包。

性能：QUIC 加密开销大，性能可能低于 TCP。

调试：QUIC 协议复杂，调试困难。

QUIC 是下一代传输协议，解决 TCP 的问题，适合低延迟、高吞吐场景。理解 QUIC 的原理和权衡，有助于选择合适的传输协议。
