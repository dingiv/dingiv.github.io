---
title: WebSocket
order: 5
---

# WebSocket

WebSocket 是 HTML5 提供的全双工通信协议，客户端和服务端可以随时发送数据。WebSocket 基于 TCP，握手阶段使用 HTTP，升级后使用自定义协议。

## 为什么需要 WebSocket

HTTP 的限制：HTTP 是半双工协议，客户端发送请求，服务端响应，服务端不能主动推送数据。

轮询的问题：客户端定期发送请求，服务端返回数据或空响应。轮询延迟高、浪费资源。

长轮询的问题：客户端发送请求，服务端保持连接，有数据时响应。长轮询减少请求次数，但服务端需要维护大量连接。

WebSocket 的优势：全双工通信，服务端可以主动推送数据。一个 TCP 连接，减少开销。低延迟，数据实时传输。

## WebSocket 握手

### 握手流程

客户端发送 HTTP 请求，请求方法为 GET，包含 Upgrade: websocket、Connection: Upgrade、Sec-WebSocket-Key 等头。

服务端回复 101 Switching Protocols，包含 Sec-WebSocket-Accept，升级为 WebSocket 协议。

握手后，TCP 连接保持，双方可以随时发送数据帧。

### Sec-WebSocket-Key

客户端发送的随机字符串，base64 编码。

服务端将 Sec-WebSocket-Key 拼接 GUID "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"，计算 SHA-1 哈希，base64 编码，得到 Sec-WebSocket-Accept。

验证握手是合法的 WebSocket 握手，防止普通 HTTP 请求被误认为 WebSocket 握手。

## WebSocket 数据帧

### 帧格式

FIN：1 位，是否是最后一帧。

RSV1-3：3 位，保留位，扩展使用。

Opcode：4 位，操作码。0x0 表示继续帧，0x1 表示文本帧，0x2 表示二进制帧，0x8 表示关闭帧。

MASK：1 位，是否掩码。客户端发送的帧必须掩码，服务端发送的帧不掩码。

Payload Length：7 位、7+16 位、7+64 位，负载长度。

Masking Key：32 位，掩码密钥，只有 MASK 为 1 时存在。

Payload Data：负载数据。

### 掩码

客户端发送的帧必须掩码，防止恶意脚本构造数据帧，攻击代理服务器。掩码算法：Payload Data[i] = Masked Data[i] ^ Masking Key[i % 4]。

服务端收到帧后，检查 MASK 是否为 1，如果是则解掩码。服务端发送的帧 MASK 为 0，不掩码。

### 操作码

0x0（继续帧）：消息的后续帧，用于分包传输。

0x1（文本帧）：UTF-8 文本数据。

0x2（二进制帧）：二进制数据。

0x8（关闭帧）：关闭连接，包含状态码和关闭原因。

0x9（Ping 帧）：心跳检测，接收方必须回复 Pong 帧。

0xA（Pong 帧）：对 Ping 帧的响应。

### 心跳检测

WebSocket 握手后，TCP 连接可能中间断网，心跳检测可以及时发现连接断开。

Ping/Pong 机制：一方发送 Ping 帧，另一方回复 Pong 帧。如果长时间未收到 Pong 帧，认为连接断开，关闭连接。

应用层心跳：定期发送应用层数据，如果长时间未收到响应，认为连接断开，关闭连接。

## WebSocket 的应用

### 实时聊天

客户端发送消息，服务端转发消息给其他客户端。WebSocket 全双工通信，消息实时到达，延迟低。

### 实时协作

多人同时编辑文档，WebSocket 同步编辑内容，避免冲突。

### 在线游戏

客户端发送操作，服务端广播游戏状态。WebSocket 低延迟，适合实时游戏。

### 股票行情

服务端推送股票价格，客户端实时显示。WebSocket 服务端主动推送，无需客户端轮询。

### 弹幕

客户端发送弹幕，服务端广播弹幕给所有客户端。WebSocket 全双工通信，弹幕实时显示。

## WebSocket 的安全

### 输入验证

服务端验证客户端发送的数据，防止 XSS、SQL 注入等攻击。

### 来源验证

验证 Origin 头，确保请求来自合法的网站，防止 CSRF 攻击。

### 速率限制

限制客户端发送消息的频率，防止消息洪水攻击。

### WAF

Web 应用防火墙（WAF）检测和阻止恶意 WebSocket 流量。

## WebSocket 的性能

### 连接管理

服务端维护大量 WebSocket 连接，需要高效的数据结构（如 epoll）。

连接心跳：定期发送 Ping 帧，检测连接是否存活，清理僵尸连接。

连接复用：WebSocket 连接复用 TCP 连接，减少连接建立的开销。

### 消息压缩

文本消息可以使用 gzip、brotli 压缩，减少传输数据量。

### 消息分片

大消息分片传输，避免阻塞其他消息。接收方收到分片后组装完整消息。

## WebSocket vs HTTP

### 全双工 vs 半双工

WebSocket 全双工通信，服务端可以主动推送。HTTP 半双工通信，客户端发送请求，服务端响应。

### 持久连接 vs 短连接

WebSocket 连接持久化，减少连接建立的开销。HTTP 短连接每次请求建立新连接，开销大。

### 低延迟 vs 高延迟

WebSocket 低延迟，数据实时传输。HTTP 轮询延迟高，定期发送请求。

### 复杂度 vs 简单

WebSocket 协议复杂，需要处理握手、分片、心跳。HTTP 协议简单，易于调试。

WebSocket 是实时通信的首选协议，理解 WebSocket 的原理和应用，有助于构建实时应用。
