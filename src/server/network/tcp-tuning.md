---
title: TCP 参数调优
order: 4
---

# TCP 参数调优

Linux 提供了大量可调优的 TCP 参数，正确配置可以显著提升服务端性能。本节从连接建立、数据传输、连接释放三个阶段介绍关键参数。

## 连接建立阶段

### somaxconn

syns 队列的最大长度，默认 128。高并发场景下，128 太小，导致连接被丢弃。建议设置为 4096 或更高。sysctl -w net.core.somaxconn=4096。

### synack_retries

发送 SYN+ACK 的重试次数，默认 5 次（约 30 秒）。重试次数太多，半连接队列占用时间长。建议设置为 2-3 次。sysctl -w net.ipv4.tcp_synack_retries=2。

### syncookies

开启 syncookies 可以防止 SYN 泛洪攻击，但会禁用一些 TCP 选项（如窗口扩大）。默认关闭，攻击时开启。sysctl -w net.ipv4.tcp_syncookies=1。

### tw_reuse

允许 TIME_WAIT 状态的 socket 用于新连接。默认关闭，建议开启。高并发短连接场景下，TIME_WAIT 状态的 socket 太多，占用端口。sysctl -w net.ipv4.tcp_tw_reuse=1。

### abort_on_overflow

syns 队列满时，是否直接发送 RST。默认关闭，建议保持关闭。开启会导致客户端连接失败，而不是排队等待。

## 数据传输阶段

### rmem_max/wmem_max

接收/发送缓冲区的最大值，默认约 128KB。高带宽延迟积（BDP）的网络需要更大的缓冲区。计算公式：带宽 × 延迟。例如 1Gbps、100ms 延迟，BDP = 1Gbps × 0.1s = 12.5MB。建议设置为 BDP 的两倍。

### rmem_default/wmem_default

接收/发送缓冲区的默认值，默认约 128KB。建议根据应用调整，Web 服务建议 256KB，大文件传输建议更大。

### tcp_fastopen

开启 TCP Fast Open（TFO），允许在三次握手期间传输数据。默认关闭，建议开启。TFO 减少 1 个 RTT，但需要客户端支持。

### tcp_notsent_lowat

发送缓冲区未发送数据的最低阈值，低于此阈值时不再发送。默认 -1（禁用），建议设置为 16KB。可以减少小包发送，提高吞吐量。

## 连接释放阶段

### fin_timeout

FIN_WAIT_2 状态的超时时间，默认 60 秒。太长会占用资源，太短可能导致连接关闭不完整。建议设置为 30 秒。

### keepalive_time

TCP keepalive 的空闲时间，默认 7200 秒（2 小时）。太长会导致僵尸连接占用资源，太短会增加网络流量。建议设置为 600 秒。

### keepalive_probes

TCP keepalive 的探测次数，默认 9 次。探测失败后关闭连接。建议设置为 3-5 次。

### keepalive_intvl

TCP keepalive 的探测间隔，默认 75 秒。建议设置为 15-30 秒。

## 性能参数

### tcp_slow_start_after_idle

空闲一段时间后，慢启动重新开始。默认开启，建议关闭。关闭后可以避免拥塞窗口不必要的缩小。

### tcp_no_metrics_save

不保存连接的性能指标（如 RTT、拥塞窗口）。默认关闭，建议开启。开启后可以避免不同连接间的性能指标干扰。

### tcp_mtu_probing

开启 MTU 探测，避免 IP 分片。默认关闭，建议开启。开启后可以自动发现 MTU，减少分片开销。

## 监控与调试

### ss 命令

ss -s：统计摘要。ss -t -a：显示所有 TCP 连接。ss -t -i：显示 TCP 详细信息（RTT、拥塞窗口）。

### tcpdump

tcpdump -i eth0 -w capture.pcap：抓包保存到文件。tcpdump -r capture.pcap -A：以 ASCII 显示包内容。

### /proc/net/tcp

/proc/net/tcp 文件包含所有 TCP 连接的状态，可以用于监控和调试。

TCP 参数调优需要根据实际场景调整，没有万能配置。理解每个参数的含义和影响，才能做出正确的调优决策。
