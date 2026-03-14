---
title: 性能分析工具
order: 2
---

# 性能分析工具

性能分析工具帮助找到性能瓶颈，定位热点代码。本节介绍常用的性能分析工具和使用方法。

## CPU 性能分析

### perf

perf 是 Linux 的性能分析工具，基于硬件性能计数器。perf 可以分析 CPU 周期、指令、缓存命中率、分支预测。

常用命令：perf top（实时查看 CPU 热点函数）、perf record（记录性能数据）、perf report（分析性能数据）、perf annotate（注释代码，显示每条指令的 CPU 时间）。

perf 的工作原理：定期采样当前执行的函数，统计函数的 CPU 时间。采样频率越高，越精确但开销越大。默认频率是 99 Hz，每秒采样 99 次。

### 火焰图

火焰图是可视化 CPU 性能数据的图表，x 轴是样本数（越宽表示 CPU 时间越长），y 轴是调用栈（从下到上是调用关系）。火焰图有两种：火焰图（ Flame Graph，向上生长）表示 CPU 时间，冰柱图（Icicle Graph，向下生长）表示 CPU 时间。

生成火焰图：perf record -F 99 -p `<pid>` -g -- sleep 60（记录性能数据）、perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg（生成火焰图）。

### pprof

pprof 是 Go 的性能分析工具，可以分析 CPU、内存、锁。pprof 可以在代码中导入 _ "net/http/pprof"，然后访问 /debug/pprof/ 查看性能数据。

pprof 命令：`go tool pprof http://localhost:6060/debug/pprof/profile`（采集 CPU 性能数据）、go tool pprof -http=:8080 profile（可视化）、top（显示 CPU 热点函数）、list 函数名（显示函数的源码和 CPU 时间）。

### async-profiler

async-profiler 是 Java 的性能分析工具，低开销（1%）、高精度。async-profiler 可以分析 CPU、内存、锁。async-profiler 生成火焰图，与 Java Flight Recorder（JFR）兼容。

## 内存分析

### valgrind

valgrind 是 Linux 的内存调试工具，可以检测内存泄漏、非法内存访问、使用未初始化的内存。valgrind 的 memcheck 工具最常用：valgrind --leak-check=full --show-leak-kinds=all ./program。

valgrind 的问题：开销大（程序变慢 10-100 倍）、误报（认为某些分配是泄漏）、不适用于所有语言（需要符号信息）。

### pprof 内存分析

pprof 也可以分析内存：`go tool pprof http://localhost:6060/debug/pprof/heap`（采集堆内存）、top（显示内存分配热点）、list 函数名（显示函数的源码和内存分配）。

pprof 的内存模式：alloc_space（分配的内存）、inuse_space（使用的内存）、alloc_objects（分配的对象数）、inuse_objects（使用的对象数）。

### heapster

heapster 是 Java 的堆内存分析工具，可以分析堆转储（heap dump）。heapster 找到大对象、重复对象、GC 根。`jmap -dump:format=b,file=heap.dump <pid>` 生成堆转储，然后使用 heapster 分析。

## I/O 分析

### iostat

iostat 是 Linux 的磁盘 I/O 监控工具，可以查看磁盘的使用率、吞吐量、延迟。iostat -x 1 显示详细信息，每秒刷新一次。

iostat 的关键指标：%util（设备使用率，接近 100% 表示饱和）、await（平均 I/O 等待时间，包括队列和服务时间）、svctm（平均服务时间，不包括队列时间）。

### iotop

iotop 是 Linux 的磁盘 I/O 进程监控工具，类似 top，但显示 I/O。iotop 可以找出哪些进程在读写磁盘。iotop -o 只显示有 I/O 的进程。

### strace

strace 是 Linux 的系统调用追踪工具，可以查看程序的系统调用。`strace -p <pid>` 追踪运行中的进程，strace ./program 追踪新进程。

strace 可以分析 I/O 密集型程序：read、write、open、close 的次数和时间。strace -T 显示系统调用的时间，strace -c 统计系统调用的次数和时间。

## 锁分析

### perf lock

perf lock 是 perf 的锁分析工具，可以分析锁的竞争。perf lock record 记录锁事件，perf lock report 显示锁的统计信息：锁的名称、等待时间、持有时间。

### pprof lock

pprof 也可以分析锁：`go tool pprof http://localhost:6060/debug/pprof/mutex`（采集锁竞争）、top（显示锁竞争热点）。

### JStack

JStack 是 Java 的线程栈分析工具，可以查看线程的状态和调用栈。JStack `<pid>` 可以找出死锁（Found one Java-level deadlock）、等待锁（`waiting to lock <0x...>`）。

## 网络分析

### tcpdump

tcpdump 是 Linux 的网络抓包工具，可以抓取网络包并分析。tcpdump -i eth0 -w capture.pcap 抓包保存到文件，tcpdump -r capture.pcap 读取抓包文件。

tcpdump 的常用过滤器：host、port、src、dst、tcp、udp。例如 tcpdump -i eth0 tcp port 80 抓取 HTTP 流量。

### wireshark

wireshark 是图形化的网络分析工具，可以打开 tcpdump 的抓包文件。wireshark 可以分析协议（HTTP、TCP、UDP）、统计流量、重建会话。

### iftop

iftop 是 Linux 的网络流量监控工具，类似 top，但显示网络连接的流量。iftop 可以找出哪些连接占用了带宽。

## 性能分析流程

1. 确定分析目标：CPU、内存、I/O、锁、网络
2. 选择工具：perf、pprof、valgrind、iostat、tcpdump
3. 采集数据：压测环境或生产环境
4. 分析数据：找热点、找瓶颈、找异常
5. 优化代码：针对热点优化
6. 验证效果：压测对比基线

性能分析工具是性能优化的眼睛，选择合适的工具可以事半功倍。理解工具的原理和限制，才能正确解读分析结果。
