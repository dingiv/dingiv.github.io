---
title: 库函数
order: 30
---

# 库函数

在 C 语言开发生态中，库的选择决定了程序的移植性、性能和开发效率。本文将常用的 C 库按功能领域进行分类整理。

## 基础标准与系统接口 (Base & System)

### C 标准库 (Standard Library)

所有 C 程序的根基，由 ISO C 标准定义。

* 主流实现：  
  * glibc：GNU 发布，功能最全，Linux 默认。  
  * musl libc：轻量级、静态链接友好，常用于 Docker 镜像和嵌入式 Linux。  
* 核心模块：stdio.h (I/O), stdlib.h (内存/进程), string.h (字符串), math.h (数学)。

### POSIX 标准接口

POSIX (Portable Operating System Interface) 定义了操作系统应提供的接口规范。

* 核心功能：  
  * 文件 I/O：open, read, write, close, 文件描述符语义。  
  * 进程管理：fork, exec, wait, exit。  
  * 线程 (Pthreads)：pthread_create, 互斥锁, 条件变量。  
  * 其他：信号处理、内存映射 (mmap)、网络套接字 (Socket)。  
* 局限性：更新较慢，无法涵盖 Linux 特有特性（如 epoll, io_uring）。

### Linux 系统编程接口

通常指封装了内核系统调用的用户态接口。

* 头文件路径：  
  * /usr/include/sys：封装系统调用（如 sys/socket.h, sys/stat.h）。  
  * /usr/include/linux：内核数据结构与常量定义（常用于驱动开发或底层底层工具）。  
* 交互方式：通过 ioctl 与内核模块通信，通过 errno 获取系统级错误。

## 通用工具库 (Utility Toolkits)

这些库弥补了 C 标准库缺乏高级数据结构（链表、哈希表）的短板。

* Glib (GNOME)：  
  * 特点：功能极度丰富，提供跨平台抽象。  
  * 组件：动态数组、平衡树、事件循环、线程池、主循环 (GMainLoop)。  
* APR (Apache Portable Runtime)：  
  * 特点：Apache 核心库，侧重于服务器端的跨平台抽象。  
  * 组件：高效的内存池 (Memory Pool)、文件锁定、共享内存。  
* Klib (Generic C Library)：  
  * 特点：轻量级、只有头文件的库，专注于极致的内存利用率和算法性能（如哈希表、排序）。

## 网络、并发与异步 (Network & Concurrency)

### 异步事件循环

* libuv：Node.js 核心库，跨平台支持最好（Windows IOCP 封装），API 设计现代。  
* libevent：老牌事件库，广泛用于 Memcached 等项目，支持多种 I/O 多路复用。  
* libev：比 libevent 更轻量、性能更高，但对 Windows 支持有限。

### 通信协议与安全

* libcurl：HTTP 客户端的事实标准，支持几乎所有应用层协议。  
* OpenSSL / BoringSSL：提供 SSL/TLS 加密及各种密码学算法（AES, RSA, SHA）。  
* ZeroMQ (ZMQ)：不仅是库，更是一种高性能异步通讯协议库，简化了复杂的 Socket 编程模型。  
* nanomsg：ZeroMQ 作者的后续作品，纯 C 实现，更轻量。

## 数据处理与持久化 (Data Processing)

### 数据库

* SQLite：零配置的嵌入式关系型数据库，C 程序存储结构化数据的首选。  
* hiredis：Redis 官方推荐的 C 客户端接口，简单高效。

### 序列化与格式解析

* JSON：jansson (易用性好), cJSON (极其轻量, 仅两个文件)。  
* Protocol Buffers (protobuf-c)：Google 序列化协议的 C 实现，适合高性能通信。  
* libxml2：功能完备的 XML 解析与生成库。

### 压缩与归档

* zlib：DEFLATE 算法的标准实现。  
* libarchive：支持多种格式（tar, zip, 7z）的读写抽象库。

## 多媒体与系统工程 (Multimedia & Engineering)

### 图像与视频

* FFmpeg (libav)：音视频编解码、协议分发的绝对霸主。  
* libpng / libjpeg-turbo：图像格式的参考实现。  
* FreeType：字体渲染的核心引擎。

### 日志与配置

* zlog：高性能、多线程友好的 C 日志库，配置文件灵活。  
* libconfig：支持结构化配置文件的读写，比 INI 更强大。  
* inih：极简的 INI 文件解析器。

### 测试与调试

* Unity：嵌入式 C 开发中常用的单元测试框架，单头文件即可。  
* Criterion：功能强大的现代 C 单元测试框架，输出友好。  
* Check：基于 Fork 机制的测试框架，能捕获段错误。
