---
title: 并发模型
order: 6
---

# 并发模型

服务端需要同时处理大量请求，并发模型决定了如何组织多个执行单元（进程、线程、协程）来共享 CPU 时间。

## 并发 vs 并行

并发是多个任务在重叠的时间段内启动、运行和完成，但不一定同时进行。并行是多个任务同时进行，需要多核 CPU 支持。

## 进程

进程是操作系统资源分配的基本单位，拥有独立的内存空间、文件描述符。进程间通信需要特殊机制：管道、消息队列、共享内存。

### 多进程模型

Nginx 的 master-worker 模式，多个 worker 进程监听同一个 socket。Chrome 的多进程架构，每个标签页一个进程。

## 线程

线程是 CPU 调度的基本单位，同一进程的线程共享进程的内存空间。线程间通信通过共享内存，但需要同步机制。

### 线程安全

互斥锁（Mutex）、读写锁（RWLock）、原子操作（Atomic）、无锁数据结构（Lock-free）。

## 协程

协程是用户态的轻量级线程，由运行时调度而非操作系统内核。

### Go goroutine

Go goroutine 是有栈协程，初始栈大小 2KB，按需扩容。Goroutine 切换只需要保存三个寄存器，切换开销小。

### async/await

async/await 是基于事件循环的异步编程模型。JavaScript、Python、C++ 都支持 async/await。

## 事件驱动

事件驱动模型基于 Reactor 模式：事件源产生事件，事件循环监听事件，事件处理器处理事件。

### Reactor 模式

单 Reactor 单线程：所有事件串行处理。单 Reactor 多线程：Handler 在线程池中执行。主从 Reactor：Main Reactor 负责 accept，Sub Reactor 负责 I/O。

## 混合模型

实际系统常采用混合模式：事件循环处理 I/O，线程池处理 CPU 密集任务。Node.js 的 worker_threads 处理 CPU 密集任务。Go 的 goroutine 结合了事件驱动和多线程的优点。
