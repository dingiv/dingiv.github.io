---
title: 事件驱动架构
order: 2
---

# 事件驱动架构

事件驱动架构是基于 Reactor 模式的并发模型，通过事件循环和事件处理器实现高并发。Node.js、Redis、Nginx 都采用事件驱动架构。

## Reactor 模式

Reactor 模式将事件分发和处理分离。Reactor 监听事件，将事件分发给对应的 Handler 处理。Handler 处理事件，完成后返回 Reactor。

### 单 Reactor 单线程

Reactor 和 Handler 在同一个线程中，所有事件串行处理。优点：简单，不需要锁。缺点：不能利用多核，Handler 阻塞会影响整个系统。

### 单 Reactor 多线程

Reactor 单线程，Handler 在线程池中执行。优点：Handler 可以并行，不阻塞 Reactor。缺点：Reactor 成为瓶颈， Handler 的并发控制复杂。

### 主从 Reactor

Main Reactor 负责 accept，Sub Reactor 负责 I/O。Main Reactor 将新连接分配给某个 Sub Reactor。优点：充分利用多核，职责分离。Netty、Nginx、Memcached 都采用主从 Reactor。

## 事件循环

事件循环是 Reactor 的核心，不断循环：获取事件，分发事件，处理事件。

### 事件循环的结构

事件队列：存储待处理的事件，可以是优先级队列。事件监听：监听 I/O 事件（可读、可写）、定时器事件、信号事件。事件分发：根据事件类型调用对应的 Handler。事件处理：Handler 执行业务逻辑，完成后返回事件循环。

### libuv 事件循环

libuv 是 Node.js 的底层库，实现跨平台的事件循环。事件循环阶段：timers（定时器）、pending callbacks（待处理的回调）、idle/resume（空闲回调）、poll（I/O 事件）、check（check 回调）、close callbacks（关闭回调）。

每个阶段执行完毕后，检查是否有待处理的 I/O 事件，如果有则进入 poll 阶段。poll 阶段阻塞等待 I/O 事件或定时器到期。

### Redis 事件循环

Redis 的事件循环（ae）包含：文件事件（socket 读写）、时间事件（定时器）。文件事件通过 epoll/kqueue/select 实现。时间事件通过最小堆实现，最近到期的定时器在堆顶。

Redis 的事件循环在单线程中执行，所有命令串行执行。Redis 6.0 引入多线程 I/O，但命令执行仍然是单线程。

## 异步编程

事件驱动架构要求非阻塞 I/O，否则会阻塞事件循环。异步编程是事件驱动的编程范式，通过回调、Promise、async/await 等机制编写异步代码。

### 回调

回调是最简单的异步编程方式，将后续操作作为回调函数传入异步操作。回调地狱（callback hell）是嵌套回调导致的代码难以阅读和维护。

### Promise

Promise 是对回调的封装，提供链式调用。Promise 有三种状态：pending、fulfilled、rejected。状态一旦改变就不能再次改变。Promise.catch 捕获异常。

### async/await

async/await 是 Promise 的语法糖，让异步代码看起来像同步代码。async 函数返回 Promise，await 等待 Promise 完成。async/await 是现代异步编程的首选方式。

## 阻塞与非阻塞 I/O

阻塞 I/O：调用后进程阻塞，直到数据准备好。非阻塞 I/O：调用后立即返回，如果数据未准备好返回 EAGAIN。事件驱动必须使用非阻塞 I/O，否则会阻塞事件循环。

### 非阻塞 I/O 的设置

Linux：fcntl(fd, F_SETFL, O_NONBLOCK)。Windows：ioctlsocket(fd, FIONBIO, &nonblock）。设置非阻塞后，read/write 可能返回 EAGAIN，表示数据未准备好或缓冲区满，需要等待下次事件。

### 边缘触发 vs 水平触发

边缘触发（ET）：状态变化时触发一次，必须一次性读完所有数据，否则数据会"丢失"。水平触发（LT）：状态就绪时持续触发，可以分批读取数据。ET 减少系统调用，但编程复杂；LT 编程简单，但可能有重复通知。

## 事件驱动 vs 多线程

事件驱动适合 I/O 密集型任务，如 Web 服务、代理服务器。多线程适合 CPU 密集型任务，如数据处理、图像渲染。

事件驱动的优势：不需要锁，编程简单；高并发，单线程处理大量连接；低延迟，没有上下文切换。劣势：不能利用多核；阻塞操作会阻塞整个循环；编程范式不同。

多线程的优势：利用多核，并行执行；编程范式与顺序编程相同。劣势：需要锁，编程复杂；线程创建和销毁开销大；上下文切换开销。

实际系统常采用混合模式：事件循环处理 I/O，线程池处理 CPU 密集任务。Node.js 的 worker_threads 处理 CPU 密集任务。Go 的 goroutine 结合了事件驱动和多线程的优点。
