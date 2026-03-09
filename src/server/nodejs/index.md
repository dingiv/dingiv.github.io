---
title: Node.js
order: 12
---

# Node.js

Node.js 是基于 V8 引擎的 JavaScript 运行时，专为服务端高并发 I/O 设计。Node.js 的事件驱动、非阻塞 I/O 模型，使其适合 I/O 密集型应用。

## 事件循环

### 事件循环的原理

Node.js 是单线程的，所有 JavaScript 代码在主线程执行。主线程运行事件循环，不断从事件队列中取出任务执行。

### 事件循环的阶段

timers（定时器）、pending callbacks（I/O 回调）、idle/prepare（内部使用）、poll（I/O 回调）、check（setImmediate）、close callbacks（close 事件回调）。

### 微任务 vs 宏任务

微任务在当前宏任务执行完毕后立即执行。宏任务在事件循环的下一轮执行。

## 异步编程

### Promise

Promise 是异步操作的封装，提供链式调用。async/await 是 Promise 的语法糖。

### 回调函数

回调是最简单的异步编程方式，但会导致回调地狱。

## V8 引擎

### V8 的架构

解析器、解释器、编译器、垃圾回收器。

### V8 的优化

JIT 编译、内联缓存、隐藏类。

## 适合场景

### 适合

I/O 密集型应用：API 服务、实时通信、代理服务器。

实时应用：聊天应用、协作编辑。

工具链：构建工具、脚手架、CLI 工具。

### 不适合

CPU 密集型应用：图像处理、视频编码。

高并发写：数据库写、日志写入。
