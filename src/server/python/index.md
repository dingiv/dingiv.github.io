---
title: Python
order: 13
---

# Python

Python 是解释型、动态类型语言，以简洁、易读、生态丰富著称。Python 在 Web 开发、数据处理、机器学习中有广泛应用。

## GIL（全局解释器锁）

### GIL 的原理

GIL 是 Python 解释器的全局锁，保证同一时刻只有一个线程执行 Python 字节码。

### GIL 的影响

多线程无法并行执行 CPU 密集型任务。多进程可以绕过 GIL。

### 绕过 GIL

多进程、C 扩展、异步编程。

## 异步编程

### asyncio

asyncio 是 Python 的异步编程库，基于事件循环和协程。async def 定义协程，await 等待 Future 完成。

### 协程的优势

协程是用户态的轻量级线程，切换开销小。协程适合 I/O 密集型任务。

## Web 框架

### Django

全栈 Web 框架，包含 ORM、模板引擎、认证系统。

### Flask

微框架，只包含核心功能。

### FastAPI

现代 Web 框架，基于 asyncio 和 Pydantic，高性能。

## 类型提示

### 类型提示的语法

Python 3.5+ 引入类型提示，使用 typing 模块。

### Pydantic

Pydantic 是数据验证库，基于类型提示。FastAPI 使用 Pydantic 验证请求体。

## 适合场景

### 适合

Web 开发、数据处理、机器学习、脚本和自动化。

### 不适合

高性能服务、CPU 密集型任务、内存受限应用。
