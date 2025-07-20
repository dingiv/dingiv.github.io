---
title: 异步
order: 50
---

# 异步编程
异步编程是一种编程范式和技术合集，它主要目的是优化阻塞 IO 造成的程序挂起问题。阻塞 IO 使得程序无法充分竞争 CPU，造成资源浪费，而异步编程允许程序在等待某些操作（如 I/O 操作、网络请求等）完成的同时继续执行其他任务。因此，与同步编程相比，异步编程能够提高程序的响应性、优化资源利用率、提升系统吞吐量并改善用户体验。

## 异步编程的核心概念

### 阻塞与非阻塞
阻塞式操作必须等待完成才能继续，而非阻塞式操作可以立即返回，不等待完成。这个操作指的是**系统调用**，阻塞式的系统调用往往会导致进程被挂起，这是异步编程被提出的首要目的。

### 同步与异步
同步操作按顺序执行，一个完成后才能执行下一个，而异步操作可以并发执行，不按固定顺序。异步编程与同步编程的显著区别在于，同步编程的程序执行顺序与代码出现的顺序一致，程序的代码往往体现出严格的先后顺序，而异步编程的代码往往依赖于高阶函数的特性，使得代码的执行顺序被更改。

### 并发与并行
并发是多个任务交替执行，而并行是多个任务同时执行。并发是操作系统营造的进程间同时执行的假象，通过时间片轮转的方式让多个进程交替使用 CPU，而并行则需要多核 CPU 的支持，真正实现多个任务的同时执行。

### 协程（Coroutine）
协程是一种由程序员控制调度的轻量级线程，具有可**在任意位置暂停和恢复**的特点，相比传统线程它具有资源消耗低、并发性能高和编程模型简单的优势。在现代的应用中，协程是实现异步编程的重要手段，它可以让程序在发生了 IO 调用的时候不再等待，而是切换上下文去执行其他任务。

## 常见的异步编程模式
异步编程的本体就是回调函数，程序不再等待一个费时操作的执行，而是事先注册一个回调函数，当操作执行完成后由等待者调用回调函数处理结果。

### 回调函数（Callback）
回调函数是最基础的异步编程模式，通过将后续操作封装成函数作为参数传递给异步操作，在操作完成后执行。例如在 Node.js 中读取文件时，可以通过回调函数处理读取结果：

```javascript
fs.readFile('file.txt', (err, data) => {
    if (err) throw err;
    console.log(data);
});
```

### Promise
Promise 是一种更高级的异步编程模式，它通过状态机来实现异步操作的状态管理，并提供了一个链式调用风格的 API 解决了回调地狱的问题，提供了更清晰的错误处理机制。Promise 对象代表一个异步操作的最终完成或失败，并返回其结果：

```javascript
fetch('https://api.example.com/data')
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
```

### async/await
Async/Await 是异步编程中中断当前函数的语法糖，使得程序员可以像书写同步代码那样来书写异步代码，让异步代码的编写和阅读更加直观和简单。它本质上是一种更优雅的异步编程方式。

```javascript
async function getData() {
    try {
        const response = await fetch('https://api.example.com/data');
        const data = await response.json();
        console.log(data);
    } catch (error) {
        console.error(error);
    }
}
```

### 事件驱动
事件驱动模式通过事件发射器和监听器实现异步通信，当特定事件发生时触发相应的处理函数。这种模式特别适合处理用户交互和系统事件，如 Node.js 中的 EventEmitter：

```javascript
const EventEmitter = require('events');
const emitter = new EventEmitter();

emitter.on('event', (data) => {
    console.log('Received:', data);
});

emitter.emit('event', 'Hello World');
```

### 协程
协程是实现异步编程的一个重要方法，Python 通过 async/await 语法实现协程，而 Go 语言则通过 goroutine 实现轻量级线程。

```python
async def fetch_data():
    response = await aiohttp.get('https://api.example.com/data')
    data = await response.json()
    return data
```

```go
func fetchData() {
    go func() {
        resp, err := http.Get("https://api.example.com/data")
        if err != nil {
            log.Fatal(err)
        }
        defer resp.Body.Close()
        // 处理响应
    }()
}
```
