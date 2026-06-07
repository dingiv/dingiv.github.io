# Node.js 性能调优

Node.js 基于 V8 引擎和单线程事件循环模型，这决定了它的性能瓶颈与传统的多线程服务端（Java、Go）有本质区别。Java 和 Go 可以通过增加线程数直接利用多核 CPU，而 Node.js 的单线程意味着 CPU 密集型任务会直接阻塞事件循环，导致所有请求排队等待。理解事件循环的运行机制和 Node.js 的资源限制，是做性能调优的前提。

## 事件循环与阻塞

Node.js 的事件循环是整个运行时的核心调度机制。每一次事件循环迭代（tick）按顺序执行：处理定时器回调 → 执行 I/O 回调 → 处理内部事件 → 执行 `setImmediate` 回调 → 关闭回调。在这个循环中，所有回调都在同一个线程上执行——如果某个回调函数执行时间过长（超过几十毫秒），后续的所有回调（包括新的网络请求处理、定时器触发、文件读取回调等）都会被阻塞。

这就是为什么 Node.js 中"不要阻塞事件循环"是一条铁律。判断阻塞的标准是：如果一个同步操作在所有 CPU 核心上都耗时超过几毫秒（通常以 10ms 作为警戒线），就应该考虑将其移出主线程。常见的阻塞源包括：大 JSON 文件的 `JSON.parse()`/`JSON.stringify()`、大数组的排序和遍历、加密计算（bcrypt 哈希计算在慢速因子较高时可能耗时数百毫秒）、正则表达式的回溯匹配等。

判断阻塞的量化标准是事件循环延迟（Event Loop Delay）。优秀的延迟应控制在 5ms 以内，超过这个值就说明主线程被某个同步任务绑架了。`eventLoopLag` 模块通过计算定时器的实际触发延迟来量化阻塞程度——如果预设 5ms 间隔的定时器实际触发间隔变成了 50ms，说明事件循环被阻塞了约 45ms。

对于无法完全移出主线程但耗时较长的同步计算，可以使用时间切片（Time Slicing）技术，利用 `setImmediate()` 或 `process.nextTick()` 将一个大循环拆分为多段，每执行一小段就让出主线程，让 I/O 请求和事件处理有机会穿插执行。这种做法不改变总计算量，但将阻塞打散为多次小暂停，避免事件循环的延迟突破临界值。

```javascript
// 时间切片：将大任务拆分为多段，每段执行后让出主线程
function processLargeArray(items, chunkSize, processor, callback) {
  let index = 0
  function processChunk() {
    const end = Math.min(index + chunkSize, items.length)
    for (; index < end; index++) {
      processor(items[index])
    }
    if (index < items.length) {
      setImmediate(processChunk) // 让出主线程，下一轮继续
    } else {
      callback()
    }
  }
  processChunk()
}
```

诊断事件循环阻塞的工具有两种。`node --inspect` 启动应用后，通过 Chrome DevTools 的 Performance 面板录制，可以直观地看到主线程上的长任务。在线上环境中，可以通过 `chrome://inspect` 远程连接到 Node.js 进程进行实时分析，无需在服务器上安装额外的工具。

## Worker Threads 与计算卸载

Node.js 10.5 引入的 `worker_threads` 模块是处理 CPU 密集型任务的标准方案。每个 Worker 线程拥有独立的 V8 实例和事件循环，与主线程通过 `postMessage` 通信（基于结构化克隆算法，支持大部分 JS 类型的序列化传输）。

Worker Threads 的适用场景很明确：文件压缩、图片处理、加密运算、大数据排序等计算密集型任务。但需要注意，创建 Worker 的开销不低（需要启动一个完整的 V8 实例），不适合为每个请求创建一个 Worker。工程上的做法是维护一个 Worker 线程池（Worker Pool），复用固定数量的 Worker 处理任务队列中的计算请求。

```javascript
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads')

if (isMainThread) {
  // 主线程：创建 Worker 处理计算任务
  const worker = new Worker(__filename, {
    workerData: { filePath: './huge-data.json' }
  })
  worker.on('message', (result) => console.log('处理结果:', result))
  worker.on('error', (err) => console.error('Worker 错误:', err))
} else {
  // Worker 线程：执行 CPU 密集型计算
  const data = JSON.parse(require('fs').readFileSync(workerData.filePath, 'utf-8'))
  const result = heavyComputation(data)
  parentPort.postMessage(result)
}
```

对于已有的 C/C++ 编写的计算密集型库（如图片压缩的 sharp、加密的 openssl），更高效的方案是使用 Node.js 的 N-API 编写原生插件，直接在 C++ 层执行计算，绕过 V8 的单线程限制。sharp 就是将 libvips（C 图像处理库）封装为 Node.js 原生插件的典型案例，性能远超纯 JS 实现的图片处理库。

## 内存管理

Node.js 默认的堆内存限制远小于浏览器中的 V8——64 位系统下约为 1.4GB，32 位系统下约为 0.7GB。这个限制可以通过 `--max-old-space-size` 参数调整（如 `node --max-old-space-size=4096 app.js`），但调大内存并不是解决内存问题的正确方式，它只是推迟了 OOM（Out of Memory）的崩溃时间。

Node.js 中的内存泄漏与浏览器端有相似的原理，但也有服务端特有的场景。Express/Koa 中最常见的是请求级别的中间件在闭包中累积数据。如果中间件在每次请求中将处理结果推入一个模块级的数组或 Map 而不清理，内存会随请求量线性增长。

```javascript
const requestHistory = []

app.use((req, res, next) => {
  requestHistory.push({
    url: req.url,
    headers: req.headers, // 每个请求的完整 headers 都被保留
    timestamp: Date.now()
  })
  next()
})
// 高流量服务中，requestHistory 会在数小时内耗尽可用内存
```

排查 Node.js 内存泄漏有两种方式。线上环境通过 `node --inspect` 启动后，在 Chrome 浏览器中访问 `chrome://inspect`，可以直接将 DevTools 的 Memory 面板远程挂载到 Node.js 进程上进行在线分析。离线排查可以使用 `heapdump` 模块在运行时生成堆快照文件（.heapsnapshot），然后用 Chrome DevTools 打开。

服务端堆快照对比的操作流程与浏览器端一致，但在触发泄漏的方式上有所不同——线上环境需要使用压测工具（如 `autocannon` 或 `ab`）模拟高并发请求，而非手动点击 UI。具体步骤是：服务刚启动时拍下快照 1 作为基准；使用压测工具发起数千次并发请求后拍下快照 2；对比两个快照中 Closure（闭包）和 Array 类型的增量，定位泄漏对象后通过 Retainers 引用链追溯到具体的代码位置。

服务端特有的泄漏场景还有两种值得注意。全局闭包缓存在业务代码中很常见——为了方便，开发者会在模块顶层用一个全局数组或 Map 缓存用户 Session 或请求结果，但没有设置过期和淘汰机制（TTL）。正确的做法是使用 Redis 等外部缓存服务，或使用支持 LRU 淘汰的内存库（如 `lru-cache`）。未清除的 EventEmitter 监听器在 Node.js 服务端同样常见——高频创建 EventEmitter 实例并频繁调用 `emitter.on()` 注册监听器，但在请求结束后忘记 `emitter.off()`，导致监听器回调闭包持有的数据持续累积。Node.js 默认会对单个 EventEmitter 的监听器数量设置 10 个的上限警告（`MaxListenersExceededWarning`），这个警告出现时就是排查泄漏的最佳时机。

V8 的 GC 日志也是诊断内存问题的有效手段。通过 `--trace-gc` 参数启动 Node.js，GC 的每一次触发（包括新生代 Scavenge 和老生代标记清除）都会输出到控制台。如果老生代 GC 的频率异常增高，说明大量对象在新生代存活后被晋升到老生代，可能存在内存泄漏或对象生命周期过长的问题。当内存占用接近 V8 堆上限时，Full GC 会导致几十到几百毫秒的服务停顿，在高并发场景下会直接表现为请求超时。结合 `--trace-gc-verbose` 可以看到每次 GC 回收的具体字节数，进一步判断是哪些对象没有被及时回收。

## 集群与进程管理

Node.js 的单进程单线程模型无法利用多核 CPU 的全部算力。在生产环境中，标准做法是通过 `cluster` 模块或进程管理器（PM2）启动多个工作进程，每个进程监听同一个端口，由操作系统内核将网络请求分发到不同的进程（基于 Round-Robin 或操作系统的负载均衡策略）。

```javascript
const cluster = require('cluster')
const os = require('os')

if (cluster.isPrimary) {
  // 主进程：根据 CPU 核心数创建工作进程
  const cpuCount = os.cpus().length
  for (let i = 0; i < cpuCount; i++) {
    cluster.fork()
  }
  cluster.on('exit', (worker) => {
    console.log(`工作进程 ${worker.process.pid} 退出，重启新进程`)
    cluster.fork() // 自动重启
  })
} else {
  // 工作进程：启动 HTTP 服务
  require('./app')
}
```

PM2 在 cluster 模式之上提供了更完善的进程管理能力：进程状态监控、日志管理、零停机重启（graceful reload，先启动新进程再关闭旧进程，避免请求中断）、以及基于 CPU/内存使用率的自动重启策略。在生产环境中，PM2 是 Node.js 应用的标配进程管理工具。

需要注意的是，多进程环境下每个进程都有独立的内存空间，如果应用中使用了进程级缓存（如内存中的 Session 存储），不同进程之间的缓存无法共享。解决方案是使用 Redis 等外部存储作为共享缓存，或将缓存放在主进程中通过 `postMessage` 与工作进程通信——但后者增加了通信开销，通常只在读多写少的场景下适用。

## I/O 优化

Node.js 的 I/O 模型本身就是非阻塞的——文件读取、网络请求、数据库查询等操作都通过 libuv 线程池异步执行，不会阻塞事件循环。但 I/O 的异步不等于性能足够好，以下几个方面仍需关注。

文件系统 I/O 方面，`fs.promises` 的 API 默认使用 libuv 的线程池处理文件操作，线程池默认大小为 4（可通过 `UV_THREADPOOL_SIZE` 环境变量调整）。当同时有大量文件 I/O 和 DNS 查询（DNS 查询也使用 libuv 线程池）时，线程池可能成为瓶颈——4 个大文件读取任务占满线程池后，第 5 个请求只能排队等待，造成严重的 I/O 阻塞假象。增大线程池大小是直接的优化手段，但有一个关键细节：`UV_THREADPOOL_SIZE` 必须在加载任何 I/O 模块之前设置，否则不会生效。

```javascript
// 必须在加载任何 I/O 模块之前执行
process.env.UV_THREADPOOL_SIZE = 64
```

在业务代码中，严格禁止使用任何带有 `Sync` 后缀的 API。`fs.readFileSync()`、`crypto.createHashSync()` 等同步 API 会让整个 Node.js 主线程进入完全阻塞状态——在文件读取完成前，即使服务器并发进来数百个网络请求，它们全部无法被接收和处理。必须使用 Promise 版本的 `fs.promises.readFile()` 或回调版本替代。这个禁令在 Code Review 中应该作为红线执行。

数据库连接是另一个关键点。使用连接池管理数据库连接，避免每次请求都创建和销毁连接（TCP 三次握手、身份认证等开销）。连接池的大小需要根据业务特征调整——过大的连接池会增加数据库的内存开销和连接管理负担，过小则会导致请求排队。通常设置为 CPU 核心数的 2-10 倍，具体取决于数据库的处理能力和查询的平均耗时。

网络层方面，Node.js 的 HTTP 服务在高并发场景下需要注意连接数管理。每个 TCP 连接都会占用一个文件描述符，Linux 系统默认的文件描述符上限（`ulimit -n`）通常是 1024，生产环境需要调高到数万。Node.js 的 `http.Server` 默认使用 `keep-alive` 保持连接，避免频繁的 TCP 握手开销，但也意味着空闲连接会持续占用文件描述符。通过设置 `server.keepAliveTimeout` 和 `server.headersTimeout` 合理控制连接的空闲超时时间，防止文件描述符被空闲连接耗尽。

## 流式处理

Node.js 的 Stream 模块是处理大体积数据的核心工具。当一个接口需要处理大文件上传下载、大量数据库查询结果的聚合、或实时数据推送时，将整个数据加载到内存中再处理是不可行的——一个 1GB 的文件意味着至少 1GB 的堆内存占用，在并发场景下很容易触发 OOM。

Stream 的核心思想是将数据分割为小块（chunk），每个 chunk 独立处理和传输，同一时刻内存中只保留一个 chunk 的数据。Node.js 内置了四种流类型：Readable（可读流）、Writable（可写流）、Duplex（双工流）、Transform（转换流）。通过 `pipe()` 方法可以将多个流串联成处理管线，数据从上游自动流向下游，背压（Backpressure）机制确保当下游处理速度跟不上时，上游暂停数据发送，防止内存溢出。

```javascript
const fs = require('fs')
const { pipeline } = require('stream/promises')
const { createGzip } = require('zlib')

// 将大文件流式压缩后写入目标文件，内存占用恒定
await pipeline(
  fs.createReadStream('./large-data.log'),
  createGzip(),
  fs.createWriteStream('./large-data.log.gz')
)
```

在 HTTP 服务中，使用 Stream 返回大体积响应也是标准做法。`res.write()` 和 `res.end()` 本身就是一个 Writable 流的接口，将 Readable 流 pipe 到 `res` 可以实现流式响应，无需将完整响应体缓存到内存中。这对于大文件下载、SSE（Server-Sent Events）推送、数据库结果流式返回等场景至关重要。

## 安全与性能陷阱

### 正则表达式拒绝服务（ReDoS）

Node.js 的正则匹配引擎是同步执行的，这决定了它既是性能敏感点也是安全风险点。如果正则表达式中包含了嵌套量词（如 `(a+)+`、`(a|a)+`）这类容易触发回溯爆炸的模式，面对恶意构造的特定输入字符串时，匹配过程会陷入指数级的时间复杂度，直接让 CPU 锁死在 100%，这种攻击被称为 ReDoS（Regular Expression Denial of Service）。

```javascript
// 危险模式：嵌套量词在特定输入下触发灾难性回溯
const dangerous = /^(a+)+b$/
dangerous.test('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaac')
// CPU 耗时从毫秒级暴涨到数十秒甚至分钟级
```

防范 ReDoS 需要从两个层面入手。编写正则时避免嵌套量词和过于宽松的模糊匹配模式，必要时使用 `safe-regex` 等工具在 CI 中检测正则的时间复杂度。在用户输入作为正则匹配目标的高风险场景中，可以对输入长度做截断限制，或使用 `RegExp.prototype` 配合超时机制（Node.js 20+ 支持 `--regex-backtrack-limit` 和 `--regex-stack-limit` 参数）兜底。

## 诊断与监控

## 诊断与监控

线上 Node.js 应用的性能诊断依赖于 APM（Application Performance Monitoring）工具和自建的可观测性体系。Node.js 内置了 `diagnostics_channel` 模块，支持对 HTTP 请求、DNS 查询、文件系统操作等关键事件进行埋点。基于此构建的 APM 工具（如 OpenTelemetry、Datadog、New Relic）可以实时追踪每个请求的处理链路、耗时分解和资源占用。

`process.cpuUsage()` 返回当前进程的 CPU 时间消耗（用户态和系统态），可以用于计算 CPU 利用率。`process.memoryUsage()` 返回堆内存、堆外内存、数组和缓冲区占用的详细数据。定期采样这两个指标并上报到监控系统，可以构建出应用的资源使用趋势图。当 CPU 利用率持续接近 100% 时，通常意味着存在计算密集型的任务需要卸载到 Worker 线程；当堆内存持续增长且 GC 无法有效回收时，则需要排查内存泄漏。

Node.js 的 `perf_hooks` 模块提供了高精度的时间测量能力（`performance.now()` 精度到微秒级），用于在代码中对关键路径进行耗时打点。与简单的 `Date.now()` 相比，`performance.now()` 不受系统时钟调整的影响，测量结果更准确。

在告警策略上，除了 CPU 和内存的绝对值告警，更有效的是基于趋势的告警。当 GC 暂停时间占比超过请求总耗时的 20% 时，即使内存绝对值尚未触及上限，也应该发出告警——这通常是内存泄漏的早期信号。当事件循环延迟（Event Loop Lag）持续超过 100ms 时，说明主线程被频繁阻塞，需要定位阻塞源并优化。
