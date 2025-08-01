# 架构
现代浏览器是复杂的软件系统，通常由以下几个核心组件构成：排版引擎、JavaScript 引擎、网络模块、存储模块、浏览器主应用等。在运行时，浏览器会启动多个进程，负责运行不同的组件和功能。

## 多进程架构
现代浏览器采用多进程架构以提高稳定性、安全性和性能。以 Google Chrome 为例，主要进程包含有：

1. 浏览器主进程
   浏览器程序的入口，负责呈现浏览器自身的 UI 界面，
   - 控制浏览器的主用户界面
   - 管理标签页和插件
   - 处理用户权限（如下载请求）
   - 协调其他进程

2. 渲染进程

   - 每个标签页通常都有自己的渲染进程
   - 负责标签页内网页的渲染
   - 运行 JavaScript 和处理 DOM
   - 在沙箱环境中运行，提高安全性
3. GPU 进程（GPU Process）

   - 处理 GPU 任务，加速渲染
   - 跨标签页共享

4. 网络进程（Network Process）

   - 处理网络请求
   - 实现网络栈
   - 管理 HTTP 缓存

5. 插件进程（Plugin Process）

   - 运行浏览器插件（如 Flash）
   - 隔离插件代码，防止影响浏览器稳定性

6. 存储进程（Storage Process）（较新版本）
   - 管理浏览器的数据存储
   - 处理文件系统访问

### 多进程架构的优势

1. **稳定性提升**

   - 一个标签页崩溃不会影响整个浏览器
   - 进程隔离避免资源冲突

2. **安全性增强**

   - 沙箱限制渲染进程的系统访问权限
   - 进程间通信受到控制

3. **性能优化**
   - 多核 CPU 上可并行处理
   - 内存占用虽然增加，但可实现更精细的资源管理

### 进程间通信（IPC）

浏览器进程之间通过 IPC（进程间通信）机制进行交互：

- 使用消息传递模式
- 通过共享内存传输大块数据
- 实现通信频道进行请求/响应模式

## 现代浏览器架构发展趋势

1. **服务化（Service-oriented Architecture）**

   - Chrome 正在将浏览器功能拆分为独立服务
   - 每个服务可以在不同进程中运行
   - 允许更灵活的资源分配

2. **站点隔离（Site Isolation）**

   - 不同站点在不同渲染进程中运行
   - 更严格的跨源边界
   - 缓解类似 Spectre 的侧信道攻击

3. **进程模型优化**

   - 在低内存设备上合并进程
   - 在高性能设备上使用更多进程
   - 动态调整进程分配

4. **WebAssembly 沙箱**
   - 更细粒度的代码隔离
   - 提高性能的同时保持安全性
