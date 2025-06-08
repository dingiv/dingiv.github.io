---
title: 浏览器
order: 40
---

# 浏览器
浏览器是现代 Web 应用的运行环境，与普通的 GUI 程序不同，浏览器包含着众多的组件，需要经过复杂的实现逻辑。

## 架构与组成
现代浏览器采用多进程架构，并至少包含几个核心组件
|组件|功能|例子|
|-|-|-|
|web 渲染引擎|实现 HTML 和 CSS 语言，解析二者的代码，生成渲染指令，绘制 HTML 元素|Blink(下层基于 Skia)|
|JavaScript 引擎|实现 JavaScript 语言|V8|
|网络协议客户端|实现各种网络协议的客户端逻辑|Http(s), WebSocket|
|客户端数据存储|实现 Web 客户端数据持久化标准|Cookie, LocalStorage, IndexedDB|
|浏览器主应用|封装和整合所有组件，并提供浏览器自身的 GUI|chromium|

除此之外，还有一些其他的模块需要处理：例如：实现 Webgl/WebGPU 图形 API 标准、Web 媒体资源编码和压缩、HTML 的 SVG 语法扩展等等

### 主流浏览器及其内核
| 特性    | Gecko           | Blink (Chrome) | WebKit (Safari) |
| ----- | --------------- | -------------- | --------------- |
| 所属项目  | Mozilla Firefox | Chromium 项目    | Apple Safari    |
| JS 引擎 | SpiderMonkey    | V8             | JavaScriptCore  |
| 样式计算  | Stylo（Rust）     | Blink（C++）     | WebCore         |
| 多进程支持 | 支持              | 支持             | Safari 支持       |
| 渲染架构  | WebRender + GPU | Skia           | CoreGraphics    |
| 开源许可  | MPL             | BSD-like       | LGPL/MPL 兼容     |


### 事件循环（Event Loop）

1. **调用栈（Call Stack）**：记录函数调用顺序
2. **任务队列（Task Queue）**：存储待执行的宏任务
3. **微任务队列（Microtask Queue）**：存储待执行的微任务
4. **事件循环（Event Loop）**：不断检查调用栈是否为空，为空则先执行所有微任务，然后执行一个宏任务

## 浏览器缓存

### 浏览器缓存机制

1. 强缓存：浏览器在本地缓存中查找资源，如果找到且未过期，则直接使用缓存资源，否则继续请求服务器。
2. 协商缓存：浏览器在本地缓存中查找资源，如果找到且未过期，则向服务器发送请求，服务器验证资源是否更新，如果更新则返回新资源，否则返回304状态码，浏览器使用本地缓存资源。

### 浏览器缓存策略

1. Cache-Control：控制缓存的策略，如max-age、no-cache、no-store等。
   1. max-age：指定缓存的过期时间，单位为秒。
   2. no-cache：表示每次请求都需要验证缓存是否有效。
   3. no-store：表示不缓存资源。
2. Expires：指定缓存的过期时间，格式为GMT时间。
3. ETag：资源的唯一标识符，用于验证缓存是否有效。
4. Last-Modified：资源的最后修改时间，用于验证缓存是否有效。
5. If-None-Match：客户端发送的ETag值，服务器根据该值判断资源是否更新。
6. If-Modified-Since：客户端发送的Last-Modified值，服务器根据该值判断资源是否更新。
7. Cache-Control优先级高于Expires，ETag优先级高于Last-Modified。
8. 浏览器会根据Cache-Control、Expires、ETag、Last-Modified等字段来判断资源是否有效，如果有效则使用缓存资源，否则向服务器请求新资源。

## 浏览器存储

浏览器提供了多种存储机制，用于在客户端保存数据。

### Cookie

- 最早的浏览器存储机制
- 容量限制：通常为4KB
- 会随HTTP请求一起发送
- 可设置过期时间、域名范围等
- 通过设置HttpOnly和Secure提高安全性

### LocalStorage

- 永久存储机制，除非手动清除
- 容量限制：通常为5MB
- 不会随HTTP请求发送
- 仅支持字符串存储
- 同源访问限制

### SessionStorage

- 会话级存储，关闭标签页后清除
- 容量限制：通常为5MB
- 不会随HTTP请求发送
- 仅支持字符串存储
- 同源且同标签页访问限制

### IndexedDB

- 结构化存储机制，支持复杂数据类型
- 容量大，通常>50MB
- 异步API，不阻塞主线程
- 支持事务和索引
- 同源访问限制

### Web Storage API

- localStorage和sessionStorage的统一接口
- 提供setItem、getItem、removeItem、clear等方法
- 支持storage事件监听变化

## 浏览器安全

### 浏览器安全机制

1. 同源策略：限制一个origin（协议+域名+端口）的文档或脚本如何与另一个源的资源进行交互。同源策略可以防止恶意文档，通过恶意脚本窃取数据。
2. 跨站脚本攻击（XSS）：攻击者通过在网页中插入恶意脚本，窃取用户数据或执行恶意操作。
3. 跨站请求伪造（CSRF）：攻击者通过诱导用户点击恶意链接或表单，以用户身份执行恶意操作。
4. 内容安全策略（CSP）：通过设置HTTP头部的Content-Security-Policy，限制网页可以加载和执行的资源，防止XSS攻击。
5. HTTPS：通过加密通信，防止中间人攻击和数据窃取。
6. Cookie安全：通过设置HttpOnly和Secure标志，防止Cookie被JavaScript访问和窃取。
7. 安全沙箱：浏览器为每个标签页或iframe创建一个独立的沙箱环境，限制其访问其他标签页或iframe的资源。
8. 安全更新：定期更新浏览器和插件，修复已知的安全漏洞。
9. 安全测试：定期进行安全测试，发现和修复安全漏洞。

### 跨域解决方案

1. **CORS（跨域资源共享）**
   - 服务器设置Access-Control-Allow-Origin等响应头
   - 支持简单请求和预检请求
   - 可控制是否发送凭证信息

2. **JSONP**
   - 利用script标签不受同源策略限制
   - 只支持GET请求
   - 有安全风险

3. **代理服务器**
   - 在同源服务器上设置代理
   - 转发请求到目标服务器
   - 返回响应给客户端

4. **WebSocket**
   - 建立持久连接
   - 不受同源策略限制
   - 支持双向通信

## 浏览器性能优化

### 关键渲染路径优化

1. **减少关键资源**
   - 减少阻塞渲染的CSS和JavaScript
   - 内联关键CSS
   - 异步加载非关键JavaScript

2. **减少资源大小**
   - 压缩HTML、CSS、JavaScript
   - 使用Gzip/Brotli压缩
   - 图片优化

3. **减少请求数量**
   - 合并CSS和JavaScript文件
   - 使用CSS Sprite
   - 使用字体图标或SVG

4. **优化加载顺序**
   - CSS放在head中
   - JavaScript放在body底部
   - 使用async/defer属性

### 渲染性能优化

1. **减少回流（Reflow）**
   - 批量修改DOM
   - 使用document fragment
   - 避免频繁读取布局信息

2. **减少重绘（Repaint）**
   - 使用CSS transform和opacity代替修改位置和可见性
   - 使用will-change提示浏览器
   - 合理使用GPU加速

3. **帧率优化**
   - 使用requestAnimationFrame
   - 避免长任务阻塞主线程
   - 使用Web Workers分担计算密集型任务

### 网络优化

1. **资源预加载**
   - preload关键资源
   - prefetch可能需要的资源
   - preconnect提前建立连接

2. **HTTP优化**
   - 使用HTTP/2多路复用
   - 使用HTTP/3 QUIC协议
   - 合理设置缓存策略

3. **CDN加速**
   - 使用CDN分发静态资源
   - 选择离用户最近的节点
   - 使用多CDN提供冗余

## 浏览器开发者工具

现代浏览器提供了强大的开发者工具，帮助开发者调试和优化Web应用。

### Elements（元素）

- 检查和修改DOM结构
- 实时编辑CSS样式
- 查看事件监听器
- 断点调试DOM变化

### Console（控制台）

- 输出调试信息
- 执行JavaScript代码
- 查看错误和警告
- 使用console API

### Network（网络）

- 监控网络请求
- 分析资源加载时间
- 查看HTTP头信息
- 模拟网络条件

### Performance（性能）

- 记录和分析页面性能
- 查看CPU和内存使用情况
- 识别性能瓶颈
- 分析帧率和渲染时间

### Memory（内存）

- 分析内存使用情况
- 查找内存泄漏
- 查看内存分配
- 生成堆快照

### Application（应用）

- 管理本地存储
- 查看和修改Cookie
- 管理Service Worker
- 查看Web应用清单

### Security（安全）

- 检查HTTPS证书
- 识别混合内容问题
- 查看内容安全策略
- 分析安全漏洞

## 浏览器兼容性

### 检测和解决兼容性问题

1. **特性检测**
   - 检测浏览器是否支持特定功能
   - 根据支持情况提供不同实现
   - 避免使用用户代理检测

2. **Polyfill**
   - 为旧浏览器提供新功能的模拟实现
   - 只在需要时加载
   - 使用现代工具自动添加

3. **渐进增强**
   - 从基本功能开始构建
   - 逐步添加高级特性
   - 确保核心功能在所有浏览器中可用

4. **工具支持**
   - Babel转译现代JavaScript
   - PostCSS处理CSS兼容性
   - Autoprefixer自动添加厂商前缀
   - Browserslist定义目标浏览器

### 常见兼容性资源

- **Can I use**：查询特性兼容性数据
- **MDN Web Docs**：详细的API兼容性信息
- **Modernizr**：特性检测库
- **core-js**：JavaScript标准库polyfill



