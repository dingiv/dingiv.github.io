# 性能优化
Web 性能优化不是一个通用清单，而是一个"量化指标 → 建立基准 → 定位瓶颈 → 局部微操 → 线上监控"的闭环工程。没有度量就没有优化，所有的优化决策都应该基于明确的性能指标和真实的瓶颈定位，而非凭感觉调参。

## Core Web Vitals
Google 提出的 Core Web Vitals 是当前业界衡量 Web 性能的核心指标体系，涵盖了加载体验、交互响应和视觉稳定性三个维度。理解这些指标的底层物理意义，是做性能优化的第一步。

### LCP（最大内容渲染）
LCP（Largest Contentful Paint）衡量的是页面主要内容首次渲染完成的时间，通常是首屏最大的图片、文本块或视频元素完成加载并渲染到屏幕上的时刻。LCP 小于 2.5 秒为优秀，超过 4 秒则需要优化。

LCP 元素的资源加载是最大的瓶颈。常见的优化手段是提高 LCP 元素的下载优先级。在 HTML 的 `<head>` 中通过 `<link rel="preload" as="image" href="hero.jpg">` 预加载关键图片，配合 `fetchpriority="high"` 属性告诉浏览器这个资源优先级高于页面中的其他静态资源。同时确保 LCP 元素的 CSS 不被阻塞——如果首屏大图是通过 CSS 背景图实现的，它的加载优先级低于 `<img>` 标签，这种情况下应改用 `<img>` 标签或使用 `<link rel="preload" as="image">` 提升优先级。

服务端渲染（SSR）是另一个重要的 LCP 优化手段。传统 SPA 的首屏需要等 JS 下载、解析、执行后才能渲染内容，中间是一段"白屏等待期"。SSR 在服务端直接生成 HTML，浏览器收到 HTML 即可渲染首屏内容，LCP 时间等于 HTML 传输时间而非 JS 下载执行时间。React Server Components（RSC）进一步推进了这个思路，服务端组件不生成 JS，只生成 HTML，减少了客户端 JS 的体积。

### CLS（累积布局偏移）
CLS（Cumulative Layout Shift）衡量的是页面加载过程中视觉内容的意外位移。常见的场景包括：图片加载完成后撑开布局导致下方内容跳动、异步加载的广告位突然插入页面、Web 字体加载后文本回流等。CLS 小于 0.1 为优秀，超过 0.25 会导致用户操作困难。

CLS 优化的核心原则是为所有异步加载的内容预留空间。对图片和视频，始终在 HTML 中声明明确的 `width` 和 `height` 属性或使用 CSS 的 `aspect-ratio`，浏览器在资源加载前就能计算出元素的空间占位，避免加载后的布局跳动。对广告位和异步组件，使用 CSS `min-height` 或骨架屏占位。对动态注入的内容（如通过 AJAX 加载的评论列表），在容器元素上预留足够的空间。

字体加载也是 CLS 的常见来源。`@font-face` 加载完成后文本从回退字体切换为设计字体，字符宽度的变化可能导致整行甚至整段文本回流。解决方案是使用 `font-display: optional`（如果回退字体可接受，直接放弃字体加载）或在字体加载完成前隐藏受影响的文本（`font-display: block` 配合较短的 block period），避免可见的文本跳动。

### INP（交互延迟）
INP（Interaction to Next Paint）在 2026 年已经全面取代 FID（First Input Delay）成为衡量交互响应能力的核心指标。INP 观察用户在整个页面访问期间的所有交互（点击、键盘、触摸），取其中最差的交互响应时间作为最终得分。INP 小于 200ms 为优秀，超过 500ms 需要优化。

INP 劣化的根本原因是主线程被长任务（Long Task，执行时间超过 50ms 的任务）阻塞。当用户点击按钮时，浏览器需要等待当前正在执行的长任务完成后才能处理点击事件的回调，这段时间就是交互延迟。

定位长任务的工具是 Chrome DevTools 的 Performance 面板。录制一段用户交互后，在火焰图（Flame Chart）中寻找标红的长任务条，查看是哪个函数调用链导致了主线程的长时间占用。常见的长任务来源包括大数据量的 DOM 操作、复杂的计算逻辑、同步的 JSON 解析等。

解决长任务的标准手法是任务切片。将一个耗时数百毫秒的大任务拆分为多个小于 16ms 的小任务（与一帧的时间对齐），通过 `requestIdleCallback`、`MessageChannel` 或 `setTimeout(fn, 0)` 让出主线程控制权，让浏览器有机会处理用户的交互事件和渲染更新，然后在下一个空闲时段继续执行剩余任务。Web Workers 则适合处理计算密集型且与 DOM 无关的任务，将计算过程完全移出主线程。

```javascript
// 任务切片：将大任务拆分为小任务，避免阻塞主线程
async function processInChunks(items, chunkSize, processor) {
  for (let i = 0; i < items.length; i += chunkSize) {
    const chunk = items.slice(i, i + chunkSize)
    processor(chunk)
    // 让出主线程，允许浏览器处理交互事件和渲染
    await new Promise(resolve => setTimeout(resolve, 0))
  }
}
```

## 首屏加载优化
首屏加载优化需要按照"网络传输 → 资源体积 → 执行时机"的管线流转来系统性思考，而不是零散地堆砌优化手段。

### 网络传输层
HTTP/2 的多路复用消除了浏览器对同一域名的并发连接限制（HTTP/1.1 下通常为 6 个），一个 TCP 连接上可以并行传输多个请求和响应。HTTP/3（QUIC）进一步解决了 TCP 的队头阻塞问题——HTTP/2 在一个 TCP 连接上多路复用，但 TCP 层的丢包重传会导致所有流被阻塞，HTTP/3 基于 UDP 实现的 QUIC 协议让每个流独立恢复，不存在队头阻塞。

缓存策略的设计需要在请求频率和缓存有效性之间取得平衡。第三方基础库（React、Vue、工具库等）版本变动极少，使用 `Cache-Control: max-age=31536000, immutable` 强缓存一年，配合内容哈希的文件名——只有文件名变了浏览器才会请求新版本。入口文件 `index.html` 使用 `no-cache` 配合 `ETag` 协商缓存，确保每次都能拿到最新版本的资源引用。这种"长期缓存 + 入口不缓存"的组合是前端部署的标准实践。

CDN 将静态资源分发到离用户最近的边缘节点，大幅缩短网络传输的物理距离。在 CDN 配置中，需要注意缓存层和源站的一致性——当 CDN 节点缓存了旧版本的资源但用户获取到了新版本的 HTML 时，会出现资源引用不匹配的 404 错误。工程上通常在资源文件名中嵌入内容哈希来规避这个问题。

### 资源体积层
代码分割（Code Splitting）是控制首屏加载体积的核心手段。通过路由级别的动态导入（`import()`），首屏只下载当前路由必需的代码，其余路由的代码在用户导航时按需加载。构建工具（Vite 的 `manualChunks`、Webpack 的 `splitChunks`）将框架代码、工具库、业务代码分离为独立的 chunk，变化频率不同的代码分开缓存——框架代码几乎不变，业务代码频繁变动，混在一起会导致业务改动让框架缓存全部失效。

压缩方面，服务端配置 Brotli 或 Gzip 压缩传输文本资源（JS、CSS、HTML、SVG），通常能减少 60%-80% 的传输体积。Brotli 的压缩率比 Gzip 高 15%-25%，但压缩耗时更长，适合静态资源在构建时预压缩后直接部署，而非实时压缩。

图片优化在首屏体积中占比极大。现代方案包括：根据设备 DPR 使用 `<picture>` 和 `srcset` 提供不同分辨率的图片，使用 WebP/AVIF 格式替代 JPEG/PNG（AVIF 在同等画质下比 JPEG 小 50%），以及按视口尺寸裁剪图片避免下载远超屏幕尺寸的大图。

### 关键渲染路径
浏览器从收到 HTML 到渲染首屏，经历了一系列有序的步骤：解析 HTML 构建 DOM 树 → 解析 CSS 构建 CSSOM 树 → 合并为渲染树 → 计算布局 → 绘制。这个流程中，阻塞渲染的关键资源（通常是 CSS）越多，首屏渲染越晚。

CSS 放在 `<head>` 中尽早加载，JS 使用 `defer` 或 `async` 属性避免阻塞 HTML 解析。`defer` 告诉浏览器在 HTML 解析完成后、DOMContentLoaded 之前按顺序执行脚本；`async` 让脚本在下载完成后立即执行，不保证顺序也不等待 HTML 解析完成。对于不参与首屏渲染的第三方脚本（统计、客服、广告），使用 `async` 或动态注入 `<script>` 标签，将它们从首屏关键路径中完全剥离。

## 虚拟滚动
长列表渲染是前端性能的经典问题。当列表包含上千甚至上万条数据时，直接将所有节点渲染到 DOM 中会导致严重的性能问题——DOM 节点数量过多，每次滚动都触发大量节点的布局计算和绘制，帧率急剧下降。

虚拟滚动的核心原理是只渲染当前可视区域内的少量 DOM 节点。通过监听容器的滚动事件，计算 `scrollTop` 和每项高度，动态切片数据数组（`slice(startIndex, endIndex)`），只将可视区间内的条目渲染为 DOM 节点。为了保持滚动条的正确长度和滚动时的视觉连续性，通常在容器内添加一个高度等于总列表高度的空间撑开元素（spacer），然后通过 `transform: translateY()` 将可视节点定位到正确的物理位置。

```javascript
function getVisibleRange(scrollTop, itemHeight, viewportHeight, totalItems) {
  const startIndex = Math.floor(scrollTop / itemHeight)
  const endIndex = Math.min(
    startIndex + Math.ceil(viewportHeight / itemHeight) + 1,
    totalItems
  )
  return { startIndex, endIndex }
}
```

当列表项高度不固定时（如朋友圈、带评论的卡片），虚拟滚动的实现复杂度显著增加。核心策略是"预估高度 + 动态修正"：初始化时为每条数据设定一个预估高度，计算出大致的总高度撑开滚动条；当节点真正渲染到 DOM 后，通过 `ResizeObserver` 或 `getBoundingClientRect()` 捕获其真实高度，动态更新一个位置索引缓存表（Offset Map），记录每条数据到顶部的累计偏移量。后续滚动时通过二分查找这个索引表快速定位可视区间。为了防止高频滚动导致 `ResizeObserver` 密集触发引发掉帧，需要配合 `requestAnimationFrame` 进行节流。

## 图片懒加载
图片懒加载的目标是推迟视口外图片的加载时机，减少首屏的网络请求和带宽消耗。

早期的实现方案是在 `scroll` 事件中通过 `getBoundingClientRect()` 判断图片是否进入视口。这个方案有明显的性能问题：`scroll` 事件触发极其频繁，且 `getBoundingClientRect()` 的调用会强制浏览器执行同步布局重算（Forced Synchronous Layout），在滚动期间反复触发会导致严重的卡顿。

现代方案是使用 `IntersectionObserver`。它基于浏览器底层的异步机制，只在元素与视口发生交叉时触发回调，不需要在主线程上频繁计算位置。被观察的图片在进入视口时才设置 `src` 属性发起加载，离开视口后可以选择清除 `src` 释放内存。

更简洁的方案是使用浏览器原生的 `loading="lazy"` 属性。在 `<img>` 标签上声明后，浏览器自动推迟视口外图片的加载，无需任何 JavaScript。现代浏览器（Chrome、Firefox、Edge、Safari）均已支持该属性。在实际项目中，首屏可见的关键图片不使用懒加载（确保 LCP 不受影响），首屏以下的图片统一使用原生懒加载，这是成本最低、效果最好的图片优化策略。

## 内存泄漏排查
内存泄漏是区分"真有实战经验"和"纯背八股"的分水岭。典型的泄漏场景包括三种：组件销毁时未清除定时器（`setInterval`、`setTimeout`）或事件监听器（`addEventListener`），导致闭包持有组件上下文无法被 GC 回收；在 JavaScript 中保存了 DOM 节点的引用后通过 `removeChild` 从页面删除了该节点（Detached DOM），DOM 节点虽然不可见但仍占据内存；全局变量或缓存不断增长但从不清理，随着应用运行时间增长内存持续膨胀。

排查内存泄漏的标准流程基于 Chrome DevTools 的 Memory 面板。第一步，在页面初始化时拍下堆快照（Heap Snapshot）作为基准。第二步，反复执行疑似泄漏的操作（如打开关闭弹窗十次），拍下第二张快照。第三步，点击垃圾桶图标强制触发垃圾回收，拍下第三张快照。第四步，使用 Memory 面板的 Comparison（对比）功能，将第三张快照与第一张进行比对，筛选出增长异常的对象。重点查看 `Detached HTMLDivElement`（游离 DOM 节点）和特定组件的构造函数实例。通过 Retainers（持有者引用链）面板向上追溯，可以精确定位是哪个全局变量、闭包或未清理的事件监听器锁死了这块内存，然后在代码中对应的清理逻辑处进行修复。

在日常开发中，养成在组件销毁生命周期（如 Vue 的 `onUnmounted`、React 的 `useEffect` 返回清理函数）中主动清除副作用（移除事件监听、清除定时器、断开 WebSocket 连接）的习惯，可以避免绝大多数内存泄漏问题。使用弱引用（`WeakMap`、`WeakSet`）替代强引用来缓存对象，也能让 GC 在需要时自动回收这些对象。
