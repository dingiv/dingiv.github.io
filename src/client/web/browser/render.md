# 渲染周期
本文梳理浏览器从用户输入 URL 到页面完全加载、可交互、最终关闭的完整生命周期过程。理解这个周期对前端性能优化有直接帮助——你知道每一毫秒花在了哪里，就知道该在哪个环节下手。

## 导航与网络请求
用户在地址栏输入 URL 后，浏览器首先判断输入内容是合法 URL 还是搜索关键词，对不完整的地址补全协议前缀（如 `http://`），同时根据历史记录和书签提供补全建议。如果地址栏判断这是一个即将跳转的 URL，还会提前进行 DNS 预取和预连接以缩短后续等待时间。

确定目标 URL 后，浏览器检查当前页面是否需要触发 `beforeunload` 事件（例如用户有未保存的表单数据），然后决定是复用已有渲染进程还是为新页面创建一个独立进程。现代 Chrome 采用面向服务（Service-Oriented）的架构，不同的站点可能分配到不同的渲染进程，以实现站点隔离（Site Isolation）。

接下来进入网络请求阶段。浏览器首先将域名解析为 IP 地址，查找顺序依次是浏览器 DNS 缓存、操作系统 DNS 缓存、路由器 DNS 缓存，最后才向本地 DNS 服务器发起递归查询。拿到 IP 后，通过 TCP 三次握手建立连接，如果是 HTTPS 还要完成 TLS 握手协商加密参数并验证服务器证书，之后才发送 HTTP 请求并接收响应。整个过程中，DNS 预解析（`dns-prefetch`）和预连接（`preconnect`）是前端可以主动优化的手段，在 HTML 的 `<head>` 中提前声明目标域名可以让浏览器提前完成 DNS 查询和 TCP 连接的建立。

```html
<link rel="dns-prefetch" href="//cdn.example.com">
<link rel="preconnect" href="//cdn.example.com" crossorigin>
```

## 解析
浏览器接收到服务器返回的 HTML 响应后，解析阶段正式开始。HTML 解析器将字节流按照指定编码（如 UTF-8）转换为字符，再通过标记化（Tokenization）将字符流分解为一个个标记（开始标签、结束标签、文本内容等），最终构建出 DOM 树。这个过程是增量的——HTML 解析器不需要等待整个文档下载完毕，而是边下载边解析，这也是为什么浏览器能做到"渐进式渲染"，用户在页面还没完全加载时就能看到部分内容。

在构建 DOM 树的同时，预加载扫描器（Preload Scanner）会快速扫描 HTML 中已经到达的部分，尽早发现引用的外部资源（CSS、JavaScript、字体、图片等）并提前发起请求。这个机制对首屏加载速度影响很大——如果关键 CSS 被放在 `<body>` 底部，预加载扫描器虽然能发现它，但浏览器会等到 CSSOM 构建完成后才开始渲染，首屏依然是空白的。所以将关键 CSS 内联到 `<head>` 中或者用 `<link>` 标签尽早声明，是工程中常见的优化手段。

JavaScript 的处理比较特殊。当 HTML 解析器遇到 `<script>` 标签时，默认会暂停 DOM 构建转而下载并执行脚本，因为脚本可能通过 `document.write` 等方法修改 DOM。这种阻塞行为是首屏加载慢的常见原因。工程上有几种应对方式：给 `<script>` 标签加 `async` 属性让脚本异步加载和执行、加 `defer` 属性让脚本在 DOM 解析完成后按顺序执行、或者将脚本放在 `<body>` 末尾。`async` 和 `defer` 的区别在于执行时机——`async` 是下载完立即执行，不保证顺序，适合独立的第三方脚本（如统计代码）；`defer` 是在 DOMContentLoaded 之前按声明顺序执行，适合有依赖关系的业务脚本。

CSS 解析与 HTML 解析并行进行，将 CSS 规则解析为 CSSOM（CSS Object Model）。CSSOM 构建完成后，浏览器才能知道每个 DOM 节点应该呈现什么样式。CSS 是渲染阻塞的——浏览器在 CSSOM 构建完成前不会渲染任何内容，这也是为什么"关键渲染路径"优化的核心就是尽快让 CSS 加载完毕并构建 CSSOM。相比之下，`@import` 嵌套导入会让 CSS 加载变成串行的，应该尽量避免。

```javascript
// DOM 树的简化结构
{
  nodeType: 9, // Document
  children: [
    {
      nodeType: 1, // Element
      tagName: 'html',
      children: [
        { nodeType: 1, tagName: 'head' },
        { nodeType: 1, tagName: 'body' }
      ]
    }
  ]
}
```

## 渲染管线
DOM 树和 CSSOM 构建完成后，浏览器将两者合并生成渲染树（Render Tree）。渲染树只包含需要显示的 DOM 节点及其计算样式，`<head>`、`<script>`、`display: none` 的元素等不会出现在渲染树中。这个阶段涉及样式计算（Style Calculation），浏览器需要确定每个节点的最终样式，考虑 CSS 优先级、继承、层叠等因素。

### 布局
布局阶段计算渲染树中每个节点的精确几何信息——位置和尺寸。浏览器从渲染树的根节点开始，自上而下遍历计算每个元素的盒模型尺寸，处理正常流、浮动、定位等布局模式。布局的计算结果是一棵布局树，其中每个节点都记录了在视口坐标系中的位置和大小。

布局是一个 potentially expensive 的操作，因为它依赖整个 DOM 树的结构——任何一个节点的尺寸或位置变化都可能需要重新计算其祖先或后续兄弟节点的布局。这也是为什么频繁读取布局属性（如 `offsetWidth`、`scrollTop`）和频繁修改样式交替进行会导致严重的性能问题，这种现象叫做强制同步布局（Forced Synchronous Layout）。

### 绘制与合成
布局完成后，浏览器将渲染树中的内容转换为屏幕上的像素。绘制阶段会生成绘制记录（Paint Records），按照特定的层叠顺序将每个元素的视觉效果记录下来。现代浏览器为了优化绘制性能，会将内容分为多个合成层（Compositing Layer），具有 `transform`、`opacity`、`will-change` 等属性的元素通常会被提升为独立的合成层。每个合成层单独进行光栅化（Rasterization）——将矢量信息转换为位图像素，这个过程可以利用 GPU 加速。

合成阶段将各个合成层按照正确的顺序和变换关系组合成最终的屏幕画面。合成的好处在于，如果只有某个合成层的属性发生变化（比如通过 `transform` 移动一个元素），浏览器只需要重新光栅化那个层并重新合成，而不需要触发整个页面的布局和绘制。这就是为什么 `transform` 和 `opacity` 的动画性能优于修改 `width`、`height` 或 `top`、`left`——前者只触发合成，后者会触发完整的布局和绘制。

```javascript
// 高性能动画 vs 低性能动画
// 高性能：只触发合成
element.style.transform = 'translateX(100px)'
element.style.opacity = '0.5'

// 低性能：触发布局 + 绘制 + 合成
element.style.left = '100px'
element.style.width = '200px'
```

整个渲染管线的流程可以概括为：样式计算 → 布局 → 绘制 → 分层 → 光栅化 → 合成。其中布局和绘制是最昂贵的部分，优化的核心思路就是尽量减少这两个阶段的触发频率。

## 交互与渲染更新
页面首次渲染完成后进入交互阶段。用户点击、滚动、输入等操作会产生事件，事件通过事件循环驱动主线程执行 JavaScript 回调。如果 JavaScript 回调修改了 DOM 或样式，就会触发重新渲染。浏览器会尽量做增量更新——只重新布局受影响的部分，只重绘变化的区域，只重新合成变化的层。

但并非所有 DOM 操作触发的更新代价都相同。根据触发的管线阶段不同，渲染更新分为几个层次：

1. 最小代价：只触发合成。通过修改 `transform` 和 `opacity`，不需要布局和绘制，浏览器直接在合成阶段完成更新。
2. 中等代价：触发重绘（Repaint）。修改颜色、背景等不影响布局的属性，不需要重新计算布局，但需要重新绘制受影响区域。
3. 最大代价：触发回流/重排（Reflow）。修改尺寸、位置等影响布局的属性，需要重新执行布局和绘制。

```javascript
// 批量 DOM 操作避免多次回流
const fragment = document.createDocumentFragment()
for (let i = 0; i < 1000; i++) {
  const el = document.createElement('div')
  el.textContent = `Item ${i}`
  fragment.appendChild(el)
}
document.getElementById('container').appendChild(fragment)

// 避免强制同步布局：先批量读取，再批量写入
requestAnimationFrame(() => {
  const width = element.offsetWidth // 读取布局属性
  elements.forEach(el => {
    el.style.width = width + 'px' // 批量写入
  })
})
```

在实际工程中，批量 DOM 操作、避免读写交替、使用 DocumentFragment、通过 `class` 切换而非逐个修改内联样式，都是减少回流次数的常见做法。

## 页面生命周期事件
浏览器提供了一系列生命周期事件，让开发者可以在页面加载、交互、关闭的各个阶段执行逻辑。理解这些事件的触发顺序对于编写可靠的初始化代码和清理逻辑至关重要。

DOMContentLoaded 在 DOM 解析完成后触发，此时外部样式表、图片等资源可能还在加载，但 DOM 树已经可用。`load` 事件在页面及所有依赖资源（图片、样式表、iframe 等）全部加载完成后触发。两者的区别是：如果你只需要操作 DOM 节点，在 DOMContentLoaded 时就可以开始；如果你需要获取图片的尺寸或进行依赖资源的计算，则需要等到 `load`。

`visibilitychange` 事件在页面可见性变化时触发，常用于在页面不可见时暂停动画或定时任务以节省资源。`beforeunload` 在用户即将离开页面时触发，可用于提醒用户保存未提交的表单数据，但不应在 `beforeunload` 中执行耗时操作——浏览器可能在没有等待回调完成的情况下就关闭页面。

```javascript
document.addEventListener('DOMContentLoaded', () => {
  // DOM 可用，可以操作节点
})

window.addEventListener('load', () => {
  // 所有资源加载完毕，可以获取图片尺寸等
})

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    // 暂停非必要操作，节省 CPU 和电量
  } else {
    // 恢复操作
  }
})
```

## 性能度量与优化工具
衡量页面加载性能的核心指标是 Web Vitals：LCP（Largest Contentful Paint）衡量最大内容元素的渲染时间，FID（First Input Delay）衡量首次交互响应延迟，CLS（Cumulative Layout Shift）衡量页面加载过程中的视觉稳定性。LCP 受关键渲染路径的影响最大，优化手段包括内联关键 CSS、延迟非关键 CSS、预加载关键资源。CLS 主要由没有预设尺寸的图片和动态注入的 DOM 引起，给图片设置 `width` 和 `height` 属性、避免在已有内容上方插入新元素是常见的优化方式。

浏览器开发者工具的 Performance 面板可以录制完整的页面加载过程，直观展示每个阶段的耗时分布——导航、DNS、TCP、请求、响应、解析、布局、绘制、合成。配合 Performance API 可以在代码中精确测量各阶段的耗时：

```javascript
window.addEventListener('load', () => {
  const perf = window.performance.timing
  const pageLoadTime = perf.loadEventEnd - perf.navigationStart
  const domReadyTime = perf.domComplete - perf.domLoading
  console.log(`页面加载: ${pageLoadTime}ms, DOM处理: ${domReadyTime}ms`)
})
```

## 关闭与资源清理
当用户关闭标签页、导航到新页面或浏览器关闭时，页面进入关闭阶段。浏览器依次触发 `beforeunload` 和 `unload` 事件，执行开发者注册的清理函数，保存会话历史和滚动位置以供前进/后退导航使用，并可能通过 `navigator.sendBeacon` 发送统计数据。随后浏览器终止 JavaScript 执行，取消待处理的网络请求，释放图形和内存资源。

需要注意的是，`unload` 事件中不应执行异步操作或依赖回调完成，因为现代浏览器（尤其是 Chrome 的 bfcache 策略）可能在 `unload` 回调完成之前就冻结或丢弃页面。推荐使用 `visibilitychange` 事件配合 `sendBeacon` API 来处理关闭时的数据上报，`sendBeacon` 能保证请求在页面关闭后仍能发送。
