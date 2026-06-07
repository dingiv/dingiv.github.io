# 内存泄漏

内存泄漏是指程序中已不再使用的内存由于某种原因无法被垃圾回收器（GC）释放，随着应用运行时间增长，内存占用持续膨胀，最终导致页面卡顿甚至浏览器崩溃。在单页应用（SPA）中，内存泄漏的影响尤其严重——用户可能在同一个页面上停留数小时甚至一整天，即使每次泄漏的量很小，累积起来也会让页面变得不可用。

## V8 的垃圾回收机制

理解内存泄漏之前，需要先理解 V8 引擎的垃圾回收机制。V8 采用分代回收策略，将堆内存分为新生代（Young Generation）和老生代（Old Generation）两个区域。

新生代存放存活时间短的对象，容量较小（通常几 MB 到几十 MB），采用 Scavenge 算法进行回收。Scavenge 将新生代内存一分为二（From 空间和 To 空间），新对象分配在 From 空间，当 From 空间满时触发 GC，将存活对象复制到 To 空间，然后两个空间角色互换。复制过程中存活时间超过一定阈值的对象会被晋升到老生代。

老生代存放存活时间长的对象，容量较大（通常几百 MB），采用标记-清除（Mark-Sweep）和标记-整理（Mark-Compact）算法。标记阶段从根节点（Root，包括全局对象、当前调用栈中的局部变量等）出发，遍历所有可达对象并标记为活跃状态；清除阶段回收未被标记的对象。为了解决标记-清除后产生的内存碎片问题，标记-整理算法会在清除后对存活对象进行内存搬运，将它们紧凑排列。

V8 的 GC 是自动运行的，开发者不需要手动触发。但 V8 的判断标准很简单：一个对象从根节点出发是否可达。如果某个对象被任何可达对象引用（直接或通过闭包链间接引用），GC 就不会回收它。内存泄漏的本质就是：本应被释放的对象被某个仍然存活的引用意外持有，导致 GC 认为它仍然"可达"。

## 常见泄漏场景

### 未清除的事件监听与定时器

这是最常见的泄漏类型。在组件挂载时注册了事件监听器或启动了定时器，但组件销毁时忘记清除。事件监听器的回调函数通常通过闭包捕获了组件的上下文变量，只要监听器还在 DOM 上，闭包以及它引用的所有变量都不会被 GC 回收。

```javascript
class Modal {
  constructor() {
    this.data = new Array(10000).fill('leak data')
    // 注册了全局事件监听，但从未移除
    window.addEventListener('resize', this.handleResize)
  }
  handleResize = () => {
    // 闭包引用了 this.data
    console.log(this.data.length)
  }
}
```

这个 Modal 实例即使从 DOM 中移除，由于 `handleResize` 仍然挂载在 `window` 上，`this` 指向的整个实例（包括内部的 `data` 数组）都不会被 GC 回收。修复方式是在组件销毁时显式移除监听器：`window.removeEventListener('resize', this.handleResize)`。

定时器同理。`setInterval` 的回调通过闭包持有组件上下文，如果 `clearInterval` 未被调用，即使组件已销毁，回调及其闭包引用的数据都会持续存在。更隐蔽的是 `setTimeout` —— 开发者可能认为 `setTimeout` 是一次性的不需要清除，但如果在某个高频操作中不断创建 `setTimeout`（且在组件销毁前未清除），这些定时器的回调闭包会同时存活在内存中。

在框架层面，Vue 的 `onUnmounted` 生命周期和 React 的 `useEffect` 清理函数是处理这些副作用的正确位置。Vue 3 的组合式 API 中，`onUnmounted(() => window.removeEventListener(...))` 应该与监听器的注册成对出现；React 中 `useEffect` 返回的函数会在组件卸载时执行，用于清理副作用。

### 游离 DOM 节点（Detached DOM）

游离 DOM 节点是指已经从文档树中移除（通过 `removeChild` 或设置父元素的 `innerHTML`），但仍然被 JavaScript 变量引用的 DOM 元素。DOM 节点本身占用的内存不大，但它关联的渲染上下文（如绑定的事件、子节点树）可能不小。

```javascript
const elements = []

function createAndRemove() {
  const div = document.createElement('div')
  div.textContent = 'temporary element'
  document.body.appendChild(div)
  document.body.removeChild(div)
  // removeChild 只是从文档树中移除，但 div 仍被 elements 数组引用
  elements.push(div)
}
```

每次调用 `createAndRemove` 都会在文档树中创建和移除一个节点，但节点被 `elements` 数组持有着，GC 无法回收。随着调用次数增加，`elements` 中积累的 DOM 节点越来越多。排查时在 Chrome DevTools Memory 面板的堆快照中搜索 "Detached"，可以直接找到这些游离节点。

这类泄漏在处理动态列表和弹窗时尤其常见。弹窗组件被关闭后，如果全局的某个数据结构仍然引用着弹窗的 DOM 根节点，整个弹窗及其子节点树都会留在内存中。

### 闭包持有大对象

闭包捕获外部变量后，即使外部函数已经执行完毕，被捕获的变量仍然存活在闭包的作用域中。如果闭包捕获的是一个大对象（如大型数组、DOM 文档片段、完整的响应式状态树），且闭包本身被长期持有，就会造成明显的内存泄漏。

```javascript
function createHandler() {
  const hugeData = new Array(1000000).fill('x')
  return function handler() {
    // 闭包捕获了 hugeData，但 handler 内部只用了 hugeData.length
    return hugeData.length
  }
}
// hugeData 的完整数据被闭包持有，但实际只需要 length
```

这类泄漏的隐蔽之处在于：泄漏发生在闭包捕获变量的那一刻，而不是在闭包被调用的时候。即使 `handler` 函数内部只读取了 `hugeData.length`，V8 的闭包实现会保留整个 `hugeData` 对象的引用，因为理论上闭包可以在任何时候访问 `hugeData` 的任意属性。修复方式是只捕获真正需要的数据，而非整个大对象。

### 无限增长的缓存

在应用中使用全局 Map 或对象作为缓存，但从不清理过期条目，是长时间运行的单页应用中最常见的泄漏来源。例如，一个页面路由缓存机制将每次访问过的路由组件存入 Map，但不限制缓存数量，用户浏览的页面越多，缓存中的组件实例和关联数据越多。

```javascript
const routeCache = new Map()

function loadRoute(path) {
  if (!routeCache.has(path)) {
    routeCache.set(path, buildComponent(path))
  }
  return routeCache.get(path)
}
// routeCache 只增不减，随着路由访问不断膨胀
```

修复方式是给缓存设置上限（如 LRU 策略，保留最近使用的 N 个条目，超出后淘汰最早未使用的条目），或使用 `WeakMap` 替代 `Map`。`WeakMap` 的键是弱引用——当键对象没有其他引用时，该条目会被 GC 自动回收，不会造成泄漏。但 `WeakMap` 的键只能是对象，不能是原始值，这限制了它的使用场景。

### 全局变量和模块顶级变量

在 ES Modules 环境下，模块作用域本身就是一个长期存活的作用域——模块被导入后，其顶层变量在应用运行期间始终存在，不会被 GC 回收。如果开发者在模块顶层使用数组、Map、对象等数据容器存放业务数据，且不设计清理逻辑，这些容器就会随着业务操作的累积不断膨胀。

```javascript
// data-store.ts — 模块顶层变量在整个应用生命周期内存活
const requestLog: Array<{url: string; timestamp: number}> = []
const userSessionCache = new Map<string, unknown>()

export function logRequest(url: string) {
  requestLog.push({ url, timestamp: Date.now() })
  // requestLog 只增不减，应用运行越久占用越大
}

export function cacheSession(userId: string, data: unknown) {
  userSessionCache.set(userId, data)
  // 用户退出登录后未清除对应条目
}
```

这种模式在业务代码中相当常见。开发者习惯在模块顶层定义一个共享的数据容器，各组件和函数直接导入使用，看似方便，但忽视了这些数据的生命周期管理。与组件级的状态（如 React 的 `useState` 或 Vue 的 `ref`）不同，组件销毁时框架会自动回收组件内的状态，但模块顶层的变量不受组件生命周期控制。

这类泄漏的修复思路取决于数据的使用场景。如果数据只在当前会话中需要，应该在用户退出登录或页面卸载时清空容器。如果数据需要跨会话持久化，则应该使用 `localStorage` 或 `IndexedDB` 等持久化存储，而非堆内存。如果数据确实需要在内存中缓存，则需要明确淘汰策略——按时间过期（TTL）、按容量上限淘汰（LRU），或在对应的业务操作完成后主动清除。

更根本的解决方式是尽量避免在模块顶层维护可变的数据容器。将数据的管理权交给状态管理库（如 Pinia、Zustand）或框架的依赖注入机制，让数据的生命周期与组件或应用状态绑定，而不是作为一个孤立的模块级变量长期存在。


### 第三方库集成

在现代 Web 框架中集成第三方库时，一个容易被忽视的泄漏来源是第三方库内部未遵守组件生命周期管理。许多第三方库（图表库、富文本编辑器、动画库、 WebSocket 客户端等）在初始化时会创建内部状态、注册事件监听或建立长连接，但它们的销毁逻辑需要开发者手动调用，框架无法替你完成。

以图表库为例。ECharts 在初始化时创建了一个绑定到 DOM 元素的实例，内部持有对容器的引用、注册了 `resize` 事件监听器、维护了渲染上下文。如果组件销毁时没有调用 `chart.dispose()`，ECharts 的内部状态（包括事件监听器和渲染缓冲区）会持续持有对已移除 DOM 容器的引用，产生游离 DOM 和闭包泄漏。

```jsx
function Dashboard({ data }) {
  const chartRef = useRef(null)

  useEffect(() => {
    const chart = echarts.init(chartRef.current)
    chart.setOption(data)
    // 缺少 chart.dispose() — 组件销毁时 ECharts 实例未被释放
  }, [data])

  return <div ref={chartRef} style={{ width: 600, height: 400 }} />
}
```

富文本编辑器（如 Quill、TipTap、TinyMCE）的情况类似。编辑器在初始化时会创建大量内部 DOM 节点和事件监听器，如果组件卸载时没有调用销毁方法，这些内部状态全部会留在内存中。动画库（GSAP、Lottie）在运行动画时可能通过 `requestAnimationFrame` 持有对 DOM 元素的引用，如果动画未在组件销毁时停止，引用也不会被释放。

WebSocket 客户端和 SSE（Server-Sent Events）连接是另一类常见问题。连接建立后，客户端会持续接收服务端推送的消息，消息回调通常持有业务上下文。如果组件卸载时不断开连接，不仅回调闭包持有的数据不会被释放，连接本身也会持续消耗系统资源。

排查第三方库泄漏的思路与其他泄漏相同——在 Memory 面板中找到泄漏对象的 Retainers 引用链，如果引用链指向了某个第三方库的内部变量或回调，就说明需要在该库的销毁阶段手动调用清理方法。具体需要调用什么方法取决于库的文档——大多数成熟的库都提供了 `destroy`、`dispose`、`disconnect`、`cleanup` 之类的销毁 API，关键是确保在组件的卸载钩子中调用它们。

## 排查方法

### 从现象到假设

内存泄漏通常不是直接被发现的，而是通过间接现象察觉：页面打开时间越长越卡顿，Performance 面板中 JS Heap 曲线呈阶梯式上升且从不回落，或者移动端页面使用一段时间后触发浏览器的内存警告甚至白屏崩溃。

确认泄漏的第一步是建立基线。在 Chrome DevTools 的 Performance 面板中录制一段包含用户典型操作的交互过程，观察 JS Heap 的变化曲线。如果内存在每次执行某个操作后都增长一段但从不回落（呈阶梯状），就可以高度怀疑该操作触发了内存泄漏。记录下具体是哪个操作导致了内存增长，作为后续深入排查的线索。

### 堆快照对比

堆快照对比是定位泄漏对象的标准手段。操作步骤如下：在页面初始状态拍下快照 1 作为基线；反复执行疑似泄漏的操作（如打开关闭弹窗 10 次、切换路由 10 次等），拍下快照 2；点击垃圾桶图标强制触发 GC，拍下快照 3。

使用 Memory 面板的 Comparison 视图，将快照 3 与快照 1 进行对比。面板会列出两张快照之间新增的对象（New）、删除的对象（Deleted）以及数量变化（Delta）。重点关注 Delta 为正且数值较大的对象类型。如果反复打开关闭弹窗 10 次后，某个组件的构造函数实例增加了 10 个，且这些实例没有被删除，就可以确认该组件在销毁时没有被正确清理。

在确认了泄漏的对象类型后，选中其中一个实例，查看底部的 Retainers（持有者引用链）面板。Retainers 从目标对象向上追溯，列出所有持有其引用的变量和闭包，形成一条完整的引用链。引用链的起点通常是某个全局变量、事件监听器回调或闭包捕获变量——这就是需要修复的泄漏点。

```text
示例 Retainers 引用链：
HTMLDivElement → window.eventHandler (closure) → handler.scopeVars → componentInstance.data
```

### Allocation Timeline 追踪

当已知某个操作会导致内存增长但不确定具体是哪段代码分配了内存时，可以使用 Allocation Timeline。启动录制后执行该操作，面板会以时间轴的形式展示内存分配的动态过程。

查看柱状图中与操作时刻对应的蓝色条目（新分配的内存），底部的构造函数列表会显示这些新分配对象的类型和大小。结合源码中的调用关系，可以判断是哪个函数创建了这些对象。与堆快照对比的定位精度不同，Allocation Timeline 更适合用于"缩小排查范围"——先确定泄漏对象的大致类型和创建时机，再用堆快照做精确定位。

## 线上监控

线上环境中的内存泄漏排查比本地困难得多，因为无法在用户的浏览器中打开 DevTools。线上监控依赖的是浏览器暴露的 `performance.memory` API（Chrome 专有）和自研的内存上报机制。

`performance.memory.usedJSHeapSize` 返回当前 JS 堆的使用量。可以在应用中定期采样（如每 30 秒采集一次），将内存数据上报到服务端。结合用户的操作日志（路由切换、弹窗打开关闭等事件），可以在线上复现本地难以复现的泄漏场景。当内存使用量超过预设阈值时，也可以在前端触发告警通知，提醒用户刷新页面释放内存。

Long Tasks API 和 `reportError` 等前端可观测性手段也可以间接反映内存问题——当内存紧张时，GC 的频率和耗时都会增加，Long Tasks 的数量可能上升。虽然这不是直接监控内存的方式，但可以作为辅助信号帮助判断是否存在内存问题。

## 预防策略

预防内存泄漏比事后排查成本低得多。在工程实践中，养成以下习惯可以避免绝大多数泄漏问题。

在组件销毁生命周期中清理所有副作用。Vue 3 的 `onUnmounted`、React 的 `useEffect` 返回函数、Svelte 的 `$effect` 清理函数等框架机制都是为了这个目的设计的。每个在 `mounted`/`onMounted` 中注册的监听器、定时器、WebSocket 连接，都应该在对应的销毁钩子中显式移除。可以将监听器的注册和移除封装为统一的工具函数，减少遗漏的可能性。

对于缓存，明确缓存的生命周期和容量上限。使用 LRU 缓存限制条目数量，或使用 `WeakMap`/`WeakRef` 让 GC 自动管理缓存条目的生命周期。`WeakRef` 是 ES2021 引入的弱引用包装器，可以创建对对象的弱引用并通过 `.deref()` 方法访问原始对象，当原始对象被 GC 回收后 `.deref()` 返回 `undefined`。

在 Code Review 中将内存管理作为审查项之一。重点关注全局事件监听器、闭包捕获大对象、未设限的缓存增长、WebSocket/EventSource 连接未断开等模式。很多内存泄漏不是设计问题，而是开发者在编码过程中遗漏了清理逻辑，通过 Review 可以在合并前拦截。
