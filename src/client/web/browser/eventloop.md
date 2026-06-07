# 事件循环
事件循环是浏览器实现异步 IO 的核心机制。JavaScript 是单线程语言，如果所有操作都同步执行，那么一个网络请求就会阻塞整个页面的渲染和交互。事件循环通过将异步任务分配到不同的队列中，在主线程空闲时依次取出执行，使得单线程的 JavaScript 也能处理并发操作而不阻塞 UI。

## 运行时架构

浏览器的运行时由三个核心区域协作组成，理解它们的分工是理解事件循环的前提。

### 调用栈

JavaScript 引擎维护一个调用栈来追踪当前正在执行的函数上下文。当代码调用一个函数时，引擎将该函数的执行上下文压入调用栈顶部；函数执行完毕后，上下文从栈中弹出。调用栈遵循后进先出的原则，栈顶的上下文始终是当前正在执行的代码。在执行上下文中，引擎还会维护词法环境来管理变量作用域和闭包。

### 宿主环境

调用栈只能执行同步代码，但浏览器本身是用 C++ 实现的多线程程序。当主线程遇到 `setTimeout`、`fetch`、DOM 事件监听等异步操作时，并不是让主线程去等待，而是将这些任务交给浏览器内核的后台线程去处理——定时器线程负责计时、网络线程负责请求和响应、GUI 线程负责渲染。主线程在交接任务后立即继续往下执行同步代码，不会被阻塞。

### 任务队列

后台线程完成任务后（定时器到时、网络响应到达等），会将对应的回调函数移入任务队列，等待主线程来取。任务队列又分为宏任务队列和微任务队列两种，优先级不同。

```javascript
function foo() {
  bar()
}

function bar() {
  console.log('bar')
}

foo()
```

上面的代码中，`foo` 先被压入栈中，`foo` 内部调用 `bar`，`bar` 被压入栈顶执行，`bar` 执行完毕弹出后，`foo` 继续执行并最终弹出。整个过程调用栈中最多只有一个函数在执行，这就是单线程的本质。

当调用栈中所有同步代码执行完毕后，引擎并不会就此停止，而是进入事件循环，从任务队列中取出待执行的任务继续执行，形成了一个持续运转的循环。

## 任务队列

事件循环中有两类优先级不同的任务队列：宏任务队列和微任务队列。

### 宏任务

宏任务（Macrotask）是由浏览器提供的宿主 API 产生的异步任务。每次事件循环从宏任务队列中取出一个任务执行。常见的宏任务来源包括：

1. 定时器回调：`setTimeout` 和 `setInterval` 的回调函数，当定时器触发时回调被加入宏任务队列。
2. IO 回调：网络请求的响应处理、文件读取等异步 IO 操作完成后的回调。
3. UI 事件回调：用户点击、输入、滚动等交互事件的回调函数。
4. `requestAnimationFrame` 回调：浏览器下一帧渲染前执行。
5. `MessageChannel` 和 `postMessage` 消息。

`setTimeout(fn, 0)` 并不是立即执行，而是将 `fn` 加入宏任务队列的末尾，等待当前执行栈清空后的下一次事件循环来执行，所以它的实际延迟至少是 4ms（浏览器对嵌套定时器的最小延迟限制）。

### 微任务

微任务（Microtask）的优先级高于宏任务。每当一个宏任务执行完毕后，事件循环会先清空所有微任务，然后才进入下一个宏任务。这意味着微任务可以打断宏任务之间的执行间隙，获得更快的响应。

1. `Promise` 的 `then`、`catch`、`finally` 回调：当 Promise 状态变更时，对应的回调被加入微任务队列。
2. `MutationObserver` 回调：监听 DOM 变动的观察者回调。
3. `queueMicrotask` 手动添加的微任务。

```javascript
console.log('1')
setTimeout(() => console.log('2'), 0)
Promise.resolve().then(() => console.log('3'))
console.log('4')
// 输出顺序：1, 4, 3, 2
```

上面的例子中，同步代码 `1` 和 `4` 先执行，Promise 回调属于微任务在当前宏任务结束后立即执行，`setTimeout` 回调属于宏任务排在下一个事件循环中执行。

### 微任务的连锁反应

微任务的一个关键特性是：在微任务执行过程中产生的新的微任务也会在当前微任务队列清空前被依次执行，直到微任务队列为空。

```javascript
Promise.resolve().then(() => {
  console.log('micro 1')
  Promise.resolve().then(() => {
    console.log('micro 2')
  })
}).then(() => {
  console.log('micro 3')
})
// 输出顺序：micro 1, micro 2, micro 3
```

第一个 `then` 回调执行时产生了新的微任务，这个新微任务会排在当前 `then` 链的第二个 `then` 回调之前，因为在同一次微任务清空中，新产生的微任务会追加到队列尾部并被立即处理。这个特性在实际工程中需要注意，避免在微任务中不断产生新的微任务导致长时间占用主线程。

## 事件循环流程

一个完整的事件循环周期包含以下步骤：

1. 执行当前调用栈中的同步代码，将其视为一个宏任务。
2. 调用栈清空后，检查微任务队列，依次取出并执行所有微任务，直到队列为空。
3. 浏览器判断是否需要执行渲染更新（基于 `requestAnimationFrame` 调度）。
4. 取出宏任务队列中的下一个任务，重复上述过程。

用伪代码表示这个流程：

```javascript
while (true) {
  // 执行一个宏任务
  const task = macroTaskQueue.dequeue()
  execute(task)

  // 清空所有微任务
  while (microTaskQueue.size > 0) {
    const microtask = microTaskQueue.dequeue()
    execute(microtask)
  }

  // 渲染（如果需要）
  if (shouldRender()) {
    render()
  }
}
```

这个循环在浏览器进程的整个生命周期中持续运行，只要页面不关闭，事件循环就不会停止。

## UI 事件

浏览器的事件系统是事件循环的重要组成部分。底层操作系统将键盘、鼠标、触摸等硬件事件传递给浏览器进程，浏览器进程经过解析和封装后将其分发给对应的渲染进程，最终通过事件循环将回调放入任务队列中执行。

### 事件传播

当一个事件发生在某个 DOM 元素上时，浏览器按照三个阶段依次传播事件：

1. 捕获阶段：事件从 `window` 开始，沿着 DOM 树向下传递到目标元素的父节点。`addEventListener` 的第三个参数传入 `true` 可以在捕获阶段处理事件。

2. 目标阶段：事件到达目标元素本身，按照注册顺序执行该元素上的事件监听器。

3. 冒泡阶段：事件从目标元素开始，沿着 DOM 树向上冒泡到 `window`。大多数事件监听器默认在冒泡阶段处理，这样父元素可以统一处理子元素触发的同类事件，避免在每个子元素上都绑定监听器。

```javascript
const parent = document.getElementById('parent')
const child = document.getElementById('child')

parent.addEventListener('click', () => console.log('parent capture'), true)
parent.addEventListener('click', () => console.log('parent bubble'))
child.addEventListener('click', () => console.log('child capture'), true)
child.addEventListener('click', () => console.log('child bubble'))

// 点击 child 元素，输出顺序：
// parent capture -> child capture -> child bubble -> parent bubble
```

捕获到冒泡的传播路径使得事件委托成为可能——将事件监听器绑定在父元素上，通过 `event.target` 判断实际触发的子元素，从而用一个监听器处理多个子元素的事件。这是前端工程中减少事件绑定数量、优化内存占用的常用技巧。

### 事件循环与 UI 响应

由于 JavaScript 的执行和 UI 渲染共享同一个主线程，长时间运行的同步代码会阻塞事件循环，导致页面无法响应用户的点击、滚动等操作。用户感知到的是页面卡顿甚至浏览器弹出"页面未响应"的对话框。

在工程实践中，将耗时计算拆分为小块，通过 `setTimeout(fn, 0)` 或 `requestIdleCallback` 将后续计算推迟到下一个事件循环周期中执行，可以在每个任务之间让出主线程处理 UI 事件和渲染更新。不过 `setTimeout(fn, 0)` 有最小 4ms 的延迟限制，`requestIdleCallback` 则可能因为浏览器一直忙碌而迟迟得不到执行。更精确的做法是使用 `MessageChannel`：它将回调作为宏任务加入队列，但不受定时器最小延迟的限制，适合需要高频让出主线程的场景。

```javascript
// 使用 MessageChannel 拆分长任务
const channel = new MessageChannel()
const items = [...] // 大量待处理数据
let index = 0

channel.port2.onmessage = () => {
  const start = Date.now()
  while (index < items.length && Date.now() - start < 5) {
    process(items[index++])
  }
  if (index < items.length) {
    channel.port1.postMessage(null) // 将剩余工作推入下一个宏任务
  }
}

channel.port1.postMessage(null)
```

Vue 2 的 `nextTick` 在某些场景下就使用了 `MessageChannel` 而非 `setTimeout`，正是为了获得更快的响应速度。

## 渲染与合成

浏览器在事件循环中的渲染时机并非每次循环都执行。现代浏览器采用一种启发式策略来决定是否需要在当前事件循环中进行渲染更新：通常以 60fps 为目标（约 16.6ms 一帧），在每帧的事件循环中，如果检测到 DOM 变动或样式变化触发了布局或重绘的需要，就会在微任务清空后执行渲染管线。

渲染管线的流程包括：样式计算 → 布局（Layout） → 分层 → 绘制（Paint） → 合成（Composite）。其中布局和绘制是比较昂贵的操作，频繁触发会导致性能问题。`requestAnimationFrame` 的回调在渲染之前执行，适合执行动画逻辑；`requestIdleCallback` 在渲染之后、浏览器空闲时执行，适合执行不紧急的后台任务。

```javascript
// 使用 requestAnimationFrame 实现平滑动画
let start = null
function animate(timestamp) {
  if (!start) start = timestamp
  const progress = timestamp - start
  element.style.transform = `translateX(${Math.min(progress / 10, 300)}px)`
  if (progress < 3000) {
    requestAnimationFrame(animate)
  }
}
requestAnimationFrame(animate)
```

## async/await 与事件循环

`async/await` 是 Promise 的语法糖，`await` 会暂停当前 async 函数的执行，将后续代码作为微任务加入队列。这一点在理解事件循环执行顺序时很重要。

```javascript
async function foo() {
  console.log('foo start')
  await bar()
  console.log('foo end')
}

async function bar() {
  console.log('bar start')
  await Promise.resolve()
  console.log('bar end')
}

foo()
// 输出顺序：foo start -> bar start -> bar end -> foo end
```

`foo` 中 `await bar()` 会等待 `bar` 执行完成，而 `bar` 中的 `await Promise.resolve()` 会将后续代码作为微任务推迟执行。整个调用链的执行依赖微任务队列来驱动，而不是像同步递归那样直接在调用栈中完成。在工程中需要注意 async 函数的 `await` 位置，避免将耗时的同步代码放在 `await` 之前导致阻塞后续微任务的执行。

## 框架中的事件循环：Vue 的 nextTick

理解事件循环后就能看透许多框架的内部设计。Vue 的 `$nextTick`（Vue 3 中是 `nextTick` 函数）就是一个典型例子。

当你修改 Vue 的响应式数据时，Vue 并不会立刻同步更新真实 DOM——如果数据被高频修改（例如在循环中连续修改多个属性），同步更新 DOM 会导致严重的重复渲染。Vue 的做法是将 DOM 更新操作推入一个微任务队列中批量执行。而 `$nextTick(callback)` 的本质就是通过 `Promise.then`（或降级到 `MessageChannel` / `setTimeout`）把传入的回调注册为一个微任务，排在 Vue 内部 DOM 更新的微任务之后。这保证了在 `$nextTick` 回调中取到的永远是更新后的 DOM 结构。

```javascript
// Vue 3 中 nextTick 的简化原理
function nextTick(fn) {
  return Promise.resolve().then(fn)
}

// 使用场景
const app = Vue.createApp({ ... })
app.mount('#app')

app.data.message = 'updated'
// 此时 DOM 还没更新

Vue.nextTick(() => {
  // 这里拿到的已经是更新后的 DOM
  console.log(document.querySelector('#text').textContent) // 'updated'
})
```

类似的，React 的 `flushSync` 用于同步刷新状态更新，而默认的批量更新机制（batching）同样利用了事件循环中的微任务来合并多次 `setState`。理解了事件循环，就能理解为什么框架的异步更新策略是合理的，以及在需要精确控制 DOM 更新时机时该用什么工具。

## Node.js 中的事件循环

Node.js 的事件循环机制与浏览器有相似之处，都基于宏任务和微任务的模型，但宏任务的类型和优先级有所不同。Node.js 的事件循环包含多个阶段：

1. timers：执行 `setTimeout` 和 `setInterval` 的回调。
2. pending callbacks：执行系统级别的回调。
3. idle, prepare：内部使用的阶段。
4. poll：获取新的 IO 事件，执行 IO 相关的回调。
5. check：执行 `setImmediate` 的回调。
6. close callbacks：执行关闭事件的回调。

Node.js 的微任务在每次事件循环阶段切换之间都会被清空，而不是像浏览器那样只在宏任务结束后清空。此外，`process.nextTick` 的优先级高于其他微任务，会在当前操作完成后、进入下一个事件循环阶段之前立即执行。这些差异在编写跨端可移植的异步代码时需要特别注意。
