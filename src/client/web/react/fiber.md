# Fiber 与调和
Vue 的虚拟 DOM 与 Diff 算法一文介绍了 diff 的核心约束（同层比较 + key 识别身份），React 遵循同样的假设，但在实现架构上有本质区别。Vue 的 diff 是同步的——一次递归调用完成整棵树的比较，不可中断；React 16 引入了 Fiber 架构，将 diff 拆解为可中断的增量工作单元，为并发渲染奠定了基础。

## Fiber 的数据结构

Fiber 是 React 对虚拟 DOM 节点的重新设计。传统的树形结构通过递归遍历（调用栈）来完成 diff，一旦开始就无法中断，当树很深时，长时间的同步计算会阻塞主线程导致页面卡顿。Fiber 将树形结构改为链表结构，每个 Fiber 节点包含三个指针：`child` 指向第一个子节点、`sibling` 指向下一个兄弟节点、`return` 指向父节点。遍历时通过这三个指针按深度优先顺序访问所有节点，不再依赖调用栈，因此可以在任何两个节点之间暂停和恢复。

```
// Fiber 节点的简化结构
{
  type: 'div',           // 对应的 DOM 元素类型或组件
  key: null,             // 列表 diff 时的身份标识
  stateNode: domElement, // 对应的真实 DOM 节点
  return: parentFiber,   // 父 Fiber
  child: firstChild,     // 第一个子 Fiber
  sibling: nextSibling,  // 下一个兄弟 Fiber

  // 调和相关的标记
  alternate: currentFiber, // 指向对应的另一棵树中的 Fiber
  flags: Placement,        // 本次更新需要执行的操作（插入、更新、删除等）
}
```

React 在任意时刻维护着两棵 Fiber 树：当前屏幕上显示的内容对应的 current 树，以及正在构建的 workInProgress 树。diff 过程就是在 workInProgress 树上打标记（哪些节点需要插入、更新、删除），标记完成后一次性提交到真实 DOM 上。这种双缓冲机制让 React 可以随时丢弃 workInProgress 树重新开始（比如更高优先级的更新到来时），而不影响当前屏幕上已经显示的内容。

## 调和流程

调和（Reconciliation）分为两个阶段：render 阶段和 commit 阶段。render 阶段遍历 Fiber 树进行 diff 比较，在 workInProgress 节点上标记需要执行的变更（`flags`），这个过程是可中断的。commit 阶段将标记的变更应用到真实 DOM 上，这个过程是同步的、不可中断的。

render 阶段的工作流程是：从根节点的 `child` 开始深度优先遍历，对每个 Fiber 节点调用 `reconcileChildren` 比较新旧子节点列表。比较过程中，如果新旧节点的 `type` 和 `key` 都相同，则复用旧 Fiber（更新 props 和 state），创建一个新的 workInProgress Fiber 指向同一个 `stateNode`；如果 `key` 相同但 `type` 不同，或者 `key` 不同，则标记旧 Fiber 为删除、创建新 Fiber。遍历顺序是深度优先的交替遍历——先处理 `child`，再处理 `sibling`，最后通过 `return` 回到父节点。每处理完一个 Fiber，React 会检查剩余时间片是否用完，如果用完就暂停并将控制权交还给浏览器处理高优先级工作（如用户输入），下一帧再从暂停的位置恢复。

commit 阶段在 render 阶段完成后同步执行，分为三个子阶段：Before Mutation 阶段读取 DOM 状态（如获取滚动位置、获取焦点元素），Mutation 阶段执行 DOM 变更（插入、删除、更新），Layout 阶段执行副作用（`useLayoutEffect` 回调、类组件的 `componentDidMount`/`componentDidUpdate`）。 Mutation 阶段之前，React 会先处理所有标记为删除的 Fiber 的 `useEffect` 清理函数；Layout 阶段之后，React 会调度 `useEffect` 的执行（异步的，不阻塞浏览器绘制）。

### 列表 Diff

React 的列表 diff 策略比 Vue 2 简单——不做双端比较，而是采用单轮遍历 + key 查找。算法遍历新子节点列表，对每个新节点用 key 在旧子节点中查找：

1. 遍历新子节点，如果能在旧子节点中找到相同 key 且 type 相同的节点，复用该 Fiber 并更新 props。
2. 如果找到了相同 key 但 type 不同的节点，标记旧节点删除、创建新节点（因为元素类型变了无法复用 DOM）。
3. 如果没找到匹配的 key，创建新节点。
4. 遍历完新子节点后，剩余的旧子节点全部标记为删除。

```
旧子节点: [A] [B] [C] [D]
新子节点: [A] [E] [B] [C]

处理过程:
A: key 匹配 → 复用
E: key 未找到 → 创建新节点
B: key 匹配 → 复用，需要移动（旧位置 1 → 新位置 2）
C: key 匹配 → 复用
D: 旧节点未匹配 → 删除
```

React 复用旧 Fiber 时会通过一个 `Map` 结构记录旧子节点的 key 到索引的映射，查找的时间复杂度为 $O(1)$，整体复杂度为 $O(n)$。这个策略比 Vue 2 的双端比较简单，在大多数实际场景中性能差异不大，但在头部插入或列表反转等特定模式下，双端比较的移动操作更少。React 团队认为这种差异在真实应用中可忽略不计，选择了更简单易懂的实现。

### key 的工程实践

React 中 key 的作用与 Vue 完全一致：用唯一标识帮助 diff 算法识别节点的移动关系。用数组索引作为 key 的问题也一样——列表插入、删除时索引会错位，导致 diff 算法无法正确复用 DOM 节点。一个典型的坑是带输入框的列表：用 index 作 key 时，删除中间一项会导致后续所有项的 index 减 1，diff 算法认为是每一项的内容变了而非一项被删除，结果是最后一项的 DOM 被删除、每个已有项被"更新"为新内容，用户输入框中的值全部错乱。

```jsx
// 不推荐：index 作为 key
{list.map((item, index) => <Row key={index} data={item} />)}

// 正确：唯一标识作为 key
{list.map(item => <Row key={item.id} data={item} />)}
```

## 并发渲染

React 18 引入了并发渲染（Concurrent Rendering），这是 Fiber 架构带来的核心能力。在 Fiber 之前，React 的更新是同步的——一次 `setState` 触发整棵组件树的同步 diff 和 DOM 更新，如果树很深或组件很重，用户会明显感知到卡顿。并发渲染允许 React 将更新工作拆分成多个小的时间片，在浏览器每帧的空闲时间中执行，高优先级的更新（如用户输入）可以打断低优先级的更新（如数据列表渲染）。

并发渲染通过 lanes（优先级车道）系统来管理更新的优先级。每个更新被分配一个 lane，优先级从高到低依次是：同步更新（SyncLane，如用户触发的点击事件）、连续输入（InputContinuousLane，如拖拽、滚动）、默认更新（DefaultLane，如数据请求的响应）、过渡更新（TransitionLane，如 `startTransition` 中的状态更新）、空闲更新（IdleLane，如离屏内容预渲染）。当高优先级的 lane 被调度时，React 会中断当前正在处理的低优先级工作，优先完成高优先级更新。

```
// 高优先级更新打断低优先级更新的示意
时间轴: |------帧1------|------帧2------|------帧3------|

低优先级工作（渲染长列表）: [====>          |====>          |====]
高优先级工作（响应用户点击）:                |==>|            |==|

低优先级工作在每帧中被高优先级打断，剩余部分推迟到下一帧继续
```

`useTransition` 和 `useDeferredValue` 是暴露给开发者的并发 API。`useTransition` 将状态更新标记为低优先级，让 React 在处理这个更新时可以被高优先级更新打断，并返回一个 `isPending` 状态用于显示加载指示器。`useDeferredValue` 接受一个值并返回一个延迟更新的版本，当高优先级更新到来时，延迟值会暂时保持旧值，让页面先响应用户交互。

```jsx
const [searchTerm, setSearchTerm] = useState('')
const [isPending, startTransition] = useTransition()

function handleChange(e) {
  // 紧急更新：立即更新输入框的值
  setSearchTerm(e.target.value)
  // 过渡更新：列表过滤标记为低优先级
  startTransition(() => {
    setFilter(e.target.value)
  })
}

// isPending 为 true 时可以显示加载状态
return <List filter={filter} isLoading={isPending} />
```

需要注意的是，`startTransition` 内的更新必须是纯的状态更新，不能包含副作用（如异步请求、DOM 操作）。如果过渡更新依赖异步数据，应该结合 Suspense 使用——Suspense 会在数据未就绪时显示 fallback UI，数据就绪后恢复渲染，整个过程同样受并发调度控制。

## 跳过更新

React 提供了多种机制让开发者手动跳过不必要的 diff 和重新渲染。这些机制与 Vue 的编译时优化（Patch Flags、Block Tree）思路不同——Vue 通过编译器自动分析哪些节点是动态的，React 则需要开发者显式声明。

`React.memo` 对函数组件做浅比较包装：如果 props 没变（浅层比较），就跳过本次渲染，直接复用上次的结果。`useMemo` 和 `useCallback` 分别缓存计算结果和回调函数引用，防止每次渲染都创建新的对象导致子组件的 `React.memo` 失效。`useRef` 缓存不触发重新渲染的可变值。对于类组件，`shouldComponentUpdate` 和 `PureComponent` 提供类似的能力。

```jsx
// React.memo 跳过 props 未变化的渲染
const ExpensiveList = React.memo(({ items }) => {
  return items.map(item => <Item key={item.id} {...item} />)
})

// useMemo 缓存计算结果，避免传入新的引用
const filtered = useMemo(() => items.filter(i => i.active), [items])
// useCallback 缓存回调引用，避免传入新的函数
const handleClick = useCallback((id) => doSomething(id), [])
```

这些手动优化手段需要在性能分析确认存在瓶颈后按需使用。React 的默认行为是"父组件重新渲染时所有子组件都重新渲染"，这在大多数场景下性能足够——一个组件的 render 函数本身并不昂贵（它只是创建虚拟 DOM 对象），真正的性能开销在 Fiber 的 diff 阶段和 DOM 操作的 commit 阶段。过早使用 `React.memo` 和 `useMemo` 会增加代码复杂度，且每个 memo 本身也有浅比较的开销，在 props 频繁变化的情况下反而比不用 memo 更慢。
