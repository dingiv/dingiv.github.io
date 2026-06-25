# 服务端组件
React Server Components（RSC）是 React 18 引入的架构变革，在 React 19 和 Next.js App Router 中成为核心范式。它打破了"React 组件只在浏览器端运行"的传统模型，将组件明确划分为服务端组件（Server Components）和客户端组件（Client Components），让计算和依赖留在服务端，动态交互留在浏览器端。

## 为什么需要服务端组件
传统的 React 单页应用（CSR）有几个根深蒂固的问题。首先是 bundle 膨胀——为了让前端组件完成数据处理，不得不将 markdown 解析器、日期格式化库、国际化资源等大型依赖打包进 JavaScript bundle 发送给浏览器，这些代码在客户端运行但完全可以只在服务端使用。其次是瀑布流请求——父组件 fetch 数据后渲染子组件，子组件又发起自己的 fetch，层层嵌套的网络请求叠加出大量的往返延迟。再者，敏感数据（如 API 密钥、数据库查询逻辑）不能暴露给前端，必须在后端维护一套 API 层做中转，增加了开发和维护成本。

RSC 的思路是：让组件直接在 Node.js 服务端运行，读取数据库、访问文件系统、调用内部微服务，完成后将结果以特殊的流式数据格式（RSC Payload）发送给浏览器。组件中引用的服务端依赖（如 markdown 解析库、数据库驱动）不会被打包进前端 bundle，用户的浏览器永远不会下载这些代码。

## 服务端组件与客户端组件
在 RSC 架构下，所有组件默认都是服务端组件。服务端组件可以直接使用 `async/await` 获取数据，可以调用 Node.js 的 `fs` 模块、数据库驱动等服务端 API，但它不能使用 `useState`、`useEffect` 等客户端 Hooks，不能绑定 `onClick` 等浏览器事件，不能访问 `window`/`document` 对象。

客户端组件需要在文件顶部声明 `'use client'` 指令。声明后该组件及其导入的所有依赖都会被打包进前端 bundle，可以使用全部的 React Hooks 和浏览器 API。`'use client'` 是一个边界声明——它标记的是"从这里往下都进入客户端"，但不影响该文件被服务端组件导入和使用。

两者的分工本质上是：服务端组件负责数据获取和静态内容渲染，客户端组件负责交互和状态管理。在实际项目中，页面级别的组件通常是服务端组件（负责获取数据并编排布局），具体交互组件（表单、弹窗、带动画的列表等）是客户端组件。

## 与传统 SSR 的区别
RSC 经常被误解为 SSR 的替代品，但它们处于不同维度，是互补关系。

传统 SSR 关注的是"首次渲染的 HTML"。服务端将整棵组件树跑一遍，生成 HTML 字符串发送给浏览器，用户立即看到页面内容，然后浏览器下载 JS 并执行 Hydration（客户端激活），让页面变得可交互。SSR 优化的是首屏加载速度，但首屏完成后的页面跳转和交互仍然回到纯 CSR 模式，需要下载完整的 JS bundle。

RSC 关注的是"组件在整个生命周期中的运行位置"。RSC 发送给浏览器的不是 HTML 字符串，而是一种结构化的流式数据（RSC Payload），描述了虚拟 DOM 的树形结构。浏览器端的 React 运行时接收到这个 payload 后，与当前的组件树进行融合（Reconciliation），完成局部更新。

这个差异带来了几个关键优势。首先是交互中依然有效——SSR 只作用于首次加载，而 RSC 在用户导航、表单提交、局部刷新时都可以在服务端重新执行，将最新的组件状态流式推送给前端。其次是状态不丢失——RSC 返回的是结构化数据而非 HTML，局部刷新时客户端组件已有的状态（输入框内容、展开/折叠状态、`useState` 的值）会被完整保留，不会被 Hydration 重置。第三是持续的 bundle 优势——SSR 的组件 JS 仍然需要在客户端加载和 Hydration，而 RSC 的服务端组件永远不会被打包进 bundle。

| 维度       | SSR                  | RSC                       |
| ---------- | -------------------- | ------------------------- |
| 输出格式   | HTML 字符串          | RSC Payload（结构化流）   |
| 作用范围   | 仅首次加载           | 整个生命周期              |
| JS bundle  | 完整组件代码仍需加载 | 服务端组件不进入 bundle   |
| 局部刷新   | 回到 CSR 模式        | 服务端重新执行 + 流式推送 |
| 客户端状态 | Hydration 时重置     | 保留不变                  |

## 组件的嵌套规则
RSC 和客户端组件之间的数据流是单向的：服务端 → 客户端。服务端组件可以向客户端组件传递 props，但传递的数据必须是可序列化的——JSON 对象、字符串、数字、数组都可以，但不能传递函数或类实例，因为函数无法跨越网络边界在浏览器端复活。

嵌套规则是：服务端组件可以直接导入并渲染客户端组件，但客户端组件不能直接导入服务端组件。如果尝试在客户端组件中 import 一个服务端组件，那个服务端组件会"退化"为客户端组件被一起打包进 bundle，失去在服务端运行的能力。

当客户端组件确实需要渲染服务端生成的内容时，正确的做法是使用 children 插槽。由一个顶层的服务端组件负责编排布局，将服务端组件作为 `children` 传入客户端组件中：

```jsx
// Layout.tsx — 服务端组件，负责编排
import Sidebar from './Sidebar'          // 客户端组件
import ServerContent from './ServerContent' // 服务端组件

export default function Page() {
  return (
    <Sidebar>
      <ServerContent />  {/* 通过 children 传入 */}
    </Sidebar>
  )
}

// Sidebar.tsx — 客户端组件
'use client'

export default function Sidebar({ children }) {
  const [collapsed, setCollapsed] = useState(false)
  return (
    <div className={collapsed ? 'sidebar collapsed' : 'sidebar'}>
      {children}
    </div>
  )
}
```

这种模式下，`ServerContent` 在服务端执行后将渲染结果序列化为 RSC Payload 的一部分，`Sidebar` 在浏览器端执行，`children` 接收到的是服务端组件已经渲染完毕的结果，不需要知道它的来源。

## Server Actions
React 19 引入了 Server Actions，打通了"客户端表单提交直接调用服务端函数"的链路。在传统模式中，前端提交表单需要 `fetch('/api/xxx', { method: 'POST' })` 调用后端 API 路由，前端写请求逻辑、后端写路由处理，中间还要处理 loading 状态和错误。Server Actions 将这个过程简化为：在服务端定义一个标记了 `'use server'` 的异步函数，直接作为 `<form>` 的 `action` 属性传入。

```jsx
// actions.ts — 服务端函数
'use server'

export async function createPost(formData) {
  const title = formData.get('title')
  await db.posts.insert({ title })
}

// page.tsx — 直接使用
import { createPost } from './actions'

export default function NewPost() {
  return (
    <form action={createPost}>
      <input name="title" />
      <button type="submit">发布</button>
    </form>
  )
}
```

React 在底层自动将 `action={createPost}` 封装为一个标准的 POST 请求，传递表单数据。Server Action 函数在服务端执行，可以直接访问数据库、文件系统等服务端资源，不需要额外的 API 路由层。结合 `useActionState` 可以获取 action 的执行状态（pending、success、error），配合 `useFormStatus` 可以在子组件中获取表单的提交状态，实现无阻塞的表单交互。

Server Actions 也可以通过 `startTransition` 或直接调用（而非表单 action）来触发，适用于非表单场景的异步操作。在底层，React 会自动处理请求序列化、网络传输、服务端执行和响应返回，开发者只需关注业务逻辑本身。

## 组件设计边界
在混合使用服务端组件和客户端组件的项目中，组件的拆分策略与纯 CSR 项目有本质区别。核心理念是：默认所有组件都是服务端组件，只有当组件确实需要浏览器交互（`useState`、事件监听、`useEffect` 等）时，才将其标记为 `'use client'`，并且尽量让客户端组件成为组件树末端的"叶子节点"。

以一个商品详情页为例，整个页面布局、商品名称、描述、规格参数都是服务端组件——它们从服务端直接获取数据并渲染，不进入客户端 bundle。只有最底部的"点赞按钮"或"加入购物车"涉及 `useState` 和 `onClick`，才需要抽离为一个 `'use client'` 的独立按钮组件。这种叶子节点策略最大化了服务端组件的覆盖范围，最小化了客户端 bundle 体积。

```
页面布局 (RSC)
├── 商品标题 (RSC)
├── 商品描述 (RSC)
├── 规格参数 (RSC)
├── 图片画廊 (RSC)
└── 操作栏 (RSC)
    ├── 价格显示 (RSC)
    ├── 点赞按钮 ('use client') ← 叶子节点
    └── 加入购物车 ('use client') ← 叶子节点
```

## Context 的局限性
React Context 在服务端组件中不可用。如果在一个服务端组件的 `layout.tsx` 中包裹了 `ContextProvider`，它下面的整棵组件树会被强制降级为客户端组件，失去 RSC 的优势。这是 RSC 架构中一个需要特别注意的约束——传统 CSR 项目中习惯的"在根节点包裹全局 Provider"的模式不能直接搬过来。

对于服务端组件之间的数据共享，不需要 Context。因为它们运行在同一次服务器请求中，可以直接在需要的地方调用数据获取函数或使用 React 的 `cache()` 进行请求级缓存，数据天然一致且实时。对于客户端组件之间的共享状态，应在客户端子树的根部（声明了 `'use client'` 的组件内部）包裹 Context Provider，仅让需要交互的客户端子组件消费，避免影响上层的服务端组件。

## 防止服务端代码泄漏
服务端组件可以直接使用 Node.js 的 `fs`、`database driver` 等模块，这些代码绝不能泄漏到客户端 bundle。虽然有 `'use client'` 的边界声明，但在团队协作中很容易出现误操作——有人在客户端组件中意外导入了一个包含数据库连接逻辑的工具函数，打包工具不会自动阻止这种行为，直到线上运行时才暴露问题。

`server-only` 包提供了编译阶段的防护。在所有只应在服务端运行的文件顶部加入 `import 'server-only'`，一旦有任何客户端代码导入了这个文件，Webpack/Vite 会在编译阶段直接报错，从根源上杜绝服务端代码泄漏到浏览器的可能。

```ts
// utils/database.ts — 只允许服务端使用
import 'server-only'

export async function queryUser(id) {
  return db.users.findById(id)
}
```

## 架构演进视角
理解 RSC 的一个有效方式是将其放在 Web 渲染架构的演进脉络中来看。传统 PHP 模式下，`.php` 文件在服务端执行，直接查询数据库并将结果嵌入 HTML 输出，但一旦页面到达浏览器，PHP 的生命周期就结束了——交互需要依赖 jQuery 等手动 DOM 操作，服务端和客户端的代码完全割裂。传统 SSR 解决了首屏渲染的问题，在服务端运行 React 生成 HTML 字符串发送给浏览器，但首屏之后所有交互和页面跳转仍然需要下载完整的组件 JS bundle，bundle 体积是持续的负担。

RSC 可以看作是这两个阶段的合流——它继承了 PHP"直接连接数据库、零 bundle 体积"的优势，同时保留了 React 组件化、局部刷新、响应式交互的能力。服务端组件在服务端执行，遇到客户端组件时留下一个占位符和序列化的 props，最终以 RSC Payload 的流式格式发送给浏览器。浏览器端的 React 运行时接收到 payload 后，将客户端组件"拼"进服务端渲染的虚拟 DOM 树中——服务端负责数据和静态内容，浏览器负责交互和状态，两者组成一棵跨越端云边界的统一组件树。

这种架构在 AI Agent 和 RAG 应用开发中尤其有优势：服务端组件可以直接调用大模型 API、读取向量数据库，将结构化数据注入页面；需要流式打字机效果、用户交互中断、画布拖拽的部分，标记为 `'use client'` 交给浏览器处理。前后端的物理边界被框架抹平，开发者编写的是一个跨越端云的统一组件树，而不是"前端代码调用后端接口"。
