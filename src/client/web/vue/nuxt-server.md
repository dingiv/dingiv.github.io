# 服务端组件
React 的服务端组件一文介绍了 RSC 的原理与机制。Vue 生态对服务端组件的探索走了一条不同的路线——React 为服务端组件设计了全新的序列化协议（RSC Payload）和自定义渲染器，而 Vue/Nuxt 选择在已有的 SSR 能力基础上，通过组件岛屿（Component Islands）架构和文件命名约定来实现类似的效果，不需要修改 Vue 核心的渲染模型。

## 架构演进
Nuxt 2 时代就已经支持服务端渲染（SSR），但所有路由只能使用同一种渲染模式——要么全部 SSR（`mode: 'universal'`），要么全部 CSR（`mode: 'spa'`）。Nuxt 3 引入了混合渲染（Hybrid Rendering），通过 `routeRules` 让不同路由使用不同的渲染策略，包括 SSR、CSR、SSG（静态生成）、ISR（增量静态再生）和 SWR（过期重验证）。

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  routeRules: {
    '/': { prerender: true },             // 构建时预渲染为静态 HTML
    '/products/**': { swr: 3600 },         // 按需生成，缓存 1 小时后过期重验证
    '/blog/**': { isr: 3600 },            // CDN 缓存 1 小时后增量再生
    '/admin/**': { ssr: false },          // 纯客户端渲染
    '/old-page': { redirect: '/new-page' }, // 服务端重定向
  }
})
```

`isr` 和 `swr` 的区别在于缓存位置：`isr` 利用部署平台（Netlify、Vercel）的 CDN 缓存，`swr` 通过响应头控制服务器或反向代理的缓存行为。两者都会生成 `_payload.json` 文件，用于客户端导航时的数据恢复。

Nuxt 3.9 开始引入交互式服务端组件（Component Islands），3.11 进一步扩展了服务端组件的能力，包括服务端专用页面（`.server.vue` 页面）和客户端专用页面（`.client.vue` 页面）。

## 组件岛屿
Nuxt 的服务端组件通过文件后缀来标记。`.server.vue` 后缀表示该组件只在服务端运行，不会被打包进客户端 bundle。

服务端组件有两种使用方式。第一种是独立使用（Islands 模式）：组件通过 `<NuxtIsland>` 在服务端渲染为 HTML 片段，当 props 变化时触发一次网络请求重新渲染，HTML 就地更新。这种方式适合内容展示型组件——比如一个需要大量 markdown 解析库的富文本渲染组件，解析库只在服务端加载，浏览器永远不会下载它。

```vue
<!-- components/MarkdownRenderer.server.vue -->
<template>
  <div v-html="renderedMarkdown" />
</template>

<script setup>
import { marked } from 'marked' // 这个库不会进入客户端 bundle
const props = defineProps<{ markdown: string }>()
const renderedMarkdown = computed(() => marked(props.markdown))
</script>
```

第二种是配对使用：当同一目录下同时存在 `Comments.server.vue` 和 `Comments.client.vue` 时，它们被视为同一个组件的两个"半面"。服务端版本在 SSR 阶段渲染初始 HTML，客户端版本在 Hydration 后接管交互。这比 React 的 `'use client'` 指令更加显式——每个半面的代码完全独立，各自决定自己的实现方式。

```vue
<!-- components/Comments.server.vue — 服务端渲染初始内容 -->
<template>
  <div>
    <div v-for="comment in comments" :key="comment.id">
      {{ comment.text }}
    </div>
  </div>
</template>

<!-- components/Comments.client.vue — Hydration 后接管交互 -->
<template>
  <div>
    <div v-for="comment in comments" :key="comment.id">
      {{ comment.text }}
    </div>
    <CommentForm @submit="addComment" />
  </div>
</template>
```

组件岛屿的底层实现是：`<NuxtIsland>` 在服务端创建一个独立的 Vue 应用实例来渲染组件，生成的 HTML 作为响应返回。每次 props 变化时，触发一次内部 fetch 请求获取最新的渲染结果。这种隔离意味着岛屿组件内部无法直接访问外部应用的上下文（如 `useState` 创建的全局状态），数据传递只能通过 props。props 通过 URL 查询参数传递给服务端渲染接口，所以不适合传递大量数据。

## 选择性客户端激活
Nuxt 3.9 引入了 `nuxt-client` 指令，允许在服务端组件内部标记某些子组件进行客户端激活。3.11 版本将支持扩展为 `deep` 模式，可以在服务端组件树的任意位置使用。

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  experimental: {
    componentIslands: {
      selectiveClient: 'deep'
    }
  }
})
```

```vue
<!-- components/PageSection.server.vue -->
<template>
  <div>
    <!-- 服务端渲染，不进入客户端 bundle -->
    <MarkdownRenderer markdown="# Static Content" />

    <!-- 标记为客户端激活，可以绑定事件和状态 -->
    <Counter nuxt-client :initial="5" />

    <button nuxt-client @click="handleClick">
      交互按钮
    </button>
  </div>
</template>
```

这比 React 的模型更灵活——React 需要在文件顶部声明 `'use client'`，整个文件要么全是服务端组件要么全是客户端组件；Nuxt 允许在同一个 `.server.vue` 文件中按需选择哪些节点需要客户端激活，粒度更细。不过 `nuxt-client` 目前仍是实验性功能，API 可能变化。

## 服务端与客户端专用页面
Nuxt 3.11 引入了页面级别的服务端/客户端专用约定。`.server.vue` 页面在客户端导航时也会走服务端渲染，并且当链接出现在视口中时自动预取，实现接近瞬时加载。`.client.vue` 页面则完全跳过服务端渲染，行为等同于整个页面被 `<ClientOnly>` 包裹——需要注意的是，纯客户端页面在首屏加载时会有明显的空白闪烁，应谨慎使用。

```vue
<!-- pages/dashboard.server.vue — 始终服务端渲染 -->
<!-- pages/settings.client.vue — 纯客户端渲染 -->
```

## 数据获取
Nuxt 提供了 `useFetch` 和 `useAsyncData` 两个核心 composable 来处理 SSR 安全的数据获取。它们在服务端执行数据请求后，将结果序列化到 `_payload.json` 中发送给浏览器，客户端导航时直接从 payload 中读取数据，避免重复请求。

```vue
<script setup>
// useFetch: 自动推断 key，SSR 安全
const { data, pending, error, refresh } = await useFetch('/api/users')

// useAsyncData: 手动指定 key，更灵活
const { data } = await useAsyncData('users', () => $fetch('/api/users'))
</script>
```

两者的返回值都包含 `data`、`pending`、`error`、`refresh` 等响应式状态。`useAsyncData` 的第一个参数是缓存 key，用于在不同组件间复用同一个数据请求。常用的选项包括 `lazy`（非阻塞加载，不等待数据就渲染页面）、`server: false`（仅在客户端执行）、`pick`（从响应中选取需要的字段，减少 payload 体积）、`watch`（监听响应式数据变化自动重新请求）。

服务端 API 路由定义在 `server/api/` 目录下，每个文件自动注册为一个 API 端点，无需手动配置路由。

```ts
// server/api/hello.ts
export default defineEventHandler((event) => {
  return { message: 'Hello' }
})
```

## 与 React RSC 的差异
React 和 Nuxt 的服务端组件在目标上相似——让计算留在服务端，减少客户端 bundle 体积——但实现路径完全不同。

React 引入了全新的序列化协议（RSC Payload），这是一种描述虚拟 DOM 树结构的自定义流式格式。服务端组件渲染后生成 RSC Payload 发送给浏览器，React 运行时解析这个 payload 并与客户端组件树融合。这需要 React 内部实现一个自定义的服务端渲染器（`react-server-dom-webpack`），本质上是在 React 核心层面做了架构改造。

Nuxt 的服务端组件基于已有的 Vue SSR 能力构建。服务端组件渲染为标准 HTML，通过 `<NuxtIsland>` 的内部 fetch 请求获取渲染结果并插入页面。没有自定义的序列化协议，没有特殊的 wire format——输出就是普通的 HTML 片段。服务端组件的实现完全在 Nuxt 框架层面，Vue 核心本身不需要做任何改动。

| 维度 | React RSC | Nuxt Server Components |
|------|-----------|------------------------|
| 序列化协议 | 自定义 RSC Payload | 标准 HTML |
| 实现层面 | React 核心改造 | Nuxt 框架层面 |
| 组件隔离 | 函数不可传（跨网络） | 岛屿内独立 Vue 实例，props 通过 URL 传递 |
| 选择性激活 | 文件级 `'use client'` 声明 | `nuxt-client` 指令，节点级粒度 |
| 数据获取 | Server Actions / fetch in RSC | useFetch / useAsyncData / server routes |

这种差异也体现在生态成熟度上。React 的 RSC 有 Next.js App Router 作为成熟的参考实现，社区资源丰富。Nuxt 的服务端组件目前仍标记为实验性功能，API 可能变化，但在混合渲染和 SSR 方面 Nuxt 3 本身已经非常成熟，服务端组件更多是锦上添花的优化手段。
