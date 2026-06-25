---
title: 项目搭建
---

# 从零搭建企业级项目
从零搭建一个企业级前端项目，核心工作不是写业务代码，而是建立一套让团队能够高效协作、持续交付的基础设施。一个设计合理的项目结构应该让新成员在半小时内跑通项目，让十个人的团队像一个人一样提交代码而不产生冲突，让功能迭代和缺陷修复能够独立并行而不互相阻塞。

## 技术选型
技术选型的原则是"选主流、选稳定、选团队能 hold 住的"。框架层面，Vue 3 和 React 19 是当前的两个主流选择，两者在组件模型、状态管理和生态完善度上各有优势，具体选哪个取决于团队的技术背景和项目需求。构建工具在 2026 年的新项目中几乎不存在争议——Vite 是默认选项，只有在需要兼容大量 Webpack 存量配置的特殊场景下才考虑 Rspack。

TypeScript 在企业级项目中已经是必选项，不是可选项。它的价值不仅在于类型安全，更在于为团队提供了统一的代码契约——接口定义即文档，类型检查即代码审查的第一道防线。使用 `strict: true` 开启所有严格检查，不要为了短期方便降低类型精度。

CSS 方案的选择需要权衡开发体验和产物体积。Tailwind CSS 在实用类优先的开发模式下效率极高，适合设计规范明确、组件复用度高的项目；CSS Modules 提供了局部作用域的样式隔离，适合需要精细控制样式优先级的场景。两者不冲突，实际项目中可以混用——Tailwind 处理布局和间距等通用样式，CSS Modules 处理组件特有的复杂样式。

包管理器推荐 pnpm。它的硬链接机制节省磁盘空间和安装时间，严格的 `node_modules` 结构避免了幽灵依赖（phantom dependencies），workspace 协议对 Monorepo 的原生支持也是其他包管理器不具备的。

## 项目结构
一个典型的企业级前端项目结构如下：

```
project/
├── public/                    # 静态资源（不经过构建工具处理）
│   └── favicon.ico
├── src/
│   ├── api/                    # 接口层：封装所有 HTTP 请求
│   │   ├── request.ts          # Axios 实例配置（拦截器、超时、错误处理）
│   │   └── modules/            # 按业务域拆分的接口模块
│   │       └── user.ts
│   ├── assets/                 # 需要构建工具处理的静态资源
│   │   ├── images/
│   │   └── styles/
│   │       └── global.css
│   ├── components/             # 通用组件（跨页面复用）
│   │   ├── Button/
│   │   │   ├── Button.vue
│   │   │   ├── Button.test.ts
│   │   │   └── index.ts        # 统一导出
│   │   └── index.ts            # 所有通用组件的统一导出入口
│   ├── composables/            # 可复用的组合式函数（Vue）或自定义 Hooks（React）
│   │   ├── useAuth.ts
│   │   └── useRequest.ts
│   ├── layouts/                # 页面布局组件
│   │   ├── DefaultLayout.vue
│   │   └── BlankLayout.vue
│   ├── pages/                  # 页面级组件（与路由一一对应）
│   │   ├── home/
│   │   │   ├── index.vue
│   │   │   └── components/     # 页面私有组件（不跨页面复用）
│   │   └── login/
│   │       └── index.vue
│   ├── router/                 # 路由配置
│   │   ├── index.ts
│   │   └── modules/
│   ├── stores/                 # 全局状态管理（Pinia / Zustand）
│   │   ├── user.ts
│   │   └── app.ts
│   ├── types/                  # 全局 TypeScript 类型定义
│   │   ├── api.d.ts
│   │   └── global.d.ts
│   ├── utils/                  # 工具函数
│   │   ├── storage.ts
│   │   └── format.ts
│   ├── App.vue                 # 根组件
│   └── main.ts                 # 入口文件
├── .env                        # 默认环境变量
├── .env.development            # 开发环境变量
├── .env.production             # 生产环境变量
├── .gitignore
├── .prettierrc                 # Prettier 配置
├── eslint.config.js            # ESLint 配置
├── tsconfig.json               # TypeScript 配置
├── vite.config.ts              # Vite 构建配置
├── package.json
└── pnpm-lock.yaml
```

这个结构的组织原则是按职责分层，而非按技术类型分层。`api` 层负责所有与后端通信的逻辑，`components` 存放跨页面复用的通用组件，`pages` 存放与路由一一对应的页面组件及其私有子组件，`composables` 存放可复用的有状态逻辑，`stores` 存放全局状态，`utils` 存放纯函数工具。每一层都有明确的职责边界，开发者新增功能时能快速判断代码应该放在哪个目录。

## 接口层设计
接口层是企业级项目中最早需要稳定下来的基础设施。所有 HTTP 请求不应该在业务组件中直接调用 `fetch` 或 `axios`，而是通过统一的接口层封装。接口层负责三件事：配置 Axios 实例（基础 URL、超时时间、请求拦截器、响应拦截器）、按业务域拆分接口模块、统一错误处理。

请求拦截器负责在请求发出前注入认证令牌（如 `Authorization: Bearer <token>`）、设置请求的 trace ID 用于日志追踪。响应拦截器负责统一处理 HTTP 错误码——401 跳转登录页、403 显示无权限提示、500 显示服务器错误提示。业务组件只需要调用接口方法并处理成功回调，不需要关心网络层面的异常处理。

```typescript
// api/request.ts — Axios 实例配置
import axios from 'axios'
import { useAuthStore } from '@/stores/auth'

const request = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 10000,
})

request.interceptors.request.use((config) => {
  const token = useAuthStore().token
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

request.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore().logout()
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default request
```

```typescript
// api/modules/user.ts — 按业务域拆分的接口
import request from '../request'

export function getUserInfo() {
  return request.get('/user/info')
}

export function updateUserProfile(data: UpdateUserDTO) {
  return request.put('/user/profile', data)
}
```

## 路由与权限
路由配置应该与页面组件分离，集中管理在一个或多个路由模块文件中。路由守卫（Navigation Guards）负责权限校验——在每次路由跳转前检查用户是否已登录、是否有权限访问目标页面。未登录时重定向到登录页并记录目标路由，登录成功后自动跳转回原目标页面。

动态路由是后台管理类项目中常见的模式。用户的权限列表从后端接口获取后，前端根据权限列表动态注册对应的路由，而不是在路由配置中硬编码所有页面的权限标识。这种方式让权限的变更只需在后端调整，无需前端发版。

```typescript
// router/index.ts
import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/login',
      component: () => import('@/pages/login/index.vue'),
      meta: { requiresAuth: false }
    },
    {
      path: '/',
      component: () => import('@/layouts/DefaultLayout.vue'),
      children: [
        {
          path: '',
          component: () => import('@/pages/home/index.vue'),
          meta: { requiresAuth: true }
        }
      ]
    }
  ]
})

router.beforeEach((to) => {
  const auth = useAuthStore()
  if (to.meta.requiresAuth && !auth.isLoggedIn) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }
})
```

## 状态管理
全局状态管理存放的是跨组件、跨页面共享的业务数据——用户信息、应用配置、权限列表等。并非所有状态都需要放入全局 Store，组件内部的私有状态应该用组件自身的响应式变量管理。过度使用全局状态会导致数据流难以追踪，维护成本随项目规模急剧上升。

Store 的拆分粒度按业务域划分，每个 Store 文件对应一个业务域（用户、订单、应用配置等），而不是将所有全局状态塞进一个巨大的 Store 对象。Pinia（Vue）和 Zustand（React）都天然支持这种按模块拆分的模式。

```typescript
// stores/user.ts — 按业务域拆分的 Store
import { defineStore } from 'pinia'
import { getUserInfo } from '@/api/modules/user'

export const useUserStore = defineStore('user', {
  state: () => ({
    userInfo: null as UserInfo | null,
    token: localStorage.getItem('token') || '',
  }),
  actions: {
    async fetchUserInfo() {
      this.userInfo = await getUserInfo()
    },
    logout() {
      this.userInfo = null
      this.token = ''
      localStorage.removeItem('token')
    }
  }
})
```

## 组件设计
通用组件（`components/` 目录）和页面私有组件（`pages/xxx/components/` 目录）的区分是项目结构中的关键约定。通用组件是跨页面复用的，应该做到 props 定义清晰、事件命名规范、无业务逻辑耦合。页面私有组件只在当前页面内使用，不需要考虑复用性，但也不应该直接写在页面文件中——超过 50 行的模板或超过 100 行的脚本逻辑就应该拆分为独立的子组件。

组件目录内的文件组织遵循"单一导出入口"原则：每个组件一个独立目录，目录名即组件名，目录内包含组件文件、样式文件和测试文件，通过 `index.ts` 统一导出。外部引用时使用 `@/components/Button` 而非 `@/components/Button/Button.vue`，减少路径嵌套层级。

## 构建配置
Vite 配置文件是企业级项目中需要持续维护的基础设施。初始阶段的配置通常包括：路径别名（`@` → `src/`）、开发服务器代理（解决本地跨域）、环境变量声明类型（`env.d.ts`）。环境变量和多环境配置的管理方式在工程化搭建中详细展开。

开发服务器代理是最常用的配置项，用于将前端请求中 `/api` 前缀的请求转发到后端开发服务器，避免在本地开发中配置 CORS 或修改 hosts。

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      }
    }
  }
})
```

TypeScript 的路径别名需要与 Vite 配置同步声明，否则编辑器的类型推断和路径跳转无法正常工作。

```json
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    },
    "strict": true
  }
}
```

## 启动流程
从零搭建一个企业级项目，实际的执行顺序是：初始化项目（`pnpm create vite`）→ 配置 TypeScript 严格模式 → 搭建目录结构 → 配置路径别名 → 封装接口层 → 配置路由和权限守卫 → 搭建状态管理。项目骨架稳定后，再接入代码规范（ESLint + Prettier + husky + lint-staged）和 CI/CD 流水线——这两个体系的搭建方式在工程化搭建中详细介绍。

这个过程中，每一层的搭建都应该以"让下一个功能能顺畅开发"为目标，而非追求配置的完备性。项目结构是演进的，不是一次性设计完成的——随着业务复杂度的增长，按需拆分目录、提取公共组件、引入新的工具，比在项目初期就设计一个"完美"的结构要务实得多。
