# 工程化搭建

构建优化关注的是产物的体积和加载速度，工程化搭建则关注整个团队的开发效率和交付流程。一个设计良好的工程化体系应该覆盖代码规范、环境管理、版本控制和持续集成，让 20 个人的团队像 1 个人一样高效协作。

## 代码规范

代码规范是团队协作的底线。没有统一的规范约束，不同风格的代码混在一起，Git diff 中充斥着格式差异，review 时很难分辨有意义的逻辑改动和纯粹的格式调整。

EditorConfig 统一不同编辑器的基础配置（缩进风格、缩进宽度、换行符等），放在项目根目录作为最低限度的格式约定。ESLint 负责静态语法检查——发现未使用的变量、潜在的类型错误、不安全的代码模式等逻辑问题。Prettier 负责代码格式化——缩进、换行、引号风格、行宽等纯粹的视觉格式。两者的分工是 ESLint 管"对不对"，Prettier 管"好不好看"。

将规范检查自动化是关键。husky 可以在 Git 的 `pre-commit` 钩子中注册检查脚本，lint-staged 配合 husky 只对本次提交涉及的文件运行检查和格式化，而不是扫描整个项目。这意味着开发者提交代码时，如果有格式问题会自动修复，如果有语法错误会阻止提交——从根源上杜绝不规范的代码进入仓库。

```
项目根目录/
├── .editorconfig          # 编辑器基础配置
├── .prettierrc            # Prettier 格式化规则
├── eslint.config.js       # ESLint 语法检查规则
├── .husky/
│   └── pre-commit         # Git 提交钩子
└── package.json           # lint-staged 配置
```

```json
// package.json 中的 lint-staged 配置
{
  "lint-staged": {
    "*.{js,ts,vue}": ["eslint --fix", "prettier --write"],
    "*.{css,scss,md}": ["prettier --write"]
  }
}
```

commitlint 规范 Git 提交信息的格式（如 `feat: xxx`、`fix: xxx`），使提交历史清晰可读，方便后续生成 changelog 和定位问题引入的版本。

## 环境管理

前端项目通常需要在多个环境中运行：本地开发、测试环境、预发布环境、生产环境。每个环境的 API 地址、功能开关、调试开关等配置不同。环境管理的核心是让这些差异通过配置文件驱动，而不是在代码中硬编码或通过注释切换。

Vite 内置了 `.env` 文件支持。`.env` 是所有环境共享的默认配置，`.env.development` 在 `npm run dev` 时加载，`.env.production` 在 `npm run build` 时加载。Vite 会将环境变量注入为 `import.meta.env` 上的属性，但只有以 `VITE_` 为前缀的变量才会暴露给客户端代码，其余变量仅服务端可用（SSR 场景下），这是出于安全考虑的设计——数据库密码、API 密钥等敏感信息不应暴露给浏览器。

```bash
# .env.development
VITE_API_BASE_URL=http://localhost:3000/api
VITE_SHOW_DEBUG_PANEL=true

# .env.production
VITE_API_BASE_URL=https://api.example.com
VITE_SHOW_DEBUG_PANEL=false
```

```js
// 在代码中使用
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL
```

更复杂的环境管理可能需要根据部署目标区分更多的环境（如 staging、canary），此时可以通过自定义 `.env.staging` 文件配合 `--mode` 参数指定加载哪个环境文件：`vite build --mode staging`。

## Monorepo

当项目规模增长到涉及多个子应用或共享组件库时，Monorepo（单仓多包）是比多仓库更优的组织方式。所有相关项目放在同一个 Git 仓库中，共享配置（ESLint、TypeScript、Vite 配置）、工具链和依赖版本，跨项目修改只需一个 PR，避免了多仓库中依赖版本对齐和联调环境的繁琐。

pnpm 是 Monorepo 场景下的首选包管理器。它的硬链接机制（hard link）让多个项目共享同一份依赖的磁盘文件，配合 `workspace` 协议（`"vue": "workspace:*"`）实现包之间的本地引用，无需每次修改都发布到 npm registry。Turborepo 则在 pnpm 之上提供任务编排能力——通过分析包之间的依赖关系图，只重新构建受影响的包，跳过未变化的下游包，在大型 Monorepo 中能显著减少构建时间。

```
monorepo/
├── packages/
│   ├── ui-components/     # 共享 UI 组件库
│   ├── utils/             # 共享工具函数
│   └── types/             # 共享 TypeScript 类型
├── apps/
│   ├── web/               # 主 Web 应用
│   └── admin/             # 管理后台
├── pnpm-workspace.yaml    # pnpm workspace 配置
├── turbo.json             # Turborepo 任务配置
└── package.json
```

Monorepo 的代价是初始配置复杂度的增加，以及对 CI/CD 流水线的影响——需要设计增量构建策略避免每次提交都构建所有包。对于小型项目（1-3 个应用），Monorepo 的收益不明显，多仓库反而更简单直接。但在涉及共享组件库、多端应用、微前端等场景下，Monorepo 的收益远超配置成本。

## CI/CD

持续集成和持续部署（CI/CD）是将代码从提交到上线的自动化流程。Git Push 触发 CI 流水线，自动执行代码检查、测试、构建、部署，每一步都有明确的通过/失败状态，减少人工操作的失误和遗漏。

典型的前端 CI/CD 流水线包含以下阶段：

1. 安装依赖和代码检查：`pnpm install` → `eslint` + `prettier` 检查，确保代码质量。
2. 单元测试和集成测试：运行自动化测试，确保功能正确性。
3. 构建：`vite build` 生成静态产物。
4. 部署：将构建产物上传到 CDN 或部署服务器。

在部署策略上，前端项目通常使用 Docker 容器化部署或直接将静态文件上传到 CDN/对象存储（如 AWS S3、阿里云 OSS）。Docker 多阶段构建（Multi-stage Build）是一个实用的模式：第一阶段使用 Node 镜像执行 `pnpm install` 和 `vite build`，第二阶段使用轻量级的 Nginx 镜像只复制构建产物，最终的生产镜像只包含 Nginx 和静态文件，体积极小。

```dockerfile
# 第一阶段：构建
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
RUN pnpm build

# 第二阶段：部署
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

缓存策略对 CI/CD 的执行速度影响很大。`pnpm install` 的产物（node_modules）和 Vite 的构建缓存可以跨流水线复用——在 CI 环境中配置缓存目录（如 GitHub Actions 的 `actions/cache`），避免每次都从头安装依赖和构建，将流水线执行时间从分钟级降低到秒级。

工程化搭建的核心目标不是追求某一项技术的极致，而是在开发体验（DX）和用户体验（UX）之间取得平衡。开发阶段的构建速度、代码规范的自动化程度、CI/CD 的执行效率影响的是整个团队的产出；产物体积、缓存命中率、首屏加载速度影响的是终端用户。一个成熟的工程化体系应该同时优化这两个端。
