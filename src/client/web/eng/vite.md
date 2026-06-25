# Vite
Vite 是当前前端新项目的事实标准构建工具。Vue 官方强推 Vite 作为默认脚手架，React 社区在 create-react-app 停止维护后也大规模转向 Vite，Svelte、Solid 等新兴框架同样首选 Vite 作为开发环境。它的成功并非偶然——在 Webpack 统治了前端构建近十年之后，Vite 从根本上重新思考了"开发环境究竟需要什么"这个问题。

## 核心设计
Vite 的核心思想是：开发环境和生产环境的需求本质不同，不应该用同一套构建策略处理。

开发环境下，Vite 利用浏览器原生 ESM（ES Modules）按需加载源文件，不进行全量打包。浏览器请求哪个模块，Vite 就编译哪个模块，然后即时返回。这意味着冷启动不需要从入口文件开始递归扫描整个依赖图谱，启动时间基本不随项目规模增长。热更新时只需要重新编译单个被修改的模块，速度通常在百毫秒级别——在 Webpack 时代，大型项目的热更新可能需要数秒。

生产环境下，Vite 使用 Rollup 进行打包。Rollup 基于 ESM 的静态分析能力更强，Tree Shaking 效果更精确，产物的体积通常比 Webpack 更小。开发和生产使用不同的构建策略，是 Vite 相比 Webpack（开发和生产都用同一套 Bundle 机制）最本质的区别。

## 预构建
Vite 的开发服务器基于浏览器原生 ESM，但 `node_modules` 下的第三方依赖通常不提供 ESM 格式——它们大多是 CommonJS 规范，且一个库内部可能包含几百个小模块。如果让浏览器直接请求这些文件，会触发大量 HTTP 请求和格式转换开销，严重拖慢加载速度。

Vite 在启动开发服务器时使用 Esbuild 对 `node_modules` 中的依赖进行预构建（Pre-bundling），将 CommonJS 转换为 ESM，并将数百个小模块合并为少量 bundle 文件。预构建的结果缓存在 `node_modules/.vite` 目录中，只有在 `package.json` 的依赖列表变化或 Lock 文件变化时才会重新执行。整个预构建过程基于 Esbuild 的 Go 语言实现，速度极快，通常在几百毫秒内完成。

## HMR 与依赖预构建
Vite 的热模块替换（HMR）同样基于原生 ESM。当文件被修改时，Vite 只重新编译该文件本身，然后通过 WebSocket 通知浏览器重新请求这个模块。浏览器收到通知后，利用 ESM 的动态导入能力在运行时替换模块的导出，无需刷新页面，且应用状态保持不变。

但这里有一个需要理解的技术细节：当被修改的文件是一个被预构建的依赖的内部模块时，Vite 并不会重新触发整个依赖的预构建，而是将变更的模块从预构建 bundle 中排除，以单独的 ESM 模块形式提供给浏览器。这种机制让第三方依赖的开发调试变得可行——你可以在 `node_modules` 中直接修改源码（如调试场景），Vite 会自动适配。

## 配置体系
Vite 的配置文件是 `vite.config.ts`（或 `.js`），基于 ESM 语法导出一个配置对象。配置项分为几类：开发服务器选项（`server.port`、`server.proxy`）、构建选项（`build.rollupOptions`、`build.target`）、依赖优化选项（`optimizeDeps`）、插件配置（`plugins`）。

环境变量通过 `.env` 文件管理，Vite 会将环境变量注入到 `import.meta.env` 对象上。只有以 `VITE_` 前缀开头的变量才会暴露给客户端代码，其余变量仅在 SSR 场景下可用。这个设计是出于安全考虑——数据库密码、API 密钥等敏感信息不应暴露到浏览器端。开发模式下 `import.meta.env.MODE` 为 `development`，生产构建时为 `production`，可通过 `--mode` 参数自定义。

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      }
    }
  },
  resolve: {
    alias: {
      '@': '/src'
    }
  }
})
```

路径别名（`resolve.alias`）是 Vite 配置中使用频率最高的选项之一。将 `@` 映射到项目 `src` 目录后，所有导入语句都可以使用 `@/components/Button.vue` 这种短路径，避免相对路径嵌套过深导致的可读性问题。但需要注意，别名配置需要同时声明给 TypeScript（通过 `tsconfig.json` 的 `paths` 字段），否则编辑器的类型推断和跳转功能无法正常工作。

## 生产构建
Vite 的生产构建使用 Rollup，这意味着 Rollup 的所有配置能力都可以通过 `build.rollupOptions` 使用。代码分割通过 `output.manualChunks` 配置，将框架、工具库、业务代码分离为独立的 chunk，配合内容哈希实现长期缓存。CSS 在生产构建中默认被提取为独立的 `.css` 文件，同样包含内容哈希。

Vite 的构建目标是现代浏览器，默认只转译语法到 ES2020（如可选链、空值合并等保留原样）。如果需要兼容旧版浏览器，需要通过 `@vitejs/plugin-legacy` 引入 Babel 进行降级转译和 Polyfill 注入，这会增加产物的体积和构建时间。工程上建议明确项目的浏览器支持范围，避免不必要的转译开销。

对于库开发（而非应用开发），Vite 提供了库模式（Library Mode）。通过配置 `build.lib`，Vite 可以将项目打包为 ES Module 和 UMD 格式，同时生成类型声明文件。这种模式通常用于开发 npm 组件库或工具库，产出可以直接发布到 npm registry 供其他项目使用。

## 与 Webpack 的本质区别
理解 Vite 的最好方式是把它和 Webpack 做对比。Webpack 在开发模式下基于 Bundle 范式——启动时从 Entry 递归构建完整的依赖图谱，打包成有限数量的 bundle 后才能启动开发服务器。项目越大，冷启动越慢，因为它必须处理完所有模块。Vite 在开发模式下基于原生 ESM 范式——不打包，浏览器按需请求模块，服务端按需编译。冷启动时间与项目规模基本无关。

这个差异在热更新时更加明显。Webpack 的热更新需要沿着依赖链找到所有受影响的模块并重新编译，链越长耗时越多。Vite 的热更新只编译被修改的文件本身，然后通过 ESM 动态导入替换模块导出，耗时几乎恒定。

但需要认识到，Vite 的原生 ESM 范式也有代价。在模块数量极多的巨型项目中，浏览器首次加载页面时会发起大量 HTTP 请求（每个 ESM 模块一个请求），在 HTTP/1.1 环境下可能触发瀑布流问题。Vite 通过预构建缓解了第三方依赖的请求数量，但业务代码本身的模块数量仍然会影响首屏加载速度。这也是为什么 Vite 在大型存量项目迁移中面临挑战——而 Rspack 保持了 Webpack 的 Bundle 范式，在巨型项目中反而有更稳定的性能表现。

从工程实践的角度来看，Vite 是当前新项目的最佳选择——开发体验好、配置简单、生态完善。但如果项目有大量 CommonJS 依赖、自定义 Webpack Loader/Plugin、或特殊的构建流程，迁移到 Vite 的成本需要认真评估。工具的选择永远应该服务于项目的实际需求，而非追逐技术趋势。
