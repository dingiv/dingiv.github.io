# 迁移

Webpack 存量项目的迁移是前端工程化中最高频也最棘手的议题之一。一个维护了五年的巨型项目，本地冷启动动辄数分钟，团队开发效能被严重拖累。面对这个问题，有三种路线：原地优化 Webpack 5、迁移到 Rspack、迁移到 Vite。三者的迁移成本和适用场景截然不同，选择错误会导致数月返工。

## 迁移 Vite 的痛点

Vite 在开发环境下基于浏览器原生 ESM 运行，不经过打包直接按需加载源文件。这个机制在新项目中表现优异，但对积累了大量非标准规范的巨型老项目来说，迁移成本极高。

### CommonJS 与 ESM 的混合依赖

这是迁移 Vite 最致命的障碍。老项目沉淀了大量没有 ESM 产物的第三方旧库或团队自研的 npm 私有包，内部充斥着 `require()` 和 `module.exports`。虽然 Vite 在预构建阶段使用 Esbuild 尝试将 CommonJS 转为 ESM，但在以下复杂场景中会失效。

动态 `require()` 是最典型的翻车点。老代码中常见的 `require('./modules/' + varName)` 这种运行时动态路径拼接，在原生 ESM 中根本无法在编译期静态解析——ESM 要求导入路径必须是明确的字符串 URL。强行迁移后浏览器会直接抛出语法错误，而这类代码在大型老项目中往往散落在各处，逐个排查的工程量巨大。

循环依赖的处理差异同样容易引发诡异 Bug。CommonJS 导出的是值的拷贝，中途遇到循环依赖时会返回一个未完成的空对象；而 ESM 导出的是动态引用（binding）。当项目中 CommonJS 和 ESM 模块交织使用且存在循环依赖时，迁移到 Vite 后极易触发运行时变量为 `undefined` 的问题，且这类 Bug 的排查难度极高，因为它只出现在特定的模块加载顺序下。

### 非标准语法的历史包袱

Webpack 拥有极为庞大的 Loader 生态，老项目往往深度依赖了 Webpack 独有的语法特性，迁移到 Vite 后需要全量重构。

样式和资源的导入方式是首要问题。老代码中充斥着 `require('image.png')` 或 `import 'style.css?raw'` 这种 Webpack 风格的资源引入。Vite 的静态资源处理有自己的一套规范，如 `import img from './img.png'` 获取 URL，或使用 `?raw`、`?url` 后缀声明意图。所有不符合 Vite 规范的资源引入语句都需要逐一改造。

环境变量的切换同样繁琐。Webpack 项目中上万个文件可能都在使用 `process.env.NODE_ENV`，而 Vite 使用 `import.meta.env`。虽然可以引入 `vite-plugin-compatible` 等插件伪造 `process` 全局变量，但这等于在运行时加了一层胶水代码，增加了不确定性和调试难度。对于一个上万模块的项目，这种"兼容层"本身的稳定性就是一个风险点。

### HTTP 请求瀑布流

巨型项目包含上万个源代码模块。在 Webpack 模式下，本地开发会将这些模块聚合成几个大 bundle 交给浏览器；而 Vite 按需拉取原生 ESM 模块，用户首次打开开发页面时，浏览器会同时发起成百上千个 `.tsx`/`.ts` 的 HTTP 请求。即便 HTTP/2 有多路复用，如此高并发的编译和文件 IO 也会让本地 Node 服务的 CPU 飙满，导致首屏加载出现严重的瀑布流卡顿，开发体验反而发生倒退。这个问题在模块数量较少的新项目中不存在，但恰恰是巨型项目迁移 Vite 时最容易被忽视的风险。

## 迁移 Rspack 的优势

Rspack 是字节跳动开源的构建工具，定位是 Webpack 的高性能现代化平替。它的设计哲学是"重写底层引擎，保留上层生态"，因此在巨型项目的迁移场景中优势明显。

### 配置与产物兼容

Rspack 的配置 API 几乎完全兼容 Webpack。项目中原有的 `splitChunks` 分包策略、`entry` 入口划分、`alias` 别名、`resolve` 模块查找逻辑，可以零成本直接复制到 `rspack.config.js` 中使用。同时，Rspack 打包出来的运行时同样基于 Jsonp 模块加载机制，完美支持 Webpack 体系下的 CommonJS、动态异步导入以及各种资源处理语法。这意味着你不需要为了迎合原生 ESM 去重构上万个文件的历史代码——这是 Rspack 相比 Vite 在存量项目迁移中最核心的优势。

### 插件生态兼容的实现

Rspack 在架构上采用了 Rust Core + JS Bridge 的双层设计。核心的 AST 解析、依赖图谱构建、Tree Shaking 算法全部用 Rust 编写，这是它比 Webpack 快 5 到 10 倍的根本原因。但关键在于，它在编译生命周期中通过 Node-API（C++ 与 JS 的跨语言通信）向外暴露了和 Webpack Tapable 一模一样的钩子机制，如 `compilation.hooks.processAssets`。

在此基础上，Rspack 官方在 Rust 侧硬编码实现了 Webpack 最常用的重型插件和 Loader，如 `html-webpack-plugin` 对应的 `builtin:html-rspack-plugin`，以及内置了 `css-loader`、`babel-loader` 的能力。对于社区中长尾的纯 JS 编写的 Webpack 插件，Rspack 提供了兼容层，这些老插件可以原封不动地挂载到 Rspack 的 JS 钩子上。跨语言通信会有微乎其微的性能损耗，但换来的是存量巨型项目惊人的低迁移门槛。

实际操作中，迁移 Rspack 通常只需将 `webpack.config.js` 重命名为 `rspack.config.js`，调整少量不兼容的配置项，替换几个内置插件即可完成。对于配置复杂度中等、没有重度依赖 Webpack 特殊机制的项目，迁移周期通常在一到两周内。

## 原地优化 Webpack 5

如果短期内有更紧迫的业务需求无法投入迁移工作，或者项目的 Webpack 配置已经极度复杂、迁移风险不可控，Webpack 5 本身也提供了足够强大的原生能力来压榨性能。在不修改任何业务代码的前提下，通常可以把分钟级的启动时间压缩到十几秒。

### 持久化缓存

这是 Webpack 5 最具性价比的原生特性。Webpack 默认将构建缓存存在内存中，进程退出后缓存就失效了。将其改为磁盘持久化缓存后，Webpack 会把全量模块的 AST 节点、依赖图谱、甚至压缩后的字节码全部序列化并固化在本地磁盘中。二次冷启动时直接跳过文件读取、Babel 解析和编译阶段，速度通常能提升 90% 以上。

```javascript
// webpack.config.js
module.exports = {
  cache: {
    type: 'filesystem',           // 从内存缓存切换为磁盘持久化
    allowCollectingMemory: true,
    buildDependencies: {
      config: [__filename],        // 配置文件变更时自动失效重算
    },
  },
}
```

缓存默认存储在 `node_modules/.cache/webpack` 目录下。需要注意的是，首次启动仍然是全量编译，持久化缓存从第二次启动开始生效。在 CI 环境中，也可以通过配置缓存目录的持久化（如 GitHub Actions 的 `actions/cache`）让流水线复用缓存。

### Resolve 剪枝

巨型项目启动慢，很大一部分时间浪费在 Node.js 递归遍历 `node_modules` 查找模块的磁盘 IO 上。Webpack 的 `resolve` 配置默认会沿着父目录逐级向上查找，直到找到匹配的模块或到达文件系统根目录。必须通过严格限定查找范围来减少无效的磁盘 IO。

```javascript
module.exports = {
  resolve: {
    // 限定第三方库的查找目录，避免逐级向上查找
    modules: [path.resolve(__dirname, 'node_modules')],
    // 减少后缀自动补全的尝试次数
    extensions: ['.tsx', '.ts', '.js'],
    // 高频大库直接锁定到具体文件，跳过内部的 package.json 解析
    alias: {
      'react': path.resolve(__dirname, 'node_modules/react/cjs/react.production.min.js'),
    }
  }
}
```

后缀自动补全是一个容易被忽视的性能陷阱。Webpack 在解析一个不带后缀的导入时，会按照 `extensions` 数组的顺序依次在磁盘上尝试 `stat` 系统调用检查文件是否存在。数组越长、导入语句越多，累计的磁盘 IO 开销越大。只保留项目实际使用的后缀，将不存在的组合从查找路径中剔除，对大型项目的启动速度有显著影响。

### 编译加速

多线程编译和编译器替换是两个互相独立的优化方向，可以同时使用。

针对耗时最长的 `babel-loader` 或 `ts-loader`，引入 `thread-loader` 启动 Worker 线程池，将 AST 解析这类 CPU 密集型任务分发给多核 CPU 并行处理。配合 `include` 配置严格限定 Loader 的作用范围到 `src` 目录，避免 `node_modules` 中的代码被重复编译。如果 `thread-loader` 的 Worker 池启动开销在热更新场景下成为瓶颈，可以通过 `poolTimeout` 配置控制 Worker 的空闲回收策略。

更激进的方案是用 `esbuild-loader` 直接替换 `babel-loader` 和 `ts-loader`。Esbuild 使用 Go 编写，在多线程并发能力上远超基于 Node.js 的 Babel。在大多数项目中，替换后编译单步速度能提升数倍，且 AST 解析的准确度完全满足工程需求。这个方案的优势在于它完全在 Webpack 体系内运作，不改变产物格式和运行时行为，风险极低。

## 迁移路线选择

三种路线的适用场景可以用一个简单的决策框架来概括。项目如果对 Webpack 插件生态有重度依赖（自定义 Loader、复杂的 Plugin 链、特殊的构建流程），优先考虑 Rspack，因为它能最大程度复用现有配置和插件，迁移成本最低。如果项目代码规范、ESM 支持良好且没有复杂的自定义构建逻辑，Vite 是更轻量的选择，适合作为全新独立子应用或彻底重构时的选型。如果短期内有业务压力无法投入迁移工作，先用 Webpack 5 的持久化缓存和 esbuild-loader 做原地优化，见效最快、风险为零。

这三条路线不是互斥的，在实际工程中往往是渐进式的：先用原地优化止血，为团队争取时间评估迁移方案，再选择 Rspack 或 Vite 作为最终形态。盲目追求一步到位的迁移往往适得其反。
