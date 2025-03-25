# Vite 插件
vite 的插件兼容于 rollup，因此 rollup 的插件都可以在 vite 中使用。同时，社区中 **unplugin** 项目提出了一种构建通用于目前主流的构建工具的插件系统。以 **unplugin** 开头的插件也使用于vite。

除去具体的整合插件，用于将vite和第三方的框架和技术进行整合，vite 还有一些通用的插件，用于优化打包、调试、开发等。
+ vite-plugin-md
+ vite-plugin-electron
+ vite-plugin-tauri


## vite-plugin-components
自动按需导入组件，无需手动导入，并且会自动生成 TS 工具类进行类型提示。比较适合于导入框架的组件，启到简化代码的效果。

## vite-plugin-compression
用于生成 gzip 和 brotli 格式的压缩文件，用于优化网络传输。

## unplugin-auto-import
适用于自动按需导入需要的 js 模块中的 export 变量，而无需手动导入，并且会自动生成 TS 工具类进行类型提示。比较适合于导入框架的组件和一些重复使用的 API，启到简化代码的效果。

## vite-plugin-svg-icons
svg 雪碧图，将多个 svg 合并到一起，然后在使用时通过 svg 的`<use />`标签进行引用，实现减少网络请求的目的。可以使用专门的构建工具来进行操作。该方案提供了解决在一个项目中管理大量本地 svg 图标的方法。自动将项目中某个目录下的所有 svg 文件进行打包，并将 svg 雪碧图内容注入到**index.html**中，然后再使用时通过如下语法使用。

```html
<svg aria-hidden="true">
  <use xlink:href="#targetId" fill="red" />
</svg>
```

需要在**main.js**中引入必要逻辑
```js
import "virtual:svg-icons-register";
```

## rollup-plugin-visualizer
可视化打包后的文件分块，查看各个模块的体积大小，方便进行打包优化。

## vite-plugin-pwa
实现 PWA，可以参考 [vite-plugin-pwa](https://github.com/antfu/vite-plugin-pwa) 的文档。

## vite-plugin-mock
用于在开发阶段模拟接口数据，方便前端开发。

## vite-plugin-inspect
用于调试 vite 的插件，可以查看 vite 的各个插件和中间件执行情况。

## vite-plugin-html
用于自定义 index.html 的内容，可以用于添加一些全局的 script、link 等。

## vite-plugin-ssr
用于服务端渲染，可以用于将前端代码打包成服务端可执行的代码，方便部署到服务器上。

## vite-plugin-legacy
用于兼容老版本的浏览器，可以用于将现代的 JavaScript 代码转换为兼容老版本的浏览器代码。

## vite-plugin-svgo
用于优化 svg 文件，可以用于减少 svg 文件的大小，提高加载速度。

## vite-plugin-dts
用于生成 TypeScript 声明文件，可以用于在项目中使用 TypeScript。

## vite-plugin-cp
用于在构建过程中复制文件或目录，可以用于将一些静态资源复制到构建目录中。

## vite-plugin-singlefile
用于将整个项目打包成一个文件，可以用于将整个项目打包成一个文件，方便部署。

## vite-plugin-robots
用于生成 robots.txt 文件，可以用于控制搜索引擎的爬取规则。

## vite-plugin-windicss
用于集成 Windi CSS，可以用于在项目中使用 Windi CSS。

## vite-plugin-native
用于将前端代码打包成原生应用，可以用于将前端代码打包成原生应用，方便部署到移动设备上。

## vite-plugin-remove-console

## vite-plugin-pages
用于自动生成路由，可以用于自动生成路由，方便前端开发。
