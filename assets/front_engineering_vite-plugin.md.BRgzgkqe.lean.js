import{_ as e,c as a,a0 as t,o as n}from"./chunks/framework.p2VkXzrt.js";const v=JSON.parse('{"title":"Vite 插件","description":"","frontmatter":{},"headers":[],"relativePath":"front/engineering/vite-plugin.md","filePath":"front/engineering/vite-plugin.md"}'),l={name:"front/engineering/vite-plugin.md"};function s(p,i,r,o,h,u){return n(),a("div",null,i[0]||(i[0]=[t(`<h1 id="vite-插件" tabindex="-1">Vite 插件 <a class="header-anchor" href="#vite-插件" aria-label="Permalink to &quot;Vite 插件&quot;">​</a></h1><p>vite 的插件兼容于 rollup，因此 rollup 的插件都可以在 vite 中使用。同时，社区中 <strong>unplugin</strong> 项目提出了一种构建通用于目前主流的构建工具的插件系统。以 <strong>unplugin</strong> 开头的插件也使用于vite。</p><p>除去具体的整合插件，用于将vite和第三方的框架和技术进行整合，vite 还有一些通用的插件，用于优化打包、调试、开发等。</p><ul><li>vite-plugin-md</li><li>vite-plugin-electron</li><li>vite-plugin-tauri</li></ul><h2 id="vite-plugin-components" tabindex="-1">vite-plugin-components <a class="header-anchor" href="#vite-plugin-components" aria-label="Permalink to &quot;vite-plugin-components&quot;">​</a></h2><p>自动按需导入组件，无需手动导入，并且会自动生成 TS 工具类进行类型提示。比较适合于导入框架的组件，启到简化代码的效果。</p><h2 id="vite-plugin-compression" tabindex="-1">vite-plugin-compression <a class="header-anchor" href="#vite-plugin-compression" aria-label="Permalink to &quot;vite-plugin-compression&quot;">​</a></h2><p>用于生成 gzip 和 brotli 格式的压缩文件，用于优化网络传输。</p><h2 id="unplugin-auto-import" tabindex="-1">unplugin-auto-import <a class="header-anchor" href="#unplugin-auto-import" aria-label="Permalink to &quot;unplugin-auto-import&quot;">​</a></h2><p>适用于自动按需导入需要的 js 模块中的 export 变量，而无需手动导入，并且会自动生成 TS 工具类进行类型提示。比较适合于导入框架的组件和一些重复使用的 API，启到简化代码的效果。</p><h2 id="vite-plugin-svg-icons" tabindex="-1">vite-plugin-svg-icons <a class="header-anchor" href="#vite-plugin-svg-icons" aria-label="Permalink to &quot;vite-plugin-svg-icons&quot;">​</a></h2><p>svg 雪碧图，将多个 svg 合并到一起，然后在使用时通过 svg 的<code>&lt;use /&gt;</code>标签进行引用，实现减少网络请求的目的。可以使用专门的构建工具来进行操作。该方案提供了解决在一个项目中管理大量本地 svg 图标的方法。自动将项目中某个目录下的所有 svg 文件进行打包，并将 svg 雪碧图内容注入到<strong>index.html</strong>中，然后再使用时通过如下语法使用。</p><div class="language-html vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">html</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&lt;</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">svg</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> aria-hidden</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;true&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">  &lt;</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">use</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> xlink:href</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;#targetId&quot;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> fill</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;red&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> /&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&lt;/</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">svg</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;</span></span></code></pre></div><p>需要在<strong>main.js</strong>中引入必要逻辑</p><div class="language-js vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">js</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;virtual:svg-icons-register&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span></code></pre></div><h2 id="rollup-plugin-visualizer" tabindex="-1">rollup-plugin-visualizer <a class="header-anchor" href="#rollup-plugin-visualizer" aria-label="Permalink to &quot;rollup-plugin-visualizer&quot;">​</a></h2><p>可视化打包后的文件分块，查看各个模块的体积大小，方便进行打包优化。</p><h2 id="vite-plugin-pwa" tabindex="-1">vite-plugin-pwa <a class="header-anchor" href="#vite-plugin-pwa" aria-label="Permalink to &quot;vite-plugin-pwa&quot;">​</a></h2><p>实现 PWA，可以参考 <a href="https://github.com/antfu/vite-plugin-pwa" target="_blank" rel="noreferrer">vite-plugin-pwa</a> 的文档。</p><h2 id="vite-plugin-mock" tabindex="-1">vite-plugin-mock <a class="header-anchor" href="#vite-plugin-mock" aria-label="Permalink to &quot;vite-plugin-mock&quot;">​</a></h2><p>用于在开发阶段模拟接口数据，方便前端开发。</p><h2 id="vite-plugin-inspect" tabindex="-1">vite-plugin-inspect <a class="header-anchor" href="#vite-plugin-inspect" aria-label="Permalink to &quot;vite-plugin-inspect&quot;">​</a></h2><p>用于调试 vite 的插件，可以查看 vite 的各个插件和中间件执行情况。</p><h2 id="vite-plugin-html" tabindex="-1">vite-plugin-html <a class="header-anchor" href="#vite-plugin-html" aria-label="Permalink to &quot;vite-plugin-html&quot;">​</a></h2><p>用于自定义 index.html 的内容，可以用于添加一些全局的 script、link 等。</p><h2 id="vite-plugin-ssr" tabindex="-1">vite-plugin-ssr <a class="header-anchor" href="#vite-plugin-ssr" aria-label="Permalink to &quot;vite-plugin-ssr&quot;">​</a></h2><p>用于服务端渲染，可以用于将前端代码打包成服务端可执行的代码，方便部署到服务器上。</p><h2 id="vite-plugin-legacy" tabindex="-1">vite-plugin-legacy <a class="header-anchor" href="#vite-plugin-legacy" aria-label="Permalink to &quot;vite-plugin-legacy&quot;">​</a></h2><p>用于兼容老版本的浏览器，可以用于将现代的 JavaScript 代码转换为兼容老版本的浏览器代码。</p><h2 id="vite-plugin-svgo" tabindex="-1">vite-plugin-svgo <a class="header-anchor" href="#vite-plugin-svgo" aria-label="Permalink to &quot;vite-plugin-svgo&quot;">​</a></h2><p>用于优化 svg 文件，可以用于减少 svg 文件的大小，提高加载速度。</p><h2 id="vite-plugin-dts" tabindex="-1">vite-plugin-dts <a class="header-anchor" href="#vite-plugin-dts" aria-label="Permalink to &quot;vite-plugin-dts&quot;">​</a></h2><p>用于生成 TypeScript 声明文件，可以用于在项目中使用 TypeScript。</p><h2 id="vite-plugin-cp" tabindex="-1">vite-plugin-cp <a class="header-anchor" href="#vite-plugin-cp" aria-label="Permalink to &quot;vite-plugin-cp&quot;">​</a></h2><p>用于在构建过程中复制文件或目录，可以用于将一些静态资源复制到构建目录中。</p><h2 id="vite-plugin-singlefile" tabindex="-1">vite-plugin-singlefile <a class="header-anchor" href="#vite-plugin-singlefile" aria-label="Permalink to &quot;vite-plugin-singlefile&quot;">​</a></h2><p>用于将整个项目打包成一个文件，可以用于将整个项目打包成一个文件，方便部署。</p><h2 id="vite-plugin-robots" tabindex="-1">vite-plugin-robots <a class="header-anchor" href="#vite-plugin-robots" aria-label="Permalink to &quot;vite-plugin-robots&quot;">​</a></h2><p>用于生成 robots.txt 文件，可以用于控制搜索引擎的爬取规则。</p><h2 id="vite-plugin-windicss" tabindex="-1">vite-plugin-windicss <a class="header-anchor" href="#vite-plugin-windicss" aria-label="Permalink to &quot;vite-plugin-windicss&quot;">​</a></h2><p>用于集成 Windi CSS，可以用于在项目中使用 Windi CSS。</p><h2 id="vite-plugin-native" tabindex="-1">vite-plugin-native <a class="header-anchor" href="#vite-plugin-native" aria-label="Permalink to &quot;vite-plugin-native&quot;">​</a></h2><p>用于将前端代码打包成原生应用，可以用于将前端代码打包成原生应用，方便部署到移动设备上。</p><h2 id="vite-plugin-remove-console" tabindex="-1">vite-plugin-remove-console <a class="header-anchor" href="#vite-plugin-remove-console" aria-label="Permalink to &quot;vite-plugin-remove-console&quot;">​</a></h2><h2 id="vite-plugin-pages" tabindex="-1">vite-plugin-pages <a class="header-anchor" href="#vite-plugin-pages" aria-label="Permalink to &quot;vite-plugin-pages&quot;">​</a></h2><p>用于自动生成路由，可以用于自动生成路由，方便前端开发。</p>`,46)]))}const d=e(l,[["render",s]]);export{v as __pageData,d as default};
