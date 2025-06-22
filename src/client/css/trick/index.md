# CSS trick
在工程化项目中，css 的规模非常大，并且 css 本身具有混乱 debuff，管理起来非常痛苦。css 的解决方案有很多，但是多也造成了技术选型的困难。

## 常见的技术

1. css 预处理器，代表技术 **Sass**。
   Sass 赋予了 css 一定的高级语言特性，增强了 css 简陋的语言能力，其中包括：
   - css 模块化。通过`@use`和`@forward`指令实现，在一定程度上解决了，css 的样式模块管理和样式隔离；
   - 变量、数据容器。分别通过`$`符号和`#{}`插值语法，`list`和`map`等进行使用，增加了 css 语言的动态性和灵活性，减少了代码冗余；
   - 流程控制和函数。分别通过`@for`、`@while`、`@if`等流程控制语句进行使用，`@function`、`@return`和 css 函数调用，增加了代码逻辑的复用能力；
   - 继承和混入。通过`%`选择器和`@extend`，`@mixin`和`@include`进行使用，提高了 css 样式复用能力；
2. css 后处理器，**PostCSS**技术。
   被广泛集成在打包工具中的技术，统一对 css 源代码进行额外的加工处理。PostCSS 是一套 css 的处理流水线，上面的每一道工序都作为一个插件进行添加，开发者可以选择自己想要的插件进行灵活添加。常见的 PostCSS 插件有：
   - css module。在 js 中以 JSON 的形式导入 css 文件，返回开发时类名到混淆后的类名的映射，解决 css 命名冲突的问题。
   - autoprefixer。自动对 css 规则进行浏览器前缀添加。
   - nanocss。将 css 文件进行压缩和分包。
   - stylelint。对 css 文件进行 lint 检查。
   - preset env。对 css 较新的特性和语法进行 polyfill 和降级处理。
   - tailwindcss。css 框架。预设类名工具集、预设主题设计系统。
   - ……
3. cij，css in js。在 React 框架的生态中广泛使用。主要的形式包括：
   - inline style，直接使用 js object 绑定到行内样式属性和行内 css 变量，js 联动 css 首选方案。
   - style runtime，直接在 js 中写 css 文本、js object，返回混淆过的类名或者组件，通过 style runtime 自动在 head 里添加 style 标签或者 link 标签，如 emotion、styled-component，基本满足需要普通开发需求，不能使用 scss 语法，不享受 post css，不要进行响应式绑定(因为性能)。
   - styled-jsx，直接在 jsx 中写 style 元素，通过编译将 style 元素进行移除，从而生成 style 标签。
4. vue scoped。这是 vue 单文件编译能力的附加品，可以说是业界中非常全能的方案了，不过，在进行 SSR 渲染的时候，会导致 html 文件膨胀。
