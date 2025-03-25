---
title: CSS
order: 10
---

# CSS
在工程化项目中，css的规模非常大，并且css本身具有混乱debuff，管理起来非常痛苦。css的解决方案有很多，但是多也造成了技术选型的困难。

## 常见的技术
1. css预处理器。代表技术**Sass**。Sass赋予了css一定的高级语言特性，增强了css简陋的语言能力，其中包括：
   + css模块化。通过`@use`和`@forward`指令实现，在一定程度上解决了，css的样式模块管理和样式隔离；
   + 变量、数据容器。分别通过`$`符号和`#{}`插值语法，`list`和`map`等进行使用，增加了css语言的动态性和灵活性，减少了代码冗余；
   + 流程控制和函数。分别通过`@for`、`@while`、`@if`等流程控制语句进行使用，`@function`、`@return`和css函数调用，增加了代码逻辑的复用能力；
   + 继承和混入。通过`%`选择器和`@extend`，`@mixin`和`@include`进行使用，提高了css样式复用能力；
2. css后处理器，**PostCSS**技术。被广泛集成在打包工具中的技术，统一对css源代码进行额外的加工处理。PostCSS是一套css的处理流水线，上面的每一道工序都作为一个插件进行添加，开发者可以选择自己想要的插件进行灵活添加。常见的PostCSS插件有：
   + css module。在js中以JSON的形式导入css文件，返回开发时类名到混淆后的类名的映射，解决css命名冲突的问题。
   + autoprefixer。自动对css规则进行浏览器前缀添加。
   + nanocss。将css文件进行压缩和分包。
   + stylelint。对css文件进行lint检查。
   + preset env。对css较新的特性和语法进行polyfill和降级处理。
   + tailwindcss。css框架。预设类名工具集、预设主题设计系统。
   + ……
3. cij。css in js。在React框架的生态中广泛使用。主要的形式包括：
   + inline style，直接使用js object绑定到行内样式属性和行内css变量，js 联动 css首选方案。
   + style runtime，直接在js中写css文本、js object，返回混淆过的类名或者组件，通过style runtime自动在head里添加style标签或者link标签，如emotion、styled-component，基本满足需要普通开发需求，不能使用scss语法，不享受post css，不要进行响应式绑定(因为性能)。
   + styled-jsx，直接在jsx中写style元素，通过编译将style元素进行移除，从而生成style标签。
4. vue scoped。这是vue单文件编译能力的附加品，可以说是业界中非常全能的方案了，不过，在进行SSR渲染的时候，会导致html文件膨胀。

