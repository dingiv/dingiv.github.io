---
title: 浏览器
order: 40
---

# 浏览器
浏览器是现代 Web 应用的运行环境，与普通的 GUI 程序不同，浏览器包含着众多的组件，需要经过复杂的实现逻辑。

## 组成
现代浏览器至少包含几个核心组件
|组件|功能|例子|
|-|-|-|
|web 渲染引擎|实现 HTML 和 CSS 语言，解析二者的代码，生成渲染指令，绘制 HTML 元素|Blink(下层基于 Skia)|
|JavaScript 引擎|实现 JavaScript 语言|V8|
|网络协议客户端|实现各种网络协议的客户端逻辑|Http(s), WebSocket|
|客户端数据存储|实现 Web 客户端数据持久化标准|Cookie, LocalStorage, IndexedDB|
|浏览器主应用|封装和整合所有组件，并提供浏览器自身的 GUI|chromium|

除此之外，还有一些其他的模块需要处理：例如：实现 Webgl/WebGPU 图形 API 标准、Web 媒体资源编码和压缩、HTML 的 SVG 语法扩展等等

## 主流浏览器
| 所属项目 | Google Chrome | Mozilla Firefox | Apple Safari   |
| -------- | ------------- | --------------- | -------------- |
| 排版引擎 | Blink         | Gecko           | WebKit         |
| 样式计算 | Blink         | Stylo           | WebCore        |
| JS 引擎  | V8            | SpiderMonkey    | JavaScriptCore |
| 渲染架构 | Skia          | WebRender + GPU | CoreGraphics   |
| 开源许可 | BSD-like      | MPL             | LGPL/MPL 兼容  |

Edge 和 Opera 已经将核心组件迁移到 Chrome，可以理解成同一个浏览器的换皮版本。目前在浏览器市场上，Chrome 一家独大，占据了较多的市场份额，并且自恃开源复刻引擎 Chromium，积极跟进 Web 标准和新特性，目前没有动摇的迹象。