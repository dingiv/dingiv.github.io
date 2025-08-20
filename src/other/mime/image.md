# 图片文件
图片文件分为两种大类型，一种是位图类型，一种是矢量类型。两种类型的生态和操作方式具有明显的不同，从他们对应的编辑处理软件中就可以看出来。

## 图片文件格式
浏览器原生支持主流的图片格式

- svg（Scalable Vector Graphics）。**矢量**。svg 图片是基于 xml 语法的纯文本文件，因此它在 Web 上获得一等公民的兼容能力和支持度，能够直接内联到 html 当中，svg 本身存储的是图形绘制的指令，记录了简单图形的绘制指令，本身能够使用滤镜、动画、渐变，并获得 dom 元素的身份与 css 进行联合控制，svg 是 web 平台上相较于纯 css 更高一级的图形表达方式。适合存储简单的图片。支持透明色。
- webp（Web Picture）。**位图**。谷歌为了提高图片在网络上传输所开发的图片格式，提供了一种在 Web 上，保持非常高的图片质量的同时拥有大幅度压缩能力的图片格式，为了 Web 图片而生。支持透明色。
- png（Portable Network Graphics）。**位图**。PNG 使用无损压缩技术和透明度支持，PNG 支持透明背景和部分透明（alpha 通道），在处理图标、徽标和需要透明背景的图像时非常有用。颜色深度高达 48 位的颜色深度，能够表现出非常丰富和细腻的颜色。
- jpg（Joint Photographic Experts Group）。**位图**。普通的无透明图片，支持很高的图片压缩能力。
- ico（Icon File）。**位图**。ICO 文件是一种用于存储计算机图标的图像文件格式，具有灵活的多尺寸支持和透明功能，非常适合 Web 网页的图标。在 windows 系统上使用较多。
- gif（Graphics Interchange Format）。**位图**。适合简单的动图，支持有限的色彩（256 色编码）。支持透明色编码。

## 图片绘制软件
图片绘制软件是用来从 0 到 1，设计、创作的软件。

- Krita。开源的数字绘画软件，提供丰富的绘画工具和笔刷，满足高级的绘画创作需求。

## 图片编辑软件
图片编辑软件是主要用来处理和加工已有图片的软件。

- PS（Adobe Photoshop）。强大的图像编辑和绘图软件，适合于位图类型，行业龙头软件。
- GIMP（GNU Image Manipulation Program）。开源免费的 PS 的对标品，适合于位图。
- AI（Adobe Illustrator）。矢量图形设计软件，适用于创建标志、插图和排版。
- Inkscape。开源免费的矢量图形设计软件，对标 AI。

## 动画制作软件
动画一般是通过多个连续的关键帧形成的流畅画面效果，也有通过骨骼绑定来形成动画的软件。

- spine。2D 游戏动画制作软件，支持 2D 骨骼动画，游戏行业龙头软件。

## vs code 图片插件
- [Draw.io Integration](https://github.com/hediet/vscode-drawio)。开源项目[draw.io](https://github.com/jgraph/drawio)的 vs code 集成，该项目是使用 Web 技术编写的流程图编辑器，使用矢量图进行编辑，内置了丰富的基本图形，包括表格等，其数据文件`.drawio`或者`.dio`是基于 xml 语法的纯文本文件，可以输入为 svg、html 和 png 文件，具有非常的高通用性。该插件，额外支持了`.drawio.svg`、`.drawio.png`、`.dio.svg`和`.dio.png`文件，在 vs code 中，该插件将覆盖默认文件行为。
- [Luna Paint - Image Editor](https://github.com/lunapaint/vscode-luna-paint)。该插件覆盖了 vs code 的图片预览器，可以直接在 vs code 当中编辑图片，支持裁剪、绘制、基本图形、颜色填充等。
- [Paste Image](https://github.com/mushanshitiancai/vscode-paste-image)。该插件主要用以将剪贴板中的图片直接粘贴到工作区目录当中，自动命名，并且在文件中粘贴该图片文件的名字，在`.md`文件中，将直接把引用链接粘贴进去。
