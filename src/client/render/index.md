---
title: 渲染引擎
order: 70
---

# 渲染引擎
渲染引擎是根据**高级语言的绘图指令**和图形源数据计算生成**屏幕显示所需要的像素颜色数据**的核心组件。

现代 GUI 程序底层依赖于通用渲染引擎提供图形的绘制能力，以此向用户呈现 UI 界面。在高级渲染场景中，例如影视和游戏程序中，渲染引擎需要暴露更多的高级接口，使得用户可以实现高级渲染效果。

渲染引擎集成了计算机图形学的众多算法成果，以实现高效和美观的像素颜色计算，包括 2D 图形和 3D 场景的渲染计算；为了提高计算速度，渲染引擎往往会操作 GPU 完成加速计算；同时，为了实现跨平台的渲染能力，渲染引擎需要屏蔽不同操作系统平台上的视图系统的差异，参考：[linux 视图系统](/kernel/linux/video/)。

## 渲染管线
渲染引擎是在各个图形 API 所提供的渲染管线上工作的

## 常见引擎
渲染引擎分为 2D 渲染引擎、3D 实时渲染引擎和 3D 非实时渲染引擎。

2D 图形和 3D 场景的渲染有着显著的差异，2D 图形的绘制可以直接基于一个有限大小的画布进行绘制，但是 3D 场景的渲染就要引入复杂计算机图形学理论，使用场景、灯光、相机等概念，将一个 3D 场景变成一张二维的图形。而且 2D 渲染一般使用简单的指令，3D 渲染往往需要使用 3D 模型作为输入，构建 3D 场景，其数据量陡然提升。

### 2D 渲染引擎

| 引擎名称              | 主要用途          | 代表作/应用               | 备注                      |
| --------------------- | ----------------- | ------------------------- | ------------------------- |
| Skia                  | 通用 2D 图形      | Chrome, Flutter, Android  | 多后端，性能优异          |
| Direct2D              | Windows 原生 2D   | Office, Edge, UWP         | 微软原生，集成度高        |
| Core Graphics         | macOS/iOS 原生 2D | Safari, Xcode, AppKit     | 苹果原生，集成度高        |
| Blend2D               | 矢量图形          | 图形编辑器、嵌入式        | 高性能矢量，C++实现       |
| AGG                   | 高质量矢量        | 图形编辑器、嵌入式        | 软件渲染，抗锯齿          |
| Pixman                | 像素图形          | Cairo, X11, Wayland       | linux 像素混合，底层库    |
| Cocos2d/Cocos Creator | 2D/轻量 3D 游戏   | 捕鱼达人，开心消消乐      | 中国流行，移动端友好      |
| PixiJS                | Web 2D 渲染       | Web 动画、游戏            | WebGL 高性能              |
| libGDX                | 2D/3D 游戏框架    | Slay the Spire            | Java 开发，跨平台         |
| RPG Maker             | 2D RPG            | Yume Nikki, OneShot       | 专注 RPG，零编程门槛      |
| Ren'Py                | 视觉小说          | Doki Doki Literature Club | Python 脚本，视觉小说专用 |

### 3D 渲染引擎
在 3D 引擎中又分为实时渲染引擎和非实时渲染引擎，前者强调在短时间内完成快速连续的渲染，帧率要求 60 FPS，以适应例如游戏在内的软件交互式需求，为了达到这个目标，实时渲染必须舍弃部分画面质量，从而换取更快的渲染速度；后者强调高质量和高保真的图形渲染，不必追求实时性，以获得极高的视觉效果，往往适合于电影和静态图片制作。

其实区分实时和非实时渲染的原因是在于当前的图形渲染技术还不够高效，或者是硬件条件还没有能够完整覆盖常见的渲染质量需求，人们不得不做出妥协，从而将有限的资源进行特化。

光线追踪技术是一项典型的高质量渲染技术，需要消耗大量的资源。一般地，实时渲染引擎支持有限的光线追踪功能，并使用近似光照进行尽可能地模拟；而非实时渲染引擎力求获得更高级的渲染效果，往往能够支持更加复杂和完善的光线追踪。

| 引擎名称         | 主要用途            | 实时性 | 代表作/应用                  | 备注                     |
| ---------------- | ------------------- | ------ | ---------------------------- | ------------------------ |
| Arnold           | 高质量渲染          | 否     | 3ds Max, Maya, 影视动画      | 光线追踪，电影级         |
| Cycles           | 高质量渲染          | 否     | Blender, Maya                | Blender 内置，开源高质量 |
| Unity            | 游戏/AR/VR          | 是     | Ori, Cuphead, Genshin Impact | 2D/3D 均支持，生态丰富   |
| Unreal           | 高保真游戏/虚拟制作 | 是     | Fortnite, 虚幻演播厅         | Nanite, Lumen, 高端渲染  |
| Godot            | 游戏                | 是     | 3D 独立游戏                  | 2D/轻量 3D，社区活跃     |
| CryEngine        | 高端游戏            | 是     | Crysis, Hunt: Showdown       | 画质极高，VR 支持        |
| Babylon.js/WebGL | Web 3D 渲染         | 是     | Web3D 演示、游戏             | JS/TS，WebGL 渲染        |
| Three.js/WebGL   | Web 3D 渲染         | 是     | Web3D 演示、可视化           | JS，WebGL 渲染           |
