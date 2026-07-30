---
title: Rust UI 框架
order: 50
---

# Rust UI 框架
Rust 的 UI 生态在过去三年里从"几乎不可用"发展到"多个框架各有擅长领域"。与 JavaScript/TypeScript 在 Web 端的统治地位不同，Rust UI 框架的核心竞争维度是**渲染方式**和**目标平台**——不同选择决定了应用的性能、体积、开发体验和平台覆盖。

## 核心分类
Rust UI 框架按渲染架构分为三大流派。

**原生渲染（Native）**：直接调用操作系统的图形 API 进行绘制，不依赖 Web 引擎。典型代表 iced、Slint。优势是极小的二进制体积（几 MB）、极低的资源占用、原生外观和手感。劣势是组件库不如 Web 生态丰富，自定义 UI 需要手动绘制。

**Web 渲染（WebView）**：将 HTML/CSS/JS 嵌入 Rust 编译的原生壳中。典型代表 Tauri、Dioxus（桌面端使用 WebView）。优势是复用整个 Web 生态——任何 CSS 框架（Tailwind、Ant Design）、任何 JS 图表库、任何动画能力直接可用。劣势是二进制体积大（几十 MB）、内存占用高（内嵌完整浏览器引擎）、启动速度慢于原生。

**即时模式（Immediate Mode）**：每一帧重新绘制整个 UI，不维护 UI 状态树。典型代表 egui。优势是无生命周期管理、无 borrow checker 与 UI 状态的复杂交互、代码极其简洁。劣势是性能不如保留模式（Retained Mode）在极复杂场景下的增量更新，且默认外观偏向"工具/调试面板"风格而非精致的桌面应用。

## 框架总览
| 框架                        | 渲染方式              | 平台            | 语言                 | 适合场景                              |
| --------------------------- | --------------------- | --------------- | -------------------- | ------------------------------------- |
| **Tauri**                   | WebView               | 桌面/Mobile     | Rust 后端 + Web 前端 | 跨平台桌面应用、想复用 Web 能力的团队 |
| **iced**                    | 原生（wgpu/glow）     | 桌面/Web/Mobile | 纯 Rust              | 桌面应用、需要原生控件的工具软件      |
| **Slint**                   | 原生（自研渲染）      | 桌面/嵌入式/MCU | Rust/C++/JS          | 嵌入式设备 UI、资源极度受限的环境     |
| **egui**                    | 即时模式（wgpu/glow） | 桌面/Web        | 纯 Rust              | 开发工具、调试面板、数据可视化面板    |
| **Dioxus**                  | WebView / 原生实验    | 桌面/Web/Mobile | 纯 Rust              | 类 React 开发体验、全栈 Rust          |
| **Leptos**                  | WASM + DOM            | Web             | 纯 Rust              | Web 应用、类 SolidJS 的细粒度响应式   |
| **Rust Web** (Yew/Sycamore) | WASM + DOM            | Web             | 纯 Rust              | Web 应用，各有组件模型偏好            |

## 选型指南
需要精致的桌面原生应用、不依赖 Web 生态 → **iced**。拥有前端团队、想快速复刻已有 Web 应用为桌面版 → **Tauri**。嵌入式 Linux 或 MCU 上的 GUI → **Slint**（这个场景外框架无法覆盖）。开发工具/面板类应用、追求最少代码量 → **egui**。类 React 的 Rust 全栈开发体验 → **Dioxus**。纯 Web 应用、Rust 替代 JS → **Leptos**（细粒度响应式）。

iced、Slint、Tauri 各有独立篇幅展开介绍。
