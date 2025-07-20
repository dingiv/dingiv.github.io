---
title: WASM
order: 50
---

# Web Assembly
Web Assembly 被描述为一个语言运行时，为支持**多语言前端**而生。该运行时由 JavaScript 引擎来实现，从而支持其他语言编译成 Web Assembly 目标，然后可以被 JavaScript 引擎运行，这样就可以让不同的语言都能够像 JavaScript 那样运行在 Web 平台上，并实现和 JavaScript 的互操作性。

目前在所有主流语言中，Rust 是第一顺位的 WASM 支持者，拥有高性能、安全性、打包体积、开发效率等诸多优势，是 Web 平台的新一代公民。

JavaScript 和 Web 平台往往被单线程和脚本语言性能困扰，可以采用 Web Worker 技术和 Web Assembly 技术来缓解 CPU 密集任务的压力。使用 Rust 编写 WASM 可以作为 JavaScript 的二进制插件，从而胜任 CPU 密集型任务。

另一方面，WASM 也能够直接支持在 WASM 运行环境中访问 DOM 结构，从而不依赖显示的 HTML/CSS 文件来进行 UI 展示。