---
title: 应用实战
order: 25
---

# Agent 应用实战
前面的章节从理论层面分析了 Agent 的核心架构：循环控制、工具调用、记忆机制、MCP 协议。这些概念在生产级的 Agent 应用中如何落地？本文档选取了两个有代表性的开源/公开 Agent 应用——Cline 和 Claude Code，通过阅读它们的源码和文档，分析核心机制的工程实现。

Cline 是 VS Code 中的 AI 编程助手插件，开源（TypeScript），代码结构清晰，适合深入阅读。Claude Code 是 Anthropic 官方的 CLI 编程工具，基于 Claude Agent SDK 构建，架构设计精良。两者虽然产品形态不同（IDE 插件 vs 命令行工具），但核心都是 ReAct 模式的 Agent 循环，且都实现了工具调用、会话管理、权限控制、MCP 集成等关键机制。

通过对比分析两者的实现差异，可以加深对 Agent 工程的理解：同样的 Agent 理论，在不同的产品形态和技术约束下，会做出不同的工程取舍。Cline 的 IDE 插件形态决定了它需要 WebView 通信和丰富的 UI 状态同步；Claude Code 的 CLI 形态决定了它需要轻量级的安全沙箱和流式输出。这些取舍反映了 Agent 应用的核心权衡：通用性 vs 专用性、安全性 vs 灵活性、性能 vs 可扩展性。
