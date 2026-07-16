# A2A 协议
A2A（Agent-to-Agent）是 Google 于 2025 年提出的 Agent 间互操作协议，旨在解决多 Agent 系统中异构 Agent 之间的发现、通信和协作问题。与面向工具调用的 MCP 协议互补——MCP 解决 Agent↔Tool 的交互，A2A 解决 Agent↔Agent 的交互。

## 为什么需要 A2A
当业务系统由多个专业 Agent 协作完成时，没有标准协议意味着每个 Agent 之间的通信都是定制化的胶水代码。财务 Agent 需要理解客服 Agent 的请求格式，客服 Agent 需要适配物流 Agent 的消息协议——N 个 Agent 之间需要 O(N²) 个适配器。这种方式脆且不可扩展。

A2A 的核心价值在于标准化三个关键环节：Agent 发现（如何找到能处理某个任务的 Agent）、任务委托（如何将任务描述、上下文和约束发送给另一个 Agent）、状态同步（如何在多个 Agent 之间协调任务进度和结果）。这三个环节形成了一套完整的 Agent 间协作协议栈，类似于微服务架构中服务发现、RPC 调用和分布式事务的角色。

## 与 MCP 的关系
MCP（Model Context Protocol）由 Anthropic 提出，定义的是 Agent 如何发现和调用外部工具。MCP Server 暴露工具列表（Tool List），Agent 通过标准化的 JSON-RPC 调用这些工具。A2A 解决的是不同层次的问题：当一个 Agent 发现自己无法完成某个任务（缺乏特定领域知识或能力）时，它通过 A2A 协议将任务（或子任务）委托给另一个具备相应能力的 Agent。

两者的关系可以类比为函数调用与进程间通信——MCP 是一个 Agent 调用一个工具函数，A2A 是一个 Agent 向另一个 Agent 发起跨进程通信。在企业级 Agent 系统中，两者通常是并存的：Agent A 通过 MCP 调用向量数据库查询知识，发现查询结果不足以回答用户问题时，通过 A2A 将问题转发给具备专业分析能力的 Agent B。

A2A 协议的主要推动者 Google 将 MCP 视为 A2A 的补充而非替代。Anthropic 在 2025 年末宣布了对 MCP 生态的扩展，引入了 Agent 间通信的原语，逐渐向 A2A 的领域靠拢。OpenAI 也对 MCP 表示了支持，未来两个协议可能在实践中逐渐融合或形成明确的分层边界。

## 核心机制
A2A 协议定义了以下核心概念。Agent Card 是 Agent 的"名片"，描述了 Agent 的身份、能力、端点地址和通信方式。当一个 Agent 需要寻找协作者时，首先查询 Agent Registry（注册中心），获取目标 Agent 的 Agent Card。

Task 是 A2A 中的工作单元。A2A 采用异步任务模型——委托方创建 Task 并发送给执行方，双方通过 Task ID 跟踪任务的生命周期：Submitted → Processing → Completed/Failed。执行方在处理过程中可以推送中间状态更新（如进度百分比、阶段性成果），委托方可以主动取消任务。

消息格式基于 JSON-LD 和 Schema.org 标准，保证跨系统的语义互操作性。A2A 支持同步请求-响应和异步发布-订阅两种通信模式，可以根据任务的性质（即时查询 vs. 长时间分析）选择合适的交互模式。

安全层面，A2A 要求 Agent 之间的通信必须经过双向认证（mTLS），任务委托附带明确的能力范围声明（Capability Token），接收方 Agent 只能在其声明的权限范围内执行操作。这防止了一个被恶意利用的 Agent 通过 A2A 协议越权操作其他 Agent 的系统资源。

## 工程实践
A2A 目前仍处于早期生态建设阶段。Google 开源了 A2A SDK（支持 Python 和 TypeScript），提供了 Agent Card 注册、任务创建和状态跟踪的标准实现。在实际项目中，如果没有使用 Google 的 Agent 基础设施（如 Agent Development Kit），可以通过自行实现简化的任务委托机制来获得 A2A 的核心价值——标准化的任务描述格式、统一的状态机管理和 Agent 能力声明。

在微服务和 Agent 共存的架构中，A2A 的引入不应取代现有的服务间通信机制。一个务实的方案是：将 Agent 间的复杂业务协作通过 A2A 规范的任务格式进行描述，但底层传输仍使用成熟的消息队列（如 Kafka、NATS）或 gRPC，在现有基础设施之上叠加 A2A 的语义层而非替换传输层。
