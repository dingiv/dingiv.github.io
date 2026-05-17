---
title: Claude Code
order: 2
---
# Claude Code 源码分析
Claude Code 是 Anthropic 官方的 CLI 编程工具，基于 Claude Agent SDK 构建，代表了 Agent 应用的另一种工程范式。与 Cline 的 IDE 插件形态不同，Claude Code 运行在终端中，这使得它在架构上更轻量、更面向流式处理，同时在安全性和可扩展性上有独特的设计。

## 架构概览
Claude Code 的核心架构围绕 `QueryEngine` 类构建，这是一个有状态的对话引擎，管理消息历史、工具执行和上下文压缩。与 Cline 的 View-Service-Model 三层架构不同，Claude Code 的架构更接近 Pipeline 模式——数据从用户输入流经处理管道，每个阶段对消息进行变换和丰富。

核心组件的职责划分：

**QueryEngine**（`src/QueryEngine.ts`）是入口和协调器。它持有对话消息数组（`mutableMessages`），管理权限拒绝记录、token 用量统计和文件状态缓存。`submitMessage()` 方法是主要的 API 入口，接收用户输入，处理后启动 Agent 循环。

**query() 函数**（`src/query.ts`）是 Agent 循环的核心。它是一个异步生成器（`AsyncGenerator`），每一步产生一条消息——可能是助手响应、工具调用结果或系统事件。这种生成器模式让调用方可以按需消费消息流，非常适合 CLI 的流式输出场景。

**StreamingToolExecutor**（`src/services/tools/StreamingToolExecutor.ts`）是工具执行引擎。与 Cline 的"先完成响应再执行工具"不同，Claude Code 可以在 LLM 还在流式生成响应时就并行启动工具执行——只要工具之间没有依赖关系。

**Tool Orchestration**（`src/services/tools/toolOrchestration.ts`）负责工具调度的并发控制。它将工具调用分为"并发安全"（concurrency-safe，如只读工具）和"非并发安全"两类，前者可以并行执行，后者必须串行执行。

**内置工具集**（`packages/builtin-tools/`）是一个独立的包，包含约 50 个工具实现。每个工具有自己的目录，包含提示词定义、参数 Schema 和执行逻辑。

## Agent 循环
Claude Code 的 Agent 循环体现在 `query()` 异步生成器中。QueryEngine 的 `submitMessage()` 调用 `query()` 并通过 `for await` 迭代其输出：

```
用户输入 → QueryEngine.submitMessage()
  → processUserInput()  // 处理 slash commands、附加文件等
  → 构建 system prompt（包含工具定义、CLAUDE.md 内容、权限上下文）
  → for await (message of query({...})):
      → 调用 Claude API（流式）
      → 解析响应中的文本块和工具调用块
      → 工具调用块 → StreamingToolExecutor 并行执行
      → 工具结果追加到消息历史
      → 继续下一次 API 调用（如果 LLM 还在调用工具）
      → 当 LLM 不再调用工具时，循环结束
```

循环的终止条件包括：LLM 不再请求工具调用（`stop_reason` 为 `end_turn`）、达到最大轮次（`maxTurns`）、超过预算（`maxBudgetUsd`）、用户中止（`AbortController`）。多个终止条件可以组合使用，形成多重安全保障。

一个值得注意的设计是 **Structured Output** 支持。当用户通过 API 指定了 `jsonSchema` 参数时，Claude Code 会注册一个特殊的 `SyntheticOutputTool`，LLM 通过调用这个工具来输出符合 Schema 的结构化数据。这比让 LLM 直接输出 JSON 更可靠——工具调用的参数天然是结构化的，模型在生成工具参数时的格式遵从性远高于自由文本输出。

## 工具系统
### 工具规模与分类
Claude Code 注册了约 50 个内置工具，数量远超 Cline 的 20 个。这种差异反映了产品定位的不同——CLI 工具需要更丰富的自包含能力，而 IDE 插件可以借助 IDE 已有的功能。

核心文件操作工具：`FileReadTool`（读取文件/图片/PDF/Notebook）、`FileEditTool`（精确字符串替换编辑）、`FileWriteTool`（创建/覆盖文件）、`GlobTool`（文件模式匹配搜索）、`GrepTool`（内容搜索）。

执行工具：`BashTool`（shell 命令执行）、`PowerShellTool`（Windows 环境）、`REPLTool`（交互式代码执行环境）、`TerminalCaptureTool`（终端输出捕获）。

搜索与网络工具：`WebSearchTool`、`WebFetchTool`、`WebBrowserTool`（浏览器自动化）。

Agent 与任务管理：`AgentTool`（子 Agent 调用，支持嵌套）、`TaskCreateTool`/`TaskGetTool`/`TaskUpdateTool`（任务追踪）、`SendMessageTool`（Agent 间通信）。

MCP 工具：`MCPTool`（调用 MCP Server 工具）、`ReadMcpResourceTool`（读取 MCP 资源）、`ListMcpResourcesTool`（列出 MCP 资源）。

元操作工具：`AskUserQuestionTool`（向用户提问）、`EnterPlanModeTool`/`ExitPlanModeTool`（计划模式切换）、`EnterWorktreeTool`/`ExitWorktreeTool`（Git 工作树管理）。

### 流式工具执行
Claude Code 的工具执行架构是其最独特的设计之一。`StreamingToolExecutor` 可以在 LLM 流式响应过程中就开始执行工具，而不是等待整个响应完成。这意味着 LLM 还在生成后续工具调用的同时，前面已经完整的工具调用已经在并行执行了。

工具调用的并发控制通过 `partitionToolCalls()` 实现：它将一批工具调用分为并发安全组和非并发安全组。只读工具（如 `FileReadTool`、`GlobTool`、`GrepTool`）标记为并发安全，可以与其他只读工具同时执行。写操作工具（如 `FileEditTool`、`BashTool`）标记为非并发安全，必须独占执行。

```
LLM 流式响应中产生工具调用: [ReadFile, ReadFile, Grep, Bash, ReadFile]
                    ↓ partitionToolCalls
并发安全组: [ReadFile, ReadFile, Grep, ReadFile]  → 并行执行
非并发安全组: [Bash]                                → 独占执行
```

这种设计的工程价值在于减少 Agent 循环的端到端延迟。传统的 Agent 循环是"LLM 生成 → 等待完成 → 执行工具 → 下一步"，而 Claude Code 实现了"LLM 生成 → 流式解析 → 立即执行 → 下一步"，将 LLM 生成和工具执行的时间重叠。

### 工具发现与搜索
一个有趣的特性是 `ToolSearchTool`——让 LLM 在大量工具中搜索合适的工具。当注册的工具数量超过模型的处理能力时（模型的 tool_choice 在几十个工具时性能下降），`ToolSearchTool` 提供了一个元工具，LLM 可以通过关键词搜索找到需要的工具，而不是每次都加载全部工具定义。

## 权限与安全系统
Claude Code 的权限模型比 Cline 更精细，设计了多种权限模式：

**Plan Mode**（计划模式）：Agent 只能读取文件和规划任务，不能执行任何写操作。适合需要先审查计划再执行的场景。

**Auto-accept Mode**（自动接受模式）：Agent 可以自动执行所有操作，包括写文件和运行命令。适合自动化场景，但需要信任 Agent 的判断。

**Default Mode**（默认模式）：Agent 需要用户确认高风险操作。系统根据操作的风险等级自动判断是否需要确认——读文件自动通过，写文件需要首次确认，删除操作始终需要确认。

权限判断通过 `canUseTool` 回调函数实现。QueryEngine 在创建时注入这个回调，每次工具执行前都会调用。回调返回 `allow`（允许执行）、`deny`（拒绝执行）或 `ask`（需要用户确认）。这种回调模式与 Cline 的 AutoApprove 机制本质上相同，但 Claude Code 的实现更注重流式集成——权限提示不会阻塞整个响应流，而是只暂停等待确认的那个工具。

**沙箱执行**：BashTool 在沙箱环境中执行命令，限制了可访问的文件系统范围和操作权限。沙箱配置可以通过环境变量和设置文件调整。

**Hook 系统**：Claude Code 支持 Hooks——在工具执行前后运行的 shell 脚本。Pre-tool hooks 可以在工具执行前注入额外的校验逻辑，Post-tool hooks 可以在工具执行后触发通知或日志。这个机制让团队能在不修改 Claude Code 源码的情况下扩展安全策略。

## 会话管理
### CLAUDE.md 上下文注入
Claude Code 的标志性特性是 CLAUDE.md 文件系统。三层配置文件定义了不同范围的上下文：

`~/.claude/CLAUDE.md`（用户级）：跨所有项目共享的个人偏好，如"使用中文回复"、"偏好函数式编程风格"。`项目根目录/CLAUDE.md`（项目级）：团队共享的项目规范，如"使用 pnpm 管理"、"测试框架是 Vitest"。`项目根目录/.claude/settings.local.json`（本地级）：个人的本地配置，不入 Git。

这些文件的内容在 `fetchSystemPromptParts()` 中被读取并拼接为 system prompt 的一部分。每次 Agent 循环开始时都会重新加载，确保使用最新的配置。

### 上下文压缩
Claude Code 实现了多层次的上下文压缩机制。当对话历史接近上下文窗口上限时，自动触发压缩——将旧消息总结为更紧凑的表示，保留关键信息（任务目标、已完成步骤、重要决策），丢弃冗余的中间过程。

压缩的触发时机通过 `calculateTokenWarningState()` 和 `getAutoCompactThreshold()` 计算。`AutoCompactTrackingState` 跟踪压缩状态，防止频繁触发。压缩产生的 `compact_boundary` 消息标记了压缩边界，后续处理可以识别哪些消息是压缩后的摘要。

### 消息持久化
Claude Code 通过 `recordTranscript()` 实现消息持久化，支持会话恢复（`--resume`）。持久化的粒度很精细：用户消息在进入查询循环前就持久化（确保即使中途崩溃也能恢复），助手消息采用 fire-and-forget 策略（不阻塞生成器迭代）。这种差异化策略在可靠性和性能之间取得了平衡。

### 内存系统
Claude Code 实现了一个基于文件的内存系统（`memdir/`），允许 Agent 在会话之间持久化信息。内存文件存储在 `~/.claude/projects/<project-hash>/memory/` 目录下，`MEMORY.md` 作为索引文件。Agent 可以通过 `loadMemoryPrompt()` 加载内存内容，这些内容作为 system prompt 的一部分注入。

## 工程启示
对比 Cline 和 Claude Code，可以看到同一 Agent 理论在不同产品形态下的工程差异：

**流式处理深度不同**。Cline 的流式处理主要在 UI 展示层面——让用户更快看到响应。Claude Code 的流式处理深入到工具执行层面——在 LLM 还在生成响应时就启动工具执行，实现了生成和执行的时间重叠。

**工具粒度不同**。Cline 的工具数量较少但每个工具更"重"（如 `write_to_file` 是全量写入），Claude Code 的工具数量更多但每个更"轻"（如 `FileEditTool` 只做精确的字符串替换）。更多但更轻的工具给 LLM 更精细的控制力，但也增加了选择复杂度——`ToolSearchTool` 的存在就是为了缓解这个问题。

**安全模型不同**。Cline 的安全主要依赖 AutoApprove 白名单，Claude Code 的安全是多层级的——权限模式、沙箱、Hook 系统、canUseTool 回调。CLI 环境下的操作风险更高（一个 `rm -rf` 可能比在 IDE 中更致命），因此需要更精细的安全控制。

**配置系统不同**。Cline 的配置主要通过 VS Code 的设置系统管理，Claude Code 的配置通过 CLAUDE.md 文件系统管理。文件式的配置可以随 Git 提交和分享，这对团队协作更有利。
