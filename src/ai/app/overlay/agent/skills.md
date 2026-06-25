---
title: Skills
order: 38
---

# Skills
MCP Server 解决的是"AI 能做什么"——把外部能力暴露给 LLM 调用。Skills 解决的是"AI 应该怎么做"——把领域知识和工作流程沉淀为可复用的指令模板。两者的区别在于作用层次：MCP Server 提供工具（函数），Skills 提供方法论（提示词）。一个查询数据库的 MCP Tool 告诉 LLM "你可以执行 SQL"，一个 Code Review 的 Skill 告诉 LLM "审查代码时关注安全性、可读性和边界条件"。

Skills 在 Claude Code 和 Cline 中都实现了两套平行的机制：用户手动触发的 Slash Commands（类似 shell alias），和模型自动触发的 Agent Skills（类似 event hook）。前者适合固定流程（提交代码、发布版本），后者适合需要上下文判断的场景（性能优化建议、架构审查）。

## Claude Code Slash Commands
Claude Code 的自定义 Slash Command 本质上是一个 markdown 文件，内容作为 prompt 注入到对话上下文中。文件存放在 `.claude/commands/` 目录下（项目级，跟随 git），或 `~/.claude/commands/` 目录下（用户级，所有项目共享）。文件名去掉 `.md` 后缀就是命令名，例如 `.claude/commands/review.md` 对应 `/project:review` 命令。

命令支持 YAML frontmatter 来配置元信息：

```yaml
---
description: Review pull request
argument-hint: [pr-number] [priority]
allowed-tools: Bash(git add:*), Bash(git diff:*), Read
---
```

`description` 是命令的简短说明，在 `/help` 中显示；`argument-hint` 是输入时的提示文字；`allowed-tools` 限制命令执行期间可以使用的工具，防止越权操作。

命令内容中可以使用 `$ARGUMENTS` 捕获所有参数，或用 `$1`、`$2` 引用单个位置参数，和 shell 脚本的写法一致：

```markdown
审查以下 PR 的代码质量和安全性：
!`git diff origin/main...HEAD --stat`
!`git log origin/main..HEAD --oneline`

重点关注：
1. 是否有安全漏洞（SQL 注入、XSS、硬编码密钥）
2. 错误处理是否完善
3. 是否有明显的性能问题

最后给出通过/修改/拒绝的建议，附上具体行号。
```

`!`backtick 语法可以在命令执行前运行 bash 命令，把输出注入到 prompt 中。上面的例子在执行 review 前自动拉取 diff 统计和 commit 日志，让 Claude 有充分的上下文。`@` 前缀可以引用文件内容，`@src/utils/helpers.js` 会把整个文件内容插入到 prompt 中。

子目录不影响命令名，只在描述中显示命名空间。`.claude/commands/frontend/component.md` 创建的命令仍然是 `/component`，但 `/help` 中会显示来源为 `project:frontend`。项目级和用户级不能有同名命令文件。

## Claude Code Agent Skills
Agent Skills 是 Claude Code 中模型自动触发的机制。和 Slash Commands 不同，用户不需要手动输入 `/xxx`，Claude 根据 skill 的描述字段自主判断是否激活。

Skill 存储为目录结构，核心文件是 `SKILL.md`：

```
.claude/skills/
  code-review/
    SKILL.md           # 必需：指令内容
    checklist.md       # 可选：辅助文档
    examples/
      bad-pr.ts        # 可选：参考示例
```

`SKILL.md` 的 frontmatter 中，`description` 字段决定了 skill 的触发时机，它会被注入到 Claude 的系统上下文中（约占用 100 token），因此描述需要同时说明功能和使用场景：

```yaml
---
name: code-review
description: Review code changes for security vulnerabilities, performance issues, and best practices. Use when the user asks to review, audit, or check code quality.
allowed-tools: Read, Grep, Glob, Bash(git diff:*)
---

# Code Review

审查代码时遵循以下标准：

## 安全性
- 检查 SQL 拼接、未转义的 HTML 输出、硬编码的密钥和 Token
- 检查文件路径是否做了规范化（防止路径遍历）
- 检查依赖是否有已知漏洞

## 性能
- 识别 N+1 查询模式
- 检查循环中的重复计算
- 关注大对象的不必要拷贝

## 可维护性
- 函数是否超过 50 行（建议拆分）
- 命名是否清晰表达意图
- 是否有充分的错误处理
```

当用户的请求匹配 `description` 中的关键词时，Claude 会自动激活这个 skill，按照 `SKILL.md` 中的指令执行。`allowed-tools` 限制了 skill 激活期间可用的工具范围，比如一个只读审查 skill 不需要写入权限。

项目级 skill 放在 `.claude/skills/`，用户级放在 `~/.claude/skills/`。项目级的跟随 git 提交，团队成员共享；用户级的跨项目通用。两者可以同名共存，Claude 会同时看到两个 skill 的描述。

## Cline Workflows
Cline 中的对等概念是 Workflows，功能上类似 Claude Code 的 Slash Commands。Workflows 是 markdown 文件，存放在 `.clinerules/workflows/` 目录（项目级）或 `~/Documents/Cline/Workflows/` 目录（用户级）。在 Cline 聊天框输入 `/` 可以触发自动补全，选择 workflow 文件名即可执行。

```markdown
# Release Preparation

1. Run `git status` and `git log --oneline -10` to check current state.
2. Verify all tests pass: run the test suite.
3. Check `package.json` version matches the intended release version.
4. Ask the user to confirm the release version and changelog highlights.
5. Update `CHANGELOG.md` with the new version and changes.
6. Create a git tag with the release version.
```

Workflow 的每一步是自然语言指令，Cline 按顺序执行。遇到需要用户决策的步骤会暂停等待确认。Workflow 不支持 `$ARGUMENTS` 或位置参数，如果需要动态输入，通过步骤中的"询问用户"来实现。

项目级 workflow 会覆盖同名全局 workflow，这意味着团队可以自定义覆盖个人的通用流程。

## Cline Agent Skills
Cline 的 Agent Skills 机制和 Claude Code 高度相似。Skill 同样是包含 `SKILL.md` 的目录结构，存放在 `.cline/skills/`（项目级）或 `~/.cline/skills/`（用户级）。需要在 Settings → Features 中开启 Enable Skills（实验功能）。

```yaml
---
name: api-design
description: Guide API endpoint design following RESTful conventions. Use when designing, creating, or modifying API endpoints.
---

# API Design Guidelines

遵循 RESTful 设计原则：

## 路径命名
- 使用名词复数：`/users`、`/orders`
- 用嵌套表示关系：`/users/:id/orders`
- 避免动词出现在路径中

## HTTP 方法语义
- GET 读取，不修改状态
- POST 创建资源
- PUT 全量更新，PATCH 部分更新
- DELETE 删除

## 响应格式
- 成功返回 2xx，附上资源表示
- 参数校验失败返回 400，附上具体字段错误
- 资源不存在返回 404
- 统一使用 JSON 格式，字段名 snake_case
```

和 Claude Code 的关键区别在于优先级：Cline 中全局 skill 优先于项目级同名 skill，设计意图是个人通用 skill 不应被项目特定 skill 覆盖。而 Claude Code 中两者可以同名共存。

## MCP 暴露的 Slash Commands
MCP Server 除了暴露 Tools，还可以暴露 Prompts（预定义的提示模板）。Claude Code 会将这些 Prompts 自动注册为 Slash Commands，命名格式为 `/mcp__<server-name>__<prompt-name>`。这意味着 MCP Server 可以同时提供工具能力和指令模板，用户通过 `/mcp__xxx` 调用 prompt，Claude 通过 `tools/call` 调用工具。Cline 对 MCP Prompts 的支持类似，同样会将其注册为可用的 workflow 或命令。

这种机制把 Skill 和 MCP 联系起来：如果某个提示模板需要依赖外部数据（比如代码库的最新变更、部署环境的配置），MCP Prompt 可以在注册时动态生成上下文，比静态 markdown 文件更灵活。

## 实践建议
Skill 的核心价值在于沉淀团队的领域知识。Code Review 标准、部署检查清单、API 设计规范——这些隐性知识通常只存在于资深成员的脑中，通过 Skill 显式化后，所有使用 AI 工具的团队成员都能自动获得这些指导。项目级的 Skill 应该纳入代码审查，和代码本身一起迭代。

Slash Command 适合高频、确定性的操作流程。如果团队每次发布都遵循相同步骤，写成 workflow 可以消除人为遗漏。`allowed-tools` 字段在实践中很有用：review 类命令限制为只读工具，deploy 类命令限制为特定的部署脚本，从工具层面防止误操作。

Agent Skill 的 `description` 字段决定了触发准确率。太宽泛的描述（如"帮助写代码"）会导致频繁误触发，太窄的描述（如"审查 user service 的 authentication middleware"）又会让 skill 几乎永远不会被激活。好的描述应该覆盖功能关键词和使用场景，让模型在相关请求出现时有足够的匹配信号。
