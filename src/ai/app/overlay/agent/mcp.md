---
title: MCP Server
order: 37
---

# MCP Server
MCP（Model Context Protocol）是 Anthropic 推出的开放协议，用于标准化 AI 应用与外部系统的连接方式。它解决的问题很直接：不同 AI 工具（Claude、Cline、Cursor 等）各自实现工具调用机制，导致同一个工具需要为每个 AI 客户端单独适配。MCP 将"AI 调用外部能力"这件事抽象成统一的客户端-服务端架构——AI 工具作为 MCP Client，自定义的能力封装为 MCP Server，双方通过 JSON-RPC 2.0 通信。理解 MCP 的关键在于类比 USB-C 接口：USB-C 统一了设备之间的物理连接标准，MCP 统一了 AI 应用与外部工具之间的通信协议。

## 协议架构
MCP 采用客户端-服务端模型，通信基于 JSON-RPC 2.0。客户端是 AI 应用（Claude Desktop、Cline、Cursor 等），服务端是开发者自定义的工具服务。

协议定义了三种通信传输方式：**stdio（标准输入输出流）**、**Streamable HTTP（HTTP POST + SSE）**、以及已废弃的纯 SSE。本地开发最常用的是 stdio——客户端直接启动服务端进程，通过 stdin/stdout 交换 JSON-RPC 消息，无需网络栈，零配置，适合单机场景。

MCP Server 可以暴露三种核心能力：Tools（可调用的函数）、Resources（可读取的数据资源）、Prompts（预定义的提示模板）。其中 Tools 是最常用的，它让 LLM 能够执行具体操作——查询数据库、调用 API、读写文件等。每个 Tool 通过 `tools/list` 请求被发现，通过 `tools/call` 请求被调用，服务端负责执行并返回结果。Client 收到工具定义后，将它们注入到 LLM 的上下文中，模型根据用户意图决定是否调用以及传什么参数。

一次完整的工具调用流程：用户在 Claude 中提问 → Claude 分析可用工具列表 → 决定调用 `get_weather` 并生成参数 → Client 通过 JSON-RPC 向 Server 发送 `tools/call` 请求 → Server 执行逻辑并返回结果 → Client 将结果反馈给 Claude → Claude 基于结果生成自然语言回答。整个过程对用户透明，用户只看到最终的回答和工具调用的摘要。

## 用 FastMCP 构建本地 Server
Python 生态下，`mcp` 官方 SDK 提供了 FastMCP 类，可以用装饰器快速定义工具。FastMCP 会自动从函数签名和 docstring 生成 JSON Schema，省去手动编写工具定义的繁琐。

```python
# weather_server.py
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("weather")

@mcp.tool()
async def get_weather(city: str) -> str:
    """获取指定城市的天气信息

    Args:
        city: 城市名称，如 北京、上海、深圳
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://wttr.in/" + city,
            params={"format": "j1"},
            timeout=10.0
        )
        data = resp.json()
        current = data["current_condition"][0]
        return (
            f"{city} 当前天气: {current['weatherDesc'][0]['value']}, "
            f"温度 {current['temp_C']}°C, "
            f"湿度 {current['humidity']}%"
        )

mcp.run(transport='stdio')
```

`@mcp.tool()` 装饰器将普通 async 函数注册为 MCP Tool。FastMCP 从函数名提取工具名（`get_weather`），从 docstring 和类型注解自动生成工具描述和参数 Schema。`mcp.run(transport='stdio')` 启动 stdio 传输模式的服务器，等待客户端连接。

stdio 模式下有一个关键注意点：绝不能向 stdout 写入任何非 JSON-RPC 内容。`print()` 函数会破坏 JSON-RPC 消息流，导致协议解析失败。日志输出应该使用 `logging` 模块（默认写入 stderr），或者写入文件。

## 更完整的 Server 示例
实际工程中，MCP Server 通常暴露多个工具，并需要处理错误、超时和参数校验。以下是一个同时暴露文件操作和系统信息查询的 Server：

```python
# dev_tools_server.py
import os
import subprocess
import logging
from pathlib import Path
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
mcp = FastMCP("dev-tools")

@mcp.tool()
def list_files(directory: str, pattern: str = "*") -> str:
    """列出指定目录下的文件

    Args:
        directory: 目录路径，绝对路径或相对路径
        pattern: 文件匹配模式，默认匹配所有文件，如 "*.py" 只列出 Python 文件
    """
    path = Path(directory).expanduser().resolve()
    if not path.exists():
        return f"错误: 目录 {directory} 不存在"
    if not path.is_dir():
        return f"错误: {directory} 不是目录"

    matched = sorted(path.glob(pattern))
    if not matched:
        return f"目录 {directory} 中没有匹配 {pattern} 的文件"

    result = []
    for f in matched[:50]:  # 限制返回数量
        size = f.stat().st_size
        size_str = f"{size}B" if size < 1024 else f"{size/1024:.1f}KB"
        result.append(f"{'[D]' if f.is_dir() else '[F]'} {f.name}  {size_str}")
    return "\n".join(result)

@mcp.tool()
def read_file(filepath: str, max_lines: int = 100) -> str:
    """读取文件内容

    Args:
        filepath: 文件路径
        max_lines: 最多读取的行数，默认 100 行
    """
    path = Path(filepath).expanduser().resolve()
    if not path.exists():
        return f"错误: 文件 {filepath} 不存在"

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[:max_lines]
        return "".join(lines)
    except UnicodeDecodeError:
        return "错误: 无法以 UTF-8 编码读取该文件，可能是二进制文件"

@mcp.tool()
def git_status(project_path: str) -> str:
    """获取 Git 仓库的状态信息

    Args:
        project_path: Git 仓库的根目录路径
    """
    path = Path(project_path).expanduser().resolve()
    if not (path / ".git").exists():
        return f"错误: {project_path} 不是一个 Git 仓库"

    result = subprocess.run(
        ["git", "-C", str(path), "status", "--short"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return f"Git 执行错误: {result.stderr}"
    if not result.stdout.strip():
        return "工作区干净，没有未提交的更改"
    return result.stdout.strip()

if __name__ == "__main__":
    mcp.run(transport='stdio')
```

这个 Server 暴露了三个工具：文件列表、文件读取、Git 状态查询。每个工具都做了参数校验和错误处理，返回结构化的文本结果供 LLM 解析。工具粒度按业务语义划分，而非按底层 API 划分——LLM 理解"列出文件"比理解"调用 os.listdir"更容易做出正确决策。

## 接入 Claude Code
Claude Code 通过配置文件注册 MCP Server。在项目根目录的 `.claude/settings.json` 中添加：

```json
{
  "mcpServers": {
    "dev-tools": {
      "command": "uv",
      "args": ["run", "python", "/absolute/path/to/dev_tools_server.py"]
    }
  }
}
```

`command` 是启动 Server 进程的命令，`args` 是参数列表。Claude Code 启动时会执行这个命令，通过 stdio 与 Server 进程通信。也可以使用全局配置 `~/.claude/settings.json`，让所有项目都能使用该 Server。

更常见的做法是用 `uvx` 直接运行 Python 包形式的 Server，无需指定绝对路径：

```json
{
  "mcpServers": {
    "dev-tools": {
      "command": "uvx",
      "args": ["dev-tools-mcp"]
    }
  }
}
```

这要求 Server 已经打包发布到 PyPI，或者通过 `--from` 参数指定本地路径。配置完成后重启 Claude Code，新工具会出现在可用工具列表中。可以通过 `/mcp` 命令查看已连接的 Server 状态。

## 接入 Claude Desktop
Claude Desktop 的配置文件位于 `~/Library/Application Support/Claude/claude_desktop_config.json`（macOS）或 `%APPDATA%\Claude\claude_desktop_config.json`（Windows）。配置格式与 Claude Code 类似：

```json
{
  "mcpServers": {
    "weather": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/weather", "run", "weather_server.py"]
    }
  }
}
```

配置完成后重启 Claude Desktop，对话界面会出现工具图标，点击可以查看已加载的工具列表。直接向 Claude 提问触发工具调用，如"北京今天天气怎么样"，Claude 会自动调用 `get_weather` 工具获取实时数据。

## 接入 Cline
Cline（原 Claude Dev）是 VS Code 中的 AI 编程助手插件，同样支持 MCP。在 VS Code 设置中搜索 `cline.mcpServers`，或直接编辑 `settings.json`：

```json
{
  "cline.mcpServers": {
    "dev-tools": {
      "command": "uvx",
      "args": ["dev-tools-mcp"]
    }
  }
}
```

Cline 的 MCP 配置与其他客户端格式一致，但配置入口在 VS Code 的设置文件而非独立配置文件。配置后在 Cline 的对话面板中可以看到新加载的工具。Cline 的特点是面向代码开发场景，因此文件操作、Git 操作类的 MCP Server 与它的契合度最高。

## 接入 Cursor
Cursor 从 0.40+ 版本开始支持 MCP。配置入口在 Settings → MCP，或者编辑 `.cursor/mcp.json` 文件：

```json
{
  "mcpServers": {
    "dev-tools": {
      "command": "uvx",
      "args": ["dev-tools-mcp"]
    }
  }
}
```

Cursor 的 MCP 工具会集成到 Agent 模式中，在 @ 符号触发的上下文菜单中选择 MCP 工具。由于 Cursor 本身已经有强大的代码索引和编辑能力，MCP Server 更适合提供 Cursor 不擅长的能力：外部 API 调用、数据库查询、自定义业务逻辑等。

## 生态进展
MCP 在 2025-2026 年间经历了从单一方推动到行业广泛接受的快速演变。

Anthropic 最初推出 MCP 时，它主要是 Claude 生态中的工具调用协议。但到 2025 年下半年，OpenAI 和 Google 先后宣布了对 MCP 的支持——OpenAI 在 ChatGPT 和 Assistants API 中集成了 MCP 客户端能力，Google 在 Gemini 和 Agent Development Kit 中提供了 MCP 兼容层。至此，MCP 从"Anthropic 的协议"升级为"行业的通用标准"。

MCP Server 市场随之爆发。从最初社区贡献的几十个参考实现（如 filesystem、postgres、slack），扩展到数千个覆盖各类服务和工具的生产级 Server。数据库类（Postgres、MySQL、MongoDB、Neo4j）、云服务类（AWS、GCP、Kubernetes）、SaaS 工具类（Slack、Notion、GitHub、Jira）都已出现稳定的第三方 MCP Server 实现。这个生态的繁荣反过来强化了 MCP 的标准化价值——开发者只需要学习一种协议，就可以让自己的工具在所有主流 AI 平台中使用。

MCP 与 A2A（Google 提出的 Agent-to-Agent 协议）的关系是互补而非竞争。MCP 定义的是 Agent↔Tool 的接口——一个 Agent 如何发现和调用外部工具。A2A 定义的是 Agent↔Agent 的接口——多个 Agent 之间如何相互发现、委托任务和同步状态。在实际系统中，一个 Agent 可能同时使用 MCP（调用数据查询、API 调用等工具）和 A2A（将子任务委托给其他专业的 Agent）。两者的标准化共同构成了 Agent 生态系统的基础设施层。

MCP 的未来演进方向包括：多模态工具支持（不仅传递文本，还支持图像、音频、视频作为工具输入输出）、流式工具调用（工具结果边生成边返回，而非等待完整结果）、以及安全沙箱的标准化（约束工具的运行环境，防止恶意工具越权操作）。这些能力一旦标准化，将推动 MCP 从"开发者工具协议"向"企业级 Agent 基础设施"的升级。

