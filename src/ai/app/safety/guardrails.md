---
title: Guardrails
order: 3
---
# 输出约束与工具安全
Guardrails 是约束 LLM 行为的工程机制，确保模型输出符合预期格式、内容在允许范围内、工具调用安全可控。与输入过滤和输出检查的"事后拦截"不同，Guardrails 在模型生成过程中或工具执行之前介入，是更主动的安全策略。

## 输出格式约束

### 结构化输出
LLM 的原生输出是自由文本，但应用通常需要结构化数据（JSON、XML、表格）。结构化输出的可靠性直接影响下游系统的稳定性——一个格式错误的 JSON 可能导致前端崩溃。

OpenAI 的 Structured Outputs 功能通过 `response_format` 参数强制模型输出符合指定 JSON Schema 的响应。模型被约束为只能生成有效的 JSON，不会产生语法错误。

```python
from openai import OpenAI
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str
    pros: list[str]
    cons: list[str]

client = OpenAI()
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "评价《星际穿越》"}],
    response_format=MovieReview
)

review = response.choices[0].message.parsed
print(f"{review.title}: {review.rating}/10")
```

当模型原生不支持 Structured Outputs 时（如某些开源模型），可以使用后处理方案：先生成自由文本，然后用 Pydantic 尝试解析，失败时用 LLM 修复格式错误，循环直到成功或达到重试上限。

### 话题约束
特定领域的应用需要约束模型只能讨论特定话题。客服机器人不应该回答政治问题，医疗助手不应该提供投资建议。

话题约束的实现方式包括：Prompt 层面的约束（"你只能回答产品相关问题"）、分类器层面的拦截（用话题分类模型判断输出的类别）、关键词层面的过滤（检测输出中是否包含禁止话题的关键词）。

## Guardrails 框架

### NVIDIA NeMo Guardrails
NeMo Guardrails 是目前最成熟的开源 Guardrails 框架，通过声明式的配置文件定义行为约束。它支持三种类型的 Guardrails：

**输入 Guardrails**（Input Rails）：在用户消息到达模型之前执行检查。可以拦截注入攻击、话题外的输入、格式不符合要求的输入。

**输出 Guardrails**（Output Rails）：在模型输出返回给用户之前执行检查。可以过滤有害内容、验证格式、检查事实准确性。

**对话 Guardrails**（Dialog Rails）：控制对话流程，定义允许的对话路径。例如，客服机器人只能走"问候 → 问题分类 → 问题解答/转人工"的流程，不能偏离。

```yaml
# config.yml - NeMo Guardrails 配置
models:
  - type: main
    engine: openai
    model: gpt-4o

rails:
  input:
    flows:
      - check injection
      - check topic
  output:
    flows:
      - check toxicity
      - check pii
  dialog:
    # 定义允许的对话路径
    user greeting -> bot greeting
    user question -> bot answer OR bot handoff
```

### 自定义 Guardrails
对于不使用框架的场景，可以实现轻量级的 Guardrails：

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class Guardrail:
    name: str
    check: Callable[[str], tuple[bool, str]]

class GuardrailsManager:
    def __init__(self):
        self.input_guardrails: list[Guardrail] = []
        self.output_guardrails: list[Guardrail] = []

    def add_input_guard(self, name: str, check: Callable):
        self.input_guardrails.append(Guardrail(name, check))

    def add_output_guard(self, name: str, check: Callable):
        self.output_guardrails.append(Guardrail(name, check))

    async def check_input(self, text: str) -> tuple[bool, list[str]]:
        violations = []
        for guard in self.input_guardrails:
            passed, reason = guard.check(text)
            if not passed:
                violations.append(f"[{guard.name}] {reason}")
        return len(violations) == 0, violations

    async def check_output(self, text: str) -> tuple[bool, list[str]]:
        violations = []
        for guard in self.output_guardrails:
            passed, reason = guard.check(text)
            if not passed:
                violations.append(f"[{guard.name}] {reason}")
        return len(violations) == 0, violations
```

## 工具安全

### 权限模型
Agent 的工具调用是最需要安全控制的环节。一个被劫持的 Agent 如果能自由执行 shell 命令、读写任意文件、调用外部 API，可能造成严重的安全事故。

权限模型的设计需要考虑三个维度：

**操作级别**：读操作（查看文件、查询数据库）的风险低于写操作（修改文件、执行命令、发送请求）。读操作可以更宽松地自动批准，写操作需要更严格的审批流程。

**资源级别**：即使都是读操作，读取 `/etc/passwd` 和读取项目 README 的风险级别完全不同。资源级别的权限控制需要基于路径白名单、数据库表权限、API 端点白名单等机制。

**用户级别**：不同用户有不同的权限等级。管理员可以执行所有操作，普通用户只能访问自己的资源，匿名用户只能执行只读操作。

### 沙箱执行
shell 命令的执行应该在沙箱环境中进行，限制可访问的文件系统范围、网络连接和系统调用。

Docker 容器是最常用的沙箱方案。每个命令在独立的容器中执行，容器只挂载必要的工作目录，网络访问受限，资源使用（CPU、内存）有上限。容器执行完毕后自动销毁，确保环境干净。

```python
import docker

client = docker.from_client()

def execute_in_sandbox(command: str, workdir: str) -> tuple[int, str]:
    container = client.containers.run(
        image="sandbox:latest",
        command=command,
        volumes={workdir: {"bind": "/workspace", "mode": "rw"}},
        working_dir="/workspace",
        mem_limit="512m",
        cpu_period=100000,
        cpu_quota=50000,  # 50% CPU
        network_mode="none",  # 禁用网络
        remove=True,
        detach=False
    )
    return container.exit_code, container.output.decode()
```

### 参数校验
工具调用的参数必须经过严格校验，防止注入攻击通过参数传递。shell 命令的参数需要转义特殊字符，文件路径需要规范化防止路径遍历（`../../etc/passwd`），SQL 参数需要参数化查询防止 SQL 注入。

```python
import shlex
from pathlib import Path

def safe_execute_command(command: str, allowed_dir: str) -> str:
    # 路径遍历检测
    resolved = Path(allowed_dir).resolve()
    if not Path(command).resolve().is_relative_to(resolved):
        return "错误: 路径超出允许范围"

    # 命令注入防护
    safe_command = shlex.quote(command)
    result = subprocess.run(
        safe_command,
        shell=True,
        cwd=allowed_dir,
        capture_output=True,
        timeout=30
    )
    return result.stdout.decode()
```

### 操作审计
所有工具调用应该被记录到审计日志，包括调用时间、调用者、工具名称、参数、执行结果。审计日志用于事后安全分析和合规审计。对于高风险操作（删除文件、发送外部请求），可以考虑实时告警机制——当检测到异常操作模式时（如短时间大量删除文件），自动暂停 Agent 并通知管理员。
