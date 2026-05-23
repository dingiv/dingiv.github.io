---
title: LCEL 与 Runnable
order: 2
---

# LCEL 与 Runnable

LangChain 的管道操作符 `prompt | llm | parser` 看起来是语法糖，但它背后是一套完整的组合框架——LCEL（LangChain Expression Language）和 Runnable 接口。几乎所有 LangChain 组件（LLM、Retriever、Tool、Chain）都实现了 Runnable 接口，这意味着它们可以自由组合、并行执行、批量处理、添加回退和缓存。理解 Runnable 是掌握 LangChain 的关键。

## Runnable 接口

Runnable 是 LangChain 的基础抽象，所有可执行的组件都继承自它。Runnable 定义了三个核心方法：`invoke(input)` 同步执行并返回结果；`batch(inputs)` 批量执行一组输入；`stream(input)` 流式输出（逐个 token 返回）。

除此之外，Runnable 提供了丰富的组合方法。`pipe(other)` 或 `|` 操作符将两个 Runnable 串联，前一个的输出作为后一个的输入。`pick(keys)` 从输出字典中选择指定字段。`map(func)` 对输出做函数变换。`with_fallbacks([r1, r2])` 添加降级链路——主 Runnable 失败时自动切换到备用。`with_retry()` 添加重试策略（指数退避）。`bind(config)` 绑定运行时配置（如 cache、metadata）。

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"topic": RunnablePassthrough()}         # 构造输入字典
    | prompt                                   # PromptTemplate
    | llm                                      # ChatOpenAI
    | StrOutputParser()                        # 解析输出
)
```

这个链中，每个 `|` 的左右两侧都是 Runnable 对象。即使是字面量 `{"topic": RunnablePassthrough()}` 也是 Runnable——它是一个字典，LangChain 会自动将其包装为 `RunnableParallel`。

## LCEL 组合模式

LCEL 支持多种组合模式，覆盖了常见的应用架构需求。

串联（RunnableSequence）是最常用的模式，用 `|` 连接，数据从左到右依次流过。上一个 Runnable 的输出作为下一个的输入，类型必须匹配。如果上游输出字典，下游的 PromptTemplate 可以通过 `{key}` 引用字典中的值。

并行（RunnableParallel）同时执行多个 Runnable，将结果合并为字典。构造方式是传入字典：`{"summary": chain_a, "translation": chain_b}`，两个 chain 并行执行，输出 `{"summary": "...", "translation": "..."}`。并行执行在需要同时调用多个工具或生成多个版本的答案时非常有用，延迟约为最慢的那个 Runnable。

路由（RunnableBranch）根据条件动态选择执行路径。通过 `RunnableBranch` 或 `chain | branch_fn` 实现，branch_fn 接收上游输出，返回对应的 Runnable。例如根据用户意图分类，路由到不同的处理链。

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: x["type"] == "search"), search_chain,
    (lambda x: x["type"] == "calc"), calc_chain,
    default_chain  # 都不匹配时的默认路径
)
```

## 输出解析

LLM 的输出是纯文本字符串，但应用通常需要结构化数据（JSON、列表、Pydantic 对象）。OutputParser 负责将文本转换为程序可处理的结构，是 LCEL 管道中 `| parser` 的实现者。

StrOutputParser 是最简单的解析器，直接返回字符串，去除首尾空白。JsonOutputParser 将 JSON 字符串解析为 Python 字典，要求 LLM 的输出是合法的 JSON 格式。PydanticOutputParser 将输出解析为 Pydantic 模型实例，提供类型检查和字段验证。

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Answer(BaseModel):
    reasoning: str = Field(description="推理过程")
    conclusion: str = Field(description="最终结论")
    confidence: float = Field(description="置信度 0-1")

parser = PydanticOutputParser(pydantic_object=Answer)

# Prompt 中需要包含格式指令
prompt = ChatPromptTemplate.from_messages([
    ("system", "按 JSON 格式输出，包含 reasoning, conclusion, confidence 字段\n{format_instructions}"),
    ("human", "{question}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
result = chain.invoke({"question": "1+1等于几"})  # 返回 Answer 对象
```

Parser 的一个常见问题是 LLM 输出不符合预期格式（多说了几句话、JSON 不完整）。解决方式是在 Prompt 中用 `get_format_instructions()` 注入格式要求，以及使用 OutputFixingParser 在解析失败时自动让 LLM 重新生成。

## 缓存与回退

缓存（Cache）是减少重复 LLM 调用、降低成本的基本手段。LangChain 通过 `InMemoryCache` 实现内存缓存，绑定方式是 `chain = chain.bind(cache=InMemoryCache())`。绑定后，相同输入的请求直接返回缓存结果，不再调用 LLM。

缓存的关键是输入的等价性判断。对于字符串输入，直接比较即可。对于字典或消息列表输入，LangChain 会序列化为字符串后比较。如果输入包含时间戳或随机数，缓存命中率会很低，需要在 bind 前用 `RunnablePassthrough.assign()` 过滤掉可变字段。

回退（Fallback）用于提高可用性。当主模型不可用（网络故障、限流、内容审查）时，自动切换到备用模型。`chain.with_fallbacks([backup_chain])` 在主 chain 抛出异常时执行 backup_chain。可以链式设置多级回退：`gpt4_chain.with_fallbacks([gpt35_chain.with_fallbacks([local_chain)]))`。

缓存和回退可以组合使用：先查缓存，缓存未命中时尝试主模型，主模型失败时回退到备用模型。LCEL 的声明式 API 让这种组合非常简洁。

## 异步与流式

LangChain 的所有 Runnable 都有异步版本：`ainvoke`、`abatch`、`astream`。异步调用在 I/O 密集型场景（如多个 LLM 调用、数据库查询）中能显著提升吞吐量。

```python
# 异步调用
result = await chain.ainvoke({"question": "什么是微服务"})

# 异步批量
results = await chain.abatch([{"question": q} for q in questions])

# 异步流式
async for token in chain.astream({"question": "解释 Docker"}):
    print(token.content, end="")
```

流式输出是 LLM 应用的标准体验。`astream` 返回 AsyncIterator，每次 yield 一个 token 的片段。前端通过 SSE 接收并渲染打字机效果。LCEL 的优势在于流式支持是透明的——任何 Runnable 链都可以直接切换到 `astream` 而不需要修改逻辑。
