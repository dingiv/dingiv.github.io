---
title: LangChain
order: 0
---

# LangChain
LangChain 是大模型应用开发的标准框架，它封装了与 LLM 交互的通用模式，让开发者专注于业务逻辑而非底层细节。从简单的单轮问答到复杂的多 Agent 协作，LangChain 提供了构建模块和编排能力，是连接模型和应用的胶水层。

## 核心概念
LangChain 的设计理念是模块化和可组合性。应用由多个组件（Model、Prompt、Memory、Tool）通过 Chain 串联，每个组件负责单一职责，组件之间通过标准接口通信。这种抽象让开发者可以替换底层实现（从 OpenAI 切换到本地 LLM）而不影响上层逻辑，也方便复用社区贡献的组件。

Model I/O 是最基础的三层抽象。Models 封装了与大模型的交互，支持 LLM（文本生成）、Chat Model（对话）、Embeddings（向量化）三种类型。Prompts 管理提示词模板，支持变量插值、示例选择、输出格式验证。Parsers 将模型输出的文本转换为结构化数据（JSON、XML、Python 对象）。这三层构成了输入-处理-输出的完整链路。

Memory 组件解决多轮对话的上下文管理。LLM 本身是无状态的，每次调用都是独立的，Memory 负责在请求之间传递历史信息。基础形态是 ConversationBufferMemory，保存所有历史对话；进阶形态是 ConversationSummaryMemory，用 LLM 总结旧对话节省 tokens；还有 ConversationTokenBufferMemory 根据长度自动截断。Memory 的关键是决定保留什么、丢弃什么，直接影响对话的连贯性和成本。

Chains 是组件的编排器，定义了数据流动的逻辑。最简单的是 LLMChain，将 Prompt 和 LLM 组合；SequentialChain 串联多个 Chain，上一个的输出是下一个的输入；RouterChain 根据输入动态选择执行路径。Chain 的价值在于将复杂流程分解为可测试、可复用的单元。

Tools 和 Agent 赋予模型调用外部能力。Tool 是可调用的函数（搜索、计算、数据库查询），Agent 是根据任务自主选择和调用 Tool 的控制器。ReAct Agent 是经典模式，模型交替执行推理（Thought）和行动（Action），"我需要搜索信息 → 调用搜索工具 → 分析结果 → 给出答案"。Tool 的选择依赖于模型对工具描述的理解，因此清晰的 doc_string 至关重要。

## 快速开始
安装 LangChain 核心库和集成包：`pip install langchain langchain-openai`。OpenAI 集成包需要配置环境变量 `export OPENAI_API_KEY=sk-xxx`，其他模型（Anthropic、Hugging Face）有对应的集成包。

一个最简单的 LLM 调用：
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
response = llm.invoke([HumanMessage(content="解释什么是量子计算")])
print(response.content)
```

使用 PromptTemplate 进行参数化：
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}，负责{task}"),
    ("human", "{input}")
])
chain = prompt | llm
result = chain.invoke({"role": "数据分析师", "task": "解释数据", "input": "什么是中位数"})
```

管道操作符 `|` 是 LangChain 的语法糖，`prompt | llm` 等价于 `RunnableSequence(prompt, llm)`。这种函数式风格让 Chain 的构建更加直观。

## 构建复杂应用
RAG 应用是 LangChain 的典型场景。完整的流程包括：文档加载 → 切分 → 向量化 → 存储 → 检索 → 生成。LangChain 的每个环节都有对应组件：DirectoryLoader 加载本地文件，RecursiveCharacterTextSplitter 智能切分，OpenAIEmbeddings 生成向量，Chroma/Pinecone 作为向量库，Retriever 检索相关文档，最后用检索到的文档增强 Prompt。

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

loader = DirectoryLoader('./docs', glob="**/*.md")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(documents)

vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
```

Agent 构建需要定义 Tool 和 Agent 类型。自定义 Tool 只需要用 `@tool` 装饰器包装一个函数：
```python
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    \"\"\"计算数学表达式，输入如 '2+3*4' \"\"\"
    return str(eval(expression))

tools = [calculate]
from langchain.agents import create_openai_functions_agent
agent = create_openai_functions_agent(llm, tools, prompt)
```

Agent 执行时会自动解析用户意图，选择合适的工具，传递参数，获取结果，最后生成自然语言回答。

## 最佳实践
提示词管理应该使用文件而非硬编码。LangChainHub 是提示词模板的共享平台，可以加载社区贡献的模板。团队内部也应该建立提示词版本控制，Git 追踪变更，A/B 测试不同版本的效果。

错误处理在生产环境不可或缺。LLM 调用可能失败（超时、限流、内容审查），需要重试机制和降级方案。LangChain 提供了 `retry` 装饰器和 `fallback` 能力，`llm.with_fallbacks([backup_llm])` 在主模型失败时自动切换。

性能优化关注延迟和成本。异步调用 `llm.ainvoke` 可以并发多个请求，适合批量处理。缓存机制 `llm.bind(cache=RedisCache())` 对相同输入直接返回缓存结果，节省 tokens 和时间。批处理 `llm.batch` 比循环调用更高效。

调试是 LangChain 开发的痛点。`langchain.debug = True` 会打印每次 Chain 执行的输入输出，但日志量大。LangSmith 是专门的调试平台，可视化 Chain 执行过程，记录中间状态，支持对比不同版本的差异。

## 生态与替代
LangChain 的生态包括 LangSmith（调试监控）、LangServe（部署服务）、LangChain.js（前端支持）。社区贡献的集成涵盖数百种模型、数据库、工具，几乎任何主流 AI 服务都有对应的 LangChain 包装器。

但 LangChain 也有局限性。抽象层过厚导致灵活性受限，定制化需求往往需要绕过框架直接实现。学习曲线陡峭，概念众多，新手容易迷失在组件森林中。版本迭代快，API 不稳定，升级时可能出现破坏性变更。

对于简单应用，直接调用 LLM API 或使用轻量级框架（LlamaIndex、SimpleAI）可能更合适。对于追求极致性能和可控性的场景，基于 Transformers、vLLM 等底层库自建框架也是选择。LangChain 的价值在于快速原型开发和中低复杂度的生产应用，它降低了 AI 落地的门槛，但不是唯一路径。
