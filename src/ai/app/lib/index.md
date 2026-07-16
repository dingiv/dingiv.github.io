---
title: 框架库
order: 20
---

# 框架库
大模型应用开发框架库是对 LLM API 的高级封装，它们将提示词模板化、工具调用标准化、RAG 流程模板化，降低大模型应用开发的入门门槛。框架的核心价值在于"编排"——将多个独立的 LLM 调用、工具调用和数据检索步骤组合成可复用的工作流。

## LangChain
LangChain 是最早出现也是生态最成熟的大模型应用框架。它的核心抽象是 Chain（链式调用）和 Agent（自主决策）。Chain 将多个步骤串联为固定的执行序列——例如"加载文档 → 切分 → Embedding → 存入向量库"就是一个典型的 Document Loading Chain。Agent 则赋予 LLM 自主选择执行路径的能力——模型根据用户输入从工具列表中选择合适的工具，决定调用顺序和参数。

LangChain Expression Language（LCEL）是 LangChain 的声明式编排语法，用管道操作符 `|` 串联多个 Runnable 组件。LCEL 的核心价值是自动处理流式传输、异步执行、重试和回退——开发者定义好数据流的拓扑结构，框架负责执行细节。LangChain 的劣势在于抽象层较重，简单的场景引入 LangChain 反而增加了代码的间接性，调试困难。如果你的应用只需要调用 LLM API 加上几次数据库查询，直接用 HTTP 调用和业务代码比引入 LangChain 更清晰。

## LlamaIndex
LlamaIndex 专注于数据索引和检索，其设计哲学围绕"将外部数据转化为 LLM 可消费的格式"。它的核心抽象是 Index——将文档、数据库表、API 响应等异构数据源统一为可查询的索引结构。LlamaIndex 在 RAG 场景中的优势在于提供了丰富的索引类型（向量索引、关键词索引、树形索引、知识图谱索引）和检索策略（递归检索、子问题拆分、混合检索），降低了构建复杂 RAG 系统的工程复杂度。

LlamaIndex 的 Ingestion Pipeline 管理从原始文档到可查询索引的完整数据流水线——包括文档解析（支持 PDF、Markdown、代码文件等）、文本切分（按语义边界或固定长度）、元数据提取（文档标题、日期、来源）和 Embedding 生成。这套数据工程能力是 LlamaIndex 与 LangChain 在 RAG 领域的主要区分点。

## 框架选型
框架选择取决于项目的阶段和复杂度。原型验证阶段，直接调用 LLM API 加简单的 Prompt 拼接就足够，不需要引入框架的间接层。当项目进入生产化阶段——需要多轮 RAG、工具调用、Agent 工作流、可观测性——框架的价值开始显现。

LangChain 的优势是生态庞大和社区资源丰富，几乎所有 LLM 提供商和向量数据库都有对应的 LangChain 集成。劣势是版本迭代快、API 变动频繁、概念抽象较重。LlamaIndex 在数据密集型 RAG 应用中表现更好，但其 Agent 能力不如 LangChain 生态成熟。对于需要自定义工作流的复杂场景，直接使用底层的 LLM API（OpenAI SDK、Anthropic SDK）加上自行实现编排逻辑，往往比依赖框架更灵活且更易维护。

工程上的务实策略是：框架是工具而非信仰。不从第一行代码就引入框架，而是在项目演化中逐步引入框架中确实有价值的模块——例如使用 LangChain 的 Document Loader 和 Text Splitter 做数据预处理（这一层与框架耦合度低，替换成本小），但不一定需要 LangChain 的 Agent Executor（这一层的约束和抽象可能与业务需求冲突）。关键判断标准是"框架省去的代码量是否大于理解框架本身所需的心智成本"。
