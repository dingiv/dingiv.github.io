---
title: Pkgs
---

# 常用第三方包

大模型应用开发涉及模型调用、向量检索、数据处理、服务构建等多个环节，以下是各环节最常用的 Python 库。这些库构成了 LLM 应用的技术基础设施，掌握它们有助于快速搭建原型和推进工程落地。

## 模型调用与推理

openai — OpenAI 官方 SDK，兼容所有 OpenAI API 协议的服务（vLLM、Ollama、Azure OpenAI 等）。提供 Chat Completions、Embeddings、Function Calling、流式输出等接口，是大模型应用的基础依赖。

vllm — 高性能推理引擎，引入 PagedAttention 技术解决 KV cache 显存碎片问题，支持连续批处理和张量并行，适合高并发推理服务部署。

transformers — Hugging Face 的核心库，提供数千个预训练模型的加载、推理和微调接口。支持 PyTorch、TensorFlow 后端，集成了 Tokenizer、Pipeline、PEFT（LoRA/QLoRA）等组件，是模型开发和微调的底层工具。

langchain — LLM 应用编排框架，封装了 Model I/O、Chain、Agent、Memory、Tool 等抽象，提供 RAG 流水线、Agent 构建、Prompt 管理等功能。LangChain 的抽象层级较高，适合快速原型，定制化需求可能需要绕过框架直接实现。

llamaindex — 专注于数据索引和检索的 LLM 应用框架，在处理大规模文档库、构建 RAG 系统方面比 LangChain 更专业。内置了文档加载器、分块器、多种检索策略和查询引擎。

## 向量存储与检索

pymilvus — Milvus 向量数据库的 Python SDK，支持 Collection 管理、向量插入、相似度搜索、标量过滤、多向量查询等操作，是构建 RAG 系统的知识存储层。

sentence-transformers — Hugging Face 的向量编码库，提供 BGE、E5、MiniLM 等主流 Embedding 模型的一行调用，支持批量编码、语义相似度计算和模型微调。

faiss — Facebook 开源的向量相似度搜索库，提供高效的 ANN 索引（IndexFlatIP、IndexIVFFlat、IndexHNSWFlat），纯内存运行，适合中小规模数据和原型验证。

chromadb — 轻量级向量数据库，纯 Python 实现，无需独立部署，适合本地开发和原型验证。支持元数据过滤和文档管理。

## 数据处理

pydantic — 数据验证和序列化库，通过 Python 类型注解定义数据模型，自动校验输入数据的类型和约束。广泛用于 LLM 应用的请求/响应校验、工具参数验证、结构化输出解析。

pypdf — PDF 文本提取库，支持文本型 PDF 的逐页提取、旋转页面处理和文档合并。适合 RAG 系统的文档预处理阶段。

python-docx — Word 文档读写库，能读取 docx 文件的标题层级、段落、表格等结构化信息，适合按文档逻辑结构做分块处理。

tiktoken — OpenAI 的 Token 计数库，支持 cl100k_base（GPT-4o）、o200k_base（GPT-4）等编码器，精确计算文本的 Token 数量，用于成本预估和上下文窗口管理。
