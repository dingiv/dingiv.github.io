---
title: 大模型应用
order: 30
---

# 大模型应用开发技术

大模型应用开发基于已有的 Web 前后端技术栈，在前后端分别引入新的组件，推动大模型的能力落地到实际的生产环节中。后端不再只是处理业务逻辑，而是成为了 AI 智能体的骨架和记忆体；前端从 GUI 演进为 LUI（语言用户界面），成为智能体的交互窗口。

## 技术栈

+ 推理引擎：Ollama、vLLM、TGI 等推理框架部署运行大模型，掌握量化（GPTQ/AWQ）、微调（LoRA/QLoRA）和参数调优
+ Embedding 模型：BGE、E5、GTE 等嵌入模型将文本映射为向量，理解对比学习训练原理和选型评估（MTEB）
+ 向量数据库：Milvus、pgvector 等向量数据库，为 RAG 应用提供语义检索能力，理解 HNSW 等索引算法
+ 提示词工程：理解提示词的表达方式对于模型生成的影响，掌握 CoT、Few-shot、Generated Knowledge 等设计模式
+ 会话管理：模型上下文窗口机制与限制、无状态调用的记忆补偿、分层记忆（短期/中期/长期）和对话状态管理
+ 应用框架：LangChain/LlamaIndex 快速构建 LLM 应用，LangGraph 构建有状态的 Agent 工作流
+ 工具调用：Function Calling 机制，定义 JSON Schema 让 LLM 主动调用外部 API
+ API 设计：OpenAI 兼容协议、SSE 流式传输、Token 计费、API 网关（LiteLLM）

## 典型应用

+ RAG 检索增强生成：结合向量检索和 LLM 生成，构建私有知识问答系统，涵盖分块、召回、精排、抗幻觉全链路优化
+ GraphRAG 知识图谱增强：结合 Neo4j 等图数据库，支持多跳推理和关系查询
+ Agent 智能体：ReAct 模式、工具调用、LangGraph 状态机工作流、多 Agent 协作（AutoGen）
+ 自动化脚本：图色脚本、AI GUI 辅助、低代码平台（Dify/Coze/n8n）
+ 媒体生成：Stable Diffusion 原理（VAE/U-Net/CLIP）、采样器、ControlNet、ComfyUI 节点式工作流

## 工程实践

+ 前端 AI 开发：Vercel AI SDK、流式渲染（SSE）、生成式 UI、Zod 结构化校验、Transformers.js 浏览器端推理
+ LLMOps 评估观测：Ragas/TruLens 评估框架、LangSmith/LangFuse 可观测性工具、A/B 测试、成本优化
