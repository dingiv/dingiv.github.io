---
title: 大模型应用
order: 30
---

# 大模型应用开发技术
大模型应用开发基于已有的 Web2.0 前后端开发技术栈，分别在前后端引入新的组件，帮助推动大模型的能力落地到实际的生产环节中。

## 技术栈
+ 推理引擎：能够使用推理引擎部署运行大模型，包括 Ollama、vLLM、TGI 等工具，掌握参数调优和模型微调技术
+ 向量数据库：使用向量数据库存取向量类型数据，为 RAG 应用提供语义检索能力，理解 HNSW 索引算法和相似度计算
+ 提示词工程：理解提示词的表达方式对于模型生成影响，掌握 CoT、Few-shot 等设计模式
+ 会话管理：理解模型上下文窗口机制与限制、模型调用的无状态性、记忆机制和对话状态管理
+ 应用框架：使用 LangChain 等框架快速构建 LLM 应用，理解 Models、Prompts、Chains、Agents、Memory 等核心组件
+ 工具调用：掌握 Function Calling 机制，定义 JSON Schema 让 LLM 主动调用外部 API

## 典型应用
+ RAG 检索增强生成：结合向量检索和 LLM 生成，构建私有知识问答系统
+ GraphRAG 知识图谱增强：结合 Neo4j 等图数据库，支持多跳推理和关系查询
+ Agent 智能体：ReAct 模式、工具调用、多 Agent 协作（AutoGen）
+ 自动化脚本：图色脚本、AI GUI 辅助、低代码平台
+ 媒体生成：Stable Diffusion、ComfyUI 节点式工作流

## 工程实践
+ 前端 AI 开发：Vercel AI SDK、流式渲染、生成式 UI、Transformers.js 浏览器端推理
+ LLMOps 评估观测：Ragas/TruLens 评估框架、LangSmith/LangFuse 可观测性工具、A/B 测试
