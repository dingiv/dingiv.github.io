---
title: 服务部署
order: 35
---

# 服务部署
模型推理最终要暴露为服务供应用调用。本地跑通模型只是第一步，生产环境的服务部署需要考虑并发控制、健康检查、监控告警和成本优化。

## 单实例部署
FastAPI 是构建推理服务的常见选择，原生支持异步和自动生成 API 文档。一个最小的推理服务包含模型加载、请求验证、推理调用、响应返回几个步骤。生产环境需要考虑请求队列（避免并发过多导致 OOM）、超时控制（长请求占用资源）、健康检查（Kubernetes 存活探针）。

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    ml_models["model"] = load_model()
    yield
    # 关闭时清理资源
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    return ml_models["model"].generate(request.messages)
```

预热机制在服务启动时加载模型并执行一次推理，避免首次请求的冷启动延迟（模型加载可能需要几十秒）。健康检查端点返回模型加载状态和显存使用情况，供 Kubernetes 判断 Pod 是否就绪。

## 多实例与负载均衡
高并发场景下，每个 GPU 运行一个推理实例，前面部署 Nginx 或 Kubernetes Service 做负载均衡。推理服务通常无状态，可以水平扩展。需要注意模型加载时间较长，Pod 不能轻易重启——设置合理的 `terminationGracePeriodSeconds`。

DP（数据并行）多实例是最简单的扩展方式——每张 GPU 独立运行完整模型，卡间零通信，前面挂负载均衡器轮询分发请求。前提是单卡能装下模型，且并发量足够撑起多实例。

## 监控与告警
推理服务的健康指标包括请求延迟（P50/P99）、吞吐量（QPS）、显存使用率、GPU 利用率和错误率。Prometheus + Grafana 是标准方案，vLLM、TGI 等框架内置了 metrics 端点。

需要设置的告警：延迟突增（可能是资源竞争或模型切换）、显存溢出（batch size 过大或 KV Cache 膨胀）、请求失败率上升（模型崩溃或 GPU 掉卡）、成本突增（可能的滥用或配置错误）。告警分级别——紧急问题（服务不可用）立即响应，性能退化可次日分析。

## 成本优化
Token 消耗直接影响运营成本。优化方向：简单任务用更小的模型（分类/摘要用 7B 而非 70B）、优化 Prompt 长度（去除冗余指令）、压缩上下文（用摘要替代原始历史）、缓存重复请求（相同输入直接返回结果）。

级联策略：先用小模型尝试，置信度低时再调用大模型。保证质量的前提下能降低 50-70% 的成本。批量处理离线任务（文档摘要、批量翻译），共享固定成本（模型加载、系统开销），提升 GPU 利用率。

模型部署的完整技术栈见 [AI 算力概览](/ai/compute/stack)，GPU 选型见 [GPU AI 部署参考](gpu-ai)。
