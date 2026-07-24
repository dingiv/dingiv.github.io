---
title: 推理优化
order: 30
---

# 推理优化
模型跑起来之后，优化目标从"能不能跑"变为"跑得快不快"。推理加速涉及三个层面：算法层减少计算量（投机解码、注意力压缩）、系统层提升硬件利用率（FlashAttention、Chunked Prefill、连续批处理）、数据层压缩存储（KV Cache 量化）。这些技术可以叠加使用，效果累乘。

## 投机解码
投机解码是最"免费"的加速手段——不改变模型权重、不损失精度、不增加通信开销，仅靠算法设计将自回归生成的串行瓶颈转化为并行验证。

原理：用一个极小的 draft model（1B-3B）快速预测 $k$ 个候选 token，大模型一次 forward pass 并行验证这 $k$ 个 token。通过的就保留，不通过的从失败点由大模型重新生成。draft model 准确率超过 80% 时，$k$ 个 token 的生成时间 ≈ 1 次大模型 forward + $k$ 次小模型 forward——而小模型的单次 forward 比大模型快 5-10 倍。

投机解码的前提是 draft model 和主模型的 token 分布高度对齐。同系列模型（Qwen3.6-32B + Qwen3.6-1.8B draft）共享词表和训练数据，通过率可达 85-95%。跨系列搭配（Llama-7B 做 Qwen-32B 的 draft）通过率可能只有 50-60%，比不用投机还慢。

```bash
# llama.cpp: draft model 投机
./llama-server -m qwen3.6-32b-Q4_K_M.gguf \
  --draft-model qwen3.6-1.8b-Q4_K_M.gguf \
  --speculative-tokens 8 -ngl 999

# SGLang: EAGLE 投机引擎（共享 embedding，通过率 90%+）
sglang serve Qwen/Qwen3.6-32B --speculative-algorithm EAGLE \
  --speculative-num-tokens 8
```

EAGLE 是比独立 draft model 更高效的方案——使用与主模型共享 embedding/lm_head 的轻量 draft head，分布对齐度更高。DFlash 进一步将 draft 阶段从"逐 token 生成"改为"一次性并行扩散"，完全消除 draft 的串行瓶颈。

本地单 GPU 运行投机解码需要为 draft model 预留额外显存——Qwen3.6-1.8B Q4_K_M 约 1.2GB，通常可以接受。

## FlashAttention 与注意力优化
自注意力机制的计算量和 KV Cache 的显存占用是长上下文推理的两大瓶颈。FlashAttention 通过分块计算和重排内存访问模式，将注意力计算的显存带宽利用率从 20% 提升到 80%+。

FlashAttention-2 需要 sm_80（Ampere）及以上 CUDA 算力，大多数现代推理框架默认启用。FA3 针对 Hopper 架构（H100）进一步优化了异步执行。长上下文（8K+）场景下，FA 对推理速度的影响最为显著——注意力计算可能占推理时间的 30-40%，FA2 将其降到 10% 以下。

MLA（Multi-Head Latent Attention，DeepSeek 提出）是算法层更激进的优化——利用低秩矩阵分解将 KV Cache 压缩到极小潜空间。相比 GQA 减少 KV 头数（4-8 倍压缩），MLA 实现 90%+ 的压缩比。KV Cache 从几十 GB 降到几 GB，单卡能容纳数倍的上下文和 batch size。MLA 是模型架构特性，不需要部署框架做额外配置——下载支持 MLA 的模型（DeepSeek-V2/V3），部署即自动生效。

GQA（Grouped Query Attention）是更广泛采用的折中方案——多个 Query 头共享一组 KV 头（如 Llama-3 使用 8 组）。KV Cache 缩小 4-8 倍，是现代开源模型能在 24GB 级显卡上运行的前提。

## Chunked Prefill 与连续批处理
Prefill（处理输入 prompt）是计算密集型，Decode（逐 token 生成）是访存密集型。传统推理将两者串行处理——一个长 prompt 的 Prefill 可能占用 GPU 数秒，期间所有其他请求的 Decode 被挂起。

Chunked Prefill 将长 prompt 切分成小块，与 Decode 步拼装到同一个 batch 中同时提交 GPU。Prefill 的矩阵计算填满 CUDA 核心，Decode 的访存利用显存带宽——计算与访存硬件级重叠，GPU 利用率显著提升。vLLM 和 SGLang 默认启用。

连续批处理（Continuous Batching）允许在生成过程中动态插入新请求。不同于静态批处理等待固定数量请求后一起处理，连续批处理在单个请求生成完成后立即释放资源给队列中的下一个请求。对 API 服务（多用户并发）影响巨大，吞吐量提升可达 10 倍。

## KV Cache 优化
KV Cache 的显存占用随上下文长度线性增长，长上下文场景下可能超过模型权重本身。优化方向：

KV Cache 量化——将 K 和 V 缓存从 FP16 压缩到 4-8 bit。TurboQuant 通过旋转处理分散异常值通道，4-bit 量化后 32K 上下文的 KV Cache 从 ~16GB 降到 ~4GB。llama.cpp 支持 `--cache-type-k q8_0 --cache-type-v q8_0`。

Prefix Caching——当多个请求共享相同前缀（system prompt、RAG 检索结果、工具定义）时，自动复用已计算的 KV Cache。SGLang 的 RadixAttention 将 prefix caching 做到自动化和细粒度——任何前缀匹配都能命中缓存，不仅是完整 prompt 匹配。多轮对话中 system prompt 和早期轮次的 KV Cache 复用，prefill 开销降低 5 倍以上。

## 本地部署的技术落地方阵
在本地单机/小集群的 PCIe 拓扑约束下，优化技术的落地优先级：

模型选择时优先挑自带 GQA/MLA 的模型（Qwen 2.5、DeepSeek-V3）——KV Cache 压缩是免费的架构红利。启动服务时确保 FlashAttention-2 已启用（现代框架默认开启）。长上下文场景开启 KV Cache 量化（4-8 bit，显存节省 > 权重量化收益）。多轮对话/RAG 场景优先用 SGLang（RadixAttention 自动缓存前缀）。单用户场景开启投机解码（EAGLE 或 draft model，1.5-2.5x Decode 提速）。

不推荐在本地落地 Prefill/Decode 分离——它需要在节点间通过 RDMA 高速网络（400Gbps+ InfiniBand）传输 GB 级 KV Cache，本地 10G/25G 网卡传输时间远超计算时间，且低并发场景下硬件利用率暴跌。Chunked Prefill 在单机内已实现了 Prefill 和 Decode 的计算-访存重叠，是 PD 分离在本地场景的最优替代。

多卡并行和投机解码分别解决显存和速度问题，可以叠加使用——PP 把模型装下，投机解码把速度拉满。
