---
title: TGI
---

# TGI

Text Generation Inference (TGI) 是 HuggingFace 开发的生产级推理框架，专为 Transformers 生态模型优化设计。它强调稳定性和可观测性，是 HuggingFace Inference Endpoints 的底层引擎。

## 核心特性

TGI 内置了 FlashAttention 实现，通过分块计算减少显存访问，推理速度比标准 Attention 快 2-3 倍。对于长序列（8K+），性能提升更为显著。

量化支持是 TGI 的强项。它原生支持 BNB（bitsandbytes）、GPTQ、AWQ 等量化格式，加载量化模型只需一行配置。TGI 还提供了详细的量化精度 benchmark，帮助开发者选择合适的量化位宽（INT8、INT4 甚至 NF4）。

TGI 的可观测性非常完善。内置 Prometheus metrics（请求延迟、吞吐量、显存使用）、日志结构化输出、健康检查接口，这些都是生产环境必需的功能。相比 vLLM 的研究导向，TGI 更注重工程实践。

## 使用方式

```bash
model=meta-llama/Llama-2-7b
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $volume:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id $model \
  --quantize bitsandbytes-nf4 \
  --max-total-tokens 4096
```

TGI 以 Docker 容器形式部署，通过参数控制量化、并发、显存限制。这比 vLLM 的 Python API 更适合生产环境，因为容器化部署易于管理和扩展。

## 与 vLLM 的选择

TGI 和 vLLM 是当前最流行的两大推理引擎。TGI 的优势在于稳定性和可观测性，适合企业级部署；vLLM 的优势在于性能和吞吐量，适合高并发场景。HuggingFace 的模型生态系统与 TGI 深度集成，加载 Hub 上的模型、 tokenizer、配置文件都开箱即用。对于非 HuggingFace 格式的模型（如 PyTorch `.pt` 文件），vLLM 的兼容性更好。
