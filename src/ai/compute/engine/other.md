# 其他推理引擎

除了 vLLM 和 DeepSpeed 外，AI 领域还有多个重要的推理和训练引擎。这些引擎各有特色：llama.cpp 专注本地部署和 CPU 推理，Megatron-LM 是超大规模模型训练的基石，TensorRT-LLM 追求极致性能，TGI 注重生产环境稳定性，SGLang 针对多轮对话场景优化。

## llama.cpp

llama.cpp 是 Georgi Gerganov 开发的轻量级 LLM 推理引擎，最初为在 MacBook 上运行 LLaMA 模型而设计，现已发展成为支持 CPU/GPU 混合推理、跨平台部署的通用框架。它的核心价值是 democratization——让没有 GPU 的开发者也能运行大模型。

### GGUF 格式

GGUF (GPT-Generated Unified Format) 是 llama.cpp 定义的二进制模型格式。它将模型权重、tokenizer 词表、配置参数打包为单个文件，支持 mmap 内存映射加载，启动速度快。更重要的是，GGUF 原生支持量化——从 Q4_0（4-bit）到 Q8_0（8-bit）再到 F16（16-bit），可根据硬件和精度需求灵活选择。

量化过程简单到极致。给定 GGUF 格式的 FP16 模型，`llama-cli` 工具可一键量化为 Q4_K_M（4-bit，中等映射质量）或 Q5_K_S（5-bit，小模型）。量化算法基于 GPTQ 思想，通过最小化权重误差保持精度，但在实现上更轻量，无需额外的校准数据集。

### CPU 推理

llama.cpp 的核心优势是 CPU 推理。通过充分利用 SIMD 指令（AVX2、AVX512、ARM NEON），在 CPU 上的推理速度远超 PyTorch。例如，在 M2 MacBook 上，llama.cpp 运行 7B Q4 模型可达 30 tokens/秒，而 PyTorch 仅 5 tokens/秒。对于边缘设备、嵌入式系统、开发测试环境，CPU 推理是唯一选择。

### 混合推理

llama.cpp 支持 CPU + GPU 混合推理。将部分层卸载到 GPU，剩余层在 CPU 上计算，通过 Metal (Apple Silicon)、CUDA (NVIDIA)、ROCm (AMD) 后端加速。这平衡了显存占用和计算速度，适合显存有限但有 GPU 加速的设备。

混合推理的关键是层卸载策略。卸载太多层会爆显存，太少则加速不明显。llama.cpp 提供了 `--n-gpu-layers` 参数让用户手动调整。一般来说，7B 模型在 8GB 显存 GPU 上可卸载 20-30 层，推理速度提升 2-3 倍。

### 使用方式

llama.cpp 的设计哲学是简单。单个二进制文件包含所有功能，无需 Python 环境、无需 CUDA toolkit、无需复杂配置，下载即用。

```bash
# 量化模型
llama-cli --model llama-2-7b.gguf --quantize-output --out-file llama-2-7b-q4.gguf --quantize-type Q4_K_M

# 运行推理
llama-cli --model llama-2-7b-q4.gguf --prompt "Hello, world" --n-predict 100

# 启动 OpenAI 兼容服务
llama-server --model llama-2-7b-q4.gguf --port 8080
```

### 适用场景

llama.cpp 最适合资源受限环境：笔记本电脑、边缘设备、开发测试机器。对于生产环境的服务化部署，vLLM/TGI/TensorRT-LLM 更合适，因为它们的并发能力和吞吐量更高。但对于个人使用、离线部署、隐私敏感场景，llama.cpp 是最佳选择。llama.cpp 的另一个优势是跨平台，Windows、macOS、Linux、Android、iOS 全平台支持，使得它成为嵌入式 AI 应用的首选推理引擎。

## Megatron-LM

Megatron-LM 是 NVIDIA 开发的超大规模模型训练框架，首创了张量并行技术，并系统性地提出了 3D 并行策略。它是训练 GPT-3 175B、Megatron-Turing NLG 530B 等里程碑模型的基石。关于 3D 并行的详细实现原理，参见 [模型并行](./parallel)。

### 张量并行原理

张量并行的核心是将矩阵乘法算子切分到多个 GPU。对于 Transformer 的 MLP 层 $Y = XW$，其中 $X \in [B, S, H]$，$W \in [H, 4H]$。列并行将 $W$ 按列切分为 $W_1, W_2, \dots, W_n$，每张卡计算 $Y_i = XW_i$，最后通过 AllConcat 拼接 $Y = [Y_1, Y_2, \dots, Y_n]$。

多头注意力更适合张量并行：$n$ 个头天然可以分配到 $n$ 张卡，每张卡计算自己的 QKV 投影和注意力输出，最后通过 AllReduce 聚合。这种切分完全符合 Transformer 的数学结构，通信开销极小。

### Sequence Parallel

长序列训练时，KV Cache 和注意力计算在序列维度上的内存和计算压力巨大。Megatron-LM 提出的 Sequence Parallel 将序列维度也进行切分，配合 Ring Attention 将通信复杂度从 $O(n^2)$ 降至 $O(n)$。这使得 128K 上下文的 GPT-3 训练成为可能。

### 使用成本

Megatron-LM 的工程复杂度远高于 DeepSpeed 和 FSDP。它要求模型代码按照特定的并行模式重写，且不支持即插即用的模型加载。但对于训练千亿级以上参数的模型，其性能优化是无可替代的。NVIDIA 的 NGC 容器预装了 Megatron-LM，可直接在 Base Command 平台上启动训练。

## TensorRT-LLM

TensorRT-LLM 是 NVIDIA 开发的推理优化框架，通过 TensorRT 的图优化和算子融合能力，将 LLM 推理性能推向极致。它是 NVIDIA 在推理领域对抗 vLLM/TGI 的王牌。

### 核心技术

TensorRT-LLM 的核心是 TensorRT，一个深度学习推理优化器。TensorRT 解析 PyTorch/ONNX 模型后，构建计算图，然后进行一系列优化：层融合（LayerNorm + Residual → 单个 kernel）、精度校准（FP32 → FP16/INT8）、内核自动调优（针对不同 GPU 架构选择最优 CUDA kernel）。这些优化将推理延迟降低 15-30%。

对于 LLM，TensorRT-LLM 额外优化了 Attention 算子。通过 in-place 更新 KV Cache、masked softmax 融合、多头 attention 并行，将 Attention 的计算效率提升到接近理论峰值。

### INT4 量化

TensorRT-LLM 支持 AWQ（Activation-aware Weight Quantization）INT4 量化。AWQ 的核心洞察是只有 1% 的权重对量化敏感，这些权重保留高精度（FP16），其余 99% 量化为 INT4。这保持了模型精度的同时，将显存占用降低 75%，计算速度提升 2-3 倍。

量化过程高度自动化。给定 FP16 模型，TensorRT-LLM 自动计算每层的量化 scale、校准激活值范围、生成量化后的 engine 文件（`.engine`）。engine 文件是针对特定 GPU 架构编译的二进制，加载后直接执行，无需 JIT 编译，启动速度快。

### In-flight Batching

In-flight Batching 是 TensorRT-LLM 的独特优化。当某个序列生成结束时，立即插入新序列，无需等待当前 batch 完成。这与 vLLM 的连续批处理类似，但 TensorRT-LLM 的实现在 CUDA 层面完成，调度开销更低，适合极高并发场景（1000+ 并发请求）。

### 使用方式

TensorRT-LLM 的 API 偏底层，需要手动构建 engine、配置 tokenizer、管理 CUDA stream。这比 vLLM/TGI 的易用性差，但换来的是极致的性能控制力。

```bash
# 构建 INT4 量化引擎
python build.py --model_dir llama-2-7b --quantization int4_awq --output_dir llama-2-7b-int4

# 运行推理
python run.py --engine_dir llama-2-7b-int4 --max_output_len 512
```

### 适用场景

TensorRT-LLM 最适合 NVIDIA GPU 架构（A100、H100、L40S）上的高性能推理。对于非 NVIDIA GPU（如 AMD ROCm），TensorRT-LLM 不支持。TensorRT-LLM 的性能优势在 H100 上尤为明显，因为 NVIDIA 针对自家的 Transformer Engine（FP8 算子）做了深度优化。在 A100 上，TensorRT-LLM 与 vLLM 性能接近；在 H100 上，TensorRT-LLM 可领先 20-30%。

## TGI

Text Generation Inference (TGI) 是 HuggingFace 开发的生产级推理框架，专为 Transformers 生态模型优化设计。它强调稳定性和可观测性，是 HuggingFace Inference Endpoints 的底层引擎。

### 核心特性

TGI 内置了 FlashAttention 实现，通过分块计算减少显存访问，推理速度比标准 Attention 快 2-3 倍。量化支持是 TGI 的强项，它原生支持 BNB（bitsandbytes）、GPTQ、AWQ 等量化格式，加载量化模型只需一行配置。

TGI 的可观测性非常完善。内置 Prometheus metrics（请求延迟、吞吐量、显存使用）、日志结构化输出、健康检查接口，这些都是生产环境必需的功能。相比 vLLM 的研究导向，TGI 更注重工程实践。

### 使用方式

TGI 以 Docker 容器形式部署，通过参数控制量化、并发、显存限制。这比 vLLM 的 Python API 更适合生产环境，因为容器化部署易于管理和扩展。

```bash
model=meta-llama/Llama-2-7b
volume=$PWD/data

docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $volume:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id $model \
  --quantize bitsandbytes-nf4 \
  --max-total-tokens 4096
```

### 与 vLLM 的选择

TGI 和 vLLM 是当前最流行的两大推理引擎。TGI 的优势在于稳定性和可观测性，适合企业级部署；vLLM 的优势在于性能和吞吐量，适合高并发场景。HuggingFace 的模型生态系统与 TGI 深度集成，加载 Hub 上的模型、tokenizer、配置文件都开箱即用。对于非 HuggingFace 格式的模型（如 PyTorch `.pt` 文件），vLLM 的兼容性更好。

## SGLang

SGLang 是 vLLM 的同类竞品，由 UC Berkeley 的 SkyLab 实验室开发（与 vLLM 同一团队）。它的核心创新是 RadixAttention 和结构化生成缓存，专门针对多轮对话、复杂 prompt、工具调用等场景优化。

### RadixAttention

RadixAttention 是 SGLang 的核心优化技术。与 vLLM 的 PagedAttention 不同，RadixAttention 使用基数树来管理 KV Cache 的共享。这带来了两个优势：一是更高效的缓存复用（不同 prompt 的公共前缀可以共享 KV Cache），二是更低的内存开销（基数树的元数据比页表更小）。

对于多轮对话场景，用户通常会重复之前的对话内容。vLLM 需要重新计算这些内容的 KV Cache，而 SGLang 可以直接复用之前的缓存。这使得 SGLang 在长上下文、多轮对话场景下比 vLLM 快 2-3 倍。

### 结构化生成缓存

SGLang 的另一大特色是结构化生成缓存（Speculation with Verification Cache，简称 SVC）。对于 JSON 格式的输出、函数调用等结构化生成任务，SGLang 会缓存已生成的部分 KV Cache，当后续输出与缓存匹配时直接复用，跳过计算。

这使得 SGLang 在 JSON API、工具调用、RAG（检索增强生成）等场景下具有显著优势。例如，对于需要输出 JSON 格式的 API，SGLang 可以在生成 key 时缓存对应的 KV Cache，生成 value 时如果复用相同的 key 结构，则无需重新计算。

### 使用方式

SGLang 兼容 OpenAI API，可通过 `python -m sglang.server` 启动兼容服务。它还支持 OpenAI 的函数调用格式，只需在 prompt 中包含工具描述，SGLang 会自动解析并调用。

```python
from sglang import function_call, Runtime, GenerationConfig

runtime = Runtime(model_path="meta-llama/Llama-2-7b", tp_size=2)

# 基础推理
output = runtime.generate("Hello, world", GenerationConfig(max_tokens=100))

# 工具调用
@function_call
def get_weather(location: str):
    return f"The weather in {location} is sunny."
```

### 与 vLLM 的对比

SGLang 和 vLLM 都来自 UC Berkeley，但定位不同。vLLM 追求极致的吞吐量和通用性，适合高并发的在线服务；SGLang 追求特定场景的优化（多轮对话、结构化生成、工具调用），适合 Agent 应用和复杂 prompt 场景。

对于大多数通用推理场景，vLLM 的性能和生态更成熟。但对于 Agent 开发、RAG 应用、需要频繁调用工具的场景，SGLang 的针对性优化会带来显著收益。对于多轮对话、结构化输出、长文档处理等场景，SGLang 的性能可领先 vLLM 2-5 倍。但对于简单的单轮问答、高并发短 prompt 场景，vLLM 的吞吐量更高。
