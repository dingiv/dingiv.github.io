---
title: mistral.rs
order: 36
---

# mistral.rs

mistral.rs 是 Rust 生态中最成熟的大模型推理引擎之一，底层基于 HuggingFace 的 candle 框架。它最独特的特性是 ISQ（In-Situ Quantization，就地量化）——在模型加载过程中直接将 FP16 权重转换为 INT4/INT8 格式，不需要预先准备量化模型文件。你只需要下载原始的 safetensors 权重，框架在加载时自动完成量化。

## 架构

```
HTTP Server (OpenAI 兼容) / Python 绑定
         │
    ┌────▼────┐
    │ Pipeline│  ← 自动检测架构、加载权重
    │ + ISQ   │
    └────┬────┘
         │
    ┌────▼──────────────┐
    │   Compute Units   │  ← 硬件加速单元（可混合调度）
    └────┬──────┬───────┘
    ┌────┼──────┼───────┐
    ▼    ▼      ▼       ▼
  CUDA  Metal  ROCm    CPU
(NV)  (Apple) (AMD)  (通用)
```

CPU、CUDA、Metal、ROCm 不是互斥的"后端选项"而是可以**混合调度**的计算单元。例如把 embed 层放在 CPU 上、attention 层放在 GPU 上，两类计算同时进行。rust 二进制启动时自动检测可用硬件并选择加速单元，不需要手动配置。

## 底层 candle 的工作原理

candle 是 HuggingFace 推出的纯 Rust 张量计算库，设计目标是"极简依赖、极快编译"。它不依赖 libtorch——核心计算有三个后端：

- CPU 后端：纯 Rust，使用 BLAS（通过 accelerate 或 Intel MKL crate）加速矩阵乘法
- CUDA 后端：通过 cudarc crate 直接调用 CUDA Runtime API，绕过 libtorch。使用 cuBLAS 和 cuDNN 加速，内存管理通过 Rust 的所有权系统保证安全
- Metal 后端：通过 metal-rs 库直接在 Apple GPU 上计算，支持 M1/M2/M3 Mac 的本地推理

candle 的核心抽象是 `Tensor`，类似 PyTorch 的 `torch.Tensor` 和 NumPy 的 `ndarray`。所有运算——矩阵乘法、注意力、softmax、层归一化——都在这个类型上进行。candle 的自动微分支持有限（仅用于推理），不适合训练场景。对于推理，candle 的关键能力是直接在 GPU 上执行量化矩阵乘法（INT4/INT8 GEMM），这是它比纯 CPU 推理快一个数量级的原因。

```rust
// candle 的核心用法：创建张量、在 GPU 上运算
let a = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::new_cuda(0)?)?;
let b = Tensor::new(&[[5f32, 6.], [7., 8.]], &Device::new_cuda(0)?)?;
let c = a.matmul(&b)?;
// c = [[19, 22], [43, 50]]
```

## ISQ 就地量化的实现

ISQ 是 mistral.rs 最突出的特性。传统量化流程是：下载 FP16 模型 → 用 AWQ/GPTQ 工具量化 → 保存量化文件 → 加载量化文件推理。这个过程需要额外的时间和磁盘空间（量化工具的安装、量化过程的 GPU 资源，以及量化文件的存储）。ISQ 将量化嵌入到模型加载流程中，一步完成。

ISQ 的原理基于权重矩阵的 channel-wise 量化。对于每个线性层权重 $W \in \mathbb{R}^{d_{out} \times d_{in}}$，对每个输出通道计算 scale $\Delta$ 和零点 $z$：

$$
\Delta = \frac{\max(W_i) - \min(W_i)}{2^b - 1}, \quad z = \text{round}\left(\frac{-\min(W_i)}{\Delta}\right)
$$

$W_{quant} = \text{round}(W / \Delta + z)$，限制在 $[0, 2^b-1]$ 范围内。$b$ 为量化位宽（4 或 8）。

mistral.rs 的 ISQ 实现了多种量化方案：Q4K（4-bit K-quant，性能和体积的最佳平衡）、Q8_0（8-bit 整数量化）、以及 FP8（NVIDIA H100 原生的 8-bit 浮点格式，损失比 INT8 更小但需要特定硬件）。量化直接在 CPU 或 GPU 上执行，不需要生成中间文件。

ISQ 的量化时间约为：7B 模型 1-2 分钟（CPU 量化）、70B 模型 8-12 分钟（CPU 量化）。量化精度损失：Q4K 通常导致 PPL 退化 0.3-0.8（相较于 FP16）。

## 硬件加速

**CUDA（NVIDIA）**是功能最完整的加速单元。支持 FlashAttention V2/V3（长上下文推理的关键加速）、PagedAttention（高效 KV Cache 管理的核心）、cuDNN 加速 kernel、FP8 blockwise 量化。多 GPU 张量并行通过 NCCL（NVIDIA 集合通信库）实现——同一节点内 NVLink 互联的 GPU 使用 NCCL 做 AllReduce。NCCL 也支持跨节点通信，可扩展到多机集群。

**Metal（Apple Silicon）**在 M1/M2/M3 Mac 上实现 GPU 加速。使用 Metal Shading Language（MSL）编写的 kernel，包括优化的 paged attention 实现和量化 GEMM。v0.5.0 之后 ISQ 量化时间在 Metal 上缩短了 30 倍。M1 Max 32 核 GPU 上 7B 模型的推理速度约 20-30 token/s。

**ROCm（AMD）**支持 AMD 显卡。与 CUDA 单元类似，通过 ROCm 的 HIP 运行时调用 GPU kernel。编译需要 ROCm 工具链。对于安装了 ROCm 的 AMD 用户，mistral.rs 可以直接利用 AMD GPU 进行推理加速。

**CPU**始终可用，底层通过 candle 的纯 Rust 实现或使用 BLAS 库加速矩阵乘法。适合没有 GPU 的场景或卸载部分轻量计算层（如 embed/lm_head）。

## 使用方式

最简启动——下载模型 + ISQ 量化 + 启动 HTTP 服务：

```bash
# 安装 mistral.rs
cargo install mistralrs-server

# 从 HuggingFace 下载模型、自动 INT4 量化、启动服务
mistralrs-server -i plain \
  -m microsoft/Phi-3.5-mini-instruct \
  --isq Q4K
# 服务启动后：http://localhost:8080
```

Rust API 使用：在 Rust 应用中嵌入推理。适合桌面应用或嵌入式设备。

```rust
use mistralrs::{IsqType, DeviceMapSetting, LoaderBuilder, 
    Messages, SamplingParams};

// 创建模型加载器：自动检测架构、指定 ISQ 量化
let loader = LoaderBuilder::new(
    DeviceMapSetting::from_device(
        candle_core::Device::cuda_if_available(0)?
    )
).use_flash_attn(false)
 .build(
    "microsoft/Phi-3.5-mini-instruct".to_string(),
    IsqType::Q4K
 )?;

// 对话
let messages = Messages::new()
    .add_system("你是一个有帮助的助手。")
    .add_user("用 Rust 写一个快排函数");

let response = loader.chat(&messages, SamplingParams::default())?;
println!("{}", response);
```

HTTP 服务模式：兼容 OpenAI API，任何能调 `POST /v1/chat/completions` 的客户端都可以无缝对接。

```bash
mistralrs-server --port 8080 -i plain \
  -m Qwen/Qwen3.6-8B-Instruct --isq Q4K
```

```python
# Python 客户端（用 openai 库直接调）
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
resp = client.chat.completions.create(
    model="qwen3.6-8b",
    messages=[{"role": "user", "content": "你好"}]
)
```

## 多 GPU 支持

mistral.rs 支持张量并行（TP）将模型权重拆分到多张 GPU。底层通信使用两种方案：NCCL（NVIDIA 集合通信库，节点内 NVLink 和跨节点 InfiniBand 均可使用）和 Ring 后端（基于点对点传输的环状通信，不依赖 NCCL，适合没有 NVLink 的 PCIe 互联场景）。NCCL 的通信效率更高（特别是 NVLink 互联时），Ring 的兼容性更好（不需要 CUDA Toolkit 中的 NCCL 组件）。

除了权重层对 GPU 的映射，还支持 P2P 设备映射（GPU 之间直接传输数据而不经过 CPU）和多节点 NCCL（将张量并行扩展到多台机器）。

```rust
use mistralrs::{DeviceMapSetting, LoaderBuilder, IsqType};

// 按层映射到不同 GPU
let loader = LoaderBuilder::new(
    DeviceMapSetting::Map(vec![
        ("model.layers.0-15", Device::new_cuda(0)?),
        ("model.layers.16-31", Device::new_cuda(1)?),
    ])
).build("Qwen/Qwen3.6-32B-Instruct", IsqType::Q4K)?;
```

## 与 llama.cpp 的对比

| 维度       | mistral.rs                    | llama.cpp          |
| ---------- | ----------------------------- | ------------------ |
| 语言       | Rust                          | C++                |
| 量化       | ISQ（加载时自动量化）         | 需预先转换 GGUF    |
| 后端       | candle/CUDA/Metal/libtorch    | CPU + CUDA + Metal |
| 编译       | cargo build，纯 Rust 最简     | CMake + C++ 工具链 |
| 模型格式   | safetensors（原始格式）       | GGUF（专用格式）   |
| API        | Rust API + HTTP + Python 绑定 | CLI + HTTP server  |
| 生态成熟度 | 较新（2024-）                 | 成熟（2023-）      |

核心差异在于模型加载方式。llama.cpp 需要先将模型转为 GGUF——这是额外的一步，且 GGUF 文件通常需要在 HuggingFace 上找到特定用户提供的预量化版本。mistral.rs 直接读取 HuggingFace 上的原始 safetensors，ISQ 在加载时自动量化。对于下载模型后想立刻跑起来的体验，mistral.rs 比 llama.cpp 少一步"找 GGUF"的摩擦。

性能方面，两个框架在相同 GPU 和相同量化精度下的推理速度差异很小（5-10% 以内），通常可以忽略。如果追求绝对最快的推理速度，用 CUDA 后端的 mistral.rs 或有 nvidia-specific 优化的 llama.cpp CUDA 版本都可以。
