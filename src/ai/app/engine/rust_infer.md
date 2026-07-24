---
title: Rust 推理引擎
order: 35
---

# Rust 推理引擎
Rust 生态中有多个用于大模型推理的框架和绑定，它们在性能、安全性和部署便利性上各有取舍。相比 Python 生态，Rust 推理引擎的核心优势在于零依赖部署（单二进制文件）、内存安全保证、以及与 Rust 系统编程生态的无缝集成。

## candle
candle 是 HuggingFace 推出的纯 Rust 推理框架，设计理念是"极简依赖、极快编译"。它不依赖 libtorch 或任何 C++ ML 库——核心计算使用 wgpu（跨平台 GPU 后端）或 CUDA（通过 cudarc crate 直接调用 CUDA API），CPU 后端基于纯 Rust 实现。编译时间约 30 秒（PyTorch 的编译可能需要数分钟到半小时）。

candle 支持主流模型架构——LLaMA、Mistral、Qwen、Phi、Gemma 等，以及文本生成（LLM）、文本嵌入（BERT/Jina）、文生图（Stable Diffusion）、语音识别（Whisper）。模型权重通过 HuggingFace 的 safetensors 格式加载，与 Python 生态共享同一份权重文件。

```rust
use candle_core::{Device, Tensor};
use hf_hub::api::sync::Api;
use candle_transformers::models::quantized_llama::ModelWeights;

let api = Api::new()?;
let model_id = "TheBloke/Llama-2-7b-GGUF";
let repo = api.model(model_id.to_string());
let tokenizer = repo.get("tokenizer.json")?;
let weights = ModelWeights::from_gguf(repo.get("model-q4_k_m.gguf")?, &Device::cuda_if_available(0)?)?;
```

candle 的劣势是生态尚不如 Python 成熟——模型支持度取决于社区的 Candle 实现，部分较新的架构（如 Qwen3.6 的特定变体）可能还没有现成的配置。但对于推理场景，特别是需要将模型嵌入 Rust 应用的场景（如桌面应用内嵌 LLM、边缘设备推理），candle 是目前最成熟的纯 Rust 方案。

## mistral.rs
mistral.rs 是专门为本地大模型推理优化的 Rust 框架，底层基于 candle。它的核心特色是**开箱即用的量化加速**和**ISQ（In-Situ Quantization，就地量化）**——在模型加载过程中实时执行量化，直接从 FP16 safetensors 转换为 INT4/INT8 格式运行，不需要预先准备 GGUF 或 AWQ 文件。

mistral.rs 支持所有主流量化格式——GGUF、GPTQ、AWQ、GGML，以及自定义的 ISQ 量化。推理后端可选 candle（纯 Rust）、libtorch（PyTorch C++ API）、CUDA（直接 GPU 推理），能够在几乎不写代码的情况下启动一个兼容 OpenAI API 的 HTTP 服务。

```bash
# 安装后直接启动：自动检测 GPU、量化
mistralrs-server -i plain -m TheBloke/Llama-2-7B-GGUF -a llama
```

```rust
use mistralrs::{IsqType, LoaderBuilder, DeviceMapSetting};

// 加载时自动 INT4 量化
let loader = LoaderBuilder::new(
    DeviceMapSetting::from_device(Device::cuda_if_available(0)?)
).build("microsoft/Phi-3-mini-4k-instruct", IsqType::Q4K)?;
```

mistral.rs 对个人用户的最大价值在于它的 ISQ——下载一个 FP16 模型文件，框架在加载时自动量化并运行。这意味着不需要再去 HuggingFace 找"某某模型的 GGUF 版本"——框架帮你做了量化这一步。

## llama-cpp-rs
llama-cpp-rs 是对 llama.cpp C++ 库的 Rust FFI 安全绑定。它不是纯 Rust 实现——底层仍然依赖 llama.cpp 的 C++ 代码——但通过安全的 Rust 类型系统包装了所有 C API。继承了 llama.cpp 的全部功能：GGUF 量化模型、GPU 层卸载、投机解码、KV Cache 量化，以及完整的采样参数控制。

llama-cpp-rs 的核心价值在于让 Rust 开发者能够以惯用方式使用 llama.cpp——通过 Rust 的 ownership 保证资源的正确释放，通过 Result/Option 处理异常情况，不需要手动管理 C 内存。对于已经在用 llama.cpp 但有 Rust 二次开发需求的用户，这是最直接的路径。

```rust
use llama_cpp_rs::{LlamaModel, LlamaInference, LlamaContextParams};

let model = LlamaModel::load_from_file("qwen3.6-8b-Q4_K_M.gguf", None)?;
let mut ctx = model.new_context(LlamaContextParams::default())?;
let output = ctx.inference("你好，请介绍一下你自己。", None)?;
```

劣势在于编译依赖 llama.cpp 的 C++ 工具链（CMake + C++ 编译器），失去了纯 Rust 项目的单一 `cargo build` 体验。此外 llama.cpp 的 API 相对底层，没有 candle/mistral.rs 那种开箱即用的高级抽象。

## burn
burn 是 Rust 生态中的完整深度学习框架，定位对标 PyTorch。它不是专门的推理引擎，但提供了从训练到推理的端到端能力。

burn 的最大特色是多后端抽象——同一份模型代码可以在不同后端上运行，切换只需改一行配置。当前支持的后端包括 WGPU（跨平台 GPU，Vulkan/Metal/DX12 自动适配）、CUDA（通过 cudarc crate）、CPU（纯 Rust 实现，通过矩阵乘法优化库加速），以及正在实验中的 ROCm 后端。

burn 的前向传播图会自动优化——算子融合、内存复用、常量折叠等。训练方面支持自动微分（autodiff）、多种优化器（Adam/AdamW/SGD）、学习率调度器、数据加载管道。模型可以通过 ONNX 导出与其他生态互通。

```rust
use burn::backend::wgpu::Wgpu;
use burn::tensor::Tensor;

type Backend = Wgpu;

let x = Tensor::<Backend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]]);
let y = x.matmul(&x.transpose());
```

burn 的劣势是模型 zoo 尚不如 PyTorch 丰富——需要自己实现大部分模型架构，或从 ONNX/HuggingFace 导入预训练权重。目前在推理场景中不如 candle 或 mistral.rs 成熟，但如果需要边训练边推理（如在线微调 + 服务），burn 是 Rust 生态中最接近 PyTorch 的方案。

## 选型指南
| 场景                              | 推荐工具     | 原因                                |
| --------------------------------- | ------------ | ----------------------------------- |
| 想最快跑起来，不写代码            | Ollama       | Python 封装的 llama.cpp，一命令启动 |
| Rust 应用内嵌 LLM                 | candle       | 纯 Rust、编译快、依赖少             |
| Rust 应用中做推理服务             | mistral.rs   | 自动量化、兼容 OpenAI API           |
| 已有 llama.cpp 经验，用 Rust 开发 | llama-cpp-rs | FFI 调用，功能完整                  |
| 边训练边推理                      | burn         | 完整 DL 框架                        |

对于个人本地部署场景，最推荐的是 mistral.rs——ISQ 就地量化消除了"找 GGUF 版本"这一步，且启动命令极简。如果需要在 Rust 桌面应用或嵌入式设备中嵌入推理能力，candle 的纯 Rust 实现和低依赖特性不可替代。
