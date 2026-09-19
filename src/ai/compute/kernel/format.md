---
title: 模型格式
order: 50
---

# 模型格式
模型格式是 AI 模型的存储和交换标准，定义了模型权重、架构、元数据如何组织和序列化。一个良好的模型格式应具备**可移植性**（跨平台兼容）、**安全性**（防止恶意代码注入）、**效率**（加载速度快、占用空间小）、**可扩展性**（支持新架构和新特性）。本文聚焦**训练和分发阶段**最常用的 PyTorch 原生与 HF 风格两种格式的存储设计；ONNX 作为跨框架交换格式详见 [ONNX](./onnx)，GGUF 作为 llama.cpp 生态的本地推理格式详见 [GGUF](./gguf)。

## 各格式对比
| 格式             | 推出方             | 特点                             | 适用场景           | 详细介绍 |
| ---------------- | ------------------ | -------------------------------- | ------------------ | -------- |
| PyTorch .pt/.pth | Meta               | PyTorch 原生格式，支持完整计算图 | PyTorch 训练/推理  | — |
| HF 风格          | Hugging Face       | config + safetensors，生态标准   | 开源模型分发       | — |
| ONNX             | Microsoft/Meta     | 框架无关，跨平台推理             | 跨框架部署         | [ONNX](./onnx) |
| GGUF             | llama.cpp          | 单文件分发、mmap 友好、量化内置  | 本地部署、边缘设备 | [GGUF](./gguf) |

选型的第一性问题是**模型规模**——小模型（亿级参数）和大模型（百亿到千亿级）对格式的需求完全不同，由此衍生出两个时代、两条技术路线。2016-2020 年的传统深度学习（ResNet、YOLO、BERT）走 ONNX 静态计算图路线（详见 [ONNX](./onnx)），2023 年后的 LLM 时代走 HF safetensors 的"薄容器"路线。理解这条分界线比记住任何一个格式的细节更重要。

## PyTorch 原生格式
PyTorch 的原生模型格式包括 `.pt`、`.pth`、`.pkl`（三者本质相同），使用 Python 的 pickle 模块序列化。这种格式可以保存完整的模型对象（包括架构、权重、优化器状态、训练状态），加载后可直接使用，无需重新定义模型结构。

```python
# 保存完整模型（包含架构和权重）
torch.save(model, "model.pth")

# 保存仅权重（state_dict）
torch.save(model.state_dict(), "weights.pth")

# 加载权重（需要先定义模型结构）
model = MyModelClass()
model.load_state_dict(torch.load("weights.pth"))
```

PyTorch 格式的优势是**完整性**和**灵活性**——可以保存任意 Python 对象（包括自定义层、优化器、学习率调度器）。但这也是它的劣势：pickle 格式存在安全风险（反序列化可执行任意代码）、跨版本兼容性差（PyTorch 版本升级可能导致旧模型无法加载）、文件体积大（包含不必要的元数据）。

生产环境中建议使用 `state_dict` 方式保存权重，而非保存完整模型。原因有二：一是安全性（避免执行任意代码），二是兼容性（模型结构由代码定义，而非依赖序列化对象）。对于跨平台部署，建议转换为 ONNX 或 HF safetensors 格式。

## HF 风格模型格式
HF 风格模型格式（Hugging Face pretrained model directory format）没有独立的正式品牌名称，而是 transformers 库中 PreTrainedModel 和 PreTrainedConfig 类的标准保存/加载目录规范，**已成为开源大模型生态的事实标准**。

从系统视角看，HF 风格模型文件夹是一个**带自描述元数据的二进制发布包**——包含配置（定义架构）、权重（模型参数）、分词器（文本接口）三类核心文件，通过标准化的目录结构和 JSON 配置实现自描述，使得任何工具都可以解析和加载模型。

### 文件夹结构
**权重文件（真正的数据）**

| 文件                               | 说明                                      |
| ---------------------------------- | ----------------------------------------- |
| `model.safetensors`                | 单文件模型的权重（推荐格式）              |
| `model-00001-of-000xx.safetensors` | 大模型的分片权重（默认每片 50GB）         |
| `model.safetensors.index.json`     | 分片索引文件（映射 tensor 名 → 分片文件） |
| `pytorch_model.bin`                | 旧格式（pickle，不推荐）                  |

**配置文件（模型元数据）**

| 文件                       | 说明                                                                                                |
| -------------------------- | --------------------------------------------------------------------------------------------------- |
| `config.json`              | **最关键的文件**，定义模型架构（层数、隐藏层维度、注意力头数等）。AI 引擎通过读取它来实例化模型类。 |
| `generation_config.json`   | 推理默认参数（max_length、temperature、top_p 等）                                                   |
| `preprocessor_config.json` | 多模态模型的预处理配置（如图像/音频处理）                                                           |

**Tokenizer 文件（文本接口）**

| 文件                      | 说明                     |
| ------------------------- | ------------------------ |
| `tokenizer.json`          | 统一的分词器格式（推荐） |
| `tokenizer_config.json`   | 分词器配置和特殊 token   |
| `vocab.json / merges.txt` | BPE 分词器的词汇表       |
| `tokenizer.model`         | SentencePiece 二进制词典 |

**其他文件**

| 文件                  | 说明                           |
| --------------------- | ------------------------------ |
| `adapter_config.json` | PEFT/LoRA 适配器配置           |
| `README.md`           | 模型卡片（模型描述、使用许可） |

```python
model.save_pretrained(
    save_directory,
    max_shard_size="50GB",          # 自动分片阈值
    safe_serialization=True,        # 使用 safetensors（默认推荐）
    push_to_hub=False,              # 是否直接推送到 HF Hub
    variant=None                    # 如 "fp16" → pytorch_model.fp16.bin
)
```

几乎所有推理引擎（vLLM、TGI、TensorRT-LLM 等）都原生支持或通过少量转换支持 HF 格式。模型作者训练完后用一次 `save_pretrained`，多个下游工具就能直接加载。HF Hub 上数万模型都遵循这个规范，确保生态兼容。

### SafeTensors
SafeTensors 是 Hugging Face 推出的安全张量序列化格式，旨在替代 PyTorch 的 pickle 格式。它只保存张量数据（名称 + 形状 + dtype + 字节流），不包含任何可执行代码，彻底杜绝了反序列化攻击的风险。

SafeTensors 采用 **Header + Data** 的二进制结构：前 8 字节是 JSON 元数据的长度（无符号整数），接着是 JSON 描述的元数据（每个张量的名称、形状、dtype、偏移量），最后是纯二进制张量数据（连续存储）。这种设计使得 AI 引擎可以使用 Linux 的 `mmap` 系统调用将文件直接映射到地址空间，实现零拷贝加载——`open` 文件后读取 Header 获取每个 Tensor 的偏移量，再 `mmap` 数据部分到虚拟内存，最后直接将磁盘数据指针通过 `cudaMemcpyAsync` 泵入显存。

| 特性         | Pickle (.bin/.pt)        | SafeTensors                       |
| ------------ | ------------------------ | --------------------------------- |
| 反序列化速度 | 慢（需要 Python 解释器） | 极快（纯磁盘 IO/mmap）            |
| 安全性       | 差（可执行任意代码）     | 安全（仅包含数据）                |
| 内存开销     | 高（加载时有内存拷贝）   | 极低（支持零拷贝）                |
| 跨语言支持   | 难（强绑定 Python）      | 易（C/C++/Rust 均有轻量级解析库） |

```python
from safetensors.torch import save_file, load_file

# 保存权重
save_file({"weight1": tensor1, "weight2": tensor2}, "model.safetensors")

# 加载权重
weights = load_file("model.safetensors")
```

SafeTensors 已成为 Hugging Face 模型分发的标准格式。自 2023 年起，HF Hub 上新上传的模型默认使用 safetensors 而非 pytorch_model.bin。主流推理引擎（vLLM、TGI、TensorRT-LLM）都优先支持 safetensors 格式。**SafeTensors 是 AI 领域的 ELF 文件格式**——它规范了权重的排布，实现了高性能、高安全性的模型加载。

### transformers 三件套
transformers 是 Hugging Face 推出的大模型加载库，它本质上是一个**模型格式规范**。深度学习框架（如 PyTorch、TensorFlow）提供底层算子和自动微分能力，但如何定义模型架构、如何加载权重、如何做推理，这些都需要开发者自己编写大量样板代码。transformers 库把这些工作标准化——模型定义、权重加载、推理接口全部统一，开发者只需几行代码就能使用预训练模型。

在 transformers 出现之前，复现一篇论文的模型是极其痛苦的过程。论文作者通常只会发布训练好的权重文件，以及一段可能在特定框架版本上才能运行的代码。模型结构定义分散在各个 GitHub 仓库，API 风格五花八门，权重文件格式互不兼容。transformers 库通过统一的接口和规范的模型格式，将模型复现成本从数天降低到数分钟。

transformers 的核心贡献是把**模型结构、权重、推理接口**三者标准化。开发者不再需要"实现模型"，只需"加载模型"。这种转变的意义在于将模型从"算法"变成了"基础设施"——就像调用 HTTP API 一样简单。

三件套将模型抽象为三个独立组件：

| 组件      | 职责                                                                 | 文件来源                              |
| --------- | -------------------------------------------------------------------- | ------------------------------------- |
| Config    | 存储模型超参数和架构信息（层数、隐藏层维度、注意力头数、词汇表大小） | config.json                           |
| Tokenizer | 文本与 token 之间的双向转换，存储分词规则                            | tokenizer.json、tokenizer_config.json |
| Model     | 纯粹的神经网络实现，接收 token ID 输出 logits                        | 基于架构代码实例化                    |

这三者解耦的设计使得同一个模型结构可以使用不同的预训练权重，同一个 Tokenizer 可以服务于多个模型，同一个模型可以轻松切换不同的任务头。

### Auto 系列
Auto 系列类（`AutoTokenizer`、`AutoModel`、`AutoModelForCausalLM` 等）是 transformers 库工程化的集大成体现。传统的做法是需要明确知道使用的是哪种模型架构，然后导入对应的类。但 Auto 系列允许开发者完全不关心模型类型，只需提供模型名称或路径，库会自动推断应该加载哪个类。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

这种设计的革命性在于将模型选择权交给了权重文件，而不是代码。当你使用 `AutoModel` 加载一个本地目录时，库会读取目录中的 `config.json` 文件，根据 `model_type` 字段自动选择对应的模型类。

### 推理与训练
transformers 库的推理接口设计简洁到极致。调用 `model.generate()` 可以自动处理采样策略、温度参数、top-k/top-p 过滤等细节。对于文本分类、问答、命名实体识别等常见任务，库还提供了 `pipeline` 高级 API，一行代码就能完成从原始文本到模型输出的全流程。

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("Hugging Face is amazing!")
# 输出: [{'label': 'POSITIVE', 'score': 0.9998}]
```

训练方面，transformers 提供了 `Trainer` 类封装了训练循环的样板代码：自动批处理、混合精度训练、梯度累积、学习率调度、日志记录、检查点保存等。开发者只需定义数据集和评估指标，`Trainer` 会处理其余的工程细节。

### Hugging Face 生态
Hugging Face 起初是一家专注于聊天机器人开发的初创公司，但在 2019 年转型为 AI 开源工具和模型托管平台。如今它已成为大模型时代最重要的基础设施之一，被称为"AI 界的 GitHub"。

| 组件            | 功能                            |
| --------------- | ------------------------------- |
| 模型托管平台    | 类似 GitHub，托管模型权重和代码 |
| transformers 库 | 模型存储、加载和运行的规范      |
| Datasets 库     | 数据加载和预处理                |
| Evaluate 库     | 评估指标统一接口                |
| Spaces          | 演示环境（免费 GPU）            |

截至 2024 年，HF Hub 上已有数十万个模型被上传分享，涵盖自然语言处理、计算机视觉、音频处理、多模态等各个领域。

### 工程实践
1. **缓存管理**：初次使用时模型会下载到 `~/.cache/huggingface`，生产环境建议指定 `cache_dir` 参数
2. **内存优化**：`device_map="auto"` 会自动将模型分层分配到 CPU 和 GPU，超大规模模型可结合 `accelerate` 库使用模型并行
3. **Tokenizer 细节**：不同模型的特殊 token 不同（如 BERT 的 `[CLS]`、GPT 的 `<endoftext>`），批量推理时设置 `padding=True` 和 `truncation=True`

## 格式选择决策树
| 场景          | 推荐格式                        | 理由                 |
| ------------- | ------------------------------- | -------------------- |
| PyTorch 训练  | state_dict + safetensors        | 安全、高效           |
| 模型分发      | HF 风格（config + safetensors） | 生态标准、兼容性强   |
| 本地 CPU 推理 | GGUF                            | 量化友好、内存占用小 |
| 跨框架部署    | ONNX                            | 框架无关、硬件支持广 |
| 边缘设备部署  | GGUF 或 TFLite                  | 量化、低资源优化     |

模型格式转换是常见的工程需求。Hugging Face 提供了 `transformers` 库的转换工具，支持从 PyTorch、TensorFlow、JAX 等格式互转。对于 ONNX 转换，可使用 `torch.onnx.export` 或 `onnxruntime` 的转换工具。对于 GGUF 转换，可使用 llama.cpp 的 `quantize` 工具将 HF 模型转换为 GGUF 格式。

转换路径的设计哲学值得展开。训练阶段用 PyTorch 原生格式保留灵活性，发布时转 HF safetensors（生态兼容），要跨框架部署再转 ONNX（解耦框架），要 CPU+Apple Silicon 部署再转 GGUF（量化 + 内存映射）。每一步转换都解决特定场景的问题，没有"一统天下"的银弹格式——这正是格式多样性的合理之处。
