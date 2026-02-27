---
title: DeepSpeed
---

# DeepSpeed

DeepSpeed 是微软开发的分布式训练优化库，通过 ZeRO（Zero Redundancy Optimizer）技术突破显存瓶颈，使得在有限 GPU 资源上训练大模型成为可能。DeepSpeed 的核心思想是：**消除数据并行中的冗余存储**，将优化器状态、梯度和参数分片到不同设备，从而将显存占用从 $O(2)$ 降至 $O(1)$。

## 核心卖点

DeepSpeed 的核心卖点可以概括为四点：**极致的显存优化**、**弹性训练能力**、**与 HuggingFace 深度集成**、**成熟的工程实践**。

显存优化是 DeepSpeed 的第一大卖点。大模型训练的显存瓶颈主要体现在三个方面：模型权重、梯度、优化器状态。对于 7B 模型，FP16 权重需要 14GB，FP32 梯度需要 28GB，FP32 优化器状态（Adam 的一阶矩和二阶矩）需要 56GB，总计约 100GB 显存，远超单卡容量。DeepSpeed ZeRO-3 通过分片存储，将每卡显存占用降至约 18GB（8 卡训练），这使得在有限 GPU 资源上训练大模型成为可能。

弹性训练是 DeepSpeed 的第二大卖点。DeepSpeed 支持弹性训练（elastic training），可以在训练过程中动态增加或减少 GPU 数量，无需重启训练。这对于云环境尤为重要——当某些 GPU 被其他任务抢占时，可以动态缩小训练规模；当有更多 GPU 可用时，可以动态扩大训练规模。弹性训练提高了训练的容错性和资源利用率。

与 HuggingFace 深度集成是 DeepSpeed 的第三大优势。DeepSpeed 与 Transformers、Diffusers 库无缝集成，只需一行代码 `--deepspeed ds_config.json` 即可启用。这降低了使用门槛，使得研究者和工程师可以快速上手。DeepSpeed 还提供了丰富的文档和教程，以及预配置的配置模板，使得从零开始训练大模型变得相对简单。

成熟的工程实践是 DeepSpeed 被广泛采用的第四大原因。DeepSpeed 已被用于训练 GPT-3 175B、BLOOM 176B、Megatron-Turing NLG 530B 等超大模型，经过了生产环境的验证。其稳定性和可靠性已在多个大规模训练任务中得到证明，这对于企业级应用至关重要。

## ZeRO 优化

ZeRO 是 DeepSpeed 的核心技术，按优化激进程度分为三个阶段：ZeRO-1、ZeRO-2、ZeRO-3。

ZeRO-1 分片优化器状态。Adam 优化器需要为每个参数存储一阶矩（momentum）和二阶矩（variance），占用 $2 \times$ 参数量显存。对于 7B 模型，优化器状态需要约 42GB（FP16 权重 14GB + FP32 一阶矩 28GB）。ZeRO-1 将这些状态按 GPU 数量 $n$ 切分，每张卡仅需存储 $1/n$ 的优化器状态。

ZeRO-1 的实现依赖于 `torch.distributed.ReduceScatter` 原语。在反向传播完成后，每张 GPU 计算自己负责的参数分片的梯度，然后通过 ReduceScatter 将梯度聚合到对应的 GPU。每张 GPU 更新自己持有的优化器状态分片，然后通过 AllGather 同步最新参数。这样每张 GPU 只需存储 $1/n$ 的优化器状态，将显存从 42GB 降至约 21GB（8 卡训练）。

ZeRO-2 进一步分片梯度。在反向传播完成后，每个 GPU 产生的梯度原本需要完整存储，用于更新参数。ZeRO-2 将梯度按切分方式分散存储，更新时通过 ReduceScatter 聚合分片梯度到对应 GPU，完成本地参数更新后，再通过 AllGather 同步最新参数。

ZeRO-2 的关键设计是**通信与计算的重叠**。在反向传播计算层 $i$ 的梯度时，同时同步层 $i-1$ 的梯度，将通信隐藏在计算中。这种流水线设计将通信开销从串行的 30% 降至并行的 10% 以下。

ZeRO-3 是最激进的优化，连模型参数本身都进行分片。每张卡仅持有 $1/n$ 的参数，前向传播时通过 AllGather 动态获取所需层的参数，用完即释放。这带来两个挑战：一是通信量增加（每层都需要 AllGather），二是实现复杂度高（需要精确控制参数获取和释放时机）。

ZeRO-3 的实现依赖于 **parameter sharding** 和 **gradient checkpointing**。parameter sharding 将模型参数按维度切分，每张 GPU 只持有部分参数。gradient checkpointing 在前向传播时不保存所有中间激活值，只保存必要的 checkpoint，反向传播时重新计算激活值。这以计算换显存，将显存占用从 $O(n)$ 降至 $O(\log n)$。

## CPU 卸载

ZeRO-Infinity 是 DeepSpeed 的 CPU 卸载技术，将不常用的数据卸载到 CPU 内存甚至 NVMe SSD。优化器状态仅在参数更新时需要，可常驻 CPU；梯度和参数在计算时才加载到 GPU。这实现了"用时间换空间"，训练速度下降 30-50%，但可将训练 1T 参数模型的 GPU 需求从 1024 张 A100 降至 128 张。

CPU 卸载的关键挑战是**数据传输开销**。PCIe 带宽（64 GB/s）远低于 GPU 显存带宽（2TB/s），频繁的 GPU-CPU 数据传输会成为瓶颈。ZeRO-Infinity 通过预取（prefetching）和流水线（pipelining）来隐藏数据传输开销。在计算层 $i$ 的时候，预先将层 $i+1$ 的参数加载到 GPU，同时同步层 $i-1$ 的梯度，实现三级流水。

CPU 卸载的另一个挑战是**CPU 内存带宽**。大规模模型的参数量可达 1T+，即使使用 CPU 内存卸载，也需要数百 GB 内存。ZeRO-Infinity 支持 NVMe SSD 卸载，但 SSD 的带宽（5-7 GB/s）远低于 CPU 内存（100-200 GB/s），只适合存储不频繁访问的数据（如优化器状态）。

## 混合精度训练

DeepSpeed 原生支持混合精度训练（mixed precision），包括 FP16、BF16、FP32。FP16（half precision）将显存减半，带宽翻倍，计算速度提升 2-4 倍（Tensor Core）。BF16（brain float 16）与 FP16 的指数位相同（8 位），但尾数位减少到 7 位，数值范围更大，不需要 loss scaling。对于训练稳定性敏感的大模型，BF16 是更安全的选择。

混合精度训练需要解决**数值稳定性**问题。FP16 的动态范围有限（最大值 65504），容易出现溢出。DeepSpeed 通过动态 loss scaling 自动调整 loss scale，在溢出前降低 loss，在溢出后提高 loss。这保证了训练的数值稳定性，同时充分发挥 FP16 的速度优势。

DeepSpeed 还支持 FP8（8-bit floating point）训练。FP8 是 H100 GPU 的新特性，理论性能是 FP16 的两倍。DeepSpeed 的 FP8 训练仍处于实验阶段，需要仔细调优 loss scale 和 learning rate。

## 分布式训练框架

DeepSpeed 不仅是显存优化库，还是完整的分布式训练框架。它提供了分布式训练的完整工具链：数据加载器（DataLoader）、模型并行（Megatron-DeepSpeed）、弹性训练（elastic training）、容错恢复（fault tolerance）。

DeepSpeed 的数据加载器支持分布式数据采样，可以自动将数据分片到多个 GPU，支持预取（prefetching）和数据增强（data augmentation）。模型并行结合 Megatron-LM 的张量并行技术，支持超大模型（100B+ 参数）的训练。

弹性训练允许在训练过程中动态增加或减少 GPU 数量。当某些 GPU 故障或被抢占时，DeepSpeed 会自动调整训练规模，将受影响的 checkpoint 迁移到健康的 GPU，然后恢复训练。这对于长周期训练（如 GPT-3 175B 需要训练数周）至关重要，避免了因硬件故障导致训练前功尽弃。

## 使用方式

DeepSpeed 的使用方式非常简单，与 HuggingFace Transformers 深度集成：

```python
import deepspeed
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b")
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-4}
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    },
    "fp16": {
        "enabled": True
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
}

model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config)
```

训练时通过命令行参数启用 DeepSpeed：

```bash
python train.py \
    --deepspeed ds_config.json \
    --model_name_or_path facebook/opt-13b \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --fp16 \
    --output_dir ./output
```

DeepSpeed 的配置文件 `ds_config.json` 可以包含数十个参数，包括 train_batch_size（全局 batch size）、train_micro_batch_size_per_gpu（单卡 micro batch size）、gradient_accumulation_steps（梯度累积步数）、zero_optimization（ZeRO 优化）、fp16（混合精度）、bf16（brain float）、optimizer（优化器配置）、scheduler（学习率调度）等。

## 与 FSDP 的对比

DeepSpeed 和 FSDP（PyTorch 2.0 原生的分布式训练方案）是当前两大主流训练框架。DeepSpeed 的优势在于成熟度高、文档丰富、支持 CPU 卸载；FSDP 的优势在于与 PyTorch 生态深度集成、API 更简洁、支持 torch.compile。

从性能角度看，DeepSpeed 和 FSDP 在 ZeRO-3 场景下性能相当。但从功能角度看，DeepSpeed 支持更多特性（如 CPU 卸载、弹性训练、混合专家），适合超大规模训练（100B+ 参数）。FSDP 则更适合已有的 PyTorch 代码迁移，API 更简洁，与 torch.compile 无缝集成。

选择哪个框架需要考虑团队的技术栈和需求。如果团队已经熟悉 PyTorch 且模型规模适中（<100B），FSDP 是更简单的选择；如果需要训练超大规模模型（100B+）或需要 CPU 卸载，DeepSpeed 是更成熟的选择。
