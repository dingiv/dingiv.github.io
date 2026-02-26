---
title: DeepSpeed
---

# DeepSpeed

DeepSpeed 是微软开发的分布式训练优化库，通过 ZeRO（Zero Redundancy Optimizer）技术突破显存瓶颈，使得在有限 GPU 资源上训练大模型成为可能。其核心思想是将优化器状态、梯度和参数分片（sharding）到不同设备，消除数据并行中的冗余存储。

## ZeRO 优化

ZeRO 按优化激进程度分为三个阶段，逐步将显存占用从 $O(2)$ 降至 $O(1)$。

ZeRO-1 仅分片优化器状态。Adam 优化器需要为每个参数存储一阶矩和二阶矩，占用 $2 \times$ 参数量显存。对于 7B 模型，优化器状态需要约 28GB（FP16 参数权重 14GB + FP32 一阶矩 28GB + FP32 二阶矩 28GB）。ZeRO-1 将这些状态按 GPU 数量 $n$ 切分，每张卡仅需存储 $1/n$ 的优化器状态，将显存从 70GB 降至约 42GB。

ZeRO-2 进一步分片梯度。在反向传播完成后，每个 GPU 产生的梯度原本需要完整存储，用于更新参数。ZeRO-2 将梯度按切分方式分散存储，更新时通过 ReduceScatter 聚合分片梯度到对应 GPU，完成本地参数更新后，再通过 AllGather 同步最新参数。这又将显存降低约 25%。

ZeRO-3 是最激进的优化，连模型参数本身都进行分片。每张卡仅持有 $1/n$ 的参数，前向传播时通过 AllGather 动态获取所需层的参数，用完即释放。这带来两个挑战：一是通信量增加（每层都需要 AllGather），二是实现复杂度高（需要精确控制参数获取和释放时机）。但收益巨大：7B 模型在 8 张 A100（40GB）上，ZeRO-3 可将每卡显存占用降至约 18GB。

## CPU 卸载

ZeRO-Infinity 进一步将不常用的数据卸载到 CPU 内存甚至 NVMe SSD。优化器状态仅在参数更新时需要，可常驻 CPU；梯度和参数在计算时才加载到 GPU。这实现了"用时间换空间"，训练速度下降 30-50%，但可将训练 1T 参数模型的 GPU 需求从 1024 张 A100 降至 128 张。对于预算有限的团队，这是打破硬件天花板的关键技术。

## 使用方式

```python
import deepspeed
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b")
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
    "zero_optimization": {"stage": 3, "offload_optimizer": {"device": "cpu"}},
}
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config)
```

DeepSpeed 与 HuggingFace DeepSpeed 集成良好，只需在训练脚本中添加 `--deepspeed ds_config.json` 即可启用，无需修改模型代码。
