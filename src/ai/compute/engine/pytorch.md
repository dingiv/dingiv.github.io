# Pytorch


## Torch
Torch 框架为了使用硬件加速计算，规定各个 GPU 厂商的封装层，将各家的硬件 API 进行屏蔽，从而让上层的数据科学家无需触及糟心而混乱的 GPU 生态，专注于数据训练即可，在调用 torch 的 API 时，torch 将帮助识别当前的硬件环境，使用对应的硬件进行加速，常见的硬件平台包括：

| 平台              | 后端                              | 底层调用                |
| ----------------- | --------------------------------- | ----------------------- |
| NVIDIA GPU        | CUDA                              | cuBLAS、cuDNN、TensorRT |
| AMD GPU           | ROCm                              | hipBLAS、MIOpen         |
| Apple M 芯片      | MPS （Metal Performance Shaders） | Metal Compute           |
| Intel GPU / CPU   | XPU  （oneAPI）                   | oneDNN                  |
| Huawei Ascend NPU | Ascend C                          | CANN                    |
| Google TPU        | XLA                               | HLO / MLIR              |
| CPU               | Native                            | OpenMP / MKL / BLAS     |

包括国产的华为昇腾 NPU 生态。


# FSDP

Fully Sharded Data Parallel (FSDP) 是 PyTorch 2.0 原生引入的分布式训练方案，功能上对标 DeepSpeed ZeRO，但与 PyTorch 生态深度集成，通过 torch.compile、torch.autograd 实现自动分片和通信算子融合。

## 与 ZeRO 的关系

FSDP 的设计借鉴了 ZeRO-3，同样将参数、梯度、优化器状态全部分片。但实现上更贴近 PyTorch 的设计哲学：通过 `torch.distributed.fsdp.FullyShardedDataParallel` 包装模块，自动处理前向传播时的 AllGather 和反向传播后的 ReduceScatter。这种模块化设计使得迁移现有代码非常简单——只需将 `nn.DataParallel` 替换为 `FSDP`，无需重构模型定义。

## 混合精度

FSDP 原生支持 BF16（Brain Float 16）混合精度训练。BF16 与 FP16 的指数位相同（8 位），但尾数位减少到 7 位，相比 FP16 的 10 位尾数，精度略有损失但数值范围更大，不需要 loss scaling。对于训练稳定性敏感的大模型，BF16 是更安全的选择。PyTorch 2.0 引入的 `torch.float16` + `GradScaler` 也支持 FP16 训练，但需要手动调整 loss scale。

## 通信优化

FSDP 的通信调度经过精心设计。前向传播时，参数 AllGather 与前一层计算流水线重叠（overlap）；反向传播时，梯度 ReduceScatter 与下一层反向流水线重叠。这种"通信计算隐藏"策略将通信开销从串行的 30% 降至并行的 10% 以下。

## 使用方式

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
model = FSDP(model, sharding_strategy="FULL_SHARD", mixed_precision="bf16")
```

FSDP 的 API 比 DeepSpeed 更简洁，适合已熟悉 PyTorch DDP 的开发者。但 ZeRO 的成熟度和文档丰富度仍然更高，对于生产环境的大规模训练，DeepSpeed 仍是首选。
