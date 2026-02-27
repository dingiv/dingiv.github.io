---
title: 算子融合
---

# 算子融合
算子融合通过将多个连续算子合并为一个 CUDA kernel，减少显存读写和 kernel 启动开销，是提升模型性能的有效手段。融合后显存访问次数从 $O(n)$ 降至 $O(1)$（$n$ 为算子数量），kernel 启动次数从 $n$ 次降至 1 次，对于逐元素操作密集的模型（如 ResNet 的残差连接、Transformer 的 LayerNorm），性能提升可达 2-5 倍。

## 融合原理
深度学习模型由数百个算子组成，如卷积、ReLU、Add、LayerNorm 等。标准执行方式是逐算子调用：1) 读入输入张量，2) 计算，3) 写出输出张量。每个算子都需要一次显存读写，而显存带宽（A100 为 2TB/s）远低于计算速度（312 TFLOPS FP16），导致性能受限于显存而非计算单元。

算子融合的核心思想是**消除中间结果的显存读写**。例如对于 $y = \text{LayerNorm}(x + z)$，标准实现需要：1) 读 $x, z$，2) 计算 $x + z$ 写入 $t$，3) 读 $t$，4) 计算 $\text{LayerNorm}(t)$ 写入 $y$。融合后：1) 读 $x, z$，2) 计算 $x + z + \text{LayerNorm}$ 写入 $y$。显存读写从 4 次减少到 2 次，且 kernel 启动次数从 2 次减少到 1 次。

融合的关键是**识别可融合的算子组合**。一般来说，逐元素操作（Add, Mul, ReLU, LayerNorm）容易融合，因为它们对每个元素独立计算，无需跨元素同步。聚合操作（Sum, Max, Softmax）也可以融合，但需要实现高效的归约算法（如 warp shuffle）。矩阵乘法、卷积等复杂算子也可以与前后算子融合（如 Conv + Bias + ReLU → ConvBiasReLU），但需要手工编写 kernel。

## 自动融合
PyTorch 2.0 的 `torch.compile` 会自动识别可融合的算子并生成融合后的 kernel。基于 TorchDynamo（捕获 Python 字节码）、AoTAutograd（自动微分）、Inductor（codegen）的编译栈，`torch.compile` 可将常见的算子组合（如 Linear + ReLU → LinearReLU）融合，无需手动编写 CUDA 代码。

```python
import torch

def model(x):
    x = torch.nn.functional.linear(x, weight, bias)
    x = torch.nn.functional.relu(x)
    x = torch.nn.functional.layer_norm(x, normalized_shape=(128,))
    return x

compiled_model = torch.compile(model)
output = compiled_model(input_tensor)
```

`torch.compile` 的融合能力受限于算子的兼容性。如果模型包含自定义算子或第三方库（如 FlashAttention），可能无法融合。此时需要手动编写 CUDA kernel 或使用 NVFuser（PyTorch 的融合 kernel 编译器）。

TensorRT 的融合能力更强。它通过解析 ONNX 模型，构建计算图，然后应用一系列融合规则：LayerNorm + Residual → LayerNormResidual，Conv + Bias + Activation → ConvBiasActivation。TensorRT 还支持跨多个节点的融合，如 Conv + Pooling + Concat → ConvPoolingConcat，这是 `torch.compile` 目前做不到的。

## 手动融合
对于性能关键的算子，手动编写 CUDA kernel 可获得最优性能。以 LayerNorm 为例，标准实现需要三次 kernel 调用（计算均值、计算方差、归一化），手动融合后可减少到一次 kernel 调用，且在 kernel 内部使用 shared memory 存储中间结果，避免全局显存访问。

```cpp
// 手动融合的 LayerNorm kernel（简化版）
__global__ void layer_norm_fusion(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int batch, int seq, int hidden, float eps) {

    // 计算均值（warp shuffle 加速）
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += input[idx];
    }
    sum = warp_reduce_sum(sum);
    float mean = sum / hidden;

    // 计算方差（复用之前的 shared memory）
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float diff = input[idx] - mean;
        var_sum += diff * diff;
    }
    var_sum = warp_reduce_sum(var_sum);
    float var = var_sum / hidden;

    // 归一化（与前两步融合，无需中间结果）
    float std = sqrtf(var + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        output[idx] = (input[idx] - mean) / std * weight[i] + bias[i];
    }
}
```

手动融合的缺点是开发成本高、维护困难。CUDA kernel 的调试需要 Nsight Compute、cuda-gdb 等专用工具，且不同 GPU 架构（Ampere vs Hopper）的最优配置不同，需要针对每种架构调优。

## 融合的限制
算子融合并非万能。首先，融合会增加 kernel 的复杂度和编译时间，过度融合可能导致寄存器压力（register pressure）、shared memory 不足，反而降低性能。其次，融合后的 kernel 可复用性差，ConvBiasReLU 只能用于 Conv + Bias + ReLU 的组合，其他组合需要重新编写 kernel。

最后，融合需要考虑数值稳定性。例如 Softmax + CrossEntropy 的融合（LogSoftmax）在数值上等价，但 Softmax 在大指数时容易溢出，需要特殊处理（减去最大值）。融合时需要保证数值精度不变，否则可能导致模型精度下降。

## 融合工具
NVFuser 是 PyTorch 2.0 引入的融合 kernel 编译器，可将多个逐元素操作融合为单个 kernel。它基于 TVM（Tensor Virtual Machine）的 IR（中间表示），支持自动调优（针对不同 GPU 架构选择最优的 block size、tilling 策略）。N VFuser 目前支持 LayerNorm、Dropout、Softmax 等常见算子的融合，但不如 TensorRT 成熟。

Triton 通过高层抽象简化了融合 kernel 的编写。开发者无需编写 CUDA，只需用 Python 描述计算逻辑，Triton 编译器自动生成融合后的 CUDA kernel。这降低了手动融合的门槛，同时保持了接近 CUDA 的性能。

```python
# Triton 实现的融合 LayerNorm
import triton
import triton.language as tl

@triton.jit
def layer_norm_fusion(x, weight, bias, y, mean, rstd, stride, eps, BLOCK_SIZE: tl.constexpr):
    # 计算 LayerNorm（融合均值、方差、归一化）
    x = tl.load(x + offset)
    mean = tl.sum(x, axis=0) / x.shape[0]
    var = tl.sum((x - mean) ** 2, axis=0) / x.shape[0]
    y = (x - mean) / tl.sqrt(var + eps) * weight + bias
    tl.store(y + offset, y)
```

Triton 的抽象级别高于 CUDA，但低于 PyTorch，适合有一定 CUDA 基础的开发者。对于完全不了解 GPU 编程的开发者，`torch.compile` 或 TensorRT 是更简单的选择。
