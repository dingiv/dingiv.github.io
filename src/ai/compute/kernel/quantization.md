---
title: 量化
---

# 量化
量化是将模型权重和激活从高精度（FP32、FP16）转换为低精度（INT8、INT4）的技术，通过牺牲少量精度换取显存占用和计算速度的大幅降低。对于资源受限的部署场景（边缘设备、移动端、显存有限的服务器），量化是不可或缺的优化手段。

## 量化基础
量化的数学定义是将连续的浮点数映射到离散的整数。对于对称量化，公式为 $x_{\text{quant}} = \text{round}(x / s)$，其中 $s$ 是 scale（缩放因子），将 FP32 范围映射到 INT8 范围 [-128, 127]。反量化为 $x_{\text{dequant}} = x_{\text{quant}} \times s$。

非对称量化引入零点（zero point），支持不对称的值域：$x_{\text{quant}} = \text{round}(x / s + z)$，其中 $z$ 是零点偏移。非对称量化在权重分布不均匀时（如 ReLU 后的激活值，全为正）精度更高，但计算稍复杂。

量化误差来源于两个因素：**精度损失**（INT8 仅 256 个离散值，FP32 是连续值）和**溢出截断**（超出范围的值被截断到 [-128, 127]）。前者是不可避免的舍入误差，后者可通过选择合适的 scale 和 zero point 减少超出范围的概率。

## 量化方法
PTQ（Post-Training Quantization，训练后量化）是最简单的方案，直接对训练好的模型进行量化，无需重新训练。PTQ 分为动态量化和静态量化：动态量化在推理时根据激活值的范围动态计算 scale，精度高但推理慢；静态量化使用校准数据集预先计算激活值的 scale，推理快但精度略低。GPTQ、AWQ、SpQR 是 PTQ 的代表，通过最小化权重误差或激活误差来保持精度。

QAT（Quantization-Aware Training，量化感知训练）在训练过程中模拟量化误差，让模型适应量化带来的精度损失。具体做法是在前向传播时插入 fake quantize 算子（模拟量化和反量化的误差），反向传播时通过 Straight-Through Estimator（STE）近似量化操作的梯度。QAT 的精度通常高于 PTQ，但需要重新训练，成本较高。

## LLM 量化
大语言模型的量化有独特挑战。Transformer 的激活值范围动态变化——不同 prompt、不同位置的值域差异大，静态量化容易溢出；LayerNorm 和 Softmax 的数值稳定性对量化敏感，需要特殊处理；大模型的参数量决定了显存占用是部署的最大瓶颈（7B 模型 FP16 需要约 14GB），量化是解决这个瓶颈的核心手段。

下面重点展开目前工业界使用最广泛的两种 LLM 量化方法：GPTQ 和 AWQ。

### GPTQ
// TODO: 基本介绍一下 GPTQ 这个名字


GPTQ（Frantar et al., 2023, ICLR）并不是凭空产生的技术，而是沿着一条长达三十年的二阶优化研究脉络逐步演化而来：OBD（LeCun, 1990，用二阶信息做剪枝）→ OBS（Hassibi et al., 1993，用 Hessian 矩阵计算权重移除后的补偿）→ OBQ（Frantar et al., 2022，将 OBS 从剪枝扩展到量化）→ GPTQ（将 OBQ 扩展到 175B 规模的 LLM）。

GPTQ 的核心目标可以用一个简洁的公式表达：对每一层，寻找量化后的权重矩阵 $\widehat{W}$，使得该层的输出误差最小化：

$$
\argmin_{\widehat{W}} \ ||WX - \widehat{W}X||_2^2
$$

OBS 提供了一个关键的洞察：当你量化（或移除）一个权重 $w_q$ 时，可以通过调整剩余的权重来补偿输出误差，而不是让误差累积。补偿量有闭式解——$\delta_q = -\frac{w_q - \text{quant}(w_q)}{[H^{-1}]_{qq}} \cdot H^{-1}_{:,q}$，其中 $H = 2XX^T$ 是校准数据在权重空间的 Hessian 矩阵。每量化一个权重后，用这个公式更新所有尚未量化的权重，"分摊"量化误差。

OBQ 的问题在于计算复杂度是 $O(d_{row} \cdot d_{col}^3)$，对 7B 模型来说需要数年的计算时间。GPTQ 做了三个关键改进使其可用。

第一个改进是**固定顺序替代贪心选择**。OBQ 每一步都挑选"量化后误差最小"的权重先量化（贪心策略），这需要昂贵的全局排序。GPTQ 发现：只要从左到右按固定顺序量化，让前面的误差被后面的大量权重吸收，最终效果与贪心策略相当甚至更好——因为大模型中未量化的权重足够多，有足够的自由度来补偿早期误差。这一步将复杂度从 $\Theta(d_{row} \cdot d_{col}^3)$ 降到 $O(\max(d_{row} \cdot d_{col}^2, d_{col}^3))$。

第二个改进是**Cholesky 分解保证数值稳定性**。大模型的 Hessian 矩阵直接求逆会出现严重的数值不稳定性（矩阵接近奇异）。GPTQ 改为预计算 $H^{-1}$ 的 Cholesky 分解 $G = \text{Cholesky}(H^{-1})^T$，所有后续的补偿更新都通过 G 来完成，避免了对 $H^{-1}$ 的重复操作和累积误差。

第三个改进是**分块处理与延迟批量更新**（Lazy Batch Update）。GPTQ 将权重列按块（Block，通常 B=128）处理。块内的每一列量化后立即更新块内的后续列；但块外（更右边）的列不立即更新——等整个块处理完后，一次性批量更新所有剩余列。这个设计精妙地利用了 GPU 的算力特征：将多次小规模的内存操作合并为一次大规模的矩阵乘法，把原本内存受限的操作变成了计算受限的操作。一个 175B 的模型在单张 GPU 上约 4 小时完成量化，而 OBQ 需要数年。

工程落地层面，GPTQ 的输出是 per-channel 的 INT4 权重——每个输出通道有独立的量化 scale，而非全局共享一个 scale，这保留了不同通道的动态范围差异。在精度表现上，LLaMA-65B 从 FP16 的 WikiText-2 PPL 3.53 变为 INT4 的 3.84（仅退化 0.31），模型大小从约 130GB 降至约 35GB。显存受限场景下的单 batch 推理可获得 3-4.5 倍加速——因为在 batch=1 时推理是显存带宽瓶颈（memory-bound）而非算力瓶颈，模型变小直接等于读取变快。

GPTQ 的局限性也需要了解。它需要一个能放下完整 FP16 模型的 GPU 来做量化（因为需要逐层加载并计算 Hessian），这意味着要量化一个 70B 的模型，你仍然需要一张能装下 70B FP16 模型的 GPU（约 140GB）。量化过程对校准数据有一定过拟合倾向——校准数据（通常是从 C4 中采样的 128 段文本）如果与推理场景的数据分布差异大，精度退化会明显加剧。

### AWQ
// TODO: 基本介绍一下 AWQ 这个名字

AWQ（Lin et al., MLSys 2024）的出发点建立在一个关键的实验观察之上：在所有权重通道中，只有约 0.1%-1% 的通道对模型质量有显著影响——但这些"关键通道"不是由权重本身的大小决定的，而是由激活值的大小决定的。通过激活值的分布来识别关键通道，效果显著优于通过权重大小来识别（后者接近随机选择）。

直觉上这很好理解：对于激活值 $X$ 很大的通道，$W \cdot X$ 的结果受 $W$ 的精度影响很大——一个小的量化误差被大的激活值放大后会产生显著的输出偏差。而对于激活值接近零的通道，即使权重被大幅量化，对最终输出的影响也微不足道。

基于这个观察，AWQ 的做法不是使用混合精度（将关键通道保留在 FP16），而是使用一种更硬件友好的方案——**等价变换**（Equivalent Transformation）。对于每个关键通道，在量化前将其权重乘以一个 scale 因子 $s > 1$（放大权重，使其在量化网格中占据更多的离散层级），同时将对应的输入激活值除以 $s$（缩小激活值以保持 $W \cdot X$ 不变）。

这个操作的巧妙之处在于：量化网格的绝对宽度不变，但放大后的权重分布在更多量化层级上，相对量化误差减小了约 $1/s$ 倍。同时，因为只有极少数的通道被放大（约 1%），量化 scale $\Delta$ 基本不受影响（$\Delta' \approx \Delta$）。而激活值被缩小后，量化误差在激活侧的传播也被抑制——相当于在送入下一层之前将"噪声"缩小了 $1/s$。

scale 因子 $s$ 的搜索是 AWQ 的精简之处。它不逐个通道搜索最优 scale，而是将问题降维到单个超参数 $\alpha \in [0, 1]$：

$$
s = s_X^{\alpha}
$$

其中 $s_X$ 是每个通道的平均激活值大小。$\alpha^* = \argmin_\alpha \mathcal{L}(s_X^\alpha)$。这只需要在 $[0, 1]$ 区间内做 20 个点的网格搜索，选定最优 $\alpha$ 后，每个通道的 scale 自动确定。整个过程不需要 Hessian 计算、不需要迭代补偿更新、不需要昂贵的矩阵分解。

AWQ 相比 GPTQ 的优势体现在多个维度：
- 量化速度：AWQ 快 2-3 倍，因为省去了 Hessian 计算和逐列补偿更新
- 校准数据需求：AWQ 仅需约 16 段文本（~32K tokens）即可获得稳定结果，GPTQ 通常需要 128-192 段
- 跨域鲁棒性：AWQ 对校准数据与推理数据分布不一致的容忍度更高（GPTQ 的 Hessian 会过拟合校准数据的统计特征）
- 量化时显存占用：AWQ 在量化过程中不需要存储 Hessian 的 Cholesky 分解，显存峰值更低

在精度上两者接近，但 AWQ 在 4-bit 场景下通常略优或持平于 GPTQ，在多领域测试中展现更好的一致性。AWQ 的限制在于目前仅支持 INT4 量化（GPTQ 支持 2/3/4/8-bit），且仅在 weight-only 量化（W4A16，即权重 INT4、激活保持 FP16）的场景下验证充分。

AWQ 配套的推理引擎 TinyChat 在消费级设备上实现了显著的加速。通过将反量化操作融合到 GEMM kernel 的主循环中（不写出中间 FP16 结果）、针对 ARM NEON 指令集优化权重的 SIMD 解包（32 个 INT4 权重仅需 3 条 SIMD 指令完成解包）、以及 kernel 融合（LayerNorm + QKV 投影 + Attention + KV Cache 更新合并为一个 kernel），TinyChat 在单张 GPU 上实现 Llama-2-7B 的 3.7 倍推理加速，让 70B 模型能在移动端 GPU 上运行、13B 模型能在 8GB 显存的笔记本 GPU 上运行。

### AWQ vs GPTQ 选择指南
选择 AWQ 的场景：只需要 4-bit 量化、量化速度优先、校准数据有限或推理数据域不确定、量化设备的显存受限（无法装下完整 FP16 模型做 GPTQ 的 Hessian 计算）。

选择 GPTQ 的场景：需要 4-bit 以外的位宽（如 2-bit 或 8-bit）、精度有极致要求且校准数据与推理数据高度一致、已有一套成熟的 GPTQ 推理管线（如 vLLM 的 `gptq_marlin` kernel）。

在实践中，2025 年起新发布的 INT4 量化模型大多采用 AWQ（HuggingFace 上 AWQ 格式的新模型占比已在 60% 以上），GPTQ 凭借成熟的推理生态和存量模型仍然大量存在于生产环境中。两者可以在同一套推理框架中共存——vLLM 既支持 AWQ 也支持 GPTQ 格式，选择哪个更多是模型供应和生态兼容的考量，而非技术优劣的绝对划分。

### SpQR
SpQR（Sparse Quantization）将量化问题建模为稀疏优化。大部分权重可以安全量化为 INT4，少量异常值（outliers）保留 FP16。SpQR 自动检测异常值（通过统计权重分布），构建稀疏矩阵，推理时稀疏矩阵乘法通过专门优化的 kernel 加速。SpQR 在 4-bit 量化下的精度优于 AWQ，但推理速度较慢（稀疏矩阵乘法比密集矩阵慢），工业采用率较低。

## 推理加速
量化的推理加速来自两方面：**显存带宽减少**和**计算单元加速**。INT8 矩阵乘法的显存读写是 FP16 的一半，因此受限于显存带宽的算子（如逐元素操作）可加速 2 倍。对于计算密集型算子（如矩阵乘法），INT8 可使用 Tensor Core（NVIDIA GPU）或 INT8 SIMD 指令（CPU），加速比可达 4-8 倍。

但量化加速的前提是硬件支持。NVIDIA GPU 从 Turing 架构开始支持 INT8 Tensor Core，A100 的 INT8 理论性能为 624 TFLOPS（FP16 是 312 TFLOPS）。CPU 的 AVX512 VNNI、ARM 的 NEON dot 指令也支持 INT8 加速。如果硬件不支持 INT8 加速（如旧款 GPU），量化反而可能变慢（量化和反量化的额外开销）。

## 使用方式
HuggingFace 的 `bitsandbytes` 库提供了最简单的量化方案：

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_8bit=True,  # INT8 量化
    torch_dtype=torch.float16,
)
```

对于 INT4 量化，需要使用 AWQ 或 GPTQ：

```python
# AWQ 量化
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_quantized("meta-llama/Llama-2-7b", quant_path="llama-2-7b-awq")

# GPTQ 量化
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized("meta-llama/Llama-2-7b", use_triton=True)
```

量化后可通过 `torch.save` 保存为 GGUF 格式（llama.cpp）或 `.bin` 格式（HuggingFace），推理时直接加载，无需重新量化。

## 量化精度
量化精度用 PPL（Perplexity，困惑度）衡量，数值越低越好。FP16 的 PPL 视模型和任务而定，INT8 的 PPL 通常与 FP16 接近（差距 < 5%），INT4 的 PPL 可能比 FP16 高 10-20%（视量化方法而定）。AWQ 和 SpQR 在 INT4 下可保持与 INT8 接近的 PPL，但推理速度略慢。

选择量化位宽需要权衡精度、速度、显存。对于生产环境，INT8 是最稳妥的选择，精度损失小且硬件加速成熟。对于边缘设备或显存极度受限的场景，INT4 是可行方案，但需要仔细验证输出质量（尤其是对于数值敏感的任务，如数学计算、代码生成）。
