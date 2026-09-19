---
title: 概念索引
order: 5
---

# 大模型推理优化的概念索引
大模型推理优化的概念看似零散（量化、内核、显存、调度），但实际上可以被组织成一张清晰的五层结构。每一层有自己的核心区分维度，上下游之间是单向依赖——下层演进决定上层可能，理解这张图就能避免绝大多数"我应该选什么"的迷茫。本文是 kernel/ 目录下所有文章的概念地图，每篇文章会聚焦在某一层或跨层关系上。

## 五层结构概览
```
┌─────────────────────────────────────────────────────────────┐
│ 推理引擎层  vLLM / TensorRT-LLM / SGLang / llama.cpp   
├─────────────────────────────────────────────────────────────┤
│ 内核层    Marlin / Machete / CUTLASS NVFP4 / DeepGEMM  
├─────────────────────────────────────────────────────────────┤
│ 算法层    AWQ / GPTQ / EXL2 / EXL3 / QuIP# / NVFP4 算法 
├─────────────────────────────────────────────────────────────┤
│ 精度/格式层  W4A16 / W8A8-FP8 / NVFP4 / K-quant·I-quant·GGUF  
├─────────────────────────────────────────────────────────────┤
│ 硬件层    Volta / Turing / Ampere / Ada / Hopper / Blackwell 
└─────────────────────────────────────────────────────────────┘
```

| 层 | 核心区分维度 | 回答什么问题 |
| -- | ------------ | ------------ |
| 硬件层 | 架构代际（SM 版本） | "这块卡能跑什么" |
| 精度/格式层 | 位宽 × 数字格式 | "用什么格式存权重和激活" |
| 算法层 | 量化方法思路 | "怎么把 FP16 模型压成那个格式" |
| 内核层 | 硬件能力 × 输入格式 | "用硬件的新指令怎么算出矩阵乘" |
| 推理引擎层 | 服务特性（批处理/缓存/调度） | "多个请求怎么串成生产级服务" |

上下游对应关系：上层概念往往直接锚定下层概念——量化算法产出某种权重格式，权重格式必须由特定内核消费，内核必须跑在支持对应硬件能力的 GPU 上。

## 硬件层：GPU 架构代际
所有上层决策的物理根基。每代架构都有标志性的新能力，这些能力是上层优化的物质基础。

| 代际 | compute capability | 关键新能力 | 代表卡 |
| ---- | ------------------ | ---------- | ------ |
| Volta | 7.0 | 第一代 Tensor Core | V100 |
| Turing | 7.5 | INT4 Tensor Core | RTX 20 系、T4 |
| Ampere | 8.0 / 8.6 | BF16、TF32、cp.async、Sparse Tensor Core | RTX 30 系、A100、A10 |
| Ada Lovelace | 8.9 | FP8 Tensor Core | RTX 40 系、L40S |
| Hopper | 9.0 | wgmma、TMA、FA3、Transformer Engine | H100、H200 |
| Blackwell | 10.0+ | FP4 Tensor Core、tcgen05、TMEM | B100/B200、RTX 50 系 |

被上层直接消费的硬件能力：
- `cp.async` — Ampere 起的异步内存拷贝指令，Marlin 的流水线依赖它
- `mma.sync` — 标准 Tensor Core 矩阵乘指令，所有代际都有
- `wgmma` — Hopper 起的 warp group 矩阵乘，Machete 利用它
- `TMA` — Hopper 起的 Tensor Memory Accelerator，Hopper+ 内核的标配
- `tcgen05.mma` — Blackwell 起的 block-scaled 矩阵乘，NVFP4 GEMM 用它
- `TMEM` — Blackwell 的 Tensor Memory，存累加器

关键约束：没有 cp.async 就跑不了 Marlin，没有 wgmma+TMA 就用不到 Machete 的全部优化，没有 FP4 Tensor Core 就只能模拟 NVFP4。上层概念在某一层断档，下层就吃不到对应优化。

## 精度/格式层：数据怎么存
由位宽维度和数字格式维度交叉得到。这层定义的是"权重和激活用什么 bit 表示"。

位宽维度命名规则：
- `W` = Weight，`A` = Activation
- 后跟数字 = bit 数
- 例：`W4A16` = 权重 4-bit + 激活 16-bit

数字格式维度：
- `INT4` / `INT8`：整数
- `FP8` / `FP16` / `BF16`：浮点
- `FP8` 又有 `E4M3` / `E5M2` 两种细分

常见组合：

| 名称 | 权重 | 激活 | 典型硬件 | 备注 |
| ---- | ---- | ---- | -------- | ---- |
| W4A16 | INT4 | BF16 | Ampere+ | weight-only，主流方案 |
| W8A8 (INT8) | INT8 | INT8 | 通用 | 经典 CV 量化 |
| W8A8-FP8 | FP8 | FP8 | Ada/Hopper/Blackwell 原生 | Hopper 起最成熟 |
| W4A8-FP8 | INT4 | FP8 | Hopper | CUTLASS 路径 |
| NVFP4 | FP4 + 微块缩放 | FP4 | Blackwell | 终极压缩 |

几个容易混淆的点：
- `W4A16` 中的 16 通常是 BF16，不是 FP16。BF16 指数位和 FP32 同宽（8 位），跑 LLM 的 Softmax 不溢出；FP16 指数位只有 5 位，跑大模型长序列会 NaN。
- `FP8` 不指定位宽，只指定数字格式——既可以用于权重（`W8A8-FP8`），也可以独立存在。
- `NVFP4` 是浮点 4-bit，不是 INT4。微块缩放 + E4M3 scale 让 4-bit 浮点的动态范围接近 FP8，精度比 INT4 高一截。

## 算法层：怎么压成低 bit
给定一个高精度模型，用什么方法算出低精度版本。这一层产出的是"格式 + 量化参数"的组合。

| 算法 | 思路 | 典型输出 |
| ---- | ---- | -------- |
| GPTQ | Hessian 矩阵误差最小化（重建法） | W4A16 权重 + per-channel scale |
| AWQ | 保护显著权重（per-channel scaling） | W4A16 权重 + 放大关键通道 |
| RTN | 朴素舍入 | 任意位宽，质量最差 |
| EXL2 | GPTQ 思想 + 混合比特 | 私有格式，2~8 bpw 混合 |
| EXL3 | QTIP / Trellis 向量量化 | 私有格式，1~8 bpw 混合 |
| QuIP# / AQLM | 高阶向量量化（学术派） | W4A16，质量极佳但慢 |
| NVFP4 量化 | 微块缩放 + E4M3 scale | NVFP4 权重 |

算法层的关键区分：
- GPTQ vs AWQ：GPTQ 是"误差补偿"思路（量化一列后用 Hessian 调整剩余权重），对校准数据敏感；AWQ 是"特征保护"思路（识别大激活值通道并放大），量化更快、泛化更好
- EXL2 vs GPTQ/AWQ：EXL2 是私有格式（ExLlamaV2 引擎专用），不能被 Marlin 等通用内核消费；GPTQ/AWQ 是开放格式，工具链最成熟
- NVFP4 量化是 NVIDIA 官方配套的 Blackwell 算法，与 NVFP4 格式强绑定

## 内核层：怎么算
给定硬件能力 + 权重格式，怎么用 GPU 的指令把矩阵乘算出来。这一层是上层所有优化的"执行落地"。

| 内核 | 出品 | 输入 | 主要目标硬件 | 关键依赖 |
| ---- | ---- | ---- | ------------ | -------- |
| Marlin | IST-DASLab | W4A16 (AWQ/GPTQ) | Ampere（消费级 W4A16 王者） | cp.async |
| Sparse-Marlin | IST-DASLab | W4A16 + 2:4 稀疏 | Ampere | 稀疏 TC |
| Machete | vLLM | W4A16 | Hopper（Marlin 继任者） | wgmma + TMA |
| CUTLASS W4A8 | NVIDIA | INT4 + FP8 | Hopper | wgmma + TMA |
| CUTLASS NVFP4 GEMM | NVIDIA | NVFP4 | Blackwell | tcgen05.mma + TMEM |
| DeepGEMM | DeepSeek-AI | FP8 / NVFP4 | Blackwell SM100 | tcgen05.mma |
| TensorRT-Edge-LLM CuTe DSL | NVIDIA | NVFP4 grouped | Blackwell SM120 | mma.sync.aligned.block_scale |
| ExLlamaV2 q_gemm | turboderp | EXL2 | Ampere/Ada（小 batch 极快） | 私有手写 CUDA |
| cuDNN / cuBLASLt | NVIDIA | FP8/BF16 | 全硬件 | 通用库 |

内核层的几个关键映射（这是选型矩阵的底层依据）：
- AWQ/GPTQ → Marlin 是 Ampere 时代 W4A16 的事实标准（[Marlin](./marlin)）
- W4A16 → Machete 是 Hopper 上替代 Marlin 的选择
- NVFP4 → CUTLASS + DeepGEMM + TensorRT-Edge-LLM 是 Blackwell 上的多源方案，目前没有一个统一的内核名字（[量化选型](./quant_matrix)）
- EXL2 → ExLlamaV2 q_gemm 是单用户极致速度方案，与 Marlin 不兼容

## 推理引擎层：怎么服务
把单次推理调用串成生产级服务。这一层关心的是"多个请求怎么排队、怎么共享显存、怎么调度"。

| 引擎 | 标志特性 | 内核消费 |
| ---- | -------- | -------- |
| vLLM | Continuous Batching + PagedAttention | Marlin / Machete / CUTLASS / DeepGEMM 全选 |
| TensorRT-LLM | NVIDIA 官方优化 | CUTLASS + TensorRT 引擎 |
| SGLang | RadixAttention（前缀缓存激进） | FlashAttention + 各类量化内核 |
| llama.cpp | GGUF + CPU+GPU 混合 | 自家 q_gemm |
| ExLlamaV2 / TabbyAPI | 单卡单用户极致速度 | 自家 q_gemm（EXL2） |
| Transformers | HF 官方，慢但通用 | 通用 GEMM |

这一层的关键概念：
- `Prefill` vs `Decode` — 两种推理阶段，对应不同瓶颈（Compute-Bound vs Memory-Bound）
- `Continuous Batching` — vLLM 标志特性，请求随时进出
- `PagedAttention` — vLLM 的显存分页管理
- `RadixAttention` — SGLang 的前缀缓存
- `Chunked Prefill` — 长 prompt 切块与 Decode 并发
- `Prefix Caching` — 通用概念，SGLang 做到极致

引擎与内核的关系：引擎是"调度框架"，内核是"计算单元"。vLLM 选择哪个内核，取决于模型权重格式 + GPU 架构；用户不需要直接选内核，vLLM 会自动路由。

## 跨层对应关系速查
关键约束：上层概念往往直接锚定下层概念。错位使用会失去硬件加速的根基。

| 上层 | 下层 | 错位后果 |
| ---- | ---- | -------- |
| Marlin | Ampere 起（cp.async） | Turing 缺 cp.async，无法流水线 |
| Machete | Hopper（wgmma + TMA） | Ampere 上没新指令可用 |
| NVFP4 | Blackwell（FP4 TC） | Ada 上没原生 FP4，软件模拟 |
| W4A16 + Marlin | 显存 ~1/4 of FP16 | 因为 weight-only，不动激活 |
| FP8 | Ada/Hopper/Blackwell 原生 | Ampere 上自动反量化到 FP16 |
| EXL2 权重 | ExLlamaV2 引擎专用 | 不能直接用 vLLM + Marlin |

## 五层的一句话总结
硬件层决定"能跑什么"，精度层决定"用什么格式存"，算法层决定"怎么压成那个格式"，内核层决定"用硬件能力怎么算那个格式"，推理引擎层决定"把多个调用串成什么服务模式"——五层之间是单向依赖，下层的演进决定上层的可能。

## 阅读路径建议
- 刚接触推理优化：先读 [Marlin](./marlin) 建立单内核的完整心智模型，再读 [量化选型](./quant_matrix) 看跨架构选型
- 关心部署：先读 [量化](./quantization) 理解算法层（GPTQ/AWQ），再读 [模型格式](./format) 理解权重存储
- 关心底层：先读 [算子优化](./optimization) 看通用 kernel 设计原则，再回来看 [Marlin](./marlin) 和 [FlashAttention](./flashattention) 的具体实现
- 关心推理服务：先读本文的"推理引擎层"，再看 vLLM / SGLang 等专门文章
