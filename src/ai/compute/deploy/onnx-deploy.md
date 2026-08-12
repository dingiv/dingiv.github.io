---
title: ONNX 部署
order: 12
---

# ONNX 模型部署
ONNX 的部署逻辑与当前 LLM 生态（vLLM/llama.cpp/Ollama）有根本差异。LLM 部署的核心挑战是"大"——模型太大单卡装不下、KV Cache 太大显存放不下、自回归生成太慢需要投机解码。ONNX 部署的核心挑战是"杂"——模型来自不同框架（PyTorch/TF/Keras/sklearn）、部署目标覆盖从手机 NPU 到浏览器 WASM 的异构硬件、推理管线的预处理和后处理需要与模型本身无缝衔接。

理解这个差异是做好 ONNX 部署的前提。你在 ONNX 部署中遇到的不是"70B 模型装不进 24GB 显存"这种问题，而是"这个 YOLO 模型用 TensorRT 快 3 倍还是用 OpenVINO 快 2 倍"这种问题。

## ONNX Runtime 后端选型
ONNX Runtime 的核心设计是 Execution Provider（EP）——可插拔的硬件后端抽象。同一个 `.onnx` 模型，切换 EP 就可以在不同硬件上运行，不需要重新导出。EP 的选择决定了推理性能的上限和部署复杂度。

| 后端              | 硬件               | 适用场景                | 性能特征                                   |
| ----------------- | ------------------ | ----------------------- | ------------------------------------------ |
| CPU (MLAS)        | x86/ARM CPU        | 小模型、低延迟不敏感    | 足够用，无需 GPU                           |
| CUDA              | NVIDIA GPU         | 中等模型、需要 GPU 加速 | 通用性好，比 PyTorch 快 1.5-2x             |
| TensorRT          | NVIDIA GPU         | 追求极限吞吐            | 编译慢（几分钟），推理比 CUDA EP 快 30-50% |
| OpenVINO          | Intel CPU/GPU      | Intel 硬件优化          | 比 CPU EP 快 20-40%                        |
| CoreML            | Apple Silicon      | iOS/macOS 部署          | Apple 原生加速，零额外功耗                 |
| DirectML          | Windows GPU (任何) | Windows 桌面应用        | AMD/Intel/NVIDIA 通吃，不挑卡              |
| QNN               | Qualcomm NPU       | Android 手机            | 移动端最低功耗                             |
| Web (WASM+WebGPU) | 浏览器             | Web 端推理              | 无需安装，但性能逊于原生                   |

EP 支持 fallback 链——`providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` 表示优先 CUDA，CUDA 不可用时降级到 CPU。这个机制使 ONNX 的部署代码在不同硬件环境上零修改运行。

```python
import onnxruntime as ort

# 自动选择最优 EP
session = ort.InferenceSession(
    "model.onnx",
    providers=ort.get_available_providers()
)

# 或显式指定优先级
session = ort.InferenceSession(
    "model.onnx",
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

TensorRT EP 的性能最优但条件最苛刻——需要模型中的所有算子都被 TensorRT 支持。如果遇到不支持的算子，ORT 会回退到 CUDA EP。检查 TensorRT 编译日志确认哪些算子未被编译。

## 模型转换与优化管线
从训练好的模型到生产推理，ONNX 的标准管线是四步：

**导出**：PyTorch/TF → ONNX。`torch.onnx.export()` 的陷阱在于——动态控制流（if/for）无法导出、自定义算子需要注册、`dynamic_axes` 配置不对导致 batch 推理失败。验证导出成功的最简方法是用 `onnx.checker.check_model()` 检查模型合法性，然后用 `onnxruntime.InferenceSession()` 做一次样例推理对比数值误差。

**简化**：用 onnxsim 清理计算图——常量折叠、去除无用节点、合并连续运算。这一步是自动的但效果显著——典型的 PyTorch 导出的 ONNX 经 onnxsim 后体积减小 15-30%，推理速度提升 5-10%。

**量化**：FP32→INT8。ONNX Runtime 支持静态量化（需要校准数据集，精度更高）和动态量化（在线计算 scale/zero point，无需校准数据但精度略低）。对于 CV 模型，INT8 精度损失 < 0.5%，但推理速度提升 2-3 倍。量化是移动端和边缘设备部署的必经步骤。

**硬件适配**：根据目标硬件选择 EP 并验证性能。TensorRT EP 需要额外的编译时间（几分钟）——编译时尝试所有可行的算子融合策略，生成的 engine 文件绑定当前 GPU 型号。将 TensorRT engine 缓存到磁盘，避免每次启动重新编译。

```python
# 完整优化管线示例
import onnx
from onnxsim import simplify
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. 导出 (PyTorch)
torch.onnx.export(model, dummy_input, "model_raw.onnx", opset_version=17)

# 2. 简化
model = onnx.load("model_raw.onnx")
model_simplified, check = simplify(model)
onnx.save(model_simplified, "model_sim.onnx")

# 3. 量化
quantize_dynamic("model_sim.onnx", "model_int8.onnx", weight_type=QuantType.QInt8)

# 4. 推理
session = ort.InferenceSession("model_int8.onnx", providers=["CUDAExecutionProvider"])
```

## 边缘与 Web 部署
ONNX 在边缘设备上的优势是 HF 生态无法覆盖的。

**移动端**：ORT Mobile 是精简版 ONNX Runtime（~2MB），支持 Android（QNN EP 调用高通 NPU）和 iOS（CoreML EP 调用 Apple Neural Engine）。模型经 INT8 量化后体积 < 10MB，在 NPU 上的推理速度是 CPU 的 5-10 倍，功耗只有 1/5。

**浏览器**：ONNX Runtime Web 通过 WASM + WebGPU 在浏览器中直接运行 ONNX 模型。不需要服务器端推理——用户上传的图片在本地浏览器中完成分类/检测/分割。典型的 Web ML 应用（背景虚化、实时姿态估计）都在使用这个路径。WebGPU EP（Chrome 113+、Edge 113+）比 WASM EP 快 2-3 倍。

**嵌入式 Linux**：ORT 支持 ARM64 Linux，可集成到 Yocto/Buildroot 构建系统中。适用于工业相机（缺陷检测）、边缘网关（传感器数据分析）、机器人（实时目标检测）等场景。搭配 OpenVINO EP 在 Intel 的 Movidius/Myriad VPU 上运行，功耗 < 5W。

## ONNX 部署 vs LLM 部署
ONNX 的部署工具链在 LLM 时代并没有过时——它只是不在 LLM 的部署路径上。两者的适用场景不同：

| 维度       | ONNX 部署                 | LLM 部署 (vLLM/llama.cpp)      |
| ---------- | ------------------------- | ------------------------------ |
| 模型规模   | 百万~亿参数               | 十亿~千亿参数                  |
| 推理模式   | 单次前向传播              | 自回归循环生成                 |
| 主要瓶颈   | 计算吞吐（Compute-Bound） | 显存带宽（Memory-Bound）       |
| 硬件范围   | CPU/GPU/NPU/手机/浏览器   | GPU + CPU 混合                 |
| 优化重点   | 算子融合、量化、图优化    | KV Cache、投机解码、连续批处理 |
| 部署复杂度 | 模型小，部署简单          | 模型大，需要分区和并行         |

一个典型的混合场景：用 ONNX 部署 Embedding 模型（BERT/sentence-transformers）做向量检索，用 vLLM 部署 LLM 做对话生成。两者在同一台服务器上通过不同的推理引擎服务不同的功能，共享 GPU 资源。

ONNX 仍然是跨硬件推理的事实标准——在需要"模型到处跑"的场景中不可替代。LLM 专用引擎则解决了"大模型跑起来"的问题。两条路径在未来会继续并行——因为它们解决的根本问题不同。
