---
title: ONNX
order: 55
---

# ONNX
ONNX（Open Neural Network Exchange）是 Microsoft 和 Facebook 主推的开放式神经网络交换格式，旨在实现 AI 模型的跨框架互操作性。ONNX 定义了一套标准的算子集合和序列化格式，使得在一个框架（如 PyTorch）中训练的模型可以导出到另一个框架（如 TensorFlow、ONNX Runtime）中进行推理。

## ONNX 生态
ONNX 生态包含三个核心组件：ONNX 格式（模型序列化）、ONNX Operators（标准算子集）、ONNX Runtime（高性能推理引擎）。ONNX 格式使用 Protocol Buffers 序列化，包含模型结构（计算图）、权重、元数据。ONNX Operators 定义了数百个标准算子（如 Conv、MatMul、Softmax），各推理引擎只需实现这些算子即可运行任何 ONNX 模型。

ONNX 的核心价值在于**解耦**——训练框架和推理引擎可以独立演进。研究者可以用 PyTorch 快速实验，工程师用 ONNX Runtime 部署到生产环境，企业用 TensorRT 加速到 NVIDIA GPU，医生用 CoreML 部署到 iPhone。这种解耦极大降低了模型迁移成本。

```python
# 将 PyTorch 模型导出为 ONNX
import torch
import torch.onnx

model = MyModel()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["output"],
    dynamic_axes={"image": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17
)
```

## ONNX Runtime
ONNX Runtime 是微软开发的高性能推理引擎，支持 CPU、GPU（CUDA、ROCm、TensorRT）、NPU、TPU 等多种硬件后端。ONNX Runtime 的核心优化包括：算子融合（将多个连续算子合并为一个）、图优化（常量折叠、死代码消除）、内存规划（减少内存分配和拷贝）。

ONNX Runtime 的性能通常优于原生框架。对于 ResNet-50，ONNX Runtime 在 NVIDIA GPU 上的推理性能比 PyTorch 高 1.5-2 倍，比 TensorFlow 高 2-3 倍。这得益于 ONNX Runtime 的针对性优化——它只关注推理，无需考虑训练需求，因此可以更激进地优化计算图。

```python
import onnxruntime as ort

# 创建推理 session
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

# 获取输入输出信息
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 推理
output = session.run([output_name], {input_name: dummy_input.numpy()})
```

## ONNX 算子集版本

ONNX 算子集（Opset）是 ONNX 的版本控制机制，每个新版本会增加新算子或修改现有算子的行为。截至 2024 年，ONNX 的最新 Opset 版本是 19。导出 ONNX 模型时，需要选择合适的 Opset 版本。

| Opset 版本 | 发布年份 | 主要特性 |
|------------|----------|----------|
| Opset 7 | 2017 | 早期版本，算子有限 |
| Opset 11 | 2019 | 支持 dynamic axes、算子标准化 |
| Opset 13 | 2020 | 改进 RNN 支持、标准化常量传播 |
| Opset 17 | 2023 | 支持 FlashAttention、GPTQ 量化 |
| Opset 19 | 2024 | 支持 Stable Diffusion 优化 |

选择 Opset 版本时需要权衡：新版本支持更多算子和优化，但旧推理引擎可能不支持。生产环境中建议选择推理引擎支持的最新 Opset 版本，以获得最佳性能。

## ONNX 的局限

ONNX 的主要挑战在于**算子覆盖不完整**。虽然 ONNX 定义了数百个算子，但各框架的自定义算子（如 PyTorch 的 `torch.complex`、TensorFlow 的 `tf.nn.depthwise_conv2d_native`）可能无法直接导出。解决方案包括：使用 ONNX 的自定义算子（Custom Operator）机制、将自定义算子分解为标准算子组合、等待 ONNX 新版本支持。

另一个挑战是**动态图支持有限**。ONNX 的计算图是静态的，导出时需要固定输入形状（虽然 dynamic axes 可以支持部分维度动态，但并非所有算子都支持）。对于控制流（如 if-else、for loop），ONNX 提供了 If 和 Loop 算子，但 PyTorch 的动态控制流导出时可能失败。

## 未来展望
ONNX 的未来发展方向包括：更好的算子覆盖（支持最新的 Transformer 架构，如 Mixture of Experts）、更强的动态图支持（通过 ONNX Script 定义可微分函数）、更丰富的量化支持（INT4、GPTQ、AWQ）。ONNX Runtime 也在向边缘设备扩展（ONNX Runtime Mobile、ONNX Runtime for Microcontrollers），满足 IoT 和嵌入式场景的需求。

ONNX 的成功在于它是**真正开放的标准**——任何人都可以实现 ONNX Runtime，任何人都可以扩展 ONNX 算子。这与 TensorFlow 的 SavedModel（只能用 TensorFlow 运行）和 PyTorch 的 TorchScript（PyTorch 特有）形成鲜明对比。虽然 ONNX 在学术研究中使用较少（研究者更习惯原生框架），但在工业部署中，ONNX 已成为跨平台推理的事实标准。
