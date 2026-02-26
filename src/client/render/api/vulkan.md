# Vulkan


# SPIR-V
SPIR-V（Standard Portable Intermediate Representation）是 Khronos 制定的跨平台中间表示，服务于 Vulkan、OpenGL 和 OpenCL 生态。着色器源代码（GLSL、HLSL、WGSL）先编译为 SPIR-V 二进制，驱动再将 SPIR-V 转换为 GPU 机器码。这种分层设计降低了驱动复杂度，也使得着色器可以跨应用复用。

SPIR-V 的设计借鉴了 LLVM IR 的思想，提供一套稳定的、平台无关的中间表示。与 GLSL 的即时编译相比，SPIR-V 的预编译可消除启动时的卡顿。SPIR-V 也支持着色器模块化（import/export），多个着色器可共享函数定义，减少代码重复。SPIR-V 采用二进制格式，体积小且解析快，运行时开销远小于文本格式的着色器语言。

## 编译流程
SPIR-V 的编译流程分为两个阶段：前端编译将着色器源代码编译为 SPIR-V，后端编译将 SPIR-V 转换为 GPU 机器码。前端编译使用 `glslangValidator`（GLSL）、`dxc`（HLSL）、`naga`（WGSL）等工具，生成 `.spv` 文件。后端编译由 Vulkan 驱动在创建管线时完成，将 SPIR-V 转换为 NVIDIA 的 SASS、AMD 的 GFX ISA、Intel 的 Xe ISA 等。

```
GLSL/HLSL/WGSL 源代码
    ↓ (前端编译器)
SPIR-V 中间表示 (.spv 文件)
    ↓ (Vulkan 驱动)
GPU 机器码 (SASS/GFX/Xe ISA)
```

前端编译的优势在于可离线完成，应用启动时只需加载预编译的 SPIR-V 文件，无需运行时编译。这种模式适合生产环境，减少了首次加载的延迟。前端编译也可进行更激进的优化（如循环展开、内联），因为这些优化只需执行一次，不影响运行时性能。后端编译相对轻量，主要负责指令选择和寄存器分配，编译时间可控。

## 指令集架构
SPIR-V 采用 SSA（Static Single Assignment）形式的指令集。每个操作都是显式的，没有隐式类型转换或状态依赖。指令由操作码和操作数组成，操作数可以是 ID 引用（指向其他指令的结果）或立即数。例如，OpAdd 指令表示加法，`%result = OpAdd %type %operand1 %operand2`，`%result` 是结果 ID，`%type` 是结果类型，`%operand1` 和 `%operand2` 是操作数 ID。

SPIR-V 的类型系统包括标量（ScalarType）、向量（VectorType）、矩阵（MatrixType）、数组（ArrayType）、结构体（StructType）、指针（PointerType）、图像（ImageType）、采样器（SamplerType）等。类型声明使用 OpType 指令，如 `OpTypeFloat 32` 声明 32 位浮点类型，`OpTypeVector %float 3` 声明三维向量。类型系统是强类型的，不支持隐式转换，确保了跨平台的正确性。

```spirv
; SPIR-V 示例：简单的加法
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpName %main "main"

%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%float = OpTypeFloat 32
%vec3 = OpTypeVector %float 3

%main = OpFunction %void None %void_fn
%label = OpLabel
%a = OpConstant %float 1.0
%b = OpConstant %float 2.0
%sum = OpFAdd %float %a %b
OpReturn
OpFunctionEnd
```

## 装饰与布局
SPIR-V 使用装饰（Decorations）将变量连接到管线阶段和资源。`Location` 装饰指定顶点输入输出位置，`BuiltIn` 装饰指定内置变量（如 Position、PointSize、FragCoord），`Binding` 和 `DescriptorSet` 装饰指定资源绑定（纹理、采样器、缓冲区）。`Block` 装饰标记统一缓冲区和存储缓冲区，`Offset` 装饰指定结构体成员偏移。

内存布局由 `std140` 和 `std430` 规则决定。`std140` 用于统一缓冲区（Uniform Buffer），标量对齐到 4 字节，向量对齐到 16 字节（vec3 例外），数组元素对齐到 16 字节。`std430` 用于存储缓冲区（Storage Buffer），标量和向量对齐到其元素大小，数组元素对齐到其元素大小（无额外填充）。这些规则确保 CPU 和 GPU 对数据布局的理解一致，避免了隐式对齐问题。

```spirv
; 统一缓冲区布局示例
OpMemberDecorate %Uniforms 0 Offset 0      ; modelMatrix
OpMemberDecorate %Uniforms 1 Offset 64     ; viewMatrix
OpMemberDecorate %Uniforms 2 Offset 128    ; projectionMatrix
OpDecorate %Uniforms Block

%Uniforms = OpTypeStruct %mat4 %mat4 %mat4
%uniform_ptr = OpTypePointer Uniform %Uniforms
%uniforms = OpVariable %uniform_ptr Uniform
```

## 优化与验证
SPIR-V 的优化可在前端编译或后端编译进行。前端优化（如 `glslangValidator -O`）包括常量折叠、死代码消除、函数内联、循环展开。后端优化（驱动负责）包括指令调度、寄存器分配、SIMD 打包。优化级别越高，编译时间越长，但运行时性能更好。对于移动平台，可关闭激进优化以减少编译时间。

SPIR-V 的验证（Validation）确保着色器符合规范。`spirv-val` 工具可检查类型错误、未定义 ID、无效装饰、控制流错误（如函数返回不一致）。验证失败意味着着色器可能在某些 GPU 上崩溃，必须修复。Vulkan 驱动在创建管线时也会验证 SPIR-V，但验证信息可能不如 `spirv-val` 详细（驱动可能只报"invalid SPIR-V"，不指出具体位置）。

## 工具链
SPIR-V 的工具链包括编译器、反汇编器、优化器、验证器。`glslangValidator` 是 GLSL 到 SPIR-V 的编译器，支持 Vulkan 和 OpenGL。`dxc` 是微软的 HLSL 编译器，可输出 SPIR-V（通过 `-spirv` 选项）。`naga` 是 Rust 编写的 WGSL 到 SPIR-V 编译器，用于 WebGPU 后端。`spirv-reflect` 可提取 SPIR-V 的着色器反射信息（资源绑定、输入输出），用于引擎自动绑定。

`spirv-opt` 是 SPIR-V 优化器，可独立于编译器运行。优化项包括：`--strip-debug`（删除调试信息，减小文件体积）、`--inline-entry-points-exhaustive`（激进内联）、`--convert-local-access-chains`（优化局部数组访问）、`--ccp`（常量传播）。`spirv-opt` 可多次运行，每次优化可基于前一次结果。

```bash
# 编译 GLSL 到 SPIR-V
glslangValidator -V -o shader.vert.spv shader.vert

# 优化 SPIR-V
spirv-opt --strip-debug shader.vert.spv -o shader.opt.spv

# 验证 SPIR-V
spirv-val shader.vert.spv

# 反汇编 SPIR-V（查看文本形式）
spirv-dis shader.vert.spv
```

## 跨平台编译
SPIR-V 的跨平台能力是其核心价值。同一份 SPIR-V 可在 NVIDIA、AMD、Intel 的 GPU 上运行，也可转换为 WebGPU 的 WGSL。这种跨平台能力降低了引擎开发的成本，Vulkan、OpenGL、OpenCL 的开发者都可使用 SPIR-V 作为中间层。

SPIR-V 到其他着色语言的转换使用 `spirv-cross` 工具。`spirv-cross` 可将 SPIR-V 转换为 GLSL、HLSL、MSL（Metal Shading Language）、CPP（C++ 伪代码）。这种转换适用于不支持 Vulkan 的平台（如 macOS 的 Metal、iOS 的 Metal、WebGL 的 GLSL）。转换质量取决于 SPIR-V 的复杂度，简单着色器通常转换良好，复杂着色器可能需要手动调整。

```bash
# SPIR-V 转 MSL（用于 macOS/iOS）
spirv-cross --msl shader.vert.spv -o shader.vert.metal

# SPIR-V 转 HLSL（用于 DirectX）
spirv-cross --hlsl shader.vert.spv -o shader.vert.hlsl

# SPIR-V 转 GLSL（用于 WebGL）
spirv-cross --es --version 300 shader.vert.spv -o shader.vert.glsl
```

## 调试与性能分析
SPIR-V 的调试比源代码困难，因为它是中间表示，不保留原始变量名（除非保留调试信息）。`spirv-val` 可验证 SPIR-V 的正确性，但无法检查逻辑错误。Vulkan 的验证层（Validation Layers）可检测运行时错误（如资源未绑定、描述符集不匹配），但开销较大，仅用于开发环境。

性能分析工具包括 RenderDoc、Nsight、Radeon GPU Profiler。这些工具可捕获 Vulkan 帧并分析每个着色器的执行时间。着色器的 SPIR-V 代码可通过 `spirv-dis` 反汇编查看，但可读性较差。更好的方法是在源代码级别分析，使用 `glslangValidator -g` 保留调试信息，然后映射回源代码。

## 扩展与版本
SPIR-V 版本与 Vulkan 版本对应。SPIR-V 1.0 对应 Vulkan 1.0，支持基本的图形和计算着色器。SPIR-V 1.3 对应 Vulkan 1.1，增加子组操作（Subgroup Operations）、设备组（Device Groups）。SPIR-V 1.5 对应 Vulkan 1.2，增加 ray query（射线查询）、整数点积。SPIR-V 1.6 对应 Vulkan 1.3，增加 shader clock（着色器时钟）、矩阵访问（动态索引矩阵）。

SPIR-V 扩展提供额外的功能，但需要硬件支持。`SPV_KHR_ray_tracing` 增加射线追踪着色器，`SPV_KHR_ray_query` 增加射线查询（可在片段着色器中发射射线）。`SPV_EXT_demote_to_helper_invocation` 增加丢弃线程的替代方案（比 `discard` 更高效）。`SPV_KHR_subgroup_rotate` 增加子组旋转操作（用于快速傅里叶变换）。扩展需要前端编译器支持（如 `glslangValidator -target-env vulkan1.2`），也需要驱动支持。

SPIR-V 的未来发展包括更好的优化、更丰富的扩展、更紧凑的表示。Khronos 正在开发 SPIR-V 2.0，计划增加更强大的类型系统、更好的模块化支持、更高效的编码格式。SPIR-V 也将成为更多图形 API 的中间表示，如 WebGPU 的后端可使用 SPIR-V，进一步统一图形编程生态。
