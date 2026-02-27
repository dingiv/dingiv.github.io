---
title: 标准接口
order: 7
---

# 图形计算 API
图形 API（Application Programming Interface）是连接应用程序与 GPU 的桥梁，定义了如何提交渲染命令、管理资源、同步操作。

图形 API 的核心职责是将高级绘图指令（如"绘制这个模型"）转换为 GPU 可理解的命令序列。这个转换过程涉及状态管理（当前着色器、当前纹理、当前混合模式）、资源绑定（顶点缓冲区、索引缓冲区、常量缓冲区）、命令提交（绘制调用、分发调用）。早期的图形 API（OpenGL 1.x）采用固定功能管线，开发者只能配置参数；现代图形 API 采用可编程管线，开发者可编写着色器自定义每个阶段。

图形 API 的演进反映了 GPU 架构的变化。早期 GPU 功能有限，只能处理光栅化，图形 API 对应提供简单接口。现代 GPU 支持可编程着色器、几何处理、曲面细分、计算着色器、射线追踪，图形 API 不断扩展以暴露这些能力。另一个趋势是减少驱动开销，早期 API 依赖驱动做大量验证和优化（运行时编译着色器、隐式状态管理），现代 API 将责任转移给开发者（预编译着色器、显式状态管理），减少了不可预测的性能波动。

不同的图形 API 有不同的设计哲学和适用场景，OpenGL 强调易用性，Vulkan 强调性能和可预测性，DirectX 是 Windows 平台的标准，Metal 是 Apple 生态的高效接口。

## Vulkan
Vulkan 是 Khronos 于 2016 年发布的现代图形 API，设计目标是提供低开销、跨平台的抽象，让开发者可以精确控制 GPU 执行。Vulkan 采用显式编程模型，开发者需要显式管理状态（Pipeline State Object）、内存（Buffer 和 Image 的分配）、同步（Fence 和 Semaphore）。这种设计减少了驱动的运行时开销，但也增加了开发复杂度。

Vulkan 的核心概念包括：实例和设备（Instance 和 Device，连接物理 GPU）、队列（Queue，提交命令的通道）、命令缓冲区（Command Buffer，记录命令的缓冲区）、渲染通道（Render Pass，定义渲染目标的布局）、管线（Pipeline，封装状态的不可变对象）、描述符集（Descriptor Set，绑定资源到着色器）。这些概念互相配合，实现了高效的并行渲染。例如，多线程可同时记录命令缓冲区，然后在主线程一次性提交，充分利用多核 CPU。

Vulkan 支持预编译着色器为 SPIR-V，应用启动时加载二进制，无需运行时编译。这与 OpenGL 的即时编译形成对比，消除了首次加载的卡顿。Vulkan 也支持更细粒度的资源管理，如显存分配（VMA，Vulkan Memory Allocator）、内存别名（同一内存用于不同资源）、资源重用（避免频繁分配/释放）。这些优化对于 AAA 游戏和大型应用至关重要。

Vulkan 的学习曲线陡峭，需要理解 GPU 架构和图形管线的每个阶段。初始化 Vulkan 应用需要数百行代码（创建实例、选择物理设备、创建逻辑设备、创建交换链、创建渲染通道、创建帧缓冲区、创建命令缓冲区、创建同步对象）。这种复杂性使得很多开发者选择高级引擎（如 Unreal、Unity）或封装库（如 GLFW、Vulkan-Hpp）来简化开发。但 Vulkan 的性能收益是显著的，特别是在多线程渲染和复杂场景中，Vulkan 可比 OpenGL 提升 20-50% 的帧率。

Vulkan 作为现代的 OpenGL 的继任者，且支持跨厂商硬件的图形 API 标准，是学习现代渲染的必备技术。

## OpenGL
OpenGL（Open Graphics Library）是最早的跨平台图形 API 之一，由硅谷图形公司（SGI）于 1992 年发布。OpenGL 采用立即模式设计，开发者通过一系列函数调用设置状态并提交绘制命令，驱动负责管理资源、验证状态、生成命令序列。这种设计简单易用，降低了图形编程的门槛，但也带来了性能问题：驱动的开销不可预测，状态变化可能导致隐式的 flush，优化依赖于驱动的"智能"程度。

OpenGL 的版本演进反映了图形硬件的发展。OpenGL 1.x 提供固定功能管线（顶点变换、光照、光栅化），OpenGL 2.0 引入可编程着色器（GLSL），OpenGL 3.0 引入核心模式（Core Profile，废弃立即模式），OpenGL 4.0 引入曲面细分和计算着色器。每个版本都增加了新功能，但也保持了向后兼容（通过扩展机制）。这种兼容性使得 OpenGL 在学术界和跨平台应用中流行，但也导致 API 臃肿，多个函数完成类似任务（如 `glBegin`/`glEnd` 和 VAO 绘制）。

OpenGL 的问题在于驱动开销大。每次 API 调用都需要验证状态、检查错误、可能触发隐式操作（如着色器编译）。这种开销在简单应用中不明显，但在复杂应用中积累成瓶颈。OpenGL 的状态机模型使得优化困难，开发者无法精确控制 GPU 执行，只能依赖驱动的"魔法"。这些问题促成了 Vulkan 的诞生。

OpenGL 在 Web 平台的变体是 WebGL。WebGL 1.0 基于 OpenGL ES 2.0，提供 ES 2.0 的子集；WebGL 2.0 基于 OpenGL ES 3.0，增加了 3D 纹理、UBO 等功能。WebGL 的优势是跨平台（所有现代浏览器支持），但性能受限于 JavaScript 和浏览器的沙箱。WebGPU 是 WebGL 的继任者，提供更低的开销和更强大的功能。

## DirectX
DirectX 是微软的图形和多媒体 API，专为 Windows 平台设计。DirectX 包含多个组件：Direct3D（3D 图形）、Direct2D（2D 图形）、DirectWrite（字体渲染）、XAudio2（音频）、XInput（输入）。Direct3D 是最核心的部分，经历了多次重大更新：Direct3D 9（固定功能管线与可编程着色器并存）、Direct3D 10（统一着色器架构）、Direct3D 11（曲面细分、计算着色器）、Direct3D 12（低开销、显式管理，类似 Vulkan）。

Direct3D 11 是目前广泛使用的版本，提供了良好的平衡：功能丰富（曲面细分、几何着色器、计算着色器）、易于使用（状态管理比 Vulkan 简单）、性能可接受（驱动优化成熟）。Direct3D 11 的立即上下文（Immediate Context）负责记录命令并提交到 GPU，延迟上下文（Deferred Context）可在多线程记录命令。这种设计比 OpenGL 的单线程模型进步，但仍不如 Vulkan 的细粒度控制。

Direct3D 12 是微软对 Vulkan 的回应，提供了类似的低开销、显式管理模型。Direct3D 12 引入了命令列表（Command List，类似 Vulkan 的命令缓冲区）、命令队列（Command Queue）、描述符堆（Descriptor Heap，类似 Vulkan 的描述符集）、根签名（Root Signature，绑定资源的元数据）。Direct3D 12 与 Vulkan 的主要区别在于抽象级别：Direct3D 12 保留了部分驱动辅助（如自动转换资源布局），Vulkan 完全由开发者控制。

DirectX 的优势是与 Windows 深度集成，支持最新的 Windows 特性（如 DirectX Raytracing、DXR 光线追踪、Variable Rate Shading 可变速率着色、Mesh Shaders 网格着色器）。DirectX 的文档和工具（PIX、Visual Studio Graphics Debugger）也比 OpenGL/Vulkan 完善，是 Windows 游戏开发的首选。DirectX 的缺点是跨平台支持有限（仅 Windows 和 Xbox），开发者需要为其他平台编写不同的代码路径。

## Metal
Metal 是 Apple 于 2014 年发布的图形 API，专为 iOS、macOS、tvOS 设计。Metal 的目标是提供低开销的 GPU 访问，充分利用 Apple Silicon（A 系列芯片、M 系列芯片）的统一内存架构。Metal 的设计比 Vulkan 简洁，API 调用次数更少，但仍保留了显式状态管理和多线程命令缓冲区。

Metal 的核心概念包括：设备（MTLDevice，代表 GPU）、命令队列（MTLCommandQueue，提交命令的通道）、命令缓冲区（MTLCommandBuffer，记录命令的缓冲区）、渲染通道描述符（MTLRenderPassDescriptor，定义渲染目标）、渲染管线状态（MTLRenderPipelineState，封装着色器和状态）、可编程着色器（Metal Shading Language，基于 C++14）。Metal 与 Apple 的操作系统深度集成，可在应用间共享 GPU 资源，支持低延迟的渲染。

Metal 的独特优势是与 Apple Silicon 的统一内存架构协同。在 Apple Silicon 上，CPU 和 GPU 共享内存，数据传输无需复制（零拷贝），延迟远低于传统架构的 PCIe 传输。Metal 也支持 SIMD 组（SIMD Group）和 SIMD 屏幕（SIMD Screen）等 Apple 特有的优化，可提升并行计算性能。Metal Performance Shaders（MPS）提供了高性能的卷积、矩阵乘法、归约等算子库，性能接近手写 Metal 代码。

Metal 的缺点是平台限制，仅支持 Apple 设备。对于跨平台应用，需要为不同平台编写不同的图形后端（Windows 用 Direct3D/Vulkan，Android 用 Vulkan，Web 用 WebGL/WebGPU）。但对于 Apple 生态的开发者，Metal 是最优选择，性能和开发效率都优于跨平台方案。Swift 和 Objective-C 的 Metal 绑定也比其他语言的 API 更友好，减少了样板代码。

## API 对比与选择
选择图形 API 需考虑目标平台、性能需求、开发团队经验。对于跨平台应用（Windows、Linux、Android），Vulkan 是最优选择，一次编写，多处运行，性能可预测。对于 Windows 独占游戏，Direct3D 12 是行业标准，工具链完善，性能优化成熟。对于 Apple 生态，Metal 是唯一选择，性能和开发效率最优。对于快速原型或学习，OpenGL 易于上手，适合教育和研究。

跨平台引擎（如 Unreal、Unity）通常使用多后端架构，上层应用使用统一的 API，引擎内部根据平台选择合适的图形 API。这种架构的优势是开发者只需编写一次代码，引擎负责处理平台差异。缺点是引擎的抽象可能掩盖平台特性的使用，无法充分发挥硬件能力。对于追求极致性能的 AAA 游戏，可能需要为不同平台编写定制代码。

Web 平台的图形 API 正在从 WebGL 向 WebGPU 演进。WebGL 2.0 基于 OpenGL ES 3.0，提供了基本的 3D 渲染能力，但性能受限于浏览器的沙箱和 JavaScript 的开销。WebGPU 是现代图形 API 的 Web 版本，提供了更低的开销、更强大的功能（计算着色器、SPIR-V 支持），有望在 2025 年成为主流。WebGPU 的设计受到 Vulkan、Direct3D 12、Metal 的影响，是跨平台图形编程的未来方向。

图形 API 的未来趋势包括：更好的多线程支持（减少 CPU 开销）、更灵活的资源管理（显式分配、内存别名）、更强大的着色器能力（射线追踪、网格着色器、机器学习加速）、更高效的开发工具（着色器热重载、性能分析器）。随着硬件的进步，图形 API 会不断演进，但核心原则保持不变：让开发者可以充分利用 GPU 的并行计算能力，创建令人惊叹的视觉体验。
