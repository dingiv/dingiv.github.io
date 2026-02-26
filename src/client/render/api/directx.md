---
title: DirectX
order: 20
---

# HLSL

High-Level Shading Language（HLSL）是微软 DirectX 图形 API 的标准着色语言，用于编写 Windows 平台的游戏和图形应用。HLSL 语法基于 C++，支持模板、运算符重载、命名空间等高级特性，在游戏开发领域占据主导地位。

HLSL 与 DirectX 的深度集成是其核心竞争力。着色器通过 HLSL 编译为字节码（DXBC 或 DXIL），与应用一起分发，无需在运行时编译，消除了启动时的卡顿。DirectX 的调试工具（PIX、Visual Studio Graphics Debugger）对着色器提供了强大支持，可单步执行、检查变量、查看 GPU 状态。HLSL 也是 Xbox 和 Windows 的统一图形语言，跨平台游戏可共享着色器代码。

## 着色器模型

HLSL 的着色器模型（Shader Model）版本控制机制确保了硬件兼容性。每个 SM 版本对应一组 GPU 功能和指令集，开发者可在着色器中指定最低版本要求。SM 2.0 支持 DirectX 9.0c，包含基本的顶点和像素着色器。SM 3.0 增加动态流控制，支持更复杂的算法。SM 4.0 引入几何着色器，支持整数运算和纹理数组。SM 5.0 支持 DirectX 11，增加曲面细分着色器和计算着色器。SM 6.0+ 支持 DirectX 12，增加波前操作、射线追踪等高级特性。

着色器模型通过着色器编译目标指定。例如 `#pragma target 5.0` 声明着色器需要 SM 5.0 支持。如果硬件不支持，编译器会报错或驱动会拒绝创建着色器。这种机制避免了运行时崩溃，但也增加了复杂性：开发者需要为不同硬件级别编写多个着色器版本，或在运行时选择合适的着色器。

```hlsl
#pragma target 5.0

// DirectX 11 特性：纹理数组
Texture2DArray texArray : register(t0);
SamplerState samplerState : register(s0);

float4 PSMain(VSOutput input) : SV_Target {
    return texArray.Sample(samplerState, float3(input.uv, input.slice));
}
```

## 基础语法

HLSL 的变量声明类似 C++，但增加了语义（Semantic）和寄存器绑定。语义将变量连接到管线阶段，如 `POSITION`（顶点位置）、`NORMAL`（法向量）、`TEXCOORD`（纹理坐标）、`SV_Target`（渲染目标输出）。寄存器绑定将变量分配到硬件资源槽，如 `register(t0)` 绑定到纹理寄存器 0，`register(b0)` 绑定到常量缓冲区 0。这种显式绑定使得着色器与应用的数据交换清晰明确。

```hlsl
// 常量缓冲区
cbuffer ConstantBuffer : register(b0) {
    matrix modelMatrix;
    matrix viewMatrix;
    matrix projectionMatrix;
};

// 顶点着色器输入
struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

// 顶点着色器输出
struct VSOutput {
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    float4 worldPos = mul(float4(input.position, 1.0), modelMatrix);
    output.position = mul(worldPos, viewMatrix);
    output.position = mul(output.position, projectionMatrix);
    output.normal = mul(input.normal, (float3x3)modelMatrix);
    output.uv = input.uv;
    return output;
}
```

HLSL 支持函数重载和模板，提供了比 GLSL 更强大的抽象能力。`max()` 函数可接受 int、float、向量等不同类型的参数，编译器自动生成适当的版本。模板函数可编写通用算法，如向量加法可同时支持 vec2、vec3、vec4。这种类型安全机制减少了运行时错误，但编译时间可能增加。

## 缓冲区资源

HLSL 提供多种缓冲区类型，用于在应用和着色器间传递数据。常量缓冲区（Constant Buffer，cbuffer）用于频繁更新的少量数据（变换矩阵、材质参数），大小限制为 64KB（硬件限制），适合每帧更新。纹理缓冲区（Texture Buffer，tbuffer）是常量缓冲区的纹理版本，可通过纹理采样器访问，支持随机读取。

结构化缓冲区（Structured Buffer）是 DX10+ 引入的通用缓冲区，可存储任意结构体数组，支持随机读取和写入。追加缓冲区（Append Buffer）和消费缓冲区（Consume Buffer）支持原子操作，用于粒子系统等无序数据流。读写缓冲区（RWStructuredBuffer）支持原子操作和线程间同步，是计算着色器的核心数据结构。

```hlsl
// 结构化缓冲区
struct Particle {
    float3 position;
    float3 velocity;
    float life;
};

StructuredBuffer<Particle> inputParticles : register(t0);
RWStructuredBuffer<Particle> outputParticles : register(u0);

[numthreads(256, 1, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID) {
    Particle p = inputParticles[DTid.x];
    p.position += p.velocity * deltaTime;
    p.life -= deltaTime;
    outputParticles[DTid.x] = p;
}
```

## 纹理采样

HLSL 的纹理系统比 GLSL 更灵活，支持纹理数组的多维资源。Texture2D、Texture3D、TextureCube 等类型对应不同的纹理维度，Texture2DArray 支持纹理数组（常用于地形切片、立方体阴影贴图）。SamplerState 定义采样模式（滤波、包裹、边界），可与纹理分离绑定，实现一个采样器用于多个纹理。

纹理采样方法包括 Sample（标准采样，自动应用 mipmap 和滤波）、SampleLevel（显式指定 mipmap 层级）、SampleGrad（显式指定导数，手动计算 mipmap）、Load（直接读取纹素，无滤波）。计算着色器常用 Load 进行精确访问，图形着色器常用 Sample 获得平滑结果。

```hlsl
Texture2D diffuseTexture : register(t0);
Texture2D normalTexture : register(t1);
Texture2D roughnessTexture : register(t2);
TextureCube environmentTexture : register(t3);
SamplerState samplerState : register(s0);

float4 PSMain(VSOutput input) : SV_Target {
    // 标准采样
    float4 diffuse = diffuseTexture.Sample(samplerState, input.uv);

    // 法线贴图采样
    float3 normal = normalTexture.Sample(samplerState, input.uv).rgb;
    normal = normalize(normal * 2.0 - 1.0);

    // 环境反射采样
    float3 reflectDir = reflect(-viewDir, normal);
    float3 environment = environmentTexture.Sample(samplerState, reflectDir).rgb;

    return float4(diffuse.rgb * environment, diffuse.a);
}
```

## HLSL 与 FXC/DXC

HLSL 的编译器有两个：FXC（Legacy 编译器）和 DXC（现代编译器）。FXC 用于 DirectX 9-11，生成 DXBC 字节码，功能成熟但不再更新。DXC 基于 LLVM，用于 DirectX 12 和 SPIR-V 生成，支持最新的 HLSL 特性（SM 6.0+）和更好的优化。选择哪个编译器取决于目标平台：DirectX 11 可用 FXC 或 DXC，DirectX 12 必须用 DXC，Vulkan 需要用 DXC 生成 SPIR-V。

编译命令行工具分别是 `fxc.exe` 和 `dxc.exe`。Visual Studio 内置了这两个编译器，可通过项目配置自动编译着色器。编译选项包括优化级别（/O0-O3）、调试信息（/Zi）、着色器模型（/target sm_5_0）、输出文件名（/Fo）。编译错误和警告会显示在输出窗口，点击可跳转到对应行。

```hlsl
// 编译命令示例
// fxc.exe /T ps_5_0 /E PSMain /Fo pixelShader.cso pixelShader.hlsl
// dxc.exe -T ps_6_0 -E PSMain -Fo pixelShader.cso pixelShader.hlsl
```

运行时加载着色器使用 D3DReadFileToBlob 读取编译后的字节码，然后调用 CreateVertexShader/CreatePixelShader 创建着色器对象。这种预编译模式避免了运行时编译开销，但需要维护多个字节码文件（顶点、像素、几何、曲面细分、计算）。DirectX 12 的根签名系统进一步简化了资源绑定，着色器可在 HLSL 中声明根签名，与应用自动同步。

## DirectX 12 特性

DirectX 12 引入了波前操作（Wave Intrinsics），允许着色器访问 SIMD 群内的线程。WaveGetLaneCount 返回波的大小（通常 32 或 64），WaveGetLaneIndex 返回当前线程在波内的索引。WaveActiveAnyTrue、WaveActiveAllTrue 检测波内是否有线程满足条件，WaveActiveSum、WaveActiveMax 计算波内线程的聚合值。这些操作可用于优化算法（减少原子操作、实现快速前缀和）。

射线追踪（Ray Tracing）是 DirectX 12 的重要特性，通过 Raytracing Shaders 实现。RayGeneration Shader 发射射线，Intersection Shader 计算射线与几何的交点，AnyHit Shader 检测射线是否击中几何，ClosestHit Shader 处理最近的交点，Miss Shader 处理未击中的情况。射线追踪需要硬件支持（RT Core），可实现真实反射、阴影、全局光照等效果。

```hlsl
// 射线生成着色器
RaytracingShaderConfig MyConfig = {
    16, // max payload size
    8   // max attribute size
};

[shader("raygeneration")]
void RayGen() {
    float2 uv = (float2(DispatchThreadID.xy) + 0.5) / ViewDimensions;
    float3 origin = CameraPosition;
    float3 direction = normalize(uv.x * CameraRight + uv.y * CameraUp + CameraForward);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 1000.0;

    float4 color = TraceRay(ray);
    Output[DispatchThreadID.xy] = color;
}
```

网格着色器（Mesh Shader）是 DirectX 12 的另一种新特性，取代了传统的顶点/几何着色器流程。网格着色器直接生成图元（点、线、三角形），可动态改变几何拓扑，实现更灵活的几何处理。放大着色器（Amplification Shader）可启动多个网格着色器实例，用于细分和剔除。这种新的几何管线比曲面细分更高效，适合复杂几何和程序化生成。

## 性能优化

HLSL 的性能优化原则与 GLSL 类似，但有一些 DirectX 特定的技巧。使用 `flattens` 和 `branch` 属性控制分支编译，`[flatten]` 将 if-else 编译为条件移动（无分支开销），`[branch]` 保留分支（适合 if 的一个分支总是成立的情况）。使用 `groupshared` 内存共享线程组内数据，减少全局内存访问。使用 `WaveActiveMax` 等波前操作代替原子操作，提升性能。

早期深度测试（Early-Z）是重要的优化，在片段着色器前执行深度测试，剔除被遮挡的片段。确保着色器不修改深度（`earlydepthstencil` 属性），驱动会自动启用 Early-Z。对于 Alpha 测试，使用 `clip()` 或 `discard` 会禁用 Early-Z，可考虑用 Alpha-to-Coverage 或预乘 Alpha 替代。

性能分析工具包括 PIX（微软官方）、RenderDoc（开源）、Nsight（NVIDIA）、Radeon GPU Profiler（AMD）。这些工具可捕获帧并分析着色器执行时间、资源使用、管线状态。PIX 的调试器支持单步执行着色器、检查变量值、查看 GPU 内存。优化应基于实际测量，热点分析（profiling）比猜测更可靠。

## 与 GLSL 的差异

HLSL 与 GLSL 的语法类似，但有一些重要差异。HLSL 使用 `mul(向量, 矩阵)`，GLSL 使用 `矩阵 * 向量`（HLSL 是行向量，GLSL 是列向量）。HLSL 的矩阵是行主序（row-major），GLSL 是列主序（column-major），需要在应用中正确设置转置。HLSL 的 `Texture2D::Sample()` 对应 GLSL 的 `texture()`，参数顺序相同。HLSL 的 `SV_Position` 对应 GLSL 的 `gl_Position`，系统值都用 `SV_` 前缀。

SPIR-V Cross、glslang 等工具可实现 HLSL 与 GLSL 的互相转换。跨平台引擎（如 Unreal、Unity）通常用 HLSL 作为主语言，然后转换为其他平台的着色器语言。这种策略减少了维护成本，但也限制了平台特定特性的使用。
