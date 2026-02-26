---
title: WebGPU
order: 30
---

# WGSL

WebGPU Shading Language（WGSL）是 WebGPU 图形 API 的标准着色语言，专为 Web 平台设计。WGSL 语法类似 Rust，采用显式类型系统和严格的数据流分析，保证了着色器的安全性和可预测性。作为 WebGL 时代的 GLSL 继任者，WGSL 为 Web 应用提供接近原生的图形性能。

WGSL 的设计反映了 Web 平台的特殊需求。着色器以文本形式嵌入 JavaScript，浏览器在运行时编译为后端格式（NVIDIA/AMD 使用 SPIR-V，Intel 使用 DXIL），再转换为 GPU 机器码。这种"二次编译"模式确保了跨平台兼容性，但编译时间成为瓶颈。WGSL 通过显式阶段标注（`@vertex`、`@fragment`、`@compute`）简化了编译器分析，减少验证开销。安全性是另一个核心考虑，WGSL 的类型系统和内存模型防止了数据竞争和非法访问，避免了 GPU 崩溃导致的浏览器沙箱逃逸。

## 基础语法

WGSL 的语法与 Rust 类似，但针对图形编程做了简化。变量声明使用 `let`（不可变）或 `var`（可变），类型必须显式标注。基本类型包括 `f32`（32 位浮点）、`i32`（32 位整数）、`u32`（32 位无符号整数）、`bool`（布尔）。向量类型使用 `vec2<f32>`、`vec3<f32>`、`vec4<f32>` 语法，矩阵使用 `mat2x2<f32>`、`mat4x4<f32>` 语法（注意行列顺序）。

```wgsl
// 顶点着色器
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct Uniforms {
    model_matrix: mat4x4<f32>,
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = uniforms.model_matrix * vec4<f32>(input.position, 1.0);
    output.clip_position = uniforms.projection_matrix * uniforms.view_matrix * world_pos;
    output.normal = (uniforms.model_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    output.uv = input.uv;
    return output;
}
```

WGSL 使用属性（attributes）将变量连接到管线阶段。`@location(N)` 绑定顶点输入或片段输出，`@builtin(position)` 输出裁剪空间坐标，`@group(N) @binding(M)` 绑定资源（纹理、采样器、缓冲区）。`@vertex`、`@fragment`、`@compute` 标注着色器阶段，编译器据此验证输入输出类型。这种显式标注比 GLSL 的全局变量更清晰，也方便工具链分析。

## 内置函数

WGSL 提供与 GLSL/HLSL 类似的内置函数，覆盖三角函数、指数对数、几何计算、向量运算等类别。三角函数（sin、cos、tan）接受弧度输入，`radians()` 和 `degrees()` 转换角度单位。几何函数包括 dot（点积）、cross（叉积）、reflect（反射）、refract（折射）、distance（距离）、length（长度）、normalize（归一化）。这些函数在 GPU 硬件上实现，通常比手写实现更快。

```wgsl
// Blinn-Phong 光照模型
fn compute_lighting(normal: vec3<f32>, light_dir: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let ambient = material.ambient * light.ambient;

    let diff = max(dot(normal, light_dir), 0.0);
    let diffuse = diff * material.diffuse * light.diffuse;

    let half_dir = normalize(light_dir + view_dir);
    let spec = pow(max(dot(normal, half_dir), 0.0), material.shininess);
    let specular = spec * material.specular * light.specular;

    return ambient + diffuse + specular;
}
```

纹理采样函数需要纹理和采样器两个对象。`textureSample(texture, sampler, uv)` 对 2D 纹理标准采样，`textureSampleLevel(texture, sampler, uv, level)` 显式指定 mipmap 层级，`textureSampleGrad(texture, sampler, uv, ddx, ddy)` 显式指定导数。`textureLoad(texture, coords, level)` 直接读取纹素，无滤波。WGSL 的纹理采样设计比 GLSL 更明确，纹理和采样器分离，避免了隐式状态。

## 数据类型与对齐

WGSL 的数据对齐规则严格，确保 CPU 和 GPU 对数据布局的理解一致。标量（f32、i32、u32、bool）对齐到 4 字节。向量的对齐等于其元素大小的 2 倍（vec2 是 8 字节，vec3 和 vec4 是 16 字节）。矩阵的对齐等于其列的对齐，`mat4x4<f32>` 是 16 字节对齐（每列 16 字节）。数组元素的对齐等于其元素类型的对齐，数组大小必须常量表达式（运行时大小需要存储缓冲区）。

```wgsl
struct Material {
    ambient: vec3<f32>,      // offset 0,  size 16
    shininess: f32,          // offset 16, size 4
    diffuse: vec3<f32>,      // offset 32, size 16
    roughness: f32,          // offset 48, size 4
    specular: vec3<f32>,     // offset 64, size 16
    metallic: f32,           // offset 80, size 4
}

// 总大小 96 字节，对齐 16 字节
```

WGSL 的 `struct` 内存布局默认为 `std430`（与 GLSL 的 `std430` 类似），成员按对齐规则排列，可能有填充（padding）。使用 `@size(N)` 和`@align(N)` 属性可手动控制布局，确保与 JavaScript 的 `DataView` 或 `Float32Array` 对齐。存储缓冲区（storage buffer）和对齐更严格，需要显式 `@align` 避免 validation error。

## 计算着色器

WGSL 的计算着色器使用 `@compute` 标注，通过 `@workgroup_size(x, y, z)` 指定工作组大小。全局调用 ID（`@builtin(global_invocation_id)`）是工作组 ID 和本地 ID 的组合，可用于索引数组。工作组内共享内存使用 `var<workgroup>` 声明，需要在 `workgroupBarrier()` 前同步所有线程。

```wgsl
struct Particle {
    pos: vec3<f32>,
    vel: vec3<f32>,
    life: f32,
}

@group(0) @binding(0) var<storage, read> input: array<Particle>;
@group(0) @binding(1) var<storage, read_write> output: array<Particle>;
struct Uniforms {
    delta_time: f32,
}
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    var p = input[index];
    p.pos = p.pos + p.vel * uniforms.delta_time;
    p.life = p.life - uniforms.delta_time;
    output[index] = p;
}
```

计算着色器的常见应用包括粒子模拟、物理计算、图像处理、后处理。WGSL 的计算着色器与 WebGPU 的计算管线集成，可与图形管线共享资源（纹理、缓冲区）。`textureStore()` 可直接写入纹理，无需通过渲染通道，适合计算着色器生成纹理数据（如阴影贴图、环境贴图）。

## 与 JavaScript 交互
WGSL 着色器通过 JavaScript 的 WebGPU API 加载。`device.createShaderModule({ code: wgslCode })` 编译着色器字符串为着色器模块。编译错误可通过 `module.getCompilationInfo()` 获取，包括错误位置和消息。着色器模块不会立即编译为 GPU 机器码，而是在创建管线（`device.createRenderPipeline()` 或 `device.createComputePipeline()`）时编译，这是"二次编译"的第二步。

```javascript
// JavaScript 加载 WGSL 着色器
const shaderCode = `
    @vertex
    fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
        return vec4<f32>(pos, 1.0);
    }
`;

const shaderModule = device.createShaderModule({ code: shaderCode });

const compilationInfo = await shaderModule.getCompilationInfo();
if (compilationInfo.messages.length > 0) {
    for (const message of compilationInfo.messages) {
        console.error(`${message.lineNum}:${message.linePos}: ${message.message}`);
    }
}
```

WebGPU 的绑定组（BindGroup）机制将资源绑定到着色器。`@group(N) @binding(M)` 对应 `device.createBindGroupLayout()` 和 `device.createBindGroup()`。绑定组在创建管线时指定，允许在运行时动态切换资源（如不同的纹理、缓冲区）。这种设计比 OpenGL 的全局绑定更灵活，也更容易优化（驱动可提前验证资源类型）。

## 性能优化

WGSL 的性能优化原则与其他着色语言类似，但有一些 WebGPU 特定的考虑。减少分支是关键，WebGPU 的实现可能在 GPU 上将 WGSL 转换为 SPIR-V 或 DXIL，分支发散仍会导致性能损失。使用 `select()`（三元运算符）代替简单的 if-else，编译器可将其编译为条件移动（无分支）。纹理采样是昂贵的操作，尽量减少采样次数，或使用 mipmap 减少内存访问。

WebGPU 的 `timestamp-query` 可测量着色器执行时间。`device.createQuerySet({ type: "timestamp", count: 2 })` 创建查询集，在编码命令时插入 `writeTimestamp`，渲染后读取查询结果。这种性能分析比 WebGL 的 `EXT_disjoint_timer_query` 更精确，也更容易使用。

计算着色器的优化关注内存访问模式。合并访问（coalescing）可减少内存事务，工作组内线程应访问连续的内存地址。`workgroupBarrier()` 同步工作组内线程，但不应过度使用（同步有开销）。使用 `var<workgroup>` 共享内存减少全局内存访问，但要注意同步正确性（避免数据竞争）。

## 与 GLSL/HLSL 的差异

WGSL 与 GLSL/HLSL 的语法类似，但有一些重要差异。WGSL 使用 `vec3<f32>` 类型标注语法，GLSL 使用 `vec3`（隐式 float），HLSL 使用 `float3`。WGSL 的矩阵是列主序（column-major），与 GLSL 一致，但语法是 `mat4x4<f32>`（注意行列顺序）。WGSL 的 `textureSample()` 需要纹理和采样器两个参数，GLSL 的 `texture()` 只需纹理（采样器隐式绑定）。

WGSL 的属性系统更明确，`@location`、`@builtin`、`@group`、`@binding` 是显式标注，GLSL 使用全局变量和 layout 限定符。WGSL 的结构体成员访问用点（`struct.member`），GLSL 也用点，HLSL 也用点（基本一致）。WGSL 的函数调用参数类型必须匹配，不支持隐式转换，GLSL/HLSL 有一些隐式转换（如 float 到 vec3）。

工具链方面，`naga`（Rust 编写的库）可将 WGSL 转换为 SPIR-V、GLSL、MSL，实现跨平台着色器。`wgpu`（Rust 实现的 WebGPU）使用 naga 作为着色器后端，支持 Windows、macOS、Linux、Android。WebGPU 的 polyfill（`webgpu.hxx`）可将 WebGPU 调用转换为 WebGL，在不支持 WebGPU 的浏览器中运行，但性能会降低。

## 调试与工具

WGSL 的调试比 GLSL/HLSL 更困难，因为 WebGPU 仍处于快速发展阶段。Chrome DevTools 的 WebGPU Inspector 可查看管线状态、资源绑定、着色器代码。RenderDoc 支持 WebGPU 捕获，可分析着色器输入输出。`wgsl-analyzer`（VSCode 插件）提供语法高亮、类型检查、错误提示。

WGSL 的编译错误信息通常包含行号和列号，错误描述比 GLSL 更详细。常见的错误包括：类型不匹配（`cannot add vec3<f32> and vec3<i32>`）、对齐错误（`struct member has misaligned offset`）、资源绑定冲突（`@binding(0) already used`）。这些错误在着色器编译时检测，不会导致 GPU 崩溃，体现了 WGSL 的安全设计。

未来的 WGSL 发展包括更好的 IDE 支持、更多内置函数、更高效的编译器。WebGPU 也将新增特性（间接绘制、射线追踪、网格着色器），WGSL 会相应扩展。WGSL 有望成为 Web 图形编程的标准语言，取代 WebGL 时代的 GLSL，为 Web 应用提供接近原生的图形性能。
