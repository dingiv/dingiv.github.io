---
title: OpenGL
order: 10
---

# GLSL
OpenGL Shading Language（GLSL）是 OpenGL 图形 API 的标准着色语言，用于编写运行在 GPU 上的可编程管线阶段。GLSL 语法类似 C 语言，增加了向量、矩阵、纹理采样等图形专用数据类型和内置函数，是学习图形编程的入门语言。

GLSL 的发展与 GPU 硬件演进同步。早期 GLSL（OpenGL 2.0）仅支持顶点着色器和片段着色器，对应可编程管线的两个端点。OpenGL 3.0 引入几何着色器，可动态生成图元。OpenGL 4.0 引入曲面细分着色器，支持自适应细分表面。OpenGL 4.3 引入计算着色器，将 GPU 用于通用计算（GPGPU）。尽管现代 Vulkan/SPIR-V 成为新趋势，GLSL 仍然广泛用于教学、原型开发和移动平台（OpenGL ES）。

## 基础语法
GLSL 的变量声明与 C 类似，但增加了图形相关的类型限定符。`attribute`（仅顶点着色器）表示从顶点缓冲区传入的逐顶点数据，如位置、法线、纹理坐标。`varying` 表示从顶点着色器传递到片段着色器的插值数据，如光照计算所需的法向量。`uniform` 表示在整个绘制调用中恒定的数据，如变换矩阵、光源位置、材质参数。现代 GLSL（OpenGL 3.3+）用 `in`/`out` 代替 `attribute`/`varying`，概念相同但术语更统一。

```glsl
#version 330 core

// 顶点着色器
layout(location = 0) in vec3 aPosition;  // 顶点属性
layout(location = 1) in vec3 aNormal;

uniform mat4 uModelMatrix;   // uniform 变量
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

out vec3 vNormal;  // 传递到片段着色器

void main() {
    gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(aPosition, 1.0);
    vNormal = mat3(uModelMatrix) * aNormal;  // 法线变换
}
```

GLSL 的向量类型支持分量访问（swizzling）。`vec3 color` 可通过 `color.rgb` 获取三分量，`color.rgba` 转换为 vec4，`color.bgr` 交换蓝色和红色。Swizzling 可重复分量（`color.rrr`），用于构造新向量。矩阵采用列主序存储，`mat4 m` 的 `m[0]` 是第一列，`m[2][1]` 是第三行第二列元素。矩阵乘法 `mat4 * vec4` 从右向左结合，注意顺序与数学公式一致。

## 内置函数

GLSL 提供丰富的内置函数，覆盖三角函数、指数对数、几何计算、向量运算等类别。这些函数在 GPU 硬件上实现，通常比手写实现更快。三角函数（sin、cos、tan）接受弧度输入，返回弧度输出。`radians()` 和 `degrees()` 可转换角度单位。指数函数（pow、exp、log）用于光照计算（Phong 模型的镜面反射需要 pow(specular, shininess)）。

几何函数处理向量运算。`dot(x, y)` 计算点积，用于判断光照强度和夹角。`cross(x, y)` 计算叉积，用于计算法向量和切线空间的副切线。`reflect(I, N)` 计算反射向量，用于镜面反射和环境映射。`refract(I, N, eta)` 计算折射向量，用于透明材质渲染。`length(v)` 返回向量长度，`normalize(v)` 归一化，`distance(a, b)` 计算两点距离。

```glsl
// Phong 光照模型
vec3 computePhong(vec3 normal, vec3 lightDir, vec3 viewDir) {
    vec3 ambient = material.ambient * light.ambient;

    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * material.diffuse * light.diffuse;

    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = spec * material.specular * light.specular;

    return ambient + diffuse + specular;
}
```

纹理采样函数从纹理中读取颜色数据。`texture(sampler2D, uv)` 对 2D 纹理采样，自动应用 mipmap 和滤波模式。`textureLod(sampler2D, uv, lod)` 显式指定 mipmap 层级，用于某些特殊效果。`texelFetch(sampler2D, ivec2(coord), lod)` 直接读取指定像素，无滤波，用于计算着色器中的随机访问。`textureGrad(sampler2D, uv, ddx, ddy)` 显式指定导数，用于手动计算 mipmap 层级。

## 数据类型

GLSL 的标量类型包括 `float`、`int`、`uint`、`bool`，分别对应浮点数、整数、无符号整数、布尔值。向量类型是标量的组合，`vec2/vec3/vec4` 是浮点向量，`ivec2/ivec3/ivec4` 是整数向量，`uvec2/uvec3/uvec4` 是无符号向量，`bvec2/bvec3/bvec4` 是布尔向量。矩阵类型是浮点数的二维数组，`mat2/mat3/mat4` 是方阵，`mat2x3/mat4x2` 等是非方阵。

GLSL 没有指针和引用，所有函数参数传值。对于大型结构体，这可能导致性能问题。解决方法是使用 `inout` 限定符传递引用，或将数据放入缓冲区（UBO、SSBO）避免复制。结构体可用于组织复杂数据，但要注意对齐。`std140` 布局规定了数组、结构体、矩阵在缓冲区中的对齐规则，确保 CPU 和 GPU 对数据布局的理解一致。

```glsl
struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

layout(std140) uniform UniformBlock {
    mat4 viewMatrix;
    mat4 projectionMatrix;
    Material material;
    Light lights[MAX_LIGHTS];
};
```

## 顶点着色器

顶点着色器处理每个顶点，执行坐标变换和属性计算。输入来自顶点缓冲区（通过 `layout(location = N)` 绑定），输出是裁剪空间坐标（`gl_Position`）和传递到片段着色器的插值数据。顶点着色器不能创建或销毁顶点，每个输入顶点精确对应一个输出顶点。

坐标变换是顶点着色器的核心任务。模型矩阵将顶点从模型空间变换到世界空间，视图矩阵从世界空间变换到相机空间，投影矩阵从相机空间变换到裁剪空间。法向量需要用法线矩阵（模型视图矩阵的逆转置矩阵）变换，确保非均匀缩放下仍垂直于表面。切线空间用于法线贴图，需要变换切线（T）、副切线（B）、法线（N）三个基向量。

```glsl
#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

out vec3 vFragPos;
out vec3 vNormal;
out vec2 vTexCoord;

void main() {
    vec4 worldPos = uModelMatrix * vec4(aPosition, 1.0);
    vFragPos = worldPos.xyz;
    gl_Position = uProjectionMatrix * uViewMatrix * worldPos;

    mat3 normalMatrix = transpose(inverse(mat3(uModelMatrix)));
    vNormal = normalMatrix * aNormal;
    vTexCoord = aTexCoord;
}
```

## 片段着色器

片段着色器处理每个片段（候选像素），计算最终颜色。输入来自顶点着色器的插值数据，输出是帧缓冲区的颜色（`fragColor`）或深度（`gl_FragDepth`）。片段着色器是光照计算、纹理采样、高级效果（阴影、环境光遮蔽、后处理）的实现位置。

光照模型模拟光线与表面的交互。环境光是全局常量，模拟间接光照。漫反射遵循 Lambert 余弦定律，强度等于光线方向与法线的点积。镜面反射遵循 Phong 或 Blinn-Phong 模型，强度等于反射方向与视线方向的点积的 shininess 次方。PBR 模型基于微表面理论，使用物理参数（粗糙度、金属度、IOR）计算 BRDF（双向反射分布函数）。

```glsl
#version 330 core

in vec3 vFragPos;
in vec3 vNormal;
in vec2 vTexCoord;

uniform sampler2D uDiffuseMap;
uniform sampler2D uNormalMap;
uniform vec3 uLightPos;
uniform vec3 uViewPos;

out vec4 fragColor;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(uLightPos - vFragPos);
    vec3 viewDir = normalize(uViewPos - vFragPos);

    // 光照计算
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * texture(uDiffuseMap, vTexCoord).rgb;

    // 纹理采样
    vec3 result = diffuse;

    fragColor = vec4(result, 1.0);
}
```

## 高级着色器

几何着色器位于顶点着色器和光栅化之间，可接收整个图元（点、线、三角形）作为输入，生成新的图元。典型应用包括法线可视化（将法线渲染为线段）、粒子系统（从点生成分叉的粒子）、公告板技术（从点生成始终面向相机的四边形）。几何着色器灵活性高但性能开销大，过度使用会降低帧率。

曲面细分着色器（Tessellation Shader）用于细分几何，增加顶点密度。曲面细分控制着色器（TCS）指定细分级别，曲面细分评估着色器（TES）计算细分后顶点的位置。典型应用包括自适应地形细节（近处细分更多）、平滑曲线（PN 三角形）、置换贴图（根据纹理高度顶点偏移）。曲面细分需要硬件支持，并非所有 GPU 都可用。

计算着色器（Compute Shader）将 GPU 用于通用计算，不依赖图形管线。计算着色器可访问共享内存（线程组内通信）、原子操作（线程间同步）、图像存储（直接读写纹理）。典型应用包括粒子模拟、物理计算、图像处理、后处理。计算着色器的灵活性使其成为 GPGPU 的首选接口，OpenCL 逐渐被计算着色器取代。

```glsl
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba8) uniform readonly image2D uInput;
layout(rgba8) uniform writeonly image2D uOutput;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec4 color = imageLoad(uInput, pos);

    // 简单的反转效果
    color.rgb = 1.0 - color.rgb;

    imageStore(uOutput, pos, color);
}
```

## 调试与性能

GLSL 的调试比 CPU 代码困难，没有打印语句和断点。常用方法包括：输出颜色作为调试信息（将变量值映射到 RGB 颜色，通过视觉观察验证）、使用 RenderDoc 捕获帧并检查着色器输入输出、利用 `gl_FragColor` 输出中间计算结果。NVIDIA Nsight 和 AMD Radeon GPU Profiler 提供高级调试功能，可单步执行着色器、检查变量值。

性能优化应基于实际测量。RenderDoc 的 Pipeline Statistics 可查看顶点/片段着色器的调用次数，判断是否有过度的几何处理。GPU 时间戳可测量着色器的执行时间，识别瓶颈。常见优化包括：减少分支（分支导致 warp 发散）、使用内置函数（mix 比 if-else 高效）、减少纹理采样（mipmap 线性滤波有开销）、使用早期深度测试（在片段着色器前执行深度测试）。

GLSL 的编译由驱动在运行时完成，首次加载着色器可能卡顿。解决方案包括：预编译着色器为二进制（使用 `glGetProgramBinary` 保存编译结果）、缓存编译结果（应用启动时加载）、使用 SPIR-V（Vulkan 的中间表示，OpenGL 4.6+ 支持）。WebGL 2.0 支持并行编译，异步加载着色器减少阻塞。
