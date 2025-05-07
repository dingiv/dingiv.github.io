# 图形学
计算机图形学是研究如何在计算机上生成、处理和显示图形的学科。它涵盖了从 2D 图像处理到 3D 场景渲染的广泛领域。

## 基本概念

### 坐标系统
1. **世界坐标系**：描述物体在3D空间中的位置
2. **相机坐标系**：以相机为原点的坐标系
3. **屏幕坐标系**：2D显示设备的坐标系
4. **纹理坐标系**：用于纹理映射的UV坐标系

### 基本变换
1. **平移变换**：改变物体的位置
2. **旋转变换**：改变物体的方向
3. **缩放变换**：改变物体的大小
4. **投影变换**：将3D场景投影到2D平面

```js
// 4x4变换矩阵示例
class Matrix4x4 {
  constructor() {
    this.elements = new Float32Array(16);
    this.identity();
  }

  identity() {
    this.elements.fill(0);
    this.elements[0] = 1;
    this.elements[5] = 1;
    this.elements[10] = 1;
    this.elements[15] = 1;
  }

  translate(x, y, z) {
    this.elements[12] = x;
    this.elements[13] = y;
    this.elements[14] = z;
  }

  rotate(angle, axis) {
    // 旋转矩阵实现
  }

  scale(x, y, z) {
    this.elements[0] = x;
    this.elements[5] = y;
    this.elements[10] = z;
  }
}
```

## 渲染管线

现代图形渲染管线包含以下主要阶段：

1. **顶点处理**
   - 顶点着色器
   - 模型变换
   - 视图变换
   - 投影变换

2. **图元装配**
   - 三角形装配
   - 裁剪
   - 背面剔除

3. **光栅化**
   - 扫描转换
   - 深度测试
   - 模板测试

4. **片段处理**
   - 片段着色器
   - 纹理采样
   - 混合

5. **输出合并**
   - 深度测试
   - 模板测试
   - 颜色混合

## 着色器编程

### 顶点着色器
```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
```

### 片段着色器
```glsl
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1;

void main()
{
    FragColor = texture(texture1, TexCoord);
}
```

## 光照模型

1. **环境光**：模拟场景中的全局光照
2. **漫反射**：模拟物体表面的散射光
3. **镜面反射**：模拟物体表面的高光
4. **法线贴图**：增强表面细节

```glsl
// 光照计算示例
vec3 calculateLight(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 lightColor)
{
    // 环境光
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // 漫反射
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // 镜面反射
    float specularStrength = 0.5;
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    return ambient + diffuse + specular;
}
```

## 图形 API

随着图像处理的需求越来越多，质量要求越来越高，速度需求越来越快，GPU被发明出来，用以加速图形处理。为了能够调度这些GPU进行图形计算，需要学习相应的API。

### 主要图形API
1. **OpenGL**：开源的通用API
   - 跨平台支持
   - 丰富的扩展
   - 成熟的生态系统

2. **DirectX/Direct3D**：Windows平台专用API
   - 高性能
   - 与Windows系统深度集成
   - 游戏开发首选

3. **Vulkan**：新一代通用API
   - 低开销
   - 更好的多线程支持
   - 更细粒度的控制

4. **WebGL/WebGPU**：Web平台API
   - 基于OpenGL ES
   - 浏览器原生支持
   - 适合Web应用开发

5. **Metal**：苹果系统专用API
   - 针对Apple硬件优化
   - 低开销
   - 优秀的性能

## 图形渲染引擎
当我们兴致勃勃地开始去学习图形 API 时，又会发现软件生态中一直存在的痛点，一个同样的技术，需要学习多个不同厂商的 API，这无疑加大了程序员的学习负担，因此，我们往往会去学习上层框架，让框架帮我们隐藏底层细节，并获得一致性的开发体验和跨平台的能力。在图形编程当中，需要学习的上层框架就是渲染引擎或者游戏引擎，这些引擎提供了图形 API 的高层封装和有用的工具，可以极大地减轻我们的学习负担，提高开发效率。游戏引擎往往包含了一下套件：
+ 渲染引擎：核心套件
+ 物理引擎：用于模拟物理现象，如碰撞检测、重力、摩擦力等
+ 音频引擎：处理游戏中的声音效果和背景音乐
+ 脚本系统：允许开发者使用编程语言（如C#、Python、Lua等）编写游戏逻辑
+ AI 系统：提供NPC（非玩家角色）行为和决策的 AI 系统
+ 输入管理：处理来自键盘、鼠标、游戏手柄等设备的输入
+ 网络支持：管理网络连接和数据传输，用于多人游戏和在线功能
+ 场景管理：处理游戏场景的加载、管理和切换
+ 资源管理：管理和处理美术资源的工具

### 主流游戏引擎
1. **Unity**
   - 跨平台支持
   - 丰富的资源商店
   - C#脚本支持
   - 适合独立游戏开发

2. **Unreal Engine**
   - 强大的渲染能力
   - 蓝图可视化编程
   - C++支持
   - 适合3A级游戏开发

3. **Godot**
   - 开源免费
   - 轻量级
   - 内置脚本语言
   - 适合2D游戏开发

## 高级渲染技术

1. **全局光照**
   - 光线追踪
   - 路径追踪
   - 光子映射
   - 辐射度

2. **后处理效果**
   - 抗锯齿
   - 景深
   - 运动模糊
   - 色调映射

3. **粒子系统**
   - 流体模拟
   - 烟雾效果
   - 火焰效果
   - 天气系统

4. **虚拟现实**
   - 立体渲染
   - 头部追踪
   - 空间音频
   - 交互系统
