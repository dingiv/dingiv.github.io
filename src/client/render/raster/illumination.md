---
title: 光照模型
order: 3
---
# 光照模型
着色是渲染管线中的核心任务，光照模型是光栅化管线中着色技术的核心原理。光照模型通过考虑光源特性、物体表面材质属性以及观察视角的相互关系，来模拟光照照射物体模型的表面时，物体应当具有的颜色。

## 着色频率
着色频率决定了法线采样的位置和频率。Flat Shading 对每个三角面片使用统一法线（面片法线通过顶点位置叉积计算）导致面片感强烈，适合低多边形艺术风格。Gouraud Shading 在顶点处计算光照后使用重心坐标插值光照强度到内部像素，但无法正确处理高光且会导致马赫带效应。Phong Shading（注意与 Phong 反射模型区分）对每个像素插值法线后独立计算光照，通过双线性插值获得像素级法线，能获得最平滑效果但计算量最大，是现代 GPU 的默认选择。

## Blinn-Phong 反射模型
Blinn-Phong （布林冯）模型是一个被广泛使用的经典光照模型理论，是当今光栅化管线的奠基理论。

### Phong 反射模型

传统 Phong 模型首先计算反射向量，其中 $\vec{l}$ 是光线方向、$\vec{n}$ 是表面法线：

$$
\vec{r} = \text{reflect}(\vec{l}, \vec{n}) = 2(\vec{n} \cdot \vec{l})\vec{n} - \vec{l}
$$

然后计算反射向量与视线方向 $\vec{v}$ 的夹角余弦值。高光项为：

$$
k_s (\max(0, \vec{r} \cdot \vec{v}))^\alpha
$$

指数 $\alpha$ 控制高光集中程度，典型值在 8 到 256 之间。该模型需要计算反射向量，涉及一次向量减法和点积，在 shader 中开销较大。

### Blinn-Phong 改进

Blinn 提出使用半程向量：

$$
\vec{h} = \text{normalize}(\vec{l} + \vec{v})
$$

替代反射向量，高光项改为：

$$
k_s (\max(0, \vec{n} \cdot \vec{h}))^\alpha
$$

半程向量是光线和视线角平分线方向，当视线与光线关于法线对称时与反射向量相同。该改进省去了反射向量计算，仅需一次向量加法和归一化，在高光指数较大时计算效率提升明显。完整的 Blinn-Phong 光照公式为：

$$
L = k_a I_a + k_d I_d \max(0, \vec{n} \cdot \vec{l}) + k_s I_s (\max(0, \vec{n} \cdot \vec{h}))^\alpha
$$

其中 $k_a, k_d, k_s$ 分别是环境光、漫反射和高光系数，$I_a, I_d, I_s$ 是对应的光照强度。

### 能量不守恒问题
Phong 和 Blinn-Phong 模型都存在能量不守恒的问题，当 $k_d + k_s > 1$ 时反射光能量会超过入射光能量。工程中常强制 $k_d + k_s = 1$ 或直接归一化高光项，但这仍不能完全解决问题，这也是该模型被称为"经验模型"而非"物理模型"的主要原因。

## PBR 理论
基于物理的渲染（PBR）通过微表面理论、能量守恒和菲涅尔效应来保证渲染结果的真实性，使得材质在不同光照环境下表现一致。是现代引擎（如 UE5、Unity）的实现标准。

### 微表面理论
微表面理论认为宏观表面由无数微小的镜面组成，每个微镜面都是理想的光滑反射镜。微镜面的法线方向服从某种概率分布，粗糙度描述了法线分布的离散程度，粗糙度越高意味着微镜面法线方向越分散。当表面越粗糙时，微镜面之间相互遮挡和阴影效应越明显，这是高光在掠射角下衰减的主要原因。

### 菲涅尔效应
菲涅尔效应描述了反射率随观察角度变化的物理现象，垂直观察时反射率最低，随着观察角度增大反射率逐渐上升至 1。完整的菲涅尔公式需要考虑材质的折射率，实时渲染中常用 Schlick 近似：

$$
F(\theta) = F_0 + (1 - F_0)(1 - \cos \theta)^5
$$

其中 $F_0$ 是垂直入射时的反射率，对于大多数非金属约为 0.04，金属则可高达 0.8-1.0。这意味着掠射角下几乎所有材质都会变成类似镜面的反射体。

### 能量守恒
能量守恒要求出射光能量不能超过入射光能量，对于不透明材质意味着反射率和透射率之和为 1。在 BRDF 建模中体现为漫反射和镜面反射的能量分配，镜面反射越强的材质漫反射越弱，金属几乎全部能量都用于镜面反射因此没有漫反射项。

## BRDF 双向反射分布函数

BRDF 定义为：

$$
f_r(\omega_i, \omega_o) = \frac{dL_o(\omega_o)}{L_i(\omega_i) \cos \theta_i d\omega_i}
$$

描述了给定入射光照方向 $\omega_i$ 和出射观察方向 $\omega_o$ 时的反射光比例。BRDF 的单位是 $sr^{-1}$（球面度倒数），物理意义是单位立体角内的反射率。

### BRDF 性质

有效的 BRDF 需要满足：线性性允许叠加多个光源的贡献，能量守恒要求：

$$
\int_{\Omega^+} f_r(\omega_i, \omega_o) \cos \theta_o d\omega_o \leq 1
$$

亥姆霍兹互易性 $f_r(\omega_i, \omega_o) = f_r(\omega_o, \omega_i)$ 允许交换光线方向，非负性要求 BRDF 值永远非负。实际渲染中常将 BRDF 拆分为漫反射和镜面反射两项：

$$
f_r = k_d f_r^{diffuse} + k_s f_r^{specular}
$$

其中 $k_s$ 实际上是菲涅尔项在特定角度下的值。

### 漫反射 BRDF

理想的朗伯漫反射 BRDF 是常数：

$$
f_r^{lambert} = \frac{\rho}{\pi}
$$

其中 $\rho$ 是反照率。除以 $\pi$ 的原因是 BRDF 是辐射亮度的比值而非能量的比值，朗伯表面对半球空间积分后反射率为 $\rho$。工程实践中 Oren-Nayar 模型通过引入微表面间的相互遮挡来模拟粗糙表面的漫反射效果，公式涉及入射角和出射角的正弦值与方位角差，能表现出粗糙表面在掠射角下的暗化现象，特别适合模拟布料、泥土等粗糙材质。

## Cook-Torrance 镜面反射 BRDF

Cook-Torrance 模型将镜面反射 BRDF 分解为三项：

$$
f_r^{specular} = \frac{D(\omega_h) F(\omega_o) G(\omega_i, \omega_o)}{4 (\vec{n} \cdot \omega_i)(\vec{n} \cdot \omega_o)}
$$

分母中的几何项修正了微表面面积与宏观表面积的差异。$D$ 是法线分布函数（NDF）描述微镜面法线为半程向量 $\omega_h$ 的概率密度，$F$ 是菲涅尔项描述反射比例，$G$ 是几何遮蔽函数描述微镜面相互遮挡造成的能量损失。

### 法线分布函数 D

Beckmann 分布：

$$
D_{beckmann} = \frac{1}{\pi \alpha^2 \cos^4 \theta_h} \exp\left(-\frac{\tan^2 \theta_h}{\alpha^2}\right)
$$

基于高斯分布假设，$\alpha$ 是粗糙度参数。

GGX 分布（也称 Trowbridge-Reitz）：

$$
D_{GGX} = \frac{\alpha^2}{\pi ((\alpha^2 - 1)\cos^2 \theta_h + 1)^2}
$$

的长尾特性更好，能产生现实中观察到的明亮高光周围柔和的衰减效果。

Blinn-Phong 分布：

$$
D_{blinn} = \frac{\alpha + 2}{2\pi} \cos^{\alpha} \theta_h
$$

计算简单但长尾效果不如 GGX，$\alpha$ 与 Blinn-Phong 高光指数的关系是 $\alpha = shininess/2$。

### 几何遮蔽函数 G

Smith 函数是现代 PBR 中的主流选择，将几何项分解为：

$$
G = G_1(\omega_i) G_1(\omega_o)
$$

对于 GGX 分布，Smith 函数的解析形式为：

$$
G_1(\omega) = \frac{2 \cos \theta}{\cos \theta + \sqrt{\alpha^2 + (1 - \alpha^2) \cos^2 \theta}}
$$

该公式在粗糙表面时能有效模拟遮挡效应。更精确的 Smith 相关函数还考虑了入射光和出射光的相关性，但计算复杂度更高。早期的 Blinn 函数：

$$
G_{blinn} = \min\left(1, \frac{2(\vec{n} \cdot \omega_h)(\vec{n} \cdot \omega_o)}{\vec{\omega}_o \cdot \omega_h}, \frac{2(\vec{n} \cdot \omega_h)(\vec{n} \cdot \omega_i)}{\vec{\omega}_o \cdot \omega_h}\right)
$$

通过取三个值的最小值来近似，计算简单但不够准确。

### 菲涅尔项 F

Schlick 近似：

$$
F_{schlick} = F_0 + (1 - F_0)(1 - \cos \theta)^5
$$

是实时渲染中的标准选择，其中：

$$
F_0 = \left(\frac{n_1 - n_2}{n_1 + n_2}\right)^2
$$

是垂直入射时的反射率。对于导体（金属），$F_0$ 通常在 0.5 到 1.0 之间且与波长相关，因此金属高光带有颜色。对于绝缘体（非金属），$F_0$ 约为 0.04 且几乎不随波长变化，这也是为什么金属和塑料看起来质感不同的根本原因。完整的菲涅尔公式需要考虑 s 偏振和 p 偏振分量，计算过于复杂不适合实时渲染。

## 各向异性反射

各向异性 BRDF 考虑了表面存在方向性纹理（如拉丝金属、头发、布料）时反射特性的变化。这类材质需要定义切线方向 $\vec{t}$ 和副切线方向 $\vec{b}$，微表面分布函数需要在切线平面内分别考虑沿 $\vec{t}$ 和 $\vec{b}$ 方向的粗糙度 $\alpha_t$ 和 $\alpha_b$。各向异性 GGX 的公式为：

$$
D_{aniso} = \frac{1}{\pi \alpha_t \alpha_b \cos^4 \theta_h \left[\frac{(\vec{\omega}_h \cdot \vec{t})^2}{\alpha_t^2} + \frac{(\vec{\omega}_h \cdot \vec{b})^2}{\alpha_b^2} + (\vec{\omega}_h \cdot \vec{n})^2\right]^2}
$$

当 $\alpha_t = \alpha_b$ 时退化为各向同性的 GGX。头发渲染使用 Marschner 模型，考虑了 R（表面反射）、TT（进入后穿出）、TRT（进入后内部反射再穿出）三种光路，需要单独的着色模型处理。

## 折射模型
// TODO: 