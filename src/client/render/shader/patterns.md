---
title: 常见模式
order: 45
---

# 常见模式
图形渲染中的常见效果有标准的着色器实现模式。理解这些模式可快速实现复杂的视觉效果，也为自定义实现提供了参考模板。这些模式涵盖了光照、材质、天空、后处理等多个方面，是图形编程的"标准库"，开发者可以组合和扩展这些模式，实现独特的渲染风格。

## 光照模型

光照模型模拟光线与表面的交互，是片段着色器的核心。Phong 模型是最简单的光照模型，计算环境光、漫反射、镜面反射三个分量。环境光是全局常量，模拟间接光照。漫反射遵循 Lambert 余弦定律，强度等于光线方向与法线的点积。镜面反射遵循 Phong 模型，强度等于反射方向与视线方向的点积的 shininess 次方。Blinn-Phong 使用半程向量代替反射向量，减少一次向量运算，性能更好。

```glsl
// Phong 光照模型
vec3 computePhong(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 color) {
    vec3 ambient = 0.1 * color;

    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * color;

    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = spec * vec3(1.0);

    return ambient + diffuse + specular;
}
```

PBR（Physically Based Rendering）基于微表面理论，使用物理参数（粗糙度、金属度、IOR）计算 BRDF（双向反射分布函数）。PBR 的核心是能量守恒（反射和透射能量总和等于入射能量）和微表面分布（表面由无数微小镜面组成）。Cook-Torrance BRDF 包含漫反射项（Lambert）和镜面反射项（Fresnel + 几何遮蔽 + 微表面分布）。PBR 的优势是跨光照条件保持一致的外观，材质参数更直观（粗糙度而非 shininess）。

```glsl
// Cook-Torrance BRDF (PBR)
vec3 computePBR(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness, float metallic) {
    vec3 H = normalize(V + L);

    // Fresnel (Schlick 近似)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);

    // 几何遮蔽 (Smith)
    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
    vec3 specular = numerator / max(denominator, 0.001);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    float NdotL = max(dot(N, L), 0.0);
    return (kD * albedo / PI + specular) * NdotL;
}
```

## 材质系统

材质系统定义表面的外观属性，包括颜色、粗糙度、金属度、法线贴图、高度贴图等。标准材质使用 PBR 参数（albedo、roughness、metallic、normal），可通过纹理采样或常量指定。法线贴图存储切线空间的法向量，用于模拟表面细节（凹凸、划痕），无需增加几何复杂度。高度贴图（Displacement Map）存储表面高度，用于视差贴图（Parallax Mapping）或几何置换。

```glsl
// 标准材质采样
struct Material {
    sampler2D albedoMap;
    sampler2D roughnessMap;
    sampler2D metallicMap;
    sampler2D normalMap;
    sampler2D aoMap; // 环境光遮蔽
};

vec3 sampleMaterial(Material mat, vec2 uv, vec3 N, vec3 V, vec3 L) {
    vec3 albedo = texture(mat.albedoMap, uv).rgb;
    float roughness = texture(mat.roughnessMap, uv).r;
    float metallic = texture(mat.metallicMap, uv).r;
    vec3 normal = texture(mat.normalMap, uv).rgb;
    normal = normalize(normal * 2.0 - 1.0); // 从 [0,1] 映射到 [-1,1]
    float ao = texture(mat.aoMap, uv).r;

    // 从切线空间变换到世界空间
    mat3 TBN = cotangentFrame(N, -V, uv);
    N = normalize(TBN * normal);

    vec3 color = computePBR(N, V, L, albedo, roughness, metallic);
    color *= ao; // 应用环境光遮蔽
    return color;
}
```

视差贴图（Parallax Mapping）模拟高度贴图的深度效果。基本思想是根据高度贴图偏移纹理坐标，使表面看起来有凹凸感。视差遮挡贴图（Parallax Occlusion Mapping）是改进版本，通过多次迭代和深度测试，减少伪影。视差贴图适用于墙面、地面等平面表面，对于陡峭角度或复杂几何，效果有限。

## 天空渲染

天空渲染模拟大气散射，使天空呈现蓝色，太阳周围呈现白色。Rayleigh 散射（分子散射）与波长的四次方成反比，蓝光散射更强，天空呈现蓝色。Mie 散射（粒子散射）与波长关系较小，太阳光直接穿透，太阳周围呈现白色。程序化天空通过高度和仰角计算散射颜色，避免预计算纹理。

```glsl
// Rayleigh + Mie 散射
vec3 computeSkyColor(vec3 rayDir, vec3 sunDir) {
    float sunHeight = sunDir.y;
    float rayleigh = rayleighPhase(dot(rayDir, sunDir));
    float mie = miePhase(dot(rayDir, sunDir));

    vec3 rayleighColor = vec3(0.05, 0.1, 0.2) * rayleigh;
    vec3 mieColor = vec3(0.2, 0.2, 0.2) * mie;

    return rayleighColor + mieColor;
}
```

天空盒（Skybox）是立方体纹理，存储环境的光照信息。天空盒采样使用视线方向作为纹理坐标，不考虑相机位置（假设天空在无穷远）。天空盒可用于环境反射、漫反射光照（IBL，Image-Based Lighting）。预过滤的天空盒（Prefiltered Environment Map）根据粗糙度预卷积 BRDF，用于 PBR 的环境光照。

空间渲染需要精确的物理模型。大气散射的数值积分可模拟日出日落、云层散射。大气参与介质（volumetric）模拟光线穿过云层、雾气的散射，使用光线步进（Ray Marching）或阴影贴图。空间渲染的性能开销大，通常预计算为 3D 纹理查找表，运行时采样。

## 阴影技术

阴影贴图（Shadow Mapping）是最常用的阴影技术。基本思想是从光源视角渲染深度到纹理，然后在相机视角比较片段深度与阴影贴图深度，判断是否在阴影中。阴影贴图的问题是分辨率有限、阴影边缘锯齿（aliasing）、自身阴影遮挡（shadow acne）。解决方法包括：使用更高的分辨率、PCF（Percentage Closer Filtering）平滑边缘、深度偏移（depth bias）减少 acne。

```glsl
// 阴影贴图采样
float computeShadow(vec3 fragPos, vec3 lightPos, sampler2D shadowMap) {
    vec4 fragPosLightSpace = lightProjectionMatrix * lightViewMatrix * vec4(fragPos, 1.0);
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5; // 从 [-1,1] 映射到 [0,1]

    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;

    // PCF: 采样周围像素并平均
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    return shadow;
}
```

级联阴影贴图（CSM，Cascaded Shadow Maps）处理大面积阴影（如户外场景）。基本思想是将视锥体分割为多个级联（近处小、远处大），每个级联使用独立的阴影贴图。级联边界处的阴影需要平滑过渡（dithering 或 blending）。CSM 的开销与级联数量成正比，通常 3-4 个级联足够。

阴影体积（Shadow Volumes）是几何阴影技术，适用于点光源和聚光灯。基本思想是从光源边缘投射阴影体积，使用模版缓冲区统计进入和退出体积的次数，奇数次在阴影内，偶数次在阴影外。阴影体积的优势是像素完美阴影，缺点是几何复杂度高、不支持软阴影。射线追踪（Ray Traced Shadows）是物理准确的阴影技术，发射射线向光源，检测遮挡。射线追踪支持软阴影（面积光源）、半透明阴影，但计算成本高，需要硬件支持（RT Core）。

## 后处理效果

后处理效果在渲染完成后对整个图像操作。Bloom 提取高亮区域并模糊叠加，模拟相机镜头的散射效果。实现步骤：提取高亮（阈值过滤）、降采样（减少模糊开销）、高斯模糊（横向+纵向）、升采样叠加到原图。Bloom 的强度、半径、阈值可调，用于模拟 HDR 效果（光源、反射）。

```glsl
// Bloom 提取高亮
vec3 extractBright(vec3 color, float threshold) {
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > threshold) {
        return color;
    }
    return vec3(0.0);
}

// Bloom 叠加
vec3 applyBloom(vec3 color, vec3 bloom) {
    return color + bloom; // 简单叠加
    // return mix(color, bloom, 0.5); // 混合
}
```

色调映射（Tone Mapping）将 HDR 颜色映射到 SDR 显示范围。线性映射（`color / (color + 1.0)`）简单但有失真。Reinhard 映射（`color / (1.0 + color.luminance)`）保留细节但压缩高光。ACES（Academy Color Encoding System）映射是电影工业标准，保留高光细节且色彩准确。曝光调整（`color * exposure`）可在色调映射前调整亮度。

```glsl
// ACES 色调映射
vec3 tonemapACES(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}
```

抗锯齿（Anti-Aliasing）消除边缘锯齿。MSAA（Multi-Sample Anti-Aliasing）在光栅化阶段采样多次，每个像素多个样本，深度/模板测试后平均颜色。MSAA 的开销与样本数成正比（4x MSAA 约 4 倍开销）。TAA（Temporal Anti-Aliasing）复用前一帧数据，通过时域累积减少锯齿，但可能引入重影（需要修复速度/深度）。FXAA（Fast Approximate Anti-Aliasing）检测边缘并模糊，快速但可能模糊细节。

景深（Depth of Field）模拟相机的对焦效果。基本思想根据深度计算模糊半径，对焦区域清晰，前景和背景模糊。实现方法：降采样、循环模糊（多个 pass）、升采样叠加。性能取决于模糊半径和质量，高质量景深需要多次 pass。散景（Bokeh）效果模拟光圈形状，可用纹理或程序化生成。

## 粒子系统

粒子系统模拟大量小物体（火花、烟雾、水滴），用简单的几何表示（点、四边形）。粒子属性包括位置、速度、生命周期、颜色、大小。粒子更新在顶点着色器或计算着色器中完成，每个粒子独立更新（重力、阻力、碰撞）。粒子渲染使用点精灵或四边形公告牌，始终面向相机。

```hlsl
// 粒子更新（计算着色器）
struct Particle {
    float3 position;
    float3 velocity;
    float life;
    float3 color;
    float size;
};

[numthreads(256, 1, 1)]
void CSUpdate(uint3 DTid : SV_DispatchThreadID) {
    Particle p = particles[DTid.x];
    p.velocity += gravity * deltaTime;
    p.position += p.velocity * deltaTime;
    p.life -= deltaTime;
    particles[DTid.x] = p;
}
```

软粒子（Soft Particles）解决粒子与几何的交界问题。基本思想根据深度差调整粒子透明度，靠近几何时淡出，减少 pop-in。软粒子需要场景深度缓冲区，性能开销较大（额外的纹理采样）。粒子纹理（Sprite Sheet）存储多个动画帧，通过 uv 偏移切换帧，实现粒子动画。

粒子性能优化是关键。GPU 粒子使用计算着色器更新，避免 CPU-GPU 同步。粒子排序（按深度）用于正确的透明渲染，但开销较大（排序算法）。粒子剔除（视锥体、遮挡）减少不可见粒子的渲染。粒子池复用粒子内存，避免频繁分配/释放。

## 程序化生成

程序化生成用算法生成纹理、几何、动画，无需预计算资源。噪声函数（Perlin、Simplex、Worley）是程序化生成的基础，生成平滑的随机值，用于地形、云层、大理石纹理。FBM（Fractal Brownian Motion）叠加多个频率的噪声，增加细节。噪声函数的计算成本高，应预计算为纹理或使用低精度近似。

```glsl
// Perlin 噪声（简化版）
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // 平滑插值

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// FBM
float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < octaves; ++i) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}
```

程序化材质用噪声生成纹理（木纹、大理石、火焰）。 Voronoi 图（Worley 噪声）生成细胞状图案，用于鳞片、裂纹。程序化几何用噪声生成地形高度，用 Marching Cubes 生成网格。程序化动画用噪声驱动顶点位置（水面波动、旗帜飘动），节省纹理内存。

射线步进（Ray Marching）是程序化渲染的重要技术。基本思想沿射线步进，检测碰撞（距离场 SDF），用于渲染程序化几何、体积云、液体。SDF（Signed Distance Field）存储点到表面的距离，可程序化计算（球体、盒子的 SDF 简单）。射线步进的性能取决于步进次数，通常需要限制最大步数和早期退出（距离小于阈值）。

这些常见模式是图形编程的基础，开发者可根据需求组合和扩展。理解模式的原理和实现，有助于调试性能问题、创造新的视觉效果、优化着色器代码。图形编程的进步往往来自对这些模式的创新应用，而非完全重新发明。
