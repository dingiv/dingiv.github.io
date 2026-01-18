# 硬件加速

## DLSS
Deep Learning Super Sampling 深度学习超采样技术，是英伟达（NVIDIA）利用 AI（人工智能）来辅助提升游戏帧数（FPS）且保持高画质的一项黑科技。

DLSS 深度集成在驱动和 DirectX/Vulkan/Metal 接口中。可以由开发者显式调用，从而提升游戏的表现能力，

版本,核心功能,硬件要求,效果
DLSS 1.0,空间放大,RTX 20 系列及以上,初代尝试，画面偶尔会模糊。
DLSS 2.x,时域超采样 (Super Resolution),RTX 20 系列及以上,行业标配。画面极其稳定，甚至有时比原生画质更锐利。
DLSS 3.x,帧生成 (Frame Generation),RTX 40 系列及以上,AI 直接在两帧之间插一帧。帧数翻倍的神器。
DLSS 3.5,光线重建 (Ray Reconstruction),RTX 20 系列及以上,专门优化光线追踪。让水面反射、阴影细节更真实，减少噪点。
DLSS 4.0,多帧合成与 Transformer 模型,RTX 50 系列及以上,2025-2026 年的新技术，极大减少了高速运动下的“鬼影”。

就是人们俗称的**超分和插帧**技术。

## 光追单元
专门用于加速光线追踪计算的硬件单元。