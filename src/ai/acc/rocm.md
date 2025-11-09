# ROCm


Radeon Open Compute Platform (ROCm)

核心定位：面向高性能计算（HPC）和 AI 的统一平台，类似于 NVIDIA 的 CUDA + cuDNN 体系。

语言与框架：

支持 HIP (Heterogeneous-compute Interface for Portability) → 类 CUDA 语法，可以轻松移植 CUDA 代码；

支持 OpenCL；

支持 PyTorch / TensorFlow 等 AI 框架的后端；

提供 ROCm Libraries（如 rocBLAS、rocFFT、MIOpen 等）。

ROCm 是现在 AMD GPU 计算的主力生态，尤其面向数据中心、AI 训练、科研计算等领域。