# 梯度下降法
梯度下降（Gradient Descent）是训练机器学习模型的核心优化算法。它的基本思想是：沿着损失函数梯度的**负方向**迭代更新参数，逐步找到使损失函数最小的参数值。它是计算机实践中无法一步求解多层神经网络的最值，而采用的逐步微调逼近的思想，模型的参数在逐步逼近损失函数最小值的过程中得以确定。

### 基本原理

对于参数向量 $\theta$，梯度下降的更新规则为：

$$
\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)
$$

其中：
- $\eta$ 是学习率（learning rate），控制每次更新的步长
- $\nabla J(\theta_t)$ 是损失函数在当前参数处的梯度
- 负号表示沿着梯度下降的方向（使损失减小）

### 三种变体

#### 批量梯度下降（Batch GD）

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{m} \sum_{i=1}^{m} \nabla L(y_i, f(x_i; \theta_t))
$$

- **特点**：每次使用全部 $m$ 个训练样本计算梯度
- **优点**：梯度准确，收敛稳定，理论保证强
- **缺点**：大数据集计算开销巨大，无法在线更新

#### 随机梯度下降（SGD）

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla L(y_i, f(x_i; \theta_t))
$$

- **特点**：每次仅使用一个随机样本
- **优点**：更新速度快，可逃离浅层局部最优，支持在线学习
- **缺点**：梯度噪声大，收敛路径震荡，需要精细调节学习率

#### 小批量梯度下降（Mini-batch GD）

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{b} \sum_{i \in B_t} \nabla L(y_i, f(x_i; \theta_t))
$$

其中 $B_t$ 是大小为 $b$ 的随机小批量（batch size 常取 32、64、128 等）。

- **特点**：折中方案，深度学习的标准做法
- **优点**：
  - 平衡计算效率和收敛稳定性
  - 充分利用 GPU 并行计算
  - 噪声适中，有助于泛化
- **缺点**：需要调节 batch size 超参数

### 现代优化器

为了加速收敛和提高训练稳定性，研究者提出了许多改进算法：

#### Momentum（动量法）

$$
\begin{aligned}
v_{t+1} &= \beta v_t + \eta \nabla J(\theta_t) \\
\theta_{t+1} &= \theta_t - v_{t+1}
\end{aligned}
$$

- **思想**：引入速度（velocity）概念，积累历史梯度
- **优点**：加速收敛，减少震荡，更容易越过小坑
- **典型参数**：$\beta = 0.9$

#### AdaGrad（自适应梯度）

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla J(\theta_t)
$$

其中 $G_t$ 是历史梯度平方和。

- **思想**：为每个参数自适应调整学习率
- **优点**：稀疏特征学习效果好（如NLP）
- **缺点**：学习率单调递减，可能过早停止

#### RMSProp

$$
\begin{aligned}
G_{t+1} &= \beta G_t + (1-\beta) (\nabla J(\theta_t))^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_{t+1} + \epsilon}} \odot \nabla J(\theta_t)
\end{aligned}
$$

- **思想**：改进 AdaGrad，使用指数加权移动平均
- **优点**：避免学习率过快衰减，适合非平稳目标

#### Adam（Adaptive Moment Estimation）

$$
\begin{aligned}
m_{t+1} &= \beta_1 m_t + (1-\beta_1) \nabla J(\theta_t) \\
v_{t+1} &= \beta_2 v_t + (1-\beta_2) (\nabla J(\theta_t))^2 \\
\hat{m}_{t+1} &= \frac{m_{t+1}}{1-\beta_1^{t+1}}, \quad \hat{v}_{t+1} = \frac{v_{t+1}}{1-\beta_2^{t+1}} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}_{t+1}
\end{aligned}
$$

- **思想**：结合 Momentum 和 RMSProp
- **优点**：
  - 对学习率不敏感
  - 收敛快速稳定
  - 适用范围广
- **典型参数**：$\beta_1=0.9, \beta_2=0.999, \eta=0.001$
- **地位**：深度学习最常用的优化器

#### AdamW

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon} + \lambda \theta_t \right)
$$

- **改进**：修正了 Adam 中权重衰减的实现方式
- **优点**：更好的泛化性能，推荐用于 Transformer 等大模型

### 学习率调度策略

固定学习率往往不是最优选择，常用调度策略包括：

1. **Step Decay**：每 N 个 epoch 降低学习率
   $$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/N \rfloor}$$

2. **Exponential Decay**：指数衰减
   $$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

3. **Cosine Annealing**：余弦退火
   $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t\pi}{T}))$$

4. **Warmup**：前期逐渐增大学习率，避免初期不稳定
   $$\eta_t = \eta_{base} \cdot \min(1, \frac{t}{T_{warmup}})$$

## 实践建议

### 调试技巧

1. **监控损失曲线**
   - 训练损失持续下降：模型正在学习
   - 训练损失不降：学习率可能过大或过小，或模型容量不足
   - 训练损失下降但验证损失上升：过拟合，需要正则化

2. **检查梯度**
   - 梯度消失：考虑改变激活函数、使用 BatchNorm、残差连接
   - 梯度爆炸：降低学习率、使用梯度裁剪、检查网络初始化

3. **损失为 NaN**
   - 学习率过大
   - 数值不稳定（如 log(0)）
   - 梯度爆炸

### 超参数选择

| 超参数   | 典型范围              | 调整建议                     |
| -------- | --------------------- | ---------------------------- |
| 学习率   | 1e-4 ~ 1e-2           | 最重要！建议用学习率查找器   |
| Batch    | 16 ~ 512              | 显存允许的情况下尽量大       |
| Optimizer | Adam / AdamW         | 首选 Adam，大模型用 AdamW    |
| 权重衰减 | 1e-5 ~ 1e-3           | 防止过拟合，从 1e-4 开始尝试 |
| Dropout  | 0.1 ~ 0.5             | 过拟合时使用                 |

### 常见陷阱

1. 错误的损失函数选择
   - 回归任务用交叉熵 ❌
   - 多分类用 MSE ❌
   - Softmax 后接 Sigmoid 交叉熵 ❌

2. 数据预处理不当
   - 回归任务目标值未归一化
   - 分类任务标签编码错误
   - 训练集和测试集归一化参数不一致

3. 忽略类别不平衡
   - 99% 准确率可能毫无意义
   - 考虑使用加权损失或重采样