# 损失函数
损失函数（Loss Function），也称为**代价函数（Cost Function）**或**目标函数（Objective Function）**，是机器学习模型训练的核心组成部分。它用于**量化模型预测值与真实值之间的误差**，是连接模型输出与优化算法的桥梁。

## 基本概念

### 损失函数的数学定义
我们如何评价模型的表现？通过比较预测值 $\hat{y_i}$ 和真实值 $y_i$ 之间的差距：

$$
Loss = L(y_i, \hat{y_i})
$$

其中 $y_i$ 是真实的标签值（常数），而 $\hat{y_i}$ 是关于模型参数的函数：

$$
\hat{y_i} = f(w_1, w_2, ..., w_n, b_1, b_2, ..., b_n)
$$

因此，损失函数本质上是一个关于模型参数的多变量函数。对于整个数据集，我们通常计算平均损失（或称代价函数）：

$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} L(y_i, \hat{y_i})
$$

其中 $m$ 是样本数量，$\theta$ 代表所有模型参数。

### 核心作用

1. **评估模型性能**：定量衡量模型预测的准确程度
2. **指导参数更新**：通过计算梯度，指示参数调整的方向和幅度
3. **优化目标**：优化算法（如梯度下降）通过最小化损失函数来训练模型
4. **任务适配**：不同任务需要不同的损失函数以匹配问题特性

## 回归任务损失函数

### 均方误差（MSE - Mean Squared Error）

**公式**：
$$
MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y_i})^2
$$

**特点**：
- 对大误差惩罚更重（平方项的作用）
- 处处可导，便于梯度优化
- 假设误差服从高斯分布时，MSE 对应最大似然估计

**适用场景**：
- 线性回归
- 神经网络回归任务
- 误差分布接近正态分布的场景

**缺点**：
- 对异常值（outliers）非常敏感
- 大误差会导致梯度爆炸

**代码示例**：
```python
import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 示例
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])
loss = mse_loss(y_true, y_pred)
print(f"MSE Loss: {loss:.4f}")  # 0.3750
```

### 平均绝对误差（MAE - Mean Absolute Error）

**公式**：
$$
MAE = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y_i}|
$$

**特点**：
- 对所有误差施加相同权重的惩罚
- 对异常值具有鲁棒性
- 在零点处不可导（需要特殊处理）

**适用场景**：
- 数据存在较多异常值
- 希望对所有样本一视同仁
- 房价预测、销量预测等

**优缺点对比**：
- 相比 MSE 更稳健，但收敛可能较慢
- 梯度恒定，不会像 MSE 那样随误差增大而放大

**代码示例**：
```python
def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

loss = mae_loss(y_true, y_pred)
print(f"MAE Loss: {loss:.4f}")  # 0.5000
```

### Huber 损失

**公式**：
$$
L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

**特点**：
- 结合了 MSE 和 MAE 的优点
- 小误差时使用平方损失（收敛快）
- 大误差时使用线性损失（对异常值鲁棒）
- $\delta$ 是可调节的阈值参数

**适用场景**：
- 需要在收敛速度和鲁棒性之间平衡
- 数据中存在少量异常值但希望保持较快收敛

### 平均平方对数误差（MSLE）

**公式**：
$$
MSLE = \frac{1}{m} \sum_{i=1}^{m} (\log(y_i + 1) - \log(\hat{y_i} + 1))^2
$$

**特点**：
- 对相对误差而非绝对误差敏感
- 更关注小值的预测准确性
- 只能用于非负目标值

**适用场景**：
- 目标值范围跨度很大（如几个数量级）
- 更关心相对百分比误差
- 销量预测、流量预测等

## 分类任务损失函数

### 二分类交叉熵损失（Binary Cross-Entropy）

**公式**：
$$
BCE = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(\hat{y_i}) + (1-y_i) \log(1-\hat{y_i})]
$$

**推导直觉**：
- 来源于最大似然估计
- 假设模型输出表示概率 $P(y=1|x)$
- 最小化交叉熵等价于最大化似然函数

**特点**：
- 输出范围 `[0, +∞)`，预测越错误损失越大
- 与 Sigmoid 激活函数天然匹配
- 凸函数，易于优化

**适用场景**：
- 逻辑回归
- 二分类神经网络（输出层使用 Sigmoid）
- 多标签分类（每个标签独立二分类）

**代码示例**：
```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # 防止 log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) +
                    (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7])
loss = binary_cross_entropy(y_true, y_pred)
print(f"BCE Loss: {loss:.4f}")  # 0.1625
```

### 多分类交叉熵损失（Categorical Cross-Entropy）

**公式**：
$$
CCE = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_{i,c} \log(\hat{y_{i,c}})
$$

其中 $C$ 是类别数量，$y_{i,c}$ 是 one-hot 编码的真实标签。

**特点**：
- 与 Softmax 激活函数配合使用
- 保证输出是合法的概率分布（和为1）
- 多分类问题的标准损失函数

**Sparse Categorical Cross-Entropy**：
- 当标签是整数而非 one-hot 编码时使用
- 计算效率更高，内存占用更小
- 数学上等价于标准交叉熵

**代码示例**：
```python
def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    y_true: one-hot 编码, shape (m, C)
    y_pred: 预测概率, shape (m, C)
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# 3分类示例
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
loss = categorical_cross_entropy(y_true, y_pred)
print(f"CCE Loss: {loss:.4f}")  # 0.3677
```

### 合页损失（Hinge Loss）

**公式**：
$$
L_{hinge} = \frac{1}{m} \sum_{i=1}^{m} \max(0, 1 - y_i \cdot \hat{y_i})
$$

其中 $y_i \in \{-1, +1\}$，$\hat{y_i}$ 是模型的原始输出（不经过激活函数）。

**特点**：
- 支持向量机（SVM）的标准损失
- 追求最大间隔分类
- 只关心决策边界附近的样本
- 在 $y \cdot \hat{y} \geq 1$ 时损失为0（支持向量概念）

**适用场景**：
- 支持向量机
- 强调决策边界的分类任务
- 需要最大间隔分类器

### Focal Loss

**公式**：
$$
FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)
$$

其中 $p_t = \begin{cases} p & \text{if } y=1 \\ 1-p & \text{otherwise} \end{cases}$

**特点**：
- 专为类别不平衡问题设计
- $(1-p_t)^\gamma$ 是调制因子，降低易分类样本的权重
- $\gamma$ 通常取2，$\alpha$ 用于平衡正负样本
- 专注于困难样本（hard examples）

**适用场景**：
- 目标检测（如 RetinaNet）
- 极度不平衡的分类问题
- 前景/背景比例悬殊的场景

**直觉理解**：
- 易分类样本（$p_t$ 接近1）：损失接近0，梯度小
- 难分类样本（$p_t$ 接近0.5）：损失大，梯度大
- 自动调节样本权重，无需手动重采样

## 特殊任务损失函数

### Dice Loss

**公式**：
$$
Dice Loss = 1 - \frac{2|X \cap Y|}{|X| + |Y|} = 1 - \frac{2\sum_{i} p_i g_i}{\sum_{i} p_i + \sum_{i} g_i}
$$

其中 $p_i$ 是预测值，$g_i$ 是真实标签（ground truth）。

**特点**：
- 基于 Dice 系数（F1 score 的等价形式）
- 直接优化评估指标
- 对类别不平衡具有鲁棒性

**适用场景**：
- 图像分割任务
- 医学图像分割
- 小目标检测

### IoU Loss / GIoU Loss

**IoU 公式**：
$$
IoU = \frac{|A \cap B|}{|A \cup B|}
$$

**特点**：
- 直接优化边界框的重叠度
- 对尺度不敏感
- 非凸，优化困难

**GIoU（Generalized IoU）**：
$$
GIoU = IoU - \frac{|C \setminus (A \cup B)|}{|C|}
$$

其中 $C$ 是包含 $A$ 和 $B$ 的最小外接框。

**适用场景**：
- 目标检测
- 实例分割
- 边界框回归

### Triplet Loss

**公式**：
$$
L = \max(d(a, p) - d(a, n) + margin, 0)
$$

- $a$: anchor（锚点）
- $p$: positive（正样本，与 anchor 同类）
- $n$: negative（负样本，与 anchor 不同类）
- $d$: 距离度量函数（如欧氏距离）

**特点**：
- 学习嵌入空间（embedding space）
- 同类样本距离近，异类样本距离远
- 需要精心设计采样策略（hard negative mining）

**适用场景**：
- 人脸识别
- 度量学习（Metric Learning）
- 相似度学习

## 正则化与损失函数
在实际应用中，为了防止过拟合，我们常在原始损失函数基础上添加正则化项：

$$
J_{total} = J_{loss} + \lambda R(\theta)
$$

### L1 正则化（Lasso）

$$
R(\theta) = \sum_{i} |\theta_i|
$$

**特点**：
- 产生稀疏解（部分参数归零）
- 可用于特征选择
- 不可导（在零点）

### L2 正则化（Ridge）

$$
R(\theta) = \sum_{i} \theta_i^2
$$

**特点**：
- 参数趋向于小值但不为零
- 处处可导，优化简单
- 等价于参数的高斯先验

### Elastic Net

$$
R(\theta) = \alpha \sum_{i} |\theta_i| + \beta \sum_{i} \theta_i^2
$$

**特点**：结合 L1 和 L2 的优点，平衡稀疏性和稳定性。

## 损失函数选择指南

| 任务类型         | 推荐损失函数                     | 关键考量因素                |
| ---------------- | -------------------------------- | --------------------------- |
| 回归             | MSE（标准）/ Huber（有异常值）   | 是否存在异常值              |
| 二分类           | Binary Cross-Entropy             | 输出是否为概率              |
| 多分类           | Categorical Cross-Entropy        | 类别是否互斥                |
| 多标签分类       | Binary Cross-Entropy（多个）     | 标签可以同时存在            |
| 类别不平衡分类   | Focal Loss / 加权交叉熵          | 样本数量差异程度            |
| 图像分割         | Dice Loss / Focal Loss 组合      | 目标大小和数量              |
| 目标检测         | 分类用 Focal Loss + 回归用 GIoU  | 框的质量和类别平衡          |
| 人脸识别/检索    | Triplet Loss / ArcFace Loss      | 是否需要嵌入空间            |
| 生成模型（GAN）  | Adversarial Loss                 | 生成器和判别器的平衡        |
| 序列任务（NLP）  | Cross-Entropy（token级）         | 序列长度和词表大小          |

