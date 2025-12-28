# PyTorch

PyTorch 是由 Meta (Facebook) AI Research 开发的开源深度学习框架，于 2016 年发布。它以动态计算图、Python 原生接口和灵活性著称，已成为学术研究和工业应用中最流行的深度学习框架之一。

## 核心特性

- **动态计算图（Define-by-Run）**：与静态图不同，PyTorch 在运行时构建计算图，使得调试更加直观
- **Pythonic 设计**：API 设计符合 Python 习惯，学习曲线平缓
- **强大的 GPU 加速**：通过 CUDA 实现高效的张量运算
- **丰富的生态系统**：包括 torchvision（计算机视觉）、torchaudio（音频处理）、torchtext（自然语言处理）等

## 张量（Tensor）

张量是 PyTorch 中最基本的数据结构，类似于 NumPy 的 ndarray，但支持 GPU 加速和自动微分。

### 创建张量

```python
import torch

# 从列表创建
x = torch.tensor([[1, 2], [3, 4]])

# 创建特殊张量
zeros = torch.zeros(2, 3)        # 全零张量
ones = torch.ones(2, 3)          # 全一张量
rand = torch.rand(2, 3)          # 随机张量 [0, 1)
randn = torch.randn(2, 3)        # 标准正态分布

# 指定数据类型和设备
x = torch.tensor([1.0, 2.0], dtype=torch.float32, device='cuda')
```

### 张量操作

```python
# 基本运算
y = x + 2
z = torch.matmul(x, y)   # 矩阵乘法
z = x @ y                # 矩阵乘法的简写

# 形状操作
x = x.view(4, 1)         # 改变形状
x = x.reshape(-1, 2)     # 自动推断维度
x = x.transpose(0, 1)    # 转置

# 索引和切片
x[0, :]                  # 第一行
x[:, 1]                  # 第二列
```

### 张量与 NumPy 互转

```python
# Tensor -> NumPy
numpy_array = tensor.cpu().numpy()

# NumPy -> Tensor
tensor = torch.from_numpy(numpy_array)
```

## Module

`nn.Module` 是 PyTorch 中所有神经网络模块的基类，是构建网络的核心抽象。

### 定义模块

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
```

### Module 的重要特性

- **参数管理**：自动追踪所有子模块的参数
- **设备转移**：通过 `.to(device)` 轻松在 CPU/GPU 间移动
- **训练/评估模式**：`.train()` 和 `.eval()` 控制 Dropout、BatchNorm 等层的行为
- **状态保存**：通过 `state_dict()` 保存和加载模型

```python
# 查看所有参数
for name, param in model.named_parameters():
    print(name, param.shape)

# 移动到 GPU
model = model.to('cuda')

# 保存和加载
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

## 自动求导（Autograd）

PyTorch 的自动微分系统是其核心功能，能够自动计算张量操作的梯度。

### 基本用法

```python
# 创建需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x

# 反向传播
y.backward()

# 查看梯度
print(x.grad)  # dy/dx = 2x + 3 = 7
```

### 计算图

- PyTorch 通过 `requires_grad=True` 追踪张量操作
- 每个操作都会在计算图中创建一个节点
- `.backward()` 从输出节点反向遍历计算图，应用链式法则

### 梯度控制

```python
# 临时禁用梯度计算（推理时使用）
with torch.no_grad():
    y = model(x)

# 梯度清零（训练循环中必须）
optimizer.zero_grad()

# 梯度累积
loss.backward()  # 梯度会累加到 .grad

# 分离计算图
y = x.detach()  # y 不再追踪梯度
```

## 神经网络

PyTorch 提供了 `torch.nn` 模块用于构建神经网络。

### 常用层

```python
import torch.nn as nn

# 全连接层
nn.Linear(in_features, out_features)

# 卷积层
nn.Conv2d(in_channels, out_channels, kernel_size)

# 池化层
nn.MaxPool2d(kernel_size)
nn.AvgPool2d(kernel_size)

# 归一化层
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)

# 激活函数
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=-1)

# Dropout
nn.Dropout(p=0.5)
```

### 损失函数

```python
# 分类任务
criterion = nn.CrossEntropyLoss()

# 回归任务
criterion = nn.MSELoss()

# 二分类
criterion = nn.BCELoss()
```

### 优化器

```python
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### 训练循环

```python
model.train()
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}')

# 评估
model.eval()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        # 计算准确率等指标
```

## 数据加载

### Dataset 和 DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(train_data, train_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

## 模型保存与加载

```python
# 方法 1：仅保存参数（推荐）
torch.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(torch.load('model_weights.pth'))

# 方法 2：保存整个模型
torch.save(model, 'model_full.pth')
model = torch.load('model_full.pth')

# 保存检查点（包含优化器状态）
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## GPU 加速

```python
# 检查 CUDA 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 移动模型和数据到 GPU
model = model.to(device)
inputs = inputs.to(device)

# 多 GPU 训练
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 常见实践

### 学习率调度

```python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

### 梯度裁剪

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_x, batch_y in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 相关资源

- [官方文档](https://pytorch.org/docs/)
- [官方教程](https://pytorch.org/tutorials/)
- [PyTorch Lightning](https://lightning.ai/)：高层封装框架
- [Hugging Face Transformers](https://huggingface.co/transformers/)：基于 PyTorch 的 NLP 库