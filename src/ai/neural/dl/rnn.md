# RNN
（循环神经网络）

循环神经网络（Recurrent Neural Network）是最早成功应用于序列建模的深度学习架构，特别适合处理变长序列数据。

### 基本结构

RNN 的核心思想是在序列的每个时间步共享参数，并维护一个**隐藏状态（hidden state）**来记忆历史信息。

**标准 RNN 公式**：
$$
\begin{aligned}
h_t &= \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h) \\
y_t &= W_{hy} h_t + b_y
\end{aligned}
$$

其中：
- $x_t$：时刻 $t$ 的输入（如词向量）
- $h_t$：时刻 $t$ 的隐藏状态
- $y_t$：时刻 $t$ 的输出
- $W_{hh}, W_{xh}, W_{hy}$：权重矩阵（所有时间步共享）
- $b_h, b_y$：偏置项

### 网络展开与前向传播

RNN 可以"展开"成一个时间序列的前馈网络：

```
输入序列: x₁  →  x₂  →  x₃  →  ...  →  xₜ
          ↓      ↓      ↓              ↓
隐状态:   h₁  →  h₂  →  h₃  →  ...  →  hₜ
          ↓      ↓      ↓              ↓
输出:     y₁     y₂     y₃     ...     yₜ
```

### RNN 的优点

1. **参数共享**：不同时间步使用相同参数，模型规模不随序列长度增长
2. **处理变长序列**：可处理任意长度的输入序列
3. **保留时序信息**：隐藏状态充当"记忆"，捕捉历史依赖

### RNN 的严重缺陷

#### 1. 梯度消失与梯度爆炸

**梯度消失**：
- 通过时间反向传播（BPTT）时，梯度连乘导致指数级衰减
- 数学上：$\frac{\partial h_t}{\partial h_1} = \prod_{i=2}^{t} \frac{\partial h_i}{\partial h_{i-1}}$
- 当权重矩阵的最大特征值 < 1 时，梯度趋向于 0
- **后果**：无法学习长距离依赖，只能记住最近几步的信息

**梯度爆炸**：
- 当权重矩阵的最大特征值 > 1 时，梯度指数级增长
- **解决方法**：梯度裁剪（Gradient Clipping）

#### 2. 长期依赖问题

由于梯度消失，RNN 难以捕捉跨度超过 10-20 个时间步的依赖关系。

**例子**：
```
"The cat, which ate the fish that was on the table, is sleeping."
```
RNN 很难学会 "cat" 和 "is" 之间的主谓一致关系（中间隔了太多词）。

#### 3. 训练效率低

由于时间步之间存在依赖，无法并行化训练，速度慢。

### RNN 的变体

- **双向 RNN（Bi-RNN）**：同时从前向后和从后向前处理序列
- **深层 RNN**：堆叠多层 RNN
- **GRU（Gated Recurrent Unit）**：简化版的 LSTM，参数更少

### 实际应用（RNN 时代）

- 语言模型（Language Modeling）
- 机器翻译（配合 Encoder-Decoder 架构）
- 语音识别
- 时间序列预测

**代码示例**：
```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, hn = self.rnn(x, h0)
        # out shape: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步
        return out

# 使用示例
model = SimpleRNN(input_size=100, hidden_size=128, output_size=10)
```

## LSTM（长短期记忆网络）

长短期记忆网络（Long Short-Term Memory）是 Hochreiter 和 Schmidhuber 在 1997 年提出的，专门设计用来解决 RNN 的梯度消失和长期依赖问题。

### 核心创新：门控机制

LSTM 引入了三个"门"来控制信息流动：
1. **遗忘门（Forget Gate）**：决定从细胞状态中丢弃什么信息
2. **输入门（Input Gate）**：决定存储什么新信息到细胞状态
3. **输出门（Output Gate）**：决定输出什么信息

### LSTM 数学公式

LSTM 维护两个状态向量：
- **细胞状态（Cell State）** $C_t$：长期记忆，像"传送带"贯穿整个序列
- **隐藏状态（Hidden State）** $h_t$：短期记忆，传递给下一层

**完整的 LSTM 计算过程**：

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{（遗忘门）} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{（输入门）} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{（候选细胞状态）} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{（更新细胞状态）} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{（输出门）} \\
h_t &= o_t \odot \tanh(C_t) \quad \text{（输出隐藏状态）}
\end{aligned}
$$

其中：
- $\sigma$ 是 Sigmoid 函数，输出 [0, 1]，用于门控
- $\odot$ 表示逐元素乘法（Hadamard product）
- $[h_{t-1}, x_t]$ 表示向量拼接

### 门控机制详解

#### 1. 遗忘门（Forget Gate）

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**作用**：决定从上一时刻的细胞状态 $C_{t-1}$ 中保留多少信息。
- $f_t = 1$：完全保留
- $f_t = 0$：完全遗忘

**直觉**：在阅读新句子时，可能需要忘记上一句的主语。

#### 2. 输入门（Input Gate）

$$
\begin{aligned}
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
\end{aligned}
$$

**作用**：决定将多少新信息加入细胞状态。
- $i_t$ 控制"要不要更新"
- $\tilde{C}_t$ 是候选的更新内容

**直觉**：当遇到新的主语时，将其存入记忆。

#### 3. 更新细胞状态

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

这是 LSTM 的核心：
- 第一项：保留旧记忆（经遗忘门过滤）
- 第二项：添加新记忆（经输入门过滤）

**关键优势**：这个加法操作使得梯度可以直接回传，缓解梯度消失！

#### 4. 输出门（Output Gate）

$$
\begin{aligned}
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

**作用**：决定输出细胞状态的哪些部分。

**直觉**：我们有很多记忆，但只输出当前相关的部分。

### LSTM 的优势

1. **解决长期依赖**：细胞状态像"高速公路"，信息可以跨越很多时间步传递
2. **缓解梯度消失**：加法操作 + 门控机制，梯度可以更好地回传（可处理 100+ 时间步）
3. **选择性记忆**：门控机制允许模型学习何时记住、何时遗忘

### LSTM 的局限

1. **计算复杂**：每个时间步需要计算 4 组权重矩阵，参数量是标准 RNN 的 4 倍
2. **训练慢**：由于时间步依赖，仍然难以并行化
3. **长序列仍有限制**：虽然比 RNN 好，但处理 1000+ token 的长文本仍力不从心
4. **无法建模全局依赖**：只能基于历史信息（单向）或有限的双向上下文

### 双向 LSTM（Bi-LSTM）

许多 NLP 任务需要同时考虑前后文（如命名实体识别）。

**结构**：
- 前向 LSTM：从左到右处理序列
- 后向 LSTM：从右到左处理序列
- 最终输出：拼接两个方向的隐藏状态

$$
h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]
$$

### 实际应用（LSTM 时代 2010-2017）

- **机器翻译**：Seq2Seq with Attention（Google NMT, 2016）
- **语音识别**：DeepSpeech
- **文本生成**：CharRNN（Karpathy）
- **情感分析**：IMDb 电影评论分类
- **命名实体识别**：Bi-LSTM + CRF

**代码示例**：
```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                           num_layers=2,
                           bidirectional=True,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.3)

    def forward(self, text):
        # text shape: (batch, seq_len)
        embedded = self.dropout(self.embedding(text))
        # embedded shape: (batch, seq_len, embed_dim)

        output, (hidden, cell) = self.lstm(embedded)
        # output shape: (batch, seq_len, hidden_dim*2)

        # 取最后一个时间步的输出
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden_cat))

# 使用示例
model = LSTMClassifier(vocab_size=10000, embed_dim=300,
                      hidden_dim=256, output_dim=2)
```

### LSTM 的历史地位

LSTM 统治了 NLP 领域约 7 年（2010-2017），直到 Transformer 的出现。尽管如今大多数任务已被 Transformer 取代，LSTM 仍有其价值：
- 序列长度适中的任务
- 计算资源受限的场景
- 时间序列预测（非 NLP）
- 理解序列模型的基础

