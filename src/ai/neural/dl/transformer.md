





# transformer


## 注意力机制

seq2seq

## 结构
引入多头自注意力机制。

多头自注意力层
前馈全连接层
残差层 和 归一化层


## Transformer（变换器）

Transformer 由 Google 在 2017 年的论文《Attention is All You Need》中提出，彻底改变了 NLP 的格局。它完全抛弃了 RNN 的递归结构，仅依赖**自注意力机制（Self-Attention）**来建模序列。

### 为什么需要 Transformer？

#### RNN/LSTM 的根本问题

1. **无法并行化**：时间步 $t$ 依赖于 $t-1$，必须串行计算
2. **长距离依赖仍困难**：即使 LSTM，信息传递路径仍然是线性的
3. **计算效率低**：长序列训练极慢

#### Transformer 的核心思想

**"直接建模任意两个位置之间的关系"**

- 不再逐步传递信息，而是让每个词直接"看到"序列中的所有其他词
- 通过注意力权重动态决定关注哪些词
- 所有位置可以并行计算

### Transformer 整体架构

Transformer 采用**编码器-解码器（Encoder-Decoder）**架构：

```
输入序列 → [编码器] → 语义表示 → [解码器] → 输出序列
```

**编码器（Encoder）**：
- 6 层（原论文），每层包含：
  - 多头自注意力层（Multi-Head Self-Attention）
  - 前馈全连接层（Feed-Forward Network）
  - 残差连接 + 层归一化（Residual + LayerNorm）

**解码器（Decoder）**：
- 6 层，每层包含：
  - 掩码多头自注意力（Masked Multi-Head Self-Attention）
  - 编码器-解码器注意力（Cross-Attention）
  - 前馈全连接层
  - 残差连接 + 层归一化

### 自注意力机制（Self-Attention）

自注意力是 Transformer 的核心创新，它允许每个词关注序列中的所有词（包括自己）。

#### 数学公式

**输入**：
- 序列的词向量矩阵 $X \in \mathbb{R}^{n \times d}$（$n$ 个词，每个词 $d$ 维）

**三个投影矩阵**：
$$
\begin{aligned}
Q &= XW^Q \quad \text{（查询 Query）} \\
K &= XW^K \quad \text{（键 Key）} \\
V &= XW^V \quad \text{（值 Value）}
\end{aligned}
$$

其中 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ 是可学习的参数矩阵。

**注意力计算**：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

#### 直觉理解

1. **Query（查询）**：我想找什么信息？
2. **Key（键）**：我有什么信息？
3. **Value（值）**：我实际包含的信息内容

**步骤**：
1. 计算相似度：$QK^T$ 得到注意力分数矩阵 $\in \mathbb{R}^{n \times n}$
   - 第 $i$ 行第 $j$ 列表示词 $i$ 对词 $j$ 的关注程度
2. 缩放：除以 $\sqrt{d_k}$ 防止梯度过小（当 $d_k$ 很大时，点积会很大，softmax 梯度接近 0）
3. Softmax：归一化为概率分布（每行和为 1）
4. 加权求和：用注意力权重对 $V$ 加权

**例子**：
```
句子：The cat sat on the mat

"sat" 的 Query 与所有词的 Key 计算相似度：
  The: 0.05
  cat: 0.35  ← 主语
  sat: 0.10
  on:  0.15
  the: 0.05
  mat: 0.30  ← 宾语

"sat" 的最终表示 = 0.05*V(The) + 0.35*V(cat) + ... + 0.30*V(mat)
```

#### 为什么要缩放？

$$
\frac{QK^T}{\sqrt{d_k}}
$$

当 $d_k$ 很大时（如 512），点积的方差会是 $d_k$，导致 softmax 进入饱和区，梯度消失。除以 $\sqrt{d_k}$ 将方差归一化为 1。

### 多头注意力（Multi-Head Attention）

**动机**：单个注意力头可能只能捕捉一种关系（如主谓关系），多头可以捕捉不同类型的依赖。

**公式**：
$$
\begin{aligned}
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\
\text{MultiHead}(Q,K,V) &= \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
\end{aligned}
$$

- 原论文使用 8 个头（$h=8$）
- 每个头的维度 $d_k = d_{\text{model}} / h = 512 / 8 = 64$

**优势**：
- 不同的头可以关注不同的语言学特征
  - 头 1：语法关系（主谓）
  - 头 2：共指关系（代词指代）
  - 头 3：语义关系（近义词）
- 增强模型的表达能力

### 位置编码（Positional Encoding）

**问题**：自注意力是**顺序不变的**（permutation-invariant），无法区分词序。

**解决方案**：在输入嵌入中加入位置信息。

**公式**（原论文使用正弦函数）：
$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{aligned}
$$

其中：
- $pos$ 是位置索引（0, 1, 2, ...）
- $i$ 是维度索引
- $d$ 是嵌入维度

**特点**：
- 不需要学习，直接根据公式生成
- 对相对位置敏感（不同频率的正弦波）
- 可以外推到训练时未见过的序列长度

**现代改进**：
- 可学习的位置嵌入（BERT）
- 相对位置编码（T5）
- 旋转位置编码 RoPE（LLaMA, GPT-NeoX）

### 前馈网络（Feed-Forward Network）

每个 Transformer 层除了注意力，还有一个**位置无关的前馈网络**：

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

- 两层全连接，中间用 ReLU（或 GELU）激活
- 原论文：$d_{\text{model}} = 512, d_{ff} = 2048$（扩大 4 倍）
- 每个位置独立计算（不跨位置交互）

**作用**：
- 增加模型的非线性表达能力
- 注意力负责"信息交互"，FFN 负责"信息变换"

### 残差连接与层归一化

每个子层（注意力或 FFN）后都有：

$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$

- **残差连接（Residual Connection）**：$x + \text{Sublayer}(x)$
  - 缓解梯度消失
  - 允许信息直接流动
- **层归一化（Layer Normalization）**：
  - 稳定训练
  - 加速收敛

### 掩码机制（Masking）

#### 1. Padding Mask

处理变长序列时，短序列会填充（padding），需要在注意力计算时忽略填充位置。

```python
# 注意力分数中，padding 位置设为 -∞，softmax 后变为 0
mask = (seq == PAD_TOKEN)
scores = scores.masked_fill(mask, -1e9)
```

#### 2. Look-Ahead Mask（解码器）

训练时，解码器不能"偷看"未来的词（因果性约束）。

```
时刻 t 只能看到位置 ≤ t 的词：
[1, 0, 0, 0]
[1, 1, 0, 0]  (下三角矩阵)
[1, 1, 1, 0]
[1, 1, 1, 1]
```

### Transformer 的优势

1. **并行化**：所有位置同时计算，训练速度快 10-100 倍
2. **长距离依赖**：任意两词间距离为 1（RNN 中为序列长度）
3. **可解释性**：注意力权重可视化，理解模型关注什么
4. **可扩展性**：增加层数和参数量效果显著（GPT-3: 175B 参数）

### Transformer 的缺点

1. **计算复杂度**：$O(n^2 \cdot d)$（$n$ 是序列长度）
   - 长文本（如书籍）计算量巨大
   - 改进：Sparse Transformer、Linformer、Performer
2. **内存占用**：需要存储 $n \times n$ 的注意力矩阵
3. **缺乏归纳偏置**：不像 CNN 有局部性，需要大量数据学习基础模式

### Transformer 变体

#### 仅编码器（Encoder-Only）

**代表模型**：BERT（Bidirectional Encoder Representations from Transformers）

- **结构**：多层 Transformer 编码器
- **训练任务**：
  - 掩码语言模型（MLM）：随机遮盖 15% 的词，预测被遮盖的词
  - 下一句预测（NSP）：判断两个句子是否连续
- **特点**：双向上下文，适合理解任务
- **应用**：文本分类、NER、问答、情感分析

#### 仅解码器（Decoder-Only）

**代表模型**：GPT（Generative Pre-trained Transformer）系列

- **结构**：多层 Transformer 解码器（带掩码）
- **训练任务**：语言模型（预测下一个词）
- **特点**：单向上下文，自回归生成
- **应用**：文本生成、对话、代码生成、指令遵循
- **著名模型**：GPT-3、GPT-4、LLaMA、Claude

#### 编码器-解码器（Encoder-Decoder）

**代表模型**：T5、BART

- **结构**：完整的 Transformer 架构
- **训练任务**：序列到序列任务
- **应用**：机器翻译、文本摘要、问答

### 代码示例

**简化版自注意力实现**：
```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # 线性变换
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape

        # 计算 Q, K, V
        qkv = self.qkv(x)  # (batch, seq_len, embed_dim*3)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim*3)
        q, k, v = qkv.chunk(3, dim=-1)  # 各自 (batch, num_heads, seq_len, head_dim)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores shape: (batch, num_heads, seq_len, seq_len)

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        attn_output = torch.matmul(attn_weights, v)
        # attn_output shape: (batch, num_heads, seq_len, head_dim)

        # 合并多头
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)

        # 输出投影
        output = self.out(attn_output)
        return output, attn_weights

# 使用示例
model = SelfAttention(embed_dim=512, num_heads=8)
x = torch.randn(2, 10, 512)  # (batch=2, seq_len=10, embed_dim=512)
output, weights = model(x)
print(f"Output shape: {output.shape}")  # (2, 10, 512)
print(f"Attention weights shape: {weights.shape}")  # (2, 8, 10, 10)
```

### Transformer 的影响

Transformer 架构引发了 NLP 的**预训练-微调范式**：

1. **预训练阶段**：在大规模无标注文本上训练（如维基百科、Common Crawl）
   - BERT：340M 参数，16GB 文本
   - GPT-3：175B 参数，45TB 文本

2. **微调阶段**：在下游任务的少量标注数据上微调

这一范式实现了：
- **迁移学习**：预训练模型学到通用语言知识
- **小样本学习**：下游任务只需少量数据
- **统一框架**：一个模型适配多种任务

### 当代大语言模型（LLM）

基于 Transformer 的大语言模型已成为 AI 的基石：

- **GPT-4**（OpenAI）：多模态，指令遵循
- **Claude**（Anthropic）：长上下文（200K tokens）
- **LLaMA**（Meta）：开源，高效
- **Gemini**（Google）：多模态，推理增强

这些模型展示了**涌现能力（Emergent Abilities）**：
- 上下文学习（In-Context Learning）
- 思维链推理（Chain-of-Thought）
- 代码生成与执行
- 多语言翻译

## 三种架构的对比总结

| 特性           | RNN                  | LSTM                   | Transformer                |
| -------------- | -------------------- | ---------------------- | -------------------------- |
| **提出时间**   | 1980s                | 1997                   | 2017                       |
| **核心机制**   | 隐藏状态递归         | 门控 + 细胞状态        | 自注意力                   |
| **长期依赖**   | 差（梯度消失）       | 好（100+ steps）       | 优秀（任意距离）           |
| **并行化**     | 否（必须串行）       | 否                     | 是（完全并行）             |
| **计算复杂度** | $O(n)$               | $O(n)$                 | $O(n^2)$                   |
| **训练速度**   | 慢                   | 慢                     | 快（GPU 加速）             |
| **参数量**     | 小                   | 中（RNN 的 4 倍）      | 大（可扩展到 1000 亿+）    |
| **适用长度**   | < 20 tokens          | < 200 tokens           | < 100K tokens              |
| **可解释性**   | 差                   | 差                     | 好（注意力可视化）         |
| **代表应用**   | 早期语言模型         | Seq2Seq 翻译           | BERT、GPT、现代 LLM        |
| **当前地位**   | 基本被淘汰           | 少数场景仍在使用       | 主流架构                   |

## NLP 发展时间线

```
1950s: 规则系统（机器翻译的开端）
1990s: 统计方法（N-gram, HMM）
2003: 神经语言模型（Bengio）
2013: Word2Vec（词向量革命）
2014: Seq2Seq（机器翻译）
2015: Attention 机制（Bahdanau）
2017: Transformer（Attention is All You Need）
2018: BERT（预训练-微调范式）
2018: GPT-1（生成式预训练）
2019: GPT-2（展示 zero-shot 能力）
2020: GPT-3（175B，few-shot 学习）
2022: ChatGPT（RLHF，对话能力）
2023: GPT-4、Claude、LLaMA（多模态，长上下文）
```

## 学习建议

### 初学者路径

1. **基础知识**
   - 线性代数（矩阵运算）
   - 概率统计（条件概率、贝叶斯）
   - Python + PyTorch/TensorFlow

2. **词向量**
   - Word2Vec（Skip-gram, CBOW）
   - GloVe
   - FastText

3. **RNN/LSTM**
   - 实现简单的文本分类
   - 理解梯度消失问题
   - 尝试情感分析任务

4. **Transformer**
   - 从头实现自注意力
   - 使用 Hugging Face Transformers 库
   - 微调预训练模型（如 BERT）

5. **大语言模型**
   - Prompt Engineering
   - Fine-tuning vs Few-shot
   - RAG（检索增强生成）

### 实践项目推荐

1. **文本分类**：IMDb 情感分析、新闻分类
2. **命名实体识别**：CoNLL-2003 数据集
3. **机器翻译**：WMT 翻译任务
4. **问答系统**：SQuAD 数据集
5. **对话系统**：基于 GPT-2 的聊天机器人

### 学习资源

**在线课程**：
- Stanford CS224N（NLP with Deep Learning）
- Hugging Face Course（免费）
- Fast.ai NLP Course

**经典论文**：
- "Attention is All You Need" (Transformer)
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "Language Models are Few-Shot Learners" (GPT-3)

**工具库**：
- Hugging Face Transformers
- spaCy（工业级 NLP）
- NLTK（传统 NLP）
- OpenAI API / Anthropic API

## 总结

自然语言处理经历了从规则、统计到深度学习的演进：
- **RNN** 开创了神经网络处理序列的先河，但受限于梯度消失
- **LSTM** 通过门控机制解决了长期依赖问题，统治了 NLP 近十年
- **Transformer** 抛弃递归，用自注意力实现并行化和全局建模，成为当今主流

如今，基于 Transformer 的大语言模型已经展现出惊人的能力，从文本生成到代码编写，从多轮对话到复杂推理。理解这些架构的演进过程，不仅能掌握技术细节，更能洞察 AI 发展的底层逻辑。

NLP 的未来仍在快速演进：更长的上下文、更高效的架构、多模态融合、可控生成、可解释性等方向都充满机遇。掌握这些基础知识，是在 AI 时代保持竞争力的关键。
