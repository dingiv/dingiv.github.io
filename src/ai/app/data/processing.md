---
title: 数据处理
order: 2
---
# 数据处理
解析后的原始文本通常不能直接使用——包含格式残留、编码不一致、重复内容、过长或过短的片段。数据处理是将原始文本转化为高质量、可检索、可训练数据的关键环节，包括清洗、分块和标准化。

## 数据清洗

### 格式清理
原始文本中常见以下格式噪声需要清理：HTML 标签残留（`<p>`, `<div>`）、多余空白字符（连续空格、制表符、换行）、PDF 解析产生的页码和页眉页脚、OCR 错误（常见于扫描件）、特殊字符（零宽空格、BOM 标记）。

```python
import re

def clean_text(text: str) -> str:
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 标准化空白字符
    text = re.sub(r'\s+', ' ', text)
    # 去除零宽字符
    text = text.replace('​', '').replace('﻿', '')
    # 去除首尾空白
    text = text.strip()
    return text
```

格式清理的边界要谨慎把握。过度清理可能丢失有价值的结构信息——Markdown 的标题标记 `#` 在纯文本中看起来像噪声，但在分块时是重要的段落分隔符。策略是先保留结构信息做分块，分块后再做最终的文本清理。

### 去重
文档库中经常存在重复或近似重复的文档：同一篇文章的不同版本、不同来源的相同内容、模板化的重复文本。去重可以减少索引大小、避免检索结果重复、降低 LLM 处理冗余信息的成本。

精确去重用文档的哈希值（如 SHA-256），完全相同的文档直接去重。模糊去重用 MinHash + LSH（Locality Sensitive Hashing），检测内容高度相似但不完全相同的文档。模糊去重的相似度阈值通常设为 0.8-0.9，低于阈值的保留为独立文档。

### 编码统一
混合来源的文档经常遇到编码问题：UTF-8 和 GBK 混用、特殊 Unicode 字符、全角半角混用。统一编码是数据清洗的第一步：

```python
def normalize_encoding(text: str) -> str:
    # 统一为 Unicode
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    # 全角转半角（标点除外）
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    return text
```

## 分块策略
分块（Chunking）是 RAG 系统中对检索质量影响最大的环节。分块太大，一个 chunk 包含多个主题，检索时噪音多；分块太小，丢失上下文，检索到的片段缺乏完整语义。

### 固定长度分块
最简单的策略，按固定字符数（如 500 字）切分，相邻 chunk 之间保留重叠（如 50 字）避免关键词被截断。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
)
chunks = splitter.split_text(document_text)
```

LangChain 的 `RecursiveCharacterTextSplitter` 不是简单的按字符切分，而是按分隔符层级递归切分——先尝试按双换行（段落）切分，如果段落太长则按单换行（句子）切分，再按句号切分。这种递归策略尽量保持语义完整性。

### 语义分块
语义分块不依赖固定长度或分隔符，而是通过 Embedding 相似度判断语义边界。将文档按句子分割，计算相邻句子的向量相似度，当相似度低于阈值时认为出现了主题切换，在此处切分。

```python
from semantic_text_splitter import TextSplitter

splitter = TextSplitter(max_chunk_size=500)
chunks = splitter.split_text(document_text)
```

语义分块的优势是 chunk 内的主题一致性高，检索时不会返回跨主题的混合内容。劣势是计算成本高（每个句子都需要 Embedding），且对短文档（如 FAQ 条目）没有优势。

### 层级分块
生产环境通常使用多粒度的层级分块策略：文档 → 章节 → 段落 → 句子，每个粒度都有独立的索引。查询时先在大粒度（章节）上定位相关范围，再在小粒度（段落）上精确检索。

层级分块的实现依赖文档的结构信息。Markdown 文档的标题层级天然适合层级分块——`#` 一级标题、`##` 二级标题、`###` 三级标题分别作为不同粒度的 chunk。Word 文档的标题样式、HTML 的 `<h1>`-`<h6>` 标签、PDF 的书签结构都可以用于构建层级。

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)
# 每个 chunk 的 metadata 包含所属的标题层级路径
```

### 代码分块
代码的分块需要基于语法结构而不是字符数。将函数、类作为独立的 chunk 保留完整的语法单元，比按行数切分更有效。AST（抽象语法树）解析器可以精确识别函数和类的边界。

代码分块的一个特殊考虑是"上下文窗口"——一个函数的实现可能引用了文件头部的 import 和全局变量。常见的做法是在每个代码 chunk 前附加文件路径和相关 import 语句，让 LLM 在分析代码时有足够的上下文。

## 元数据管理
每个文档块都应该附带元数据，用于过滤和排序检索结果：

| 元数据字段 | 用途 | 示例 |
|-----------|------|------|
| source | 来源追踪 | `confluence/engineering/page-123` |
| title | 展示和引用 | "API 网关配置指南" |
| section | 章节定位 | "第三章 > API 设计 > 认证" |
| created_at | 时效性判断 | `2026-01-15` |
| updated_at | 增量更新 | `2026-05-10` |
| doc_type | 格式过滤 | `api_doc`, `tutorial`, `faq` |
| language | 语言过滤 | `zh`, `en` |
| tokens | 长度信息 | 450 |

元数据过滤在检索时可以显著提升精度。查询"2025 年的财务报告"时，先用 `updated_at` 和 `doc_type` 过滤，再在候选集上做向量检索，比纯向量检索更准确。Milvus、Weaviate 等向量数据库都支持标量字段过滤与向量检索的组合。

## 数据版本管理
文档库是动态变化的——新文档持续加入、旧文档更新、部分文档删除。数据版本管理追踪这些变化，支持回滚和 A/B 测试。

最简单的方案是基于时间戳的增量处理：记录每个文档的最后处理时间，只重新处理新增或修改的文档。更完善的方案使用 Git-like 的版本管理，每次变更产生一个新的版本快照，可以对比不同版本之间的差异。

DVC（Data Version Control）是数据版本管理的标准工具，它与 Git 集成，用 `.dvc` 文件追踪数据文件的版本，实际数据存储在远程存储（S3、GCS）中。这避免了将大文件直接提交到 Git 仓库，同时保持了版本追踪的能力。
