---
title: 文档解析
order: 1
---
# 文档解析
文档解析是数据工程的第一步，目标是从各种格式的文件中提取结构化文本。看似简单的任务，实际上充满挑战——PDF 的表格和公式、扫描件的 OCR、HTML 的噪声标签、代码文件的特殊结构，每种格式都有独特的解析策略。

## PDF 解析
PDF 是企业文档最常见的格式，也是最难的解析对象。PDF 的设计目标是"在所有设备上看起来一样"，而不是"便于提取结构化内容"。PDF 内部是一系列绘制指令（文本位置、字体、图片），没有显式的标题、段落、表格等语义结构。

文本型 PDF 可以直接提取文字，但需要处理多栏布局（左右两栏的文字需要分别读取，而不是按行交错）、页眉页脚（需要识别并去除）、脚注和尾注（需要关联到正文引用位置）。`pypdf` 是最基础的 Python PDF 库，适合结构简单的文档：

```python
from pypdf import PdfReader

reader = PdfReader("report.pdf")
for page in reader.pages:
    text = page.extract_text()
    # 简单的页眉页脚过滤
    lines = text.split("\n")
    content_lines = [l for l in lines if not is_header_footer(l)]
    print("\n".join(content_lines))
```

表格是 PDF 解析的难点。PDF 中的表格是通过线条和文字定位实现的，提取时需要重建单元格与行列的对应关系。`pdfplumber` 和 `camelot` 是两个专门的 PDF 表格提取工具，前者通过分析文字位置重建表格，后者通过检测线条来识别表格结构。

```python
import pdfplumber

with pdfplumber.open("financial_report.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            # table 是二维数组，每行是一个列表
            for row in table:
                print(row)
```

扫描件 PDF（图片型）无法直接提取文字，需要 OCR（光学字符识别）。`pytesseract` 是 Python 中最常用的 OCR 工具，底层使用 Tesseract 引擎。对于中文文档，需要额外下载中文语言包。OCR 的准确率受图片质量、字体、排版影响，通常需要人工校验。

```python
from pdf2image import convert_from_path
import pytesseract

images = convert_from_path("scanned_doc.pdf")
for image in images:
    text = pytesseract.image_to_string(image, lang="chi_sim+eng")
    print(text)
```

专业级 PDF 解析工具正在向多模态方向发展。`marker` 和 `nougat` 使用视觉模型直接理解 PDF 页面的布局，自动识别标题、正文、表格、公式、代码块，输出结构化的 Markdown。这种方法的准确率远高于传统规则解析，尤其擅长处理混合布局（半页文字半页表格）的文档。

## Word 文档
`.docx` 文件本质是一个 ZIP 包，内含 XML 格式的文档结构。`python-docx` 可以读取标题层级、段落、表格、列表等结构化信息，适合按文档逻辑结构做分块处理。

```python
from docx import Document

doc = Document("report.docx")
for para in doc.paragraphs:
    if para.style.name.startswith("Heading"):
        level = int(para.style.name.split()[-1])
        print(f"{'#' * level} {para.text}")
    else:
        print(para.text)

for table in doc.tables:
    for row in table.rows:
        cells = [cell.text for cell in row.cells]
        print(" | ".join(cells))
```

Word 文档的解析难点在于样式的多样性。同一个标题级别可能用不同的样式名（"Heading 1" vs "标题 1" vs 自定义样式），需要建立样式到语义的映射。嵌入的图片和对象（Excel 图表、OLE 对象）需要单独提取和处理。

## HTML 与网页
网页内容提取的核心挑战是区分"内容"和"噪声"（导航栏、广告、侧边栏、评论区）。`trafilatura` 是目前最好的网页正文提取库，基于启发式规则和机器学习模型识别正文区域，准确率高于传统的 `newspaper3k` 和 `readability`。

```python
from trafilatura import fetch_url, extract

url = "https://example.com/article"
html = fetch_url(url)
text = extract(html, include_tables=True, favor_precision=True)
# include_tables: 保留表格内容
# favor_precision: 宁可少提取也不要混入噪声
```

HTML 表格的提取可以用 `pandas.read_html()`，它会自动解析 `<table>` 标签为 DataFrame。对于复杂表格（合并单元格、嵌套表格），可能需要用 `BeautifulSoup` 手动解析。

Markdown 文件是最容易解析的格式，本身就是结构化文本。但需要注意 YAML frontmatter（`---` 包围的元数据区域）和代码块的识别，避免将代码内容误当作正文分块。

## 代码文件
代码文件的解析需要识别语法结构——函数定义、类定义、注释块。LangChain 的 `LanguageParser` 利用语言语法树（AST）进行智能分块，将每个函数或类作为一个独立的文档块，保留函数签名和文档字符串。

```python
from langchain_community.document_loaders import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

loader = GenericLoader.from_filesystem(
    "src/",
    glob="**/*.{py,js,ts}",
    parser=LanguageParser()
)
documents = loader.load()
# 每个文档对应一个函数/类，metadata 包含文件路径和代码结构信息
```

## 解析工具选型

| 工具 | 格式 | 优势 | 适用场景 |
|------|------|------|----------|
| pypdf | PDF | 轻量、纯 Python | 简单文本型 PDF |
| pdfplumber | PDF | 表格提取能力强 | 含表格的 PDF |
| marker | PDF | 多模态布局识别 | 复杂布局 PDF |
| python-docx | Word | 结构化读取 | .docx 文件 |
| trafilatura | HTML | 正文提取准确 | 网页内容 |
| LanguageParser | 代码 | AST 感知分块 | 代码文件 |
| unstructured | 全格式 | 统一接口 | 混合格式场景 |

`unstructured` 是一个值得关注的通用解析工具，它提供了统一的 API 处理 PDF、Word、HTML、Markdown、PPT 等多种格式，内部根据文件类型自动选择最佳解析策略。对于混合格式的文档库，使用 `unstructured` 可以减少维护多种解析器的成本。
