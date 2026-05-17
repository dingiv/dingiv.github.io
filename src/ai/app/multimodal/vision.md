---
title: 视觉理解
order: 1
---
# 视觉理解
视觉大模型（GPT-4o、Claude 3.5、Qwen-VL）能够理解图像内容——识别物体、读取文字、解析图表、理解空间关系。这项能力为 RAG 系统处理非文本文档、Agent 操控 GUI、自动化测试验证等场景打开了新的可能。

## 图像理解
多模态模型的图像理解基于 Vision Transformer（ViT）架构。图像被切分为固定大小的 patch（如 16×16 像素），每个 patch 经过线性投影后作为 Transformer 的输入 token，与文本 token 一起参与注意力计算。这种方式让模型能够在统一框架内处理图像和文本。

实际使用中，图像通过 base64 编码或 URL 作为 API 请求的一部分发送：

```python
import openai

client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片的内容"},
            {"type": "image_url", "image_url": {
                "url": "https://example.com/photo.jpg",
                "detail": "high"  # high/low/auto 控制分辨率
            }}
        ]
    }]
)
print(response.choices[0].message.content)
```

`detail` 参数控制图像的解析精度。`high` 模式会将图像放大并切分为多个 tile 分别处理，token 消耗更高但能识别更细节的内容；`low` 模式使用缩略图，消耗约 85 tokens，适合只需要整体理解的场景。

## 文档 OCR
传统 OCR（Tesseract、PaddleOCR）将图像中的文字逐字符识别，但丢失了文档的布局和结构信息。视觉大模型的 OCR 能力更强——它不仅识别文字，还理解标题层级、段落结构、表格行列关系，输出结构化的文档内容。

```python
# 用视觉模型解析含表格的 PDF 页面
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": """请解析这张文档图片，按以下格式输出：
1. 识别所有标题和层级
2. 提取所有文字内容
3. 表格以 Markdown 表格格式输出
4. 保留段落结构"""},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }}
        ]
    }]
)
```

视觉 OCR 的优势在于对复杂布局的处理能力——多栏排版、图文混排、跨页表格等传统 OCR 的痛点，视觉模型能通过整体理解来正确处理。缺点是成本高（每页约 1000+ tokens）和延迟大（每页 2-5 秒），不适合大批量处理。生产环境通常是先用传统 OCR 做批量处理，对 OCR 失败的页面再用视觉模型兜底。

## 图表解析
图表（柱状图、折线图、饼图）是文档中信息密度最高的部分，也是传统文本解析最难处理的内容。视觉模型可以直接理解图表的语义——读取轴标签、数据点、图例、趋势，将视觉信息转换为结构化数据。

对于 RAG 系统，图表解析的策略是将图表的语义描述作为文档的一部分索引。例如，一个折线图的 chunk 包含"图 3.2：2020-2025 年营收增长趋势，从 500 万增长到 2000 万，年复合增长率约 30%"，这样用户查询"营收增长率"时可以检索到这个图表。

## Computer Use
Computer Use 是视觉理解在 Agent 领域的前沿应用——模型通过截图理解当前屏幕状态，决定下一步操作（点击、输入、滚动），形成"看→思考→操作"的闭环。Claude 的 Computer Use 和 Cline 的 browser_action 就是这种模式的实现。

Computer Use 的技术链路：定时截屏 → 将截图发送给视觉模型 → 模型分析界面状态和任务目标 → 输出操作指令（坐标 + 动作类型）→ 执行操作 → 再次截屏 → 循环。

这种模式的可靠性目前还不够高。视觉模型可能将相似按钮混淆，坐标定位存在像素级偏差，动态内容（动画、视频）的处理更加困难。工程上的改进方向包括：操作前的目标确认（高亮即将操作的元素让用户确认）、操作后的结果验证（截图对比确认操作是否生效）、失败时的自动回退。
