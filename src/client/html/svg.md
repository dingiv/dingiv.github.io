# SVG
SVG（Scalable Vector Graphics，可缩放矢量图形）是一种基于 XML 纯文本的图片格式，广泛用于网页设计、图标和图形展示。与 JPEG 和 PNG 等位图格式不同，SVG 是矢量格式，具有独特的特性和优势。

现代 Web 浏览器为 HTML 扩展了 SVG 的标准，使得 HTML 具有了 SVG 的能力。SVG 基于 XML 语法，使用类似于 HTML 元素的方式定义图形。根元素是 svg。

## 特点
+ 无损缩放：适合高分辨率显示（如 Retina 屏幕），放大不模糊；
+ 小文件大小：对于简单图形（如图标、logo），文件比位图小；
+ 可编辑性：可以用文本编辑器或代码修改内容；
+ 样式灵活：支持 CSS 和 JavaScript，易于动态调整；
+ SEO 友好：SVG 中的文本可被搜索引擎索引；
+ 透明支持：天然支持透明背景，无需额外处理；
- 复杂图形性能：对于非常复杂的图形（例如包含大量路径），渲染可能变慢；
- 不适合照片：SVG 基于矢量，难以表示复杂的位图图像（如照片）；

## 基本元素
+ svg，文档根元素，定义图片和画布
  ```xml
  <svg width="100" height="100" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  </svg>
  ```
  width，height，图片的尺寸，默认单位为像素，支持绝对单位和相对单位，相对单位例如 %，语义和 CSS 兼容，表示父元素包含块的尺寸；viewBox 定义画布，包括坐标逻辑大小和坐标系原点；例子中的 `0 0 200 200`，代表了将整个图片的 100 个像素均分为 200 份坐标网格，左上角的坐标为 `(0,0)`，右下角的坐标为 `(200,200)`。
+ rect，矩形
  ```xml
  <rect x="10" y="10" width="80" height="80" fill="blue" />
  ```
  x, y：左上角坐标。width, height：宽高。fill：填充颜色。
+ circle，圆形
  ```xml
  <circle cx="50" cy="50" r="40" fill="red" />
  ```
  cx, cy：圆心坐标；r：半径。
+ ellipse，椭圆
  ```xml
  <ellipse cx="50" cy="50" rx="40" ry="20" fill="green" />
  ```
  rx, ry：水平和垂直半径。
+ line，直线
  ```xml
  <line x1="10" y1="10" x2="90" y2="90" stroke="black" />
  ```
  x1, y1：起点坐标。x2, y2：终点坐标。stroke：描边颜色。
+ polyline，折线
  ```xml
  <polyline points="10,10 50,50 90,10" stroke="black" fill="none" />
  ```
  points：一系列坐标点。
+ polygon，多边形
  ```xml
  <polygon points="50,10 90,90 10,90" fill="yellow" />
  ```
  points：定义封闭多边形的顶点。
+ path：定义封闭多边形的顶点。
  ```xml
  <path d="M10 10 L50 50 L90 10" stroke="black" fill="none" />
  ```

## 进阶元素
+ g，分组元素，虚元素，将多个元素组合，统一应用样式或变换。
  ```xml
  <g fill="blue">
      <circle cx="30" cy="30" r="20" />
      <rect x="50" y="50" width="40" height="40" />
  </g>
  ```
+ text，文本元素，用于向图片中嵌入文本
  ```xml
  <text x="50" y="50" font-size="20" text-anchor="middle">Hello</text>
  ```
  text-anchor：文本对齐（start、middle、end）。
+ image，用于嵌入位图（如 PNG/JPEG）
  ```xml
  <image x="10" y="10" width="80" height="80" href="example.png" />
  ```

## SVG 动画
SVG 自带 SMIL（Synchronized Multimedia Integration Language）动画功能。

```xml
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="50" r="40" fill="red">
        <animate attributeName="r" from="40" to="20" dur="2s" repeatCount="indefinite" />
    </circle>
</svg>
```
