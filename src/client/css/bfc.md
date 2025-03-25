# 块级格式上下文
区块格式化上下文（Block Formatting Context，BFC）是 Web 页面的可视 CSS 渲染的一部分，是块级盒子的布局过程发生的区域，也是浮动元素与其他元素交互的区域。其定义洋洋洒洒在[mdn](https://developer.mozilla.org/zh-CN/docs/Web/CSS/CSS_display/Block_formatting_context)写了一堆。

但是总结起来，可以将其作用认为是识别浮动边界、隔离影响。BFC是一个独立的区域，在这个容器中按照一定规则进行物品摆放，并且不会影响其它环境中的物品。
如何创建：总结起来就是，普通的display为block的元素不是，其他大部分都是bfc。例如：
1. 根元素或包含根元素的元素
2. 浮动元素 float ＝ left | right 或 inherit（≠ none）
3. 绝对定位元素 position ＝ absolute 或 fixed
4. display ＝ inline-block | flex | inline-flex | table-cell 或 table-caption
5. overflow ＝ hidden | auto 或 scroll (≠ visible)

在同一个BFC中的儿子元素之间的margin会发生折叠，使用一个激活了BFC的元素包裹可以防止折叠，BFC可以防止float元素影响自己，激活自己的BFC可以防止float改变自己内部的文字的排布。