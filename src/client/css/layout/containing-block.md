# 包含块
一个元素的尺寸和位置经常受其包含块（containing block）的影响。大多数情况下，包含块就是这个元素最近的祖先块元素的内容区域，但也不是总是这样。理解包含块的定义是解决复杂布局问题的基础和前提。 

## 包含块对元素位置的影响
一个元素的偏移属性（包括绝对定位absolute、固定定位fixed、粘性定位sticky的元素）是相对于其包含块进行计算的，即top、bottom、left、right这4个偏移属性；相对定位relative的元素的偏移是基于自身的，静态定位static的元素无法偏移；

## 包含块对元素尺寸的影响
一个元素的width、height、padding、margin、top、bottom、left、right的百分值是根据其包含块的尺寸来进行计算的。具体的计算见

## 确定元素的包含块
确定一个元素的包含块的过程完全依赖于这个元素的`position`属性：当这个元素的`position`为不同的值时，该元素的包含块也会有不同的取向。
+ static、relative、sticky：选择距离该元素最近的祖先块元素的**内容区**作为包含块，这些祖先块元素可以是inline-block,、block、list-item、table container、flex container、grid container等；
  + relative布局的偏移规则：根据自身的尺寸进行偏移；
  + sticky布局的偏移生效需要有另一个限制，就是scroll上下文，**在该元素到其包含块的所有中间嵌套元素必须保持overflow为visiable**，否则将无法生效。基于此特点，建议在使用sticky时尽量少地跨越太多层级嵌套的去寻找包含块元素。
+ absolute：选择距离该元素最近的`position`的值不`static`，也就是值为fixed、absolute、relative、sticky的祖先元素的**内边距区**作为包含块。
+ fixed：在连续媒体的情况下 (continuous media) 包含块是**视口（viewport）**,在分页媒体 (paged media) 下的情况下包含块是**分页区域 (page area)**。
+ 在特定情况下，如果`position`属性是absolute或fixed，包含块也可能是由满足以下条件的最近父级元素的内边距区的边缘组成的：
  + transform 或 perspective 的值不是 none
  + will-change 的值是 transform 或 perspective
  + filter 的值不是 none 或 will-change 的值是 filter（只在 Firefox 下生效）。
  + contain 的值是 layout、paint、strict 或 content（例如：contain: paint;）
  + backdrop-filter 的值不是 none（例如：backdrop-filter: blur(10px);）
  - ⚠️ps: perspective 和 filter 属性对形成包含块的作用存在浏览器之间的不一致性。

## 百分比长度的坑
width、height的百分比，相对的是父元素的content-width或者content-height，如果该元素为绝对定位，则相对于偏移父元素的width或者height
padding和margin的百分比，相对的是父元素的content-width，注意，padding-top和padding-bottom也是相对于content-width！！如果该元素为绝对定位，则相对于偏移父元素的width
translate的百分比，相较于自身的margin-width和margin-height。
line-height的百分比或者无符号数字，相对于自身font-size的计算值。

可以通过在内部增加一个占满高度和宽度的新元素从而实现padding设置基准的内移。