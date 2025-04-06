# 继承性

某些 CSS 属性会自动从父元素传递到子元素，而无需在子元素中显式定义。这种行为被称为继承。继承属性通常与文本和字体相关。

## 常见的自动继承的属性：

- 文本相关属性：color, font-family, font-size, font-style, font-variant, font-weight, line-height, text-align, text-indent, text-transform, visibility, white-space, word-spacing, letter-spacing
- 列表属性：list-style-type, list-style-position, list-style-image
- 表格属性：border-collapse, border-spacing, caption-side, empty-cells, table-layout

## 常见的不继承的属性：

- 盒模型属性：margin, padding, border, width, height
- 布局属性：position, top, right, bottom, left, z-index, display, float, clear
- 背景属性：background-color, background-image, background-position, background-repeat, background-size

## 显式控制继承性

使用全局 CSS 常量 inherit、initial、unset 来对相应的属性进行设置即可。

- inherit。要继承
- initial。不继承
- unset。如果属性原本是可继承的，则表现为 inherit，如果原本不是可继承的，则表现为 initial

## 注意 ⚠️

在 CSS 继承时，继承的是父元素的计算值，而不是继承 CSS 属性的指定值。
例如，一个父元素的字体大小为 16px，那么它的子元素通过指定 1.5em 的字体大小，则其计算值为 24px，而孙元素指定字体大小为 inherit，则继承父元素的字体大小 24px，如果孙元素指定其字体大小为 1.5em，其实际的字体大小将为 36px，这是因为继承的值是计算值而不是指定值。
而 line-height 为了提供稳定的行为，相对于元素自身的字体大小，而提供了纯数值的形式，这样可保证不受继承性的影响，推荐在设置 line-height 时使用无单位数值。如下为 MDN 上给的 demo。

```html
<div class="box green">
  <h1>Avoid unexpected results by using unitless line-height.</h1>
  length and percentage line-heights have poor inheritance behavior ...
</div>

<div class="box red">
  <h1>Avoid unexpected results by using unitless line-height.</h1>
  length and percentage line-heights have poor inheritance behavior ...
</div>
```

```css
.green {
  line-height: 1.1;
  border: solid limegreen;
}

.red {
  line-height: 1.1em;
  border: solid red;
}

h1 {
  font-size: 30px;
}

.box {
  width: 18em;
  display: inline-block;
  vertical-align: top;
  font-size: 15px;
}
```

执行效果
![alt text](inheritance.png)
