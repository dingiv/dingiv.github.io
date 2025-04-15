# grid布局
grid布局可以减少元素层级的嵌套，并提供双轴控制的弹性布局。使用grid的时候需要规定两个轴，有三种情况，两个固定轴，固定行轴+自由列轴，自由行轴+固定列轴，所谓自由就是该方向上的轴数量不固定。

布局时，和flex布局一样，需要对父子元素都进行定义。
## 父元素属性
在父元素上规定三种轴类型中一种。
```
grid-template-row/column
grid-auto-row/column
grid-template-area
```

也可以规定对齐方式
```
align-content/items
justify-content/items
```

## 子元素属性：
规定子元素的弹性
```
grid-column
grid-row
grid-area
```

规定子元素的自对齐
```
justify/align/place-self
```

## grid keyword
用于指定特殊的弹性值，这些弹性可以达到特定的grid效果，独属于grid布局
```
min-content、max-content、fr、auto
fit-content()、minmax()、min()、max()
repeat():  repeat(auto-fill/auto-fit/\<number\>, \<keyword1, keyword2\>)
```
