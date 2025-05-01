# tailwindcss
tailwind 是 css 原子化框架，以 css 后处理器的形式实现，为 css 提供了一组开箱即用的原子类型、一套优雅规范的设计系统、一个简单易用的样式生成器，为开发者提供便利和约束。它显著提高开发效率的同时，提高了 css 的复用性和组合性。

其实现机制基于简单的文件扫描，将选定的文件中的单词进行统计，然后于框架中配置的类名进行比对，如果匹配成功，则在指定的一个 css 文件中，生成该 css 规则的具体代码。

## 主题系统
强大的预定义变体，可以灵活地定义原子级的样式，从而轻松更换，包含三个子系统
颜色系统 colors、尺寸系统 spacing、响应系统 screens（responsive），三个子系统精确地抽象了平时开发中最常用的内容，并将其可操作化

## 指令
`@tailwind`指令可以让用户无须关心路径，对 tailwind 的内置资源进行导入。
`@apply`指令让用户在指定区域内展开一组 tailwind 类的内容。

## 内置资源
包括在三个 layer 中

1. 基础规则 Base，在 layer base 层中，直接作用于普通元素的全局样式；
2. 工具集合 Utilities，在 layer utilities 中，代表一个 css 样式的一系列预定义的原子值，表现为一个类名；
3. 组件 Components，在 layer components 中，代表一个预定预定义的 css class 和样式，它可以被 tailwind 识别并加入 tailwind 的组合系统当中，表现为一组类名

- 注：Utilities 更加松散，为一些基础类，Components 是成块的逻辑块，并且 Component 中的类一般依赖于已经定义好的 Utilities，通过@apply 衍生基础类

## 基本语法

```
Modifier:class
```

`Modifier`可以有多个，最后一个是 class，这两个位置也可以是方括号中加上一个"任意值"，表示临时属性。

## 变体
变体 Variant 与修饰符 Modifier，tailwind 给定一个修饰符代表一个类别产生作用时需要的条件，它被抽象成一个 Variant，同时具象为一个 css 选择器加上一个花括号，被修饰的 component 将被限定在其中，这使得 tailwind 可以简单抽象一种状态、一类元素、一个通用属性，并能够方便快捷地与其他类别进行组合。
这个操作相当于条件判断，只有满足了该条件再声明相应的样式。例如：`hover:`、`xl:`、`after:`等等。

## 任意值
任意值 Arbitrary 和声明式函数调用。

- 类名位置处，`class1-[p1]`相当于调用动态 Utilities "class1"或者动态 Components "class1"函数，并传入参数"p1"，并产生动态样式，`[styleName:styleValue]`相当于退化为完全的行内样式，与 style 属性中填写的行内样式类似；
- 修饰符位置处，`modi1-[p1]:`相当于调用动态 Variant "modi1"，并出入参数"p1"，并产生动态条件，匹配动态的条件，`[atrr1=value1]`或者`[&:hover]`相当于退化为手动写选择器，需要有"&"站位,不占位则默认在最前面,"\["后面一个的位置处。
