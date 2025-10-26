# js 对象基本操作
在 js 中我们可以对一个对象进行一些语言层面上的基本操作，这些操作是潜藏在语言表面语法下的底层执行逻辑。

## 属性操作
属性操作就是指对象上携带的数值，js 中的对象从表现上就是其他语言中的 HashMap 数据结构，用于快速获取散列数据。一个对象上可以有多个属性，每个属性需要使用要给 key 来索引，key 只能是 string 或者 symbol 这两种基本类型（注意数字不是，数字会被转化为 string）。一个属性往往有以下的定义和描述符。
| 描述符       | 定义                             |
| ------------ | -------------------------------- |
| value        | 代表了该属性的值                 |
| writeale     | 代表了属性当前是否能被“修改”     |
| enumeratable | 代表了该属性当前是否能够被“枚举” |
| configurable | 代笔了当前属性是否能够被“配置”   |


+ 属性读取：obj.key
+ 属性赋值：obj.key = value
+ 属性定义：Object.defineProperty
+ 属性删除：delete obj.key
+ 检查属性："key" in obj
+ 获取描述符：Object.getOwnPropertyDescriptor
+ 冻结对象：Object.freeze
+ 密封对象：Object.seal
+ 获取原型：Object.getPrototypeOf
+ 设置原型：Object.setPrototypeOf
+ 获取属性名：Object.keys / Object.getOwnPropertyNames
+ 获取符号属性：Object.getOwnPropertySymbols

+ [[GET]]
  该操作用于在对象的一个属性上，并获取一个值，一个 slot 对外，，然后再用来获取对象的 slot 上的值。

+ [[SET]]
  该操作作用于对象 slot 上的值，如果该 slot 上没有属性定义则施加定义，如果有定义则修改 value 值。

+ [[DELETE]]
  该操作作用于

+ [[Define Property]]

+ [[ENUMERATE]]
  该操作代表了一个属性是否能够被枚举，

