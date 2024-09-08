# 属性的可枚举性和所有权

![alt text](own-p.png)

## Own
hasOwnProperty、Object.getOwnPropertyNames……
凡是强调**Own**指的是在对象本身上定义的属性，而不是从原型链上继承而来的。

## name、symbol、#
对象有三种属性，key类型为字符串的属性，key类型为symbol类型的，而完全私有的`#`开头的属性，只有谷歌浏览器的控制台能够访问，其他的方法访问不了 :dog:
```js
Object.getOwnPropertyNames
Object.getOwnPropertySymbols
Object.getOwnPropertyDescriptors
```

## Enumerable
可枚举属性是指那些内部“可枚举”标志设置为true的属性，而平常我们在使用一些内置方法的时候，会默认将对象设置一些初始值，
+ 对于通过直接的赋值和属性初始化的属性，该标识值默认为即为 true；
+ 对于通过 Object.defineProperty 等定义的属性，该标识值默认为false；
+ 对于原型链上的方法默认为false；
+ Symbol类型的key值的属性，默认为false；
- **可枚举的属性并且不是Symbol类型的key可以通过for...in循环进行遍历**


## Descriptor
用于描述一个JavaScript对象的某个key值所代表的属性的配置对象。
+ value。该值表示该key值的属性的值。
+ get。该函数使用一个空的参数列表，以便有权对值执行访问时，获取属性值。参见 getter。可能是 undefined。
+ set。使用包含分配值的参数调用的函数。每当尝试更改指定属性时执行。参见 setter。可能是 undefined。
+ enumerable。一个布尔值，表示是否可以通过 for...in 循环来枚举属性。另请参阅枚举性和属性所有权，以了解枚举属性如何与其他函数和语法交互。
+ configurable