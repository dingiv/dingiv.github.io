# 属性的可枚举性和所有者
![alt text](own-p.png)

## 自有属性——Own properties
hasOwnProperty、Object.getOwnPropertyNames……
凡是强调 **Own** 指的是在对象本身上定义的属性，而不是从原型链上继承而来的。

### ... 展开操作符——浅拷贝语义
该操作符的目的是浅拷贝。

它主要分成两个场景，一个是在方括号 `[]` 里面展开，一个是在花括号 `{}` 里面展开；前者遵从可迭代协议，使用对象的迭代器展开一个可迭代对象。但是，它还实现了一些扩展能力，它可以展开一个普通对象，其展开逻辑是可以认为是浅拷贝
```js
const arr = [1, 2]
const arr2 = [2, 4]

//
const newArr = [...arr, ...arr2]

const obj = {
  name: 'zs'
}

const zs = {
  ...obj,
  age: 18
}
```

## key 类型——name、symbol、#
对象的 key 可以有三种属性，key 类型为字符串的属性，key 类型为 symbol 类型的，而完全私有的 `#` 开头的属性，只有谷歌浏览器的控制台能够访问，其他的方法访问不了 :dog:
```js
Object.getOwnPropertyNames
Object.getOwnPropertySymbols
Object.getOwnPropertyDescriptors
```

## 可枚举属性——Enumerable
可枚举属性是指那些内部“可枚举”标志设置为 true 的属性，而平常我们在使用一些内置方法的时候，会默认将对象设置一些初始值
+ 对于通过直接的赋值和属性初始化的属性，该标识值默认为即为 true；
+ 对于通过 `Object.defineProperty` 等定义的属性，该标识值默认为 false；
+ 对于原型链上的方法默认为 false，而访问器默认为 true；这是由于使用 class 语法定义的类，其原型链上的方法默认是不可枚举的，该行为是根据 ES 规范决定的；
+ Symbol 类型的 key 值的属性，默认为 false；

### for..in 循环
对象的可枚举属性且为 string key，可以通过 for...in 循环进行遍历，该遍历包括对象自身的和继承的可枚举属性。

## Descriptor
用于描述一个对象的某个 key 值所代表的属性的配置对象。
+ value。该值表示该 key 值的属性的值。
+ get。该函数使用一个空的参数列表，以便有权对值执行访问时，获取属性值。参见 getter。可能是 undefined。
+ set。使用包含分配值的参数调用的函数。每当尝试更改指定属性时执行。参见 setter。可能是 undefined。
+ enumerable。一个布尔值，表示是否可以通过 for...in 循环来枚举属性。另请参阅枚举性和属性所有权，以了解枚举属性如何与其他函数和语法交互。
+ configurable

## 对象的私有属性实现方法
1. 使用 `_` 开头的属性名，表示该属性是私有的，但是这种方式并不是真正的私有属性，只是约定俗成的，几乎没有限制，完全依赖于开发人员的意识，并且在使用 `for...in` 时，`_` 开头的属性也会被遍历出来，这种行为很可能是我们不希望的。因此，**不推荐使用**。
2. 使用 `Symbol` 类型作为属性名，因为 `Symbol` 类型的属性名不会被 `for...in` 遍历出来，并且 `Object.getOwnPropertyNames` 也无法获取到 `Symbol` 类型的属性名，但是 `Object.getOwnPropertySymbols` 可以获取到 `Symbol` 类型的属性名。比较**推荐使用**。
3. 使用 `WeakMap` 类型实现，将一个类和一个 `WeakMap` 实例关联起来，将类的实例作为 `WeakMap` 的 key，将类的私有属性作为 `WeakMap` 的 value，这样就可以实现类的私有属性，除了能够获得该 `WeakMap`，否则完全无法获取相应的属性，但是在实现上略微复杂，**推荐使用**。
4. 使用 `#` 开头的属性名，这是 ES6 中新增的语法，表示该属性是私有的，并且该属性无法被 `for...in` 遍历出来，也无法被 `Object.getOwnPropertyNames` 获取到，但是 `Object.getOwnPropertyDescriptors` 可以获取到 `#` 开头的属性，但是即使 key 被获取到也无法在 class 外部访问。该种方式在和 Proxy、Reflect 一起使用时，非常容易发生错误，因此，**不推荐使用**。
