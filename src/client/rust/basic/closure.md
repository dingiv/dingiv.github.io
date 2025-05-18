# 闭包
闭包是 Rust 中的匿名函数类型，是支持函数式编程的基础特性。

闭包是一个动态生成的对象、匿名函数，它实现了 Fn trait，可以像函数一样进行调用，并且最重要地，闭包可以动态地捕获环境中的变量，从而形成一个具有隐式依赖的脏函数。

## 闭包的使用
```rust
let cl = |x| {
  println!("{}", 12);
};
```

## Fn trait
有三个 fn triat：
+ FnOnce：实现 `call_once` 方法，表示闭包可以被调用一次，调用后闭包可能会消耗捕获的变量（通过移动所有权）；
  ```rust
  pub trait FnOnce<Args> {
    type Output;
    fn call_once(self, args: Args) -> Self::Output;
  }
  ```
+ FnMut：继承 FnOnce，并额外实现 `call_mut` 方法，表示闭包可以被调用多次，并且可以修改捕获的变量；
  ```rust
  pub trait FnMut<Args>: FnOnce<Args> {
    fn call_mut(&mut self, args: Args) -> Self::Output;
  }
  ```
+ Fn：继承 FnMut，并额外实现 `call` 方法，表示闭包可以被调用多次，但只读访问捕获的变量；
  ```rust
  pub trait Fn<Args>: FnMut<Args> {
    fn call(&self, args: Args) -> Self::Output;
  }
  ```

自动声明语法，编译可以自动为闭包类型声明定义，使用 `|x| {}` 语法进行声明；默认情况下，声明的类型是 Fn。

## 区别于普通函数
