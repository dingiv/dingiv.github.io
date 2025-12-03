# 特性

## 内置 trait
内置的 trait 是与编译器实现交互的一个接口，为 rust 项目提供规范，同时也能为开发者带来约定俗成的语法糖。
1. Clone / Copy
   Clone：允许类型创建自己的副本。Copy：允许类型通过简单的按位复制进行复制，适用于像整数这样的简单类型。

   ```rust
   #[derive(Clone, Copy)]
   struct MyStruct;

   let a = MyStruct;
   let b = a; // `b` 是 `a` 的副本
   ```

2. Debug / Display
   Debug：允许类型使用 `{:?}` 占位符快速打印复杂类型，适用于调试输出。Display：允许类型使用 `{}` 打印出来，适用于用户友好的输出。

3. Eq / PartialEq
   Eq：用于类型的值相等的比较，如果一个类型要实现 Eq，必须先实现 PartialEq，这是因为 Eq 比 PartialEq 的要求更加严格。PartialEq：用于类型的值相等的比较，可以定义哪些值是相等的。Partial 纯纯是为给 IEEE 标准定的 NaN 这个东西擦屁股，因为标准规定了浮点数的计算错误不会引发不可恢复的错误，而是返回一个 NaN。因此浮点数只是实现了 PartialEq 而不是 Eq，因为
   ```rust
   let a = 0.0 /0.0; // NaN
   let b = (-1.0_f64).sqrt() // NaN
   ```
   一般的，我们实现 Eq 即可，Partial 也会默认实现。除非比较涉及浮点数的比较，且浮点数的这个 NaN 和 NaN 比较的场景不可被忽略，那么

   ```rust
   #[derive(PartialEq, Eq)]
   struct MyStruct {
       value: i32,
   }

   let a = MyStruct { value: 5 };
   let b = MyStruct { value: 5 };
   assert_eq!(a, b); // 使用 `PartialEq` 和 `Eq`
   ```

4. Ord / PartialOrd
   Ord：用于实现类型的排序，如果一个值实现了 Ord 相当于隐式实现了 Eq 和 PartialOrd。PartialOrd：用于类型的排序，但支持不完全比较（例如 NaN）。因此，在平时的使用中，只需要实现了 Ord 即可。

   ```rust
   #[derive(PartialOrd, Ord)]
   struct MyStruct {
       value: i32,
   }

   let a = MyStruct { value: 5 };
   let b = MyStruct { value: 10 };
   assert!(a < b); // 使用 `PartialOrd` 和 `Ord`
   ```

5. From / Into / FromStr
   From：用于定义类型之间的转换；
   Into：与 From 相关，允许进行类型转换。

   类型转化是在两个类型之间调用一个类型转化的函数进行值映射，A -> B。为 B 类型实现泛型 trait，`From<A>`，就可以让在需要一个 B 的地方填入 A 类型，编译器将自动调用类型转化的函数。同时，A 类型的身上也会隐式实现一个 `into<B>()` 方法。

   ```rust
   let a: i32 = 10;
   let b: f64 = a.into(); // `i32` 转换为 `f64`
   ```

6. Drop
   Drop：允许你在类型销毁时执行清理工作。通常用于关闭文件、释放内存等。

7. Default
   Default：提供一个默认值，常用于结构体或枚举。

   ```rust
   #[derive(Default)]
   struct MyStruct {
       value: i32,
   }

   let x: MyStruct = Default::default(); // `value` 默认为 `0`
   ```

8. Iterator 和 IntoIterator
   Iterator：定义一个可以迭代的类型，通常用于实现 next() 方法。
   IntoIterator：用于将集合类型转换为迭代器。

   一般地，我们可以为自己的类型实现一个 IntoIterator，从而让其变成一个迭代器，该方法会尝试获取结构体的所有权。
   ```rust
   let v = vec![1, 2, 3];
   let mut iter = v.into_iter();
   assert_eq!(iter.next(), Some(1)); // 使用 `IntoIterator` 和 `Iterator`
   ```

9. Fn、FnMut 和 FnOnce
   Fn：用于表示可以调用的函数，要求不修改闭包捕获的变量。
   FnMut：表示可以调用并且可以修改捕获变量的函数。
   FnOnce：表示只能调用一次的函数，通常用于闭包。

   ```rust
   let closure = |x| x + 1;
   assert_eq!(closure(2), 3); // 使用 `Fn` trait
   ```

10. AsRef 和 AsMut
    AsRef：用于将类型转换为某种引用类型。

    AsMut：用于将类型转换为可变引用类型。

    ```rust
    let s = String::from("Hello");
    let s_ref: &str = s.as_ref(); // `AsRef` 将 `String` 转换为 `&str`

    let mut s = String::from("Hello");
    let s_mut: &mut str = s.as_mut(); // `AsMut` 将 `String` 转换为 `&mut str`
    ```

11. Hash 和 Eq
    Hash：用于定义如何计算类型的哈希值，通常与 Eq 一起使用。

    ```rust
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert("Hello");
    set.insert("World");
    ```

12. Sized
    Sized：所有已知大小的类型默认都实现此 trait。几乎所有类型都隐式地实现了 Sized，除非它是动态大小类型（DST），如 str、dyn Trait。

13. Unsize 和 CoerceUnsized
    Unsize：用于允许类型变得“不完全”或变为动态大小类型（DST）。
    CoerceUnsized：允许自动将类型转换为动态大小类型。

14. Add +
15. Sub - 减法
16. Mul * 乘法
17. Div /
18. Rem %
19. Neg - 取负值
20. Deref / DerefMut * 解引用
21. Index / IndexMut `[n]` 下标访问
22. \[Xxx\]Assign
23. ...


## trait bound 自约束


## 泛型实现
为任意类型实现一个trait

