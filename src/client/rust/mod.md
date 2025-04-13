# 模块系统和包管理
rust 的模块系统有三个主要抽象，package, crate, mod。理解模块是建构大型应用和复用社区生态的前提。

## package
一个 rust package/project 通常是一个文件夹，内部包含一个 Cargo.toml，它是 Cargo 包管理的基本单位。一个 package 通常至少含有一个 crate。

## crate
一个二进制的构建单元，在编译之后，就是一个二进制的可执行文件。crate 有两种，一种是独立执行二进制，其代码入口通常是 `<package>/src/main.rs` 文件；另一种是链接库二进制文件，其代码入口是 `<package>/src/lib.rs`。往往，我们的实际工程程序入口就是 main.rs，如果我们要发布链接库，那么入口就是 lib.rs。

## mod
一个 mod 模块是 Rust 代码的一个逻辑单元，可以用来组织和封装功能。Rust 模块可以在 crate 内部进行嵌套和组织。

程序的编译入口是 src/main.rs 文件，这是程序的根模块。你可以在这个文件中使用 `mod mod_name {}` 关键字进行模块声明，模块定义支持嵌套。

### 模块拆分
当代码量变大的时候，我们需要将模块的代码拆分到不同的文件中进行保存。如何拆分？在一个模块文件中，可以将 `mod mod_name {}` 中的实现部分，拆分到**与该文件同目录**中的 mod_name.rs 文件中，然后留下 `mod mod_name;`，此语句表示，模块的实现被拆分到了同目录下的 `mod_name.rs` 文件中了，编译器将自行寻找它。

嵌套拆分，如果 mod_name.rs 中的代码还是多，那么还可以继续拆分。现在，将该文件变成 `mod_name/mod.rs` 文件，然后，将 mod.rs 文件中的子模块mod_a, mod_b 使用同样的方式，拆到同目录 mod_name 中的 mod_a.rs 和 mod_b.rs 文件中。

> mod_name/mod.rs 文件类似于 JavaScript 当中的 mod_name/index.js 文件，不过，JavaScript 当中的模块不存在和目录结构捆绑的父子关系，而 rust 中，还有其他的语言，如 Java、Python、Golang 等，有这样的关系。而像 JavaScript、C 语言、C++ 就没有模块和目录的捆绑。

### 访问控制和模块导入
使用 `pub` 关键字控制内容对于其所属模块外部代码的可见性。对于一个模块内部的代码，其默认对外不可见，但是对于该模块内部的同级内容可见。

如果是子模块 a，那么，哪怕模块 a 本身被 pub 修饰，子模块内部的代码页不会被兄弟模块 b 可见，兄弟模块 b 只是知道有这么个兄弟 a，而不知道 a 的内部有什么。

使用 `use` 关键字进行模块导入
+ `use foo;` 导入子模块 foo；
+ `use foo::bar;` 导入子模块 foo 中的 bar，具体是什么不知道；
+ `use foo::{bar1,bar2};` 导入子模块 foo 中的 bar1 和 bar2，具体是什么不知道；
+ `use foo::*;` 导入子模块 foo 中的所有东西，具体是什么不知道；
+ `use super::foo;` 导入兄弟模块 foo，即同目录下的 foo.rs 文件或者 foo/mod.rs 文件，或者是同文件中的 mod foo 声明；
+ `use crate::foo;` 导入 src 目录下的文件，即 src/foo.rs 或者 src/foo/mod.rs 文件，或者 main.rs 中 mod foo 声明；
