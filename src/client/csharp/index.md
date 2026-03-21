---
title: C#
order: 50
---

# C#

C# 是由微软在 2000 年发布的现代编程语言，由 Anders Hejlsberg 领导设计，他是 Turbo Pascal 和 Delphi 的首席架构师。C# 的设计目标是成为一种简单、现代、通用的面向对象编程语言，同时兼顾 C++ 的强大功能和 Visual Basic 的易用性。

## 语言特性

C# 运行在 .NET 运行时之上，编译为中间语言 (IL) 由 JIT 编译器转换为本地代码。C# 结合了 C++ 的性能和 Java 的易用性，引入了许多现代语言特性，如垃圾回收、泛型、LINQ、async/await 等。

C# 的类型系统包括值类型和引用类型。值类型直接存储数据，分配在栈上或内联在对象中；引用类型存储对象的引用，分配在托管堆上。垃圾回收器自动管理内存，开发者无需手动释放。

C# 支持多种编程范式，包括面向对象、函数式、泛型、反射等。现代 C# 版本持续引入新特性，如模式匹配、记录类型、可空引用类型等，保持语言的现代感。

## 应用领域

C# 在多个领域有广泛应用。Web 开发方面，ASP.NET Core 是高性能的 Web 框架，支持 MVC、Web API、Blazor 等开发模式。桌面应用方面，WinForms、WPF、WinUI 3 提供了丰富的桌面应用开发能力。游戏开发方面，Unity 引擎使用 C# 作为脚本语言，是独立游戏和手游开发的首选。云计算方面，Azure 是微软的云平台，.NET 应用可以无缝部署到 Azure。

## 技术文章

- [基础语法](./basic.md) - 类型系统、变量、方法、泛型、命名空间
- [异步编程](./async.md) - async/await、Task、ValueTask、异步流、取消操作
- [NuGet 包管理](./nuget.md) - 包结构、dotnet CLI、版本范围、私有源、包发布
- [.NET 运行时](./dotnet.md) - JIT/AOT 编译、GC、IDisposable、线程池、异常处理
- [MSBuild](./msbuild.md) - 项目文件、目标任务、构建配置、自定义任务
- [Unity 脚本](./unity.md) - MonoBehaviour、协程、组件系统、物理、预制体
