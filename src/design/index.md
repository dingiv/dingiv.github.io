---
title: 软件设计
order: 20
---

# 软件设计
软件设计的基础意义是根据实际的业务需求编写软件进行实现。但是，在实际的环境中，事情往往没有想象中的那么简单，我们需要额外面临更多考验。

+ 可行性问题，这是最基本的要求，要求软件的实现能够切实解决需求；
+ 成本问题，要求使用尽量低的时间成本、金钱成本；
+ 可维护问题，软件需求持续服役，新增功能、修复 bug、回滚修改等；
+ 质量问题，拥有较好的性能、健壮性、可用性、稳定性；

## 复杂软件
一些运行在用户态的软件承担着较为底层的基础工作，并且需要具有极高的运行性能，是软件设计的标杆。编写这些软件需要深刻理解领域内的专业知识和良好的工程设计，学习它们有助于提高个人的软件设计能力，例如：
+ 模拟器，用于模拟和虚拟化硬件平台，从而在一个操作系统上运行另一个独立的操作系统。QEMU、VirtualBox、VMWare...
+ 编译器/运行时，用于将一个语言的源码文件编译成另一种语言并可以在不同的平台上进行执行，GCC、MSVC、汇编器、V8、JVM...
+ 浏览器，用于实现 Web 标准和渲染 Web 程序，Chrome、FireFox、Safari...
+ 数据库，用于高效管理磁盘数据，实现 sql 标准，并对外提供数据服务，提供数据并发处理和事务功能，MySQL、Redis...
+ 游戏引擎，用于提供高效的游戏渲染解决方案和一站式游戏开发能力，Unity、Unreal、Godot...
+ AI 基架，用于训练和驱动 AI 模型，Pytorh...
