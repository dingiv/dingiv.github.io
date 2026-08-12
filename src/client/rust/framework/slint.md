---
title: Slint
order: 60
---

# Slint
Slint 是 Rust 生态中唯一将"嵌入式设备"作为一等目标的 UI 框架。它能在 Linux MCU（如 STM32、ESP32）、嵌入式 Linux（Yocto/Buildroot）和桌面端运行——同一个 `.slint` 声明式 UI 文件在不同平台上渲染效果一致。Slint 的渲染引擎完全自研，不依赖 Web 引擎、不依赖 Qt/GTK，在资源极度受限的设备上也能保持 60 fps。

## 声明式 UI 语言
Slint 使用自己的 `.slint` DSL 描述 UI，语法接近声明式 JSON/YAML 但加入了响应式绑定（类似 QML）。UI 和逻辑分离——`.slint` 文件定义界面结构和布局，Rust（或 C++/JS）代码处理业务逻辑。

```txt
// ui.slint
export component MainWindow {
    width: 400px;
    height: 300px;
    title: "Counter";

    // 声明属性——Rust 端可以访问和修改
    in-out property<int> counter: 0;

    VerticalLayout {
        padding: 20px;
        spacing: 10px;
        alignment: center;

        Text {
            text: "Count: " + counter;
            font-size: 24px;
        }
        HorizontalLayout {
            spacing: 10px;
            alignment: center;
            Button {
                text: "+";
                clicked => { counter += 1; }
            }
            Button {
                text: "-";
                clicked => { counter -= 1; }
            }
        }
    }
}
```

```rust
// main.rs
slint::include_modules!();  // 编译时内联 .slint 文件

fn main() -> Result<(), slint::PlatformError> {
    let ui = MainWindow::new()?;
    let ui_weak = ui.as_weak();

    // 从 Rust 侧响应 UI 状态变更
    ui.on_counter_changed(move |val| {
        println!("Counter changed to: {}", val);
    });

    ui.run()
}
```

`.slint` 的核心特性：内建布局系统（VerticalLayout/HorizontalLayout/GridLayout），支持响应式绑定（`counter` 变更自动更新绑定它的所有 `Text`），回调信号（`clicked =>` 直接内联处理），动画和过渡效果（`animate` 关键字，`duration` 指定时长）。编译时 `slint!` 宏将 `.slint` 文件编译为原生 Rust 代码——不是运行时解释，没有脚本引擎开销。

## 自研渲染引擎
Slint 不依赖任何第三方渲染框架。核心渲染流程：

- **软件渲染器**：纯 CPU 绘制，通过 `embedded-graphics` 和 `tiny-skia` 实现。支持 `no_std` 环境，不需要操作系统，可在 RTOS 或 bare-metal 上运行。渲染输出通过 `FrameBuffer` trait 抽象——适配任意显示驱动（SPI 屏、LVDS、HDMI 帧缓冲）。
- **Femtovg 渲染器**：基于 GPU 的矢量图形，使用 OpenGL ES 2.0 或 Metal。目标平台是嵌入式 Linux 和桌面——Femtovg 是极小体积的 Canvas 2D 式渲染库（约 100KB），渲染质量接近 Skia 但体积小一个数量级。
- **Qt 渲染器**：作为 fallback，在需要 Qt 兼容性（如集成到已有 Qt 应用的 KDE 桌面）的场景使用。

软件渲染器使 Slint 能够在 MCU 上运行 GUI——这个能力在 Rust UI 生态中只有 Slint 具备。典型场景：STM32H7 + SPI 触摸屏 + FreeRTOS，运行一个温控器控制面板，Slint 软件渲染输出到帧缓冲驱动。

## 平台跨度
桌面（Windows/macOS/Linux）、嵌入式 Linux（Yocto/Buildroot, 基于 Femtovg 或软件渲染）、MCU（ARM Cortex-M, 基于软件渲染, 需要至少 32KB RAM 和 256KB Flash）、Android（实验性）。Web 端暂不支持。

## 与 iced 的对比
| 维度 | Slint | iced |
|------|-------|------|
| UI 描述 | `.slint` 声明式 DSL | Rust 代码（Elm 架构） |
| 渲染 | 自研（软件/Femtovg/Qt） | wgpu/glow（GPU） |
| 嵌入式/MCU | 是（软件渲染 + no_std） | 否 |
| 组件库 | 中等（表格、列表视图等内建） | 基础（需自定义 Canvas 绘制） |
| 二进制体积 | MCU: < 500KB, 桌面: ~3MB | 桌面: ~3-8MB |
| 学习曲线 | 需学 `.slint` 语言 | 纯 Rust，Elm 架构概念 |

Slint 的核心价值在嵌入式端——没有其他 Rust UI 框架能在 STM32 上跑 GUI。如果你的目标平台是桌面且不需要嵌入式兼容，iced 的"全 Rust 代码"体验更流畅。如果你的产品线同时覆盖嵌入式显示面板和桌面配置工具，同一套 `.slint` 文件跨平台复用是 Slint 的独特优势。
