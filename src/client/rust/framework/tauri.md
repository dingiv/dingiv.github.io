---
title: Tauri
order: 65
---

# Tauri
Tauri 是 Rust 生态中"Web 渲染"流派的旗舰框架。它用 Rust 编译一个极小的原生壳（约 3-6 MB），内嵌操作系统的 WebView 组件（Windows 用 WebView2/Edge、macOS 用 WKWebView、Linux 用 WebKitGTK），前端用任何 Web 技术栈（React/Vue/Svelte/纯 HTML）。Rust 后端和 Web 前端通过 IPC 双向调用——Rust 函数可以暴露为前端 `import`，前端事件可以触发 Rust 处理。

Tauri 与 Electron 的功能定位完全一致——"用 Web 技术做桌面应用"。但 Electron 捆绑整个 Chromium（~120MB 基线体积），Tauri 复用操作系统自带的 WebView（~3MB 基线体积 + 系统 WebView 组件）。

## 架构
```
┌──────────────────────────────────────┐
│            Web 前端                    │
│  React / Vue / Svelte / HTML+CSS+JS   │
│  (运行在系统 WebView 中)               │
├──────────────────────────────────────┤
│         IPC (invoke / event)          │
├──────────────────────────────────────┤
│            Rust 后端                   │
│  ┌──────────┐  ┌──────────────────┐   │
│  │ Commands │  │   Tauri Core     │   │
│  │ (暴露给  │  │ (窗口管理/菜单/   │   │
│  │  前端)   │  │  托盘/更新/存储)  │   │
│  └──────────┘  └──────────────────┘   │
└──────────────────────────────────────┘
```

前端调用 Rust 通过 `#[tauri::command]` 标注的函数——Rust 侧写纯 Rust 函数，前端用 `invoke('fn_name', { args })` 异步调用，参数自动序列化/反序列化。Rust 向前端推送事件通过 `app.emit("event-name", payload)`。

```rust
// Rust 后端: src-tauri/src/main.rs
#[tauri::command]
fn read_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![read_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

```javascript
// Web 前端: 调用 Rust 函数
import { invoke } from '@tauri-apps/api/core';

const content = await invoke('read_file', { path: '/path/to/file.txt' });
console.log(content);
```

Tauri 的内置能力包括窗口管理（多窗口、无边框、透明背景）、系统托盘、全局快捷键、自动更新（支持增量更新和签名验证）、文件系统访问（路径隔离和权限控制）、原生菜单栏和右键菜单。这些是 Electron 需要额外安装包才能覆盖的功能。

## IPC 通信机制
Tauri 的前后端通信是框架最核心的设计——它决定了前端 JS 如何调用 Rust 函数、参数如何传递、结果如何返回、错误如何在两个语言运行时之间传播。

### invoke：前端调用后端
前端调用 `invoke('command_name', { arg1: val1, arg2: val2 })` 时发生以下流程：

```
JS: invoke('read_config', { path: '/etc/app.json' })
    │
    ▼  JSON 序列化参数 { "path": "/etc/app.json" }
    │
    ▼  WebView IPC 桥 —— 通过系统 WebView 的 postMessage 机制传到 Rust 侧
    │   (Windows: WebView2 host object, macOS: WKWebView message handler)
    │
    ▼  Rust 侧反序列化 JSON → 匹配到 #[tauri::command] fn read_config(path: String)
    │   调用函数，获取 Result<String, String>
    │
    ▼  返回值序列化为 JSON → 通过 IPC 桥传回前端
    │
    ▼  JS: await invoke(...) 解析为 Promise<...>
```

两端各做一次 JSON 序列化/反序列化。前端传入的 JS 对象被 `JSON.stringify` → Rust 侧 `serde_json::from_value` 还原为 Rust 类型 → 函数执行 → 返回值 `serde_json::to_value` → 前端 `JSON.parse` 得到 JS 对象。整个过程对开发者透明——JS 侧像是调用了一个异步函数，Rust 侧像是被本地调用。

### `#[tauri::command]` 的类型契约
`#[tauri::command]` 宏在编译时生成胶水代码，对函数的参数和返回值有严格的类型约束：

所有参数必须实现 `serde::Deserialize`。所有返回值必须实现 `serde::Serialize`。错误类型需要实现 `serde::Serialize`（Rust 侧的 `Result::Err` 会被序列化后作为 JS Error 抛出）。支持的类型包括标准 Rust 类型（`String`, `i32`, `bool`, `Vec<T>`, `HashMap<K,V>`, `Option<T>`）和自定义 struct/enum（需派生 `Serialize`/`Deserialize`）。

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Config {
    theme: String,
    font_size: u32,
    plugins: Vec<String>,
}

#[tauri::command]
fn save_config(config: Config) -> Result<String, String> {
    // JS 传入 { theme: "dark", font_size: 14, plugins: ["git"] }
    // Rust 侧自动反序列化为 Config struct
    let json = serde_json::to_string(&config).map_err(|e| e.to_string())?;
    std::fs::write("config.json", &json).map_err(|e| e.to_string())?;
    Ok("Config saved".to_string())
}
```

前端调用完全不知道 `Config` struct 的存在——它只看到 JS 对象：

```javascript
const result = await invoke('save_config', {
    config: { theme: 'dark', font_size: 14, plugins: ['git'] }
});
console.log(result); // "Config saved"
```

### 错误传播：Rust Error → JS Exception
Rust 函数的 `Result::Err` 会转换为 JS 的异常。Tauri 将 `Err` 中的值序列化为 JSON 字符串，作为 JS Error 的 message：

```rust
#[tauri::command]
fn read_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| e.to_string())
}
```

```javascript
try {
    const content = await invoke('read_file', { path: '/nonexistent.txt' });
} catch (error) {
    console.log(error); // "No such file or directory (os error 2)"
}
```

错误传播的关键在于 Rust 侧的 `Err` 类型必须实现 `Serialize`。如果 `Err` 类型不可序列化（如 `Box<dyn Error>`），Tauri 会在编译时报错——`#[tauri::command]` 的代码生成强制检查这一点。实践中通常用 `String` 作为错误类型（简单场景）或自定义错误 enum（复杂场景，前端可根据错误变体做分支处理）：

```rust
#[derive(Serialize)]
enum AppError {
    NotFound(String),
    PermissionDenied,
    NetworkError { code: u16, message: String },
}

#[tauri::command]
fn read_file(path: String) -> Result<String, AppError> {
    if !path.starts_with("/allowed/") {
        return Err(AppError::PermissionDenied);
    }
    std::fs::read_to_string(&path).map_err(|_| AppError::NotFound(path))
}
```

```javascript
try {
    await invoke('read_file', { path: '/forbidden/file.txt' });
} catch (error) {
    const parsed = JSON.parse(error);  // 尝试解析序列化后的错误
    // { "PermissionDenied": null }
}
```

### 事件：后端推送前端
`invoke` 是前端拉取（pull）——JS 主动调用 Rust。后端推送（push）——Rust 主动向 JS 发送数据——通过事件（Event）机制实现：

```rust
use tauri::Emitter;

#[tauri::command]
fn start_download(app: tauri::AppHandle, url: String) {
    std::thread::spawn(move || {
        // 每下载 1MB 推送一次进度事件
        for progress in 0..=100 {
            let _ = app.emit("download-progress", progress);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let _ = app.emit("download-complete", "Done");
    });
}
```

```javascript
import { listen } from '@tauri-apps/api/event';

// 监听后端推送的事件
const unlisten = await listen('download-progress', (event) => {
    console.log(`Progress: ${event.payload}%`);
});

await listen('download-complete', (event) => {
    console.log(event.payload); // "Done"
    unlisten(); // 取消监听
});
```

事件的 payload 同样经过 serde 序列化/反序列化——Rust 侧 `emit("event", value)` 中的 value 必须实现 `Serialize`，JS 侧的 `event.payload` 已经是反序列化后的 JS 对象。

### 序列化性能与二进制数据
默认 JSON 序列化对小数据量（KB 级别）足够——参数和返回值在 IPC 调用的总耗时中占比极小（通常 < 5%），主要延迟来自系统 WebView 的 IPC 桥本身（~0.1-1ms）。但当传输大数据时（图片、大文件内容、大量数据库行），序列化成为瓶颈。

Tauri 针对大数据提供了几种优化路径：直接返回文件路径而非内容（前端用 WebView 原生能力读文件）；使用 Tauri 的 `tauri::api::file` 或 `tauri::api::fs` 模块读写文件（绕过 IPC，走 TAURI 的内置文件协议）；流式传输（通过事件分块推送大数据，避免单次 IPC 传输超大数据包）。

Rust 侧对大型二进制数据的推荐方式是使用 `Vec<u8>` + serde bytes 优化——默认情况下 `Vec<u8>` 序列化为 `[0, 1, 2, ...]`（每个 byte 一个 JSON 数字，膨胀 4-6×），开启 serde 的 `bytes` feature 后直接序列化为 base64 字符串或通过 Binary 通道传输。

### 与 Electron IPC 的对比
| 维度 | Tauri | Electron |
|------|-------|----------|
| 调用方向 | `invoke` → Rust 函数, `emit` → JS 事件 | `ipcRenderer.invoke` → `ipcMain.handle` |
| 序列化 | serde（编译时类型约束） | JSON（运行时，无类型保证） |
| 参数类型 | 所有参数编译时检查 Serialize/Deserialize | `any`——运行时爆炸 |
| 错误传播 | `Result::Err` 经 Serialize 序列化为 JS Error | 手动 try/catch，错误格式自定 |
| 多窗口 | `app.emit_to(target, ...)` 精确投递 | `webContents.send` |
| 性能 | serde JSON 序列化（小数据 < 1ms） | V8 JSON 序列化（小数据 < 0.5ms） |

核心差异：Tauri 的类型契约在编译时强制执行——参数类型不匹配是编译错误而非运行时 bug。Electron 的 IPC 完全无类型——`ipcMain.handle('name', (event, ...args) => ...)` 的 `args` 是 `any[]`，参数数量不对或类型错误只有在函数内部手动检查才能发现。对于大型团队或多人协作项目，Tauri 的编译时 IPC 类型安全显著减少运行时错误。

## 与 Electron 的差异
| 维度           | Tauri                 | Electron          |
| -------------- | --------------------- | ----------------- |
| 基线二进制体积 | 3-6 MB                | ~120 MB           |
| 内存基线       | ~40-80 MB             | ~150-300 MB       |
| Web 引擎       | 系统 WebView          | 捆绑 Chromium     |
| 后端语言       | Rust                  | Node.js           |
| 兼容性         | 系统 WebView 版本差异 | Chromium 版本统一 |
| 移动端         | 实验性支持            | 无官方支持        |

Tauri 的核心优势是二进制体积和内存占用——系统 WebView 已经常驻内存（Windows 的 Edge WebView2 被系统和其他应用共享），Tauri 自身只增加了原生壳的体积。核心劣势是系统 WebView 的版本差异——Windows 7 不支持 WebView2（需额外安装）、Linux 的 WebKitGTK 版本在不同发行版上行为可能不一致。如果目标用户群覆盖旧系统（Windows 7/8）或小众 Linux 发行版，WebView 兼容性需要额外测试。

## 适用边界
Tauri 最适合拥有 Web 前端团队的场景——现有的 React/Vue 应用可以直接套进 Tauri 壳变成桌面应用，业务逻辑可以同时跑在 Web 和桌面端。Rust 后端处理性能敏感操作（文件 IO、加密、图像处理），前端处理 UI 渲染——这是最自然的分工。

不适合的场景：需要原生外观和极致性能的桌面工具（用 iced）、嵌入式或 MCU 上的 GUI（用 Slint）、不希望依赖系统 WebView 的环境（Electron 捆绑 Chromium 虽重但行为统一）、需要大量复杂桌面级 UI 组件（Web 生态虽强但某些桌面控件——如虚拟文件树、多窗口拖拽——在 WebView 中的体验不如原生）。
