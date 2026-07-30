---
title: iced
order: 55
---

# iced
iced 是 Rust 生态中最纯粹的"原生渲染 + Elm 架构"UI 框架。它不依赖任何 Web 引擎——通过 wgpu（Vulkan/Metal/DX12）或 OpenGL 直接调用 GPU 绘制，二进制体积极小（几 MB），启动近乎瞬时。iced 的设计哲学受 Elm 语言深刻影响：UI 是状态的纯函数，所有交互通过消息（Message）驱动状态更新，状态变更触发 UI 重绘。

## Elm 架构
iced 采用四层单向数据流：State（应用状态）→ View（状态到 UI 的纯函数）→ Message（用户交互产生的事件）→ Update（消息处理函数，返回新状态）。这个模式消除了 UI 框架中常见的状态同步 bug——永远只有一个状态来源，View 是纯函数，Update 是纯函数引用。

```rust
use iced::widget::{button, column, text};
use iced::{Element, Sandbox, Settings};

struct Counter {
    value: i32,
}

#[derive(Debug, Clone, Copy)]
enum Message {
    Increment,
    Decrement,
}

impl Sandbox for Counter {
    type Message = Message;

    fn new() -> Self { Self { value: 0 } }
    fn title(&self) -> String { String::from("Counter - iced") }

    fn update(&mut self, message: Message) {
        match message {
            Message::Increment => self.value += 1,
            Message::Decrement => self.value -= 1,
        }
    }

    fn view(&self) -> Element<Message> {
        column![
            button("+").on_press(Message::Increment),
            text(self.value).size(50),
            button("-").on_press(Message::Decrement),
        ]
        .padding(20).spacing(20).into()
    }
}
```

`Sandbox` 是最简 trait——适合不需要异步操作的简单应用。需要网络请求、文件 IO 时升级到 `Application` trait，支持 `Command` 返回值用于执行副作用（`Command::perform(async_fn, Message::Loaded)`）。两种 trait 的 View 和 Update 签名完全一致，升级只需改 trait impl。

## Message 机制详解
理解 iced 的 Message 系统是理解 Elm 架构与 React/Vue 式框架本质差异的钥匙。在 React 中，状态分散在组件树各处，每个组件独立管理自己的 `useState`。在 iced 中，整个应用只有一个全局状态（根 struct），所有状态变更都必须通过一个中心化的 `enum Message`。

### Message 的三种来源
**用户交互**——最常见。按钮点击 `on_press(Message::Increment)`、文本输入 `on_input(|s| Message::TextChanged(s))`、滑块拖动 `on_change(|v| Message::VolumeChanged(v))`。Widget 自身不持有状态——它将用户动作转换为 Message 发出，由 Update 函数统一处理。这意味着按钮不知道也不关心"点击后会发生什么"——它只是发出一个信号。

**异步 Command**——Side Effect 的唯一切入点。iced 的函数签名保证 `view()` 和 `update()` 是纯函数——它们不能直接发起网络请求、读写文件、启动定时器。任何副作用都通过 `Command` 表达：`update()` 返回 `Command::perform(async_fn, Message::Loaded)`，Runtime 在后台执行异步函数，完成后将结果包装为 Message 投回 `update()`。

```rust
// Application trait 示例——Command 异步流
impl Application for WeatherApp {
    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::FetchWeather(city) => {
                self.loading = true;
                // 发起异步请求，完成后将结果投回 update() 作为 Message::WeatherLoaded
                Command::perform(fetch_weather(city), Message::WeatherLoaded)
            }
            Message::WeatherLoaded(Ok(data)) => {
                self.weather = Some(data);
                self.loading = false;
                Command::none()
            }
            Message::WeatherLoaded(Err(e)) => {
                self.error = Some(e.to_string());
                self.loading = false;
                Command::none()
            }
        }
    }
}
```

**事件订阅（Subscription）**——外部事件的持续来源。与一次性 Command 不同，Subscription 是持续的——定时器每 1 秒触发、键盘全局快捷键监听、窗口尺寸变化。`fn subscription(&self) -> Subscription<Message>` 返回需要监听的持续事件，Runtime 自动管理订阅的生命周期。

### 从 Message 到 UI 更新：完整链路
```
用户点击按钮
    │
    ▼
Widget 将 Message::Increment 投递到 Runtime 的消息队列
    │
    ▼
Runtime 从队列取出消息，调用 update(&mut state, Message::Increment)
    │
    ▼
update() 修改 state.value += 1，返回 Command::none()
    │
    ▼
Runtime 检测到 state 已变更，调用 view(&state) 生成新的 Element 树
    │
    ▼
差分算法（diff）比较新旧 Element 树，仅更新变化的节点
    │
    ▼
GPU 重新绘制受影响的 Widget
```

关键：`update()` 和 `view()` 被 Runtime 交替调用——先更新状态，后重建视图。开发者不控制调用时机——Runtime 在每一帧中批量处理消息，与浏览器的 requestAnimationFrame 类似。多个消息在同一帧内到达时，Runtime 批量调用 update 后只做一次 view 重建和渲染。

### 与 React useState 的对比
两个框架的表面功能一样——"状态变了，UI 自动更新"。但底层机制完全不同，这决定了代码组织方式、调试体验和重构难易度。

**状态管理模式**：React 的状态分散——每个组件有自己的 `useState`、`useReducer`、context。组件 A 的状态只有 A 知道怎么改。iced 的状态集中——整个应用的 state 在一个中心 struct 中，所有修改通过 `enum Message` 穷举。要理解"应用可能处于哪些状态、怎样从一个状态变到另一个状态"，只需要看 `Message` 枚举和 `update()` 函数。

这类似于 Redux 的 `reducer(state, action)` 模式——但 iced 把它做成了框架级的唯一路径，而不是一个可选的库。

**类型安全**：iced 的 `enum Message` 在编译时保证——`match` 的所有分支必须处理，遗漏任何变体都是编译错误。React 的 `setState(42)` 没有类型级保证——状态类型是泛型参数，但状态变更路径是运行时行为，不会在编译时检查"是否遗漏了某个状态转换的处理"。重构时 iced 更安全：删掉一个 Message 变体，编译器会告诉你所有 `match` 分支需要更新；React 删掉一个 action type，不会有任何编译时警告。

**异步和副作用**：

```rust
// iced: 异步通过 Command——类型明确、链式可追踪
Command::perform(fetch_data(), Message::DataLoaded)

// React: 异步通过 useEffect——闭包捕获、依赖数组手动维护
useEffect(() => {
    fetchData().then(setData);
}, [query]);  // 漏了依赖 → 陈旧的闭包引用 → bug
```

iced 的异步路径清晰——`update()` 返回 `Command`，Runtime 执行，结果作为 Message 回到 `update()`。消息类型固定（`Message::DataLoaded(Result<Data, Error>)`），调用链可追踪。React 的 `useEffect` 依赖数组需要手动维护——遗漏依赖导致闭包捕获陈旧变量、多余的 deps 导致重复执行。这两类 bug 在 iced 中不存在——没有依赖数组的概念，状态只在 `update()` 中修改、`view()` 在每次 `update()` 后被 Runtime 重建。

**重渲染粒度**：React 通过 Virtual DOM diff 决定哪些 DOM 节点需要更新——setState 触发组件及所有子组件重渲染（除非 `React.memo`），但 DOM 操作只作用于变化的节点。iced 同样做差分——但差分对象是 iced 自己的 Element 树（不是 DOM），结果只更新变化的 Widget 的 GPU 绘制指令。两者的开销模式类似，但 iced 的"diff→渲染"全程在 Rust 侧完成，没有 JS→Native 桥接开销。

**代码量**：React 的样板更少——`const [value, setValue] = useState(0)` 一行搞定。iced 需要定义 `enum Message` + `impl update` + `view` 中的 `on_press`。对简单计数器，React 更短。但对复杂表单——有验证状态、加载状态、错误状态、提交状态的表单——React 的 `useState × N` 开始产生"状态散落在组件各处"的问题，iced 的中心化 `enum` + `match` 反而更容易理清所有状态转换路径。

核心取舍：React 胜在写起来快、组件生态丰富；iced 胜在状态转换可穷举、重构安全、异步路径可追踪。选择取决于项目是"快速迭代、功能经常变"还是"状态复杂度高、正确性要求高"。

## 组件化与开发实践
iced 没有 React 意义上的"组件"。没有生命周期钩子、没有 props、没有 `useEffect`。iced 的代码复用方式更接近函数式编程：UI 被组织为返回 `Element<Message>` 的纯函数，状态变更通过 `Message` 枚举的分层和委托实现。

### 无组件 = 只有函数
在 iced 中，"组件"就是一个签名为 `fn(&State, &impl Fn(Message)) -> Element<Message>` 的函数。这个函数接收当前状态和一个消息发送闭包，返回一棵 Element 树。它不是框架的抽象——只是 Rust 的函数。

```rust
// 一个"计数器组件"——实际上是一个 view 函数
fn counter_view(value: i32, on_msg: impl Fn(CounterMsg) -> Message + 'static) -> Element<Message> {
    column![
        button("+").on_press(on_msg(CounterMsg::Increment)),
        text(value).size(30),
        button("-").on_press(on_msg(CounterMsg::Decrement)),
    ]
    .spacing(10)
    .into()
}
```

这个函数的复用性来自参数化——`value` 决定了显示什么，`on_msg` 决定了交互产生什么 Message。调用者控制状态存储位置和消息路由目标——函数本身不持有状态、不决定生命周期。这与 React 组件有本质区别：React 组件持有自己的 state 和 effects，iced 的 view 函数只是一个渲染函数。

### 大型应用的组织：Message 分层与 update 委托
当应用从单个 Counter 扩展到多页面、多面板时，把所有 Message 变体塞进一个 `enum` 会迅速失控。

iced 的标准实践是"嵌套枚举 + 委托 update"：每个功能模块定义自己的 `enum Msg`，根 `Message` 通过一个变体包装子模块的消息，`update()` 中匹配到包装变体时委托给子模块的 `update()`。

```rust
// 子模块: 设置页面
mod settings {
    #[derive(Debug, Clone)]
    pub enum Msg { ThemeChanged(String), FontSizeChanged(u32) }

    pub struct State { pub theme: String, pub font_size: u32 }

    pub fn update(state: &mut State, msg: Msg) -> Command<Msg> {
        match msg {
            Msg::ThemeChanged(t) => { state.theme = t; Command::none() }
            Msg::FontSizeChanged(s) => { state.font_size = s; Command::none() }
        }
    }

    pub fn view(state: &State) -> Element<Msg> {
        column![
            text_input("Theme", &state.theme).on_input(|s| Msg::ThemeChanged(s)),
            slider(8..=32, state.font_size, |s| Msg::FontSizeChanged(s)),
        ].into()
    }
}

// 根模块
#[derive(Debug, Clone)]
enum Message {
    Settings(settings::Msg),  // 包装子模块消息
    Counter(crate::CounterMsg),
}

struct App {
    settings: settings::State,
    counter: CounterState,
}

impl Application for App {
    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::Settings(msg) => {
                // 委托给子模块，同时映射子模块的 Command<Msg> 到根 Command<Message>
                settings::update(&mut self.settings, msg)
                    .map(Message::Settings)
            }
            Message::Counter(msg) => { /* 同样委托 */ }
        }
    }

    fn view(&self) -> Element<Message> {
        column![
            settings::view(&self.settings).map(Message::Settings),
            counter_view(self.counter.value, |m| Message::Counter(m)),
        ].into()
    }
}
```

关键操作在 `.map(Message::Settings)`——子模块的 `view()` 返回 `Element<settings::Msg>`，`.map()` 将子模块的 Message 类型提升为根的 Message 类型。代码复用通过 view 函数实现，业务逻辑隔离通过子模块的 `update()` 实现，Message 路由通过 enum 包装 + `.map()` 实现。三层各自独立。

这个模式与 Redux 的"combineReducers + mapDispatchToProps"理念相同，但 iced 把它做成了框架内置机制而非第三方库。

### 子组件如何向父组件传递数据
iced 没有 React 的"lifting state up"概念——因为状态一开始就集中在根 struct 中，不存在"提升"。子模块向父模块"传递数据"实际上通过 Message 包装实现：

1. 子模块 view 产生 `settings::Msg::ThemeChanged("dark")`
2. view 调用链上的 `.map(Message::Settings)` 将其包装为 `Message::Settings(settings::Msg::ThemeChanged("dark"))`
3. 根的 `update()` 匹配 `Message::Settings(msg)` → 委托给 `settings::update(&mut self.settings, msg)`
4. `settings::update()` 修改 `self.settings.theme = "dark"`——这个状态本来就存储在根 struct 中

子模块不"拥有"状态——它借用了根 struct 中的 `settings::State`。子模块不"传递"数据给父——它只是产生一个 Message，而这个 Message 最终导致存储在同一位置的状态被修改。所有权从未转移。

### 可复用 Widget 的开发模式
iced 的可复用 widget 分三个层级：

**纯 view 函数**（无需状态，只需参数）：导航栏、分隔线、卡片容器。函数签名 `fn navbar(items: &[NavItem]) -> Element<Message>`，调用方传数据，返回 Element。

**带状态的 view + Message 组合**（需要状态但状态由调用方管理）：搜索框（`fn search_bar(query: &str, on_input: impl Fn(String) -> Msg) -> Element<Msg>`）、表单输入组。调用方负责存储 `query: String` 在自身的 State 中。

**带状态的 view + Message + update 模块**（需要自有状态和逻辑）：完整的"模块"——文件夹浏览器、设置面板、数据表格。调用方将模块的 State 嵌套在自身 State 中，将模块的 Msg 作为自身 Message 的一个变体，update 委托给模块的 `update()`。

iced 没有 runtime 的"注册组件"机制——三个层级都是编译时的函数组合。这在初次接触时显得样板代码更多（你需要显式指定 enum 变体和委托逻辑），但随着规模增大，强制性的分层边界使重构更容易——修改一个子模块的 State 不会影响其他模块，删除一个页面只需要删掉对应的 enum 变体和 match 分支，编译器会告诉你所有未处理的地方。

### 大型应用的路由与页面切换
多页面应用（设置页、主面板、关于页）通过 enum 表示当前页面，view 中 match 渲染不同内容：

```rust
enum Page { Home, Settings, About }

struct App {
    current_page: Page,
    home: home::State,
    settings: settings::State,
}

fn view(&self) -> Element<Message> {
    match self.current_page {
        Page::Home => home::view(&self.home).map(Message::Home),
        Page::Settings => settings::view(&self.settings).map(Message::Settings),
        Page::About => about_view(),
    }
}
```

页面切换是一个 Message 变体：`Message::NavigateTo(Page::Settings)`。`update()` 中 `self.current_page = Page::Settings`，view 自动切换到对应页面。没有路由库、没有 `react-router`——就是 enum + match。

iced 通过 wgpu 实现跨平台 GPU 渲染——wgpu 是 WebGPU 标准的 Rust 实现，在 Windows 上映射到 DX12/Vulkan、macOS 上映射到 Metal、Linux 上映射到 Vulkan。也支持 OpenGL 作为 fallback（通过 glow crate）。GPU 直接绘制意味着不需要内嵌浏览器引擎——一个完整桌面应用编译后约 3-8 MB。

内置组件覆盖标准桌面 UI 需求：button、text、text_input、slider、checkbox、radio、pick_list、scrollable、container、column、row、grid、toggler、tooltip、progress_bar。自定义绘制通过 `Canvas` 组件实现，提供类似 HTML5 Canvas 的程序化绘图 API（路径、形状、文字渲染）。

## 平台与生态
桌面平台（Windows/Mac/Linux）是一等公民。Web 通过 WASM + WebGL 编译到浏览器运行。移动端（iOS/Android）有实验性支持。iced 的组件库相对精简——没有表格、树形视图、富文本编辑器等复杂组件，这些需要基于 `Canvas` 自己实现。自定义 UI 和复杂交互的样板代码量高于 React/Vue 等声明式 Web 框架。

**适合**：工具软件、系统面板、不需要表格/树/富文本的桌面应用。**不适合**：需要大量复杂交互组件的企业应用、设计驱动的像素级定制产品、内嵌地图/视频/动画的多媒体应用。在这些场景下，Tauri 的 Web 生态复用是更好的选择。
