# ReactNative
rn 和 flutter 技术对比，diff 树



## React Native 与 Flutter
React Native 和 Flutter 是移动端跨平台开发的两条主线，它们回答同一个问题——"一套代码，双端运行"——但实现路径截然不同。RN 走"JS 驱动原生控件"的路线：UI 逻辑在 JavaScript 中描述，渲染交给平台原生的 UIKit/Android View。Flutter 走"自绘引擎"的路线：UI 全部由 Dart 描述，渲染由自带引擎（Skia/Impeller）直接画像素，平台只提供窗口和输入。

这个根本差异决定了两个框架的性能模型、调试体验和生态形态。理解它们的 diff 机制差异，是理解一切表层体验差异的钥匙。

## 渲染架构对比
```
React Native:
  JS 代码 (React 组件树)
      │  diff (Fiber reconciler)
      ▼
  Virtual DOM patch
      │  Bridge/JSI 序列化
      ▼
  原生组件 (UIView / android.view.View)
      │  平台渲染管线
      ▼
  屏幕像素

Flutter:
  Dart 代码 (Widget 树)
      │  每次重建（Widget 不可变）
      ▼
  Element 树 (diff 复用状态)
      │  标记脏节点
      ▼
  RenderObject 树 (布局与绘制)
      │  Skia/Impeller 直接光栅化
      ▼
  屏幕像素
```

RN 的渲染最终依赖平台组件——一个 `<Text>` 编译后是 iOS 的 UILabel 或 Android 的 TextView。这意味着 RN 应用的原生感和平台一致性天然好（用的就是原生控件），但跨平台一致性差（iOS 和 Android 的控件行为、默认样式不同）。Flutter 的渲染不经过任何平台组件——一个 Text 是 Flutter 引擎自己画出来的文字，双端像素级一致，但原生感依赖 Flutter 对平台设计规范的模仿质量。

## diff 树机制：Fiber 与 Element
两个框架都有"diff"——对比新旧 UI 描述，最小化更新。但 diff 的对象和粒度完全不同。

**RN 的 Fiber diff（React Reconciler）**：diff 发生在 JS 线程的 Virtual DOM 上。React 把组件树表示为 Fiber 节点链表，setState 后从变更节点开始向上标记更新优先级，随后 diff 产生 patch 列表（创建/更新/删除哪些组件），patch 通过 Bridge 序列化为原生命令（创建 UIView、设置属性、调用方法）。关键特征：diff 的对象是"组件"——粒度到 React 组件级别；diff 的执行者是 JS 线程——大组件树的 diff 会阻塞 JS 线程导致掉帧；patch 的传递跨语言边界——旧架构（Bridge）的序列化开销是经典性能瓶颈。

**Flutter 的 Element diff（三树结构）**：Flutter 维护三棵树——Widget 树（不可变的配置描述，每次 build 重建）、Element 树（Widget 的实例化状态，执行 diff）、RenderObject 树（布局和绘制对象）。build 时 Widget 树全量重建（代价极低——Widget 只是轻量配置对象），Element 树做 diff：相同类型和 key 的 Widget 复用对应 Element（保留其状态），变化的 Widget 更新 Element 关联的 RenderObject。关键特征：diff 的对象是"RenderObject"——粒度到布局/绘制节点；Widget 全量重建但 Element 增量复用——"重建的是配置，复用的是状态"；diff 和渲染同线程（UI 线程），但 diff 成本被 Widget 的轻量性摊薄。

这个差异在代码形态上的体现：RN 的状态更新触发组件重渲染（开发者用 memo/useMemo 控制重渲染范围），Flutter 的 setState 触发 build 重建 Widget 子树（开发者用 const 构造和状态拆分控制重建范围）。两个框架的优化手段不同——RN 优化"diff 的输入"，Flutter 优化"重建的成本"。

## 性能模型
**RN 的瓶颈在三处**：JS 线程的 diff（大型列表滚动时的 reconciler 开销）、Bridge 的序列化（旧架构每次 UI 更新跨桥传 JSON）、以及 JS 与原生线程的调度。新架构（Fabric + JSI）针对后两者：JSI 让 JS 直接调用原生 C++ 对象方法（替代 JSON 序列化），Fabric 把渲染调度改到原生线程。但 JS 线程 diff 的本质没有变——RN 的性能调优核心仍然是"让 JS 线程少干活"。

**Flutter 的瓶颈在两处**：build 阶段（Widget 树重建 + Element diff 在 UI 线程执行）和光栅化阶段（Raster 线程把 RenderObject 转为 GPU 指令）。Flutter 的性能模型更接近游戏引擎——固定帧预算（60fps = 16.6ms），超预算掉帧。调优核心是"让 build 更轻"（const 构造、状态下沉）和"让绘制更简单"（减少 layer 数、避免 saveLayer）。

延迟方面，Flutter 的触摸响应（120Hz 采样）和动画流畅度通常优于 RN——因为整个链路（输入 → build → 绘制）都在同一引擎内，没有跨语言边界。RN 的优点是原生滚动惯性和系统级手势——用的就是原生控件，滚动物理和手势识别是平台原生的。

## 生态与开发体验
RN 的最大资产是 React 生态：前端开发者零学习成本、npm 的数百万包（部分兼容）、TypeScript 全链路类型。Flutter 的最大资产是一致性和工具链：双端像素级一致（设计稿还原度高）、Flutter DevTools 的 widget inspector 和 hot reload、pub.dev 的组件生态。

包管理的成熟度差异显著：RN 依赖 npm + 原生模块（CocoaPods/Gradle）的双重依赖管理——版本冲突是经典痛点（React 版本 vs 原生模块版本 vs 平台 SDK 版本的三方矩阵）。Flutter 单一 pubspec.yaml + Dart 的 pub 依赖解析——依赖管理问题少一个量级。

热重载：Flutter 的 hot reload 保留状态注入新代码（百毫秒级），RN 的 Fast Refresh 类似但受原生模块变更限制（改了原生代码必须重新编译）。两个框架的 UI 层迭代体验都远好于原生开发，但 Flutter 的一致性略胜。

## 选型视角
**选 RN**：团队有 React 经验；应用需要深度调用平台原生能力（相机、地图、推送的定制化）；已有 Web 端 React 代码希望部分复用；需要集成大量既有原生模块（RN 的"逃生舱"——任何原生能力都可以桥接）。

**选 Flutter**：追求双端视觉一致性（设计稿还原）；动画和自定义 UI 复杂度高（Flutter 的自绘引擎做自定义绘制远超 RN 的组件组合）；团队愿意学习 Dart；目标还包括桌面端和嵌入式（Flutter 的跨端覆盖面比 RN 更广）。

两个框架的 diff 机制差异在长期维护中的体现：RN 的升级跟随 React 生态的演进（React 18 并发特性 → RN 新架构的逐步迁移），Flutter 的升级跟随引擎演进（Skia → Impeller 的渲染引擎替换是底层透明的）。选型的最终依据是团队技能栈和 UI 一致性需求的权重——两个框架的性能差异在大多数业务场景中都不是决定性因素。
