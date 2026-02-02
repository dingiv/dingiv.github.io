# 视图系统
linux 的视图系统是应用进程通过显示器形成可视化的图形用户界面的整套链路。

## 层次结构
为了理解 linux 系统的桌面视图的生成原理，需要自上而下地了解每一个层次。其中包括的层次有多个，并且需要区分有无 GPU 加速的情况：

### 无 GPU 加速
无 GPU 加速的场景适用于老式设备、嵌入式设备、虚拟化场景，依赖 CPU 进行软件渲染，效率较低但硬件要求简单。
+ UI 组件库  
  基于 UI 框架封装常见的 UI 组件，提供易用、美观、强大的 UI 体验；封装复杂交互逻辑，开发者只需调用高级 API（如 gtk_button_new）；
+ UI 框架：GTK、Qt、Flutter  
  封装在一个平台上绘制窗口和组件的基本能力，提供输入输出的回调机制，给定最基本的程序架构的模型，并提供跨平台的 UI 构建能力；
  - GTK：GNOME 生态核心，基于 C，使用 Cairo 渲染。
  - Qt：KDE 生态核心，基于 C++，使用 QPainter/Skia。
  - Flutter：跨平台框架，基于 Dart，使用 Skia 渲染。
  ```c
  #include <gtk/gtk.h>
  int main(int argc, char *argv[]) {
      gtk_init(&argc, &argv);
      GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
      gtk_window_set_default_size(GTK_WINDOW(window), 800, 600);
      gtk_widget_show_all(window);
      gtk_main();
      return 0;
  }
  ```
+ 渲染引擎：Cairo、Skia、Pixman  
  一个 UI 框架内部可能会包含的用于封装绘图指令和绘制基本图形的核心组件，与窗口系统交互，申请窗口资源，从而获得键鼠输入；开辟一块内存作为绘制图形的画布，在画布上执行绘制作业；
  - Cairo：GTK 的矢量渲染引擎，擅长 2D 图形。
  - Skia：Flutter 和 Qt 的渲染引擎，支持软件和硬件渲染。
  - Pixman：低级像素操作库，Cairo 的后端之一。
+ 窗口系统：X11、Wayland  
  直接与内核交互，封装图形设备的**系统调用**和**帧缓冲区**，形成窗口管理系统，以本地服务的方式监听 unix socket，提供窗口管理相关的调用；
+ 图形显示内核模块 fbdev： 帧缓冲区、鼠标键盘模块  
  负责管理直接管理帧缓冲区，协调和管理各个图形设备，帧缓冲区的数据将被屏幕通过 DMA 读取，并进行显示；
+ 图形设备驱动：屏幕驱动、鼠标驱动、键盘驱动  
  linux 的内核设计，驱动程序用于直接与设备交互；
+ 屏幕：  
  通过 DMA 读取帧缓冲区，根据显示结果的像素数据控制像素的颜色显示；

### GPU 加速
现代的计算机普遍支持 GPU 加速的能力，使用 GPU 加速的视图系统将会经历更加复杂的图形链路。
+ UI 组件库；
+ UI 框架：GTK、Qt、Flutter；
+ 渲染引擎：Cairo、Skia  
  该组件在 GPU 加速的场景下，将会**调用 GPU 的图形 API 进行加速**，渲染引擎会将图形资源和计算指令放入显存中，然后 GPU 访问资源进行计算，GPU 将计算的结果放入指定的显存区域中；
+ 窗口系统：X11、Wayland + OpenGL、Metal
  
  在存在 GPU 的状态下，窗口系统会将帧缓冲区申请在显存中；接受其他窗口进程发送的渲染指令，然后读取结果的 fd，并**再次调用图形 API** 通过 GPU 进行加速叠加合成；
  ```c
  #include <EGL/egl.h>
  #include <GLES2/gl2.h>
  int main() {
      EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      eglInitialize(display, NULL, NULL);
      EGLConfig config;
      EGLint num_config;
      eglChooseConfig(display, NULL, &config, 1, &num_config);
      eglTerminate(display);
      return 0;
  }
  ```
+ 图形 API：  
  存在于用户态的供用户态应用调用 GPU 进行图形计算的 API，并提供着色器语言 DSL，让开发者编写运行在 GPU 之上的程序着色器，从而计算渲染逻辑；
  
+ 图形显示内核模块 drm：帧缓冲区、鼠标键盘模块  
  在 GPU 场景下，drm 模块负责管理帧缓冲区和 GPU，在 GPU 的中显存中创建帧缓冲区；
+ 图形设备驱动：GPU 驱动、显示适配器、鼠标驱动、键盘驱动
  i915（Intel）、amdgpu（AMD）、nvidia（NVIDIA）；
+ GPU(图形适配器)：  
  GPU 向内核暴露**显示适配器**，内核将其当作一个屏幕处理；屏幕不再连接到主板之上，而是连接到 GPU 的输出口上，此时，屏幕由 GPU 接管；
+ 屏幕：  
  从 GPU 的显存中读取像素数据，控制像素颜色，显示结果；


### Windows 视图系统结构对比
Windows 是闭源系统，其窗口系统被封装在系统内部，运行在内核态。上层应用程序通过 win32 API 创建和管理窗口。在 GPU 加速方面，使用 Windows 自有的 Direct2D API 进行加速。在没有 GPU 的机器上使用 GDI 渲染模式，对标 Linux 上的软件渲染。


## 渲染引擎
一个渲染引擎面向上层框架，背靠窗口系统和 GPU 的图形 API 进行工作。渲染引擎向上暴露一些基本的图形绘制指令和基本图形的绘制能力，向下将这些指令转化为像素的数据填充到画布之上。

### 窗口上下文
渲染引擎需要向窗口系统申请窗口，也可以理解为一个绘制图形的上下文。渲染引擎通过窗口上下文获得鼠标和键盘的输入。

渲染引擎会在内存中或者显存中创建一个 buf 来存放像素计算的结果，这个 buf 使用一个 usigned int 类型的一维数组表示，但是可以看作一个矩形区域，例如一个有 2k 屏幕大小的窗口为 `2560*1440`，长度就是 `2560*1440`。buf 中的每个元素代表了一个像素的颜色数据，占用 4 个字节，分别表示 R、G、B、A 四个通道，取值 0~255，使用 RGBA 标准。当然也可以使用其他的标准绘制，这样的话，申请的画布属性就会有所不同。人眼能识别的极限色彩大概有200万~1000万种，而 256^3=16,777,216，能够覆盖人眼的视觉极限，同时，32 位的大小满足计算机的内存对齐，能够方便进行硬件加速和优化。

渲染引擎通过图形 API 来在显存中开辟空间，并通过图形 API 向 GPU 提交需要运行的任务代码——着色器程序，然后 GPU 开始工作，将图形任务完成后输出到指定的区域。然后，渲染引擎每一次渲染完成一帧之后，将这块显存区域的 dma-buf 文件描述符 fd 发送给窗口系统，窗口系统进行合成。

### 软件渲染
在没有 GPU 设备进行硬件渲染的环境上，渲染引擎申请到窗口之后，将高级指令（如 draw_circle）转为像素颜色，向其中填充像素数据；然后提交给窗口系统（如 Wayland 的 wl_surface.commit），窗口将会被窗口系统进行托管和显示，并传递 UI 交互。

### GPU 渲染
在大多数现代的设备上，往往可以使用 GPU 对图形显示进行加速，包括独立显卡或者集成显卡。驱动 GPU 进行图形渲染加速需要使用 GPU 能够**听懂**的语言，也就是图形 API，不同的操作系统上的图形 API 有所不同，渲染引擎将会为上层屏蔽这些差异，在不同的平台上调用不同的图形 API，操作 GPU 进行图形计算，这样就可以减轻 CPU 的压力，从而获得快速的图形体验。

在 GPU 渲染场景下，渲染引擎申请的 buf 是直接放在显存中的，如果是集成显卡，那么是放在系统从主内存中预留给集成显卡的内存区域中的，这块区域是核显显存。GPU 加速场景下的一个重要特点在于，像素结果数据的搬运往往是由 GPU 搬运和处理，无需 CPU 进行搬运，这一点非常重要，因为 CPU 的数据拷贝性能较差，CPU 更擅长逻辑运算，在纯 CPU 渲染的情况下，CPU 的占用率非常高。

渲染引擎通常不直接操作帧缓冲区，而是通过窗口系统提交渲染结果；渲染引擎将渲染结果的 buf 通过 dma-buf fd 的形式与窗口系统共享，传递时通过 unix socket 发送 dma-buf 的文件描述符。

### 图形 API
GPU 提供给上层应用的调用的接口，往往由渲染引擎和窗口系统等用户态需要利用 GPU 进行加速计算的程序进行调用。图形 API 提供的接口包括：显存管理，绘制调用等核心功能；
- OpenGL 就是一个开放的图形 API 接口，用于调用 GPU 和 GPU 驱动层进行交互，它是跨平台的标准。
- DirectX 是一个 windows 平台专属的图形 API，用于在 Windows 平台上操作 GPU。
- Metal 是一个苹果系统的专属 API，使用在 IOS 和 MAC 系统上。
- Vulkan 是下一代开放图形 API。

为了指挥 GPU 做事，图形 API 会支持一套用于编写 GPU 程序的编程语言——着色器语言，通过编写着色器，然后负责具体的渲染逻辑的实现。
- OpenGL -> glsl，文本
- DirectX -> hlsl，文本
- Metal -> msl，文本
- Vulkan -> spir-v，字节码

## 窗口系统
linux 窗口系统，利用系统调用操作和管理包括屏幕、键盘、鼠标、GPU等用户图形交互设备，抽象和管理桌面窗口的数据结构，通过监听本地的 unix socket 对其他进程提供服务，其他进程通过发起请求，便可获得一个窗口上下文，获得用户的桌面交互。需要强调的是，linux 的窗口系统与 Windows 和苹果不同，linux 窗口在用户空间实现。X11 是 linux 系统长期以来使用的窗口系统，但是今年来，新的 Wayland 标准在 linux 系统中逐渐替换，X11 的标准将会被逐步取代。

### 合成器
合成器是窗口系统的核心组件，它需要完成的一个核心任务便是计算多个窗口直接的层叠关系，控制窗口的显示和隐藏，同时向上传递鼠标和键盘的输入事件，形成 UI 反馈。窗口系统还需要负责鼠标光标的绘制。

窗口系统接受其他进程的渲染调用，其他进程使用 dma-buf 的 fd 访问其他进程的渲染内容，然后计算合成结果，统一刷新到帧缓冲区。这一过程中，Wayland 依然也会进行判断，如果有 GPU，那么 Wayland 会把帧缓冲区申请在显存上，然后合成逻辑会运行在 GPU 上，在 GPU 计算完成后，输出的位置就是帧缓冲区，然后屏幕直接读显存显示图像。

### 桌面环境
**桌面环境**软件，是一种特殊的 GUI 程序，它会向窗口系统申请一个占满屏幕大小的窗口，并显示出各种各样的用于系统管理的交互界面和组件，如任务栏、桌面背景，从而方便用户对计算机进行可视化管理。linux 环境上最主流的两个桌面系统是 GNOME（基于 GTK 框架，Cairo 引擎）和 KDE（基于 Qt，Skia 引擎）。桌面环境工作在 UI 框架层，它内部封装了一套自己选择的渲染引擎，并通过提供 API，让上层的应用使用自己风格的 UI 组件。

普通 GUI 软件，普通 GUI 程序往往会通过**桌面环境**提供的原生 UI 库，来展示自己的窗口，他们不直接处理绘制逻辑，使用现有的原生组件和 UI，原生组件的特点是性能较好，开箱即用，但是不能跨**桌面环境**，样式简单，美化起来比较麻烦，数量有限，可定制化程度低。在 linux 上，平台原生的 UI 框架是 GTK。这些 GUI 程序往往具有统一的窗口修饰组件，因此他们在长相上非常相似。

自绘制 GUI 软件，为了实现跨平台的 UI 开发能力和跨平台组件，一些 UI 框架（如：Blink、Flutter、Qt）使用跨平台的渲染引擎（如 Skia）进行图形渲染，然后提供自己的上层组件库。这些窗口的长相风格各异，可定制化程度高，并且能够提供跨平台的 UI 一致性体验。


## 视图管理内核模块
内核中与视图显示相关的模块为上层提供系统接口，从而让上层获得操作视图设备的能力。

### 帧缓冲区
linux 内核中维护着一个特殊的内存区域——帧缓冲区，该缓冲区可以理解为屏幕显示器的数据输入源，屏幕显示器通过 DMA 技术直接访问这块内存区域，从而刷新自己的图像。帧缓冲区可能不止一个，出于多显示器的支持，双缓冲区优化，虚拟帧缓冲区的支持，一个内核中可以有多个帧缓冲区，帧缓冲区是整个系统共享的资源，为了防止资源的竞争，一般由窗口系统来操作帧缓冲区，并规划窗口，其他的程序需要经过窗口系统获得窗口来进行图形渲染。

### fbdev
这个模块 `fbdev.ko` 用于管理和申请**帧缓冲区**，可以通过操作 `/dev/fb0` 来访问，往往由窗口系统来进行操作或者在嵌入式设备中，也可以由开发者自己直接操作帧缓冲区，从而实现全部的屏幕显示控制。但是，随着 GPU 的普及，现代的 CPU 往往都直接集成了一个核显，所以，fbdev 在现代硬件平台上逐渐减少了使用。在现代的 Linux 系统上，即使没有 GPU，DRM 依然会在启动之后接管 fbdev 的工作。
```c
#include <fcntl.h>
#include <sys/mman.h>
int main() {
    int fb = open("/dev/fb0", O_RDWR);
    char *fbmem = mmap(NULL, 800 * 600 * 4, PROT_WRITE, MAP_SHARED, fb, 0);
    for (int i = 0; i < 800 * 600; i++) fbmem[i * 4 + 2] = 255; // 红色
    munmap(fbmem, 800 * 600 * 4);
    close(fb);
    return 0;
}
```

### DRM（Direct Rendering Manager）
内核模块 `drm.ko` 负责管理 Linux 图形显示相关的硬件，提供用户空间程序（如 Wayland 合成器、X11 服务器、渲染引擎）与硬件之间的接口，抽象不同 GPU 的差异（如 Intel、AMD、NVIDIA）。依赖具体的、不同厂商的硬件驱动程序。

+ GPU 渲染管理：分配显存，执行渲染指令（如 OpenGL、Vulkan）。支持 3D 渲染（Unity、Unreal）和 2D 渲染（Skia、Cairo）。
+ 显示管理：配置显示模式（分辨率、刷新率）。管理帧缓冲区（显存中的像素数据）。
+ 硬件抽象：屏蔽不同 GPU 厂商（Intel、AMD、NVIDIA）的差异。
+ 同步与调度：协调多进程访问 GPU，防止资源冲突。
+ 支持垂直同步（V-Sync），避免画面撕裂。

文件：drivers/gpu/drm/drm_*。提供通用接口（如 ioctl），供用户空间调用。管理设备文件（/dev/dri/card0、/dev/dri/renderD128）。协调多 GPU 和多显示器。
设备节点：  
`/dev/dri/card0`：主设备，用于显示和渲染。
`/dev/dri/renderD128`：渲染专用，供 Vulkan/OpenGL。

#### KMS (Kernel Mode Setting)
定义：DRM 的显示管理子系统，负责配置显示硬件。设置分辨率、刷新率（如 1920x1080@60Hz）。管理 CRTC（显示控制器）、Encoder（信号编码）、Connector（HDMI/DP）。
支持多显示器（热插拔）。
```c
#include <libdrm/drm.h>
#include <libdrm/drm_mode.h>
int main() {
    // 通过 KMS 设置显示模式
    int fd = open("/dev/dri/card0", O_RDWR);
    drmModeRes *res = drmModeGetResources(fd);
    drmModeConnector *conn = drmModeGetConnector(fd, res->connectors[0]);
    drmModeSetCrtc(fd, res->crtcs[0], conn->modes[0].mode, 0, 0, NULL, 0, NULL);
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(fd);
    return 0;
}
```

还有其他模块
+ GEM (Graphics Execution Manager)，定义：显存管理子系统。分配和管理显存中的缓冲区（Buffer Objects，BO）。支持帧缓冲区、纹理、顶点数据。提供共享内存（如 DMA-BUF）给 Wayland/X11。Skia 的 Vulkan 后端通过 GEM 分配纹理。Cycles 用 GEM 管理 GPU 内存。
+ TTM (Translation Table Maps)，定义：显存分页管理（较老，部分 GPU 使用）。处理显存不足时的交换（GPU 到系统内存）。优化内存分配。新 GPU（如 AMD、Intel）更依赖 GEM。
+ Vendor-Specific Drivers，作用：实现 DRM 的硬件特定逻辑。抽象硬件细节，提供统一的 DRM API。
+ DMA-BUF，定义：跨设备缓冲区共享机制。Flutter 的 Skia Vulkan 后端通过 DMA-BUF 提交渲染结果。

#### DRM 在视图系统中的工作流程
以 GPU 加速场景为例，DRM 如何协作：
+ 初始化：Wayland 合成器（Mutter）打开 /dev/dri/card0。KMS 检测显示器，设置 1920x1080@60Hz。
+ 渲染：Flutter（Skia）用 Vulkan 渲染 UI，GEM 分配显存。Cycles 用 CUDA 调用 DRM，计算光线追踪。
+ 显示：渲染结果存入帧缓冲区（显存）。KMS 通过 CRTC 输出到 HDMI。
+ 输入：DRM 配合 evdev（/dev/input）传递鼠标/键盘事件。Mutter 分发给 Flutter 窗口。

## 完整绘制流程
1. 生产阶段：App 进程的“私下操作”
   
   在 App 进程中，渲染引擎（如 Skia 或游戏引擎）并不会直接和屏幕对话。

   分配 Buf：App 调用图形 API（Vulkan/EGL），驱动程序会在显存中申请一块内存（Buffer）。

   计算存储：GPU 接收指令，将渲染结果（像素数据）填入这个 Buffer。

   导出句柄：重点来了。由于 Linux 进程间内存是隔离的，App 必须通过 dma-buf 机制，将这块显存导出为一个文件描述符（File Descriptor, FD）。

   类比：你做了一桌菜（Buffer），你没法把桌子搬走，但你给了 Wayland 一把房间的钥匙（FD）。

2. 传递阶段：Wayland 的“协议握手”
   
   App 渲染完一帧后，会通过 Wayland 协议 发送一个请求（通常是 wl_surface.commit）：

   发送 FD：App 把刚才那个 dma-buf 的 FD 发给 Wayland 进程。

   元数据同步：同时告知 Wayland 这个 Buffer 的尺寸、像素格式（如 RGBA8888）以及步长（Stride）。

   同步栅栏（Fence）：App 还会发送一个“栅栏”，告诉 Wayland：“Buffer 还在写入中，等这个栅栏信号亮了，你再动它。”

3. 合成与显示：Wayland 进程的“接力”
   
   Wayland 进程（Compositor，如 GNOME 的 Mutter）收到通知后：

   导入 Buf：Wayland 调用图形 API（Vulkan/OpenGL），通过 dma-buf 接口直接把 App 的那块显存挂载到自己的渲染管线中。

   合成（Compositing）：Wayland 把你的 App 窗口（现在只是它的一张纹理）、背景墙纸、其他窗口，通过 GPU 进行一次“大拼装”，生成最终的一张覆盖全屏的 屏幕 Buffer。

   放入 DRM：Wayland 调用 libdrm。它通过 drmModePageFlip 或现代的 Atomic KMS 接口，告诉内核中的 DRM 模块：“嘿，这块新的 Buffer 已经合好了，下次显示扫描时，直接读这个地址。”

   显示刷新：显示器硬件在下一次 VSync（垂直同步）信号到来时，直接从显存中读取该 Buffer，像素就这样亮了起来。

步骤,操作主体,关键技术,本质
1. 绘制,App (Client),Vulkan / OpenGL,填充显存
2. 分享,App → Wayland,dma-buf (FD),跨进程传递显存钥匙
3. 合成,Wayland (Server),Vulkan / OpenGL,纹理叠加
4. 显示,Wayland → Kernel,DRM / KMS,显存地址翻转 (Flip)