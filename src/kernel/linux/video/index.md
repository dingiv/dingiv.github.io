# 视图
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
  一个 UI 框架内部可能会包含的用于封装绘图指令和绘制基本图形的核心组件，与窗口系统交互，通过申请窗口资源，从而获得绘制图形的画布，在画布上执行绘制作业；渲染引擎通常不直接操作帧缓冲区，而是通过窗口系统提交内容；
  - Cairo：GTK 的矢量渲染引擎，擅长 2D 图形。
  - Skia：Flutter 和 Qt 的渲染引擎，支持软件和硬件渲染。
  - Pixman：低级像素操作库，Cairo 的后端之一。
+ 窗口系统：X11、Wayland  
  直接与系统交互，封装图形设备的**系统调用**和**帧缓冲区**，形成窗口管理系统，以本地服务的方式，提供窗口管理相关的调用；形成窗口层叠上下文，管理窗口之间的显示层级，协调鼠标键盘等用户输入设备，传递输入反馈；提供光标渲染；
+ 图形显示内核模块：fbdev 帧缓冲区、鼠标键盘模块  
  负责管理帧缓冲区，协调和管理各个图形设备，帧缓冲区的数据将被屏幕通过 DMA 读取，并进行显示；
+ 图形设备驱动：屏幕驱动、鼠标驱动、键盘驱动  
  linux 的内核设计，驱动程序用于直接与设备交互；

### GPU 加速
现代的计算机普遍支持 GPU 加速的能力，使用 GPU 加速的视图系统将会经历更加复杂的图形链路。
+ UI 组件库；
+ UI 框架：GTK、Qt、Flutter；
+ 渲染引擎：Cairo、Skia  
  该组件在 GPU 加速的场景下，将会调用 GPU 的图形 API 进行加速，并协调 GPU 绘制的输入输出，将 GPU 的输出内容显示到窗口上；
+ 窗口系统 + 图形 API：X11、Wayland + OpenGL、Metal  
  在该层窗口系统将为渲染引擎提供一个窗口或者说渲染上下文或者说画布，GPU 图形 API，是在应用空间封装的用于和 GPU 进行交互的库。在该层，二者进行协作完成任务，而协调二者的正是渲染引擎；
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
+ 图形显示内核模块：drm、帧缓冲区、鼠标键盘模块  
  在 GPU 场景下，drm 模块负责管理帧缓冲区和 GPU，在 GPU 的中显存中创建帧缓冲区，然后数据被显示到屏幕上；
+ 图形设备驱动：GPU 驱动、屏幕驱动、鼠标驱动、键盘驱动  
  i915（Intel）、amdgpu（AMD）、nvidia（NVIDIA）；


### Windows 视图系统结构对比
Windows 是闭源系统，其窗口系统被封装在系统内部，运行在内核态。上层应用程序通过 win32 API 创建和管理窗口。在 GPU 加速方面，使用 Windows 自有的 Direct2D API 进行加速。在没有 GPU 的机器上使用 GDI 渲染模式，对标 Linux 上的软件渲染。


## 渲染引擎
一个渲染引擎面向上层框架，背靠窗口系统和 GPU 的图形 API 进行工作。渲染引擎向上暴露一些基本的图形绘制指令和基本图形的绘制能力，向下将这些指令转化为像素的数据填充到画布之上。

### 窗口
渲染引擎需要向窗口系统申请窗口，也可以理解为一个画布、一个绘制图形的上下文。一个窗口可以看作一个矩形区域，例如一个有 2k 屏幕大小的窗口为 2560*1440。它使用一个 usigned 类型的一维数组表示，其中的每个元素代表了一个像素的颜色数据，占用 4 个字节，分别表示 R、G、B、A 四个通道，取值 0~255，使用 RGBA 标准。当然也可以使用其他的标准绘制，这样的话，申请的画布属性就会有所不同。人眼能识别的极限色彩大概有200万~1000万种，而 256^3=16,777,216，能够覆盖人眼的视觉极限，同时，32 位的大小满足计算机的内存对齐，能够方便进行硬件加速和优化。

### 软件渲染
在没有 GPU 设备进行硬件渲染的环境上，渲染引擎申请到窗口之后，将高级指令（如 draw_circle）转为像素颜色，向其中填充像素数据；然后提交给窗口系统（如 Wayland 的 wl_surface.commit），窗口将会被窗口系统进行托管和显示，并传递 UI 交互。

### GPU 渲染
在大多数现代的设备上，往往可以使用 GPU 对图形显示进行加速，包括独立显卡或者集成显卡。驱动 GPU 进行图形渲染加速需要使用 GPU 能够**听懂**的语言，也就是图形 API，不同的操作系统上的图形 API 有所不同，渲染引擎将会为上层屏蔽这些差异，在不同的平台上调用不同的图形 API，操作 GPU 进行图形计算，这样就可以减轻 CPU 的压力，从而获得快速的图形体验。

在 GPU 渲染场景下，渲染引擎依然需要申请一个窗口作为绘制图形的画布，但是这个画布是为 GPU 申请的，渲染引擎通过图形 API 指挥 GPU 完成绘制工作。二者在系统层中有着显著的差异。

### 图形 API
GPU 提供给上层应用的调用的接口，往往由渲染引擎进行调用。
- OpenGL 就是一个开放的图形 API 接口，用于调用 GPU 和 GPU 驱动层进行交互，它是跨平台的标准。
- DirectX 是一个 windows 平台专属的图形 API，用于在 Windows 平台上操作 GPU。
- Metal 是一个苹果系统的专属 API，使用在 IOS 和 MAC 系统上。
- Vulkan 是安卓系统的专属 API。

## 窗口系统
linux 窗口系统，利用系统调用操作和管理包括屏幕、键盘、鼠标、GPU等用户图形交互设备，抽象和管理桌面窗口的数据结构，向其它应用程序提供窗口 API，应用程序只需调用其 API，便可获得一个窗口，并能够在上面绘制自己的图形。

### 合成器
合成器是窗口系统的核心组件，它需要完成的一个核心任务便是计算多个窗口直接的层叠关系，控制窗口的显示和隐藏，同时向上传递鼠标和键盘的输入事件，形成 UI 反馈。窗口系统还需要负责鼠标光标的绘制。

需要强调的是，linux 的窗口系统与 Windows 和苹果不同，linux 窗口在用户空间实现。X11 是 linux 系统长期以来使用的窗口系统，但是今年来，新的 Wayland 标准在 linux 系统中逐渐替换，X11 的标准将会被逐步取代。

### GUI 程序
桌面软件，是一种特殊的 GUI 程序，它会向窗口系统申请一个占满屏幕大小的窗口，并显示出各种各样的用于系统管理的交互界面和组件，如任务栏、桌面背景，从而方便用户对计算机进行可视化管理。linux 环境上最主流的两个桌面系统是 GNOME（基于 GTK 框架，Cairo 引擎）和 KDE（基于 Qt，Skia 引擎）。

普通 GUI 软件，普通 GUI 程序往往会通过平台提供的原生 UI 库，来展示自己的窗口，他们不直接处理绘制逻辑，使用现有的原生组件和 UI，原生组件的特点是性能较好，开箱即用，但是不能跨平台，样式简单，美化起来比较麻烦，数量有限，可定制化程度低。在 linux 上，平台原生的 UI 框架是 GTK。这些 GUI 程序往往具有统一的窗口修饰组件，因此他们在长相上非常相似。

自绘制 GUI 软件，为了实现跨平台的 UI 开发能力和跨平台组件，一些 UI 框架（如：Blink、Flutter、Qt）使用跨平台的渲染引擎（如 Skia）进行图形渲染，然后提供自己的上层组件库。这些窗口的长相风格各异，可定制化程度高，并且能够提供跨平台的 UI 一致性体验。


## 视图管理内核模块
内核中与视图显示相关的模块为上层提供系统接口，从而让上层获得操作视图设备的能力。

### 帧缓冲区
linux 内核中维护着一个特殊的内存区域——帧缓冲区，该缓冲区可以理解为屏幕显示器的数据输入源，屏幕显示器通过 DMA 技术直接访问这块内存区域，从而刷新自己的图像。帧缓冲区可能不止一个，出于多显示器的支持，双缓冲区优化，虚拟帧缓冲区的支持，一个内核中可以有多个帧缓冲区，帧缓冲区是整个系统共享的资源，为了防止资源的竞争，一般由窗口系统来操作帧缓冲区，并规划窗口，其他的程序需要经过窗口系统获得窗口来进行图形渲染。

### fbdev
这个模块 `fbdev.ko` 用于管理和申请**帧缓冲区**，可以通过操作 `/dev/fb0` 来访问，往往由窗口系统来进行操作或者在嵌入式设备中，也可以由开发者自己直接操作帧缓冲区，从而实现全部的屏幕显示控制。但是，随着 GPU 的普及，在现代的系统中其逐渐减少了使用。
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
内核模块 `drm.ko` 负责管理 GPU 硬件，提供用户空间程序（如 Wayland 合成器、X11 服务器、渲染引擎）与硬件之间的接口，抽象不同 GPU 的差异（如 Intel、AMD、NVIDIA）。依赖具体的、不同厂商的硬件驱动程序。

+ GPU 渲染管理：分配显存，执行渲染指令（如 OpenGL、Vulkan）。支持 3D 渲染（Unity、Unreal）和 2D 渲染（Skia、Cairo）。
+ 显示管理：配置显示模式（分辨率、刷新率）。管理帧缓冲区（显存中的像素数据）。
+ 硬件抽象：屏蔽不同 GPU 厂商（Intel、AMD、NVIDIA）的差异。
+ 同步与调度：协调多进程访问 GPU，防止资源冲突。
+ 支持垂直同步（V-Sync），避免画面撕裂。

#### DRM 核心
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
