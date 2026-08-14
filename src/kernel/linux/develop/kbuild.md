# linux 内核构建系统
linux 内核构建系统是基于 gcc 和 makefile，并在此基础上添加了适用于内核构建的特性


 内核没有标准库

  -nostdinc          # 不使用系统头文件
  -ffreestanding     # 不假设标准库存在
  -fno-builtin       # 不使用编译器内置函数替换
  内核自己实现了 memcpy、printf（printk）等，不依赖 glibc。

  NULL 不是无效地址

  -fno-delete-null-pointer-checks    # 禁止编译器优化掉 NULL 检查
  在用户态，NULL (0x0) 永远不可访问，编译器会优化掉对 NULL 的检查。但在内核空间，地址 0 是合法的（尤其在 x86-32 上），不能被优化掉。

  禁用浮点

  -mno-sse -mno-mmx -mno-sse2 -mno-avx -mno-80387
  内核代码不能使用浮点运算。因为切换 FPU/SSE 上下文代价高昂，内核选择完全不使用。需要浮点时，必须手动保存/恢复 FPU 状态。

  红区不可用

  -mno-red-zone    # 仅 x86-64
  x86-64 ABI 规定函数栈帧下方有 128 字节的"红区"可供叶函数使用而不必调整栈指针。但内核的中断随时会打断执行，红区会被覆盖，所以必须禁用。

  内核代码模型

  -mcmodel=kernel   # 仅 x86-64
  x86-64 内核运行在高位地址（0xffffffff80000000 起），超过普通 -mcmodel=small 的 2GB 寻址范围，必须用 kernel 模型。

  禁止严格别名优化

  -fno-strict-aliasing
  内核大量做类型转换（type-punning），比如把一块内存既当 struct boot_params 又当 char[] 来操作。严格别名优化会破坏这些代码。

  3. GCC 插件体系

  内核还有自己的 GCC 插件（scripts/gcc-plugins/）：

  ┌─────────────────────────┬────────────────────────────────┐
  │          插件           │              作用              │
  ├─────────────────────────┼────────────────────────────────┤
  │ randomize_layout_plugin │ 随机化结构体字段布局，防御攻击 │
  ├─────────────────────────┼────────────────────────────────┤
  │ stackleak_plugin        │ 系统调用返回时擦除内核栈       │
  ├─────────────────────────┼────────────────────────────────┤
  │ latent_entropy_plugin   │ 为变量注入随机熵               │
  └─────────────────────────┴────────────────────────────────┘

  4. 编译器抽象层

  内核通过头文件抹平 GCC 和 Clang 的差异：

  include/linux/compiler_types.h    → 根据 __clang__/__GNUC__ 分发
    ├─ include/linux/compiler-gcc.h   # GCC 特定定义
    └─ include/linux/compiler-clang.h # Clang 特定定义
  include/linux/compiler_attributes.h  # 统一的 __attribute__ 宏

  常用宏映射：
  #define __always_inline  inline __attribute__((always_inline))
  #define __packed         __attribute__((packed))
  #define __noreturn       __attribute__((noreturn))
  #define __cold           __attribute__((cold))
  #define noinstr          noinline notrace  // 不能被追踪/内联

## 内核启动命令行参数
编译产出 bzImage 之后，控制内核行为的第二个入口是启动命令行（kernel cmdline）——Boot Loader 传递的 `console=ttyS0 root=/dev/sda1` 这类参数。内核启动早期解析它们并据此配置自身。调试开发场景的常用参数按功能分组：

**控制台与输出**（调试第一需要——没有输出就没有信息）：

| 参数 | 作用 |
|------|------|
| `console=ttyS0,115200` | 串口控制台（QEMU `-nographic`、无头服务器的标配） |
| `console=tty0 console=ttyS0` | 双控制台——VGA 和串口同时输出（最后的 console 是 stdin/stdout 主设备） |
| `earlyprintk=serial,ttyS0,115200` | 最早的输出通道——在 printk 初始化前就可用（调试启动早期崩溃） |
| `printk.devkmsg=on` | 允许用户态写入 /dev/kmsg（测试注入） |
| `loglevel=8` | 提高控制台日志级别（8=debug 全开） |

**内存与地址布局**（调试内存管理、KASLR 相关问题）：

| 参数 | 作用 |
|------|------|
| `nokaslr` | 禁用内核地址随机化——KASLR 让调试符号和断点全部失效，内核调试必加 |
| `mem=4G` | 限制内核可见内存（复现小内存场景） |
| `memmap=4G$0x100000000` | 在指定物理地址保留内存（设备 DMA 缓冲、大页预留） |
| `crashkernel=256M` | 为 kdump 的 crash kernel 预留内存 |
| `hugepages=1024` | 启动时预留大页 |

**调试基础设施**：

| 参数 | 作用 |
|------|------|
| `oops=panic` | oops 升级为 panic（避免带伤继续运行） |
| `panic=5` | panic 后 5 秒重启（配合看门狗做自动恢复） |
| `kgdboc=ttyS1,115200` | KGDB over console（串口连接 GDB 调试内核） |
| `debug` | 提升子系统 debug 日志级别 |
| `initcall_debug` | 打印每个 initcall 的执行时间（定位启动慢的驱动） |
| `ignore_loglevel` | 输出所有日志（无视 loglevel 设置） |

**故障注入与测试**：

| 参数 | 作用 |
|------|------|
| `failslab=` / `fail_page_alloc=` | 故障注入框架（测试错误处理路径） |
| `fault_inject=` | 通用故障注入配置 |
| `ftrace_dump_on_oops` | oops 时自动 dump ftrace 缓冲（事后分析现场） |
| `trace_event=` | 启动时启用指定 tracepoint（捕获启动期事件） |

**根文件系统与 init**（引导问题排查）：

| 参数 | 作用 |
|------|------|
| `root=/dev/sda1 rootwait` | 根设备（rootwait 等待设备就绪——USB/NVMe 枚举慢时必需） |
| `rootflags=` | 根文件系统挂载选项 |
| `init=/bin/sh` | 跳过 init 系统（救援模式——忘记密码、init 损坏） |
| `rdinit=` / `rd.break` | initramfs 阶段打断（dracut 的救援入口） |
| `systemd.unit=emergency.target` | systemd 的紧急目标 |

命令行的解析机制：早期解析（`setup_arch` 前）处理 `earlycon`/`earlyprintk` 这类"越早越好"的参数；主体解析由 `parse_args` 遍历——每个参数匹配注册的 `__setup()` 或 `early_param()` 宏处理函数（驱动通过模块参数机制 `module_param` 注册自己的参数）。未识别的参数传递给 init 进程（`/proc/cmdline` 可查看完整命令行）。调试场景的实用习惯：维护一份 QEMU 启动脚本固定 `nokaslr console=ttyS0` 等参数组合——避免每次调试重新摸索。
