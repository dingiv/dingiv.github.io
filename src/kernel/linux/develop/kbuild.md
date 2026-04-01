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
