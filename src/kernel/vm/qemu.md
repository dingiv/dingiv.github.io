---
title: QEMU 内核调试
order: 10
---

# QEMU 内核调试
QEMU 对内核开发者的价值远超"跑一个虚拟机"——它是内核调试最顺手的工具。真实的硬件调试需要 JTAG 探针、串口线缆和一块可能被调试错误烧毁的开发板，而 QEMU 把这一切虚拟化：GDB 可以附加到虚拟机内的任意状态、串口输出直接重定向到终端、物理内存和寄存器随时可以检查。内核开发中最有价值的反馈回路——"修改 → 编译 → 启动 → 观察"——在 QEMU 中可以被压缩到几十秒。

## GDB 调试内核
QEMU 内置 GDB stub——在用户态模拟虚拟机的 CPU 状态，等待 GDB 连接。启动参数 `-s -S` 的含义：`-s` 在 TCP 1234 端口开放 GDB 服务，`-S` 让虚拟机在第一条指令处暂停等待调试器。

```bash
# 启动内核调试会话
qemu-system-x86_64 -kernel bzImage \
  -initrd initramfs.cpio.gz \
  -append "console=ttyS0 nokaslr" \
  -s -S \
  -nographic
```

`nokaslr` 是内核调试的标配——KASLR 让内核地址每次启动随机化，调试符号和断点会全部失效。在另一个终端连接 GDB：

```gdb
# gdb vmlinux
(gdb) target remote :1234
(gdb) b start_kernel
(gdb) c
```

断点命中后，`bt` 查看调用栈、`info registers` 查看寄存器、`x/20x $rsp` 查看栈内容——与调试普通用户态程序完全相同的体验，但被调试对象是内核启动早期的代码。

调试内核模块的常用技巧：模块在加载时地址动态分配，断点需要延迟设置。先在 `module_init` 函数入口打断点确认模块代码段基址，再根据符号偏移计算内部函数的实际地址：

```gdb
# 模块加载后
(gdb) p &my_module_init      # 得到模块加载基址
(gdb) b *($lx_symbol("my_driver_ioctl") )   # 计算偏移后的实际地址
```

## 串口控制台：内核的眼睛
`-nographic` 把虚拟串口重定向到当前终端，内核启动参数 `console=ttyS0` 让内核的 printk 输出走串口。这解决了内核调试最基础的痛点——内核启动早期的日志在图形控制台上稍纵即逝，串口输出则可以完整捕获、搜索、保存。

```bash
# 将串口输出同时保存到文件
qemu-system-x86_64 -kernel bzImage -initrd initramfs.cpio.gz \
  -append "console=ttyS0 nokaslr" -nographic | tee boot.log
```

对于调试 panic 场景，串口是唯一可靠的输出通道——panic 时图形栈和网络栈都可能已经损坏，但串口驱动是内核中最早初始化、最晚失效的组件。`panic=-1` 参数让内核 panic 后立即重启，配合串口日志可以自动化"反复崩溃采集"的循环。

## 内核错误捕获机制
QEMU 环境适合练习内核的各种错误捕获手段，因为可以安全地制造崩溃。

**oops 消息**：内核在非致命错误（空指针解引用、栈溢出）时打印的调用栈回溯。`oops=panic` 让 oops 升级为 panic，避免状态继续恶化。oops 输出的函数地址可以用 `scripts/faddr2line` 工具解析为源码位置。

**panic 与 kdump**：panic 是内核检测到不可恢复错误后的停机。kdump 机制在 panic 时用 kexec 启动一个专门的内核（dump 内核）来保存崩溃现场的内存镜像（vmcore），供事后用 crash 工具分析。QEMU 中练习 kdump 需要额外传递 crashkernel 参数：

```bash
-append "console=ttyS0 nokaslr crashkernel=128M"
```

**动态探针**：ftrace 和 kprobe 在运行中的内核插入探测点，不需要重新编译。kprobe 可以在任意函数的入口和返回点打印参数：

```bash
echo 'p:my_probe do_sys_open filename=+0(%si):string' > /sys/kernel/debug/tracing/kprobe_events
echo 1 > /sys/kernel/debug/tracing/events/kprobes/my_probe/enable
cat /sys/kernel/debug/tracing/trace
```

这些机制在 QEMU 中的价值是练习"如何定位问题"——真实生产环境的内核崩溃只会给你一份 vmcore 或一段 oops，在 QEMU 中熟悉了分析工具的使用，生产排障时才不会手忙脚乱。

## 调试用 initramfs
内核调试的另一个基础设施是定制 initramfs。BusyBox 编译的最小 initramfs（几 MB）启动快、行为可控，是内核启动早期调试的标准环境。对于文件系统层的调试，QEMU 的 `-drive` 挂载磁盘镜像可以模拟任意块设备行为，而 `-virtfs` 通过 9p 协议直通宿主机目录，让虚拟机内的操作直接反映到宿主机文件系统——这在内核模块开发和测试时极其方便。

```bash
# 9p 共享目录：虚拟机内 mount -t 9p hostshare /mnt
qemu-system-x86_64 -kernel bzImage -initrd initramfs.cpio.gz \
  -virtfs local,path=./shared,mount_tag=hostshare,security_model=none \
  -append "console=ttyS0 nokaslr"
```

一个值得建立的工程习惯：为内核调试维护一份专用的 QEMU 启动脚本和 initramfs——脚本固定所有参数（GDB 端口、串口重定向、调试内核参数），initramfs 预装调试工具（crash、perf、调试符号）。内核调试的频率远低于应用调试，如果不预先准备好环境，每次调试都要重新摸索参数——环境的不确定性会在最需要专注的时候消耗注意力。
