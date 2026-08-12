---
title: 虚拟化
order: 40
---

# 虚拟化
虚拟化技术是使用软件模拟硬件的技术。大型的虚拟化软件可以模拟一个完整的硬件平台，并加载指定的操作系统镜像，从而在用户态运行一个虚拟的硬件平台，又在此之上运行一个完整的操作系统，广泛用于云计算场景和软件的开发调试场景。

常见的虚拟化软件，也叫模拟器有：
+ Qemu+KVM：主要运行在 Linux 平台上模拟器，支持运行完整的操作系统，用于云计算场景；
+ Hyper-V：Windows 平台原生的模拟器，需要 Windows 专业版；
+ VMware Workstation/ESXi：前者是面向桌面场景的模拟器，主要用于 windows 生态，后者是面向企业级应用的模拟器；
+ VirtualBox：Oracle 开发的开源虚拟化工具，Type-2 虚拟化管理程序，运行在主机 OS 上，用于桌面虚拟化、开发测试等；
+ Android Emulator：安卓设备模拟器，Type-2 全虚拟，用于在 PC 开发机模拟安卓设备，进行开发测试；
+ Xcode Simulator：IOS 设备模拟器，Type-2，用于在 PC 开发机模拟 IOS 设备，进行开发测试；

### 调试与高性能 IO
虚拟化不仅是运行虚拟机的技术，也是内核调试和高性能 IO 的基础设施：

+ [QEMU 内核调试](qemu)——GDB stub、串口控制台、oops/panic/kdump 错误捕获、调试用 initramfs
+ [虚拟机调试增强](debug)——QEMU Monitor 观察物理内存、设备寄存器访问监控（guest_errors/trace-events）、与 GDB 配合
+ [SPDK 与 DPDK](spdk-dpdk)——用户态数据面技术，绕开内核 IO 路径，虚拟化高性能网络的基石