# 开机流程
开机流程涉及硬件态向内核态的转化，操作系统是运行在硬件上的一个程序，其加载运行也是一个复杂的过程，一旦操作系统运行成功，应用层的软件便可以由操作系统加载和运行，乘上操作系统的快车，利用操作系统提供的 API 便捷地使用硬件完成业务需求。

## 供电
在系统通电时，主板等周边设备上的小型专用处理器将优先工作。其中挂载在 CPU 总线上的小型控制器 EC（Embedded Controller）负责计算机的电源管理工作。EC 控制电源供应单元为硬件系统供电，系统各部分进入各自的供电流程，包括 CPU、内存、南北桥芯片等。

> 周边设备的控制器芯片不承担主要的计算任务，但会帮助协调各个硬件之间的工作。

各个硬件的供电顺序有所先后，一般顺序是：CPU、内存、南北桥、扩展卡和外围设备、启动存储等。电源控制单元在确保电源供应一切顺利后，向 EC 发出信号表示供电完毕。供电成功后，EC 通知主板上的各个芯片组，正式开启计算机的启动流程。

首先，南北桥进行交互：

1. 南桥向北桥发出正常信号
2. 北桥收到南桥信号，并向 CPU 发送正常信号
3. CPU 开始工作

> 供电单元可以进行交流电到直流电的转化，并保持电源电压的稳定。

## BIOS/UEFI 寻找 Boot Loader
CPU 开始工作后的第一个程序是 BIOS/UEFI 程序，它是主板上的固件。BIOS 程序会：

1. 扫描和检查设备
2. 初始化硬件系统（主板、内存、CPU、显卡等）
3. 确保设备能够正常工作（POST - Power-On Self Test）
4. 为内存条和 MMIO 设备分配地址空间
5. 将地址空间连成连续的数组空间（地址位数取决于机器位宽，通常为 32 位或 64 位）

在硬件检查完毕后，BIOS/UEFI 程序尝试依次从多个外部存储设备中加载 Boot Loader 程序到内存中。如果某个设备中没有找到，则尝试从下一个存储设备加载。一旦加载成功，计算机的执行权就移交给 Boot Loader。

BIOS 加载 Boot Loader 时，从外部存储设备的启动扇区（MBR - Master Boot Record）加载，并将其放置到内存中的约定区域。

BIOS 查找 Boot Loader 程序的顺序可能如下：

- 硬盘
- USB
- CDROM
- 网卡（pxe 启动）

BIOS 通常提供基于终端的配置界面，允许用户自定义引导过程，例如修改启动介质的优先级。

> 一些硬件层面的功能（如网卡的 SR-IOV 功能）不一定会被启用。如需启用，需要重启机器并提前修改 BIOS 配置。

## Boot Loader 加载操作系统
Boot Loader 是存放在操作系统镜像盘中约定区域的一段程序。它被 BIOS 程序加载并执行，负责：

1. 把操作系统的文件加载到内存中
2. 初始化操作系统
3. 将计算机的执行权移交给操作系统

> Boot Loader 执行时，CPU 处于 Real 模式，只能访问 1MB 的内存空间，没有内存保护。Boot Loader 利用硬盘的分区表、文件系统信息和操作系统核心文件，实现从实模式到保护模式的切换，以及从硬盘到内存的数据传输。

Boot Loader 根据 MBR 中的磁盘分区信息，找到活动分区（操作系统文件所在的分区），然后：

1. 找到操作系统可执行文件
2. 加载操作系统到指定的内存区域
3. 跳转 CPU 到该区域，开始执行操作系统

常见的 Boot Loader 实现包括：

- GRUB（广泛用于 Linux 系统）
- LILO
- NTLDR
- BOOTMGR

在 Linux 文件系统中，Boot Loader 通常位于`/boot`目录下。

## 操作系统启动
以 Linux 系统为例，当系统启动时：

1. BIOS/UEFI 程序在上电时对设备进行扫描检查和初始化
2. 为内存条和 MMIO 设备分配地址空间
3. 将地址空间连成连续的数组空间
4. 操作系统接管硬件管理
