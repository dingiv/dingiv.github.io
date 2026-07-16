---
title: 开机上电
order: 10
---

# 开机上电
开机流程涉及硬件态向内核态的转化，操作系统是运行在硬件上的一个程序，其加载运行也是一个复杂的过程，一旦操作系统运行成功，应用层的软件便可以由操作系统加载和运行，乘上操作系统的快车，利用操作系统提供的 API 便捷地使用硬件完成业务需求。

## 供电
在硬件平台通电时，主板等周边设备上的小型专用处理器将优先工作。其中挂载在 CPU 总线上的小型控制器 EC（Embedded Controller）负责计算机的电源管理工作。EC 控制电源供应单元为硬件系统供电，系统各部分进入各自的供电流程，包括 CPU、内存、南北桥芯片等。

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

在硬件检查完毕后，硬件信息将会被放置在物理内存空间中的一些约定俗成的位置，BIOS/UEFI 程序尝试依次从多个外部存储设备中加载 Boot Loader 程序到内存中。如果某个设备中没有找到，则尝试从下一个存储设备加载。一旦加载成功，计算机的执行权就移交给 Boot Loader。

BIOS 加载 Boot Loader 时，从外部存储设备的启动扇区（MBR - Master Boot Record）加载，并将其放置到内存中的约定区域。BIOS 查找 Boot Loader 程序的顺序可能如下：

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

- GRUB（广泛用于 x86 平台）
- UBoot（广泛用于 arm 嵌入式开发板）
- LILO
- NTLDR
- BOOTMGR

在 Linux 文件系统中，Boot Loader 通常位于`/boot`目录下。

### boot_params 启动协议
TODO: 

## 操作系统启动：从汇编到 C
当 Boot Loader 将内核镜像加载到内存并移交控制权后，内核的启动流程正式展开。整个流程可以划分为四个阶段，从 16 位实模式逐步过渡到 64 位长模式，最终进入架构无关的通用内核初始化。

### 阶段一：实模式启动 (`arch/x86/boot/`)
内核入口 `_start` 位于 `arch/x86/boot/header.S:246`，Boot Loader 跳转到这里时 CPU 还处于 16 位实模式。`header.S` 中定义了与引导程序之间的协议头（`hdr`），包含启动参数、载荷位置等关键信息，由 `setup.ld` 链接脚本约束整个 setup 段不超过 32KB。

实模式主函数 `main()` 位于 `arch/x86/boot/main.c:133`，负责三件事：通过 BIOS E820 中断检测物理内存布局、设置视频模式、检测 CPU 特性。之后 `go_to_protected_mode()`（`pm.c:103`）开启 A20 地址线、设置 GDT/IDT、切换到 32 位保护模式。这一步的标志性动作是通过 `pmjump.S` 执行一次远跳转，CS 寄存器被加载为保护模式代码段选择子，CPU 正式进入保护模式。

### 阶段二：解压 stub (`arch/x86/boot/compressed/`)
进入保护模式后，执行流到达 `head_64.S:82` 的 `startup_32`。这个 32 位入口负责建立早期的页表结构，启用长模式（64 位）。随后 `startup_64`（行 278）作为 64 位入口，完成自定位、处理 5 级分页和 SEV 加密等初始化工作。

重定位完成后，调用 `extract_kernel()`（`misc.c:407`）执行内核解压。这个函数先通过 KASLR 随机化内核加载地址，然后根据压缩格式（gzip/xz/zstd）解压内核镜像，最后解析 ELF 格式将其加载到目标内存位置。解压完成后跳转到内核主体的入口。

bzImage 的构建过程反映了这个阶段的产物组织方式：
- 内核源码编译为 `vmlinux`（未压缩 ELF）
- strip 得到 `vmlinux.bin`，压缩为 `vmlinux.bin.gz`
- 通过 `piggy.S` 将压缩数据嵌入，与解压 stub 链接为 `compressed/vmlinux`
- `objcopy` 提取为 `vmlinux.bin`，与实模式 setup.bin 合并，最终生成 `bzImage`

### 阶段三：内核主体入口 (`arch/x86/kernel/`)
解压完成后，执行流到达 `arch/x86/kernel/head_64.S:38` 的 `startup_64`——这是内核本体的第一个入口。BSP（启动 CPU）在此设置 GDT/IDT、修正页表、配置 CR4/EFER 等控制寄存器。`common_startup_64`（行 198）是 BSP 和 AP（应用处理器）的共用路径，负责配置 per-CPU 栈和 APIC ID 查找。

`initial_code`（行 479）是一个函数指针，指向 `x86_64_start_kernel()`——内核执行的第一个 C 函数。这个函数（`arch/x86/kernel/head64.c:222`）清理早期页表、初始化 KASAN、拷贝启动参数、加载 CPU 微码，最后调用 `start_kernel()` 进入架构无关的通用初始化。

### 阶段四：通用内核初始化 (`init/main.c`)
`start_kernel()`（`init/main.c:1008`）是内核主体逻辑的入口。它按严格的顺序启动各个核心子系统：

1. `setup_arch()`：ACPI 解析、E820 内存映射建立、SMP 初始化等架构相关设置
2. `trap_init()`：中断和异常处理框架
3. `sched_init()`：调度器初始化
4. `console_init()`：控制台可用（此后内核可以通过 printk 输出日志）
5. 其他子系统：定时器、RCU、cgroup、VFS 等依次初始化

所有子系统初始化完成后，`rest_init()`（行 714）创建 PID 1（`kernel_init`，最终执行 `/sbin/init`）和 PID 2（`kthreadd`，内核线程守护者），BSP 自身退化为 idle 进程。至此，内核完成全部初始化，用户空间的第一个进程开始运行。

### 早期内存管理的建立
在 `start_kernel()` 之前，CPU 已经有了可用的页表——但这些早期页表只映射了内核镜像本身（约几十 MB），仅够内核勉强执行。`start_kernel()` 中的内存初始化子系统需要从零构建完整的内存管理基础设施。

第一步是摸清家底。`setup_arch()` 通过 BIOS 提供的 E820 表获取物理内存分布，调用 `e820__memblock_setup()` 填充 memblock 早期分配器，此后 `memblock_alloc()` 可用。接着 `init_mem_mapping()` 为全部物理 RAM 建立直接映射页表（替换早期临时页表），最后将 `swapper_pg_dir` 加载到 CR3。

第二步是建立 zone/node 结构。`free_area_init()` 将物理内存划分为 ZONE_DMA（ISA DMA 用）、ZONE_DMA32（32 位设备 DMA 用）和 ZONE_NORMAL（普通内存）三个区域。同时通过 `sparse_init()` 分配每个内存 section 的 `struct page` 数组（vmemmap），由 `memmap_init()` 逐个初始化每个 `struct page`。

第三步是启动伙伴分配器。`mm_core_init()` 调用 `memblock_free_all()` 将 memblock 中记录的空闲页全部移交给伙伴系统——这是最关键的转变：从"记录哪些物理页可用"的早期分配器切换到"按 2^n 页框管理"的伙伴系统。此后 `alloc_pages()` 和 `__get_free_pages()` 可用。紧接着 `kmem_cache_init()` 启动 SLUB 分配器（`kmalloc()` 可用），`vmalloc_init()` 启动 vmalloc 分配器。

早期页表与 `start_kernel()` 建立的页表之间存在本质差距：

| 维度 | 早期页表 | start_kernel 建立的页表 |
| ---- | -------- | ----------------------- |
| 映射范围 | 仅内核镜像（约几十 MB） | 所有物理 RAM |
| 分配能力 | 无，全靠 brk 预留缓冲区 | memblock → buddy → slab 三级 |
| zone 概念 | 无 | DMA/DMA32/NORMAL 分区 |
| struct page | 无 | 每个物理页都有元数据 |
| NUMA 感知 | 无 | 按节点组织内存 |

简单来说，早期页表是让内核能跑起来的脚手架，`start_kernel()` 要在此基础上建立起整栋内存管理的大厦。

### CR3 切换时间线
从 Boot Loader 到 `start_kernel()`，CR3 寄存器经历了多次切换，每一次都对应着页表结构的重大变化：

1. 解压器 `startup_32` 在汇编中手动构建 4GB 恒等映射的三级页表（L4→L3→L2，用 2MB 大页），写入 CR3
2. `initialize_identity_mappings()` 读取当前 CR3，复用或新建顶层页表，映射内核映像和命令行等区域
3. 内核本体 `startup_64` 调用 `__startup_64()` 修正 `early_top_pgt`（处理 KASLR 偏移），写入 CR3 切换到修正后的页表
4. `x86_64_start_kernel()` 调用 `reset_early_page_tables()` 清理早期页表中的恒等映射，构建 `init_top_pgt`（即最终的 `swapper_pg_dir`）
5. `start_kernel()` 中 `init_mem_mapping()` 为全部物理内存建立完整直接映射，最终加载到 CR3

关键的静态页表数据结构定义在 `head_64.S` 中：`early_top_pgt` 是启动过渡 PGD（仅 entry[511] 非零，指向 `level3_kernel_pgt`），`init_top_pgt` 是最终运行时 PGD（初始全零，逐步填充），`level2_kernel_pgt` 用 2MB 大页映射内核的 text+data+bss 段。最初的 PGD 由解压器汇编代码从零构建，内核本体的 `early_top_pgt` 则在编译时静态定义、启动时由 `__startup_64()` 做 KASLR 修正后加载。

## 一号进程启动
内核初始化的最后一步是 `rest_init()` 创建 PID 1。这个进程在 Linux 系统中具有极其特殊的地位——它是用户空间启动的第一个进程，由内核直接创建，始终拥有进程号 1。PID 1 负责启动系统服务、守护进程以及用户登录环境，决定了系统的运行模式和服务框架。在现代 Linux 发行版中，PID 1 通常是 systemd，它通过单元文件管理所有系统服务的启动顺序和依赖关系。