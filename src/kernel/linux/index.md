---
title: Linux
order: 20
---

# Linux
理论课的知识终究是基于宽泛的普适讨论，在实际的生产中，往往会基于已有的操作系统进行定制，而不是从头开始编写，利用现有的 linux 实现可以极大加速系统层开发，同时复用 linux 平台上的应用层软件生态。

## 内存管理

### 物理内存

主板和BIOS程序在上电时：
1. 检查和扫描设备
2. 初始化各种设备（内存条和各种IO设备的寄存器和缓冲区）
3. 将这些存储空间拼接成连贯的物理内存空间

物理内存空间可以看作是一个地址数组，每个地址的大小取决于计算机的位数（32位、64位等）。在操作系统引导时，BIOS将物理内存空间信息告知操作系统，包括：
- 内存地址的分区
- 各个硬件设备的地址范围

### 虚拟内存

操作系统为进程提供虚拟内存，每个应用程序进程都认为自己独占全部、连续的内存。在64位系统上：
- 虚拟内存大小为256TB
- 地址范围：0x0000000000000000 到 0x0000FFFFFFFFFFFF

### 内存映射

物理内存和虚拟内存之间存在映射关系：
- 映射以一块连续的内存为单位（通常为4000 Byte）
- 一块虚拟内存对应一块真实的物理内存
- 这个单位称为内存分页
- 操作系统通过页表（多级数组数据结构）维护映射关系

当进程创建时，操作系统为进程创建页表，维护进程的虚拟内存到物理内存的映射。当应用程序通过系统调用进行内存分配时：
1. 调用操作系统的封装函数
2. 操作系统分配物理内存
3. 执行内存映射操作
4. 返回指向虚拟内存的指针

### 自定义内存映射
```c
#include <sys/mman.h>
/**
 * @param addr 可以是 NULL，由操作系统自行分配
 * @param fd 需要映射的文件
 * @param offset 偏移量
 */
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
```

mmap函数将文件fd中的内容映射到当前进程的内存空间中addr位置处，大小为length，通过prot参数控制内存访问权限，通过flags提供更多配置选项。

## IO管理

IO设备拥有独立的控制处理器，现代IO设备通过MMIO方式将自身寄存器地址空间映射到物理内存空间中，让CPU通过直接读写物理地址空间来控制IO设备。

### 设备挂载

1. 设备识别和驱动加载
   - 识别设备：操作系统检测设备并分配设备文件（Linux中通常位于`/dev/`目录下）
   - 加载驱动：操作系统加载适当的驱动程序支持设备操作

2. 设备格式化
   - 存储设备需要经过格式化才能使用
   - 格式化将物理存储空间划分为存储区域
   - 为这些区域建立文件系统
   - 未格式化的存储设备不能直接存储文件和数据

   文件系统格式化：
   - 文件系统是操作系统管理磁盘上文件的方式
   - 不同操作系统使用不同的文件系统格式（ext4、NTFS、FAT32、exFAT等）
   - 分区表（MBR或GPT）定义设备上不同部分的布局和大小

   例如，在Linux中格式化磁盘分区：
   ```bash
   sudo mkfs.ext4 /dev/sda1
   ```

3. 挂载存储设备
   - 格式化后，存储设备的文件系统才可用
   - 挂载操作将设备上的文件系统与操作系统的目录结构连接
   - 用户可以通过路径访问存储设备的内容

   在Linux中挂载设备：
   ```bash
   sudo mount /dev/sda1 /mnt
   ```

4. 文件系统检查与修复
   - 文件系统可能因突然断电或设备损坏而不一致
   - 操作系统执行文件系统检查（fsck）修复问题

   在Linux中手动运行文件系统检查：
   ```bash
   sudo fsck /dev/sda1
   ```

5. 挂载配置（可选）
   - 可以将存储设备配置为系统启动时自动挂载
   - 通过编辑`/etc/fstab`文件完成配置

   例如，添加以下行将设备`/dev/sda1`挂载到`/mnt`：
   ```bash
   /dev/sda1 /mnt ext4 defaults 0 2
   ```

### IOMMU

IOMMU（IO设备内存空间管理单元）：
- 在一些硬件平台上支持IOMMU技术
- 添加IOMMU单元，在CPU访问物理内存地址时添加类似MMU的内存虚拟技术
- 针对IO设备
- 通常伴随DMA一同出现

### Socket套接字

Socket是操作系统提供的跨进程通信底层抽象：
- 基于bind、listen、accept等操作
- 实现对网络通信的封装
- 让上层能够使用C函数方便地调用

服务器端代码示例：
```c
// 创建套接字
int server_fd = socket(AF_INET, SOCK_STREAM, 0);

// 绑定套接字到本机地址
struct sockaddr_in address;
address.sin_family = AF_INET;
address.sin_addr.s_addr = INADDR_ANY; // 监听所有接口
address.sin_port = htons(PORT); // 绑定端口
int ret = bind(server_fd, (struct sockaddr*)&address, sizeof(sockaddr_in));

// 开始监听
listen(server_fd, 3);

// 接收客户端请求
int connect_fd;
while(1) {
    // 阻塞直到有客户端连接
    connect_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addr_len);

    char recv_buf[1024];
    ssize_t bytes_received = 0;
    while(1) {
        // 接收数据
        bytes_received = recv(connect_fd, recv_buf, sizeof(recv_buf), 0);
        // 也可以使用通用的文件描述符操作函数
        // bytes_received = read(connect_fd, recv_buf, sizeof(recv_buf));
    }
}
```

### IO 多路复用

Linux系统通过select、poll、epoll提供系统级别的事件监听机制：
- select：通过fd_set结构维护文件描述符集合
- 每次调用select函数会阻塞
- 内核循环遍历fd_set，监听文件描述符变化
- 找出可以处理的文件描述符进行处理

## 中断（Interrupt）

中断是一种信号，由硬件或软件发出，请求CPU停止当前操作，转而响应特定事件。该机制广泛用于IO设备和CPU的交互中。

### 中断的作用

1. 多任务处理
2. 资源管理
3. 事件驱动：响应IO设备的中断，实现监听IO事件

### 中断的类型

1. 硬件中断
   - IO中断（磁盘读写完成）
   - 定时器中断
   - 外设中断（鼠标键盘操作）

2. 软件中断
3. 异常

### 中断的流程

1. 触发中断：硬件或软件触发中断信号
2. 保存现场：CPU停止当前任务，保存寄存器和程序计数器到堆栈
3. 查找中断向量：CPU根据中断向量表IVT查找中断处理程序地址
4. 执行中断处理程序：CPU转到中断处理程序，完成特定任务
5. 恢复现场：执行完成后恢复现场，继续之前任务

## 万物皆文件

"一切皆文件"是UNIX的著名哲学理念。在Linux中：
- 具体文件、设备、网络socket等都可以抽象为文件
- 内核通过虚拟文件系统（VFS）提供统一界面
- 程序可以通过文件描述符fd调用IO函数访问文件
- 应用程序可以调用select、poll、epoll等系统调用监听文件变化

常见的IO函数：
- open
- read
- write
- ioctl
- close

## 内核模块

Linux是可扩展架构，允许动态加载和卸载功能模块：
- 包括设备驱动程序、文件系统、网络协议栈等
- 动态加载通过insmod/modprobe命令
- 卸载通过rmmod命令

内核模块加载后：
- 符号空间中的函数不会被加入到操作系统运行的物理内存中
- 通过EXPORT_SYMBOL宏导出函数
- 只有通过该宏显式导出的函数才能在模块外部可见

> 内核模块加载区别于动态链接库加载：
> - 内核模块加载是将二进制目标文件加载到物理内存中
> - 动态链接库由系统在创建进程时加载到虚拟内存中

## 文件管理

Linux至少需要一个存储设备来建立文件系统：
- 对数据文件进行持久化
- 满足"一切皆文件"的设计哲学
- 最小 Linux 机器实例需要挂载硬盘或基于内存的假文件系统

### VFS（虚拟文件系统）

虚拟文件系统的作用：
- 实现UNIX环境下"一切皆文件"的具体方式
- 为用户空间提供树形结构的文件目录结构
- 让用户通过文件路径访问系统资源
- 所有资源都被抽象成文件
- 用户态程序可以使用统一的操作文件接口

### 文件系统

文件系统用于描述磁盘中数据的组织方式和结构：
- 磁盘在挂载到虚拟文件系统前需要格式化
- 格式化过程就是建立文件系统的过程

> 注意区分文件系统和VFS：
> - 文件系统用于管理和描述块设备中的数据
> - VFS是Linux中的文件结构抽象

## 硬件交互

### CPU权限级别

现代处理器运行时分为4个权限级别：
- ring0：最高权限，直接访问硬件资源，运行操作系统代码
- ring1
- ring2
- ring3：最低权限，只能通过间接调用系统代码访问硬件资源，运行用户应用程序代码或用户态库

### IO设备

操作系统内核不直接操作设备：
- 通过调用设备的驱动程序完成对设备的读写
- 驱动程序由硬件厂商实现
- 使用设备时，将驱动程序以内核模块方式加载进内核
- 操作系统声明统一的SPI（Service Program Interface），由硬件厂商实现

### 可编程IO/DMA

根据数据传输过程是否需要CPU参与，IO分为两类：

1. 可编程IO
   - 操作系统通过读写IO设备寄存器控制设备
   - 分为两种类型：
     * PMIO（Port-Mapped Input/Output）：通过访问IO端口控制设备
     * MMIO（Memory-Mapped Input/Output）：设备寄存器和缓冲区映射到物理内存中

2. DMA（Direct Memory Access）
   - 外设与内存之间交换数据的接口技术
   - 数据传输过程无须CPU控制
   - 数据拷贝和搬运由外设专用处理器完成
   - 操作系统通过驱动程序提前告知外设数据拷贝位置
   - 外设直接访问内存，将数据放到指定位置
   - 完成后发起中断通知CPU

## 服务

服务是后台运行或定时运行的任务，帮助系统或其他进程更好地完成任务。

### 系统服务

系统服务软件：
- 不属于内核层代码
- 运行在用户空间中
- 协助用户与系统内核交互
- 为用户态程序提供基础设施服务

典型软件包括：
- systemd（服务管理、journald日志服务、udev设备管理）
- bash
- iproute
- grub

多数Linux发行版自带systemd：
- 配置文件位于`/etc/systemd`、`/usr/lib/systemd`、`~/.config/systemd`等文件夹
- 使用systemctl命令操作
- 使用journalctl命令管理日志内容

### 服务单元

在systemd抽象中：
- 独立运行的进程服务使用service文件描述
- 称为服务单元
- 规定服务进程的启动配置
- systemd根据配置文件管理和启动服务
- 通过systemctl和journalctl接口管理和控制系统上注册的服务程序
