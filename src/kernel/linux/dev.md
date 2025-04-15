# Linux 开发笔记

## 开发库

### Glib

Glib 是由 GNU 书写的 C 语言通用跨平台库，广泛应用于 C 语言程序中。它提供 C 标准库所没有的很多上层 API 封装，简化了 C 语言的开发和使用，包括：
- 内存管理
- 字符串处理
- 常用数据结构
- 文件管理
- 锁机制

### Klib (Linux Kernel Library)

Linux 下的程序编译环境默认包含 `/usr/include` 文件夹下的头文件：
- `/usr/include/linux`：用于编译和开发内核模块
- `/usr/include/sys`：提供与 Linux 系统强相关的函数库

系统调用机制：
- 用户态程序通过包含 `sys/xxx.h` 进行系统调用
- 通过 `sys/ioctl.h` 提供的文件操作接口与内核模块交互
- 系统调用函数在进程实例化时由系统自动加载到内核空间

### 常用第三方库

1. 网络相关
   - libcurl：HTTP 客户端库
   - OpenSSL：加密和安全通信
   - libevent：事件驱动网络库
   - libuv：跨平台异步 I/O 库

2. 数据处理
   - SQLite：轻量级数据库
   - zlib：数据压缩
   - libxml2：XML 处理
   - jansson：JSON 处理

3. 多媒体
   - FFmpeg：音视频处理
   - libpng：PNG 图像处理

## 代码实践

### 函数设计原则

1. 函数纯洁性
   - C 程序对 I/O 操作有容忍性
   - 多数函数允许使用 I/O 操作

2. 参数传递
   - 推崇"改参函数"模式
   - 函数返回值通常为 int 类型，表示操作是否成功
   - 实际结果通过参数中的指针返回
   - 内存分配由用户决定（栈或堆）

3. 内存管理
   - 动态内存分配需要传递指针的指针
   - 函数通过参数返回动态分配的内存

4. 编程规范
   - 减少全局状态引用
   - 检查所有返回 int 结果的函数
   - 验证所有接收指针的参数的合法性

### 字节序处理

1. 基本概念
   - 内存和数据流被抽象为字节数组
   - 数据编码需要先转换为 16 进制
   - 内存地址从低到高：0x00000000 -> 0xffffffff

2. 字节序转换
   - 不同平台 CPU 的大小端序不同
   - 网络通信和驱动编写需要固定字节序
   - 使用转换函数处理字节序：
     ```c
     cpu_to_le32()
     cpu_to_le64()
     ```

## 常用工具

### 文件操作
- echo：输出文本
- cat：查看文件内容
- tee：同时输出到文件和屏幕
- vi/vim/nano：文本编辑器
- tar/unzip/cpio/7z：压缩解压
- find：文件查找

### 磁盘管理
- fdisk：磁盘分区
- fio_libaio：异步 I/O 测试
- dd：数据复制
- lsblk/df/du：块设备/磁盘空间管理

### 包管理
- apt：Debian/Ubuntu 包管理
- yum/dnf：RHEL/CentOS 包管理
- rpm：RPM 包管理

### 网络工具
- ip：综合性网络管理工具
- iptables/nftables：netfilter 模块管理
- tc：流量控制
- tcpdump：网络抓包
- nc/socat：Socket 客户端/服务端
- iperf3：网络性能测试
- ss/netstat：Socket 统计
- nmap：网络扫描

### 内核工具
- dmesg：查看内核缓冲区消息
- lspci：PCI 设备信息

### 系统服务
- systemctl：服务管理
- journalctl：日志管理
- top：资源监控

## 网络编程

### Socket 编程

#### TCP 服务端流程
1. 创建 Socket
2. 绑定地址
3. 监听连接
4. 接受连接
5. 读写数据

#### TCP 客户端流程
1. 创建 Socket
2. 连接服务器
3. 读写数据

### Scatter/Gather I/O

Scatter/Gather I/O（分散-聚集 I/O）是一种高效的数据传输技术：
- 避免不必要的内存拷贝
- 减少 CPU 负担
- 提升 I/O 性能

实现方式：
1. Scatter Read：从连续数据源读取，分散存储到多个内存块
2. Gather Write：从多个内存块读取，一次性写入连续目标

## 内核开发

### 开发环境
- 使用 QEMU 模拟器进行跨平台硬件模拟
- 支持不同架构下的内核开发和调试

### 系统调用
- 通过 `sys/syscall.h` 扩展系统调用接口
- 使用 `syscall` 函数，参数为系统调用号（`SYS_xxx`）

## 日志管理

### 内核日志
- 使用 `printk` 函数打印日志
- 驱动可使用 `dev_printk`（自动携带设备信息）

### 用户态日志
- 需要自行实现日志系统
- 可使用 systemd-journald 服务

### Systemd 日志管理这是一个常常伴随系统安装的用户态进程，用于管理机器上服务，同时也自带了一个日志模块-systemd-journald，它会定期读取
内核日志，并将内核日志输出到位置ソvar／log／syslog｀文件中；同时它也向用户态的程序提供了一个＇syslog 函数接口，用户态程序可以选择使用这个函数来使用提供的日志打印系统和服务。syslog 函数会将日志消息发送给 systemd-journald 守护进程，具体的日志管理由它完成。
- systemd-journald 定期读取内核日志
- 输出到 `/var/log/syslog`
- 提供 `syslog` 函数接口
- 使用 `journalctl` 查看日志：
  ```bash
  journalctl -n 100  # 查看最近 100 条日志
  ```