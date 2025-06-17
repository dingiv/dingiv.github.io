# systemd
systemd 是 linux 上的一号进程，用于在应用层提供基础的系统服务，简化应用层的使用流程。

## 服务管理

## 集中化日志管理

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
