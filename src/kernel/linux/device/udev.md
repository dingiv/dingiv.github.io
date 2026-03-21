---
title: udev
order: 60
---

# udev 设备管理
udev 是 Linux 系统的设备管理器，负责管理 `/dev` 目录下的设备节点，以及处理用户空间中的设备事件。当系统添加或移除硬件设备时，内核会通过 netlink 机制通知 udev，udev 根据预定义的规则执行相应的动作，如创建设备节点、设置权限、加载模块、运行脚本等。

## 从 devfs 到 udev 的演进
在 udev 出现之前，Linux 使用 devfs 来管理设备节点。devfs 在内核空间运行，设备加入时自动在 `/dev` 目录下创建对应的设备文件。devfs 的问题在于设备节点命名不灵活，主次设备号分配混乱，而且内核态的代码难以维护。

udev 的设计将设备管理从内核空间移到用户空间，内核只负责检测设备变化和发送事件，用户空间的 udevd 守护进程负责处理这些事件。这种分离设计使得设备管理策略可以通过配置文件灵活调整，无需重新编译内核。

## udev 的工作机制
udev 系统由三个核心组件构成：内核的 sysfs 文件系统、内核的 netlink 事件机制、用户空间的 udevd 守护进程。

sysfs 是一个虚拟文件系统，挂载在 `/sys` 目录下，它将内核中的设备模型以文件和目录的形式暴露给用户空间。每个设备在 sysfs 中都有一个对应的目录，包含设备的属性信息。例如，`/sys/block/sda` 目录包含了 sda 硬盘的各种属性，如大小、型号、序列号等。

当系统检测到设备变化时，内核会通过 netlink socket 发送 uevent 事件到用户空间。这些事件包含设备的动作（add/remove/change）、设备路径、以及一些关键属性。udevd 守护进程监听这些事件，并根据规则库决定如何响应。

udevd 收到事件后，会按优先级顺序处理规则文件。规则文件位于 `/etc/udev/rules.d/` 和 `/lib/udev/rules.d/` 目录，文件名以数字开头，数字越小优先级越高。每条规则由匹配键和赋值键组成，匹配键用于筛选事件，赋值键用于指定动作。

## udev 规则语法
udev 规则的语法简洁但功能强大。一条规则可以包含多个匹配条件，所有条件都满足时才执行相应的动作。匹配键包括 `KERNEL`（内核设备名）、`SUBSYSTEM`（子系统类型）、`ATTRS`（设备属性）、`DRIVERS`（驱动名称）等。赋值键包括 `NAME`（设备节点名）、`SYMLINK`（符号链接）、`OWNER`（所有者）、`GROUP`（所属组）、`MODE`（权限模式）、`RUN`（执行程序）等。

一个典型的 udev 规则示例：
```
# 为特定 USB 设备创建固定命名的符号链接
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", SYMLINK+="ftdi_device"

# 为 U 盘设置权限和所有者
SUBSYSTEM=="block", ATTRS{idVendor}=="0951", ATTRS{idProduct}=="1666", OWNER="user", GROUP="storage", MODE="0660"

# 设备添加时执行自定义脚本
ACTION=="add", SUBSYSTEM=="net", RUN+="/usr/local/bin/configure-network.sh %k"
```

规则中的 `==` 表示匹配，`+=` 表示追加，`=` 表示赋值。`%k` 是内核提供的设备名替换变量，其他常用变量包括 `%n`（内核设备号）、`%p`（设备路径）、`%m`（总线主编号）等。

## 设备持久化命名
udev 最重要的功能之一是为设备提供持久化的命名。传统的设备命名如 `/dev/sda` 会因为设备发现顺序不同而改变，U 盘先插入可能是 sdb，后插入可能是 sdc。这对于需要稳定设备路径的应用（如自动挂载配置、备份脚本）是很大的问题。

udev 通过设备的唯一属性来创建稳定的符号链接。这些属性包括设备的序列号、厂商 ID、产品 ID、路径等。例如，以下规则会基于磁盘的 ID 创建 `/dev/disk/by-id/` 目录下的符号链接：
```
SUBSYSTEM=="block", ENV{ID_SERIAL_SHORT}=="S1XWNSDB012345", SYMLINK+="disk/by-id/$env{ID_SERIAL_SHORT}"
```

现代发行版已经预置了大量规则，用户可以在 `/dev/disk/by-uuid/`、`/dev/disk/by-label/`、`/dev/disk/by-path/` 等目录下找到各种持久化命名的设备链接。

## 网络设备命名
udev 也负责网络设备的命名。传统的 eth0、eth1 命名方式因设备发现顺序不稳定而被 systemd 的可预测命名方案取代。新方案根据设备的物理位置、固件信息、MAC 地址等生成稳定的设备名，如 `enp3s0`（PCI 总线 3 插槽 0 的以太网设备）、`wlp2s0`（PCI 总线 2 插槽 0 的无线设备）。

如果需要恢复传统的 eth0 命名，可以通过 udev 规则实现：
```
SUBSYSTEM=="net", ACTION=="add", DRIVERS=="?*", ATTR{address}=="00:11:22:33:44:55", NAME="eth0"
```

或者在内核启动参数中添加 `net.ifnames=0` 来禁用可预测命名。

## udevadm 工具

udevadm 是 udev 的管理工具，提供了调试和信息查询功能。`udevadm info` 可以查询设备的属性信息：
```bash
# 查询 sda 设备的所有属性
udevadm info --attribute-walk --name=/dev/sda

# 查询某个设备节点的详细信息
udevadm info --query=all --name=/dev/sda
```

`udevadm monitor` 可以实时监控内核发送的 uevent 事件：
```bash
# 监控所有 uevent
udevadm monitor

# 只监控内核层事件
udevadm monitor --kernel

# 只监控 udev 层事件
udevadm monitor --udev
```

`udevadm test` 可以模拟设备事件并测试规则，而不实际执行动作：
```bash
# 测试添加 sda 时的规则处理
udevadm test /sys/block/sda
```

`udevadm control` 用于控制 udevd 守护进程：
```bash
# 重载规则配置
udevadm control --reload-rules

# 触发所有设备的事件
udevadm trigger

# 触发特定子系统的设备事件
udevadm trigger --subsystem-match=net
```

## 内核模块自动加载
udev 还负责在检测到新设备时自动加载相应的内核模块。当内核检测到设备但缺少驱动时，会发送 uevent 事件，udevd 根据设备的 `MODALIAS` 属性查找匹配的内核模块并加载。

模块别名信息存储在 `/lib/modules/$(uname -r)/modules.alias` 文件中，这个文件在内核模块安装时由 depmod 工具生成。如果需要禁用某个设备的自动加载，可以在 `/etc/modprobe.d/` 目录下创建 blacklist 配置：
```
# 禁止自动加载 pcspkr 模块
blacklist pcspkr
```

## 与 systemd 的集成
在现代 Linux 系统中，udevd 已经被集成到 systemd 中，成为 systemd-udevd 服务。systemd 通过 udev 规则来管理设备，同时利用设备事件来触发服务启动。例如，当某个 USB 网卡插入时，systemd 可以自动启动网络配置服务；当某个存储设备插入时，可以自动启动备份服务。

这种设备事件驱动的服务启动机制，使得系统可以根据实际硬件情况动态调整服务状态，实现了更加智能和高效的资源管理。

## 故障排查
udev 相关的故障通常表现为设备节点未创建、权限不正确、规则不生效等问题。排查时首先查看 udevd 日志：
```bash
journalctl -u systemd-udevd
```

使用 `udevadm monitor` 确认内核是否发送了设备事件，如果没有事件，问题可能在内核驱动层。如果有事件但规则未生效，使用 `udevadm test` 查看规则匹配过程，检查规则语法和优先级。

设备属性信息是编写规则的基础，使用 `udevadm info --attribute-walk` 可以获取设备的完整属性树，帮助找到稳定的匹配键。
