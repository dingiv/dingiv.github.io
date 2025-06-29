# systemd
systemd 是 Linux 系统中现代化的初始化系统和服务管理器，作为一号进程（PID 1）运行，负责在系统启动时初始化用户空间，并管理系统的各类服务、进程、挂载点、设备等。systemd 的设计目标是提升系统启动速度、统一服务管理接口、增强系统的可维护性和可扩展性。

## 服务
服务是后台运行或定时运行的任务，帮助系统或其他进程更好地完成任务。

系统服务软件：
- 不属于内核层代码
- 运行在用户空间中
- 协助用户与系统内核交互
- 为用户态程序提供基础设施服务

典型软件包括：
- systemd（服务管理、journald 日志服务、udev 设备管理）
- bash
- iproute
- grub

多数 Linux 发行版自带 systemd：
- 配置文件位于`/etc/systemd`、`/usr/lib/systemd`、`~/.config/systemd`等文件夹
- 使用 systemctl 命令操作
- 使用 journalctl 命令管理日志内容

## 服务单元
systemd 采用并行化的服务启动方式，利用服务之间的依赖关系图，最大化地并发启动各项服务，从而显著缩短系统启动时间。它以“单元（Unit）”为核心抽象，将服务、挂载点、设备、套接字、计时器等都统一为不同类型的单元进行管理。systemd 还集成了日志管理（journald）、设备管理（udev）、网络管理（networkd）等多种功能，极大地简化了系统管理流程。

与传统的 SysV init 相比，systemd 提供了更强的依赖管理能力、更细粒度的控制接口、更丰富的状态监控和日志功能。它支持按需启动（socket/DBus 激活）、服务自动重启、资源限制（cgroups）、快照与恢复等高级特性。

在 systemd 抽象中，独立运行的进程服务使用 service 文件描述，称为服务单元。单元配置规定服务进程的启动配置，systemd 根据配置文件管理和启动服务。然后，用户态程序通过 systemctl 和 journalctl 接口管理和控制系统上注册的服务程序。

systemd 支持多种单元类型（Unit Type），包括：
- service：后台服务进程（如 sshd、nginx）
- socket：套接字激活单元，实现按需启动服务
- target：运行级别分组，类似于 SysV 的 runlevel
- mount：文件系统挂载点
- automount：自动挂载点
- timer：定时任务单元，替代 cron
- device：内核设备单元
- path：文件或目录监控单元
- swap：交换分区/文件单元

## 启动流程
1. 内核启动后，执行 systemd：内核完成初始化后，将控制权交给 `/sbin/init`，大多数现代发行版的 `/sbin/init` 实际上是 systemd。
2. systemd 解析配置文件：systemd 读取 `/etc/systemd/system/`、`/usr/lib/systemd/system/` 等目录下的单元配置文件，构建服务依赖关系图。
3. 并发启动单元：根据依赖关系，systemd 并发启动各类服务单元（如网络、挂载点、日志、用户会话等）。
4. 进入多用户/图形界面目标：systemd 启动所有必要服务后，系统进入多用户（multi-user.target）或图形界面（graphical.target）运行级别。
5. 持续管理与监控：systemd 持续监控服务状态，处理服务崩溃、自动重启、日志收集等任务。

每个单元通过配置文件（如 `*.service`、`*.socket`）描述启动命令、依赖关系、环境变量、资源限制等。systemd 通过 `Requires`、`Wants`、`Before`、`After` 等指令精确控制单元的启动顺序和依赖关系。

## 服务管理
systemd 向用户态提供了统一的服务管理命令 `systemctl`，用于启动、停止、重载、查看状态、设置开机自启等操作。例如：

```bash
systemctl start nginx.service      # 启动服务
systemctl stop nginx.service       # 停止服务
systemctl restart nginx.service    # 重启服务
systemctl status nginx.service     # 查看服务状态
systemctl enable nginx.service     # 设置开机自启
systemctl disable nginx.service    # 取消开机自启
```

systemd 支持服务的自动重启、资源限制（如内存、CPU 限额）、日志集成、依赖管理等高级功能。管理员可以通过 `systemctl` 轻松管理系统和用户服务。

## 日志管理
systemd 集成了 journald 日志服务，统一收集内核日志、服务日志和用户日志。日志以二进制格式存储，支持高效检索和过滤。通过 `journalctl` 命令可以方便地查看、筛选和分析日志。例如：

```bash
journalctl -u nginx.service    # 查看 nginx 服务日志
journalctl -b                 # 查看本次启动以来的所有日志
journalctl --since "1 hour ago"  # 查看最近一小时日志
```

journald 还支持日志持久化、日志轮转、远程日志收集等功能，极大提升了系统运维的可观测性。它会定期读取内核日志，并将内核日志输出到位置 `/var/log/syslog` 文件中；同时它也向用户态的程序提供了一个 `syslog` 函数接口，用户态程序可以选择使用这个函数来使用提供的日志打印系统和服务。syslog 函数会将日志消息发送给 systemd-journald 守护进程，具体的日志管理由它完成。

一般的内核日志，使用 `printk` 函数打印日志或者驱动可使用 `dev_printk`（自动携带设备信息）；而用户态日志需要自行实现日志系统或者可使用 systemd-journald 服务提供的接口。

## 特殊性
在 Linux 系统中，一号进程（PID 1）具有极其特殊和重要的地位。它是内核启动后创建的第一个用户空间进程，通常由 systemd 担任。作为系统的根进程，一号进程承担着初始化用户空间环境、启动和管理所有系统服务、维护系统运行级别等核心职责。

一号进程还有一个关键作用：它是所有孤儿进程的“收养者”。当系统中其他进程的父进程意外终止时，这些进程会被内核自动转交给一号进程，由其负责资源回收和善后处理。这一机制保证了系统中不会出现无人管理的僵尸进程，维护了进程表的整洁和系统的稳定性。

此外，一号进程的健壮性直接关系到系统的稳定运行。如果一号进程崩溃或退出，内核会认为系统处于不可恢复的状态，通常会触发内核 panic 或自动重启。因此，systemd 作为一号进程，其健壮性和可靠性对整个 Linux 系统的持续运行至关重要。