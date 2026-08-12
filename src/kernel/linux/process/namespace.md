---
title: 命名空间
order: 30
---

# 命名空间
Namespace 是 Linux 容器技术的基石。它的核心思想朴素而深刻：内核资源是全局的，但可以让每个进程看到一套**裁剪过的视图**——进程以为自己是 PID 1，以为自己是 root，以为拥有整张网络接口列表，而实际上这些视图都是内核为它专门构造的假象。

从内核实现的角度看，Namespace 做的事情是把六类全局资源"按命名空间切分"：进程号空间、网络栈、挂载点、主机标识、用户权限、IPC 对象。每个进程通过 `task_struct` 中的 `nsproxy` 指针关联到一组命名空间——这个结构体里存放的七个指针（对应七种命名空间）共同定义了进程眼中的世界。

## 六大类资源隔离

### 进程号空间：PID Namespace
PID Namespace 隔离的是进程编号的可视范围。没有 PID Namespace 之前，全局只有一个 PID 空间，所有进程按创建顺序分配唯一编号。引入 PID Namespace 后，每个命名空间内的进程从 1 开始独立编号——容器内那个"PID 1"在宿主机上可能实际编号 18472。

内核实现的核心是一个层级化的 `pid_namespace` 结构。每个命名空间记录自己内部的编号分配，同时通过 `parent` 指针链接到上层命名空间——一个进程在每个祖先命名空间中都有对应的编号。`getpid()` 系统调用返回当前命名空间内的编号，而宿主机的 `ps` 看到的是初始命名空间内的编号。

有两个实现细节值得注意。一是 PID 1 的特殊性：每个 PID Namespace 内的 init 进程负责收养孤儿进程——init 退出时，内核向命名空间内所有进程发送 SIGKILL（`pidns->reboot` 机制），整个命名空间被强制清理。这正是容器内"主进程退出则容器退出"的内核根源。二是信号发送的权限限制：父命名空间的进程可以向子命名空间内的进程发信号（通过 `kill` 系统调用带 PID 转换），反之则不允许——这是跨命名空间信号隔离的规则。

### 网络栈：Network Namespace
Network Namespace 隔离的是整套网络栈——不只是网络接口，还包括路由表、iptables 规则、ARP 表、socket 端口空间。每个 Network Namespace 拥有独立的 `struct net` 实例，内核网络子系统的所有全局变量在这个结构体中都有对应副本。

工程上最有价值的细节是 **lo 接口在命名空间间不共享**。新创建的 Network Namespace 默认只有 down 状态的 loopback——容器网络的第一步永远是"把 lo 拉起来"。宿主机与容器通信通过 veth pair：veth 是一对逻辑相连的网卡，一端留在宿主机命名空间（通常挂到 bridge 上），另一端移入容器命名空间，两者之间的流量就是容器进出网络的通道。

端口空间的隔离是 Network Namespace 最直观的收益——两个容器可以各自监听 80 端口互不冲突，因为它们的 `bind` 查找发生在各自的 `struct net` 的端口哈希表中。

### 挂载视图：Mount Namespace
Mount Namespace 隔离的是挂载点列表。每个命名空间维护自己的挂载树（内核 `struct mount` 的独立实例），`mount`/`umount` 操作只影响当前命名空间。

Mount Namespace 是容器镜像分层机制的根基。容器启动时，运行时在当前命名空间中挂载容器的 rootfs，配合 OverlayFS 把镜像层叠加为一个合并视图——这个"根目录"只存在于容器的 Mount Namespace 中，宿主机看到的仍是自己的根目录。`pivot_root` 系统调用在这里发挥作用：它把容器的 rootfs 挂载变为命名空间的根，同时把宿主机的旧根目录移走——彻底切断容器对宿主机目录树的可见路径。

Mount Namespace 也是传播机制（propagation）的试验场：`mount --make-shared/private/slave` 控制挂载事件在命名空间间的传播方向。Docker 卷挂载的某些行为（宿主机挂载在容器内可见，容器内挂载在宿主机不可见）就依赖这个机制。

### 主机标识：UTS Namespace
UTS Namespace 隔离的是主机名（hostname）和 NIS 域名——通过 `sethostname` 和 `setdomainname` 系统调用设置的标识。它是六类命名空间中实现最简单的（`struct uts_namespace` 只有两个字符串字段），但意义不小：没有它，容器内修改主机名会影响宿主机，而且 `hostname` 命令的隔离感是容器"独立系统"体验的一部分。

### 用户与权限：User Namespace
User Namespace 是六类命名空间中最后加入、也是安全影响最大的。它隔离的是 UID/GID 的映射关系：命名空间内的 UID 0（root）可以映射为宿主机上的非特权 UID（如 1000）。

实现上，每个进程的凭证（`struct cred`）在 User Namespace 的上下文中解释。容器内进程以 UID 0 运行——拥有该命名空间内全部 capabilities——但宿主机视角它只是普通用户 1000。**Capabilities 的作用域也被 User Namespace 限制**：进程持有的 capability 只在拥有它的 User Namespace 及其子命名空间内有效——容器内 root 的 `CAP_SYS_ADMIN` 不能操作宿主机的任何资源。

这是 rootless 容器（Podman rootless、Docker rootless mode）的技术基础：普通用户通过 `unshare -U` 创建 User Namespace 让自己在其中成为 root，再在这个命名空间内创建其他命名空间——全程不需要宿主机 root 权限。

### 进程间通信：IPC Namespace
IPC Namespace 隔离的是 System V IPC 对象（共享内存段 `shmget`、信号量 `semget`、消息队列 `msgget`）和 POSIX 消息队列。这些对象的命名空间是全局 ID 分配的——`shmget(key)` 中的 key 在各命名空间独立解析，一个命名空间内创建的共享内存段对另一个命名空间完全不可见。

IPC 隔离对容器安全有实际意义：不隔离时，宿主机和容器之间可能通过已知 key 的共享内存段意外共享数据。IPC Namespace 关闭了这个通道。

### 两种较新的命名空间
Cgroup Namespace 隔离的是 `/proc/self/cgroup` 和 `/sys/fs/cgroup` 的视图——容器内的进程看到自己的 cgroup 路径是 `/`，而非宿主机上的完整路径。这是 2016 年（内核 4.6）加入的，动机是防止容器内的进程通过 cgroup 路径推断宿主机布局。

Time Namespace（内核 5.6）隔离的是 CLOCK_MONOTONIC 和 CLOCK_BOOTTIME——允许容器拥有独立的单调时钟起点。这对容器快照/恢复（CRIU）有意义：恢复的容器不应看到快照期间的时钟跳变。

## 命名空间的生命周期

### 创建：clone 与 unshare
创建命名空间的两个系统调用分工明确。`clone` 在创建新进程的同时创建命名空间——`clone(CLONE_NEWPID | CLONE_NEWNET, ...)` 一次调用产生一个处于新命名空间中的新进程，容器运行时的 init 进程就是这样创建的。`unshare` 不创建进程，把调用者自身移入新命名空间——`unshare -n` 让当前 shell 获得独立网络栈。

一个容易被误解的细节：**PID Namespace 通过 clone 创建时，新进程是命名空间内的 PID 1，但通过 unshare 创建时调用者不是 PID 1**。unshare 后第一个 fork 的子进程才成为 PID 1。这意味着 unshare 方式创建 PID Namespace 时必须立即 fork——否则命名空间内没有 init 进程，后续进程无法正常创建。

### 关联：setns
`setns(fd, nstype)` 让调用进程加入一个已存在的命名空间，参数是 `/proc/<pid>/ns/<type>` 的文件描述符。这是 `docker exec` 进入容器、`nsenter` 工具的实现基础——这些工具打开目标进程的命名空间文件，然后 setns 进去。

`/proc/<pid>/ns/` 目录下的文件是命名空间对象的"引用"——内核用这些文件跟踪命名空间对象的引用计数。一个命名空间在所有进程退出、所有引用文件被关闭后才被销毁。这也是为什么一个进程可以"握住"一个已经没有进程的命名空间，随时再进去。

### 销毁
命名空间对象采用引用计数管理。进程退出时递减其所处命名空间的计数；`/proc/<pid>/ns/` 的打开文件也各持一个引用。计数归零时内核释放命名空间对象——Network Namespace 的释放会触发网络栈清理（释放接口、路由表、socket），Mount Namespace 的释放触发挂载树清理。这个"最后一个引用消失才销毁"的语义，是 CRIU 等容器迁移工具能够工作的重要前提：它们在迁移过程中持有的命名空间文件描述符，保证了命名空间在进程暂停期间不消失。

## 从内核到容器
Namespace 本身只是视图隔离——六类资源的"可见性切分"。完整的容器还需要另外三块拼图：Cgroup 提供资源计量与限制（"能用多少"）、联合文件系统提供镜像分层（"根目录从哪来"）、Capabilities/Seccomp/SELinux 提供真正的安全约束（"能做什么"）。Namespace 解决的是容器"看起来是独立系统"的问题，而容器安全是另一套机制的职责——把视图隔离误当作安全隔离，是理解容器最深的坑。

Namespace 的价值也不局限于容器：`unshare -m` 在普通用户态就提供沙箱化挂载视图，测试挂载脚本时不用污染系统；Network Namespace 被广泛用于网络功能虚拟化（每个虚拟网络功能一个命名空间）和进程级 VPN 分流。容器只是这个内核特性的最大用户，不是它的全部。
