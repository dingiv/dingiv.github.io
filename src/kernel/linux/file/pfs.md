# 伪文件系统
伪文件系统（Pseudo Filesystems）是 Linux 虚拟文件子系统（VFS）中的一类特殊文件系统，不依赖物理存储设备，而是由内核动态生成，用于暴露内核数据结构、设备信息或系统状态给用户态。它们通过 VFS 提供文件接口，遵循“一切皆文件”的哲学，方便用户和程序访问内核信息。

文件和目录内容实时反映内核状态，运行时创建；部分伪文件系统（如 `/proc`）只读，部分（如 `/sys`）支持写操作以配置内核。

## 常见伪文件
1. /proc（procfs）
   - 作用：提供进程和内核信息的接口，反映系统状态。
   - 内容：
     - 进程信息：如 `/proc/[pid]/status`（进程状态）、`/proc/[pid]/maps`（内存映射）。
     - 系统信息：如 `/proc/cpuinfo`（CPU 信息）、`/proc/meminfo`（内存使用）。
     - 内核参数：如 `/proc/sys/kernel/`（可读写配置，如 `sysctl`）。
   - 实现：
     - 注册为 `proc_fs_type`，使用 `proc_dir_entry` 管理文件和目录。
     - 通过 `proc_ops`（类似 `file_operations`）实现文件操作。
   - 示例：`cat /proc/uptime` 查看系统运行时间。

2. /sys（sysfs）
   - 作用：暴露设备、驱动和内核对象的属性，支持设备管理和配置。
   - 内容：
     - 设备拓扑：`/sys/devices/` 反映硬件层级。
     - 设备类型：`/sys/class/`（如网卡 `/sys/class/net/`）。
     - 驱动信息：`/sys/bus/`（如 PCI、USB 设备）。
   - 实现：
     - 基于 `struct kobject`，通过 `sysfs_ops` 提供读写接口。
     - 设备驱动通过 `kobject` 创建 sysfs 条目，映射到文件。
   - 示例：`echo 1 > /sys/class/leds/brightness` 控制 LED。

3. /dev（devfs 或 udev）
   - 作用：提供设备文件接口，表示字符设备和块设备。
   - 内容：
     - 字符设备（如 `/dev/tty`）、块设备（如 `/dev/sda`）。
     - 动态管理：由 `udev`（用户态）或内核动态创建设备文件。
   - 实现：
     - 传统 devfs 已废弃，现由 `udev` 配合 tmpfs 实现。udev 是一个用户态的服务程序，通过监听内核的 uevent 硬件事件，特别是设备的插拔事件，从而来动态为 /dev 目录创建设备的虚拟文件，提供设备的访问接口。其他的伪文件系统是由内核实现的，而 /dev 伪文件由用户态的 udev 服务配合实现，这一点区别于其他的伪文件系统。
     - 设备驱动通过 `register_chrdev` 或 `cdev_add` 注册，VFS 创建 `inode`（含 `i_cdev` 或 `i_bdev`）。
   - 示例：`cat /dev/random` 获取随机数。

4. /tmpfs（tmpfs）
   - 作用：基于内存的文件系统，数据存储在 RAM，速度快，断电丢失。
   - 内容：常用于临时文件或共享内存（如 `/dev/shm`）。
   - 实现：
     - 注册为 `tmpfs_fs_type`，数据存储在页面缓存，无需块设备。
     - 支持 `file_operations` 和 `inode_operations`。
   - 示例：`mount -t tmpfs tmpfs /mnt` 创建内存文件系统。

5. /debug（debugfs）
   - 作用：提供内核开发者调试接口，暴露内部数据结构。
   - 内容：自定义调试信息，由驱动或子系统注册（如 `/debug/tracing`）。
   - 实现：
     - 注册为 `debugfs_fs_type`，通过 `debugfs_create_file` 创建文件。
     - 提供简单接口（如 `debugfs_create_u32`）添加调试节点。
   - 示例：`cat /debug/tracing/trace` 查看内核跟踪日志。

### 伪文件系统的工作原理
伪文件系统首先通过 `register_filesystem` 向内核注册自己的 `struct file_system_type`（如 `proc_fs_type`），并在挂载时（如 `mount -t proc proc /proc`）由 VFS 创建对应的 `struct super_block`。伪文件和目录在内核中由 `struct inode` 和 `struct dentry` 结构体表示，并绑定特定的操作函数（如 `proc_ops` 或 `sysfs_ops`），实现与 VFS 的集成。

与传统文件系统不同，伪文件系统的文件内容并不存储在磁盘上，而是由内核在访问时动态生成。每个文件的 `read` 或 `write` 回调函数会实时从内核数据结构中获取或更新信息，例如 `/proc/cpuinfo` 的 `read` 操作会读取当前 CPU 的相关信息。

用户通过常规的文件操作命令（如 `cat`、`echo`）访问这些伪文件，VFS 会调用伪文件系统实现的 `file_operations`。对于支持写操作的伪文件（如 `/sys`），写入会触发内核回调，从而实现对设备或内核参数的动态配置和管理。

## procfs 的编程接口
内核子系统或驱动向 `/proc` 暴露信息的路径是 `proc_create`——它创建一个 `proc_dir_entry` 并注册 `proc_ops`（read/write 回调）：

```c
static int my_proc_show(struct seq_file *m, void *v) {
    seq_printf(m, "counter: %d\n", my_counter);
    return 0;
}

static int __init my_init(void) {
    proc_create_single("my_info", 0, NULL, my_proc_show);
    return 0;
}
```

`proc_create_single` 是简单只读文件的现代封装（内部用 seq_file——内核的序列化输出缓冲，解决"read 回调被多次调用、每次只填一页"的拼接问题）。复杂场景直接用 `proc_create` 注册完整的 `proc_ops`，支持 read/write/llseek。老代码中的 `create_proc_entry` 已废弃——没有类型安全且容易出错。

proc 条目按目录组织（`proc_mkdir` 创建目录），子系统各有自己的命名空间：`/proc/sys/`（sysctl 参数）、`/proc/net/`（网络状态）、`/proc/<pid>/`（每进程信息，由内核动态生成）。`/proc/<pid>/` 下的条目不经过 proc_create——它们由进程子系统的固定表生成（`pid_entry` 数组），内容直接从 `task_struct` 读取。

**procfs 与 sysfs 的分工**：procfs 暴露"进程和内核状态"（面向运维和诊断），sysfs 暴露"设备模型"（面向设备管理）。历史遗留使 `/proc/cpuinfo`、`/proc/interrupts` 这类硬件信息留在 procfs，但新代码的规则明确——设备属性进 sysfs，进程/内核状态进 procfs，调试信息进 debugfs。

## /proc 目录的瞬时快照
`/proc` 常被称为"系统状态的瞬时快照"——每次读取都是当下时刻的实时值。常用条目速查：

| 条目 | 内容 | 典型用途 |
|------|------|---------|
| `/proc/cpuinfo` | CPU 型号、频率、flags | 确认 CPU 特性（如 avx512、smx） |
| `/proc/meminfo` | 内存分区使用详情 | 排查内存去向（Cached/Slab/Swap） |
| `/proc/loadavg` | 1/5/15 分钟负载 + 运行队列 | 快速判断系统繁忙度 |
| `/proc/interrupts` | 每个 CPU 的中断计数 | 中断分布是否均衡（IRQ 亲和排查） |
| `/proc/vmstat` | 内核内存管理事件计数 | 页回收/swap 活动分析 |
| `/proc/<pid>/status` | 进程状态摘要 | 查看 VMSize/RSS/线程数/上下文切换 |
| `/proc/<pid>/maps` | 内存映射区域 | 排查 so 库加载、内存布局 |
| `/proc/<pid>/sched` | 调度器统计 | 自愿/非自愿上下文切换、等待时间 |
| `/proc/<pid>/fd/` | 打开的文件描述符符号链接 | "这个进程打开了什么文件" |
| `/proc/<pid>/cmdline` | 启动命令行 | 参数用 \0 分隔（`tr '\0' ' '` 查看） |
| `/proc/sys/*` | sysctl 可写参数 | `echo` 或 `sysctl` 运行时调参 |

注意 read 语义的坑：多数 `/proc` 条目每次 read 返回"当下的快照"，但 `seq_file` 的多页拼接（如 `cat /proc/interrupts` 在长输出时）可能读到前后不一致的拼接——内核对此的对策是 seq_file 的 `stop`/重启机制（检测到不一致时重读），应用侧则应避免把 /proc 输出当作严格一致的事务快照。

## sysfs 的对象模型：kobject 与 kset
sysfs 背后是 Linux 设备模型的核心抽象——**kobject**（kernel object）。kobject 是设备模型的最小单元：一个引用计数 + 一个名字 + 一个父指针 + 一个 ktype。它不直接代表"设备"——而是提供所有设备模型对象（device、driver、bus、class）的公共基础设施：引用计数生命周期、sysfs 目录表示、uevent 热插拔事件。

```c
struct kobject {
    const char      *name;       // sysfs 中的目录名
    struct list_head entry;      // 挂入父 kset 的链表
    struct kobject   *parent;    // 父目录（层级结构）
    struct kset      *kset;      // 所属的 kset
    struct kobj_type *ktype;     // 属性和释放回调
    struct kref      kref;       // 引用计数
};
```

**kset** 是 kobject 的集合——它自身也内嵌一个 kobject（既当容器又当成员）。kset 的价值是分组和事件聚合：`/sys/block/` 下所有块设备的 kobject 属于同一个 kset，`/sys/devices/` 的设备层级通过 kset 组织。kset 还关联一个 `kset_uevent_ops`——决定这个集合的成员在什么条件下发出 add/remove/change 事件。

kobject 与 sysfs 目录的对应关系由 `kobject_add`（创建 sysfs 目录）和 `sysfs_create_file`（在目录下创建属性文件）建立。属性的读写最终落到 `kobj_type->sysfs_ops` 的 `show`/`store` 回调——这就是 `echo 1 > /sys/class/leds/.../brightness` 的完整路径：sysfs 的 write → `sysfs_ops->store` → 驱动的处理函数。

设备模型各对象与 kobject 的关系是"内嵌而非继承"（C 语言无继承）：`struct device` 内嵌 `struct kdevice`……实际是 `device->kobj`，`device_driver`、`bus_type`、`class` 同样内嵌。对象生命周期由 kobject 的引用计数管理——`kobject_get`/`kobject_put`，计数归零时 `kobj_type->release` 回调销毁对象。这解决了设备模型最难的"何时释放"问题——热插拔场景下设备对象的最后一个引用者（可能是 sysfs 的一个打开文件）释放前对象不能消失。

用户态看到的 `/sys` 层级是 kobject 树的直接映射：`/sys/devices/pci0000:00/0000:00:1f.2/` 的目录深度对应设备在总线拓扑中的位置；`/sys/class/net/eth0` 是同一设备在"class 视角"下的符号链接——同一个 kobject，多个视角的呈现。理解"kobject 是 sysfs 的骨骼"之后，`/sys` 的目录结构不再是黑盒——它就是设备模型数据结构的文件系统投影。
