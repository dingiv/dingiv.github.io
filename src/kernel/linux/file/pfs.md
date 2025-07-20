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
