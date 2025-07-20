---
title: 设备管理
order: 50
---

# 设备管理
硬件设备管理是操作系统的核心职责之一。在冯诺依曼架构下，设备主要就是 **CPU、内存和外设**。CPU 和内存会采用一些单独的管理机制，对此之外的其他设备，采用一种较为统一的管理机制，包括**设备抽象**和**驱动程序**。

## 设备管理分层
Linux 设备管理包括从下层硬件到上层驱动接口的多个层次。
- 物理硬件层
- 设备抽象层
- 设备驱动层

## 物理硬件层
物理硬件层，操作系统需要使用**硬件接口**来完成对硬件的管理和数据传输，无须关心硬件内部的具体实现。从硬件接口类型来看，设备的类型主要有两种：**总线设备和平台设备**。总线设备的硬件接口和平台设备的硬件接口有所不同，总线设备的硬件接口需要通过**总线协议**来定义，而平台设备的硬件接口通过**内核自定义**，具体是通过**通过设备树或 ACPI** 来描述。

### [总线设备](/kernel/embed/bus/)
总线设备：通过标准总线协议（如 PCI、USB、I2C、SPI）连接的硬件设备，由总线驱动管理，支持动态探测和枚举。正如其他协议一样，协议的出现可以明确双发通信的流程，使得流程规范化、共识化。

理解总线设备硬件接口，需要结合总线协议规定的硬件行为（设备探测、设备枚举、资源配置等）来理解，在此之上操作才知道如何控制总线设备。但幸运的是，总线设备往往配有一个总线控制器，该控制器实现硬件级别的设备管理，操作系统无需关心过多的设备管理功能，只需要安装总线控制器的驱动，从而调用总线的驱动，让总线控制器来帮忙管理总线设备即可，简化了操作系统的复杂度。

### [平台设备](/kernel/embed/bus/platform)
平台设备：SoC 集成或非标准协议的硬件设备（如 UART、GPIO），通过设备树或 ACPI 静态描述，不支持动态枚举。
> **设备树源**配置文件是一个 Linux 内核提供的一个自定义硬件接口的机制，多见于 arm 平台和嵌入式设备，通过编写一个特殊的 dts （device tree source）文件，然后在内核启动的时候加载进入内核，作为内核启动配置来让内核识别自定义的硬件设备的接口。
>
> **ACPI** 是一种标准化规范，通过表格（如 DSDT、SSDT）文件描述硬件配置、电源管理和设备关系，主要用于 x86 和部分 ARM 系统，同样在内核启动的时候加载到内核中，用于作为配置让内核识别自定义的硬件，功能和设备树源类似。

平台设备的管理依赖于特定硬件实现，很多硬件厂商自研了专有硬件，嵌入到了特定的平台上，此时使用原版的 Linux 是无法探测这些这些硬件的，所有需要为 Linux 编写**设备树源**，使得 Linux 知道该硬件，即在物理内存地址空间的某一段上存在着一个怎样的硬件，帮助操作系统将物理内存地址空间中的某一段识别为硬件的寄存器空间。

### 数据传输
数据面的数据传输机制决定设备数据面编程方式和通信性能。平台设备往往更加现代和规范，其使用较为先进的 MMIO 方式，可以将设备的寄存器空间映射到物理内存空间中，从而让操作系统以访问内存的方式来访问设备，简化了操作系统的复杂度。

|设备类型|MMIO|Port I/O|
|-|-|-|
|总线设备|广泛使用（如 PCIe、USB、I2C）|老式设备（如 ISA、传统 PCI）|
|平台设备|主要接口（如 UART、GPIO）|极少使用（仅 x86 特定场景）|

## 设备抽象层
硬件设备管理的主要工作是使用系统定义的 C 语言数据结构来抽象物理设备，并且将设备的驱动关联到数据结构之上，系统操作设备的时候无须直接访问硬件，而是通过驱动来间接访问。结构化抽象的主要目标是硬件的描述信息和设备支持的操作函数。

内核提供统一的**设备模型**管理框架，用于管理设备，并其绑定驱动，提供设备发现、绑定和资源管理的结构化方式。Linux 中的设备使用一个树形结构来保存，驻留在内存中，以 `struct device` 结构作为一个树节点，支持树的正反向遍历，可以将其称为**设备拓扑树**，它是设备树源的实例化结果。
> 注意这个设备拓扑树和用于设备探测的设备树源配置文件不是一个东西。并且设备树源只定义了平台设备，不定义总线设备，而设备拓扑树还包含了总线设备。
```c
struct device {
    struct device *parent;     // 父设备（如总线或控制器）
    struct bus_type *bus;      // 所属总线（如 PCI、USB）
    struct device_driver *driver; // 关联驱动
    void *platform_data;       // 平台数据
    struct list_head bus_list; // 总线设备链表
    ...
};

// 总线设备，通过总线设备的驱动来管理这条总线上的所有设备
struct bus_type {
    const char *name;      // 总线名称
    int (*match)(struct device *dev, struct device_driver *drv);
    ...
};
```

通过设备抽象，操作系统还可以定义虚拟设备，即使操作系统没看到真实的物理设备，也可以假装有一个，然后使用软件的实现来模拟底层硬件的行为。其中虚拟网卡、虚拟磁盘就是非常典型的例子。

根据设备的访问方式，主要有三种设备类型，三种类型最终都需要关联成一个 device，然后纳入到设备拓扑树中管理。
- 字符设备：只支持顺序访问（如 /dev/ttyS0），使用 `struct cdev` 定义。由于字符设备只支持顺序访问，所以字符设备可以直接被看作是一个文件，然后可以直接对其进行读写操作，但是不可以进行文件指针的移动，也就是不支持 `lseek` 操作，通常不支持缓冲。
- 块设备：随机访问（如 /dev/sda），使用 `struct block_device` 定义。一个块设备可以支持随机访问，并且包含多个文件，我们可以随机访问其中的某一个文件，并对单个文件使用 lseek 操作，其上层是**块设备子系统**，通常支持缓冲区。
- 网络设备：如 eth0，使用 `struct net_device` 定义。处理数据时基于数据包而非字节流或块。一个网络设备上游调用者不是直接来自于用户态的读写操作，而是来自于**内核协议栈**的调用，属于**网络子系统**。

### 字符设备

```c
// 字符设备
struct cdev {
    struct kobject kobj;             // 设备模型信息，将设备纳入设备模型，并暴露到 /sys/class
    struct module *owner;            // 指向驱动模块，防止卸载
    const struct file_operations *ops; // 文件操作接口（如 read/write）
    struct list_head list;           // 链接到 inode 的 cdev 列表
    dev_t dev;                       // 设备号（major/minor）
    unsigned int count;              // 设备数量
    // ...
};

struct file_operations {
    struct module *owner;
    ssize_t (*read)(struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write)(struct file *, const char __user *, size_t, loff_t *);
    int (*open)(struct inode *, struct file *);
    int (*release)(struct inode *, struct file *);
    // ... 还有 ioctl、mmap、poll 等等
};
```

### 块设备

```c
// 块设备
struct gendisk {
   struct kobject kobj;             // 嵌入设备模型
    int major;                       // 主设备号
    int minors;                      // 次设备号范围
    struct block_device_operations *fops; // 块设备操作接口
    struct request_queue *queue;     // I/O 请求队列
    struct disk_part_tbl *part_tbl;  // 分区表
    char disk_name[DISK_NAME_LEN];   // 设备名称（如 sda）
    struct block_device *part0;      // 主分区
    // ...
};

struct block_device_operations {
    int (*open)(struct block_device *, fmode_t);
    void (*release)(struct gendisk *, fmode_t);
    int (*ioctl)(struct block_device *, fmode_t, unsigned, unsigned long);
    int (*media_changed)(struct gendisk *);
    int (*revalidate_disk)(struct gendisk *);
    // ...
};
```

### 网络设备

```c
// 网络设备
struct net_device {
    char name[IFNAMSIZ];             // 接口名称（如 eth0）
    struct net_device_ops *netdev_ops; // 网络操作接口
    struct netdev_hw_addr_list dev_addrs; // MAC 地址
    unsigned char addr_len;          // MAC 地址长度
    unsigned int flags;              // 接口状态（如 IFF_UP）
    struct net *net;                 // 网络命名空间
    struct rtnl_link_stats64 *stats; // 网络统计
    struct netdev_queue *tx_queue;   // 发送队列
    // ...
};

struct net_device_ops {
    int (*ndo_open)(struct net_device *dev);
    int (*ndo_stop)(struct net_device *dev);
    netdev_tx_t (*ndo_start_xmit)(struct sk_buff *skb, struct net_device *dev);
    int (*ndo_set_mac_address)(struct net_device *dev, void *addr);
    // ...
};
```

## 设备驱动层
驱动是操作系统定义的一套 [SPI（Service Provider Interface）](./driver)。linux 定义了一套驱动程序的规范，要求驱动程序应该长什么样。硬件厂商如果希望 linux 系统能够管理他们生产的硬件，那么就需要实现 linux 定义的 SPI。操作系统无需关心硬件接口的细节，而通过驱动来间接访问和管理设备，而驱动程序则以内核模块的方式加载进内核。编写设备驱动的时候基于设备抽象层提供的统一设备模型接口，区分为三种。

```c
struct device_driver {
    const char *name;                    // 驱动的唯一标识，用于设备匹配和 sysfs 暴露
    struct bus_type *bus;                // 所属总线类型，用于设备匹配和总线管理
    struct module *owner;                // 驱动模块，防止模块在设备绑定时卸载
    const struct of_device_id *of_match_table; // 设备树匹配表
    const struct acpi_device_id *acpi_match_table; // ACPI 匹配表
    int (*probe)(struct device *dev);    // 设备初始化函数
    void (*remove)(struct device *dev);  // 设备移除函数
    void (*shutdown)(struct device *dev); // 设备关闭函数
    int (*suspend)(struct device *dev, pm_message_t state); // 挂起函数
    int (*resume)(struct device *dev);   // 恢复函数
    // ...
};
```

+ 字符设备驱动开发时，通常需要实现一组 `file_operations` 操作函数（如 open、read、write、release 等），并将其赋值给 `cdev` 结构体的 `ops` 成员。驱动通过初始化 `cdev` 并调用 `cdev_add()` 将其注册到内核，之后用户空间可以通过 `/dev/xxx` 设备文件访问字符设备，所有操作最终会调用到驱动实现的接口，实现对底层硬件的顺序读写和管理。
+ 块设备的驱动开发通常需要实现 `block_device_operations` 结构体，定义块设备的基本操作接口。驱动首先通过 `register_blkdev()` 注册主设备号，然后初始化并注册 `gendisk` 结构体，将其挂载到内核块设备子系统。这样，用户空间就可以通过 `/dev/sda` 等设备文件访问块设备，所有的操作最终都会调用到驱动实现的接口函数，实现对底层硬件的读写和管理。
+ 网络设备驱动开发时，首先需要实现 `net_device_ops` 结构体，定义网络设备的操作方法。驱动分配并初始化 `net_device` 结构体后，通过 `register_netdev()` 将其注册到内核网络子系统。此后，内核协议栈会通过 `netdev_ops` 调用驱动实现的接口进行数据包的收发、设备的启动和关闭等操作，实现网络数据的高效传输和管理。

### 驱动绑定

### 主次设备号
