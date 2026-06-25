# 块设备子系统
块设备子系统是文件系统与磁盘驱动之间的中间层。文件系统负责在磁盘的字节数组上组织文件目录的逻辑结构，磁盘驱动负责与物理硬件交互完成数据读写，而块设备子系统屏蔽了底层硬件差异，向上为文件系统提供统一的 bio（Block I/O）请求接口，向下将请求分发给具体的磁盘驱动。同时，它还承担 I/O 调度、请求合并、队列管理等职责，是存储 I/O 路径上的核心枢纽。

## 在存储栈中的位置
从上到下，一次文件读写的完整路径依次经过：应用程序通过系统调用进入 VFS → VFS 查找文件对应的 inode 和 dentry → 文件系统（如 ext4）根据 inode 中的逻辑块映射定位数据在磁盘分区中的物理位置 → 块设备子系统接收文件系统生成的 bio 请求，进行调度和合并 → 块设备驱动将请求转换为硬件命令提交给磁盘控制器 → 磁盘控制器通过 DMA 将数据传输到内存 → 中断通知完成 → 逐层回调返回。

+ VFS
+ 文件系统
+ 块设备子系统
+ 设备管理子系统
  + 块设备驱动
  + 块设备抽象
  + 物理磁盘

```
应用程序
  ↓  open/read/write 系统调用
VFS（file → dentry → inode）
  ↓  inode->i_mapping 操作页面缓存
文件系统（ext4/xfs/...）
  ↓  逻辑块 → 物理扇区映射，生成 bio
块设备子系统
  ↓  I/O 调度、请求合并、分发
块设备驱动（NVMe/SCSI/...）
  ↓  硬件命令提交
磁盘控制器 → DMA → 物理磁盘
```

在这个层次结构中，块设备子系统向上通过 `submit_bio` 接收 bio 请求，向下通过 `gendisk->fops` 或 request queue 与驱动交互。块设备子系统与[设备管理子系统](../device/)的衔接点是 `struct block_device` 和 `struct gendisk`：`block_device` 抽象磁盘分区（如 `/dev/sda1`），`gendisk` 抽象物理磁盘（如 `/dev/sda`），两者通过设备号关联。

## 核心数据结构

### bio
bio（Block I/O）是块设备子系统的基本请求单位，由文件系统构造，描述一次连续的块 I/O 操作。文件系统不直接调用磁盘驱动，而是构造 bio 并提交给块设备子系统，由子系统负责后续的调度和分发。

```c
struct bio {
    struct bio *bi_next;              /* 请求链表 */
    struct block_device *bi_bdev;     /* 目标块设备 */
    unsigned short bi_opf;            /* 操作类型（REQ_OP_READ/WRITE） */
    struct bvec_iter bi_iter;         /* 当前段迭代器 */
    struct bio_vec *bi_io_vec;        /* 数据段向量（物理页 + 偏移 + 长度） */
    bio_end_io_t *bi_end_io;          /* 完成回调函数 */
    void *bi_private;                 /* 回调私有数据 */
    atomic_t bi_remaining;            /* 剩余完成计数 */
    ...
};
```

bio 的核心字段是 `bi_io_vec`，它是一个分散-聚集（scatter-gather）向量数组，每个元素 `struct bio_vec` 描述一个物理内存页中的数据段（页指针 + 页内偏移 + 长度）。一次 bio 可以包含多个不连续的物理页，但它们在磁盘上的逻辑块地址必须是连续的。文件系统在构造 bio 时，根据文件在磁盘上的物理块分布来填充 `bi_io_vec`。

bio 的完成通知机制通过 `bi_end_io` 回调实现。当块设备子系统完成一次 bio 的所有物理页传输后，调用 `bi_end_io(bio)` 通知文件系统。文件系统在该回调中检查错误状态、释放资源，并可能唤醒等待该 I/O 的进程。

### request 与 request_queue
request 是 I/O 调度器对 bio 进行合并和排序后的产物。当 bio 进入块设备子系统后，调度器可能将多个 bio 合并为一个 request（如果它们访问的磁盘扇区是相邻或重叠的），也可能对多个 request 按磁盘扇区顺序排序以减少寻道时间。request 是提交给块设备驱动的最终请求单位。

```c
struct request {
    struct request_queue *q;          /* 所属队列 */
    unsigned int cmd_flags;           /* 操作标志 */
    sector_t __sector;                /* 起始扇区 */
    struct bio *bio;                  /* bio 链表 */
    struct bio *biotail;              /* bio 链尾 */
    struct hlist_node hash;           /* 哈希表（查找合并候选） */
    ...
};

struct request_queue {
    struct elevator_queue *elevator;  /* I/O 调度器 */
    struct request *last_merge;       /* 最近合并的请求 */
    struct blk_queue_ctx *queue_ctx;  /* per-CPU 队列上下文 */
    unsigned int queue_flags;         /* 队列标志 */
    make_request_fn *make_request_fn; /* bio 处理函数 */
    ...
};
```

request_queue 是管理 request 的核心结构。传统上，每个 gendisk 关联一个 request_queue，所有发往该磁盘的 bio 先进入 queue，经调度器合并排序后再发给驱动。现代块层引入了 **blk-mq（Block Multi-Queue）** 架构，将单队列拆分为多级队列结构，以适应多核 CPU 和高速存储设备的需求。

### block_device 与 gendisk
`struct block_device` 和 `struct gendisk` 是块设备子系统中对上层的抽象。它们对应了块设备在 VFS 层和设备管理层中的表示。

```c
// 抽象磁盘分区（如 /dev/sda1）
struct block_device {
    dev_t bd_dev;                     /* 设备号（major:minor） */
    struct gendisk *bd_disk;          /* 所属物理磁盘 */
    struct inode *bd_inode;           /* VFS inode（/dev/sda1 的 inode） */
    struct block_device *bd_contains; /* 父设备（sda1 → sda） */
    struct hd_struct *bd_part;        /* 分区信息 */
    ...
};

// 抽象物理磁盘（如 /dev/sda）
struct gendisk {
    int major;                        /* 主设备号 */
    int first_minor;                  /* 起始次设备号 */
    int minors;                       /* 次设备号数量（分区数+1） */
    char disk_name[DISK_NAME_LEN];    /* 设备名（"sda"） */
    const struct block_device_operations *fops; /* 驱动操作接口 */
    struct request_queue *queue;      /* 请求队列 */
    struct disk_part_tbl *part_tbl;   /* 分区表 */
    ...
};
```

gendisk 是设备管理层中块设备驱动的注册产物。驱动在 probe 中分配并初始化 gendisk，通过 `add_disk` 注册到块设备子系统，内核自动在 `/sys/block/` 下创建对应条目，并通知 udev 创建 `/dev/sdX` 设备节点。block_device 是内核在首次打开某个块设备文件时创建的，一个 gendisk 对应一个 block_device（整个磁盘），每个分区对应一个额外的 block_device（通过 `bd_contains` 指向整个磁盘的 block_device）。

在 VFS 层，块设备的 inode 通过 `i_bdev` 指向 block_device，block_device 的 `bd_disk` 指向 gendisk，gendisk 的 `fops` 指向驱动操作函数。这条链路将 VFS、块设备子系统和设备驱动连接起来。block_device 对标了隔壁网络设备中的 socket 结构，是 VFS 层操作下层的 API 抽象。

## blk-mq 架构
现代 Linux 内核的块设备子系统采用 **blk-mq（Block Multi-Queue）** 架构，取代了早期的单队列架构。单队列架构下所有 CPU 核心共享一个 request_queue 和一把自旋锁，多核并发 I/O 时锁竞争严重，成为性能瓶颈。blk-mq 通过多级队列设计解决了这个问题。

blk-mq 将 I/O 路径分为两级队列：

**软件阶段（Software Stage）**：每个 CPU 核心拥有独立的**提交队列（Submission Queue, SQ）**。文件系统提交 bio 时，通过 per-CPU 的映射将 bio 放入当前 CPU 的 SQ 中。SQ 无锁操作，多核之间不存在竞争，这是 blk-mq 性能提升的关键。

**硬件阶段（Hardware Stage）**：一组**硬件队列（Hardware Queue, HW Queue）** 与底层硬件的实际并行能力对应。NVMe 设备可能支持 64 个或更多的提交/完成队列，每个队列可以绑定不同的 CPU 核。blk-mq 将 SQ 中的 request 映射到 HW Queue，由硬件队列对应的驱动上下文处理。

两级队列之间的映射关系通过 `blk_mq_tag_set` 中的 `map` 数组配置。简单的设备可以设置 1:1 映射（SQ 直接对应 HW Queue），复杂的设备可以 N:1 映射（多个 SQ 共享一个 HW Queue）。驱动在初始化时通过 `blk_mq_alloc_tag_set` 和 `blk_mq_init_queue` 创建 tag set 和 request queue。

```c
// 块设备驱动初始化 blk-mq 的典型流程
struct blk_mq_tag_set tag_set = {
    .ops = &my_mq_ops,           // 驱动实现的 mq 操作
    .nr_hw_queues = 4,           // 硬件队列数量
    .queue_depth = 128,          // 每个队列的深度
    .numa_node = NUMA_NO_NODE,
    .cmd_size = sizeof(struct my_cmd), // 驱动私有数据大小
};

blk_mq_alloc_tag_set(&tag_set);
struct request_queue *queue = blk_mq_init_queue(&tag_set);

// 驱动实现的 mq 操作
struct blk_mq_ops my_mq_ops = {
    .queue_rq = my_queue_rq,     // 处理 request
    .complete = my_complete,     // 请求完成回调
    .init_hctx = my_init_hctx,   // 初始化硬件队列上下文
};
```

`queue_rq` 是驱动最核心的回调，驱动在此将 request 转换为硬件命令（如 NVMe 提交队列条目、SCSI CDB），通过 MMIO 或 DMA 提交给磁盘控制器。`complete` 在硬件完成 I/O 后由中断处理程序调用，驱动在此回收资源并通知块设备子系统该 request 已完成。

## I/O 调度
I/O 调度器的目标是将无序的 I/O 请求整理为有利于磁盘执行的顺序，主要优化两个指标：减少机械硬盘的磁头寻道时间（对于 SSD，寻道时间不存在，但仍有其他优化空间），提高吞吐量。

对于机械硬盘（HDD），I/O 调度器通过将请求按扇区顺序排序来最小化磁头移动距离。经典的调度算法包括 **Deadline**（保证每个请求的最长等待时间不超过阈值，防止饥饿）和 **CFQ**（Completely Fair Queuing，为每个进程分配独立的队列，按时间片轮转服务）。现代内核中 CFQ 已被 **BFQ**（Budget Fair Queueing）取代，BFQ 在 CFQ 基础上引入了带宽预算机制，对交互式应用（如桌面视频播放）的延迟控制更好。

对于 SSD 和 NVMe 设备，寻道时间的概念不再适用，随机读写和顺序读写的性能差异远小于 HDD。此时调度器的核心工作从排序转向合并和限制队列深度。**mq-deadline** 是 blk-mq 架构下 Deadline 调度器的多队列版本，适用于大多数场景。**none**（Noop）是最简单的调度器，不做排序和合并，直接将请求转发给驱动，适用于自带队列管理的 NVMe 设备。内核会根据设备类型自动选择合适的调度器，也可以通过 `/sys/block/sdX/queue/scheduler` 手动切换。

```
# 查看和切换 I/O 调度器
cat /sys/block/sda/queue/scheduler
echo mq-deadline > /sys/block/sda/queue/scheduler
```

## I/O 路径详解
以 ext4 文件系统读取一个文件为例，完整追踪 I/O 从用户态到磁盘的路径。

### 读操作
用户调用 `read(fd, buf, count)` 进入内核。VFS 通过 fd 找到 `struct file`，再通过 `file->f_inode` 找到 inode。VFS 检查页面缓存（`inode->i_mapping`），如果数据已在缓存中（缓存命中），直接通过 `copy_to_user` 将数据从内核页拷贝到用户缓冲区，I/O 在此结束。

缓存未命中时，VFS 调用 `address_space->a_ops->readpage`（由 ext4 实现）。ext4 的 readpage 函数首先根据文件逻辑块号查找 ext4 的 extent 映射（`struct extent_map`），将文件的逻辑块号转换为磁盘分区的物理扇区号。然后构造 bio：`bio->bi_bdev` 指向分区的 block_device，`bi_iter.bi_sector` 设为起始物理扇区，`bi_io_vec` 填入目标内存页，`bi_end_io` 设为完成回调。构造完成后调用 `submit_bio(bio)` 将 bio 提交给块设备子系统。

块设备子系统接收到 bio 后，根据 gendisk 关联的 request_queue 进入 blk-mq 路径。bio 被放入当前 CPU 的提交队列，调度器决定是否与其他 request 合并。随后 request 进入硬件队列，驱动的 `queue_rq` 被调用。NVMe 驱动将 request 中的扇区号、长度、数据缓冲区 DMA 地址等信息填入 NVMe 提交队列条目，向控制器门铃寄存器（doorbell）写入提交计数，通知硬件处理。

NVMe 控制器通过 DMA 将数据从磁盘读取到 bio 指定的内存页中，完成后发送中断。驱动在中断处理中调用 `blk_mq_complete_request`，块设备子系统调用 bio 的 `bi_end_io`，ext4 的完成回调唤醒等待该页的进程。最终数据已在页面缓存中，后续的 `copy_to_user` 将数据拷贝到用户缓冲区。

### 写操作
写操作的路径与读操作类似，但有一个关键差异：数据先写入页面缓存（延迟写）。用户调用 `write(fd, buf, count)` 时，VFS 将数据从用户缓冲区 `copy_from_user` 到内核页面缓存中的目标页，并将该页标记为脏页（`SetPageDirty`），write 系统调用立即返回。

脏页不会立刻写回磁盘，而是由内核的回写机制（writeback）在适当时机触发。触发条件包括：脏页比例超过阈值（`vm.dirty_ratio`）、内存紧张需要回收页面、应用程序主动调用 `fsync` 或 `fdatasync`、或者定时器周期性触发（kworker 线程）。

回写时，`address_space->a_ops->writepages`（ext4 实现）被调用，ext4 将脏页构造为 WRITE 类型的 bio 提交到块设备子系统。对于日志文件系统（如 ext4），在写数据之前还需要先将元数据变更写入日志（journal），日志写入和数据写入各自生成独立的 bio，但日志 bio 必须先于数据 bio 完成以保证崩溃一致性。

### Direct I/O 与 Sync I/O
默认的 I/O 路径经过页面缓存，适合大多数场景。但在某些场景下需要绕过页面缓存：

**Direct I/O（O_DIRECT）**：用户调用 `open` 时指定 `O_DIRECT` 标志，I/O 数据直接在用户缓冲区和磁盘之间传输，不经过页面缓存。适用于数据库等自行管理缓存的场景，避免双重缓存（数据库 buffer pool + 内核页面缓存）浪费内存。Direct I/O 要求用户缓冲区的地址和大小必须对齐到文件系统逻辑块大小（通常 4KB），否则系统调用返回 EINVAL。

**Sync I/O（O_SYNC）**：每次 write 系统调用在数据写入磁盘后才返回，不依赖回写机制的延迟。适用于对数据持久性要求极高的场景（如金融交易日志），但性能开销较大。

两者可以组合使用（`O_DIRECT | O_SYNC`），此时数据直接从用户缓冲区 DMA 到磁盘，write 在硬件确认完成后返回。

## 分区管理
一块物理磁盘可以被划分为多个分区，每个分区在内核中对应一个 `struct hd_struct` 和一个 `struct block_device`。分区表存储在磁盘的第一个扇区（LBA 0）中，常见格式有 **MBR**（Master Boot Record，最多 4 个主分区，支持扩展分区）和 **GPT**（GUID Partition Table，最多 128 个分区，支持 UEFI 启动）。

内核在块设备注册后（`add_disk`），会自动读取磁盘的分区表，为每个分区创建 block_device 和对应的 `/dev/sdXN` 设备节点。分区信息存储在 `gendisk->part_tbl` 中。分区调整（如 `fdisk`、`parted`）后，需要通知内核重新读取分区表（通过 `ioctl(BLKRRPART)` 或重新插拔设备）。

分区在 I/O 路径上的体现是扇区偏移量的转换。当文件系统向分区 `/dev/sda1` 提交 bio 时，bio 中的扇区号是相对于分区起始位置的。块设备子系统在将 bio 分发给驱动之前，会将扇区号加上分区的起始偏移量，转换为相对于整块磁盘的绝对扇区号。这个转换由 `blk_partition_remap` 函数完成，对文件系统和驱动都是透明的。
