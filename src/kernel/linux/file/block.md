# 块设备子系统
块设备子系统是工作在文件系统和设备管理层中间的代理层。它负责使用设备管理层提供的块设备驱动读写磁盘文件，并提供异步 IO、IO 批处理等功能；同时，其提供了统一的接口，为文件系统软件封装了不同的磁盘的驱动差异，并为文件系统提供了一种基于 bio 请求的交互方式，让文件系统只需要专注于自身的逻辑即可。

+ VFS
+ 文件系统
+ 块设备子系统
+ 设备管理子系统
  + 块设备驱动
  + 块设备抽象
  + 物理磁盘

## 主要功能
+ I/O 请求管理：接收文件系统或用户态的 I/O 请求（struct bio），转换为设备驱动可处理的格式。通过 struct request_queue 管理请求队列，支持合并、排序和调度优化。
  ```c
  // 封装单个 io 请求
  struct bio {
      struct bio *bi_next;              /* Next bio in chain */
      struct block_device *bi_bdev;     /* Target block device */
      unsigned short bi_flags;          /* Status and command flags */
      unsigned short bi_opf;            /* Operation flags (e.g., REQ_OP_READ) */
      unsigned short bi_ioprio;         /* I/O priority */
      struct bvec_iter bi_iter;         /* Iterator for data transfer */
      struct bio_vec *bi_io_vec;        /* Vector of data pages */
      bio_end_io_t *bi_end_io;          /* Completion callback */
      void *bi_private;                 /* Private data for callback */
      atomic_t bi_remaining;            /* Reference count for completion */
      ...
  };

  // IO 请求队列
  struct request_queue {
     struct list_head queue_head;      /* List of pending requests */
     struct request *last_merge;       /* Last merged request */
     struct elevator_queue *elevator;  /* I/O scheduler (e.g., deadline) */
     make_request_fn *make_request_fn; /* Direct bio processing function */
     struct blk_queue_ctx *queue_ctx;  /* Queue context */
     unsigned int queue_flags;         /* Queue properties (e.g., QUEUE_FLAG_MQ) */
     ...
  };
  ```
+ 设备抽象：提供 struct block_device 和 struct gendisk，抽象块设备（如 /dev/sda）和分区（如 /dev/sda1）。屏蔽底层硬件差异（如 SATA、NVMe），为文件系统提供统一接口。
  ```c
  // 用于抽象一个磁盘分区
  struct block_device {
      dev_t bd_dev;                     /* Device number (major:minor) */
      struct gendisk *bd_disk;          /* Associated gendisk */
      struct inode *bd_inode;           /* VFS inode for device file */
      struct block_device *bd_contains; /* Parent device (e.g., sda for sda1) */
      struct hd_struct *bd_part;        /* Partition info, if applicable */
      struct list_head bd_list;         /* List of block devices */
      unsigned long bd_private;         /* Driver-specific data */
      ...
  };

  // 用于抽象一个物理磁盘
  struct gendisk {
      int major;                        /* Major number */
      int first_minor;                  /* First minor number */
      int minors;                       /* Number of minors */
      char disk_name[DISK_NAME_LEN];    /* Device name (e.g., "sda") */
      struct block_device_operations *fops; /* Driver operations */
      struct request_queue *queue;       /* I/O request queue */
      void *private_data;                /* Driver-specific data */
      struct disk_part_tbl *part_tbl;   /* Partition table */
      ...
  };

  ```
+ I/O 调度：使用调度器（如 CFQ、deadline、noop）优化 I/O 性能，减少磁盘寻道时间，提高吞吐量。
+ 异步 I/O 支持：支持异步 I/O，通过 bio->bi_end_io 回调通知请求完成，降低延迟。
+ 分区与设备管理：支持分区表解析（如 GPT、MBR），管理设备状态（如热插拔）。通过 sysfs（如 /sys/block/sda）暴露设备信息和配置。

## 读文件工作流程
+ I/O 请求生成：文件系统（如 ext4）或用户态程序通过 VFS 发起 I/O 请求。ext4 通过 super_block->s_bdev 创建 struct bio，指定目标扇区和数据。
+ 请求提交：调用 submit_bio，将 bio 送入 block_device->bd_disk->queue。子系统可能将多个 bio 合并为 struct request，或直接处理（make_request_fn 模式，如 NVMe）。
+ I/O 调度：request_queue 使用调度器（如 deadline）对请求排序、合并，优化性能。
+ 驱动处理：驱动通过 gendisk->fops->submit_bio 或队列的 request_fn 处理请求。请求转换为硬件命令（如 SCSI 命令），通过中断完成。
+ 完成通知：硬件完成 I/O，触发中断，驱动调用 bio->bi_end_io 通知文件系统。文件系统更新元数据，完成操作。
