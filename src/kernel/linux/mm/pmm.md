# 物理内存管理
物理内存空间可以看作是一个地址数组，每个地址对应一个字节，地址范围取决于计算机位数（32 位支持 4GB，64 位理论支持 16EB）。物理内存管理负责内核如何获知物理内存的布局、如何分配和回收物理页框。

## 物理内存探测
主板和 BIOS 程序在上电时：
1. 检查和扫描设备
2. 初始化各种设备（内存条和各种 IO 设备的寄存器和缓冲区）
3. 将这些存储空间拼接成连贯的物理内存空间

生成的必要信息，这些信息放置在一些可以在后续操作

内核在启动早期需要知道物理地址空间中哪些范围是可用 RAM、哪些被硬件设备占用了。不同平台使用不同的探测机制。

### E820 表（x86 平台）
E820 表是 x86 平台上 BIOS/UEFI 向操作系统传递物理内存布局的标准方式。Boot Loader（GRUB）在启动时通过 BIOS 中断 `INT 15h, AX=E820h` 逐条查询物理地址范围，结果存储在 `boot_params.e820_table` 中，传递给内核。现代 UEFI 系统通过 EFI 内存描述（GetMemoryMap）获取等价信息，内核启动时将其转换为 E820 格式。

E820 表的每一项描述一段连续的物理地址范围，包含三个字段：起始地址（base）、长度（length）和类型（type）。常见的类型包括：

- `E820_TYPE_RAM`（类型 1）：可用内存，内核可以自由分配使用。
- `E820_TYPE_RESERVED`（类型 2）：保留区域，硬件设备占用或 BIOS 保留，内核不得使用。
- `E820_TYPE_ACPI`（类型 3）：ACPI 数据表占用的可回收内存，内核初始化 ACPI 后可以回收使用。
- `E820_TYPE_NVS`（类型 4）：ACPI NVS（Non-Volatile Storage）内存，用于 S3 睡眠时保存硬件状态，不能被内核分配。
- `E820_TYPE_UNUSABLE`（类型 5）：内存存在但不可靠（如检测到坏的内存区域），内核不能使用。
- `E820_TYPE_PMEM`（类型 7）：持久内存（Persistent Memory，如 Intel Optane DC），断电后数据不丢失，通过特殊路径管理。

一个典型的 x86 系统的 E820 表如下（可通过 `dmesg | grep e820` 或 `cat /proc/iomem` 查看）：

```
BIOS-provided physical RAM map:
 BIOS-e820: [mem 0x0000000000000000-0x000000000009fbff] usable
 BIOS-e820: [mem 0x00000000000f0000-0x00000000000fffff] reserved
 BIOS-e820: [mem 0x0000000000100000-0x00000000bffdffff] usable
 BIOS-e820: [mem 0x00000000bff00000-0x00000000bffeffff] ACPI data
 BIOS-e820: [mem 0x00000000bfff0000-0x00000000bfffffff] reserved
 BIOS-e820: [mem 0x00000000fec00000-0x00000000fec0ffff] reserved   // IO APIC
 BIOS-e820: [mem 0x00000000fed00000-0x00000000fed00fff] reserved   // HPET
 BIOS-e820: [mem 0x00000000fee00000-0x00000000ffffefff] reserved   // Local APIC
 BIOS-e820: [mem 0x0000000100000000-0x000000041fffffff] usable     // 高端内存
```

从表中可以看到：0-640KB 是早期可用的 RAM；640KB-1MB 被 BIOS 保留（包含 VGA 显存等）；1MB 到约 3GB 是主要可用 RAM 区域；中间有 ACPI 保留区和硬件设备寄存器的 MMIO 区域（IO APIC、HPET、Local APIC 等）；4GB 以上是 64 位系统的高端内存。

内核在启动时解析 E820 表，过滤掉 reserved 和 unusable 区域，将所有 usable 和 ACPI reclaimable 区域合并为可用物理内存范围，作为后续页框分配的基础。这个过程在 `arch/x86/kernel/e820.c` 中完成。

### 设备树（ARM 平台）
ARM 平台不使用 E820 表，物理内存布局通过设备树（Device Tree）中的 `memory` 节点描述：

```
memory {
    device_type = "memory";
    reg = <0x0 0x80000000 0x0 0x80000000>;  // 起始地址 2GB，大小 2GB
};
```

`reg` 属性可以有多组，每组描述一段连续的物理内存范围。内核在启动时解析设备树的 `memory` 节点，将所有 `reg` 条目收集为可用物理内存范围。ARM 平台上的硬件设备寄存器区域通过设备树中的其他节点描述，内核不会将这些区域纳入可用内存。

## struct page 与 mem_map
内核为每个物理页框维护一个 `struct page` 结构体（定义在 `include/linux/mm_types.h`），用于跟踪该页面的状态。所有 `struct page` 组成全局数组 `mem_map`，通过物理页框号（PFN，Page Frame Number）可以直接索引。

```c
struct page {
    unsigned long flags;       // 页面状态标志（PG_locked, PG_dirty, PG_lru, PG_slab 等）
    atomic_t _refcount;        // 引用计数，为 0 时页面可被回收
    struct list_head lru;      // LRU 链表节点
    struct address_space *mapping; // 关联的地址空间（文件页）或 NULL（匿名页）
    unsigned long index;       // 页内偏移或 swap 条目编号
    void *s_mem;               // SLUB 分配器的数据起始地址
    ...
};
```

`struct page` 的大小约为 64 字节，对于 4KB 的页框，`struct page` 的开销约为 1.5%。一个 16GB 内存的系统需要约 256MB 来存储 `struct page` 数组。为了节省空间，内核将 `mem_map` 放置在可直接映射区域的尾部，紧跟物理内存之后。

## 内核虚拟地址
由于 CPU 访问内存必须经过 MMU，内核代码本身也通过虚拟地址访问物理内存。内核虚拟地址空间中有一段区域通过**线性映射**直接对应物理地址：虚拟地址 = 物理地址 + 常数偏移量（x86_64 上偏移量为 `PAGE_OFFSET`，通常为 0xffff888000000000）。通过线性映射，内核可以直接用指针访问任意物理页，无需额外建立页表项。

线性映射区域只覆盖可用物理内存的范围，中间的空洞（被硬件设备占用的 MMIO 区域）不参与映射。物理内存条在物理地址空间中本身可能是分散的，线性映射通过跳过空洞的方式将这些分散的可用区域连缀成连续的内核虚拟地址空间，简化了内核的内存管理。

内核虚拟地址空间中除了直接映射区域外，还包括 vmalloc 区域（用于分配不连续的虚拟内存）、模块映射区域、kmap 区域等，这些区域使用多级页表映射，不属于线性映射。

## 页框分配
页框分配是物理内存管理的核心功能，内核使用伙伴系统（Buddy System）和 SLUB 分配器两级结构来满足不同粒度的分配需求。

### 伙伴系统
伙伴系统（Buddy System）用于分配连续的物理页框，是内核页框分配的基础。它将空闲页面按 2 的幂次分组为多级链表数组（order 0 到 order 10），每级链表包含大小为 $2^{order}$ 个连续页框的空闲块。

![](./buddy.dio.svg)

分配过程：
1. 根据请求大小，找到对应的块链表
2. 如果链表为空，则向更大的块链表申请
3. 将大块分割成两个小块，一个用于分配，另一个加入较小的块链表
4. 如果仍然没有合适的块，则继续向更大的块链表申请

分配时，从请求大小对应的最小 order 开始查找，如果该级链表为空则向更高级申请一个大块，将其对半分裂后放入低级链表，直到得到所需大小的块。释放时，将块加入对应链表，并检查其"伙伴"（地址相邻的同大小块）是否也空闲，如果是则合并为更大的块，继续向上尝试合并。

伙伴系统的时间复杂度为 O(order)，对于 4KB 页框最大分配 4MB（order 10）。内核为分配页框提供的接口包括 `__get_free_pages(gfp_mask, order)`、`alloc_pages(gfp_mask, order)` 和 `__get_free_page(gfp_mask)`（分配单个页框的快捷方式）。

`gfp_mask`（Get Free Pages mask）是分配标志的位掩码，控制分配行为和约束：`GFP_KERNEL` 是最常用的标志，表示在进程上下文中分配，允许阻塞和 I/O；`GFP_ATOMIC` 表示在中断上下文中分配，不能阻塞；`GFP_DMA` 要求分配的内存位于低 16MB 地址范围内（用于 ISA 设备的 DMA）；`GFP_HIGHUSER` 用于用户空间页面的分配。

释放过程：
1. 将释放的块加入对应的块链表
2. 检查是否有相邻的空闲块
3. 如果有，则合并成更大的块，并加入更大的块链表
4. 重复步骤 2-3，直到无法继续合并

优点：快速分配和释放，有效减少内存碎片，支持大块内存分配；缺点：可能造成内部碎片，合并操作可能较慢，不适合小块内存分配。

### SLUB 分配器
伙伴系统以页（4KB）为最小分配单位，但内核中大量对象的体积远小于一页（如 `struct task_struct` 约 9KB、`struct inode` 约 1KB、`struct file` 约 256 字节）。如果每个小对象都占用一整页，内存浪费严重。SLUB 分配器（前身是 SLAB 分配器）在伙伴系统之上提供小对象的高效分配。

SLUB 为每种内核对象类型维护一个 **kmem_cache**（缓存），每个缓存关联一组 slab（连续页框的集合），每个 slab 内部被划分为若干等大的对象槽位。分配时从 slab 中取一个空闲槽位返回，释放时归还到 slab。当 slab 中所有槽位都被占用时，从伙伴系统申请新的页框创建 slab；当 slab 中所有槽位都空闲时，将 slab 归还给伙伴系统。

```c
// 创建 kmem_cache
struct kmem_cache *my_cache = kmem_cache_create("my_object",
    sizeof(struct my_object), 0, SLAB_HWCACHE_ALIGN, NULL);

// 分配对象
struct my_object *obj = kmem_cache_alloc(my_cache, GFP_KERNEL);

// 释放对象
kmem_cache_free(my_cache, obj);

// 通用小内存分配（基于预定义的 kmalloc 缓存）
void *ptr = kmalloc(256, GFP_KERNEL);
kfree(ptr);
```

`kmalloc` 是 SLUB 分配器的通用接口，内部根据请求大小选择合适的预定义缓存（kmalloc-64、kmalloc-128、kmalloc-256 等）。`kmalloc` 返回的内存保证物理连续（因为来自伙伴系统分配的 slab），这是它区别于 `vmalloc` 的关键特性。

Linux 内核曾使用三种 SLAB 实现：SLAB（最早引入，使用复杂）、SLUB（当前默认，实现简洁高效，支持 NUMA 和调试）、SLOB（面向嵌入式，极简但性能低）。当前主流内核默认使用 SLUB。
