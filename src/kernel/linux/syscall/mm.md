# 内存管理
内存管理接口包括用户态接口和内核态接口。

## 系统调用
Linux 的内存管理基于虚拟内存，通过分页机制将进程的虚拟地址映射到物理内存（包括 DRAM 和缓存，如 Intel H 系列的 DDR5 和 L3 缓存）。用户态程序无法直接操作物理内存，而是通过系统调用请求内核分配、释放或管理内存。这些系统调用触发软中断（`syscall` 指令），在进程上下文中运行，涉及内核的页面表更新和内存分配器（如 `slab`）。主要系统调用包括 `mmap`、`munmap`、`brk`、`sbrk`、`mremap`、`mprotect`、`msync` 和 `madvise`，以下逐一分析。

### 内存分配

#### mmap/munmap/mremap
映射内存或文件，分配匿名内存或将文件映射到进程的虚拟地址空间，广泛用于动态内存分配和文件 I/O。使用时注意成对调用。实现原理：更新进程的 `mm_struct` 和分配页面表，但是标记为`不存在`。首次访问是，发现`不存在`，页面错误触发中断上下文，调用 `do_page_fault` 分配物理页面。

```c
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int munmap(void *addr, size_t length);
int main() {
    size_t len = 4096;
    int *addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    addr[0] = 42;
    printf("Stored: %d\n", addr[0]);
    munmap(addr, len);
    return 0;
}
```

- 参数：
  - `addr`：建议映射地址（通常为 `NULL`，由内核选择）。
  - `length`：分配大小（字节，需页面对齐）。
  - `prot`：权限（`PROT_READ`、`PROT_WRITE`、`PROT_EXEC`）。
  - `flags`：映射类型（`MAP_ANONYMOUS`、 `MAP_PRIVATE`、 `MAP_SHARED`）。
  - `fd`：文件描述符（匿名内存为 -1）。
  - `offset`：文件偏移（匿名内存为 0）。
- 返回值：成功返回虚拟地址，失败返回 `MAP_FAILED`（`(void*)-1`）。

使用按需分页（lazy allocation），首次访问触发页面错误，然后分配物理内存；支持匿名映射（`MAP_ANONYMOUS`）类似 `malloc`，但直接管理虚拟内存；支持映射文件，以内存访问方式读取文件，提高文件 IO 速度。可用于：大块内存分配（替代 `malloc`）；文件映射（如共享库、mmap 文件读写）；线程栈分配（`MAP_STACK`）；共享内存分配等。

#### brk/sbrk
调整进程堆区的大小，用于动态内存分配，适合连续小块内存分配。实现原理：更新 `mm_struct` 的 `brk` 字段，扩展堆区涉及分配新页面。

```c
int brk(void *addr);
void *sbrk(intptr_t increment);
int main() {
    void *old = sbrk(0);
    sbrk(4096);
    void *new = sbrk(0);
    printf("Heap: %p -> %p\n", old, new);
    sbrk(-4096);
    return 0;
}
```

- `brk`：设置堆结束地址。
- `sbrk`：增减堆大小，返回旧堆顶。

C 标准库的 `malloc` 和 `free` 底层依赖 `sbrk` 管理小块内存（< 128KB）。用户态程序很少直接调用，`malloc` 间接使用。

### 内存优化

#### mprotect
更改 `mmap` 或堆内存的访问权限。实现原理：更新进程 mm_struct 中的页面表权限。

```c
int mprotect(void *addr, size_t length, int prot);
mprotect(addr, 4096, PROT_READ); // 设置只读
```

prot：设置 `PROT_READ`、`PROT_WRITE`、`PROT_EXEC`，用于动态调整内存区域的读写执行权限，保护内存区域（如只读代码段），动态加载可执行代码。

#### msync
将 `MAP_SHARED` 映射的内存与底层文件同步。实现原理：再内核中调用文件系统写操作，强制数据刷新到硬盘。

```c
int msync(void *addr, size_t length, int flags);
```

`flags`：`MS_SYNC`（同步写）、`MS_ASYNC`（异步写）。确保内存修改写回文件。

#### madvise
向内核提供内存使用模式建议，优化页面管理。实现原理：调整 VMA 的页面回收策略。

```c
int madvise(void *addr, size_t length, int advice);
```

设置 `advice`：如 `MADV_WILLNEED`（预读）、`MADV_DONTNEED`（释放），优化页面错误和缓存行为，提高内存访问效率（如预读大文件）。

- mlock/munlock：将内存锁定，使得无法 swap 换出
- shmget/shmat/shmdt/shmctl：System V IPC，共享内存老版本接口，现代编程使用 mmap 替代。

## 内核态接口
在 Linux 内核态编程中，内存管理是核心功能，内核提供多种接口用于分配和管理内存，如 `kmalloc`、`vmalloc`、`get_free_pages` 等。这些接口服务于内核模块、驱动程序和其他内核组件，运行在内核上下文，需高效、安全地分配内存。与用户态的 `mmap`、`brk` 等不同，内核态接口直接操作物理内存或内核虚拟地址。

#### kmalloc/kzalloc
分配小块、物理和虚拟连续的内存，使用 `slab` 分配器。

```c
void *kmalloc(size_t size, gfp_t flags);
void *kzalloc(size_t size, gfp_t flags);
```

- 参数：
  - `size`：分配大小（通常 < 128KB，最大受 `KMALLOC_MAX_SIZE` 限制，如 4MB）。
  - `flags`：分配标志（如 `GFP_KERNEL`、`GFP_ATOMIC`）。
- 返回值：分配的内存地址（内核虚拟地址），失败返回 `NULL`。
- 实现：
  - 位置：`mm/slab.c` 或 `mm/slub.c`（SLUB 分配器）。
  - 机制：
    - 使用 `slab` 分配器，从预分配的缓存池（`kmem_cache`）获取内存。
    - 缓存池按大小分级（如 8B、16B、32B），减少碎片。
    - 分配物理连续内存，映射到内核虚拟地址（`PAGE_OFFSET` 以上）。
    - `GFP_KERNEL` 在进程上下文分配，可能睡眠；`GFP_ATOMIC` 在中断上下文分配，非阻塞。
  - 上下文：`GFP_KERNEL` 需进程上下文，`GFP_ATOMIC` 支持中断上下文。

适用于小块内存分配（如驱动中的缓冲区、数据结构）、高性能场景（如网络栈、DMA）。优点：高效（缓存池预分配）、物理连续，适合 DMA。缺点：大小受限（最大 4MB），大块分配可能失败。碎片可能增加（长期运行）。

#### vmalloc
分配大块、虚拟连续的内存，物理上可能不连续，

```c
void *vmalloc(unsigned long size);
```

- 参数：`size` 可达 GB 级别。
- 返回值：内核虚拟地址，失败返回 `NULL`。
- 实现：
  - 位置：`mm/vmalloc.c`。
  - 机制：
    - 使用 `buddy` 分配器分配物理页面（非连续）。
    - 创建内核页面表，映射到连续的内核虚拟地址（`VMALLOC_START` 到 `VMALLOC_END`）。
    - 需要页面表操作，H 系列的 DDR5 加速。
  - 上下文：进程上下文，可能睡眠。

适用于大块内存分配（如内核模块加载、文件系统缓存）。非 DMA 场景（物理不连续）。页面表操作开销大。

#### get_free_pages/\_\_get_free_pages
分配页面对齐的内存（整页，4KB 倍数），底层基于 `buddy` 分配器

```c
void * __get_free_pages(gfp_t gfp_mask, unsigned int order);
```

- 参数：
  - `gfp_mask`：分配标志（如 `GFP_KERNEL`、`GFP_ATOMIC`）。
  - `order`：分配 2^order 页（如 `order=0` 为 4KB，`order=10` 为 4MB）。
- 返回值：页面起始地址（内核虚拟地址），失败返回 `NULL`。
- 实现：
  - 位置：`mm/page_alloc.c`。
  - 机制：
    - 使用 `buddy` 分配器分配物理连续页面。
    - 映射到内核虚拟地址，页面对齐。
    - `GFP_KERNEL` 在进程上下文，`GFP_ATOMIC` 在中断上下文。
  - 上下文：同 `kmalloc`，取决于 `gfp_mask`。

适用于大块、页面对齐的内存（如 DMA 缓冲区），驱动程序中需要整页分配。灵活，支持任意页面数。分配大块内存（高 `order`）可能失败，需手动管理页面。

#### alloc_pages
底层页面分配器，返回 `struct page`
```c
struct page *alloc_pages(gfp_t gfp_mask, unsigned int order);
```
适用于需要直接操作页面（如文件系统、虚拟化）。基于 `buddy` 分配器。

```c
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/gfp.h>

static void *kmalloc_ptr, *vmalloc_ptr, *pages_ptr;

static int __init mem_init(void) {
    // kmalloc 分配 1KB
    kmalloc_ptr = kmalloc(1024, GFP_KERNEL);
    if (!kmalloc_ptr) {
        printk(KERN_ERR "kmalloc failed\n");
        return -ENOMEM;
    }
    strcpy(kmalloc_ptr, "kmalloc data");
    printk(KERN_INFO "kmalloc: %s at %p\n", (char *)kmalloc_ptr, kmalloc_ptr);

    // vmalloc 分配 64KB
    vmalloc_ptr = vmalloc(64 * 1024);
    if (!vmalloc_ptr) {
        printk(KERN_ERR "vmalloc failed\n");
        kfree(kmalloc_ptr);
        return -ENOMEM;
    }
    strcpy(vmalloc_ptr, "vmalloc data");
    printk(KERN_INFO "vmalloc: %s at %p\n", (char *)vmalloc_ptr, vmalloc_ptr);

    // __get_free_pages 分配 2 页 (8KB)
    pages_ptr = __get_free_pages(GFP_KERNEL, 1); // order=1, 2^1 pages
    if (!pages_ptr) {
        printk(KERN_ERR "get_free_pages failed\n");
        vfree(vmalloc_ptr);
        kfree(kmalloc_ptr);
        return -ENOMEM;
    }
    strcpy(pages_ptr, "pages data");
    printk(KERN_INFO "get_free_pages: %s at %p\n", (char *)pages_ptr, pages_ptr);

    return 0;
}

static void __exit mem_exit(void) {
    if (pages_ptr)
        free_pages((unsigned long)pages_ptr, 1);
    if (vmalloc_ptr)
        vfree(vmalloc_ptr);
    if (kmalloc_ptr)
        kfree(kmalloc_ptr);
    printk(KERN_INFO "Memory freed\n");
}

module_init(mem_init);
module_exit(mem_exit);
MODULE_LICENSE("GPL");
```
