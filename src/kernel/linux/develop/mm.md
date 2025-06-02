# 内存分配

讲讲 kmalloc、vmalloc、get_free_pages 的区别


ChatGPT 说：
这三个函数 kmalloc、vmalloc 和 get_free_pages 是 Linux 内核中常用的 内存分配接口，虽然都是用来在内核空间中申请内存的，但它们的使用场景、分配方式和返回的内存特性都不相同。

🧱 一、三者对比表（核心总结）
函数	分配空间	分配大小	地址是否连续	物理连续性	用途
kmalloc	内核空间	小块（<=2MB）	✔ 虚拟地址连续	✔ 物理地址连续	适用于设备驱动、DMA等
vmalloc	内核空间	大块	✔ 虚拟地址连续	✘ 物理地址不连续	适合大块内存，不用于 DMA
get_free_pages	页框（低级接口）	以页为单位	✔ 虚拟地址连续	✔ 物理地址连续	分配多页连续物理内存

🔍 二、详细解释
1️⃣ kmalloc
函数原型：

c
复制
编辑
void *kmalloc(size_t size, gfp_t flags);
特点：

返回的是物理和虚拟地址都连续的内存。

分配小块内存效率高（小于 PAGE_SIZE 到几 MB）。

使用 slab/slub/slob 分配器 管理。

可用于 DMA（因为物理地址连续）。

限制：

分配大于某个限制（通常是 order-10，约等于 4MB）时可能失败。

不适合申请非常大的内存块。

示例用途：

c
复制
编辑
struct my_struct *ptr = kmalloc(sizeof(*ptr), GFP_KERNEL);
2️⃣ vmalloc
函数原型：

c
复制
编辑
void *vmalloc(unsigned long size);
特点：

返回的内存是 虚拟地址连续，但物理地址不连续。

适用于分配较大的内存块（比如几十 MB）。

内存通过页表映射到非连续物理页。

不能用于 DMA。

适合场景：

数据缓冲区、文件缓存等对性能要求不高的大块内存。

示例用途：

c
复制
编辑
void *buf = vmalloc(10 * 1024 * 1024); // 分配 10MB
3️⃣ get_free_pages
函数原型：

c
复制
编辑
unsigned long get_free_pages(gfp_t flags, unsigned int order);
特点：

分配的是 2^order 个页 的内存（页大小通常为 4KB）。

物理和虚拟地址都连续。

比 kmalloc 更底层，主要用于内核内存管理系统。

返回的是 unsigned long 类型的页首地址（非 void*）。

释放内存：

c
复制
编辑
free_pages(addr, order);
适合场景：

内核子系统内部、内核初始化时使用，或显式需要页对齐/页数量的情况。

使用示例：

c
复制
编辑
unsigned long addr = get_free_pages(GFP_KERNEL, 2); // 分配 4 页 (16KB)
🚫 注意事项
使用错误	原因说明
kmalloc 申请太大	会失败，造成内核警告或崩溃
用 vmalloc 做 DMA	错误，DMA 需要物理连续内存
忘记释放内存	内核态内存泄漏会引起严重问题

✅ 总结建议
一般用途（小块） → kmalloc

大块内存（几 MB 以上） → vmalloc

手动管理页框 / 特殊需求 → get_free_pages

是否还想进一步了解它们的源码实现，或在模块中如何正确使用这几个接口？我可以继续讲。