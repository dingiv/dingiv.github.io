# 物理内存管理
主板和 BIOS 程序在上电时：
1. 检查和扫描设备
2. 初始化各种设备（内存条和各种 IO 设备的寄存器和缓冲区）
3. 将这些存储空间拼接成连贯的物理内存空间

物理内存空间可以看作是一个地址数组，每个地址的大小取决于计算机的位数（32 位、64 位等）。在操作系统引导时，BIOS 将物理内存空间信息告知操作系统，包括：内存地址的分区和各个硬件设备的地址范围；操作系统可以在其中确定内存条设备的内存范围从而得知主内存的范围。

## 主内存管理

### 位图方法
将物理内存条划分为**头+体**的两块，头使用 bit 位标记了体中的某一块内存是否被分配使用，体中的某块内存可以直接与上层的内存页大小保持一致，采用 4k 为一块进行管理和分配；该算法实现简单，但是在新增分配的时候需要使用 O(n) 时间查找目标，时间效率不高。

### 伙伴系统
buddy system 使用多级链表数组管理空闲块，是一种用于分配连续物理页框的算法。可以用于解决位图方法在分配内存时的速度问题。内核为分配页框调用 __get_free_pages()、alloc_pages() 等接口时，就会走这个算法。当释放的两个“伙伴”空闲块连续时，可以合并为更大的块。

系统将空闲页面分组为 11 个块链表，每个块链表分别包含大小为1、2、4、8、16、32、64、128、256、512和1024个连续页框的页块。最大可以申请 1024 个连续页框，对应 4MB 大小的内存。

![](./buddy.dio.svg)

分配过程：
1. 根据请求大小，找到对应的块链表
2. 如果链表为空，则向更大的块链表申请
3. 将大块分割成两个小块，一个用于分配，另一个加入较小的块链表
4. 如果仍然没有合适的块，则继续向更大的块链表申请

分配过程：
```
空闲块（1024KB）  
→ 分配 128KB 时 → 找不到正好128KB，就分成两个 512KB → 再分两个 256KB → 再分两个 128KB

分配成功：128KB
其“伙伴”：另一个 128KB 仍空闲

当释放该 128KB，如果它的“伙伴”也空闲，则两者合并为 256KB，继续向上尝试合并。
```

释放过程：
1. 将释放的块加入对应的块链表
2. 检查是否有相邻的空闲块
3. 如果有，则合并成更大的块，并加入更大的块链表
4. 重复步骤2-3，直到无法继续合并

优点：，快速分配和释放，有效减少内存碎片，支持大块内存分配；缺点：，可能造成内部碎片，合并操作可能较慢，不适合小块内存分配。

### SLAB 分配器
SLAB 分配器是 Linux 内核中用于管理小块内存的分配器，主要用于内核对象的分配和释放。它基于以下思想：内核对象在创建和销毁时，需要频繁地分配和释放内存，如果每次都使用伙伴系统，会造成很大的开销。

SLAB分配器的主要特点：
1. 对象缓存，为每种对象类型创建专用缓存，缓存中保存已分配但未使用的对象，避免频繁的内存分配和释放
2. 内存着色，通过偏移对象在缓存中的位置，提高CPU缓存的命中率，减少缓存行冲突
3. 三级结构，缓存描述符（kmem_cache），SLAB描述符（slab），对象数组
4. 分配策略，首先从部分满的SLAB中分配，如果没有，则从空的SLAB中分配，如果都没有，则创建新的SLAB

对于小于一页的内存分配请求，Linux提供了以下机制：

1. kmalloc，基于SLAB分配器，支持不同大小的缓存，支持内存对齐要求，适用于内核对象
2. vmalloc，分配虚拟内存，不保证物理内存连续，适用于大块内存，性能较低
3. 内存池，预分配内存块，快速分配和释放，减少内存碎片，适用于特定场景

### 内存管理策略
1. 内存回收
   - 页面回收
   - 缓存回收
   - 交换机制
   - OOM killer
2. 内存压缩
   - 页面压缩
   - 内存碎片整理
   - 提高内存利用率
3. 内存监控
   - 内存使用统计
   - 内存压力检测
   - 性能分析
   - 问题诊断
4. 内存调优
   - 参数配置
   - 性能优化
   - 资源限制
   - 负载均衡