# 内存回收
当系统内存紧张时，内核需要回收部分物理页面以避免内存耗尽。内存回收是一个多层次、多策略的过程，从最轻量的页面缓存回收到最极端的 OOM Killer 终止进程，形成了一条由轻到重的处理链。

## 回收对象

内核可回收的内存分为几类，按回收代价从低到高排列：

页缓存是代价最低的回收对象。页缓存用于缓存文件内容，如果页面未被修改（clean page），可以直接丢弃，需要时从磁盘重新读取；如果页面已被修改（dirty page），需要先写回磁盘再释放。页缓存回收不会影响进程的正确性，但可能导致后续的文件访问需要重新从磁盘加载。

匿名页是进程的堆、栈等私有数据，这类页面没有对应的磁盘文件，不能直接丢弃。内核必须将其写入 swap 交换分区后才能释放物理内存。swap 的代价远高于页缓存回收，因为涉及磁盘 I/O。

内核缓存（SLUB 分配器的缓存）包括 dentry（目录项）、inode 等内核对象。这些缓存用于加速内核操作，内存紧张时可以回收部分未被使用的对象。内核通过 `drop_caches` 机制允许手动触发缓存回收：`echo 1 > /proc/sys/vm/drop_caches` 释放页缓存，`echo 2 > /proc/sys/vm/drop_caches` 释放页缓存和 dentry/inode 缓存，`echo 3 > /proc/sys/vm/drop_caches` 释放全部可回收缓存。

## LRU 链表

内核通过 LRU（Least Recently Used，最近最少使用）算法管理活跃页和不活跃页。每个 NUMA 节点维护两组 LRU 链表：**活跃 LRU（active list）** 存放最近被访问过的页面，**不活跃 LRU（inactive list）** 存放较长时间未被访问的页面。

页面首次被访问时加入活跃 LRU 的头部。当活跃 LRU 过长时，尾部的页面被移到不活跃 LRU 的头部。内核回收时优先从活跃 LRU 的尾部扫描：如果页面近期被访问过（通过 accessed 标志位判断），将其移回活跃 LRU 头部；如果未被访问，将其移到不活跃 LRU 头部。不活跃 LRU 尾部的页面是回收的首选目标。

现代内核（2.6.x 之后）将 LRU 链表按页面类型分为**匿名页 LRU**和**文件页 LRU**，每种类型各有 active 和 inactive 两条链表。这样在内存紧张时，内核可以优先回收文件页（代价低），当文件页不足以满足回收需求时再考虑回收匿名页（代价高，需要 swap）。

## kswapd 与直接回收

内存回收分为两种触发方式。**后台回收（kswapd）** 是 per-CPU 的内核线程，定期检查每个 NUMA 节点的水位线（watermark）。当空闲页面数量低于 `high watermark` 时，kswapd 开始异步回收页面，直到空闲页面恢复到 `high watermark` 以上。后台回收在工作负载低谷时进行，不影响进程的正常执行。

**直接回收（direct reclaim）** 发生在进程分配内存时发现空闲页面低于 `min watermark`。此时进程被阻塞，内核同步执行页面回收，直到回收出足够的内存满足分配请求。直接回收会导致进程的延迟显著增加，频繁触发意味着系统内存压力过大。

三个水位线的配置在 `/proc/sys/vm/` 中：`min_free_kbytes` 设置最小预留内存量，影响三个水位线的绝对位置。`watermark_scale_factor`（内核 4.x+）控制 high 和 low 之间的间距比例，间接影响 kswapd 的触发敏感度。

## [内存交换](./swap)

当内存紧张到需要回收匿名页时，内核将匿名页写入 swap 交换分区。详细的交换机制和页替换策略参见[内存交换](./swap)。

## 内存压缩

内存压缩在回收和 swap 之间提供了一层缓冲。内核将不活跃页面压缩后仍保存在内存中，而不是直接写入磁盘 swap 分区。这样在需要访问被压缩页面时，只需解压即可恢复，避免了磁盘 I/O。压缩的代价是 CPU 开销（时间换空间）。

**zswap** 是内核自带的压缩后交换缓存。被换出的匿名页先在内存中压缩存储，只有当 zswap 的内存池满了才真正写入 swap 设备。zswap 使用 LRU 算法管理压缩后的页面，长时间未访问的页面会被解压并写入 swap 设备以释放 zswap 的内存空间。

**zram** 将一块内存虚拟成块设备，作为压缩 swap 区使用。zram 适合嵌入式和低内存设备（如 Android、ChromeOS），物理内存有限且没有专用 swap 分区或磁盘的场景。zram 与 zswap 的区别在于：zram 是一个独立的块设备，直接替代 swap 分区；zswap 是 swap 路径上的压缩缓存，仍然需要后端的 swap 设备。

## 内存碎片

内存碎片是指物理内存中存在无法被有效利用的小块空闲区域，影响大块连续物理内存的分配。内存碎片在需要连续物理内存的场景下尤为突出，如驱动 DMA、大页分配、内核模块加载等。

外部碎片是指总空闲内存足够但分散在不连续的位置，无法满足大块连续分配请求。伙伴系统通过合并相邻空闲块来缓解外部碎片，但在长时间运行后仍然可能积累。内核提供了**内存紧缩（Memory Compaction）**机制来解决外部碎片：扫描物理内存，将分散的活跃页面迁移到一侧，将空闲页面集中到另一侧，形成大块连续空闲区。内存紧缩可以自动触发（大块分配失败时），也可以手动触发：`echo 1 > /proc/sys/vm/compact_memory`。

内部碎片是指分配的内存块大于实际需要，块内存在未被使用的空间。SLUB 分配器通过为不同大小的对象建立专用缓存来减少内部碎片。大页机制下内部碎片更为明显（一个 2MB 大页即使只用了 4KB，也整页被占用），需要根据实际负载权衡使用。

## OOM Killer

当所有回收手段（页缓存回收、匿名页 swap、内存压缩、内存紧缩）都无法满足内存分配需求时，内核启动 OOM Killer，选择性地终止进程以释放内存。

OOM Killer 的选择标准基于 oom_score 评分，存储在每个进程的 `/proc/<pid>/oom_score` 中。评分综合考虑以下因素：进程占用的物理内存越多，分数越高；进程是否为 root 用户（root 进程的分数被调低）；进程的 oom_score_adj（可通过 `/proc/<pid>/oom_score_adj` 手动调整，-1000 表示禁止被 OOM Kill，+1000 表示最高优先级被 Kill）。

OOM Killer 触发时会打印日志到 dmesg，包含被选中进程的 PID、名称、内存占用和评分信息。在生产环境中，可以通过设置 `vm.panic_on_oom=1` 让系统在 OOM 时直接 panic 重启（适用于对内存可用性要求极高的场景），或者配置 `oomd`（systemd-oomd）在用户空间更智能地管理 OOM。

## 监控与调优

- `/proc/meminfo`：系统整体内存使用情况（MemTotal、MemFree、MemAvailable、Buffers、Cached、SwapCached 等）。`MemAvailable` 是更准确的可用内存估算，考虑了可回收的缓存。
- `/proc/buddyinfo`：各阶空闲内存块分布，用于分析外部碎片。
- `/proc/pagetypeinfo`：按 Movable/Reclaimable/Unmovable 类型分类的空闲页面统计。
- `/proc/vmstat`：虚拟内存统计，包括 pgscan（扫描的页面数）、pgsteal（回收的页面数）、pswpin/pswpout（swap 读写字节数）、compact_*（紧缩统计）。
- `/proc/pressure/memory`（PSI）：内存压力指标（avg10/avg60/avg300），反映内存紧张程度对任务执行的影响。
- `free -h`：快速查看内存和 swap 使用概况。
- `slabtop`：实时监控 SLUB 缓存的内存占用。

常见调优参数：`vm.swappiness` 控制 swap 倾向（0 表示尽量避免 swap，100 表示积极 swap，默认 60）；`vm.min_free_kbytes` 设置预留最小空闲内存；`vm.dirty_ratio` 和 `vm.dirty_background_ratio` 控制脏页写回阈值；`vm.compaction_proactiveness` 控制内存紧缩积极性（0-100，默认不启用主动紧缩）。
