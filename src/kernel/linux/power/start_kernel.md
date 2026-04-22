# 内核启动

Linux 内核 x86 启动流程全景

启动链路如下（从上电到用户空间）：

Bootloader (GRUB/UEFI)
  └─→ arch/x86/boot/header.S    _start            [16位实模式]
      └─→ arch/x86/boot/main.c   main()            [实模式 C 代码]
          └─→ arch/x86/boot/pm.c  go_to_protected_mode()
              └─→ arch/x86/boot/pmjump.S            [切换到保护模式]
                  └─→ arch/x86/boot/compressed/head_64.S  startup_32  [32位保护模式]
                      └─→ startup_64                               [64位长模式]
                          └─→ arch/x86/boot/compressed/misc.c  extract_kernel()  [解压]
                              └─→ arch/x86/kernel/head_64.S  startup_64  [内核主体]
                                  └─→ arch/x86/kernel/head64.c  x86_64_start_kernel()
                                      └─→ init/main.c  start_kernel()  [架构无关初始化]
                                          └─→ rest_init() → kernel_init (PID 1) → /sbin/init

  四个阶段概要

  阶段1 — 实模式启动 (arch/x86/boot/)

  ┌──────────────┬────────────────────────┬───────────────────────────────────────────────────┐
  │     文件     │         关键行         │                       作用                        │
  ├──────────────┼────────────────────────┼───────────────────────────────────────────────────┤
  │ header.S:246 │ _start                 │ 引导程序跳入的入口，2字节短跳到 start_of_setup    │
  ├──────────────┼────────────────────────┼───────────────────────────────────────────────────┤
  │ header.S:234 │ hdr                    │ 与引导程序之间的协议头（启动参数、载荷位置等）    │
  ├──────────────┼────────────────────────┼───────────────────────────────────────────────────┤
  │ setup.ld     │ ENTRY(_start)          │ 链接脚本，约束 setup 不超过 32KB                  │
  ├──────────────┼────────────────────────┼───────────────────────────────────────────────────┤
  │ main.c:133   │ main()                 │ 实模式主函数：检测内存(E820)、设视频模式、检测CPU │
  ├──────────────┼────────────────────────┼───────────────────────────────────────────────────┤
  │ pm.c:103     │ go_to_protected_mode() │ 开启A20、设置GDT/IDT、切换到保护模式              │
  └──────────────┴────────────────────────┴───────────────────────────────────────────────────┘

  阶段2 — 解压stub (arch/x86/boot/compressed/)

  ┌───────────────┬──────────────────┬────────────────────────────────────────────────┐
  │     文件      │      关键行      │                      作用                      │
  ├───────────────┼──────────────────┼────────────────────────────────────────────────┤
  │ head_64.S:82  │ startup_32       │ 从保护模式进入，建页表，开启长模式             │
  ├───────────────┼──────────────────┼────────────────────────────────────────────────┤
  │ head_64.S:278 │ startup_64       │ 64位入口，自定位，处理5级分页/SEV加密          │
  ├───────────────┼──────────────────┼────────────────────────────────────────────────┤
  │ head_64.S:445 │ .Lrelocated      │ 清BSS，调用 extract_kernel() 解压，跳入内核    │
  ├───────────────┼──────────────────┼────────────────────────────────────────────────┤
  │ misc.c:407    │ extract_kernel() │ KASLR随机化地址 + 解压(gzip/xz/zstd) + ELF解析 │
  └───────────────┴──────────────────┴────────────────────────────────────────────────┘

  阶段3 — 内核主体入口 (arch/x86/kernel/)

  ┌───────────────┬───────────────────────┬────────────────────────────────────────┐
  │     文件      │        关键行         │                  作用                  │
  ├───────────────┼───────────────────────┼────────────────────────────────────────┤
  │ head_64.S:38  │ startup_64            │ BSP入口：设GDT/IDT、修页表、配CR4/EFER │
  ├───────────────┼───────────────────────┼────────────────────────────────────────┤
  │ head_64.S:198 │ common_startup_64     │ BSP和AP共用：配per-CPU栈、APIC ID查找  │
  ├───────────────┼───────────────────────┼────────────────────────────────────────┤
  │ head_64.S:479 │ initial_code          │ 指向 x86_64_start_kernel——第一个C函数  │
  ├───────────────┼───────────────────────┼────────────────────────────────────────┤
  │ head64.c:222  │ x86_64_start_kernel() │ 清页表、KASAN、拷贝启动参数、加载微码  │
  └───────────────┴───────────────────────┴────────────────────────────────────────┘

  阶段4 — 通用内核初始化 (init/main.c)

  ┌──────┬────────────────┬───────────────────────────────────────────────────────────────────────────┐
  │ 行号 │      函数      │                                   作用                                    │
  ├──────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ 1008 │ start_kernel() │ 架构无关初始化总入口                                                      │
  ├──────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ —    │ setup_arch()   │ ACPI、e820内存映射、SMP等                                                 │
  ├──────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ —    │ trap_init()    │ 中断/异常处理                                                             │
  ├──────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ —    │ sched_init()   │ 调度器                                                                    │
  ├──────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ —    │ console_init() │ 控制台可用                                                                │
  ├──────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ 714  │ rest_init()    │ 创建 PID 1 (kernel_init→/sbin/init)、PID 2 (kthreadd)，自身成为 idle 进程 │
  └──────┴────────────────┴───────────────────────────────────────────────────────────────────────────┘

  bzImage 构建结构

  内核源码全部编译 → vmlinux (未压缩ELF)
      → strip → vmlinux.bin → gzip/xz → vmlinux.bin.gz
          → piggy.S (嵌入压缩数据) + 解压stub → compressed/vmlinux
              → objcopy → vmlinux.bin
                  + setup.bin (实模式代码)
                  → bzImage (最终引导镜像)

  建议从 arch/x86/boot/header.S 的 _start 开始顺着调用链读，这是理解整个启动流程最直接的方式。需要深入哪个阶段可以继续问。


❯ 注意到在 start_kernel 之前，内存的页表已经建立，那么在 start_kernel 中内存初始化工作内核是在做什么？                                                                                                                            

● Explore(Explore start_kernel memory init)
  ⎿  Done (135 tool uses · 69.6k tokens · 9m 52s)
  (ctrl+o to expand)

● 好问题。启动阶段的页表只映射了内核镜像本身，远远不够。start_kernel() 需要从零构建整套内存管理基础设施。

  核心脉络：从 "能跑" 到 "能分配"

  启动前页表做的事：
  head_64.S: 只映射了内核代码段 + 少量 RAM → 勉强能执行

  start_kernel() 要做的事（按执行顺序）：

  第一步：摸清家底 — setup_arch() (init/main.c:1030)

  BIOS e820 表（固件给的物理内存分布）
      │
      ▼
  e820__memory_setup()          # 获取物理内存映射图
      │
      ▼
  e820__memblock_setup()        # 填充 memblock 早期分配器
      │                         # 此后 memblock_alloc() 可用
      ▼
  init_mem_mapping()            # 建立全部物理内存的直接映射页表
      │                         # arch/x86/mm/init.c:758
      ▼
  swapper_pg_dir 加载到 CR3     # 替换掉早期临时页表

  这一步的关键转变：从只映射内核镜像 → 映射所有物理 RAM 到内核虚拟地址空间。

  第二步：建立 zone/node 结构 — free_area_init() (mm/mm_init.c:1820)

  内核把物理内存划分为 zone（以 x86-64 为例）：

  物理内存: 0 ── 16MB ──── 4GB ──────────────────────── max_pfn
             │        │      │                            │
           ZONE_DMA  ZONE_DMA32    ZONE_NORMAL
    (ISA DMA)  (32位设备DMA)    (普通内存)

  free_area_init() 做的事：
  1. arch_zone_limits_init() — 确定每个 zone 的 PFN 边界
  2. sparse_init() — 分配每个内存 section 的 struct page 数组（vmemmap）
  3. free_area_init_node() — 初始化每个 NUMA 节点的 pg_data_t 和 zone 结构
  4. memmap_init() — 逐个初始化每个 struct page（清零、设 zone/node/pfn、引用计数置1）

  第三步：启动伙伴分配器 — mm_core_init() (init/main.c:1070)

  这是最关键的转变——从 memblock 到 buddy：

  memblock_free_all()              # mm/memblock.c:2340
      │
      ├─ 遍历 memblock 中所有空闲区间
      │       │
      │       ▼
      │   __free_pages_core()      # 清除 PageReserved，引用计数归零
      │       │
      │       ▼
      │   __free_pages_ok()        # 放入伙伴系统的空闲链表
      │
      ▼
    此后 alloc_pages() / __get_free_pages() 可用

  紧接着建立上层的分配器：

  kmem_cache_init()    # SLUB 分配器启动 → kmalloc() 可用
  vmalloc_init()       # vmalloc 分配器启动 → vmalloc() 可用
  mm_cache_init()      # mm_struct 的 slab 缓存 → fork() 可用

  整体流程图

  setup_arch()
    │
    ├─ e820 固件内存表 ──→ memblock 早期分配器
    ├─ init_mem_mapping() ──→ 完整直接映射页表（替换临时页表）
    │
    ▼
  free_area_init()
    │
    ├─ 划分 ZONE_DMA / DMA32 / NORMAL / HIGHMEM
    ├─ 分配并初始化所有 struct page（每个物理页一个）
    ├─ 初始化 zone 的锁、空闲链表、per-cpu 统计
    │
    ▼
  mm_core_init()
    │
    ├─ build_all_zonelists()     ← zone 回退链表（NORMAL 满了找 DMA32）
    ├─ memblock_free_all()       ← memblock 空闲页 → 伙伴系统
    ├─ kmem_cache_init()         ← SLUB 启动，kmalloc 可用
    ├─ vmalloc_init()            ← vmalloc 启动
    │
    ▼
  setup_per_cpu_pageset()         ← per-CPU 页缓存
  mem_cgroup_init()               ← 内存 cgroup 控制器

  为什么早期页表不够？

  ┌──────────────────────────┬────────────────────────────────────┐
  │         早期页表         │        start_kernel 建立的         │
  ├──────────────────────────┼────────────────────────────────────┤
  │ 只映射内核镜像 (~几十MB) │ 映射所有物理 RAM                   │
  ├──────────────────────────┼────────────────────────────────────┤
  │ 没有内存分配能力         │ memblock → buddy → slab 三级分配器 │
  ├──────────────────────────┼────────────────────────────────────┤
  │ 没有 zone 概念           │ DMA/DMA32/NORMAL/HIGHMEM 分区      │
  ├──────────────────────────┼────────────────────────────────────┤
  │ 没有 struct page         │ 每个物理页都有元数据               │
  ├──────────────────────────┼────────────────────────────────────┤
  │ 没有 NUMA 感知           │ 按节点组织内存                     │
  └──────────────────────────┴────────────────────────────────────┘

  简单说，早期页表只是让内核能跑起来的脚手架，start_kernel() 要在此基础上建起整栋内存管理的大厦。

✻ Crunched for 11m 6s

❯ 我比较好奇的是，从 memblock 到 多级页表，这个过程是如何建立的，(例如在 @arch/x86/kernel/acpi/boot.c 的 acpi_boot_init 函数中，还有 函数 @arch/x86/mm/init_64.c  fill_pmd )                                                      
  ⎿  Read arch/x86/kernel/acpi/boot.c (1851 lines)
  ⎿  Read arch/x86/mm/init_64.c (1657 lines)

● 好问题。让我追踪从 memblock 分配到多级页表建立的具体过程。

  Searched for 2 patterns, read 1 file (ctrl+o to expand)

● 现在我把整个链路串起来，用一个具体的例子说明。

  从 memblock 到多级页表的建立过程

  核心问题：先有鸡还是先有蛋？

  建立页表需要物理页来存放页表项，但分配物理页又需要页表来映射。内核用两种机制解决了这个循环依赖：

  第一步：预留页表缓冲区（brk 机制）

  setup_arch()
    └─ early_alloc_pgt_buf()        # arch/x86/mm/init.c:187
         │
         │  extend_brk() 从内核镜像末尾的 brk 区域划出 INIT_PGT_BUF_SIZE
         │  这是一块静态预留的内存，不需要分配器
         │
         ├─ pgt_buf_start = 起始 PFN
         ├─ pgt_buf_end   = 起始 PFN    （当前分配位置）
         └─ pgt_buf_top   = 起始 PFN + 预留页数 （上限）

  这就是 alloc_low_pages() 的第一种来源——从 brk 缓冲区顺序分配：

  // arch/x86/mm/init.c:126
  void *alloc_low_pages(unsigned int num)
  {
      // 来源1: brk 预留缓冲区
      if (pgt_buf_end + num <= pgt_buf_top) {
          pfn = pgt_buf_end;
          pgt_buf_end += num;    // 线性推进
      }
      // 来源2: brk 用完了，从 memblock 分配
      else {
          ret = memblock_phys_alloc_range(...);  // 只从已映射的范围内分配
          if (!ret)
              ret = __pa(extend_brk(...));       // 实在不行再扩展 brk
      }
      clear_page(__va(pfn << PAGE_SHIFT));  // 清零
      return __va(pfn << PAGE_SHIFT);       // 返回虚拟地址
  }

  关键点：memblock 在这里只做简单的物理地址区间记录，不涉及页表映射——因为这些页已经通过 __va() 直接访问（早期恒等映射还在）。

  第二步：自顶向下遍历，逐级填充页表

  以 init_mem_mapping() 为入口，完整的调用链：

  init_mem_mapping()                    # arch/x86/mm/init.c:758
    │
    ├─ init_memory_mapping(0, ISA_END_ADDRESS)   # 先映射 ISA 区域 (0-16MB)
    │     │
    │     ├─ split_mem_range()     # 按 1G/2M/4K 边界切分物理地址范围
    │     │
    │     └─ kernel_physical_mapping_init()       # arch/x86/mm/init_64.c:798
    │           │
    │           └─ __kernel_physical_mapping_init()  # :741
    │                 │
    │                 │  遍历每个 PGD 范围 (512GB):
    │                 ├─ pgd = pgd_offset_k(vaddr)
    │                 │
    │                 │  遍历每个 P4D 范围:
    │                 ├─ phys_p4d_init()           # :690
    │                 │    │
    │                 │    │  遍历每个 PUD 范围 (1GB):
    │                 │    ├─ phys_pud_init()      # :607
    │                 │    │    │
    │                 │    │    │  如果支持 1G 大页且对齐:
    │                 │    │    ├─ set_pud(pfn_pud(..., prot_sethuge()))  # 直接映射 1G
    │                 │    │    │
    │                 │    │    │  否则，分配 PMD 页:
    │                 │    │    ├─ pmd = alloc_low_page()      # ★ 从 brk/memblock 拿一页
    │                 │    │    ├─ phys_pmd_init(pmd, ...)     # 递归填充 PMD
    │                 │    │    └─ pud_populate_init(pud, pmd) # 把 PMD 页挂到 PUD 下
    │                 │    │
    │                 │    │    phys_pmd_init() 内部:
    │                 │    │    │  如果支持 2M 大页且对齐:
    │                 │    │    ├─ set_pmd(pfn_pmd(...))      # 直接映射 2M
    │                 │    │    │
    │                 │    │    │  否则，分配 PTE 页:
    │                 │    │    ├─ pte = alloc_low_page()      # ★ 再拿一页
    │                 │    │    ├─ phys_pte_init(pte, ...)     # 填充每个 PTE 项
    │                 │    │    └─ pmd_populate_kernel_init(pmd, pte)
    │                 │    │         # 每个 PTE: set_pte(pfn_pte(paddr >> PAGE_SHIFT, prot))
    │                 │
    │                 │    └─ (同上，P4D → PUD 的挂接)
    │                 │
    │                 └─ sync_global_pgds()  # 同步到所有进程的 PGD
    │
    ├─ memory_map_top_down(ISA_END_ADDRESS, end)  # 自顶向下映射剩余内存
    │     │
    │     │  按步长从高地址向低地址推进，每步调用:
    │     └─ init_range_memory_mapping()
    │           │
    │           └─ for_each_mem_pfn_range()       # 只映射 e820 中的 RAM 区域
    │                 └─ init_memory_mapping()
    │                       └─ kernel_physical_mapping_init()  # 同上
    │
    └─ load_cr3(swapper_pg_dir)   # 切换到完整页表！
        __flush_tlb_all()          # 刷新 TLB

  第三步：用具体数字说明

  假设系统有 8GB 内存，4级页表（PGD → PUD → PMD → PTE）：

  物理地址: 0x0000_0000_0000 ~ 0x0000_0001_FFFF_FFFF (8GB)

  直接映射区虚拟地址: 0xFFFF_8880_0000_0000 ~ ... (线性映射所有物理内存)

  对 0x0000_0000_4000_0000 (1GB 处) 建立映射:

    1. pgd_offset_k(vaddr)  → 找到 PGD 中对应的项
       如果为空 → alloc_low_page() 分配一个 PUD 页

    2. pud = pud_page + pud_index(vaddr)
       如果支持 1G 大页 → set_pud(pfn_pud(1GB物理页号, PAGE_KERNEL))
       一步映射 1GB！只需一个 PUD 项

       如果不支持 → alloc_low_page() 分配一个 PMD 页
       → pud_populate(pud, pmd)  把 PMD 页物理地址写入 PUD 项

    3. 对 PMD 页中的每个项:
       如果支持 2M 大页 → set_pmd(pfn_pmd(2MB物理页号, PAGE_KERNEL))
       一步映射 2MB

       如果不支持 → alloc_low_page() 分配一个 PTE 页
       → 对 512 个 PTE 逐个: set_pte(pfn_pte(4KB物理页号, PAGE_KERNEL))

  页表项的物理内容（以 PTE 为例）：
  PTE = | 物理页帧号 | FLAGS |
        | 0x40000    | PRESENT|RW|ACCESSED|DIRTY|GLOBAL|... |
        ↑
        指向物理页，CPU 硬件 MMU 自动解析

  两个关键的 "fill" 函数对比

  你提到的 fill_pmd 在 init_64.c:291，它和 phys_pmd_init 做的事类似但有区别：

  // fill_pmd — 用于 fixmap / 任意虚拟地址的单独映射
  static pmd_t *fill_pmd(pud_t *pud, unsigned long vaddr)
  {
      if (pud_none(*pud)) {
          pmd_t *pmd = (pmd_t *)spp_getpage();    // 从 memblock 分配
          pud_populate(&init_mm, pud, pmd);        // 挂到 PUD 下
      }
      return pud_offset(pud, vaddr);               // 返回对应 vaddr 的 PMD 项指针
  }

  // phys_pmd_init — 用于批量映射整段物理内存
  static unsigned long phys_pmd_init(pmd_t *pmd_page, ...)
  {
      for (i = 0; i < PTRS_PER_PMD; i++) {
          if (page_size_mask & (1<<PG_LEVEL_2M))
              set_pmd(pmd, pfn_pmd(paddr >> PAGE_SHIFT, prot_sethuge(prot)));
              // 大页: 一个 PMD 项直接映射 2MB 物理内存
          else {
              pte = alloc_low_page();
              phys_pte_init(pte, ...);              // 逐个 PTE 填充
              pmd_populate_kernel(pmd, pte);        // PMD → PTE
          }
      }
  }

  内存分配的时间线

  head_64.S 阶段:
    早期页表用静态分配（写在 .bss 段里的 pgt_buf）

  setup_arch() 阶段:
    early_alloc_pgt_buf()    → brk 预留 ~几十个页表页
    init_mem_mapping()       → 开始逐级填充
      alloc_low_pages()      → 先消耗 brk 缓冲区
                            → brk 不够时从 memblock 分配
      memblock_alloc()       → 只记录物理地址，不做页表映射

  mm_core_init() 阶段:
    memblock_free_all()      → memblock 的空闲页全部交给 buddy
    after_bootmem = 1        → 之后的页表分配走 get_zeroed_page() / slab

  所以页表页的分配在 memblock 之前就开始了——先用 brk 静态预留的内存，再用 memblock 补充，最后 memblock 完成使命后把所有剩余内存移交给 buddy 系统。


  在 start_kernel() 之前，%cr3 指向的 PGD 表项构造过程

  整个过程分为 三个阶段，涉及多次 %cr3 切换：

  ---
  阶段一：解压器（decompressor）构建最初的页表

  文件：arch/x86/boot/compressed/head_64.S

  1. startup_32（行 82）— 从 32 位保护模式进入时，在汇编中手动构建一个 4GB 恒等映射的三级页表：
    - pgtable + 0x0000：Level 4（1 项 → L3）
    - pgtable + 0x1000：Level 3（4 项 → 4 个 L2）
    - pgtable + 0x2000：Level 2（2048 项，用 2MB 大页映射 4GB）
    - 行 234-235：第一次写入 %cr3：movl %eax, %cr3
  2. initialize_identity_maps()（arch/x86/boot/compressed/ident_map_64.c）— 读取当前 %cr3，复用或新建顶层页表，调用 kernel_ident_mapping_init() 映射内核映像、命令行等区域，行 183：write_cr3(top_level_pgt)
  3. 若需在 4 级/5 级分页间切换，configure_5level_paging()（arch/x86/boot/compressed/pgtable_64.c）会再写一次 %cr3

  ---
  阶段二：内核本体 startup_64 切换到 early_top_pgt

  文件：arch/x86/kernel/head_64.S（行 38）

  此时 %cr3 指向解压器留下的恒等映射页表。内核 不从头构建页表，而是做 KASLR 修正：

  - __startup_64()（arch/x86/boot/startup/map_kernel.c 行 87）对静态页表做 fixup：
    - 修正 early_top_pgt[511]（指向 level3_kernel_pgt）
    - 在 early_dynamic_pgts[] 中创建恒等映射（用于物理→虚拟地址跳转）
    - 修正 level2_kernel_pgt 的 PMD 项
  - 回到 head_64.S 行 136：写入 %cr3，切换到修正后的 early_top_pgt

  关键的 静态页表数据结构 定义在 head_64.S 行 603-673：

  ┌───────────────────┬────────────────────────────────────────────────────────────────┐
  │       符号        │                              说明                              │
  ├───────────────────┼────────────────────────────────────────────────────────────────┤
  │ early_top_pgt     │ 启动过渡 PGD，仅 entry[511] 非零（→ level3_kernel_pgt）        │
  ├───────────────────┼────────────────────────────────────────────────────────────────┤
  │ init_top_pgt      │ 最终运行时 PGD（即 swapper_pg_dir），初始全零                  │
  ├───────────────────┼────────────────────────────────────────────────────────────────┤
  │ level3_kernel_pgt │ entry[510] → level2_kernel_pgt，entry[511] → level2_fixmap_pgt │
  ├───────────────────┼────────────────────────────────────────────────────────────────┤
  │ level2_kernel_pgt │ 用 2MB 大页映射内核 text+data+bss                              │
  └───────────────────┴────────────────────────────────────────────────────────────────┘

  ---
  阶段三：C 代码 x86_64_start_kernel() 填充 init_top_pgt

  文件：arch/x86/kernel/head64.c（行 222）

  1. reset_early_page_tables() — 清空 early_top_pgt 的 entry[0-510]，写回 %cr3（杀死恒等映射）
  2. clear_page(init_top_pgt) — 清零最终 PGD
  3. init_top_pgt[511] = early_top_pgt[511] — 把内核高位映射拷贝过来
  4. 调用 x86_64_start_reservations() → start_kernel()

  ---
  总结：%cr3 切换时间线

  bootloader → 解压器 startup_32 构建 pgtable 并写入 %cr3
                    ↓
             解压器 ident_map 完善 identity mapping，写 %cr3
                    ↓
             解压完成，跳转到内核 startup_64
                    ↓
             __startup_64() 修正 early_top_pgt，写 %cr3
                    ↓
             x86_64_start_kernel() 清理 early_top_pgt，构建 init_top_pgt
                    ↓
             start_kernel()

  核心结论：最初的 PGD 由解压器的汇编代码在 startup_32 中从零构建（4GB 恒等映射），内核本体的 early_top_pgt 则是 在编译时静态定义、在启动时由 __startup_64() 做 KASLR 修正后加载到 %cr3 的。
