---
title: IOMMU
order: 15
---

# IOMMU
DMA（Direct Memory Access）和 IOMMU（I/O Memory Management Unit）分别回答了设备 IO 的两个根本问题：DMA 让设备不经过 CPU 直接读写内存，IOMMU 让这种直接访问受到地址翻译和权限控制。理解这层关系是理解现代 IO 体系（尤其是虚拟化和设备直通）的基础。

## DMA：设备直接访问内存
CPU 参与每一次数据传输是低效的——把磁盘的一个扇区读入内存，如果由 CPU 执行，需要 CPU 从磁盘控制器逐字节取数、逐字节写内存，全程占用核心。DMA 的解决思路是把搬运工作交给 DMA 引擎：CPU 只做一次"把地址 X 处的 N 字节读入内存地址 Y"的指令（设置 DMA 描述符），之后的搬运由硬件完成，完成后通过中断通知 CPU。

DMA 引入了一个根本性的地址问题：**设备看到的地址和 CPU 看到的地址不是一回事**。CPU 通过 MMU 使用虚拟地址，通过页表转换到物理地址。而设备（至少在无 IOMMU 的系统上）直接使用物理地址——设备驱动必须把缓冲区转换成设备可用的地址。

更隐蔽的问题在"物理地址"本身：x86 的 32 位系统中，物理地址空间可能超过 32 位（PAE 模式 36 位），而老式设备（如早期的 PCI 设备）的 DMA 引擎只有 32 位寻址能力——设备根本访问不到 4GB 以上的内存。这就是 **DMA 掩码（DMA Mask）**的由来：设备声明自己可寻址的范围（如 `dma_set_mask(dev, DMA_BIT_MASK(32))`），超出范围的缓冲区需要内核在低地址区域分配（GFP_DMA 区域），或者使用 bounce buffer（反弹缓冲）——数据先拷到设备可访问的低地址区，再由 CPU 拷回目标位置。

**DMA 一致性（Cache Coherency）**是另一个经典陷阱。现代 CPU 有高速缓存——写入内存的数据可能还在缓存中未刷回物理内存。如果设备此时做 DMA 读取，读到的是旧数据。x86 架构的缓存一致性协议（MESI）硬件层面保证了 DMA 和缓存的自动一致；但 ARM 等架构不保证——驱动必须显式调用缓存刷新/失效操作（`dma_sync_single_for_device`/`for_cpu`）。跨架构开发驱动时，这类"x86 上能跑，ARM 上数据错乱"的问题是最难排查的一类。

## IOMMU：设备侧的 MMU
IOMMU 是为设备提供地址翻译的硬件单元——设备发出的 DMA 地址请求经过 IOMMU 的页表翻译后才访问真实物理内存。它是"设备的 MMU"，与 CPU 的 MMU 对称。

IOMMU 的存在使 DMA 的内存访问从"裸奔"变为"受控"。驱动通过 DMA API 分配的缓冲区被登记到 IOMMU 页表中，设备只能访问登记过的地址范围——一个被入侵的设备（或被恶意驱动的设备）向任意物理地址发起 DMA 读取内存的攻击（DMA attack）被硬件阻断。这是 IOMMU 的安全价值。

IOMMU 的第二个价值是**消除 DMA 掩码限制**：32 位寻址的老设备在 IOMMU 后面可以访问任意物理内存——IOMMU 把设备的 32 位地址翻译到高地址的物理内存。bounce buffer 不再需要，性能提升。

IOMMU 的第三个价值是**设备隔离**。IOMMU 将设备分组为 iommu_group——同一 group 内的设备共享同一套 IOMMU 页表上下文，不同 group 之间硬件层面互不可见。这个分组不是软件概念，而是由 PCIe 拓扑决定的硬件事实：通过同一个 PCIe Switch 连接的设备（对 IOMMU 而言无法区分彼此的 DMA 请求来源）必须放在同一 group；直连 Root Complex 不同端口的设备可以分到不同 group。

## iommu_group 与 VFIO 直通
iommu_group 的工程意义在设备直通（PCI Passthrough）中体现——它是直通的粒度单位。VFIO 框架把设备直通给虚拟机时，直通的单位是**整个 iommu_group 而非单个设备**：同 group 的所有设备必须一起直通（或一起留在宿主机），因为 IOMMU 无法在 group 内做隔离——把 group 内一个设备给虚拟机、另一个留给宿主机，两个世界就通过 PCIe Switch 共享了 DMA 通路，隔离形同虚设。

```bash
# 查看设备的 iommu_group
ls /sys/kernel/iommu_groups/
# 查看某个设备的 group 归属（group 编号下的设备列表）
ls /sys/kernel/iommu_groups/12/devices/

# 直通前必须确认 group 内只有目标设备
# 若 group 内有多个设备，需要 ACS 支持的 PCIe Switch 才能拆分
```

设备直通时的另一个 IOMMU 细节：虚拟机内的设备 DMA 地址是**客户物理地址（GPA）**，而 IOMMU 页表登记的是宿主机物理地址（HPA）——需要两层翻译（GPA → HPA 通过 vIOMMU，HPA → 实际地址通过 IOMMU）。启用嵌套页表（nested page table）时两层翻译合并为一次硬件查表，直通性能接近裸机。

## SMMU：ARM 平台的 IOMMU
IOMMU 是 x86 的命名——Intel 叫 VT-d，AMD 叫 AMD-Vi。ARM 平台的对应物是 **SMMU（System Memory Management Unit）**，架构角色完全对等：设备 DMA 地址经过 SMMU 页表翻译和权限检查后才访问真实内存。

SMMU 与 x86 IOMMU 的关键差异在架构代际演进。SMMUv1/v2（内核驱动 arm-smmu）采用分布式翻译模型——每个设备（或每类设备）有自己独立的 SMMU 实例，页表配置分散。SMMUv3（内核驱动 arm-smmu-v3，2017 年随 SBSA 规范推广）改为集中式——一个 SMMU 管理所有设备，翻译使用内存中的页表结构而非硬件维护的 TLB 为主要形态，向 x86 IOMMU 的模型靠拢。SMMUv3 还引入了两级 StreamID/SubstreamID 寻址（对应 PCIe 的 Requester ID），这是 PCIe 直通场景中设备标识的硬件基础。

内核层面，SMMU 通过通用 IOMMU 框架接入——`struct iommu_ops` 抽象了 VT-d、AMD-Vi、SMMUv2/v3 的差异，DMA API 和 VFIO 对"底下是哪种 IOMMU"完全无感知。驱动代码的可移植性由这个抽象保证。ARM 服务器（如 Ampere、鲲鹏）上做 GPU 直通和 VFIO 透传时，系统里跑的就是 arm-smmu-v3——排查工具同样是 dmesg 中的 fault 记录（SMMU 的格式是 `arm-smmu-v3 ... event ...`），排查思路与 x86 完全一致，只是日志前缀不同。

虚拟化层面，QEMU 提供 vSMMU 模拟——向虚拟机暴露一个虚拟 SMMU，支持嵌套翻译（GPA → HPA 两层页表合并查表），配合 VFIO 实现 ARM 平台的设备直通。ARM 生态中 IOMMU 的普及率曾是短板（早期 ARM 服务器芯片不带 SMMU，直通和防 DMA attack 都无从谈起），SBSA 规范将 SMMUv3 列为服务器芯片的强制要求后，ARM 平台的 IO 虚拟化能力才与 x86 对齐。

## 零拷贝与 IOMMU 的关系
零拷贝和 IOMMU 在 IO 数据路径上位于不同环节，但存在几处真实的相互作用。

**功能上是正交的**。零拷贝（sendfile/splice）回答"数据搬运由谁执行"——目标是让 CPU 退出数据搬运路径，DMA 引擎直接在设备间/内存与设备间搬运。IOMMU 回答"设备如何寻址内存"——DMA 请求的地址如何翻译、权限如何检查。前者优化搬运的执行者，后者控制搬运的寻址方式。

**但零拷贝的每一步 DMA 都要经过 IOMMU**。sendfile 的 scatter-gather 路径中，网卡 DMA 描述符里填的页面地址——IOMMU 开启时这些地址是 IOVA（I/O 虚拟地址），每次 DMA 传输都要经过 IOMMU 页表查表。零拷贝省掉了 CPU 的数据拷贝，但 IOMMU 的地址翻译开销被加在了 DMA 路径上。这也是"IOMMU 开启后 IO 性能下降 5-10%"这个经验数据的来源——翻译开销对大数据块传输（零拷贝的主要场景）相对比例小，对小尺寸高频 IO 相对比例大。

**IOMMU 是零拷贝的安全前提**。零拷贝把页缓存页面直接交给设备 DMA 读取——没有 IOMMU 时，设备拿到的是物理地址，一个有缺陷或被入侵的设备驱动可以让设备 DMA 读取任意物理内存（包括内核代码段、其他进程的数据）。零拷贝让"设备直接读内核内存"变得普遍，这放大了无 IOMMU 环境的风险面。开启 IOMMU 后，DMA 只能访问 `dma_map` 登记过的缓冲区——零拷贝的页面引用传递依然工作（DMA API 在背后为这些页面建立 IOMMU 映射），但设备越界读取被硬件拦截。

**IOMMU 扩展了零拷贝的适用范围**。前面提到的 32 位 DMA 掩码设备——没有 IOMMU 时只能用 bounce buffer，数据"先拷到低地址区、设备读、再拷回"——零拷贝对这类设备完全失效（bounce 本身就是一次拷贝）。IOMMU 将设备地址翻译到任意物理内存后，32 位设备也能直接 DMA 页缓存页面，零拷贝路径恢复。虚拟化场景同理：直通设备的 GPA→HPA 翻译由 IOMMU（嵌套页表）完成，虚拟机内的零拷贝路径不必因虚拟化而打断。

一句话总结两者关系：零拷贝决定**数据动不动的路径**，IOMMU 决定**设备能不能碰这些数据的地址**——零拷贝的收益在 IOMMU 开启时略有折扣（翻译开销），但安全性和适用范围（老设备、虚拟机）都依赖 IOMMU 的存在。

## 内核视角的 DMA API
驱动开发不直接操作 IOMMU 硬件——通过 DMA API 抽象。`dma_alloc_coherent` 分配设备可访问的一致性内存（IOMMU 开启时自动建立映射，关闭时退化为 GFP_DMA + 无 IOMMU 直通），`dma_map_single` 映射一个缓冲区并返回设备可用地址（IOMMU 开启时返回 IO 虚拟地址，关闭时返回物理地址）。驱动代码对"系统有没有 IOMMU、是 VT-d 还是 SMMU"保持透明——这是 DMA API 的设计目标，也是跨平台驱动代码正确性的关键抽象。

理解 DMA/IOMMU 体系对系统工程的几个直接收益：排查设备 DMA 错误的思路（先看 dmesg 中的 DMAR/IOMMU fault 或 arm-smmu-v3 event 记录——它直接给出设备试图访问的非法地址）；虚拟化场景中 GPU 直通必须确认 iommu_group 隔离性；安全敏感场景开启 IOMMU 是防 DMA attack 的标准手段（x86 的 `intel_iommu=on iommu=pt`，ARM 上检查内核是否加载 arm-smmu-v3 驱动）。
