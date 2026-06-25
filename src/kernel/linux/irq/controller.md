# 中断控制器
中断控制器是连接外部设备中断信号与 CPU 核心的硬件单元，负责信号的收集、优先级仲裁和路由分发。没有中断控制器，多个设备同时产生中断时 CPU 无法有序处理，也无法将中断定向到特定的 CPU 核心。不同 CPU 架构使用不同的中断控制器，Linux 内核通过 `irq_chip` 抽象屏蔽了这些差异。

## x86：APIC 体系
x86 平台的中断控制器体系由两部分组成：集成在 CPU 核心内部的 **Local APIC** 和位于芯片组中的 **I/O APIC**。

### Local APIC
每个 CPU 核心内部都有一个 Local APIC，它是中断信号的最终接收端。Local APIC 的职责包括接收来自 I/O APIC 的外部中断、接收来自其他 CPU 核心的 IPI（Inter-Processor Interrupt，处理器间中断）、以及管理本核心内部的定时器中断和性能监控中断。

Local APIC 内部维护一组寄存器，包括中断请求寄存器（IRR）、中断服务寄存器（ISR）、中断结束寄存器（EOI）和任务优先级寄存器（TPR）。当外部中断到达时，Local APIC 将中断向量号写入 IRR，如果该中断的优先级高于 TPR 中的当前优先级，则通过 CPU 的 INTR 引脚通知 CPU 核心执行中断处理程序。CPU 自动将中断向量号压入内核栈，内核据此查找 IDT（中断描述符表）中的处理函数入口。

中断处理完成后，内核必须向 Local APIC 的 EOI 寄存器写入以通知硬件"该中断已处理完毕"，Local APIC 才能继续接收同优先级或低优先级的中断。这个写 EOI 的操作由内核的中断退出代码自动完成。

Local APIC 还内置一个**定时器（LAPIC Timer）**，可以配置为单次触发或周期触发模式。在多核系统中，每个核心通过自己的 LAPIC Timer 独立接收定时器中断，内核的调度器 tick 和高精度定时器（hrtimer）都依赖它。

### I/O APIC
I/O APIC 位于芯片组中（Intel PCH 或传统南桥），负责接收外部设备的中断信号并通过 APIC 总线路由到目标 CPU 的 Local APIC。I/O APIC 内部有 24 个（或更多）**重定向表条目（Redirection Entry）**，每个条目对应一个中断输入引脚，可以独立配置中断向量号、目标 CPU 的 APIC ID、触发模式（边沿触发或电平触发）和极性（高电平有效或低电平有效）。

当设备的中断信号到达 I/O APIC 的某个引脚时，I/O APIC 查找对应的重定向表条目，将中断消息（包含中断向量号和目标 CPU 信息）通过 APIC 总线发送到目标 CPU 的 Local APIC。这个过程是硬件自动完成的，内核不需要在中断到达时介入路由判断。

系统中可以存在多个 I/O APIC，通过 ACPI 的 MADT（Multiple APIC Description Table）描述每个 I/O APIC 的 MMIO 基地址和中断输入范围。内核启动时解析 MADT，通过 MMIO 配置所有 I/O APIC 的重定向表。

### MSI 与 MSI-X
传统的 INTx 中断机制要求设备通过物理中断线发送信号，经过 I/O APIC 路由到 CPU。MSI（Message Signaled Interrupts）绕过了这条路径：设备直接通过向 Local APIC 的特定内存地址写入数据来触发中断，写入的数据就是中断向量号。

MSI 的优势在于：每个设备可以拥有独立的中断向量，不需要共享中断线；不需要 I/O APIC 参与，减少了中断延迟；设备可以支持多个中断向量。MSI 最多支持 32 个向量，MSI-X 扩展到最多 2048 个向量，每个向量可以独立配置目标 CPU 和触发条件。

MSI/MSI-X 在 PCIe 设备中广泛使用。网卡利用多向量能力为每个接收/发送队列绑定独立的中断，实现多核并行处理。NVMe 设备利用 MSI-X 为多个提交/完成队列分配独立中断。内核通过 `pci_alloc_irq_vectors` 统一分配 MSI/MSI-X 向量，通过 `pci_irq_vector` 获取每个向量的 Linux 中断号。

### x2APIC
传统 APIC 寄存器通过 MMIO 访问，MMIO 地址空间有限，且某些操作需要通过不可缓存的 MMIO 完成存在性能开销。**x2APIC** 扩展将 APIC 寄存器映射到 MSR（Model Specific Register），通过 `wrmsr/rdmsr` 指令访问，避免了 MMIO 的开销。更重要的是，x2APIC 将 APIC ID 从 8 位扩展到 32 位，支持超过 255 个 CPU 核心，在大型服务器系统中必不可少。

内核通过 CPUID 特性位检测 x2APIC 支持情况，在 BIOS/ACPI 启用 x2APIC 模式后自动切换到 x2APIC 驱动。x2APIC 模式下，IPI 的发送和 Local APIC 的配置都通过 MSR 完成，不再需要通过 MMIO 操作 I/O APIC。

## ARM：GIC 体系
ARM 平台使用 **GIC（Generic Interrupt Controller）** 作为中断控制器，架构版本从 GICv1 演进到 GICv4。GIC 的功能定位与 x86 的 APIC 体系类似，但实现和接口不同。

### GIC 的组件
GIC 由**分发器（Distributor）**、**CPU 接口（CPU Interface）** 和（GICv3+ 的）**重分发器（Redistributor）**组成。

分发器负责管理所有中断源的全局属性：中断使能/禁用、优先级设置、目标 CPU 的亲和性配置。所有中断信号（SPI，Shared Peripheral Interrupt）都先到达分发器，由分发器决定路由到哪个 CPU 接口。

CPU 接口是每个 CPU 核心与 GIC 之间的接口，负责优先级屏蔽、中断抢占和中断状态管理。CPU 接口向 CPU 核心发出中断信号（IRQ 或 FIQ），CPU 通过读写 CPU 接口的寄存器来应答中断（ACK）和完成中断（EOI）。

GICv3 引入了重分发器，每个 CPU 核心有一个独立的重分发器，管理该核心的私有中断（PPI，Private Peripheral Interrupt）和软件生成的中断（SGI，Software Generated Interrupt，用于 IPI）。GICv3 的 CPU 接口改为通过内存映射的系统寄存器访问（`ICC_*` 系列寄存器），不再像 GICv2 那样通过共享的 MMIO 区域，消除了多核并发访问的瓶颈。

### 中断类型
GIC 将中断分为四类：**SGI（Software Generated Interrupt）** 由软件通过写 GIC 寄存器触发，用于核间通信（IPI），中断号 0-15；**PPI（Private Peripheral Interrupt）** 是每个 CPU 核心私有的外设中断，如核内的定时器中断、看门狗中断，中断号 16-31；**SPI（Shared Peripheral Interrupt）** 是所有 CPU 核心共享的外设中断，如 PCIe 设备中断、GPIO 中断，中断号 32-1019；**LPI（Locality-specific Peripheral Interrupt）** 是 GICv3+ 引入的，专为大量中断源设计（如 MSI），中断号从 8192 开始，通过 ITS（Interrupt Translation Service）管理。

LPI 和 ITS 是 GICv3 的重要特性。ITS 维护一张设备 ID 到中断号（EventID）的映射表，当 PCIe 设备发送 MSI 时，ITS 将设备的 DeviceID + EventID 翻译为 LPI 中断号，路由到目标 CPU 的重分发器。这种机制使得 ARM 平台能够高效支持 PCIe 设备的 MSI/MSI-X 中断。

### 与 APIC 体系的主要差异
在 x86 上，Local APIC 和 I/O APIC 是两个独立的硬件单元，通过 APIC 总线通信。在 ARM 上，GIC 的分发器和 CPU 接口集成在同一个 IP 块中（尽管物理上可能分布在芯片的不同位置），通过内部总线通信。x86 的中断向量号是全局统一的（0-255），ARM 的中断类型区分了私有和共享范围。x86 通过 INTR/FIRE 中断引脚通知 CPU，ARM 通过 IRQ/FIQ 信号线通知（FIQ 具有更高优先级，常用于安全世界（Secure World）的中断）。x86 的 IPI 通过 Local APIC 的 ICR 寄存器发送，ARM 的 IPI 通过 SGI 机制发送。

## Linux 中断子系统
内核的中断子系统（`kernel/irq/`）建立了统一的抽象框架，将不同架构的中断控制器硬件差异封装在 `irq_chip` 驱动中。

### irq_domain
`irq_domain` 是连接硬件中断号和 Linux 虚拟中断号的桥梁。不同中断控制器使用不同的硬件中断号编码方式：x86 Local APIC 使用向量号（0-255），GIC 使用 SPI 中断号（32-1019），I/O APIC 使用引脚号 + GSI（Global System Interrupt）。`irq_domain` 负责将硬件中断号映射为 Linux 内部统一的虚拟中断号（Linux IRQ number），上层的 `request_irq` 和中断处理流程只使用虚拟中断号，不关心底层硬件细节。

```c
// irq_chip 描述中断控制器的操作接口
struct irq_chip {
    const char *name;
    void (*irq_ack)(struct irq_data *data);       // 确认中断
    void (*irq_mask)(struct irq_data *data);       // 屏蔽中断
    void (*irq_unmask)(struct irq_data *data);     // 解除屏蔽
    void (*irq_eoi)(struct irq_data *data);        // 中断结束
    int (*irq_set_affinity)(struct irq_data *data,
                            const struct cpumask *dest, bool force); // 设置亲和性
    ...
};
```

`irq_chip` 的方法对应了中断控制器的基本操作。以 x86 的 Local APIC 为例：`irq_ack` 在进入中断处理函数时调用，通知 Local APIC 该中断已被接收（从 IRR 移到 ISR）；`irq_eoi` 在中断处理完成时调用，向 Local APIC 写 EOI 寄存器；`irq_set_affinity` 修改 I/O APIC 重定向表或 MSI 地址，将中断路由到目标 CPU。

### 中断亲和性
中断亲和性（IRQ affinity）决定了一个中断被路由到哪个 CPU 核心处理。内核通过 `/proc/irq/N/smp_affinity` 暴露亲和性配置。对于 I/O APIC 中断，修改亲和性会更新重定向表的目标 CPU 字段；对于 MSI 中断，修改亲和性会更新 MSI 地址中的目标 APIC ID。

在多核系统中，合理配置中断亲和性可以避免所有中断集中在同一个 CPU 上造成瓶颈。常见的策略包括：将网卡中断均匀分散到所有 CPU 核心（RPS/RFS 机制），将存储设备中断绑定到执行 I/O 的核心（NUMA 亲和性），将定时器中断绑定到固定核心（减少缓存抖动）。内核也提供了 `irqbalance` 守护进程，自动根据中断负载和 CPU 拓扑调整亲和性。

对于中断控制器支持的区域亲和性（如 GICv3 的亲和性路由），内核可以根据 CPU 的 NUMA 节点自动将中断路由到距离设备最近的 CPU 核心，减少跨节点访问的延迟。x86 平台通过 I/O APIC 重定向表和 MSI 地址配置实现类似的效果，但需要驱动和子系统显式设置。
