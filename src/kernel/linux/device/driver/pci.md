# PCIe 驱动
PCIe（Peripheral Component Interconnect Express）是目前服务器和桌面平台最主要的设备互联总线，广泛用于网卡、显卡、NVMe 硬盘、加速卡等高性能设备。在 Linux 内核中，PCIe 驱动的开发围绕**总线-设备-驱动模型**展开，内核提供了完整的 PCI 子系统框架，开发者只需按照规范实现 probe/remove 等回调函数，即可完成驱动的编写。

## PCI 子系统架构

Linux 内核的 PCI 子系统分为三层。
+ PCI 配置空间访问层，负责通过 MMIO（Memory-Mapped I/O）读取设备的配置空间，获取厂商 ID、设备 ID、BAR（Base Address Register）等硬件信息。
+ 中间层是 PCI 核心层，负责设备的枚举、资源分配、电源管理、热插拔等通用逻辑。
+ 具体设备驱动层，这一层实现与特定设备通信的逻辑，也是开发者主要工作层。

内核启动时，PCI 核心层会遍历系统中的所有 PCI 总线，扫描每个总线上的设备槽位，读取配置空间来识别设备。识别到的每个 PCI 设备都会被抽象为一个 `struct pci_dev` 结构体，注册到设备模型中。随后，PCI 总线的 `match` 函数会根据设备信息在已注册的驱动中查找匹配项，匹配成功则调用驱动的 `probe` 函数。

## 设备识别与匹配

PCI 设备通过配置空间中的**厂商 ID（Vendor ID）** 和**设备 ID（Device ID）** 来标识，这两个 ID 各 16 位，组合起来可以唯一确定一种设备型号。内核中还使用**子系统厂商 ID** 和**子系统设备 ID** 做更细粒度的区分，因为同一个芯片可能被不同厂商做成不同的板卡。

驱动通过 `struct pci_device_id` 结构体定义匹配表，告知内核自己能驱动哪些设备。内核的 PCI 总线 match 函数会将 `pci_dev` 的 ID 与驱动匹配表逐条比较，命中则触发 probe。

```c
static const struct pci_device_id my_pci_ids[] = {
    { PCI_DEVICE(0x8088, 0x1234) },               // 匹配厂商 0x8088，设备 0x1234
    { PCI_DEVICE(0x8088, 0x5678), .driver_data = 1 }, // 匹配另一个型号，附带私有数据
    { 0, }  // 结束标记，必须保留
};
MODULE_DEVICE_TABLE(pci, my_pci_ids);
```

`PCI_DEVICE` 宏只指定厂商 ID 和设备 ID，屏蔽了子系统和 class 的匹配。如果需要更精确的匹配，可以使用 `PCI_VDEVICE`（自动填充厂商 ID 为 PCI_VENDOR_ID_xxx）或手动填充完整的 `pci_device_id` 结构体。`MODULE_DEVICE_TABLE` 宏是必须的，它将匹配表导出到模块信息中，使得 modprobe 和 udev 能够在用户空间完成驱动自动加载（modalias 机制）。

## 驱动注册与生命周期

PCI 驱动通过 `struct pci_driver` 结构体定义，核心是 probe 和 remove 两个回调。

```c
static int my_pci_probe(struct pci_dev *pdev, const struct pci_device_id *ent) {
    int err;
    // 1. 启用设备，唤醒处于休眠状态的设备并确认设备可访问
    err = pci_enable_device(pdev);
    if (err) return err;

    // 2. 请求内存区域，防止其他驱动访问同一 BAR 区域
    err = pci_request_region(pdev, BAR_0, "my_device");
    if (err) goto disable;

    // 3. 将 BAR 空间映射到内核虚拟地址空间
    void __iomem *mmio = pci_iomap(pdev, BAR_0, 0);
    if (!mmio) goto release;

    // 4. 设置 DMA 掩码，告知内核设备能寻址的物理地址范围
    err = dma_set_mask_and_coherent(pdev, DMA_BIT_MASK(64));
    if (err) {
        // 回退到 32 位寻址
        err = dma_set_mask_and_coherent(pdev, DMA_BIT_MASK(32));
        if (err) goto unmap;
    }

    // 5. 启用 MSI 或 MSI-X 中断
    err = pci_alloc_irq_vectors(pdev, 1, 16, PCI_IRQ_MSIX);
    if (err < 0) goto unmap;

    // 6. 注册中断处理函数
    err = request_irq(pci_irq_vector(pdev, 0), my_irq_handler,
                      IRQF_SHARED, "my_device", pdev);
    if (err) goto free_irq;

    // 7. 将私有数据挂载到 pci_dev，后续通过 pci_get_drvdata 获取
    struct my_device *dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    dev->mmio = mmio;
    pci_set_drvdata(pdev, dev);

    return 0;

free_irq:
    pci_free_irq_vectors(pdev);
unmap:
    pci_iounmap(pdev, mmio);
release:
    pci_release_region(pdev, BAR_0);
disable:
    pci_disable_device(pdev);
    return err;
}

static void my_pci_remove(struct pci_dev *pdev) {
    struct my_device *dev = pci_get_drvdata(pdev);

    free_irq(pci_irq_vector(pdev, 0), pdev);
    pci_free_irq_vectors(pdev);
    pci_iounmap(pdev, dev->mmio);
    pci_release_region(pdev, BAR_0);
    pci_disable_device(pdev);
    kfree(dev);
}

static struct pci_driver my_pci_driver = {
    .name = "my_pci_device",
    .id_table = my_pci_ids,
    .probe = my_pci_probe,
    .remove = my_pci_remove,
};

module_pci_driver(my_pci_driver);
MODULE_LICENSE("GPL");
```

`module_pci_driver` 是一个宏，展开后自动定义 `module_init` 和 `module_exit`，分别调用 `pci_register_driver` 和 `pci_unregister_driver`，省去了手动注册的样板代码。

### probe 中的关键步骤

probe 函数的执行顺序是有讲究的，必须严格遵循：先 `pci_enable_device` 使能设备，再请求资源，再映射内存，最后注册中断和 DMA。remove 中的释放顺序则完全相反。

`pci_enable_device` 完成三件事：唤醒设备（如果设备处于 D3cold 等低功耗状态）、启用设备的内存空间和 I/O 空间解码、将设备标记为活跃状态。没有这一步，后续对 BAR 的访问会失败。

BAR（Base Address Register）是 PCI 配置空间中最重要的资源字段。每个 BAR 对应设备提供的一段地址空间，可能是内存映射的寄存器（MMIO）或 I/O 端口。`pci_request_region` 向内核声明对某一段 BAR 的独占使用权，防止其他驱动或内核子系统重复占用。`pci_iomap` 将 BAR 的物理地址映射为内核虚拟地址，之后通过 `readl/writel` 等 MMIO 访问函数操作设备寄存器。

## 中断机制

PCI 设备的中断经历了从传统 INTx 到 MSI（Message Signaled Interrupts）再到 MSI-X 的演进。INTx 是电平触发的共享中断线，所有 PCI 设备共享四根中断线（INTA-D），内核需要逐个调用中断处理函数来判断中断来源，效率低下。MSI 通过向特定内存地址写入数据来触发中断，每个设备可以拥有独立的中断向量，无需共享，且支持多个中断向量。MSI-X 是 MSI 的增强版本，最多支持 2048 个中断向量，每个向量可以独立配置目标 CPU 和触发条件。

现代 PCIe 驱动应优先使用 MSI-X，其次 MSI，最后才考虑 INTx。`pci_alloc_irq_vectors` 统一了这三种中断的分配接口，通过标志位 `PCI_IRQ_MSIX`、`PCI_IRQ_MSI`、`PCI_IRQ_LEGACY` 指定期望的中断类型，函数会自动选择最佳方案。分配成功后，通过 `pci_irq_vector(pdev, index)` 获取每个中断向量的 Linux 中断号，传给 `request_irq` 注册处理函数。

对于支持多队列的设备（如网卡的多发送/接收队列），MSI-X 的多向量能力尤为关键。每个队列可以绑定独立的中断向量和 CPU 核，实现真正的并行处理，避免单个中断成为瓶颈。

## DMA 操作

PCIe 设备通常通过 DMA（Direct Memory Access）直接读写系统内存，无需 CPU 介入。驱动在使用 DMA 之前，需要通过 `dma_set_mask_and_coherent` 设置 DMA 掩码，告知内核设备能寻址的物理地址范围。64 位设备设置 64 位掩码，32 位设备设置 32 位掩码。如果 64 位掩码设置失败（可能是系统没有足够的 IOMMU 支持），需要回退到 32 位。

内核提供了两种 DMA 映射 API：一致性映射（consistent mapping）和流式映射（streaming mapping）。一致性映射通过 `dma_alloc_coherent` 分配一块物理连续且设备与 CPU 都能访问的内存，适用于设备与 CPU 频繁双向访问的共享数据结构（如命令环、状态环）。流式映射通过 `dma_map_single` 或 `dma_map_page` 将已有的内核缓冲区临时映射给设备，适用于单向的批量数据传输。流式映射完成后必须调用 `dma_unmap_single` 解除映射。

```c
// 一致性映射：分配设备与 CPU 共享的内存
struct shared_ring *ring;
dma_addr_t ring_dma;
ring = dma_alloc_coherent(&pdev->dev, sizeof(*ring), &ring_dma, GFP_KERNEL);
// ring_dma 是设备端使用的物理地址
// ring 是 CPU 端使用的虚拟地址

// 流式映射：将内核缓冲区映射给设备
void *buf = kmalloc(PAGE_SIZE, GFP_KERNEL);
dma_addr_t buf_dma = dma_map_single(&pdev->dev, buf, PAGE_SIZE, DMA_TO_DEVICE);
if (dma_mapping_error(&pdev->dev, buf_dma)) {
    kfree(buf);
    return -ENOMEM;
}
// 设备通过 buf_dma 访问数据
// 传输完成后解除映射
dma_unmap_single(&pdev->dev, buf_dma, PAGE_SIZE, DMA_TO_DEVICE);
```

在高性能场景下，建议使用分散-聚集（scatter-gather）DMA，通过 `dma_map_sg` 一次性映射多个不连续的内存页，减少映射次数和 TLB 压力。`dma_map_sg` 的参数 `struct scatterlist` 由 `sg_alloc_table` 和 `sg_set_page` 构造，或使用辅助函数 `sg_init_one` 从单个缓冲区快速构造。

## 电源管理

PCI 驱动需要支持电源管理，实现 `suspend` 和 `resume` 回调，使设备能在系统休眠时正确进入低功耗状态，在唤醒时恢复正常工作。

```c
static int my_pci_suspend(struct device *dev) {
    struct pci_dev *pdev = to_pci_dev(dev);
    // 禁用设备中断，保存设备状态
    free_irq(pci_irq_vector(pdev, 0), pdev);
    pci_save_state(pdev);
    // 将设备置于 D3 低功耗状态
    pci_set_power_state(pdev, PCI_D3hot);
    return 0;
}

static int my_pci_resume(struct device *dev) {
    struct pci_dev *pdev = to_pci_dev(dev);
    // 唤醒设备，恢复状态
    pci_set_power_state(pdev, PCI_D0);
    pci_restore_state(pdev);
    request_irq(pci_irq_vector(pdev, 0), my_irq_handler,
                IRQF_SHARED, "my_device", pdev);
    return 0;
}

static const struct dev_pm_ops my_pci_pm_ops = {
    .suspend = my_pci_suspend,
    .resume  = my_pci_resume,
};

// 在 pci_driver 中引用
static struct pci_driver my_pci_driver = {
    .driver = {
        .pm = &my_pci_pm_ops,
    },
    // ...
};
```

`pci_save_state` 和 `pci_restore_state` 配对使用，保存和恢复设备的配置空间寄存器。这是必要的，因为某些设备在进入 D3 状态后会丢失配置空间的内容。`pci_set_power_state` 控制设备的电源状态，PCI 规范定义了 D0（全功耗）到 D3（几乎关闭）的多个电源状态，驱动在 suspend 时将设备置于 D3hot 或 D3cold，resume 时恢复到 D0。

## 错误处理与恢复

PCIe 规范定义了 AER（Advanced Error Reporting）机制，用于报告和恢复总线错误。当发生不可纠正错误（如内存读超时、完成超时等）时，PCIe 设备会发送错误消息，内核 AER 驱动捕获后通知设备驱动。驱动可以通过注册 `error_handlers` 来响应这些错误事件。

```c
static const struct pci_error_handlers my_err_handlers = {
    .error_detected = my_error_detected,
    .mmio_enabled  = my_mmio_enabled,
    .slot_reset    = my_slot_reset,
    .resume        = my_error_resume,
};

static struct pci_driver my_pci_driver = {
    .error_handlers = &my_err_handlers,
    // ...
};
```

错误恢复流程分为四个阶段：`error_detected` 通知驱动发生了错误，此时驱动应停止提交新的 I/O；`mmio_enabled` 表示设备的 MMIO 空间已恢复可访问（仅在可恢复错误时调用）；`slot_reset` 表示总线进行了热重置，设备需要完全重新初始化；`resume` 表示设备已恢复正常，驱动可以恢复正常的 I/O 操作。每个阶段都有明确的职责划分，驱动需要根据错误严重程度在不同阶段采取不同的恢复策略。

## 编译与调试

将驱动源文件添加到内核的 Makefile 中（如 `obj-$(CONFIG_MY_PCI) += my_pci.o`），或者在模块化开发时编写独立的 Kbuild 文件：

```makefile
obj-m += my_pci.o

KDIR := /lib/modules/$(shell uname -r)/build
all:
    make -C $(KDIR) M=$(PWD) modules

clean:
    make -C $(KDIR) M=$(PWD) clean
```

编译完成后，通过 `insmod my_pci.ko` 加载驱动，`lsmod` 确认加载状态，`lspci -k` 查看设备与驱动的绑定关系。`dmesg` 中可以看到 probe 的日志输出。调试时常用的内核配置包括 `CONFIG_PCI_DEBUG`（开启 PCI 子系统调试信息）和 `CONFIG_PCIEASPM`（ASPM 电源管理调试）。`pci_stub` 驱动可以将设备从原驱动中"偷走"绑到 stub 上，便于在不影响原驱动的情况下进行调试。用户还可以通过 `/sys/bus/pci/devices/` 目录下的属性文件查看和修改设备的配置空间、资源映射等运行时状态。
