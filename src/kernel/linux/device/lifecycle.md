# 生命周期
设备从上电到被应用层使用，需要经历一个完整生命周期：硬件上电、内核发现设备、总线枚举、驱动匹配与绑定、设备就绪，以及运行时的热插拔和移除。理解这个生命周期，是掌握 Linux 设备管理全貌的关键。

## 内核启动与设备模型初始化

内核的入口函数 `start_kernel` 在完成进程管理、内存管理、中断系统等核心子系统的初始化后，开始建立设备管理的基础设施。这一阶段的工作围绕设备模型（Driver Model）展开，在设备真正被探测之前，内核需要先准备好"管理框架"。

设备模型的核心是 `struct bus_type`、`struct device` 和 `struct device_driver` 三个结构体，它们构成了**总线-设备-驱动**的三角关系。在 `start_kernel` 的调用链中，`driver_init` 函数最先执行，它注册了系统中所有的虚拟总线，其中最重要的是 **platform_bus_type**（平台总线）。platform_bus 本身不对应任何物理总线，它是内核为平台设备（SoC 集成外设、通过设备树或 ACPI 描述的设备）创建的统一管理总线。

紧接着，`do_basic_setup` 函数被调用，它触发了几个关键的初始化流程：PCI 总线子系统通过 `pci_driver_init` 注册 `pci_bus_type`，USB 核心通过 `usb_init` 注册 `usb_bus_type`，I2C、SPI 等其他总线也在此阶段完成注册。这些总线的注册为后续的设备探测和驱动匹配做好了准备。此时，系统中已经有了总线类型，但还没有具体的设备和驱动，它们会在后续阶段逐步出现。

## 总线设备：从控制器到设备

总线设备的管理采用分层结构：内核先初始化总线控制器，再由总线控制器负责探测和管理其下挂载的设备。以 PCI 为例，PCI 总线本身不是凭空存在的，它需要一个 PCI 主桥芯片来连接 CPU 和 PCI 设备，这个主桥芯片就是总线控制器。

### 总线控制器的初始化

总线控制器在内核启动阶段由体系结构相关的代码或 ACPI/设备树描述来发现和初始化。在 x86 平台上，ACPI 表中描述了 PCI 主桥的存在和资源配置（内存映射范围、中断路由等）。内核在解析 ACPI 时，会为每个 PCI 主桥创建一个 `struct pci_dev` 并注册到 `pci_bus_type`。在 ARM 平台上，设备树中的节点描述了 PCI 控制器的寄存器基地址和配置空间范围，内核根据设备树创建对应的平台设备，由 PCI 控制器驱动（如 `pcie-rockchip`、`pcie-qcom`）probe 后完成控制器初始化。

PCI 控制器驱动的 probe 函数会完成控制器硬件的初始化：配置控制器的寄存器、设置内存映射窗口、使能链路训练（Link Training，建立 CPU 与 PCI 设备之间的物理连接）。控制器就绪后，PCI 核心层开始扫描（enumerate）这条总线上的设备。

### 设备枚举

PCI 枚举的过程是：从总线号 0 开始，对每个设备号（0-31）读取配置空间的厂商 ID 和设备 ID。如果读取到有效值（非 0xFFFFFFFF，后者表示设备不存在），则说明该槽位有设备。内核继续读取设备的 BAR（获取资源需求）、中断引脚（获取中断路由信息）、能力链（PCIe Capability，如 MSI-X 支持、电源管理等）。如果发现设备是多功能设备（Header Type 的多函数位为 1），还会继续扫描该设备下的每个功能号。

枚举过程中有一个递归逻辑：如果某个 PCI 设备本身是一个 PCI 桥（PCI-to-PCI Bridge），内核会为其创建一条次级总线，并递归扫描次级总线上的设备。这个过程一直持续到所有总线上的所有设备都被发现为止。最终，内核在内存中构建出一棵完整的 PCI 设备拓扑树，每个节点都是一个 `struct pci_dev`，通过 `bus->self` 指向其父桥设备。

USB 总线的枚举过程类似但有其特殊性。USB 主机控制器（xHCI）初始化完成后，USB 核心层会通过根集线器（Root Hub）探测端口。发现设备连接后，内核通过默认地址 0 与设备通信，分配设备地址，读取设备描述符（VID/PID）、配置描述符（接口和端点信息），然后根据设备类或 VID/PID 在已注册的 USB 驱动中查找匹配项。USB 支持多级集线器（Hub）级联，枚举时会递归扫描每个集线器下挂载的设备。

### 驱动匹配与绑定

总线设备枚举完成后，每个 `struct device` 都已注册到对应总线上。此时总线的 `match` 函数被调用，将设备信息与所有已注册驱动的 `id_table` 进行比对。以 PCI 为例，match 函数比较设备的厂商 ID、设备 ID、子系统 ID 与 `pci_device_id` 数组中的每一项。

匹配成功后，内核调用驱动的 `probe` 函数。probe 完成设备的硬件初始化：使能设备、映射寄存器空间、注册中断、初始化 DMA 等。probe 返回 0 后，`device->driver` 被设置为当前驱动，绑定关系正式建立。此时设备处于"已绑定、就绪"状态，可以通过 `/dev` 节点或网络接口被应用层使用。

需要强调的是，总线上设备和驱动的注册顺序是不确定的。可能出现的情况是：设备先注册，驱动后注册；或者驱动先注册，设备后注册。内核的驱动核心层（Driver Core）处理了这两种情况：当新设备注册时，会在总线的驱动列表中查找匹配项；当新驱动注册时，会在总线的设备列表中查找匹配项。无论谁先到，最终都会触发 probe。如果驱动以内核模块（.ko）的形式在启动后手动加载，那么 probe 发生在 `insmod` 或 `modprobe` 的时刻。

## 平台设备：设备树与 platform 总线

平台设备（Platform Device）与总线设备不同，它们不支持动态枚举。内核无法通过扫描总线来发现这些设备，因为它们要么直接集成在 SoC 芯片上（如 UART 控制器、I2C 控制器、GPIO 控制器），要么通过非标准接口连接，没有总线协议规定的探测机制。

### 设备树解析

在 ARM 和嵌入式平台上，平台设备通过设备树（Device Tree）来描述。设备树是一种层次化的数据结构，以 dts（Device Tree Source）文件形式存在，编译为 dtb（Device Tree Blob）二进制格式。Boot Loader 在加载内核时将 dtb 传递给内核，内核在启动早期解析设备树。

设备树中每个节点描述一个设备或总线，节点的 `compatible` 属性是驱动匹配的关键。例如：

```
/ {
    soc {
        uart0: serial@fe201000 {
            compatible = "vendor,uart-v2", "ns16550a";
            reg = <0xfe201000 0x200>;
            interrupts = <0 37 4>;
            clocks = <&clk_uart0>;
            status = "okay";
        };
    };
};
```

内核在解析设备树时，会为每个 `compatible` 属性存在且 `status` 不为 `disabled` 的节点创建一个 `struct platform_device`，并注册到 `platform_bus_type`。设备节点中的 `reg`、`interrupts`、`clocks` 等属性被解析后存储在 `platform_device` 的资源列表（`struct resource`）中，供驱动 probe 时使用。

### ACPI 描述

在 x86 平台上，平台设备通过 ACPI（Advanced Configuration and Power Interface）表来描述。ACPI 使用 DSDT（Differentiated System Description Table）和 SSDT（Secondary System Description Table）中的设备对象来描述硬件。每个设备对象有 `_HID`（Hardware ID）和 `_CID`（Compatible ID）属性，功能等同于设备树中的 `compatible`。

内核的 ACPI 子系统在启动时解析这些表，为匹配的设备创建 `struct platform_device` 并注册到 platform 总线。因此，无论底层是设备树还是 ACPI，对驱动开发者而言，面对的都是同样的 `platform_driver` 和 `platform_device` 抽象，差异被内核屏蔽了。

### 驱动匹配

平台设备的驱动匹配基于 `compatible` 字符串。`platform_driver.id_table` 中的 `of_device_id` 数组定义了驱动支持的设备列表：

```c
static const struct of_device_id my_of_match[] = {
    { .compatible = "vendor,uart-v2" },
    { .compatible = "ns16550a" },
    {}
};
MODULE_DEVICE_TABLE(of, my_of_match);

static struct platform_driver my_drv = {
    .driver = {
        .name = "my-uart",
        .of_match_table = my_of_match,
    },
    .probe = my_probe,
    .remove = my_remove,
};
```

platform 总线的 match 函数会将设备的 `compatible` 与驱动的 `of_match_table` 逐条比较，命中则触发 probe。probe 函数通过 `platform_get_resource` 获取寄存器地址，通过 `platform_get_irq` 获取中断号，通过 `of_iomap` 将物理地址映射为内核虚拟地址，完成硬件初始化。

### 平台设备的加载时序

平台设备的注册发生在内核启动的非常早期。`start_kernel` → `rest_init` → `kernel_init` → `kernel_init_freeable` 调用链中，`do_basic_setup` 会调用 `driver_init` 和各子系统的 initcall。设备树的解析和 platform_device 的创建发生在 `of_platform_default_populate_init`（设备树平台）或 `acpi_init`（ACPI 平台），它们都是 `arch_initcall` 级别，比大多数模块驱动的 `module_init` 更早执行。

这意味着在内核模块加载阶段（`do_initcalls` 的后期），平台设备已经全部注册到 platform 总线上了。当平台驱动注册时，它会立即在已有的平台设备列表中查找匹配项并触发 probe。因此平台设备的生命周期可以概括为：设备树/ACPI 描述 → 内核解析创建 platform_device → 平台驱动注册 → match → probe → 就绪。

## 热插拔设备生命周期

支持热插拔的总线（如 USB、PCIe 热插拔、Thunderbolt）允许设备在系统运行期间动态插入和移除。热插拔设备的生命周期与启动时的静态枚举有所不同，它需要内核在运行时响应硬件事件。

### 设备插入

当用户将一个 USB 设备插入端口时，硬件层面的信号变化通过以下路径传递到内核：USB 集线器检测到端口电平变化 → 集线器向主机控制器（xHCI）发送状态变化通知 → 主机控制器触发硬件中断 → 内核中断处理程序调度 USB 核心层的轮询线程（kthread） → USB 核心发现新设备并开始枚举。

USB 枚举过程与启动时一致：分配设备地址、读取描述符、根据 VID/PID 或设备类查找匹配驱动、调用 probe。整个枚举过程是异步的，不会阻塞中断处理程序。枚举完成后，内核通过 netlink 向用户空间的 udev 发送 uevent 事件，udev 根据规则创建 `/dev` 节点、加载固件、设置权限等。

PCIe 热插拔的流程类似，但检测机制不同。PCIe 热插拔槽通过 PCIe 插槽的 `Presence Detect` 和 `Power Fault` 引脚检测设备插入。主板上的热插拔控制器（如 ACPI 的 PCI 热插拔事件、Thunderbolt 控制器）向内核发送通知，内核 PCI 热插拔子系统（`pciehp` 驱动）处理该事件，对热插拔槽执行链路重新训练、配置空间扫描、资源分配，然后为新发现的 PCI 设备走正常的 match → probe 流程。

### 设备移除

设备移除是设备插入的逆过程。当 USB 设备被拔出时，集线器检测到端口断开，通知主机控制器，内核 USB 核心收到事件后执行以下步骤：首先调用驱动的 `disconnect` 函数，驱动在此释放所有资源（取消 urb、释放内存、注销中断等）；然后将设备从 USB 核心的设备列表中移除；最后通过 netlink 通知 udev 清理 `/dev` 节点。

PCIe 设备热移除的流程类似：`pciehp` 驱动检测到设备拔出，通知 PCI 核心层，PCI 核心层调用驱动的 `remove` 函数，释放资源，注销设备。对于正在使用中的设备（如正在执行 I/O 的 NVMe 硬盘），内核需要先等待进行中的 I/O 完成（或超时取消），然后才执行移除。

### 用户空间的通知链

热插拔事件从内核传递到用户空间依赖 **uevent 机制**。内核在设备添加或移除时，通过 kobject_uevent 发送事件到 netlink 套接字（`NETLINK_KOBJECT_UEVENT`）。用户空间的 udevd 守护进程监听该套接字，收到事件后根据 `/etc/udev/rules.d/` 中的规则执行动作。常见的动作包括：创建设备节点（`mknod`）、设置权限（`chmod/chown`）、创建符号链接、加载内核模块（`modprobe`）、执行自定义脚本等。

uevent 事件携带设备的属性信息（如 `DEVTYPE`、`DRIVER`、`MODALIAS` 等），udev 规则可以基于这些属性进行匹配。`MODALIAS` 属性特别重要，它是驱动自动加载的关键：当 uevent 中包含 `MODALIAS=pci:v00008088d00001234sv0000...` 时，udev 会调用 modprobe 加载对应的驱动模块，实现驱动的按需加载。

## 生命周期全景

从内核启动到设备可用，整个生命周期可以归纳为以下阶段：

1. **基础设施准备**：`start_kernel` 中 `driver_init` 注册虚拟总线（platform_bus），`do_basic_setup` 中各总线子系统注册 `bus_type`（pci_bus_type、usb_bus_type 等）。
2. **总线控制器初始化**：通过 ACPI/设备树发现总线控制器硬件，总线控制器驱动的 probe 函数完成控制器硬件初始化。
3. **设备枚举**：总线控制器就绪后，内核扫描总线发现设备，为每个设备创建 `struct device`（PCI 为 `pci_dev`，USB 为 `usb_device` + `usb_interface`）。
4. **平台设备注册**：设备树/ACPI 解析创建 `platform_device`，注册到 platform_bus_type。
5. **驱动匹配与绑定**：总线 match 函数将设备与驱动配对，调用 probe 完成硬件初始化。
6. **用户空间就绪**：udev 收到 uevent，创建 `/dev` 节点，设备可被应用层使用。
7. **运行时热插拔**：设备插入时重复枚举→匹配→probe 流程；设备移除时执行 disconnect/remove 清理资源。

需要注意的是，上述流程是理想化的简化。实际中存在很多边界情况：驱动以内核模块形式延迟加载、设备在 probe 期间失败需要回退、多驱动竞争同一设备、设备树中的设备被 `status = "disabled"` 禁用后通过运行时修改 overlay 重新启用等。但核心的驱动绑定机制（match → probe/remove）贯穿所有场景，是理解设备生命周期的关键线索。
