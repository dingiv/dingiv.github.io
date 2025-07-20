# DMA
DMA 技术是用于加速 IO 操作速度，卸载 CPU 工作负载的硬件级技术。通过该技术，访问 IO 设备时，无需 CPU 在内存和设备寄存器之间进行数据拷贝工作，而是由 DMA IO 设备的内置专用处理器负责设备数据的拷贝工作。

CPU 仅设置和结束传输，空闲时可执行其他任务，减轻了 CPU 的工作量，让 CPU 专注于更高级的逻辑处理，而不是简单的数据拷贝。

> 可编程 IO，是相较于 DMA 技术的 IO 读写，即相对较老的技术，操作系统通过读写 IO 设备寄存器控制设备，目前市面上的设备已经广泛支持 DMA 技术，以提供 IO 访问速度；
> - PMIO（Port-Mapped Input/Output）：通过访问 IO 端口控制设备
> - MMIO（Memory-Mapped Input/Output）：设备寄存器和缓冲区映射到物理内存中
  
## 基本流程
DMA 功能往往只需要驱动层关注即可，由驱动负责发起 DMA 硬件操作。
+ CPU 通过驱动发起 IO 操作，配置 DMA 相关的控制器，从而指定传输参数（如源地址、目标地址、数据长度），立即返回；
+ 设备直接通过总线并经由 IOMMU 的地址翻译访问内存，传输数据；
+ 传输完成后，DMA 控制器通过中断通知 CPU；
+ CPU 收到通知，接管后续逻辑，到指定好的目标地址上获取拷贝完成的数据结果；

## IOMMU
CPU 是不能直接访问内存的，必须通过 MMU。IO 设备更不能直接访问内存，必须通过 IOMMU（IO 设备内存空间管理单元）。相较于 MMU 是 CPU 的一个硬件单元，IOMMU 可以视为一个独立的设备，需要安装驱动。

### 核心功能
+ 地址转换：将设备使用的虚拟地址（IOVA，I/O Virtual Address）映射到物理内存地址。允许设备通过 DMA 访问内存，无需了解物理内存布局。
+ 设备隔离：为每个设备分配独立的地址空间，防止未经授权的内存访问。增强虚拟化场景的安全性（如虚拟机隔离）。
+ 中断重映射：管理设备中断，确保中断信号正确路由到目标 CPU 或虚拟机。
+ 性能优化：减少 DMA 传输的内存拷贝开销。支持大页面映射，降低地址转换开头的 TLB（Translation Lookaside Buffer）开销。

### 关键数据结构
1. struct iommu_domain：
   - 表示设备的地址空间，类似虚拟机的内存上下文。
   - 字段：
     - `type`：域类型（如 `IOMMU_DOMAIN_DMA` 用于标准 DMA 映射）。
     - `ops`：`struct iommu_ops`，定义 IOMMU 硬件操作（如地址映射）。
   - 作用：为设备分配独立的虚拟地址空间。
2. struct iommu_group：
   - 表示一组共享 IOMMU 的设备（如 PCIe 设备的同一组）。
   - 字段：
     - `devices`：关联的设备列表。
     - `domain`：指向当前使用的 `iommu_domain`。
   - 作用：确保同一组设备共享相同的地址空间和隔离策略。
3. struct iommu_ops：
   - 定义 IOMMU 硬件驱动的操作，如 `map`（映射 IOVA 到物理地址）、`unmap`（解除映射）。
   - 由具体 IOMMU 硬件驱动实现（如 Intel VT-d、AMD-Vi）。

### 工作流程（以 DMA 读操作为例）
1. 设备驱动初始化：
   - 驱动（如同 NVMe 驱动 `drivers/nvme/host/core.c`）调用 `iommu_map` 或 `dma_map_single` 分配 IOVA。
   - 源码示例（`include/linux/dma-mapping.h`）：
     ```c
     dma_addr_t dma_map_single(struct device *dev, void *ptr, size_t size, enum dma_data_direction dir);
     ```
   - 内核通过 `struct device->dma_ops` 调用 IOMMU 驱动（如 `intel_iommu_map`）。
2. IOMMU 配置：
   - IOMMU 硬件驱动更新页表，将 IOVA 映射到物理地址。
   - 页表存储在 IOMMU 硬件中，类似 CPU 的 MMU 页表。
3. DMA 传输：
   - 设备（如磁盘）使用 IOVA 发起 DMA 传输，IOMMU 自动转换为物理地址。
   - 数据直接写入页面缓存（`bio->bi_io_vec` 的页面）。
4. 中断与完成：
   - 传输完成后，设备触发中断，IOMMU 重映射中断信号到正确 CPU。
   - 驱动调用 `bio_endio`，通知文件系统（如 ext4）。
5. 解除映射：
   - 驱动调用 `dma_unmap_single`，释放 IOVA，更新 IOMMU 页表。
