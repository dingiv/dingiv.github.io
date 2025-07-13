# 可编程 IO/DMA

根据数据传输过程是否需要 CPU 参与，IO 分为两类：

1. 可编程 IO

   - 操作系统通过读写 IO 设备寄存器控制设备
   - 分为两种类型：
     - PMIO（Port-Mapped Input/Output）：通过访问 IO 端口控制设备
     - MMIO（Memory-Mapped Input/Output）：设备寄存器和缓冲区映射到物理内存中

2. DMA（Direct Memory Access）
   - 外设与内存之间交换数据的接口技术
   - 数据传输过程无须 CPU 控制
   - 数据拷贝和搬运由外设专用处理器完成
   - 操作系统通过驱动程序提前告知外设数据拷贝位置
   - 外设直接访问内存，将数据放到指定位置
   - 完成后发起中断通知 CPU
  

### IOMMU

IOMMU（IO 设备内存空间管理单元）：

- 在一些硬件平台上支持 IOMMU 技术
- 添加 IOMMU 单元，在 CPU 访问物理内存地址时添加类似 MMU 的内存虚拟技术
- 针对 IO 设备
- 通常伴随 DMA 一同出现
