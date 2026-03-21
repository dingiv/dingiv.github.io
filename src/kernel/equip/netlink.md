# 网络链路


局域网通信技术，ethernet，pcie，usb，蓝牙，2.4g/5g, infiniband, fiber channel 光纤，thunder blot，can 网络 control ler area network

方案名称	物理媒介	协议栈层级	理论带宽 (单向)	实测有效负载延迟 (Latency)	CPU 负载	核心优势	
PCIe NTB (P2P)	PCIe 3.0 x4 缆线	硬件电路 / 内存映射	~32 Gbps	< 1 μs (纳秒级交换)	极低 (DMA)	性能天花板，真正异构计算	
RDMA (RoCE v2)	万兆光纤 / DAC	Verbs API (绕过内核)	~10 / 25 Gbps	2 - 10 μs	极低 (零拷贝)	高并发、大数据量无损传输	
Thunderbolt 4/USB 4.0	雷电专用线	PCIe Tunneling	40 Gbps (物理)	20 - 60 μs	中 (受 IOMMU 影响)	兼顾高带宽与外设扩展性	
USB 3.1 Gadget	USB-C 数据线	RNDIS / USB 协议栈	5 / 10 Gbps	200 - 500 μs	高 (中断频繁)	成本最低，无需额外硬件	
10G Ethernet	CAT6A / SFP+	标准 TCP/IP	10 Gbps	100 - 300 μs	中高 (协议栈开销)	兼容性最强，方案最成熟	
CAN Bus (2.0B)	双绞线	CAN Frame	1 Mbps	500 μs - 2 ms	低 (硬件过滤)	极高实时性，抗干扰强	