# 机房

1. 机房基础设施

这是最底层，决定机器能否稳定运行。主要包括：

供电系统：市电 + UPS（不间断电源）+ 发电机。核心思想是“永不断电”。

制冷系统：恒温恒湿（通常温度 22°C 左右，湿度 45%～55%）。

机柜与布线：标准 19 英寸机柜，常见 42U 高。前后进风、线缆分层（上走网线，下走电缆）。

防火与安防：气体灭火系统（FM200 或 IG541），门禁、摄像头、环境监控。

2. 网络与接入层

这里决定你的服务器能否通信。

核心交换机 / 汇聚交换机 / 接入交换机 分层结构。

双上联 / 双电源冗余，保证某一路坏掉也能通。

外网接入：通过 ISP（电信、联通、移动）光纤接入，或走 BGP 做多线冗余。

内部网络划分：通过 VLAN、子网、ACL（访问控制）区分管理网、业务网、存储网。

3. 服务器部署

实际放置和上电阶段：

规划机柜布局（每柜功率负载均衡）。

设备上架、接线、打标。

开机配置：BIOS、RAID、固件更新。

装系统（通常 PXE 网络安装，自动化 kickstart / preseed / cloud-init）。

基础监控和资产登记。

4. 软件与服务层

系统上电之后进入软件世界：

自动化配置管理：Ansible、SaltStack、Puppet。

监控系统：Prometheus + Grafana、Zabbix。

日志系统：ELK 或 Loki + Grafana。

安全与访问控制：SSH key 管理、堡垒机、NTP 同步、审计系统。

上层应用部署：容器、K8s、或者虚拟机（KVM、VMware）。

## 机架式
机架式服务器机箱的内部结构非常有逻辑感，几乎所有品牌（Dell PowerEdge、HPE ProLiant、华为、浪潮、Supermicro）都遵循相同的布局思路。你可以把它想成一台高度压缩、可持续运行 7×24 的“工业级电脑”。

下面是一个典型 2U 机架式服务器 内部的主要组件：

1. 主板（Motherboard）

核心骨架，承担所有连接。

包含 CPU 插槽（1–2 个，有的高端型号 4 个）。

多个 内存插槽（DIMM Slots），常见 8 到 32 个。

集成 PCIe 通道、SATA/SAS 控制器、BMC（Baseboard Management Controller，负责远程管理）。

2. CPU（中央处理器）

通常是 Intel Xeon 或 AMD EPYC 系列，多核高线程，支持 ECC 内存。

高端型号支持 NUMA 架构（Non-Uniform Memory Access），适合多 CPU 协作。

3. 内存（Memory / RAM）

使用 ECC（Error-Correcting Code）内存，能自动检测并修正错误。

布局在 CPU 两侧，保持气流对称以帮助散热。

4. 硬盘托架与存储控制器

前面板通常有 8–24 个热插拔托架（Hot-swap Bays）。
支持：

SATA/SAS 硬盘（传统机械盘）

NVMe SSD（通过 U.2/U.3 接口或 PCIe 通道）

RAID 控制器（RAID Card）：管理磁盘阵列（RAID 0/1/5/10 等），可缓存写操作以提升性能。

5. 电源模块（Power Supply Unit, PSU）

通常是 双电源冗余（1+1），即两块独立 PSU，可在热插拔时互为备份。

支持 80PLUS Platinum / Titanium 认证，转换效率高。

6. 风扇阵列（Cooling Fans）

多个高转速风扇位于前后中段，用于气流通道。

通常由 BMC 动态控制转速，响应温度变化。

7. 扩展插槽（PCIe Slots）

用于安装：

网络接口卡（10G/25G/100G NIC）

GPU 加速卡（如 A100、H100）

存储扩展卡（NVMe 扩展、RAID）

光纤通道 HBA（连接 SAN 存储）

8. 远程管理模块（BMC / iDRAC / iLO / iMana）

独立的小型控制系统，接入独立管理网口。

提供 Web 界面、IPMI、虚拟控制台、温度监控、远程开关机。

9. 机箱结构件

滑轨（Rail Kit）：方便抽拉维护。

EMI 屏蔽层：防电磁干扰。

导风罩（Air Shroud）：引导气流从前向后经过 CPU 与内存。