---
title: BMC 详解
order: 56
---

# BMC 主板管理器
BMC（Baseboard Management Controller）是服务器主板上一个独立的小型计算机——有自己的 ARM 处理器、内存、存储、网卡和固件，通过 I2C、SMBus、PCIe 与主板各组件相连。只要电源线插着，BMC 就保持在线——主 CPU 关机、操作系统崩溃、甚至主板故障都不影响它工作。这个"独立于主系统的第二台计算机"是服务器与消费级主板最本质的硬件差异。

## 硬件架构
主流的 BMC 芯片是 ASPEED 的 AST2500（2016）和 AST2600（2019）系列。AST2500 集成 800MHz ARM11 处理器、DDR3/DDR4 内存控制器、2D 图形引擎（用于 KVM 视频输出）；AST2600 升级为 1.2GHz 双核 Cortex-A7 和 4 核图形引擎。除了 ASPEED，服务器大厂有自研 BMC——华为的 iBMC、HPE 的 iLO 芯片、Dell 的 iDRAC 芯片（基于自有或定制硅片）。

BMC 与主系统的连接是理解其能力边界的关键：

- **I2C/SMBus**：连接温度传感器、风扇控制器、电压监控芯片——BMC 通过轮询这些总线采集主板健康数据
- **PCIe 低速链路**：连接视频捕获芯片（VGA 信号捕获用于远程 KVM）和 USB 控制器（虚拟键鼠注入）
- **NC-SI（Network Controller Sideband Interface）**：与主板网卡共享物理网口——BMC 通过 NC-SI 借用主网口的流量通道，无需专用管理网口
- **UART**：连接主板串口，实现 Serial-over-LAN

BMC 的电源来自待机电路（standby rail）——主系统关机后依然供电。这意味着 BMC 的功耗（2-5W）是服务器关机时的持续开销。

## 固件与协议栈
BMC 固件是一个完整的嵌入式 Linux 系统。主流选择是 OpenBMC（开源，由 IBM/Meta/谷歌推动）和 AMI MegaRAC（商业闭源，ASPEED 生态的默认选项）。SuperMicro/HPE/Dell 在两者之上叠加自有的 Web UI 和管理功能。

对外协议分为两代。**IPMI**（Intelligent Platform Management Interface）是 1998 年的标准：基于 UDP 623 端口的二进制协议，定义传感器读取（SDR，Sensor Data Record）、事件日志（SEL，System Event Log）、电源控制、LAN 串口重定向等命令。IPMI 的设计年代决定了它的粗糙——明文认证（IPMI 1.5）、固定格式、安全性差。

**Redfish** 是 DMTF 2015 年发布的现代替代：基于 REST API + JSON + OData，通过 HTTPS 访问。资源模型直观（`/redfish/v1/Systems/1` 代表服务器，`/redfish/v1/Chassis/1/Thermal` 代表温度数据），支持标准的认证和 TLS。现代 BMC 同时暴露两个接口，但新工具链（Ansible、Terraform 的服务器插件、云厂商的裸机 API）全部面向 Redfish。

```bash
# IPMI 命令行：读取 CPU 温度、风扇转速
ipmitool sensor | grep -E "CPU|FAN"
ipmitool sel list                        # 事件日志
ipmitool chassis power status            # 电源状态

# Redfish 的 curl 等价物
curl -k -u admin:password https://bmc-ip/redfish/v1/Chassis/1/Thermal
```

## 核心能力
**带外管理**（Out-of-Band）是 BMC 的第一价值。操作系统内的一切管理工具（SSH、远程桌面）都依赖主系统存活——主系统崩溃时它们全部失效。BMC 的独立通道在"操作系统都进不去"时依然可用：远程开机/关机/重启、BIOS 设置修改、固件更新。服务器机房"无人值守"的运维模式完全建立在这个能力上。

**远程 KVM**：BMC 的图形引擎捕获主板的 VGA 输出，压缩后通过管理网络传输，同时把远程键鼠输入注入为 USB 设备。效果是"物理坐在机房里"的完整替代——从开机自检到 BIOS 配置到操作系统安装，全程远程操作。

**虚拟介质**：把远程的 ISO/IMG 文件挂载为服务器的虚拟光驱/软驱/U 盘。远程装机的标准流程：挂载安装镜像 → 设置一次性启动项 → 重启 → KVM 中完成安装。

**传感器与告警**：BMC 持续轮询温度（CPU 核心、内存、VRM、进风口）、电压、风扇转速、电源状态。超过阈值触发事件（写入 SEL 日志，可配置 SNMP trap 或邮件告警）。风扇控制策略由 BMC 的 PID 控制器执行——这也解释了为什么服务器主板的风扇不经过 BIOS 控制。

## 安全面
BMC 是服务器安全中被攻击最频繁的目标之一——它权限极高（可以控制电源、注入键鼠、读取所有传感器）且永远在线。历史漏洞（如 2018 年的 IPMI 密码哈希泄露、多个固件 RCE）证明：BMC 的 Web 界面、IPMI 明文协议、固件更新通道都是攻击面。

工程上的防御实践：管理网络与业务网络物理隔离（BMC 网口接独立交换机/VLAN）；禁用 IPMI 只保留 Redfish（HTTPS）；定期更新 BMC 固件（漏洞修复频率高于 BIOS）；修改默认密码（出厂 `ADMIN/ADMIN` 是许多安全事件的起点）；限制 BMC 的互联网暴露（Shodan 上可以搜到大量暴露的 BMC 控制台）。

## 个人装机的意义
个人 EPYC/Xeon 装机获得的 BMC 能力，是消费级主板无法提供的：远程装机（虚拟介质 + KVM）、崩溃后的远程救援（系统 hang 时 reset）、无显示器的服务器管理（全程 Serial-over-LAN）。这正是 [EPYC 装机](epyc) 和 [Xeon 装机](xeon) 文章中反复出现 BMC 配置话题的原因——首次开机后风扇全速运转需要登录 BMC 调整风扇曲线、默认密码需要修改、NC-SI 共享网口需要配置。

一个实用的装机习惯：新主板到手先通过 `ipmitool lan print` 确认 BMC IP（或配置静态 IP），然后立即更新固件、修改密码、配置风扇曲线——这三件事做完，服务器才算"可以进机房"。
