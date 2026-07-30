---
title: Threadripper 装机
order: 52
---

# Threadripper 个人装机生态
Threadripper 是 AMD 的高端桌面平台（HEDT），定位在 Ryzen 和 EPYC 之间。与 EPYC 的纯服务器取向不同，Threadripper 同时追求高核心数和高单核频率——64 核 3990X 的基础频率 2.9 GHz、单核加速 4.3 GHz，而同代 64 核 EPYC 7742 只有 2.25/3.4 GHz。这个特性使 Threadripper 成为需要"多核数量 + 单核不差"场景的最优解：编译、渲染、EDA、本地 AI 推理的混合负载。

代价是价格。Threadripper 的二手市场远不如 EPYC 和 Xeon 繁荣——服务器退役量大、价格低；Threadripper 只有个人用户和工作室退役，流通量小、价格坚挺。

## 各代可玩性评价
Threadripper 的代际比 EPYC 更复杂——消费级 HEDT（X399/TRX40）和工作站 PRO 系列（WRX80/TRX50/WRX90）使用了不同的插座和芯片组，即使同一代 CPU 也无法跨平台使用。

| 代际       | 代号     | 平台  | 插座  | 核心数   | 内存             | 可玩性 | 备注                           |
| ---------- | -------- | ----- | ----- | -------- | ---------------- | ------ | ------------------------------ |
| 1000/2000  | Zen/Zen+ | X399  | TR4   | 最多 32C | DDR4 UDIMM       | 低     | IPC 弱、主板少、不建议         |
| 3000       | Zen 2    | TRX40 | sTRX4 | 最多 64C | DDR4 UDIMM       | **高** | 个人装机甜点，PCIe 4.0         |
| PRO 3000WX | Zen 2    | WRX80 | sWRX8 | 最多 64C | DDR4 RDIMM/UDIMM | 中     | PRO 平台贵、8 通道内存         |
| PRO 5000WX | Zen 3    | WRX80 | sWRX8 | 最多 64C | DDR4 RDIMM/UDIMM | 中     | Zen 3 IPC + 8 通道，但主板天价 |
| 7000       | Zen 4    | TRX50 | sTR5  | 最多 64C | DDR5 UDIMM       | **低** | 主板+DDR5 太贵，二手极少       |
| PRO 7000WX | Zen 4    | WRX90 | sTR5  | 最多 96C | DDR5 RDIMM       | **低** | 工作站价格，不适合个人         |

3000 系列（Zen 2, TRX40）是当前个人装机的唯一甜点。3990X（64C/128T）二手约 8000-12000 元，TRX40 主板 1500-3000 元，DDR4 UDIMM 与消费级通用（不需要 ECC RDIMM）。相比同代 EPYC 7742（CPU 800 元 + SP3 主板 1500 元），Threadripper 的 CPU 贵 10 倍、主板差不多——但换来的是翻倍的单核频率和消费级的 ATX 板型便利性。这个价格差是否值得，取决于你的负载是否真的需要高单核。

PRO 5000WX 系列（Zen 3）在 TRX40 上不可用，需要 WRX80 主板。WRX80 的价格（3000-6000 元）和 sWRX8 的 RDIMM 要求使其与 EPYC Milan 直接竞争——而 Milan 的主板选择更多、CPU 更便宜。除非需要同时满足"64 核 + 单核 4.0 GHz+ + 8 通道内存"这三个条件的组合，否则 EPYC Milan 性价比远超 PRO 5000WX。

1000/2000 系列（X399）目前不值得。Zen/Zen+ 的 IPC 偏低、X399 的 PCIe 3.0 限制了 NVMe 和 GPU 扩展、二手主板存量少且质量参差不齐。唯一可能的理由是极其便宜——如果碰到整机在 2000 元以下且有验证过的工作状态，可以作为一个"入门 16-32 核"的选择。但多数情况下，攒一台新的 AM5 Ryzen 9 是更好的替代。

7000 系列目前不适合个人装机——TRX50 主板（4000-8000 元）和 DDR5 UDIMM 成本过高，且二手市场几乎不存在。一个全新 7980X（64C, Zen 4）平台的总成本轻松超过 20000 元，而这个预算足以买一套二手 EPYC Genoa 或两块 A100 级别的 GPU。

## CPU 选购指南
Threadripper 的命名体系比 EPYC 简洁得多。消费级 Threadripper 型号为 4 位数字（如 3960X、3970X、3990X），核心数一目了然：最后两位 ÷ 10 ≈ 核心数的一半。3960X = 24C、3970X = 32C、3990X = 64C。

**消费级 Threadripper（3000 系列, sTRX4, TRX40）**：
- 3960X（24C/48T, 3.8/4.5 GHz, 128MB L3）≈ 4000-6000 元 — 入门 24 核，单核接近同代 Ryzen 9
- 3970X（32C/64T, 3.7/4.5 GHz, 128MB L3）≈ 5500-8000 元 — 性价比甜点
- 3990X（64C/128T, 2.9/4.3 GHz, 256MB L3）≈ 8000-12000 元 — 旗舰，128 个线程

超频是 Threadripper 的独特卖点。与 EPYC（锁频）不同，TRX40 上的 Threadripper 完全开放超频。3990X 在充足散热下可稳定全核 3.5-3.8 GHz——相比默认全核约 3.2 GHz 提升 10-20%。代价是功耗从 280W TDP 飙升至 400-500W，需要顶级散热和 VRM。

**PRO 系列（sWRX8, WRX80）**：
- 3945WX（12C/24T, 4.0/4.3 GHz）— 入门 PRO，单核极强
- 3955WX（16C/32T, 3.9/4.3 GHz）— 预算内高单核选择
- 3975WX（32C/64T, 3.5/4.2 GHz）— 中端工作站
- 3995WX（64C/128T, 2.7/4.2 GHz）— 旗舰

PRO 系列与消费级 Threadripper 的关键差异：支持 8 通道内存（vs 消费级 4 通道）、支持 RDIMM/LRDIMM（容量上限 2TB+ vs 消费级 256GB UDIMM）、支持 AMD PRO 管理特性（类似 Intel vPro 的远程管理和安全功能）。PRO 系列也锁频——不能用 TRX40 的超频玩法。注意：PRO 3000WX（Zen 2）的 64 核型号也命名为 3995WX，与消费级 3990X 相似。购买前确认型号后缀——X 结尾是消费级（TRX40）、WX 结尾是 PRO（WRX80）。

**ES/QS**：Threadripper 的 ES 芯片在二手市场上比 EPYC 少得多——因为 Threadripper 的量本身就小。但 Zen 2 的 ES 有一个特定问题值得注意：部分早期 ES 的 IF（Infinity Fabric）频率上限比正式版低，导致内存超频能力受限。QS 和正式版几乎没有区别。

## 主板选择
Threadripper 主板生态比 EPYC 健康得多——各大消费级主板厂商（华硕、技嘉、微星、华擎）都有 TRX40 产品线。这些是真正的消费级主板，有 RGB、有 BIOS 图形界面、有 Windows 下的超频软件。

**TRX40（sTRX4, 消费级 Threadripper）**：
- 华硕 ROG Zenith II Extreme（E-ATX, 顶级 VRM, 16 相 Infineon, 双 10G）— 价格 2000-3000 元
- 技嘉 TRX40 AORUS XTREME（E-ATX, 16 相, Intel X550 双 10G）— 价格 1800-2800 元
- 华擎 TRX40 Taichi（ATX, 8 层 PCB, 3 个 M.2, 性价比最高）— 价格 1200-1800 元
- 微星 TRX40 PRO（ATX, 基础版, 少 M.2 和 USB）— 价格 1000-1500 元

与 EPYC 的服务器主板不同，TRX40 主板有完整的消费级 I/O——3-5 个 M.2 PCIe 4.0 x4 插槽（均来自 CPU 直连）、7.1 音频、Wi-Fi 6、USB 3.2 Gen2x2。而且大部分为 ATX 板型，不需要特殊机箱。BIOS 图形界面支持鼠标操作，体验完全等同于消费级 Ryzen 主板——这是 EPYC 和 Xeon 服务器主板无法提供的便利。

**WRX80（sWRX8, PRO 系列）**：
- 华硕 PRO WS WRX80E-SAGE SE WIFI（E-ATX, 8 通道, 7 条 PCIe 4.0 x16, BMC）— 3000-5000 元
- 技嘉 WRX80-SU8-IPMI（E-ATX, 服务器取向, 双 10G, BMC）— 2800-4500 元

WRX80 主板是工作站/服务器混合体——有 BMC 远程管理（PRO 用户对可靠性要求高）、但保留消费级的 BIOS UI。板型几乎都是 E-ATX。如果已经有 M-ATX 或 ATX 机箱，需要升级。

**TR4/TRX4 插座的物理陷阱**：TR4（X399, 1st/2nd Gen）和 sTRX4（TRX40, 3rd Gen）物理尺寸完全相同——CPU 可以物理插入对方的插座。但针脚定义不同，插错后果是烧毁 CPU 或主板。AMD 在 sTRX4 插座上改了 ID pin 的识别电压，理论上主板能检测到不兼容的 CPU 并拒绝上电——但不要用这个安全机制来赌博。买 CPU 前确认对应插座。

**推荐清单**：
- 性价比：华擎 TRX40 Taichi（ATX, 1200-1800 元，3 M.2, 8 层 PCB, VRM 够 64 核超频）
- 旗舰：华硕 ROG Zenith II Extreme Alpha（E-ATX, 2000-3000 元，顶级用料）
- 紧凑：暂无真正的 M-ATX TRX40 板——所有 TRX40 板均为 ATX+，需要确认机箱兼容性

## 散热方案
Threadripper 的 IHS（集成散热顶盖）面积远大于消费级 CPU——sTRX4 的 CPU 封装为 58.5×75.4mm，与 SP3（EPYC）相同。关键差异在于热量分布：EPYC 的 chiplet 分布在 CPU 基板的外围边缘（最多 8 个 CCD 围绕一个中央 IO Die），而 Threadripper 的 chiplet 布局更紧凑（最多 8 个 CCD 围绕 IO Die，但封装尺寸比 SP3 略小，CCD 间距更近）。这个差异对风冷散热器的底面积要求没有本质区别——都需要全覆盖。

**风冷**：Noctua NH-U14S TR4-SP3（140mm 单塔，支持 350W+）是 Threadripper 风冷的实际标准。NH-U14S 的铜底面积完全覆盖 Threadripper 的 CCD 分布区域，即使 64 核超频到 400W 也能维持在 80-90°C。IceGiant ProSiphon Elite（重力热管 + 均热板混合）提供更好的极限散热能力（支持 500W），但体积巨大（280mm 高），需要确认机箱能容纳。

**水冷**：Threadripper 冷头必须使用 sTR4/sTRX4 专用冷头——消费级 AM4/AM5 冷头的微水道面积只覆盖 CCD 的中心部分，边缘 CCD 完全冷却不到。大部分主流水冷品牌有 Threadripper 专用版本：Arctic Liquid Freezer II 4U-M（420mm 一体式, Server Edition）、Enermax LiqTech II TR4（全覆盖冷头，但有早期版本堵塞的负面历史）。分体水冷方面，Optimus PC 的 Threadripper 冷头是社区评价最好的全覆盖设计。

**功耗与散热匹配**：
- 3960X（24C, 280W TDP）：NH-U14S 足够，默频满载 65-75°C
- 3990X（64C, 280W TDP）：NH-U14S 默频 70-80°C，超频 400W+ 需要 IceGiant 或 360mm+ 水冷
- 3990X 全核 3.8 GHz 超频约 480W——需要 420mm 一体式或分体水冷，且对 VRM 和机箱通风有高要求

## 内存选型
Threadripper 消费级（TRX40）使用标准 DDR4 UDIMM——与 Ryzen 消费级平台完全相同。不需要 ECC RDIMM，不需要去二手服务器市场翻找。这意味着内存采购是三个平台（EPYC/Xeon/Threadripper）中最简单的。

TRX40 支持 4 通道 DDR4，大多数 TRX40 主板有 8 个 DIMM 插槽（4 通道 × 2 DIMM/通道）。官方支持最高 DDR4-3200，实际可通过 XMP/DOCP 超频到 3600-3800 MHz。Zen 2 的 IF（Infinity Fabric）时钟与内存时钟 1:1 耦合，上限约 1800-1900 MHz（对应 DDR4-3600 到 3800）。超过 1900 MHz IF 会切换到 2:1 模式，延迟增加——所以内存频率没必要拉太高，DDR4-3600 CL16 是最佳甜点。

4 通道要求 4 或 8 条 DIMM 才能获得最大带宽。2 条 DIMM（双通道模式）也能工作，但带宽减半。预算紧张时 4×16GB DDR4-3200 （约 400-600 元，全新非 ECC）是最经济的 64GB 配置。

WRX80（PRO）支持 8 通道 DDR4，可使用 UDIMM 或 RDIMM。RDIMM 的二手价格参考 [EPYC 装机](epyc#内存采购)，但 PRO 平台通常建议直接上 RDIMM——8 通道 × 128GB LRDIMM = 1TB 内存容量的上限是 PRO 平台的核心价值之一。

## 电源选型
TRX40 主板使用标准 24 pin 主供电 + 双 8 pin EPS CPU 供电——与高端 Ryzen 9 主板的接口一致，大部分 850W+ 的 ATX 电源都有双 8 pin EPS。不需要特殊的服务器电源模块（CRPS）。

**功耗预估**：
- 3960X 默频平台 + 64GB + 2 NVMe + 1 GPU ≈ 400-500W
- 3990X 默频平台 + 128GB + 4 NVMe + 2 GPU ≈ 700-900W
- 3990X 超频 400W+ + 4 GPU（如 4×RTX 3090）≈ 1500-1800W

64 核超频 + 多 GPU 场景已经到了普通 ATX 电源的极限——需要 1600W+ 电源或双电源方案。此时消费级平台的优劣变成：便利性消失，开始接近 EPYC 服务器的复杂度。

## 装机方案参考

### 24 核入门工作站
CPU：Threadripper 3960X（24C/48T, 3.8 GHz）≈ 4000-6000 元。主板：华擎 TRX40 Taichi ≈ 1200-1800 元。内存：4×16GB DDR4-3600 CL16 ≈ 600-800 元（64GB）。散热器：Noctua NH-U14S TR4-SP3 ≈ 500 元。总成本约 6300-9100 元。适合：单 CPU 替代双路 Xeon E5、编译大型项目、中等规模 3D 渲染、同时运行多个 VM。

### 64 核旗舰
CPU：Threadripper 3990X（64C/128T, 2.9/4.3 GHz）≈ 8000-12000 元。主板：华硕 ROG Zenith II Extreme ≈ 2000-3000 元。内存：8×16GB DDR4-3600 CL16 ≈ 1200-1600 元（128GB）。散热器：Noctua NH-U14S TR4-SP3 ≈ 500 元。总成本约 11700-17100 元。适合：V-Ray/Blender 渲染主力机、EDA 仿真、64 核本地 CI 服务器。

对比方案：同预算的 EPYC Milan 64 核（7773X）配 Supermicro H12SSL-NT + 256GB RDIMM，总价约 8500-13500 元。EPYC 方案便宜 30-40% 且内存大一倍，但单核频率只有 2.2/3.5 GHz——Threadripper 3990X 的单核加速 4.3 GHz 高出 23%。如果你的编译流程中 30% 时间是单线程串行步骤，这个单核差距可能比 64 个核心更重要。

### PRO 工作站（需要 8 通道内存和超大容量）
CPU：Threadripper PRO 3995WX（64C/128T, 2.7/4.2 GHz）≈ 12000-18000 元。主板：华硕 PRO WS WRX80E-SAGE ≈ 3500-5000 元。内存：8×32GB DDR4-3200 RDIMM ≈ 1600-2400 元（256GB）。散热器：Noctua NH-U14S TR4-SP3 ≈ 500 元。总成本约 17600-25900 元。适合：需要 512GB+ 内存的 CFD 仿真、超大内存数据库、多 GPU 深度学习训练节点。

## 常见陷阱
1. **Threadripper ≠ EPYC 的散热器通用性**：TR4 和 sTRX4 的扣具与 SP3 兼容——都用 4 点 LGA 4094 扣具。买标注"TR4"或"SP3"的风冷散热器即可互用。但 sWRX8（WRX80, PRO 系列）的扣具与 SP3 略有不同——螺丝长度差了约 1mm。用 sTRX4 的散热器装到 sWRX8 主板上可能压不紧 IHS。确认散热器包装中是否包含 WRX80 兼容螺丝。

2. **SO-DIMM 支持**：部分 TRX40 的 M-ATX 或 mini-ITX 板（如华擎 TRX40D4U-2L2T）使用 SO-DIMM 槽（笔记本内存）以节省 PCB 空间。这些板的 SO-DIMM 不兼容标准 DDR4 CODIMM——确认主板规格后再买内存。

3. **VRM 散热与超频**：3990X 超频 400W+ 时 VRM 温度可能超过 100°C。TRX40 主板的 VRM 散热片通常没有主动风扇——高负载长时间运行后 VRM 热保护可能导致 CPU 降频。解决：在 VRM 散热片上加装一个小风扇（40-60mm）直吹，多数板子预留了螺丝孔。

4. **Windows vs Linux 的 NUMA 感知**：64 核 Threadripper 在 Windows 10/11 上可能被错误识别为 4 个 NUMA 节点（每 16 核一个节点）而实际上只有一个 IO Die 连接所有 CCD。在 BIOS 中将内存交错（Memory Interleaving）设为 Channel 模式（而非 Die 模式）可以解决此问题。Linux 内核 5.8+ 对 Threadripper 的 NUMA 拓扑自动检测更好，通常不需要手动干预。

5. **PCIe 通道分配**：TRX40 的 64 条 PCIe 4.0 通道（来自 CPU）+ 24 条 PCIe 4.0 通道（来自 TRX40 芯片组）看似充裕。但 3990X 如果插满 4 块 GPU（每条 x16），会用光 CPU 直连的 64 条通道。芯片组提供的通道通过 PCIe 4.0 x8 上行连接与 CPU 通信——等效带宽 16 GB/s——插在芯片组通道上的 NVMe 和 USB 共享这个带宽。确认主板手册中哪个 M.2 插哪个 PCIe 槽是什么来源（CPU 直连 vs PCH），将需要高带宽的设备（主 NVMe、主 GPU）插在 CPU 直连的槽上。

6. **BIOS 更新与 3990X 支持**：部分发布较早的 TRX40 主板需要 BIOS 更新才能支持 3990X（64C）。第一次开机如果没有显示，可能是 BIOS 版本太旧——TRX40 主板大多支持 USB BIOS Flashback（无 CPU 刷 BIOS），参考主板手册操作即可。
