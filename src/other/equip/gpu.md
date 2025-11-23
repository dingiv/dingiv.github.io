# GPU

## 上下游厂商

- 显卡核心设计者：Nvidia、AMD、Intel，Nvidia 占 90%，AMD 占 9%。
- AIC 厂商：AIC（Add-In Card）厂商，七彩虹、技嘉、华硕、微星等采购芯片，设计并制造完整显卡，包括 PCB、散热器、供电模块和 BIOS 调优。
- 零件供应商
  - 晶圆代工厂：TSMC（台积电）为主，生产 GPU 芯片（3nm/5nm 工艺）。
  - 内存供应商：三星、美光、SK 海力士，提供 GDDR6X/HBM4 等显存。
  - PCB 制造商：如 Foxconn、Pegatron，生产电路板。
  - 散热方案商：如 Delta、Cooler Master，提供风扇/水冷散热。
  - 电源组件供应商：如 TI、Infineon，提供 VRM、MOSFET 等。
- 组装厂：如纬创、和硕，负责显卡组装。
- 分销商/零售商：如京东、Newegg，将显卡销售给消费者。

## 核心参数

- 显存
  - 显存大小：决定运算的规模，大型游戏和 AI 模型往往刚需大量的显存
  - 显存带宽：可明确为 GPU 与显存间的数据传输速率（如 1TB/s）
- 计算性能
  - FLOPS：每秒浮点计算次数（FP16/FP32，16 位和 32 位浮点），直接反应 GPU 的处理速度，衡量科学计算和图形渲染性能
  - TOPS：每秒万亿次操作（INT8/FP8，8 位整型和 8 位浮点），衡量 AI 推理性能
- 功耗
  - TDP：热设计功耗（如 300W），反映功耗和散热需求

其他技术参数：

- GPU 核心：架构（如 NVIDIA Rubin）、CUDA 核心数/流处理器，决定计算能力
- 时钟频率：核心频率（如 2.5GHz）、Boost 频率，影响性能
- CUDA 核心/流处理器数：如 RTX 5090 约 21,760 个，决定并行计算能力
- 显存（VRAM）：类型（如 GDDR6X、HBM4）、位宽（如 256-bit）
- 工艺制程：如 3nm，提升能效比。

## 测试项目

- 图形基准测试：3DMark Time Spy，测试帧率、光追，反应游戏性能
- 通用计算测试：Geekbench Compute，评估通用计算能力，反应 AI 计算性能
  值得一提的是，AI 计算可以使用 GPU 并联来提高运算能力。

## Nvidia 常见型号

Nvidia 目前市场的主力是 Blackwell 系列和 Ada Lovelace 系列，Ampere 系列正在逐步退出。Hopper 系列是专注于企业级市场。

### 消费级 Geforce

Geforce RTX 型号专注于消费级市场。其中 RTX 50X0 系列属于 Blackwell，RTX 40X0 系列属于 Ada Lovelace 系列，30X0 属于 Ampere 系列。

| 型号          | 架构      | 显存     | FP32 TFLOPS |
| ------------- | --------- | -------- | ----------- |
| 5090          | Blackwell | 32       | 104-125     |
| 5090D/5090Dv2 | Blackwell | 32,24    | 104-125     |
| 5080          | Blackwell | 16       | 70          |
| 5070/5070Ti   | Blackwell | 12,16    | 37          |
| 5060/5060Ti   | Blackwell | 8,16     | 23          |
| 5050          | Blackwell | 8        | 13          |
| 4090          | Lovelace  | 24,48    | 83          |
| 4080          | Lovelace  | 16       | 40-49       |
| 4070/4070Ti   | Lovelace  | 12,16    | 29-40       |
| 4060/4060Ti   | Lovelace  | 8,16     | 15-22       |
| 3090/3090Ti   | Ampere    | 30       |             |
| 3080/3080Ti   | Ampere    | 10,12,20 |             |
| 3070/3070Ti   | Ampere    | -        | -           |
| 3060/3060Ti   | Ampere    | 8,12     |             |
| 3050          | Ampere    | 6        |             |

> 5090D/5090Dv2 中国特供版，阉割了显存或者带宽

### 专业级 Quadro

RTX 专业级，专注于图形工作站等领域，也可以用来训练 AI，但是优化的不如 Blackwell。
| 型号     | 系列      | 显存  | FP32 TFLOPS |
| -------- | --------- | ----- | ----------- |
| PRO 6000 | Blackwell | 96    | 125         |
| PRO 5000 | Blackwell | 48,72 | 65          |
| PRO 4000 | Blackwell |       |             |
| PRO 3000 | Blackwell |       |             |
| PRO 2000 | Blackwell |       |             |
| PRO 1000 | Blackwell |       |             |
| PRO 500  | Blackwell |       |             |
| 6000 Ada | Lovelace  | 48    | 91          |
| 5000 Ada | Lovelace  |       | 65          |
| 4000 Ada | Lovelace  |       | 52          |
| A6000    | Ampere    | 48    | 38          |
| A5500    | Ampere    | 24    | 22          |
| A5000    | Ampere    | 24    | 27          |
| A4500    | Ampere    | 20    | 23          |
| A4000    | Ampere    | 16    | 19          |
| A2000    | Ampere    | 12    | 19          |
| A1000    | Ampere    | 8     | 19          |
| A800     | Ampere    |       | 19          |
| A400     | Ampere    |       | 19          |

### 企业级 Tesla

当前 Hopper 架构是市场中的主力。
| 型号    | 系列      | 显存  | FP32 TFLOPS |
| ------- | --------- | ----- | ----------- |
| B300    | Blackwell |       |             |
| B200    | Blackwell |       |             |
| B100    | Blackwell |       |             |
| H100NVL | Hopper    |       |             |
| H200    | Hopper    |       |             |
| H100    | Hopper    |       |             |
| H800    | Hopper    |       |             |
| H20     | Hopper    |       |             |
| L40S    | Lovelace  | 48    | 91          |
| L40     | Lovelace  | 48    | 45          |
| L20     | Lovelace  | 48    | 45          |
| L4      | Lovelace  | 24    | 30          |
| A100    | Ampere    | 40,80 | 19          |
| A30     | Ampere    |       | 10          |
| A10     | Ampere    | 24    | 31          |
| V100S   | Volta     | 32    |             |
| GV100   | Volta     | 32    |             |
| V100    | Volta     | 32    |             |
| Titan V | Volta     | 12    |             |

## AMD 常见型号

### 消费级

AMD 设计。专注于消费级市场
| 型号           | 显存  |     |
| -------------- | ----- | --- |
| 9070/9070XT    | 12,16 |     |
| 9070GRE        | 12,16 |     |
| 9060XT         | 32    |     |
| 7900XT/7900XTX | 20,24 |     |
| 7900GRE        | 16    |     |
| 7800XT         | 16    |     |
| 7700XT         | 12,16 |     |
| 7600/7600XT    | 16    |     |

### 专业级

AMD PRO 系列
| 型号        | 显存 |     |
| ----------- | ---- | --- |
| R9700       | 32   |     |
| W7900       | 48   |     |
| W7800       | 32   |     |
| W7600/W7500 | 8,16 |     |
| W9000       | 16   |     |

### 企业级

AMD Instinct 系列
| 型号          | 显存 |     |
| ------------- | ---- | --- |
| MI350         | 288  |     |
| MI300X/MI325X | 192  |     |
| MI355X        | 32   |     |

## 国产厂商

### 摩尔线程

主力是企业级 AI 算力。

### 砺算科技

主力是消费级市场。

### 南京沐曦

### 景嘉微

### 寒武纪
