# 待写博客内容清单

## 软件工程
- 可读性与可维护性：可读不等于可维护，可维护包含可写性和可改性
- 可靠性与可用性：可靠 = 不能错，可用 = 可以错但必须马上纠正
- 需求分析：基础需求、刚需、亮点需求的分层

## 容器与虚拟化
- 容器本质：被蒙蔽了双眼的进程（多进程容器 = 一组共享同一片树叶的进程）
- 正向容器与反向容器（安全计算视角）
- 虚拟化模块新增：SPDK 和 DPDK
- QEMU 调试内核开发：Linux 错误捕获机制、内核调试专用虚拟机模拟器
- 虚拟机调试增强：实时显示物理内存值、监控 CPU 读取的设备寄存器

## 存储与 IO
- 流式存取与块式存取
- sendfile 零拷贝
- DMA、IOMMU、iommu_group 体系

## 算法
- 矩阵与图遍历算法：处理当前节点 → 试探下一个节点 → 决定是否进入
- 二分法循环条件与状态转移：left <= right 配合 mid = left ± 1，三分支
- 并发锁的思想：访问自肃
- 内存屏障：何时需要、哪些场景需要、与并发的关系

## AI 部署
- 模型集群 PD 分离（Prefill/Decode 分离）
- 会话上下文动态压缩：上下文过长时用语言模型总结压缩

## 硬件与主板
- BMC 主板管理器详解
- 4G vs 5G 模块转接板差异：总线协议（USB vs PCIe+MHI）、功耗控制（1A vs 瞬时大电流）、驱动成熟度（option/qmi_wwan）
- 局域网通信技术总览：Ethernet、PCIe、USB、蓝牙、2.4G/5G、InfiniBand、Fiber Channel、Thunderbolt、CAN

## 前端
- RN 与 Flutter 技术对比：diff 树机制差异

## 后端与网络
- API Gateway：协议转换（HTTP → RPC/gRPC）
- OpenAI RPC 库
- Session 管理：user id、session id、task id 三层身份
- 可观测性：ELK、Prometheus、监控、日志

## 图形
- SDL2 图形库

## Linux 内核（已完成）
- proc 文件系统：proc_fs、proc_create、proc 目录常见条目 → 已扩充 `file/pfs.md`
- /sys 目录与 kset/kobject 模块 → 已扩充 `file/pfs.md`
- 内核编译参数与内核启动命令行 → 已扩充 `develop/kbuild.md`（命令行参数五类分组表）
- 常用内核命令行参数（调试开发场景）→ 同上
- 内核并发机制：原子变量、RCU、per-CPU 变量、seqlock、单写者模型、中断与抢占禁用 → 新建 `irq/sync.md`
- Linux 进程树与继承性：fork 树拓扑结构 → 已扩充 `process/index.md`（fork 继承表 + 锁陷阱 + 会话）
- EFI 是一个微型操作系统 → 已扩充 `power/boot.md`（UEFI 微系统视角）
- Linux 性能调优：服务器莫名卡顿排查 → 新建 `sde/sre/tuning.md`
- 内存管理：e820 表、e820 与 memblock 的关系、x86 物理内存映射兼容性设计 → 已有覆盖（`mm/pmm.md` + `power/boot.md` 早期内存管理节）
- 手写操作系统章节（新增 blog 章节）→ 待规划（章节级）

## 数据库
- 数据库分层：访问层、执行层、存储层
- ACID 详解

## Shell
- `$()` 命令替换的错误捕获陷阱：local 赋值吞掉错误，local 声明与赋值分离可以避免
