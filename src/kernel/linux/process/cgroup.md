---
title: Cgroup
order: 35
---

# Cgroup 与 SELinux
Cgroup 和 SELinux 是 Linux 进程管理的两个"管控"子系统，但它们管控的维度完全不同：Cgroup 管"进程能用多少资源"——CPU 时间、内存、IO 带宽；SELinux 管"进程能碰什么对象"——文件、端口、其他进程。一个限流，一个授权。容器的"资源限制 + 安全约束"两翼分别由它们承担。

## Cgroup：资源的分组计量与限制
Cgroup（Control Group）的设计动机源于一个朴素的工程问题：内核的资源调度（CPU 调度器、内存回收、IO 调度）以进程为单位工作，但实际需要管控的单位是"一组进程"——一个服务的多个 worker、一个用户的所有任务、一个容器的全部进程。Cgroup 在进程和调度器之间插入了一个分组层：进程被组织为树状层级，每个节点可以附加资源限制，调度器在做决策时读取这些限制。

### 层级模型与控制器
Cgroup 采用"层级 + 控制器"的模型。层级（hierarchy）是 cgroup 目录的组织结构，控制器（controller）是一种资源的管理逻辑。常见的控制器：

| 控制器 | 资源 | 典型接口 (v2) |
|--------|------|--------------|
| cpu | CPU 时间配额与权重 | `cpu.max`, `cpu.weight` |
| memory | 内存使用上限与 OOM 行为 | `memory.max`, `memory.swap.max` |
| io | 块设备 IO 带宽与权重 | `io.max`, `io.weight` |
| pids | 进程数量上限 | `pids.max` |
| cpuset | CPU 核与 NUMA 节点亲和 | `cpuset.cpus`, `cpuset.mems` |
| freezer | 进程组冻结/恢复 | `cgroup.freeze` |

每个控制器的实现方式各异：cpu 控制器修改调度实体的权重和配额，memory 控制器在内存分配路径上记账并在超限时触发回收或 OOM，io 控制器在块层提交路径上做限速。但对外接口统一：写文件设置限制，读文件查看使用量——`echo 512M > memory.max` 和 `cat memory.current`。

### v1 与 v2：两种设计哲学
Cgroup 的 v1 和 v2 体现了两种截然不同的层级设计。

v1 的模型是"每控制器一棵树"：CPU 层级树和内存层级树相互独立，进程可以同时挂在不同控制器的不同层级位置上。灵活性高，但配置碎片化——一个服务可能要维护它在四棵树上的位置，且跨控制器的层级同步（如 CPU 亲和与内存 NUMA 的对应关系）需要管理员手动保证。

v2 的模型是"统一层级"：一棵树承载所有控制器，进程挂载位置唯一。接口收敛为 `cpu.max`、`memory.max` 这样的扁平文件，`cgroup.subtree_control` 控制哪些控制器向下层传递。v2 还引入了"无内部进程"规则——只有叶子节点可以承载进程，中间节点只做分组——这消除了 v1 中"进程和内部分组混合"导致的资源记账歧义。当前主流发行版和 Kubernetes（cgroupdriver=systemd）已默认使用 v2。

### Cgroup 与容器
容器运行时将每个容器映射为一个 cgroup 子树：`docker run --cpus 2 --memory 1g` 翻译为对容器 cgroup 目录下 `cpu.max = 200000 100000`（2 核）和 `memory.max = 1073741824` 的写入。Kubernetes 的 Pod 资源限制、limit/request 的 QoS 模型，底层都是 cgroup 配置的组合。

两个工程细节值得注意。**内存 OOM 的顺序**：容器超限时，内核先尝试回收容器内可回收页（page cache），回收失败才触发 OOM killer——OOM killer 在容器 cgroup 范围内选择进程，不会波及宿主机其他容器。**CPU 限制的调度语义**：`cpu.max` 是硬配额——超限的进程被调度器 throttled（排队等待下一个周期），表现为容器内 CPU 使用率的"锯齿状"抖动。CPU 限制设置过紧时容器内延迟会异常升高——这是配额机制的固有行为，不是 bug。

## SELinux：强制访问控制
SELinux 回答的问题与传统的 Unix 权限（DAC）完全不同。DAC 的判断规则是"进程的用户是否有权限"——root 可以访问一切。SELinux 的判断规则是"进程的类型和对象的类型是否符合策略"——即使 root 进程，类型不匹配依然被拒绝。DAC 是"谁"的权限，MAC 是"是什么"的权限。

### 类型强制（Type Enforcement）
SELinux 的核心模型是类型强制：每个进程有一个域（domain，一种类型），每个资源（文件、端口、socket）有一个类型（type），策略定义了"哪个域可以访问哪个类型的什么操作"。访问发生时内核查询策略缓存（AVC，Access Vector Cache）裁决。

一个经典的容器示例：Docker 给容器进程打上 `container_t` 域，给宿主机文件打上 `usr_t`、`etc_t` 等类型。即使容器进程以 root 运行（DAC 层面无所不能），策略规定 `container_t` 只能访问 `container_file_t` 类型的文件——容器进程访问宿主机的 `/etc/passwd` 被 SELinux 拒绝。这就是容器逃逸场景中 SELinux 能兜底的原因：逃逸出的 root 进程依然受 `container_t` 域的策略约束。

### 策略的组织：类型、角色、用户
SELinux 的完整标签是三元的 `user:role:type`。类型（type）是访问控制的主体，角色（role）限制用户可以进入哪些域，用户（SELinux user）与 Linux 用户映射。工程实践中 90% 的注意力在类型上——编写策略就是定义"哪个域可以访问哪个类型"的规则（allow 规则）。

策略编写使用 `allow httpd_t httpd_config_t:file { read open };` 这样的语句。系统预置的策略按软件包组织（`httpd.pp`、`container.pp`），管理员通过布尔开关（boolean）和自定义规则微调。策略排错的标准工具链：`ausearch -m avc` 查 AVC 拒绝日志 → `audit2allow` 生成规则建议 → 确认合理后加入策略。

### 模式与排错
SELinux 有三种模式：Enforcing（强制执行）、Permissive（只记录不拒绝）、Disabled（关闭）。排错流程的标准起点是把相关域切到 Permissive——看问题是否消失，消失则确认是 SELinux 拒绝，再用 AVC 日志定位具体规则。

最常见的排错场景是"服务起不来但日志没报错"——多半是 SELinux 拒绝被服务吞掉了。`setenforce 0` 立即验证（生产环境临时排查用），确认后用 `audit2allow` 生成永久规则，而不是让 SELinux 一直关闭。

### AppArmor 的对比
AppArmor 是 Ubuntu 系默认的 MAC 系统，与 SELinux 的模型差异：SELinux 基于类型（inode 上的标签，文件移动后标签跟随 inode），AppArmor 基于路径（profile 中写文件路径，文件移动后规则失效需要更新路径）。SELinux 策略表达力更强、粒度更细但学习曲线陡峭；AppArmor 配置直观但灵活性差。容器生态对两者的支持都成熟——Docker 可以同时应用 SELinux 标签和 AppArmor profile。选择通常跟随发行版默认：RHEL 系用 SELinux，Ubuntu 系用 AppArmor。

## Cgroup 与 SELinux 在容器安全栈中的位置
完整的容器安全栈是分层防御的：Namespace 提供视图隔离（进程以为自己独立）、Cgroup 提供资源边界（坏进程无法拖垮宿主）、Capabilities 裁剪特权（root 不再全能）、Seccomp 屏蔽危险系统调用（缩小内核攻击面）、SELinux/AppArmor 提供独立于 DAC 的最后裁决（逃逸后的兜底）。每一层的失效都不至于让整个系统失守——这是容器安全工程的核心原则：不信任任何单一机制，用纵深防御换取整体可靠性。
