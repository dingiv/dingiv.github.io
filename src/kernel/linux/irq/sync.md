---
title: 内核并发机制
order: 25
---

# 内核并发原语
内核的并发环境比用户态苛刻得多：多核真并行、中断随时打断、抢占可能发生在任何点。内核提供了一整套并发原语，从最重的锁到最轻的读侧无开销，各有明确的适用场景。选型的核心逻辑是**按数据结构的读写模式匹配原语**——写多读少用自旋锁/互斥锁，读多写少用 RCU/seqlock，每 CPU 独立数据用 per-CPU 变量，简单计数用原子变量。

## 并发的三个来源
内核代码必须防御的并发来自三个方向：**多核并行**（两个 CPU 同时执行同一代码路径）、**中断打断**（进程上下文的代码执行到一半被中断处理程序打断，后者访问同一数据）、**抢占**（内核态任务被调度器换出，另一个任务访问同一数据）。任何"这个数据只有一个地方访问"的假设在这三个来源面前都不成立——每核的统计计数器会被其他核的中断处理程序碰，链表会被其他进程上下文的软中断碰。

禁用中断和禁用抢占是应对后两个来源的暴力手段：`local_irq_disable()`/`local_irq_save()` 关中断（只保护"当前 CPU 的中断来源"，不挡其他核）、`preempt_disable()` 关抢占。它们通常不单独使用——而是作为自旋锁获取的副作用（spin_lock 在 CONFIG_PREEMPT 下隐含关抢占，需要中断安全时用 spin_lock_irqsave 关中断）。单独长时关中断是内核 bug 的高发区（中断延迟飙升、看门狗触发）。

## 原子变量：最小的并发单元
原子变量（`atomic_t`/`refcount_t`）依赖硬件原子指令（x86 的 LOCK 前缀、ARM 的 LL/SC）实现不可分割的读-改-写。适用于**单一变量的简单操作**：计数器、标志位、引用计数。

```c
atomic_t counter = ATOMIC_INIT(0);
atomic_inc(&counter);
int v = atomic_read(&counter);
atomic_cmpxchg(&counter, 5, 10);   // CAS

// refcount_t 是引用计数的强化版——内置溢出/下溢检查（防 use-after-free 类漏洞）
refcount_inc(&obj->refcnt);
```

`refcount_t`（2017 年从 atomic_t 分化）值得单独记住：它不是简单的 atomic——增加时检查溢出、减少时检查下溢到 0 后再减会警告并饱和——把"引用计数错误"从静默的内存腐败变成有日志的防御。新代码的引用计数一律用 refcount_t。

原子变量的边界：它只保护单个变量自身的操作一致性，**不保护多个变量之间的一致性**（两个原子变量的组合更新不是原子的）——那是锁的职责。

## 自旋锁与互斥锁：经典的临界区保护
**自旋锁**（spinlock）：获取失败的 CPU 忙等（自旋）——不睡眠。适用于**临界区极短**（几十纳秒到微秒）且**不能睡眠的上下文**（中断处理程序、软中断、持有自旋锁期间）。实现基于排队自旋锁（qspinlock）——等待者排队自旋，FIFO 公平，缓存行在 CPU 间传递一次。

```c
spinlock_t lock = __SPIN_LOCK_UNLOCKED(lock);
spin_lock(&lock);
/* 临界区——不能睡眠 */
spin_unlock(&lock);

// 与中断交互的变体——中断处理程序也碰这个数据时必须关中断
spin_lock_irqsave(&lock, flags);
/* 临界区 */
spin_unlock_irqrestore(&lock, flags);
```

spin_lock 的变体选择规则是内核编程的必答题：数据被中断上下文访问 → `spin_lock_irqsave`（防本核中断打断）；只被软中断访问 → `spin_lock_bh`（关软中断）；只被进程上下文访问 → `spin_lock`（CONFIG_PREEMPT 下自动关抢占）。选错变体是死锁的常见来源——进程上下文持有普通 spinlock 时被自己的中断处理程序打断，后者 spin_lock 同一把锁永远等不到。

**互斥锁**（mutex）：获取失败的任务睡眠——让出 CPU。适用于**临界区较长**（可能睡眠、可能做 IO）且**只在进程上下文使用**。持有 mutex 期间可以睡眠（这是与 spinlock 的本质区别），但不能在中断上下文使用（中断不能睡眠）。

选型速查：中断上下文 → 只能 spinlock；临界区可能睡眠（分配内存、copy_to_user）→ 只能 mutex；临界区是纯内存操作且极短 → spinlock 优先（避免睡眠的调度开销）。

## RCU：读侧零开销的读写分离
RCU（Read-Copy-Update）是内核最有特色的并发原语——它实现了**读侧几乎零开销**的读写同步：读者不加锁、不写计数、不执行原子指令——只是普通指针解引用。代价转移到写者：写者复制-修改-发布新版本，并延迟释放旧版本（等所有既有读者离开）。

RCU 的读侧：

```c
rcu_read_lock();                    // 标记读临界区开始（仅禁止抢占，无其他开销）
struct foo *p = rcu_dereference(gp);  // 带依赖屏障的指针读取
if (p) do_something(p->field);      // 安全使用——p 在读临界区内不会被释放
rcu_read_unlock();                  // 结束
```

写侧（发布-删除模型）：

```c
struct foo *old, *new = kmalloc(...);
*new = *old; new->field = new_value;   // 复制并修改
rcu_assign_pointer(gp, new);           // 发布（带 release 语义）
 synchronize_rcu();                     // 等待所有既有读者离开（宽限期）
kfree(old);                            // 现在安全释放旧版本
```

`synchronize_rcu()` 等待的"宽限期"（grace period）是 RCU 的核心概念——所有在发布新版本**之前**开始且尚未结束的读临界区都完成。宽限期通常几毫秒到几十毫秒，写者也可以用 `call_rcu(&old->rcu, delayed_free)` 异步注册延迟释放。

RCU 的适用条件苛刻但常见：**读多写少**（配置表、路由表、设备列表——读每秒百万次、写每天几次）、**数据结构通过指针访问**、**读者不需要看到最新版本**（容忍旧数据短暂可见）。内核的网络路由表、dentry 缓存、模块列表都是 RCU 的经典用户。误用场景：写频繁（每次写都付出复制+宽限期延迟，写吞吐崩溃）、需要写后读一致（读者可能读到旧值——不适用余额这类强一致数据）。

## per-CPU 变量：消灭共享本身
per-CPU 变量为每个 CPU 核心维护一个独立副本——`this_cpu_inc(&counter)` 只增加当前核的副本，天然无共享、无竞争、无缓存行弹跳。读合计时跨核汇总（`for_each_possible_cpu` 累加）。

```c
DEFINE_PER_CPU(unsigned long, my_counter);

this_cpu_inc(my_counter);                    // 原子性由"禁止抢占期间不被迁移"保证
total = 0;
for_each_possible_cpu(cpu)
    total += per_cpu(my_counter, cpu);
```

适用场景：统计计数（性能计数器、事件计数——精度要求不高，汇总时读到的各核值本来就是不同时刻的）、每核的空闲缓存/缓冲（每核自己的 slab 缓存）。边界：访问**其他核**的副本需要显式同步（通常就不该访问）；`this_cpu_*` 操作要求抢占禁用（否则操作到一半被迁移到别的核——操作了错误的副本）；需要精确全局值时 per-CPU 汇总有时间差。

## seqlock：写优先的乐观读
seqlock（顺序锁）适合**写少读多且读者不能容忍 RCU 的旧值、但读远多于写**的数据。核心思想：写者持锁并递增序列号；读者乐观读取——读前后序列号一致且为偶数则读取有效，否则重试。

```c
// 写者
write_seqlock(&sl);
data = new_value;
write_sequnlock(&sl);        // 序列号 +1（奇数→偶数）

// 读者（乐观，无锁）
do {
    seq = read_seqbegin(&sl);   // 记录起始序列号
    val = data;                 // 普通读取
} while (read_seqretry(&sl, seq));  // 序列号变了（写者介入）→ 重读
```

seqlock 的特性：**读者永不阻塞写者**（写者不需要等读者——只管递增序列号）；写者之间互斥（内部有 spinlock）；读者可能重试（检测到写介入时）。适用场景：时间戳（`jiffies` 的读取、`gettimeofday` 的实现）、地理位置更新（GPS 坐标——读者要最新值但读频率极高）、内核的 `vgettimeofday` 快路径。边界：被保护的数据读取必须是原子的（单次 load 或对齐的结构体——多字段读取中间被打断靠序列号检测重试）；写者频繁时读者反复重试反而更慢。

## 选型决策表
| 数据特征 | 推荐原语 | 理由 |
|---------|---------|------|
| 简单计数/标志 | atomic/refcount | 开销最小，无锁语义够用 |
| 每核统计 | per-CPU | 消灭共享本身，零竞争 |
| 短临界区、中断上下文可用 | spinlock（含变体） | 忙等代价低于睡眠调度 |
| 长临界区、可能睡眠 | mutex | 睡眠让出 CPU |
| 读极多写极少、容忍旧值 | RCU | 读侧零开销 |
| 读多写少、需要最新值、读取原子 | seqlock | 读者不阻塞写者 |
| 复杂读写比例 | rwlock/rw_semaphore | 传统读写锁（多数场景被 RCU 取代） |

调试工具：lockdep（内核锁依赖检测器——运行时检测锁顺序死锁、上下文违规如"中断中拿 mutex"，`CONFIG_DEBUG_LOCKDEP`）是内核并发开发的第一工具；KCSAN（并发 sanitizer——数据竞争检测）捕获遗漏的同步。这两个工具在 QEMU 调试内核（见[QEMU 内核调试](/kernel/vm/qemu)）的环境中都值得开启——并发 bug 的复现依赖竞态窗口，工具化检测比等待偶发崩溃可靠得多。
