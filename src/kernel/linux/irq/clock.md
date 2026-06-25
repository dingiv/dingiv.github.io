# 时钟中断
时钟中断是操作系统的心跳信号，由定时器硬件周期性触发，内核在每个时钟中断中执行时间维护和任务调度等基础工作。时钟中断的频率直接决定了系统的计时精度和调度粒度。

## 硬件定时器
内核依赖硬件定时器产生周期性的时钟中断。不同平台上使用的定时器硬件不同。

在 x86 平台上，历史上有三种硬件定时器：**PIT（Programmable Interval Timer，8254）** 是最早的 PC 定时器，精度约 1μs，通过 I/O 端口 0x40-0x43 访问，目前已基本淘汰；**HPET** 是高精度事件定时器，精度可达 100ns 甚至 10ns，通过 MMIO 访问，支持多个独立的比较器通道；**LAPIC Timer** 是集成在每个 CPU 核心的 Local APIC 内部的定时器，每个核心独立运行，精度高且不需要经过总线访问，是现代 x86 系统首选的 per-CPU 时钟源。

在 ARM 平台上，GIC 的分发器可以管理全局定时器（Global Timer），此外每个 CPU 核心还有私有定时器（Private Timer）和看门狗定时器。Cortex-A 系列的通用定时器（Generic Timer）通过系统寄存器访问（`CNTV_CTL_EL0`、`CNTV_TVAL_EL0`），提供 per-CPU 的虚拟定时器和物理定时器，内核使用虚拟定时器作为 per-CPU 的 tick 源。

## jiffies 与 tick
内核维护一个全局变量 `jiffies`，每个时钟中断（tick）加一。`jiffies` 是内核中最基础的时间计量单位，内核中大量超时判断、时间计算都基于 jiffies。`CONFIG_HZ` 配置项定义了每秒的 tick 次数，常见值有 100（10ms 精度）、250（4ms 精度）和 1000（1ms 精度）。服务器平台通常使用 1000Hz 以获得更好的调度响应性。

时钟中断的处理函数 `tick_handle_periodic` 主要完成以下工作：更新 `jiffies`；更新系统的墙上时间（wall time，基于 `xtime` 和 `wall_to_monotonic`）；检查是否有到期的内核定时器（`timer wheel`）；调用调度器（`scheduler_tick`）更新进程的时间片和统计信息；处理 RCU 相关的宽限期检查。

传统模式下，无论 CPU 是否有空闲任务，时钟中断都以固定频率触发。这对于需要精细计时的场景是必要的，但在 CPU 空闲时，周期性的时钟中断会频繁唤醒 CPU，阻止其进入深度休眠状态，浪费功耗。

## NO_HZ 与 tickless
`CONFIG_NO_HZ_IDLE`（也称 tickless idle）模式下，当 CPU 进入空闲状态（没有可运行的进程）时，内核停止该 CPU 上的周期性时钟中断，改为设置一个一次性定时器，在下一个最近的到期事件（最近的定时器到期或调度需求）时唤醒 CPU。这样 CPU 可以在空闲期间持续停留在低功耗状态，而不是被无意义的时钟中断反复唤醒。

`CONFIG_NO_HZ_FULL`（full tickless）更进一步，即使 CPU 在执行任务时也取消周期性 tick，改为按需触发。这适用于运行一个或少量 CPU 密集型任务的场景（如高性能网络转发、HPC 计算），消除了时钟中断带来的上下文切换和缓存抖动开销。但 full tickless 模式下，基于 jiffies 的时间统计会不准确，需要依赖高精度定时器补偿。

## 高精度定时器（hrtimer）
内核的定时器框架分为两层：基于 jiffies 的**低精度定时器**（`timer_list`，精度为 1/HZ 秒）和基于硬件时钟源的**高精度定时器**（`hrtimer`，精度可达纳秒级）。现代内核中 hrtimer 是主要的定时器实现，低精度定时器在底层也通过 hrtimer 模拟。

hrtimer 的核心是一个按到期时间排序的红黑树（per-CPU），每个 CPU 维护独立的红黑树。添加定时器时将其插入对应 CPU 的红黑树，硬件定时器被编程为在最近的到期时间触发中断。中断到达时，内核从红黑树中取出所有已到期的定时器并执行其回调函数，然后将硬件定时器编程为下一个到期时间。

hrtimer 的时钟源是 `clocksource`（如 TSC、ARM Generic Timer），精度取决于硬件时钟源的能力。在 x86 上，TSC 是最优的时钟源；在 ARM 上，Generic Timer 是首选。`/sys/devices/system/clocksource/clocksource0/` 可以查看和切换时钟源。

```c
// hrtimer 的典型使用方式
#include <linux/hrtimer.h>

enum hrtimer_restart my_timer_callback(struct hrtimer *timer) {
    struct my_device *dev = container_of(timer, struct my_device, timer);
    // 定时器到期时的处理逻辑
    handle_timeout(dev);
    // 返回 HRTIMER_NORESTART 表示一次性定时器
    // 返回 HRTIMER_RESTART 表示周期性定时器（需要重新设置到期时间）
    return HRTIMER_NORESTART;
}

// 初始化
hrtimer_init(&dev->timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
dev->timer.function = my_timer_callback;

// 启动定时器，500ms 后到期
hrtimer_start(&dev->timer, ms_to_ktime(500), HRTIMER_MODE_REL);

// 取消定时器
hrtimer_cancel(&dev->timer);
```

hrtimer 的回调函数运行在硬中断上下文（hrtimer 中断），因此不能休眠。如果定时器回调需要执行耗时操作，应结合 workqueue 使用。

## 与调度器的关系
时钟中断是调度器运行的驱动力量。每次 tick 中断都会调用 `scheduler_tick`，执行以下工作：更新当前进程的运行时间统计（`vruntime`）；检查当前进程的时间片是否用尽，如果用尽则设置 `TIF_NEED_RESCHED` 标志，在时钟中断返回用户态时触发调度。

在 NO_HZ 模式下，时钟中断不再周期性触发，调度器的时间片到期检测依赖于高精度定时器。内核为每个进程设置一个调度到期定时器（sched timer），当进程的时间片用完时该定时器触发，产生一次高精度定时器中断，在中断中完成调度切换。这意味着即使没有周期性 tick，调度器仍然能正确地剥夺超时进程的 CPU。

从驱动开发的角度来看，时钟中断通常是不可见的——驱动使用定时器 API（hrtimer 或 timer_list）来设置超时，内核负责在合适的硬件定时器上编程。只有在内核调试或性能分析时，才需要关注时钟中断的频率、NO_HZ 模式和时钟源选择。
