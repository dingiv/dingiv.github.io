# 中断处理
当中断信号到达 CPU 后，CPU 自动完成硬件级的上下文保存（将当前指令指针、状态寄存器等压入内核栈），然后跳转到内核的中断处理函数。内核负责识别中断来源、调用对应的驱动处理函数、完成后续的清理和上下文恢复。由于中断处理会抢占当前正在执行的进程，且在中断上下文中不能休眠或阻塞，内核需要精心设计中断处理的执行模型。

## 中断处理流程

以 x86 平台的外部设备中断为例，完整的中断处理流程如下：

1. **设备触发中断**：外部设备（如网卡收到数据包）通过中断线或 MSI 向中断控制器发送中断信号。
2. **中断控制器路由**：I/O APIC 或 MSI 将中断路由到目标 CPU 的 Local APIC。
3. **CPU 响应中断**：Local APIC 通过 INTR 引脚通知 CPU，CPU 检查中断是否被允许（IF 标志位），如果允许则自动完成硬件上下文切换：将 SS、RSP、RFLAGS、CS、RIP 压入内核栈，通过 IDT 查找中断向量号对应的处理函数入口。
4. **内核入口**：内核的中断入口汇编代码执行，切换到该 CPU 核心的中断栈（x86_64 的 IST 机制），保存完整的寄存器上下文（`pt_regs` 结构），调用 `do_IRQ` 或 `generic_handle_irq`。
5. **中断分发**：内核通过 Linux 虚拟中断号查找对应的 `irq_desc`，调用该中断的 **action 链**（处理函数链表）。
6. **驱动处理函数执行**：驱动的 `handler` 回调被调用，执行设备特定的中断处理逻辑（如读取设备状态、拷贝数据、确认中断）。
7. **上下半部调度**：如果驱动注册了底半部机制（softirq、tasklet 或 workqueue），顶半部在返回前将底半部任务标记为待执行。
8. **中断退出**：内核恢复寄存器上下文，向中断控制器发送 EOI（End of Interrupt），通过 `iret` 指令返回被中断的进程上下文。

整个顶半部的执行时间是关键性能指标。在此期间，当前 CPU 核心上所有同级和低级的中断都被屏蔽（x86 上 IF 位被自动清除），如果有另一个设备等待中断响应，它必须等到当前中断处理完成。因此顶半部必须尽可能短。

## 注册中断处理函数

驱动通过 `request_irq` 注册中断处理函数，通过 `free_irq` 注销。这是驱动开发中最基本的中断操作。

```c
#include <linux/interrupt.h>

// 中断处理函数原型
irqreturn_t my_handler(int irq, void *dev_id) {
    // dev_id 是注册时传入的私有数据指针
    struct my_device *dev = (struct my_device *)dev_id;

    // 1. 读取设备状态寄存器，确认中断来源
    u32 status = readl(dev->mmio + STATUS_REG);
    if (!(status & IRQ_PENDING))
        return IRQ_NONE;  // 不是本设备的中断（共享中断场景）

    // 2. 清除设备的中断标志（防止重复触发）
    writel(status, dev->mmio + STATUS_REG);

    // 3. 快速处理：拷贝数据到缓冲区，调度底半部处理
    schedule_work(&dev->bh_work);

    return IRQ_HANDLED;  // 中断已处理
}

// 注册中断
int ret = request_irq(irq_num, my_handler, IRQF_SHARED,
                      "my_device", &my_dev);
// 参数：中断号、处理函数、标志位、名称、私有数据

// 注销中断
free_irq(irq_num, &my_dev);
```

`request_irq` 的第三个参数是标志位，常用的包括：`IRQF_SHARED` 表示多个设备共享同一个中断线（此时 `dev_id` 必须唯一，内核通过遍历 action 链调用所有 handler）；`IRQF_ONESHOT` 表示在中断处理完成后才解除屏蔽，用于电平触发的共享中断防止中断风暴；`IRQF_NO_SUSPEND` 表示在系统休眠时不禁用该中断。

中断处理函数的返回值是 `irqreturn_t`，返回 `IRQ_HANDLED` 表示中断已处理，返回 `IRQ_NONE` 表示不是本设备的中断（仅在共享中断时有效，内核会据此判断是否需要处理下一个 handler）。

## 软中断（Softirq）

软中断是内核定义的一组静态编译的中断处理底半部机制，用于在进程上下文之外异步执行可延迟的工作。软中断在中断上下文中执行（不能休眠），但执行时中断是开启的（可以被新的硬件中断抢占）。

内核定义了若干种软中断类型（`enum softirq_bh`），包括：

- `HI_SOFTIRQ`：高优先级软中断，用于 tasklet 调度。
- `TIMER_SOFTIRQ`：定时器软中断，用于内核定时器的到期处理。
- `NET_TX_SOFTIRQ` / `NET_RX_SOFTIRQ`：网络发送和接收软中断，处理网络协议栈的数据包收发。这是软中断最典型的应用场景，网络设备驱动在顶半部将数据包放入队列，通过 `NET_RX_SOFTIRQ` 在底半部执行协议栈处理。
- `TASKLET_SOFTIRQ` / `HI_SOFTIRQ`：tasklet 调度用。
- `RCU_SOFTIRQ`：RCU 读侧宽限期结束的回调处理。

软中断的执行时机有两个：一是在硬件中断处理的退出路径中（`do_IRQ` 返回前检查是否有待处理的软中断），二是在 `ksoftirqd` 内核线程中（当软中断积累过多时唤醒该线程处理，避免在中断退出路径中执行过多软中断影响响应性）。`ksoftirqd` 是 per-CPU 的内核线程，每个 CPU 核心一个。

软中断的一个关键特性是**同一类型的软中断可以在多个 CPU 上并发执行**。例如 `NET_RX_SOFTIRQ` 在 CPU 0 和 CPU 1 上可以同时运行。这意味着使用软中断的代码必须自行处理并发安全问题（通常通过 per-CPU 数据结构或自旋锁）。

驱动开发者通常不直接使用软中断，而是通过更高层的抽象（tasklet 或 workqueue）来实现底半部。但理解软中断有助于理解 tasklet 的底层实现和网络的收发路径。

## Tasklet

Tasklet 是基于软中断实现的底半部机制，提供了更简单的接口和更好的并发保证。tasklet 本质上是在 `HI_SOFTIRQ` 和 `TASKLET_SOFTIRQ` 这两种软中断之上的封装。

```c
#include <linux/interrupt.h>

// 定义 tasklet
void my_tasklet_func(unsigned long data) {
    struct my_device *dev = (struct my_device *)data;
    // 处理耗时操作
    process_data(dev);
}

DECLARE_TASKLET(my_tasklet, my_tasklet_func, (unsigned long)&my_dev);

// 在顶半部中调度 tasklet
tasklet_schedule(&my_tasklet);

// 注销时取消未执行的 tasklet
tasklet_kill(&my_tasklet);
```

Tasklet 与原始软中断的关键区别在于**串行化保证**：同一个 tasklet 在同一时刻只会在一个 CPU 上执行，不会并发。这意味着 tasklet 的处理函数不需要加锁（前提是不访问其他 tasklet 共享的数据）。但不同的 tasklet 之间可以在不同 CPU 上并发执行。

Tasklet 的内部实现：`tasklet_schedule` 将 tasklet 挂载到当前 CPU 的 tasklet 链表上，然后触发 `TASKLET_SOFTIRQ`。在软中断执行时，内核遍历当前 CPU 的 tasklet 链表，依次执行每个 tasklet 的函数。如果 tasklet 在执行期间被重新调度（函数内再次调用 `tasklet_schedule`），它会在下一轮软中断中重新执行。

Tasklet 的限制是不能休眠（因为它运行在软中断上下文中），且处理函数应尽量短。如果底半部需要执行耗时操作或可能休眠，应使用 workqueue。

## Workqueue

Workqueue（工作队列）是另一种底半部机制，与前两者不同，工作队列的处理函数运行在**进程上下文**（内核线程中），可以休眠、可以持有信号量、可以访问用户空间。这是最灵活的底半部方式。

```c
#include <linux/workqueue.h>

// 定义工作项和工作队列
struct work_struct my_work;

void my_work_handler(struct work_struct *work) {
    // 在进程上下文中执行，可以休眠、持有锁、访问用户空间
    struct my_device *dev = container_of(work, struct my_device, my_work);
    msleep(100);  // 可以休眠
    mutex_lock(&dev->mutex);  // 可以使用互斥锁
    process_data(dev);
    mutex_unlock(&dev->mutex);
}

// 初始化工作项
INIT_WORK(&my_work, my_work_handler);

// 在顶半部中调度工作
schedule_work(&my_work);

// 取消工作
cancel_work_sync(&my_work);
```

内核维护了一个全局的工作队列（`system_wq`，对应 per-CPU 的 `kworker` 线程），通过 `schedule_work` 即可将工作项提交到全局队列。驱动也可以创建专用的工作队列（`create_singlethread_workqueue` 或 `alloc_workqueue`），拥有独立的内核线程，适用于需要严格串行化或优先级控制的场景。

Workqueue 与 tasklet 的选择取决于底半部是否需要休眠。如果处理逻辑涉及可能阻塞的操作（如等待信号量、拷贝用户空间数据、进行磁盘 I/O），必须使用 workqueue。如果只是简单的数据处理和状态更新，tasklet 的开销更小（无需上下文切换到内核线程）。

## 三种底半部机制对比

| 特性 | Softirq | Tasklet | Workqueue |
|------|---------|---------|-----------|
| 执行上下文 | 中断上下文（不可休眠） | 中断上下文（不可休眠） | 进程上下文（可以休眠） |
| 并发性 | 同类型可在多 CPU 并发 | 同一 tasklet 串行，不同 tasklet 可并发 | 由工作队列类型决定 |
| 锁要求 | 需要 per-CPU 数据或自旋锁 | 同一 tasklet 无需锁 | 可使用互斥锁、信号量 |
| 适用场景 | 网络收发（NET_TX/RX）等高性能场景 | 设备驱动通用底半部 | 需要休眠的耗时操作 |
| 使用方式 | 驱动一般不直接使用 | `DECLARE_TASKLET` + `tasklet_schedule` | `INIT_WORK` + `schedule_work` |
| 调度延迟 | 最低（中断退出路径直接执行） | 低 | 较高（需要调度内核线程） |

在实际的驱动开发中，大多数场景使用 tasklet 或 workqueue。软中断主要被内核子系统自身使用（网络栈、定时器、RCU），驱动开发者很少直接操作软中断，除非有极高的性能要求且能正确处理多 CPU 并发。

## 中断统计与调试

内核提供了丰富的中断统计和调试手段。`/proc/interrupts` 显示每个中断号的触发次数和分发到的 CPU 核心，是观察中断负载分布的首要工具。

```
# 查看 /proc/interrupts 的典型输出
           CPU0       CPU1       CPU2       CPU3
  0:         50          0          0          0   IO-APIC   2-edge      timer
 24:      12580       8234       6102       4856   PCI-MSI 524288-edge   eth0-TxRx-0
 25:       9200       5100       7800       6300   PCI-MSI 524289-edge   eth0-TxRx-1
```

`/proc/softirqs` 显示各类软中断在每个 CPU 上的触发次数，用于分析软中断的负载均衡情况。如果某个 CPU 上的 `NET_RX` 远高于其他 CPU，说明网络收发存在倾斜，可能需要调整 RPS（Receive Packet Steering）配置。

`/proc/irq/N/smp_affinity` 以十六进制位掩码形式显示和设置中断亲和性。例如 `echo 3 > /proc/irq/24/smp_affinity` 将中断 24 限制在 CPU 0 和 CPU 1 上处理。`/proc/irq/N/smp_affinity_list` 以 CPU 列表格式提供更直观的配置方式。

内核配置选项 `CONFIG_GENERIC_IRQ_DEBUGFS` 启用后，`/sys/kernel/debug/irq/` 下会提供更详细的中断调试信息，包括每个中断的域信息、芯片信息和时序统计。
