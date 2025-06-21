# 内核代码上下文
在 Linux 内核开发中，上下文指内核代码执行时的环境，包括 CPU 状态、可用资源和调度约束，理解不同的代码执行上下文（Context）是系统编程的基础。常见的内核上下文主要包括**进程上下文**和**中断上下文**。不同的上下文有不同的内核代码执行环境、可用资源以及编程时的注意事项。

## 进程上下文
进程上下文是指内核代码以某个进程身份运行的环境。通过系统调用处理函数（如 open、read、write 等）、内核线程（kthread）以及部分明确在进程上下文中执行的内核定时器回调等，进入进程上下文。

在进程上下文中，内核拥有该进程的全部资源和状态，可以主动让出 CPU（如调用 schedule()），也可以进行休眠和阻塞操作。此外，允许内核访问用户空间内存，例如通过 `copy_from_user/copy_to_user` 等接口与用户空间进行数据交换。在编程时需要注意，不要在持有自旋锁的情况下休眠或阻塞，并且要确保用户空间指针的合法性，防止非法访问。

- 运行环境：
  - 关联特定进程，当前进程信息存储在 `current` 宏（指向 `struct task_struct`）。
  - 使用进程的内核栈（每个进程有独立的内核栈，典型大小 8KB）。
- 资源访问：
  - 可访问用户空间内存（通过 `copy_to_user`、`copy_from_user`）。
  - 可调用阻塞函数（如 `schedule`、`sleep`），允许进程被调度。
- 调度：
  - 可被抢占（除非明确禁用抢占，如 `preempt_disable`）。
  - 支持长时间运行的任务。

```c
#include <linux/module.h>
#include <linux/kernel.h>

static int __init my_init(void) {
    printk(KERN_INFO "In process context, current PID: %d\n", current->pid);
    return 0;
}

static void __exit my_exit(void) {
    printk(KERN_INFO "Exiting module\n");
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
```

### 系统初始化上下文
系统初始化上下文是指系统在启动阶段（如内核启动、模块加载时）执行的代码环境。此时通常在单核、单线程环境下运行，不会被抢占，也不会响应中断。进入时机包括内核启动流程（如 start_kernel）和模块初始化函数（如 module_init 宏指定的函数）。在该上下文中可以执行阻塞操作，但要避免长时间阻塞影响启动进度。由于尚未进入多任务环境，部分同步机制（如信号量）不可用。此时，可以看作是早期的




## 中断上下文
中断上下文是指内核在处理中断（包括硬件中断、软中断、tasklet等）时的执行环境。进入中断上下文的典型时机包括硬件中断处理程序（IRQ handler）、软中断（softirq）、tasklet等。

在中断上下文下，内核不能阻塞或休眠，不能调用 schedule() 等可能导致阻塞的函数，也不能访问用户空间内存。中断上下文具有较高的优先级，通常会**打断进程上下文的执行**，从而**抢占 CPU**。编写中断处理代码时，必须禁止阻塞或休眠操作，尽量缩短中断处理时间，避免长时间占用 CPU，并且不要访问可能导致阻塞的资源（如等待信号量、互斥锁等）。

- 运行环境：
  - 不关联特定进程，`current` 宏可能不可靠（中断可能在任意进程中触发）。
  - 使用中断栈（每个 CPU 核有独立中断栈，较小，通常 4-8KB）。
- 资源访问：
  - **不可访问用户空间内存**，因为无进程上下文。
  - **不可阻塞或睡眠**，不能调用 `schedule` 或阻塞函数（如 `msleep`）。
  - 必须快速执行，避免阻塞其他中断。
- 调度：
  - 不可抢占，中断处理期间禁用调度。
  - 高优先级，可能嵌套（硬中断可被更高优先级中断打断）。

```c
#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/sched.h>

irqreturn_t my_irq_handler(int irq, void *dev_id) {
    printk(KERN_INFO "Interrupt context, CPU: %d\n", smp_processor_id());
    return IRQ_HANDLED;
}

static int __init my_init(void) {
    // 进程上下文：打印当前进程信息
    printk(KERN_INFO "Process context, PID: %d, Name: %s\n", 
           current->pid, current->comm);
    
    // 注册中断处理程序
    if (request_irq(10, my_irq_handler, IRQF_SHARED, "my_device", NULL)) {
        printk(KERN_ERR "Failed to request IRQ\n");
        return -1;
    }
    return 0;
}

static void __exit my_exit(void) {
    free_irq(10, NULL);
    printk(KERN_INFO "Exiting module\n");
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
```

### 软中断上下文
软中断和 tasklet 介于中断和进程上下文之间，不能阻塞，常用于中断下半部的处理。工作队列（Workqueue）则是在内核线程中运行，属于进程上下文，可以阻塞和休眠，适合处理需要较长时间的任务。

## 典型上下文示例
```c
#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/workqueue.h>
#include <linux/kthread.h>

struct work_struct my_work;
struct task_struct *my_thread;

// 硬中断处理程序
irqreturn_t my_irq_handler(int irq, void *dev_id) {
    printk(KERN_INFO "Hard interrupt context, IRQ: %d\n", irq);
    return IRQ_HANDLED;
}

// 任务let（软中断上下文）
void my_tasklet_handler(unsigned long data) {
    printk(KERN_INFO "Tasklet (softirq) context, CPU: %d\n", smp_processor_id());
}
DECLARE_TASKLET(my_tasklet, my_tasklet_handler, 0);

// 工作队列（进程上下文）
void my_work_handler(struct work_struct *work) {
    printk(KERN_INFO "Workqueue context, PID: %d\n", current->pid);
    tasklet_schedule(&my_tasklet); // 触发任务let
}

// 内核线程（进程上下文）
int my_thread_func(void *data) {
    while (!kthread_should_stop()) {
        printk(KERN_INFO "Kernel thread context, PID: %d\n", current->pid);
        msleep(1000);
    }
    return 0;
}

static int __init my_init(void) {
    // 进程上下文：初始化
    printk(KERN_INFO "Process context, PID: %d\n", current->pid);

    // 注册硬中断
    if (request_irq(10, my_irq_handler, IRQF_SHARED, "my_device", NULL)) {
        printk(KERN_ERR "Failed to request IRQ\n");
        return -1;
    }

    // 初始化工作队列
    INIT_WORK(&my_work, my_work_handler);
    schedule_work(&my_work);

    // 创建内核线程
    my_thread = kthread_run(my_thread_func, NULL, "my_kthread");
    if (IS_ERR(my_thread)) {
        free_irq(10, NULL);
        return PTR_ERR(my_thread);
    }

    return 0;
}

static void __exit my_exit(void) {
    free_irq(10, NULL);
    flush_scheduled_work();
    tasklet_kill(&my_tasklet);
    kthread_stop(my_thread);
    printk(KERN_INFO "Exiting module\n");
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
```

