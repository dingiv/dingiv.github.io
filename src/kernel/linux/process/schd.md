# 调度
调度系统是进程管理的核心组件，负责决定哪个进程在何时运行，以及运行多长时间。

调度的目标有
- 高效性，调度算法高效快速，开销小
- 实时性，对于一些时间敏感的任务，需要能够在要求的时间内完成
- 公平性，不同的任务都能够获得调度，并且得到应有的执行时间

在不同的使用场景下，调度的目标优先级会有所不同，但是主要矛盾是**实时性**和**公平性**之间的矛盾，因为重要的进程它需要特权和优先，但是公平要求的是每个进程都能够得到充分的执行，我们只能同时满足**高效和实时**或者**高效和公平**。

## 任务
任务是调度的基本单位，本质上是进程或线程（在 Linux 内核中，线程和进程本质相同，都是 `task_struct`。线程在 linux 中本质上是共享同一组资源的进程，从内核层面上，线程和进程的并没有什么太大的不同。这一点和其他系统例如 Windows 有显著的不同。这是一种巧妙的实现思路，使得进程的系统调用 API 相较于 Windows 等系统显得非常的简洁，同时在性能角度考虑也更加的轻量。

## 调度器
调度器是决定哪个任务获得 CPU 的组件。Linux 内核支持多种调度器，每种调度器针对不同的应用场景设计，使用不同的调度策略和算法。Linux 内核中的调度器按调度类（Scheduling Class）组织，主要包括：

1. CFS 调度器：用于普通进程，追求公平性，基于虚拟运行时间，使用红黑树组织就绪队列
2. 实时调度器：用于实时进程（SCHED_FIFO/SCHED_RR），基于优先级抢占，保证实时性要求
3. 截止时间调度器：用于 SCHED_DEADLINE 策略，基于任务截止时间调度，适合硬实时应用
4. 空闲调度器：当系统无其他可运行任务时使用，执行 CPU 空闲循环

### CFS 调度器
CFS（完全公平调度器）是 Linux 默认的普通进程调度器，其核心思想是通过时间记账，让运行时间最少的任务先执行，从而让每个进程获得公平的 CPU 时间。

核心概念：
- 虚拟运行时间（vruntime）：记录每个进程已经运行的虚拟运行时间
- 调度延迟（sched_latency）：所有可运行进程应该在一个调度延迟内至少运行一次
- 最小粒度（min_granularity）：进程的最小运行时间片

```c
struct sched_entity {
    struct load_weight load;        // 进程权重
    struct rb_node run_node;        // 红黑树节点
    u64 vruntime;                   // 虚拟运行时间
    u64 sum_exec_runtime;           // 实际运行时间
    // ... 其他字段
};
```

调度算法：
1. 选择虚拟运行时间 vruntime 最小的进程运行
2. 进程运行时，其虚拟运行时间按权重递增
3. 当进程的虚拟运行时间超过其他进程时，发生调度

权重计算时使用 nice 来计算权重，用户态设置的进程优先级指标，范围从 -20（最高优先级）到 19（最低优先级），默认值为 0。nice 值通过影响进程的调度权重（weight），决定 CPU 时间分配比例。nice 值每降低 1，权重增加约 10%，高权重进程获得更多 CPU 时间。

### 实时调度器
实时调度器用于满足实时性要求的进程，支持两种调度策略：

SCHED_FIFO（先进先出）：
- 高优先级进程可以抢占低优先级进程
- 同优先级进程按 FIFO 顺序运行
- 进程会一直运行直到主动让出 CPU 或被更高优先级进程抢占

SCHED_RR（轮转调度）：
- 类似 SCHED_FIFO，但同优先级进程按时间片轮转
- 每个进程运行一个时间片后被放到队列末尾
- 时间片长度可配置

```c
struct rt_rq {
    struct rt_prio_array active;    // 活跃优先级数组
    struct rt_prio_array expired;   // 过期优先级数组
    int rt_nr_running;              // 运行中的实时进程数
};
```

### 截止时间调度器
截止时间调度器基于任务的截止时间进行调度，适合硬实时应用。

核心概念：
- 运行时间（runtime）：任务需要的 CPU 时间
- 截止时间（deadline）：任务必须完成的时间点
- 周期（period）：任务的执行周期

调度算法：
1. 选择截止时间最早的任务运行
2. 任务运行时消耗其运行时间配额
3. 当运行时间用完时，任务被挂起直到下一个周期

```c
struct sched_dl_entity {
    struct rb_node rb_node;         // 红黑树节点
    u64 dl_runtime;                 // 运行时间
    u64 dl_deadline;                // 截止时间
    u64 dl_period;                  // 周期
    u64 dl_bw;                      // 带宽
};
```

### 调度器选择
Linux 使用调度类（Scheduling Class）来组织不同的调度器，每个调度类都有不同的优先级：

```c
// 调度类优先级（数字越小优先级越高）
#define SCHED_CLASS_DEADLINE    0   // 截止时间调度器
#define SCHED_CLASS_RT          1   // 实时调度器
#define SCHED_CLASS_FAIR        2   // CFS调度器
#define SCHED_CLASS_IDLE        3   // 空闲调度器
```

调度选择流程：
1. 从最高优先级的调度类开始检查
2. 如果该调度类有可运行任务，选择该调度类的任务
3. 否则检查下一个优先级的调度类
4. 重复直到找到可运行任务

## 调度流程
调度流程是调度器工作的核心机制，包括调度时机、调度决策、上下文切换等关键步骤。

### 调度时机
调度可能发生在以下时机：
+ 主动调度/进程主动让出 CPU
  - 进程调用 `sched_yield()` 主动让出 CPU
  - 进程进入阻塞状态（等待 I/O、信号量、事件等）
  - 进程退出（正常、异常、信号终止）
  ```c
  // 主动让出 CPU
  #include <sched.h>
  int sched_yield(void);

  // 进程阻塞示例
  int fd = open("file.txt", O_RDONLY);
  char buffer[1024];
  // 如果文件没有数据可读，进程会阻塞
  ssize_t bytes_read = read(fd, buffer, sizeof(buffer));
  ```

+ 被动调度/时钟中断触发
  - 时间片耗尽（CFS 中为虚拟运行时间超过阈值）
  - 统计信息更新
  - 调度检查

+ 抢占调度/优先级抢占
  - 高优先级进程变为可运行状态
  - 实时进程抢占普通进程
  - 内核抢占（如果启用，需要内核编译时启用 `CONFIG_PREEMPT`）

  ```c
  // 设置进程优先级
  #include <sched.h>
  struct sched_param param;
  param.sched_priority = 50;  // 实时优先级
  sched_setscheduler(0, SCHED_FIFO, &param);
  ```

### 调度决策过程
总体先选调度器，然后让调度器选进程，然后分时间片大小。

1. 调度器选择  
  Linux 使用调度类优先级来选择调度器：

  ```c
  // 调度类优先级（数字越小优先级越高）
  #define SCHED_CLASS_DEADLINE    0   // 截止时间调度器
  #define SCHED_CLASS_RT          1   // 实时调度器
  #define SCHED_CLASS_FAIR        2   // CFS调度器
  #define SCHED_CLASS_IDLE        3   // 空闲调度器
  ```

  1. 从最高优先级的调度类开始检查
  2. 如果该调度类有可运行任务，选择该调度类的任务
  3. 否则检查下一个优先级的调度类
  4. 重复直到找到可运行任务

2. 进程选择  
  进入不同的调度器之后，不同的调度器使用自身的策略来选择要运行的进程
  - CFS 进程选择
  ```c
  // CFS 选择虚拟运行时间最小的进程
  struct task_struct *pick_next_task_fair(struct rq *rq)
  {
      struct sched_entity *se;
      struct rb_node *left;
      
      // 从红黑树最左节点选择（vruntime 最小）
      left = rb_first_cached(&rq->cfs.tasks_timeline);
      se = rb_entry(left, struct sched_entity, run_node);
      
      return task_of(se);
  }
  ```

  - 实时进程选择
  ```c
  // 实时调度器选择最高优先级进程
  struct task_struct *pick_next_task_rt(struct rq *rq)
  {
      struct rt_rq *rt_rq = &rq->rt;
      struct rt_prio_array *array;
      
      // 从活跃数组中查找最高优先级进程
      array = &rt_rq->active;
      return array->queue[array->highest_prio];
  }
  ```

3. 时间片分配   
  CFS 时间片计算：
  ```c
  // 更新当前进程的运行时间
  void update_curr_fair(struct rq *rq)
  {
      struct sched_entity *curr = &rq->curr->se;
      u64 now = rq_clock_task(rq);
      u64 delta_exec = now - curr->exec_start;
      
      // 更新实际运行时间
      curr->sum_exec_runtime += delta_exec;
      
      // 更新虚拟运行时间（考虑权重）
      curr->vruntime += calc_delta_fair(delta_exec, curr);
      
      curr->exec_start = now;
  }
  ```

4. 上下文切换  
  在选择好调度的参数的参数，正式开始进行调度的动作，主要包括**保存当前进程状态**和**恢复目标进程状态**。
  1. 保存当前进程的执行状态，包括通用寄存器、浮点寄存器、程序计数器（PC）、栈指针（SP）以及页表基地址等关键信息到进程控制块 task_struct 中，确保下次调度时能够准确恢复进程的执行环境。
  2. 恢复目标进程的执行状态，将目标进程控制块中保存的寄存器状态加载到 CPU 中，切换页表以访问目标进程的内存空间，并恢复执行上下文，使目标进程能够从上次中断的地方继续执行。

  ```c
  // 简化的上下文切换伪代码
  void context_switch(struct task_struct *prev, struct task_struct *next)
  {
      struct mm_struct *mm, *oldmm;
      
      // 切换内存管理上下文
      mm = next->mm;
      oldmm = prev->active_mm;
      
      if (mm != oldmm) {
          switch_mm(oldmm, mm, next);
      }
      
      // 切换寄存器上下文
      switch_to(prev, next, prev);
  }
  ```

  上下文切换开销主要包括寄存器保存/恢复（~100-200个时钟周期）、页表切换（~1000-2000个时钟周期）、缓存失效和 TLB 失效，可通过延迟 TLB 失效、缓存亲和性优化和进程迁移限制等策略进行优化。

## 调度优化策略
Linux 内核采用一些常见的优化手段来确保调度的优化。

### 负载均衡
多核负载均衡通过进程迁移机制实现，当检测到某个 CPU 负载过重时，调度器会将进程从负载重的 CPU 迁移到负载较轻的 CPU 上。负载计算主要基于运行队列长度和进程权重进行，同时设置迁移阈值来避免频繁迁移造成的性能开销。

```c
// 负载均衡触发条件
static inline bool should_we_balance(struct lb_env *env)
{
    struct sched_domain *sd = env->sd;
    
    // 检查负载差异是否超过阈值
    return env->imbalance > sd->imbalance_pct;
}
```

多核负载需要考虑 NUMA 架构的影响，通过 NUMA 感知，调度通过优先在本地 NUMA 节点进行进程调度，同时考虑内存访问延迟的影响，并尽量避免跨节点的进程迁移，从而减少内存访问开销并提高系统整体性能。

### 缓存优化
缓存亲和性是指进程优先在之前运行的 CPU 上执行，通过减少缓存失效来提高内存访问效率的优化策略。

CPU 亲和性设置：
```c
#include <sched.h>

// 设置进程的 CPU 亲和性
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(0, &cpuset);  // 绑定到 CPU 0
sched_setaffinity(0, sizeof(cpuset), &cpuset);
```

### 实时性优化
实时进程配置：
```c
// 设置实时调度策略
struct sched_param param;
param.sched_priority = 99;  // 最高实时优先级
sched_setscheduler(0, SCHED_FIFO, &param);

// 设置 CPU 隔离（避免其他进程干扰）
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(1, &cpuset);  // 绑定到隔离的 CPU
sched_setaffinity(0, sizeof(cpuset), &cpuset);
```

## 调度问题诊断
在高度性能敏感的开发中，需要调优系统调度器，从而优化调度性能。

### 常见问题
1. 进程饥饿是指低优先级进程长时间无法获得 CPU 执行时间的问题。  
   当系统中存在大量高优先级进程时，低优先级进程可能会被无限期延迟，导致其任务无法及时完成。为了解决这个问题，调度器通常会采用动态优先级调整机制，随着等待时间的增加逐步提升进程优先级，或者通过时间片补偿的方式确保所有进程都能获得基本的执行机会。
2. 优先级反转是另一个常见的调度问题，发生在低优先级进程持有高优先级进程所需资源的情况下。   
   当高优先级进程等待低优先级进程释放资源时，中等优先级的进程可能会抢占 CPU，导致高优先级进程被进一步延迟。为了解决这个问题，系统通常采用优先级继承机制，让持有资源的低优先级进程临时继承等待进程的优先级，或者使用优先级天花板技术，预先设置资源访问的最高优先级。
3. 调度延迟过高是指进程从就绪状态到实际开始运行的时间间隔过长，这通常是由于系统负载过高或调度器配置不当导致的。  
   当系统中运行进程过多时，调度器需要更多时间来做出调度决策，同时频繁的上下文切换也会增加调度开销。为了诊断和解决这个问题，系统管理员需要监控调度延迟指标，优化调度器参数配置，并在必要时进行负载均衡或资源扩容。

### 调度统计与监控
内核在调度过程中会执行调度数据统计，开发者可以通过系统接口访问调度数据。
```c
struct sched_statistics {
    u64 wait_start;           // 等待开始时间
    u64 wait_max;             // 最大等待时间
    u64 wait_count;           // 等待次数
    u64 wait_sum;             // 等待时间总和
    u64 iowait_count;         // IO 等待次数
    u64 iowait_sum;           // IO 等待时间总和
    u64 sleep_start;          // 睡眠开始时间
    u64 sleep_max;            // 最大睡眠时间
    u64 sum_sleep_runtime;    // 睡眠时间总和
    u64 block_start;          // 阻塞开始时间
    u64 block_max;            // 最大阻塞时间
    u64 exec_max;             // 最大执行时间
    u64 slice_max;            // 最大时间片
    u64 nr_migrations_cold;   // 冷迁移次数
    // ...
};
```

### 诊断工具
内核态监控接口：
```c
// 获取进程调度统计
struct sched_statistics *stats = &task->se.statistics;

// 获取运行队列统计
struct rq *rq = cpu_rq(cpu);
u64 nr_running = rq->nr_running;
u64 load = rq->load.weight;
```

用户态监控数据接口：
```bash
# 查看进程调度信息
cat /proc/[pid]/schedstat

# 查看调度器统计
cat /proc/schedstat
cat /proc/schedstat | grep "cpu"

# 查看 CPU 调度信息
cat /proc/sched_debug
# 监控调度延迟
cat /proc/sched_debug | grep "avg_delay"
 
# 使用 ftrace 跟踪调度事件
echo function_graph > /sys/kernel/debug/tracing/current_tracer
echo sched > /sys/kernel/debug/tracing/trace_options

# 使用 perf 分析调度性能
perf record -e sched:sched_switch -g -p [pid]
perf report
```
