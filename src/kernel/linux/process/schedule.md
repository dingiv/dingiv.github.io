# 调度

## 调度器概述
Linux调度器是操作系统的核心组件，负责决定哪个进程在何时运行，以及运行多长时间。Linux采用完全公平调度器(CFS)作为默认调度器，实现了对CPU资源的公平分配。

进程可以被阻塞，往往是因为进程执行了一个 IO 动作，而此时进程的任务必须等待 IO 设备完成读写动作之后，才能执行下一个动作，这意味着 CPU 必须等待缓慢的 IO 设备完成它们的工作。为了减少 CPU 等待 IO 设备的时间，操作系统会在进程使用了阻塞 IO 调用的时候将进程挂起，从而主动让出该进程对 CPU 的占用，此时，调度算法将会调度其他进程，让其他进程

调度器类，不同的进程可以使用不同的调度器


时间片长短
时间记账，让运行时间最少的任务先执行


nice 值

调用优先级——多个任务都处于等待的时候，谁先进行

抢占

## 多级反馈队列
Linux系统的调度算法采用的是多级反馈队列(MLFQ)，通过动态调整进程优先级来实现公平调度。

### 优先级分类
1. 实时进程优先级(0-99)
   - SCHED_FIFO：先进先出，直到主动让出CPU
   - SCHED_RR：时间片轮转，相同优先级进程轮流执行
   - SCHED_DEADLINE：基于截止时间的调度

2. 普通进程优先级(100-139)
   - 静态优先级(nice值)：-20到19
   - 动态优先级：根据进程行为动态调整

### 调度策略
1. 完全公平调度(CFS)
   - 基于虚拟运行时间
   - 红黑树组织就绪队列
   - 动态时间片分配

2. 实时调度
   - 优先级抢占
   - 时间片轮转
   - 截止时间保证

## 内核代码上下文

### 调度器数据结构
```c
struct sched_entity {
    struct load_weight load;
    struct rb_node run_node;
    struct list_head group_node;
    unsigned int on_rq;
    u64 exec_start;
    u64 sum_exec_runtime;
    u64 vruntime;
    u64 prev_sum_exec_runtime;
    u64 nr_migrations;
    struct sched_statistics statistics;
};

struct task_struct {
    // ... 其他字段 ...
    struct sched_entity se;
    struct sched_rt_entity rt;
    struct sched_dl_entity dl;
    // ... 其他字段 ...
};
```

### 调度器实现
1. 进程选择
```c
static struct task_struct *pick_next_task_fair(struct rq *rq)
{
    struct task_struct *p;
    struct cfs_rq *cfs_rq = &rq->cfs;
    struct sched_entity *se;

    if (!cfs_rq->nr_running)
        return NULL;

    se = pick_next_entity(cfs_rq);
    p = task_of(se);

    return p;
}
```

2. 时间片分配
```c
static void update_curr(struct cfs_rq *cfs_rq)
{
    struct sched_entity *curr = cfs_rq->curr;
    u64 now = rq_clock_task(rq_of(cfs_rq));
    u64 delta_exec;

    delta_exec = now - curr->exec_start;
    curr->exec_start = now;
    curr->sum_exec_runtime += delta_exec;
    curr->vruntime += calc_delta_fair(delta_exec, curr);
}
```

## 时钟滴答

## 调度时机

### 主动调度
1. 进程阻塞
   - 等待I/O
   - 等待信号量
   - 等待事件

2. 进程退出
   - 正常退出
   - 异常退出
   - 被信号终止

### 被动调度
1. 时钟中断
   - 时间片耗尽
   - 更新统计信息
   - 检查是否需要调度

2. 优先级抢占
   - 高优先级进程就绪
   - 实时进程抢占
   - 内核抢占

## 调度优化

### 负载均衡
1. 多核调度
   - 进程迁移
   - 负载均衡
   - CPU亲和性

2. 缓存优化
   - 缓存亲和性
   - NUMA感知
   - 内存访问优化

### 性能调优
1. 调度参数
   - 时间片长度
   - 调度延迟
   - 负载权重

2. 系统配置
   - CPU隔离
   - 实时进程配置
   - 调度组设置

## 调度器扩展

### 调度类
1. 完全公平调度类
   - 普通进程调度
   - 公平性保证
   - 动态优先级

2. 实时调度类
   - 实时进程调度
   - 优先级保证
   - 截止时间保证

3. 截止时间调度类
   - 基于截止时间
   - 资源预留
   - 服务质量保证

### 调度策略
1. 批处理调度
   - 吞吐量优化
   - 资源利用率
   - 批处理作业

2. 交互式调度
   - 响应时间优化
   - 用户交互
   - 前台进程

## 调度器调试

### 性能分析
1. 调度延迟
   - 测量方法
   - 影响因素
   - 优化方案

2. 吞吐量
   - 测试方法
   - 性能指标
   - 瓶颈分析

### 问题诊断
1. 调度问题
   - 进程饥饿
   - 优先级反转
   - 死锁检测

2. 性能问题
   - CPU使用率
   - 响应时间
   - 系统负载

## 实际应用

### 系统调优
1. 进程优先级
   - nice值设置
   - 实时优先级
   - 调度策略选择

2. 资源控制
   - CPU限制
   - 内存限制
   - IO限制

### 开发建议
1. 进程设计
   - 合理划分任务
   - 避免CPU密集
   - 考虑调度特性

2. 性能优化
   - 减少调度开销
   - 优化IO操作
   - 合理使用锁


### 阻塞和空转

从操作系统层面，当进程调用阻塞式系统调用时：
1. 进程进入阻塞状态
2. 操作系统不会把CPU的时间片分配给这个进程
3. 直到调用结束，进程从阻塞状态重新进入执行状态

常见的阻塞式调用包括：

1. 进程调度和中断
   - 进程本身的调度和中断机制通过阻塞方式实现
   - 这个过程对于进程自身来说是无感知的

2. 阻塞式IO调用
   ```c
   // 阻塞式函数
   read(), write(), sleep(), wait()
   ```

3. 锁和同步机制
   - 互斥锁（Mutex）
   - 信号量（Semaphore）
   ```c
   // 线程互斥锁
   pthread_mutex_t mutex;
   pthread_cond_t cond;

   pthread_mutex_init(&mutex, NULL);
   // 尝试获取互斥锁，如果无法获取则进入阻塞状态
   pthread_mutex_lock(&mutex);
   // 开始访问被保护的资源
   // ...
   // 释放锁
   pthread_mutex_unlock(&mutex);

   // 信号量
   sem_t sem;
   sem_init(&sem, 0, 0); // 初始化信号量为0
   sem_wait(&sem); // 等待信号量
   ```

与阻塞不同，空转指的是进程进入无意义的空循环状态，可能通过不断检查条件来等待某个条件的达成。这可能是程序bug或刻意设计。在这种状态下，进程处于正常执行状态但不执行有用逻辑，导致CPU浪费。

> 死锁时，锁住的两个或多个进程会被阻塞，然后被操作系统挂起，CPU占用为0，这是典型的死锁特征。
