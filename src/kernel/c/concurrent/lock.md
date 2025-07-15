# 同步原语

在 Linux C 语言编程中，**同步原语（synchronization primitives）** 是用于协调多个线程或进程对共享资源的访问，防止数据竞争和状态不一致的关键工具。同步原语分为 **线程级（用户空间）** 和 **进程级（可能涉及内核）**，常用的有：

## 线程同步原语（基于 `pthread`）

这些原语定义在 `<pthread.h>` 中，仅在线程之间共享内存的环境下使用。

### 互斥锁
作用：实现互斥访问，**一次只允许一个线程进入临界区**。
常用函数：

```c
pthread_mutex_init()
pthread_mutex_lock()
pthread_mutex_unlock()
pthread_mutex_destroy()
```

### 自旋锁
- 比互斥锁更轻量，在锁短时间持有时效率更高。
- **线程会忙等（不断循环）直到获得锁**，适合嵌入式或无调度环境。
- 函数：

```c
pthread_spin_init()
pthread_spin_lock()
pthread_spin_unlock()
pthread_spin_destroy()
```

### 读写锁
- 允许多个线程同时读，但写操作必须独占。
- 用于读多写少的场景。
- 函数：

```c
pthread_rwlock_init()
pthread_rwlock_rdlock()
pthread_rwlock_wrlock()
pthread_rwlock_unlock()
```

### 条件变量
- 用于线程之间的**等待/通知**机制，配合互斥锁使用。
- 类似于“事件队列”中的 wait/signal。
- 函数：

```c
pthread_cond_wait()
pthread_cond_signal()
pthread_cond_broadcast()
```

### 屏障
- 使一组线程在某个点**同步并等待**，直到所有线程都到达屏障。
- 用于阶段性分批执行的线程协作。

## 原子操作原语
定义在 `<stdatomic.h>` 或 GCC 内建函数中，**无需加锁，使用 CPU 指令实现并发安全**。

`atomic_int`、`atomic_bool`、`atomic_flag` 等

```c
atomic_int count = 0;
atomic_fetch_add(&count, 1);
```

用于无锁队列、自旋锁等高性能场景。


## 进程间同步原语
用于**多个进程之间同步/通信**，多数涉及内核支持：

### 信号量
- 头文件：`<semaphore.h>`
- 区分为**线程间同步**（无名信号量）与**进程间同步**（命名信号量或放在共享内存中）
- 函数：

```c
sem_init()
sem_wait()
sem_post()
sem_destroy()
```

### futex（Fast Userspace Mutex Linux 特有）
- 系统调用级别的原语，允许**用户态等待锁竞争失败时才进入内核态**，用于构建高性能锁。
- 原型：

```c
int futex(int *uaddr, int op, int val, const struct timespec *timeout, int *uaddr2, int val3);
```
通常不直接调用 `futex()`，而是由 `pthread`、glibc、Rust、Go 等线程库内部使用。


## 使用对比表

| 原语       | 用于      | 粒度 | 需要内核支持   | 适合场景           |
| ---------- | --------- | ---- | -------------- | ------------------ |
| `mutex`    | 线程      | 中等 | 否（用户空间） | 常规临界区保护     |
| `rwlock`   | 线程      | 中等 | 否             | 多读少写的资源     |
| `cond`     | 线程      | 中等 | 是（可能休眠） | 等待事件或条件发生 |
| `atomic`   | 线程/进程 | 小   | 否             | 简单数值/标志更新  |
| `spinlock` | 线程      | 小   | 否             | 短时间争用         |
| `barrier`  | 线程      | 中等 | 否             | 多线程阶段同步     |
| `sem_t`    | 线程/进程 | 中等 | 是             | 计数信号量场景     |
| `futex`    | 线程/进程 | 高级 | 是             | 构建低延迟锁机制   |
