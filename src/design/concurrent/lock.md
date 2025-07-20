# 同步原语
同步原语（synchronization primitives）是用于协调多个线程或进程对共享资源的访问，防止数据竞争和状态不一致的关键工具。同步原语分为线程级和进程级，进程级往往需要内核的支持。

大多数语言都是 C 语言系的，C 语言中的锁为上层高级语言提供底层支持。

## 线程同步原语
线程同步原语（Linux 下基于 `pthread`）。这些原语定义在 `<pthread.h>` 中，仅在线程之间共享内存的环境下使用。

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


## 乐观锁
尝试拿锁，如果拿不到就再次尝试，如果再拿不到直接放弃操作，返回失败。乐观锁是一种设计思维，而不是固定的
| 特点     | 乐观锁                     | 悲观锁                       |
| -------- | -------------------------- | ---------------------------- |
| 加锁方式 | 不加锁，更新时检测冲突     | 加锁，访问时就互斥           |
| 冲突代价 | 冲突时失败或重试，代价较高 | 冲突时阻塞等待               |
| 并发性能 | 并发高，适合读多写少       | 并发低，适合写多或强一致性   |
| 实现方式 | 版本号 / 时间戳 / CAS      | 显式锁（mutex）/ 行锁 / 表锁 |


```sql
CREATE TABLE product (
    id INT PRIMARY KEY,
    stock INT,
    version INT
);

-- A 线程
UPDATE product SET stock = 9, version = 6 WHERE id = 1 AND version = 5;

-- B 线程，失败，因为 version 不正确
UPDATE product SET stock = 9, version = 6 WHERE id = 1 AND version = 5;
```


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
