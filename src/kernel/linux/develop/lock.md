# 同步原语
在 Linux 内核中，多个线程或进程可能同时访问共享资源，这就需要同步机制来保证数据的一致性和正确性。Linux 内核提供了多种同步原语，每种都有其特定的使用场景和特点。

## 锁的粒度

### 读写分离

## 自旋锁
自旋锁是一种忙等待的锁机制，当线程无法获取锁时，会一直循环检查锁的状态，直到获得锁为止。支持读写分离。

```c
#include <linux/spinlock.h>

spinlock_t my_lock = SPIN_LOCK_UNLOCKED;

// 获取锁
spin_lock(&my_lock);
// 临界区代码
spin_unlock(&my_lock);


rwlock_t my_rwlock = RW_LOCK_UNLOCKED;
// 读者获取锁
read_lock(&my_rwlock);
// 读取操作
read_unlock(&my_rwlock);

// 写者获取锁
write_lock(&my_rwlock);
// 写入操作
write_unlock(&my_rwlock);
```
 
自旋锁具有响应速度快的优点，因为它无需进行上下文切换，线程在等待锁时会一直循环检查锁的状态。这种机制特别适用于短时间持有的锁，在 SMP（对称多处理器）系统中能够提供较高的效率。

然而，自旋锁也存在明显的缺点。它会占用 CPU 时间进行忙等待，当锁被长时间持有时会造成资源浪费。此外，自旋锁不适用于长时间持有的锁场景，并且可能导致优先级反转问题，影响系统的实时性能。

自旋锁主要适用于三种场景：中断处理程序、短时间的临界区操作，以及不能睡眠的上下文环境。在这些场景下，自旋锁能够提供最佳的同步性能。

## 信号量
信号量是一种计数同步原语，可以控制对资源的访问数量。

### 计数量
```c
#include <linux/semaphore.h>

struct semaphore sem;

// 初始化信号量
sema_init(&sem, 1);  // 二值信号量
sema_init(&sem, 5);  // 计数信号量

// 获取信号量
down(&sem);          // 阻塞版本
down_interruptible(&sem);  // 可中断版本
down_trylock(&sem);  // 非阻塞版本

// 释放信号量
up(&sem);
```

### 完成量
完成量是一种特殊的同步原语，用于等待某个事件完成。

```c
#include <linux/completion.h>

struct completion comp;

// 初始化完成量
init_completion(&comp);

// 等待完成
wait_for_completion(&comp);

// 通知完成
complete(&comp);
complete_all(&comp);  // 唤醒所有等待者
```
 
信号量具有多个显著优点。首先，它能够精确控制资源的访问数量，通过设置不同的计数值来限制同时访问资源的线程数量。其次，信号量支持阻塞等待机制，当资源不可用时，线程可以主动让出CPU，避免忙等待造成的资源浪费。此外，信号量特别适用于生产者-消费者模式，能够有效协调多个线程之间的协作关系。

然而，信号量也存在一些明显的缺点。在实时系统中，信号量可能导致优先级反转问题，即低优先级线程持有信号量时，会阻塞高优先级线程的执行。另外，在SMP（对称多处理器）系统中，信号量的性能表现不如自旋锁，因为信号量涉及更多的上下文切换和调度开销。

信号量的使用场景非常广泛。它常用于实现资源池管理，如数据库连接池、线程池等场景。在生产者-消费者模式中，信号量能够有效控制缓冲区的大小和访问。此外，信号量还适用于需要限制并发访问数量的场景，如限制同时访问某个服务的客户端数量。

## 互斥锁
互斥锁是一种二值同步原语，确保同一时间只有一个线程能访问共享资源。支持读写分离。

```c
#include <linux/mutex.h>

struct mutex my_mutex;

// 初始化互斥锁
mutex_init(&my_mutex);

// 获取锁
mutex_lock(&my_mutex);
// 临界区代码
mutex_unlock(&my_mutex);

// 尝试获取锁（非阻塞）
if (mutex_trylock(&my_mutex)) {
    // 成功获取锁
    mutex_unlock(&my_mutex);
}
 
#include <linux/rwsem.h>

struct rw_semaphore my_rwsem;

// 初始化读写锁
init_rwsem(&my_rwsem);

// 读者获取锁
down_read(&my_rwsem);
// 读取操作
up_read(&my_rwsem);

// 写者获取锁
down_write(&my_rwsem);
// 写入操作
up_write(&my_rwsem);
```
 
**优点：**

- 支持阻塞等待，不占用 CPU
- 自动处理优先级继承
- 适用于长时间持有的锁

**缺点：**

- 需要上下文切换，开销较大
- 不能在中断上下文中使用

## 原子操作
原子操作是不可分割的操作，在 SMP 系统中保证操作的原子性。

### 原子变量

```c
#include <linux/atomic.h>

atomic_t counter = ATOMIC_INIT(0);

// 原子操作
atomic_inc(&counter);           // 递增
atomic_dec(&counter);           // 递减
atomic_add(10, &counter);       // 加法
atomic_sub(5, &counter);        // 减法
atomic_set(&counter, 100);      // 设置值
int val = atomic_read(&counter); // 读取值
```

### 原子位操作

```c
#include <linux/bitops.h>

unsigned long flags = 0;

// 原子位操作
set_bit(0, &flags);             // 设置位
clear_bit(0, &flags);           // 清除位
test_and_set_bit(0, &flags);    // 测试并设置位
test_and_clear_bit(0, &flags);  // 测试并清除位
```

## RCU（Read-Copy Update）

RCU 是一种无锁同步机制，适用于读多写少的场景。

### 基本使用

```c
#include <linux/rcupdate.h>

struct my_data {
    int value;
    struct rcu_head rcu;
};

// 读者
rcu_read_lock();
struct my_data *data = rcu_dereference(ptr);
// 使用data
rcu_read_unlock();

// 写者
struct my_data *new_data = kmalloc(sizeof(*new_data), GFP_KERNEL);
// 初始化new_data
rcu_assign_pointer(ptr, new_data);
synchronize_rcu();  // 等待所有读者完成
kfree(old_data);
```

### RCU 的特点

**优点：**

- 读者无锁，性能极高
- 适用于读多写少的场景
- 无死锁问题

**缺点：**

- 写者开销较大
- 内存回收延迟
- 实现复杂

## 顺序锁

顺序锁是一种乐观锁机制，适用于读多写少的场景。

```c
#include <linux/seqlock.h>

seqlock_t my_seqlock = SEQLOCK_UNLOCKED;

// 读者
unsigned seq;
do {
    seq = read_seqbegin(&my_seqlock);
    // 读取数据
} while (read_seqretry(&my_seqlock, seq));

// 写者
write_seqlock(&my_seqlock);
// 修改数据
write_sequnlock(&my_seqlock);
```

## 锁的选择

1. 根据使用场景选择
- 中断上下文：使用自旋锁或原子操作
- 进程上下文：使用互斥锁或信号量
- 读多写少：使用 RCU 或读写锁
- 短时间持有：使用自旋锁
- 长时间持有：使用互斥锁

### 2. 性能考虑

```
// 性能从高到低排序
原子操作 > 自旋锁 > RCU > 读写锁 > 互斥锁 > 信号量
```

### 3. 避免死锁

- 按固定顺序获取多个锁
- 使用 trylock 避免阻塞
- 设置锁的超时时间

### 4. 实际示例

```c
// 共享计数器示例
struct shared_counter {
    atomic_t count;
    spinlock_t lock;
    struct mutex mutex;
};

// 使用原子操作（最高性能）
void inc_atomic(struct shared_counter *sc) {
    atomic_inc(&sc->count);
}

// 使用自旋锁（中等性能）
void inc_spinlock(struct shared_counter *sc) {
    spin_lock(&sc->lock);
    sc->count.counter++;
    spin_unlock(&sc->lock);
}

// 使用互斥锁（最低性能，但最安全）
void inc_mutex(struct shared_counter *sc) {
    mutex_lock(&sc->mutex);
    sc->count.counter++;
    mutex_unlock(&sc->mutex);
}
```

## 调试和性能分析

### 锁竞争检测

```c
// 启用锁竞争检测
CONFIG_LOCKDEP=y
CONFIG_DEBUG_LOCK_ALLOC=y

// 运行时检测
echo 1 > /proc/sys/kernel/lock_stat
```

### 性能监控

```c
// 查看锁统计信息
cat /proc/lock_stat

// 使用perf分析锁竞争
perf record -g -p <pid>
perf report
```

通过合理选择和使用这些同步原语，可以构建高效、安全的多线程内核代码。每种同步机制都有其适用场景，理解它们的特点和性能特征是内核开发的重要基础。
