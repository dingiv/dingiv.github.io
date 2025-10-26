# 原子操作
原子操作（Atomic Operation）是指在多处理器或多线程环境下，不可被中断、不可分割的操作。即使有多个线程/CPU 并发执行，原子操作要么全部完成，要么全部不做，不会出现中间状态。原子操作由硬件级别提供支持，与硬件强相关。

## 原子指令和原子变量
**原子指令**是各个 CPU 平台提供的，能够保证原子性的特殊指令，是原子操作的基础。

- x86 架构的 `LOCK` 前缀指令（如 `LOCK XADD`、`LOCK CMPXCHG`）
- ARM 架构的 `LDREX/STREX`、`SWP` 等

这些指令通常用于实现原子加减、原子交换、原子比较并交换（CAS, Compare-And-Swap）等操作，是上层原子操作的基础。

**原子变量**是指支持原子操作的数据类型。需要特别说明的是，原子变量往往是在编译器层面实现的类型标记，其内存布局相较于普通的数据类型没有本质差异。原子变量用于提示编译器进行内存优化和语义提示。

以最简单的多线程同时对同一个变量进行加 1 操作的例子为例，在多核/多线程环境下，普通变量的操作（如 `a++`）实际上分为多步（读、加、写），可能被其他线程打断，导致数据竞争和不一致。原子操作可以保证这些操作的完整性，无需加锁即可安全并发访问。

## linux 原子接口
Linux 内核为不同平台提供了统一的原子操作接口，屏蔽了下层硬件平台的原子操作。以 C 函数的形式给出。

常见接口：

| 类别         | 接口/类型名称                       | 说明                                                              |
| ------------ | ----------------------------------- | ----------------------------------------------------------------- |
| **原子类型** | `atomic_t`                          | 32位整型                                                          |
|              | `atomic64_t` / `atomic_long_t`      | 64位整型（部分平台为 `atomic_long_t`）                            |
| **原子操作** | `atomic_set(&v, i)`                 | 赋值，把 `v` 设置为 `i`                                           |
|              | `atomic_read(&v)`                   | 读取 `v` 的值                                                     |
|              | `atomic_add(i, &v)`                 | 给 `v` 加上 `i`                                                   |
|              | `atomic_sub(i, &v)`                 | 从 `v` 减去 `i`                                                   |
|              | `atomic_inc(&v)`                    | 自增1（`v = v + 1`）                                              |
|              | `atomic_dec(&v)`                    | 自减1（`v = v - 1`）                                              |
|              | `atomic_add_return(i, &v)`          | 加上 `i` 并返回新值                                               |
|              | `atomic_sub_return(i, &v)`          | 减去 `i` 并返回新值                                               |
|              | `atomic_cmpxchg(&v, old, new)`      | 如果 `v` 等于 `old`，则设为 `new`，返回原值                       |
|              | `atomic_xchg(&v, new)`              | 把 `v` 设为 `new` 并返回旧值                                      |
|              | `atomic_add_unless(&v,a,u)`         | 如果 `v` 不等于 `u`，则加上 `a`，如果有加则返回非零               |
| **64位变体** | `atomic64_set`、`atomic64_read` ... | 操作与对应32位类似，仅操作对象为 `atomic64_t`，接口名带 `64` 后缀 |
| **初始化**   | `ATOMIC_INIT(i)`                    | 初始化 `atomic_t` 为 `i`                                          |
|              | `ATOMIC64_INIT(i)`                  | 初始化 `atomic64_t` 为 `i`                                        |
```c
#include <linux/atomic.h>

atomic_t counter = ATOMIC_INIT(0);

void example(void) {
    atomic_inc(&counter); // 原子加1
    int val = atomic_read(&counter); // 原子读取
    atomic_dec(&counter); // 原子减1
    // CAS操作
    int old = atomic_cmpxchg(&counter, 5, 10); // 如果counter==5，则设为10
}
```

## C 接口原子接口
C11 标准库中引入了语言级的原子变量接口，提供跨操作系统的原子操作抽象。

原子变量
- `atomic_int`

```c
#include <stdatomic.h>
#include <pthread.h>
#include <stdio.h>

atomic_int counter = 0;

void* increment(void* arg) {
    for (int i = 0; i < 1000; i++) {
        atomic_fetch_add(&counter, 1); // Atomic increment
    }
    return NULL;
}

int main() {
    pthread_t threads[10];
    for (int i = 0; i < 10; i++) {
        pthread_create(&threads[i], NULL, increment, NULL);
    }
    for (int i = 0; i < 10; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("Counter: %d\n", atomic_load(&counter)); // Outputs ~10000
    return 0;
}
```

| 类别               | 接口/类型名称                                     | 说明                                                             |
| ------------------ | ------------------------------------------------- | ---------------------------------------------------------------- |
| **原子类型**       | `_Atomic` 限定符，如 `_Atomic int`                | 用于声明原子类型变量                                             |
|                    | `atomic_bool`                                     |                                                                  |
|                    | `atomic_char`                                     |                                                                  |
|                    | `atomic_int`                                      |                                                                  |
|                    | `atomic_long`                                     |                                                                  |
|                    | `atomic_pointer`                                  |                                                                  |
| **原子操作**       | `atomic_store(&obj, val)`                         | 存储，将 `val` 写入 `obj`                                        |
|                    | `atomic_load(&obj)`                               | 读取，从 `obj` 读取值                                            |
|                    | `atomic_exchange(&obj, val)`                      | 替换，将 `obj` 设为 `val`，返回旧值                              |
|                    | `atomic_fetch_add(&obj, opnd)`                    | 加上 `opnd`，返回操作前的值                                      |
|                    | `atomic_fetch_sub(&obj, opnd)`                    | 减去 `opnd`，返回操作前的值                                      |
|                    | `atomic_fetch_and(&obj, opnd)`                    | 按位与（AND），返回操作前的值                                    |
|                    | `atomic_fetch_or(&obj, opnd)`                     | 按位或（OR），返回操作前的值                                     |
|                    | `atomic_fetch_xor(&obj, opnd)`                    | 按位异或（XOR），返回操作前的值                                  |
|                    | `atomic_compare_exchange_strong(&obj, &exp, des)` | 若 `obj` 等于 `exp`，则设为 `des`，否则更新 `exp` 为当前值       |
|                    | `atomic_compare_exchange_weak(&obj, &exp, des)`   | 类似 strong，可能会伪失败                                        |
| **内存序控制参数** | `memory_order_relaxed`                            | 无序保证，执行速度最快，但允许操作重排                           |
|                    | `memory_order_acquire`                            | 读取操作，保证之后的操作不会被重排到该读取之前                   |
|                    | `memory_order_release`                            | 写入操作，保证之前的操作不会被重排到该写入之后                   |
|                    | `memory_order_acq_rel`                            | 用于读改写操作（如 fetch_add），同时具备 acquire 与 release 语义 |
|                    | `memory_order_seq_cst`                            | 最严格，也是默认选项，所有线程间保证全序一致性                   |
| **辅助操作**       | `atomic_init(&obj, val)`                          | 初始化原子类型变量                                               |
|                    | `atomic_is_lock_free(&obj)`                       | 检查该对象的操作是否为无锁实现                                   |

```c
#include <stdatomic.h>
atomic_int counter = 0;
atomic_fetch_add(&counter, 1); // Increments counter atomically
int value = atomic_load(&counter); // Reads counter atomically
```

## 内存屏障
现代的编译器会在编译时做很多工作，其中有的工作是指令重排，也就是代码执行的实际顺序不一定按照我们编写的代码顺序来执行。在单线程情况下，这通常不会有问题，但是一旦到了多线程的环境下，指令重排就可能导致数据竞争和不可预期的结果。更糟糕的是，现代的 CPU 往往可以使用一些硬件加速手段，例如：分支预测和流水线并发，这些硬件级别的行为也会导致指令顺序发生重排。

为了解决这种问题，引入了内存屏障（memory barrier，或 memory fence）的机制。内存屏障是一种特殊的指令，用于限制编译器和 CPU 的指令重排，保证内存操作的可见性和有序性。常见的内存屏障分为以下几类：
+ 全屏障（Full Barrier, 通常为 mfence）：确保其前后的所有读写操作都完成，常用于关键的同步点。
+ 读屏障（Load Barrier, lfence）：保证屏障前的读操作全部完成，才会进行屏障后的读操作。
+ 写屏障（Store Barrier, sfence）：保证屏障前的写操作全部完成，之后的写操作才会开始。

内存屏障主要用途包括：
+ 防止乱序执行：确保关键顺序不可颠倒，避免多线程访问共享变量时出现“看见过时值”或“操作丢失”问题。
+ 实现同步原语：如锁（mutex）、自旋锁（spinlock）、信号量（semaphore）等都依赖内存屏障保障可见性和顺序性。
+ 确保写入对其他CPU可见：跨核心通信时，强制将数据从 CPU 缓存刷回主内存，确保其他处理器能够及时看到变更。

## 总结
- 原子操作适合简单数据类型（如整型），复杂结构体仍需加锁保护。
- 无锁并发虽然高效，但实现复杂，需谨慎设计，避免ABA问题等陷阱。
- 在 SMP（多核）系统下，原子操作通常伴随内存屏障（memory barrier）以保证可见性和有序性。

