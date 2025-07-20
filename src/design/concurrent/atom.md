# 原子操作
原子操作（Atomic Operation）是指在多处理器或多线程环境下，不可被中断、不可分割的操作。即使有多个线程/CPU 并发执行，原子操作要么全部完成，要么全部不做，不会出现中间状态。

以最简单的多线程同时对同一个变量进行加 1 操作的例子为例

**原子变量**是指支持原子操作的数据类型。对原子变量的读写、加减等操作在硬件或内核层面保证原子性，避免了竞态条件。

在多核/多线程环境下，普通变量的操作（如 `a++`）实际上分为多步（读、加、写），可能被其他线程打断，导致数据竞争和不一致。原子操作可以保证这些操作的完整性，无需加锁即可安全并发访问。

## 二、原子指令

**原子指令**是 CPU 提供的、能够保证原子性的特殊指令。例如：

- x86 架构的 `LOCK` 前缀指令（如 `LOCK XADD`、`LOCK CMPXCHG`）
- ARM 架构的 `LDREX/STREX`、`SWP` 等

这些指令通常用于实现原子加减、原子交换、原子比较并交换（CAS, Compare-And-Swap）等操作，是无锁并发的基础。

---

## 三、Linux 内核中的原子操作

Linux 内核为不同平台提供了统一的原子操作接口，常见的有：

- `atomic_t`、`atomic64_t`：原子整型变量
- 常用操作函数：
  - `atomic_read()`、`atomic_set()`
  - `atomic_inc()`、`atomic_dec()`、`atomic_add()`、`atomic_sub()`
  - `atomic_cmpxchg()`（原子比较并交换，CAS）
  - `atomic_xchg()`（原子交换）

### 示例代码

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

---

## 四、无锁并发（Lock-free Concurrency）

**无锁并发**是指在多线程环境下，不依赖传统的互斥锁（mutex、spinlock）来保证数据一致性，而是依靠原子操作和原子指令实现线程安全。其优点包括：

- 避免死锁和优先级反转
- 提高并发性能，减少上下文切换
- 适合高性能、低延迟场景

### 常见无锁并发技术

- **CAS（Compare-And-Swap）**：最常用的无锁原语，广泛用于实现无锁队列、无锁栈等数据结构。
- **原子计数器**：如引用计数、统计计数等。
- **环形缓冲区、无锁队列**：常用于内核、网络、日志等高并发场景。

### 简单无锁队列示例（伪代码）

```c
// 伪代码，仅示意
do {
    old_tail = queue->tail;
    new_tail = old_tail + 1;
} while (!atomic_cmpxchg(&queue->tail, old_tail, new_tail));
```

---

## 五、注意事项

- 原子操作适合简单数据类型（如整型），复杂结构体仍需加锁保护。
- 无锁并发虽然高效，但实现复杂，需谨慎设计，避免ABA问题等陷阱。
- 在 SMP（多核）系统下，原子操作通常伴随内存屏障（memory barrier）以保证可见性和有序性。

---

## 参考

- [Linux 内核文档：原子操作](https://www.kernel.org/doc/html/latest/core-api/atomic_ops.html)
- [Linux 源码 include/linux/atomic.h](https://elixir.bootlin.com/linux/latest/source/include/linux/atomic.h)

---

如需补充更详细的代码示例、平台差异或深入原理，请告知！