---
title: Java
order: 0
---

# Java

Java 是成熟的、面向对象的编程语言，以"一次编写，到处运行"（Write Once, Run Anywhere）著称。Java 在企业级应用中有广泛应用，如 Web 服务（Spring Boot）、大数据（Hadoop、Spark）、Android 开发。

## Java 生态

### JVM（Java Virtual Machine）

JVM 是 Java 的核心，负责将字节码编译成机器码并执行。JVM 包含：类加载器（ClassLoader，加载 class 文件）、运行时数据区（堆、栈、方法区、程序计数器、本地方法栈）、执行引擎（解释器、JIT 编译器、GC）、本地库接口（JNI）。

JVM 的优势：跨平台（字节码可以在任何有 JVM 的平台上运行）、安全（沙箱机制限制权限）、动态（运行时加载类）。JVM 的劣势：启动慢（JIT 编译需要时间）、内存占用大（堆、栈、元空间）。

### JDK vs JRE

JDK（Java Development Kit）是 Java 开发工具包，包含 JRE、编译器（javac）、工具（javadoc、jdb）。JRE（Java Runtime Environment）是 Java 运行环境，包含 JVM、类库。

### Java 版本

Java 8（LTS，2014）：Lambda 表达式、Stream API、日期时间 API。Java 11（LTS，2018）：本地变量类型推断（var）、HTTP Client、模块化系统。Java 17（LTS，2021）：密封类、模式匹配、记录类（Record）。Java 21（LTS，2023）：虚拟线程（Virtual Threads）、模式匹配增强、字符串模板。

## 内存模型

### JVM 内存结构

堆（Heap）：存储对象实例，是 GC 的主要区域。栈（Stack）：存储方法调用和局部变量，线程私有。方法区（Method Area）：存储类信息、常量、静态变量，JDK 8 后称为元空间（Metaspace）。程序计数器（Program Counter）：存储当前执行的字节码指令，线程私有。本地方法栈（Native Method Stack）：存储本地方法（native）的调用，线程私有。

### 堆内存划分

新生代（Young Generation）：Eden 区（新对象）、Survivor 区（From、To，存活对象）。老年代（Old Generation）：存活时间长的对象。JDK 8 后移除永久代，使用元空间（本地内存）。

### 对象创建与分配

对象创建流程：类加载检查 → 分配内存（TLAB，Thread Local Allocation Buffer，线程本地分配缓冲）→ 初始化零值 → 设置对象头 → 执行 `<init>` 方法。

对象分配策略：对象优先在 Eden 区分配、大对象直接进入老年代、长期存活对象进入老年代（年龄阈值，默认 15）、动态年龄判定（Survivor 区对象年龄总和超过阈值，进入老年代）。

## 垃圾回收

### GC 算法

标记清除（Mark-Sweep）：标记存活对象，清除未标记对象。问题：碎片化。标记整理（Mark-Compact）：标记存活对象，将存活对象移动到一端。问题：移动开销大。复制算法（Copying）：将存活对象复制到另一块内存，清空当前内存。问题：内存利用率低。

### 分代 GC

新生代 GC（Minor GC）：复制算法， Eden + From → To，存活对象年龄 +1。老年代 GC（Major GC/Full GC）：标记清除或标记整理，老年代和新生代都回收。

### GC 器

Serial GC：单线程 GC，适合小内存、单核。Parallel GC：多线程 GC，适合多核、吞吐量优先。CMS（Concurrent Mark Sweep）：并发标记清除，低延迟（JDK 14 移除）。G1（Garbage First）：分区 GC，可预测停顿时间（JDK 9 后默认）。ZGC：低延迟 GC，停顿时间 < 10ms（JDK 15+）。Shenandoah：低延迟 GC，与 ZGC 类似。

### GC 调优

GC 日志：-Xlog:gc*（JDK 9+）、-XX:+PrintGCDetails（JDK 8）。GC 参数：-Xms（初始堆）、-Xmx（最大堆）、-XX:NewRatio（新生代比例）、-XX:SurvivorRatio（Eden:Survivor 比例）。

GC 调优目标：低延迟（停顿时间）、高吞吐量（GC 时间占比）、小内存（堆内存占用）。调优策略：调整堆大小、调整新生代比例、选择合适的 GC 器、优化对象分配（减少对象创建、使用对象池）。

## 并发编程

### 线程

线程是 Java 并发的基本单位。创建线程：继承 Thread 类、实现 Runnable 接口、实现 Callable 接口（有返回值）。线程池：ThreadPoolExecutor、ForkJoinPool、ScheduledThreadPoolExecutor。

### 锁

synchronized：内置锁，基于 monitor，可重入。Lock 接口：ReentrantLock（可重入锁）、ReentrantReadWriteLock（读写锁）。锁优化：偏向锁（Biased Locking，无竞争时偏向线程）、轻量级锁（Lightweight Locking，无竞争时 CAS）、重量级锁（Heavyweight Locking，竞争时阻塞）。

### 原子类

AtomicInteger、AtomicLong、AtomicReference：基于 CAS（Compare-And-Swap）实现无锁并发。CAS 的问题：ABA 问题（版本号解决）、循环时间长开销大（LongAdder 解决）、只能保证一个变量（AtomicReference 解决）。

### 并发容器

ConcurrentHashMap：分段锁（JDK 7）、CAS + synchronized（JDK 8）。CopyOnWriteArrayList：写时复制，适合读多写少。BlockingQueue：阻塞队列，支持生产者-消费者模式。

## 虚拟线程（Virtual Threads）

### 虚拟线程的原理

虚拟线程是 JDK 21 引入的轻量级线程，由 JVM 调度而非 OS。虚拟线程的优势：轻量级（一个虚拟线程只占几百字节）、高并发（可以创建数百万个虚拟线程）、简单（像线程一样使用）。

### 虚拟线程 vs 平台线程

平台线程（Platform Thread）是 OS 线程的包装，1:1 映射到 OS 线程。虚拟线程（Virtual Thread）是 JVM 调度的协程，M:N 映射到 OS 线程。平台线程创建和销毁开销大，虚拟线程创建和销毁开销小。

### 虚拟线程的使用

创建虚拟线程：Thread.ofVirtual().start(() -> { ... })。虚拟线程池：Executors.newVirtualThreadPerTaskExecutor()。阻塞操作（如 I/O）会释放虚拟线程，不会阻塞平台线程。

### 虚拟线程的问题

虚拟线程不适合 CPU 密集型任务（仍然占用平台线程）。虚拟线程不支持 ThreadLocal（或慎用，可能导致内存泄漏）。虚拟线程的异常处理复杂（需要 UncaughtExceptionHandler）。

## Java 的适用场景

### 适合的场景

企业级应用：Spring Boot、微服务、RESTful API。大数据：Hadoop、Spark、Flink。Android 开发：Android SDK 是 Java（Kotlin 兼容）。科学计算：ND4J、Deeplearning4j。

### 不适合的场景

系统编程：无直接内存访问、无指针。实时系统：GC 停顿不可预测。高性能服务：启动慢、内存占用大。

Java 是成熟的、生态丰富的语言，适合企业级应用和大数据。理解 Java 的 JVM、内存模型、GC、并发编程，有助于编写高性能、稳定的 Java 应用。
