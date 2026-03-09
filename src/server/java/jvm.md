---
title: JVM 内存与 GC
order: 11
---

# JVM 内存与 GC

JVM 内存管理和垃圾回收是 Java 性能优化的核心。理解 JVM 的内存结构、GC 算法、GC 调优，有助于解决内存泄漏、GC 停顿、性能瓶颈。

## JVM 内存结构

### 程序计数器（Program Counter）

程序计数器是当前线程执行的字节码指令的行号指示器。字节码解释器通过改变程序计数器的值来选择下一条指令。程序计数器是线程私有的，每个线程都有独立的程序计数器。

### Java 虚拟机栈（Java Virtual Machine Stack）

Java 虚拟机栈存储方法调用和局部变量。每个方法调用创建一个栈帧（Stack Frame），栈帧包含：局部变量表（Local Variable Table）、操作数栈（Operand Stack）、动态链接（Dynamic Linking）、返回地址（Return Address）。Java 虚拟机栈是线程私有的。

栈帧的大小在编译时确定，动态大小会导致 StackOverflowError（栈溢出）。-Xss 参数调整栈大小。

### 本地方法栈（Native Method Stack）

本地方法栈为本地方法（native）服务，存储本地方法的调用和局部变量。本地方法栈是线程私有的。

### 堆（Heap）

堆是 JVM 中最大的一块内存，存储对象实例。堆是所有线程共享的，GC 的主要区域。堆可以通过 -Xms（初始堆）、-Xmx（最大堆）调整大小。

### 方法区（Method Area）

方法区存储类信息、常量、静态变量、即时编译器编译后的代码。方法区是所有线程共享的。JDK 8 后，方法区称为元空间（Metaspace），使用本地内存，可以通过 -XX:MetaspaceSize、-XX:MaxMetaspaceSize 调整大小。

## 对象创建与分配

### 类加载检查

虚拟机遇到 new 指令时，首先检查类是否已加载、解析、初始化。如果没有，先执行类加载。

### 分配内存

类加载检查通过后，虚拟机为新生对象分配内存。分配方式：指针碰撞（Bump the Pointer，堆内存规整时）、空闲列表（Free List，堆内存不规整时）。

TLAB（Thread Local Allocation Buffer，线程本地分配缓冲）：为了提高分配效率，每个线程在堆中分配一块私有内存，线程优先在 TLAB 中分配对象，TLAB 满了再从堆中分配新的 TLAB。

### 初始化零值

内存分配完成后，虚拟机将分配到的内存块初始化为零值（如 int 为 0，boolean 为 false）。这保证了对象的实例字段可以不赋值直接使用。

### 设置对象头

对象头（Object Header）包含：Mark Word（哈希码、锁状态、GC 年龄）、类型指针（指向类元数据）、数组长度（如果是数组）。Mark Word 的状态：无锁（hashcode + age）、偏向锁（thread ID + epoch + age）、轻量级锁（指向栈中 Lock Record 的指针）、重量级锁（指向堆中 Monitor 的指针）。

### 执行 `<init>` 方法

执行对象的构造函数（`<init>`），初始化对象的字段。

## 对象的内存布局

### 对象头

对象头包含 Mark Word 和类型指针。Mark Word（32 位 JVM 占 4 字节，64 位 JVM 占 8 字节）：无锁（25 bit hashcode + 4 bit age + 1 bit biased + 2 bit lock_state）、偏向锁（54 bit thread ID + 2 bit epoch + 4 bit age + 1 bit biased + 1 bit lock_state）、轻量级锁（62 bit 指向 Lock Record 的指针 + 2 bit lock_state）、重量级锁（62 bit 指向 Monitor 的指针 + 2 bit lock_state）。

### 实例数据

实例数据是对象的字段，包括父类的字段。字段的存储顺序：longs/doubles、ints/floats、shorts/chars、bytes/booleans、oops（普通对象指针）。相同宽度的字段分配在一起，父类的字段在子类字段之前。

### 对齐填充

对齐填充不是必需的，用于保证对象大小是 8 字节的整数倍（64 位 JVM）。Hot JVM 要求对象起始地址是 8 字节的整数倍，提高内存访问效率。

## 垃圾回收

### 对象存活判定

引用计数（Reference Counting）：对象被引用时计数 +1，引用失效时计数 -1，计数为 0 时可回收。引用计数的问题是循环引用（A 引用 B，B 引用 A，无法回收）。

可达性分析（Reachability Analysis）：从 GC Roots 开始向下搜索，走过的路径称为引用链，如果一个对象到 GC Roots 没有任何引用链，则可回收。GC Roots 包括：栈中引用的对象、方法区静态属性引用的对象、方法区常量引用的对象、本地方法栈中引用的对象。

### 引用类型

强引用（Strong Reference）：普通的对象引用，只要强引用存在，GC 不会回收。软引用（SoftReference）：内存不足时回收，适合缓存。弱引用（WeakReference）：GC 时回收，适合缓存、映射。虚引用（PhantomReference）：无法通过虚引用获取对象，用于跟踪对象回收。

### GC 算法

标记清除（Mark-Sweep）：标记存活对象，清除未标记对象。优点：不需要额外空间。缺点：碎片化。

标记整理（Mark-Compact）：标记存活对象，将存活对象移动到一端。优点：无碎片化。缺点：移动开销大。

复制算法（Copying）：将存活对象复制到另一块内存，清空当前内存。优点：无碎片、效率高。缺点：内存利用率低（需要一块空闲内存）。

分代收集（Generational Collection）：新生代用复制算法（存活对象少），老年代用标记清除或标记整理（存活对象多）。

### GC 器

Serial GC：单线程 GC，适合小内存、单核。新生代用复制算法，老年代用标记整理。

Parallel GC：多线程 GC，适合多核、吞吐量优先。新生代用复制算法，老年代用标记整理。

CMS（Concurrent Mark Sweep）：并发标记清除，低延迟。新生代用复制算法，老年代用标记清除（并发）。CMS 的问题：碎片化、CPU 敏感、浮动垃圾。

G1（Garbage First）：分区 GC，将堆分成多个 Region，可预测停顿时间。新生代和老年代不再物理隔离，而是逻辑上的。G1 的优势：可预测停顿时间、无碎片化。G1 的代价：额外内存占用（Remembered Set）。

ZGC：低延迟 GC，停顿时间 `< 10ms`。ZGC 的技术：染色指针（Colored Pointers，将元数据存储在指针中）、读屏障（Load Barrier，读取对象时检查）、并发整理（Concurrent Relocation）。

## GC 调优

### GC 日志

JDK 9+：-Xlog:gc*:file=gc.log（GC 日志输出到文件）。JDK 8：-XX:+PrintGCDetails、-XX:+PrintGCDateStamps（打印 GC 详细信息和时间戳）。

### GC 参数

-Xms：初始堆大小。-Xmx：最大堆大小。-XX:NewRatio：新生代与老年代的比例（默认 2，新生代:老年代 = 1:2）。-XX:SurvivorRatio：Eden 与 Survivor 的比例（默认 8，Eden:Survivor:Survivor = 8:1:1）。-XX:MaxTenuringThreshold：对象晋升老年代的年龄阈值（默认 15）。-XX:ParallelGCThreads：并行 GC 的线程数。

### GC 调优策略

调整堆大小：-Xms = -Xmx（避免动态扩容）、堆大小 = (存活对象 × 3) / (1 - GC 时间占比)。

调整新生代大小：新生代大一些，减少 Full GC。新生代太大，老年代太小，老年代容易满。

选择合适的 GC 器：小内存（`<4GB`）用 Serial GC 或 Parallel GC，大内存（`>4GB`）用 G1 或 ZGC。低延迟用 G1 或 ZGC，高吞吐量用 Parallel GC。

优化对象分配：减少对象创建（如用基本类型）、使用对象池（如 ThreadLocal）、避免大对象（直接进入老年代）。

JVM 内存管理和 GC 是 Java 性能优化的核心。理解 JVM 的内存结构、GC 算法、GC 调优，有助于解决内存泄漏、GC 停顿、性能瓶颈。
