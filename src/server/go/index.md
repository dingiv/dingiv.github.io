---
title: Go
---

# Go 语言

Go 是 Google 开发的静态类型语言，专为并发和大规模系统设计。Go 的简单性、高性能、内置并发，使其成为云原生时代的首选语言。

## Go 的设计哲学

### 简单性

Go 只有 25 个关键字，语法简单，学习曲线平缓。Go 没有类和继承，只有结构体和接口。Go 的接口是隐式的，类型不需要显式声明实现接口。

### 并发性

Go 的 goroutine 是轻量级线程，一个 goroutine 只占 2KB 内存。Go 的 channel 是类型安全的管道，用于 goroutine 间通信。CSP（Communicating Sequential Processes）模型：不要通过共享内存通信，通过通信共享内存。

### 性能

Go 是编译型语言，性能接近 C。Go 的垃圾回收器（GC）是并发的，停顿时间短（<1ms）。Go 的调度器是 M:N 的，多个 goroutine 映射到多个 OS 线程，充分利用多核。

## 类型系统

### 基本类型

基本类型：bool、int/int8/int16/int32/int64、uint/uint8/uint16/uint32/uint64、float32/float64、complex64/complex128、string、byte（uint8 的别名）、rune（int32 的别名）。

### 复合类型

数组：[n]T，长度是类型的一部分，数组长度不能变化。切片：[]T，动态数组，底层引用数组，可以追加元素。映射：map[K]V，哈希表，键必须可比较。结构体：struct，字段集合。指针：*T，指向变量的地址。接口：interface，方法集合。

### 接口

接口是方法的抽象，类型实现接口的所有方法，就隐式实现了接口。空接口 interface{} 没有任何方法，所有类型都实现了空接口，可以持有任意值。

接口的动态类型和动态值：接口有两个字段，类型（值的类型）和值（值本身）。空接口的类型和值都为 nil，表示"零值"。

## 并发模型

### Goroutine

Goroutine 是 Go 的协程，轻量级、低开销。创建 goroutine：go func() { ... }()。Goroutine 的调度由 Go 运行时负责，不需要 OS 参与。

Goroutine 的调度模型：GMP 模型。G（goroutine）是用户态协程，M（machine）是 OS 线程，P（processor）是逻辑处理器。P 维护一个本地 G 队列，M 从 P 的队列获取 G 执行。队列为空时，从全局队列或其他 P 偷取 G。

### Channel

Channel 是类型安全的管道，用于 goroutine 间通信。创建 channel：ch := make(chan int)。发送：ch <- value，接收：value := <-ch。

Channel 的类型：无缓冲 channel（发送和接收同步）、有缓冲 channel（发送可以缓冲，接收可以延迟）。关闭 channel：close(ch)，关闭后不能再发送，可以继续接收。

Channel 的 select：select 等待多个 channel 操作，哪个先完成就执行哪个。select 可以带 default，非阻塞地发送或接收。

### Sync 包

Sync 包提供同步原语：sync.Mutex（互斥锁）、sync.RWMutex（读写锁）、sync.WaitGroup（等待组）、sync.Once（只执行一次）、sync.Pool（对象池）、sync.Cond（条件变量）。

sync.Map 是并发安全的 map，适合读多写少的场景。写多的场景应该用普通 map + sync.RWMutex。

## 内存管理

### 堆和栈

Go 的值可以分配在堆或栈。栈分配快（移动栈指针），堆分配慢（需要 GC）。逃逸分析（Escape Analysis）决定值分配在堆或栈。如果值被返回或被闭包捕获，则逃逸到堆。

栈的大小：goroutine 的初始栈是 2KB，按需扩容（最大 1GB）。栈的扩容和收缩由 Go 运行时管理。

### 垃圾回收

Go 的 GC 是三色标记清除（Tri-color Mark and Sweep）并发 GC。三色：白色（未访问）、灰色（已访问，但子节点未访问）、黑色（已访问，子节点也访问）。

GC 的过程：1. 标记准备（stop the world，很短）、2. 并发标记（goroutine 和 GC 并发）、3. 标记终止（stop the world，很短）、4. 并发清扫（goroutine 和 GC 并发）。

GC 的触发：内存分配量达到阈值、定时触发（2 分钟）、手动触发（runtime.GC()）。

### 内存分配器

Go 的内存分配器是 tcmalloc 的变种，基于 span（内存块）。mcache 是每个 P 的本地缓存，mcentral 是全局的 span 缓存，mheap 是堆管理器。小对象（<32KB）从 mcache 分配，大对象从 mheap 分配。

## 反射与接口

### 反射

反射是程序在运行时检查类型和值的能力。reflect.TypeOf 获取类型，reflect.Value 获取值。反射可以动态调用方法、修改值（需要值可寻址）、创建对象。

反射的开销大，应该避免在热路径中使用。反射的常见用途：JSON 序列化/反序列化、ORM、配置解析。

### 接口值

接口值包含动态类型和动态值。接口值是 nil 当且仅当类型和值都为 nil。接口值比较时，类型和值都相同才相等。

接口值的断言：value, ok := iface.(Type)，ok 表示断言是否成功。类型 switch：switch value := iface.(type) { case int: ... case string: ... }。

## 错误处理

### Error 接口

Error 接口只有一个方法 Error() string。创建错误：errors.New("error")、fmt.Errorf("format %s", arg)。自定义错误：type MyError struct { Msg string }，实现 Error() string 方法。

### 错误包装

Go 1.13 引入错误包装：fmt.Errorf("...: %w", err) 包装错误，errors.Unwrap 解包错误，errors.Is 判断错误链中是否有某个错误，errors.As 获取错误链中的某个错误。

### Panic 和 Recover

Panic 是运行时错误（如数组越界），会导致程序崩溃。Recover 可以捕获 panic，恢复程序执行。Panic 和 Recover 应该只在库内部使用，应用层应该返回 error。

## Go 的标准库

### Net/Http

Net/Http 是 Go 的 HTTP 客户端和服务端库。http.Get 发送 GET 请求，http.Post 发送 POST 请求。http.Server 创建 HTTP 服务器，http.HandleFunc 注册处理器。

http.Handler 接口：ServeHTTP(ResponseWriter, *Request)。http.HandlerFunc 是函数类型，实现了 Handler 接口。

### Context

Context 是请求范围的值、取消信号、截止时间。Context 可以在 goroutine 间传递取消信号和截止时间。Context 的类型：context.Background（根 context）、context.TODO（未确定用途）、context.WithCancel（可取消）、context.WithDeadline（有截止时间）、context.WithValue（携带值）。

Context 的使用：每个请求应该创建一个 context，传递给所有依赖的 goroutine。当请求取消时，所有 goroutine 收到取消信号，退出执行。

Go 是云原生时代的 C，简单、高效、并发。理解 Go 的类型系统、并发模型、内存管理，有助于编写高性能、可维护的 Go 程序。
