# .NET

.NET 运行时 (Runtime) 是 C# 程序的执行环境，负责内存管理、类型安全、异常处理、GC、线程调度等核心功能。理解运行时的工作原理对于编写高性能、可靠的 C# 代码至关重要。

## 运行时架构

.NET 运行时由几个核心组件组成。CLR (Common Language Runtime) 是运行时的核心，提供内存管理、类型安全、JIT 编译等功能。BCL (Base Class Library) 提供了丰富的类库，包括文件 I/O、网络通信、数据结构、集合等。

```
应用程序代码
    ↓
C# 编译器 (Roslyn)
    ↓
IL 代码 (元数据 + MSIL)
    ↓
.NET 运行时 (CLR)
    ├── JIT 编译器
    ├── GC (垃圾回收)
    ├── 异常处理
    ├── 线程池
    └── 类型系统
    ↓
本地机器码
```

## JIT 编译

C# 代码编译为 IL (Intermediate Language) 而非本地机器码。IL 是一种栈式的中间语言，类似于 Java 字节码。程序运行时，CLR 的 JIT (Just-In-Time) 编译器将 IL 编译为本地机器码。

JIT 编译是按需进行的，方法首次调用时才被编译。编译后的本地码会被缓存，后续调用直接使用缓存。这种方式称为分层编译 (Tiered Compilation)，可以平衡启动速度和运行性能。

```csharp
// C# 代码
int Add(int a, int b)
{
    return a + b;
}

// 编译后的 IL (简化)
IL_0000: ldarg.0
IL_0001: ldarg.1
IL_0002: add
IL_0003: ret

// JIT 编译后的机器码 (x86-64)
mov eax, ecx
add eax, edx
ret
```

分层编译从快速编译开始，如果方法被频繁调用，JIT 会使用更激进的优化策略重新编译。这种自适应优化使得 .NET 程序既有快速启动速度，又有接近本地代码的运行性能。

## AOT 编译

.NET 6 引入了 AOT (Ahead-Of-Time) 编译，可以在编译时将 IL 编译为本地代码。AOT 编译的程序启动更快，但失去了 JIT 的运行时优化能力。

```bash
# 发布为 AOT 程序
dotnet publish -c Release -r win-x64 --self-contained /p:PublishAot=true
```

AOT 的优势包括更快的启动速度、更小的内存占用、更利于 iOS 等平台的发布（不允许 JIT）。劣势是失去了运行时优化、反射受限、程序体积更大（因为需要携带运行时）。

## 垃圾回收

GC (Garbage Collection) 是 .NET 内存管理的核心机制。GC 自动回收不再使用的对象，开发者无需手动释放内存。GC 基于世代假设：新分配的对象很可能很快不再使用（短命对象），存活时间长的对象很可能继续存活（长命对象）。

GC 堆分为三代 (Generation 0、1、2)。新对象分配在 Gen 0，Gen 0 满了触发 GC，存活对象晋升到 Gen 1。Gen 1 满了触发 GC，存活对象晋升到 Gen 2。Gen 2 是满触发完整 GC。

```csharp
// 对象生命周期
void ProcessData()
{
    // temp 是短命对象，分配在 Gen 0
    var temp = new MemoryStream();

    // 使用 temp...

    // 方法结束，temp 不再被引用
    // 下次 GC 时会被回收
}
```

大对象堆 (LOH) 单独管理大于 85KB 的对象，大对象直接分配在 Gen 2。大对象很少移动，压缩开销太大。

GC 分为工作站 GC 和服务器 GC。工作站 GC 是默认模式，适合单机应用程序。服务器 GC 针对多核优化，使用多个 GC 线程并行回收，适合服务器应用。

## IDisposable 和 using 语句

GC 只管理托管内存，非托管资源 (文件句柄、数据库连接、网络连接) 需要手动释放。IDisposable 接口定义了 Dispose 方法，用于释放非托管资源。

```csharp
// 实现 IDisposable
class FileStream : IDisposable
{
    private IntPtr handle;
    private bool disposed = false;

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposed)
        {
            if (disposing)
            {
                // 释放托管资源
            }
            // 释放非托管资源
            CloseHandle(handle);
            disposed = true;
        }
    }

    ~FileStream()
    {
        Dispose(false);
    }
}

// 使用 using 语句确保资源释放
using (var file = new FileStream("test.txt", FileMode.Open))
{
    // 使用文件
} // 自动调用 Dispose()
```

C# 8.0 引入了 using 声明，可以在变量声明时直接使用。

```csharp
// using 声明
using var file = new FileStream("test.txt", FileMode.Open);
// 使用文件...
// 作用域结束时自动 Dispose
```

## 异常处理

.NET 的异常处理基于 try-catch-finally 语句。异常是对象，继承自 System.Exception。异常发生时，CLR 会展开调用栈，寻找匹配的 catch 块。

```csharp
try
{
    // 可能抛出异常的代码
    int x = 0;
    int y = 10 / x;
}
catch (DivideByZeroException ex) when (x == 0)  // 异常过滤器
{
    Console.WriteLine("Division by zero");
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
finally
{
    // 无论是否异常都执行
    Cleanup();
}
```

异常应该只用于异常情况，不应该用于正常控制流。抛出异常有性能开销，且会破坏代码的可读性。对于可预见的错误，应该使用返回值或 `Result<T>` 模式。

## 线程池

.NET 提供了线程池，避免频繁创建和销毁线程的开销。线程池维护一定数量的线程，当需要异步操作时，从线程池获取线程；操作完成后，线程返回线程池。

```csharp
// 使用线程池
ThreadPool.QueueUserWorkItem(state => {
    Console.WriteLine($"Working on {state}");
}, "Task 1");

// Task.Run 内部使用线程池
await Task.Run(() => {
    Console.WriteLine("Background work");
});
```

线程池根据系统负载动态调整线程数量，避免过多线程导致的上下文切换开销。线程池的工作窃取算法使得工作负载在多个线程间自动平衡。

## 应用程序域

AppDomain 是 .NET 的隔离边界，类似于操作系统中的进程。一个进程可以包含多个 AppDomain，每个 AppDomain 有独立的加载器堆和安全边界。AppDomain 可以独立加载和卸载程序集。

```csharp
// 创建新的 AppDomain
AppDomain domain = AppDomain.CreateDomain("NewDomain");

// 在新 AppDomain 中执行代码
domain.DoCallBack(() => {
    Console.WriteLine("Running in new domain");
});

// 卸载 AppDomain
AppDomain.Unload(domain);
```

AppDomain 在现代 .NET Core/.NET 5+ 中被简化，AssemblyLoadContext 取代了 AppDomain 的部分功能。

## Native AOT

.NET 7 引入了 Native AOT，可以完全静态编译为本地代码，不依赖运行时。Native AOT 使用 ILTrimmer 技术裁剪未使用的代码，生成小型、快速的本地可执行文件。

```bash
# Native AOT 发布
dotnet publish -c Release -r win-x64 /p:NativeAot=true
```

Native AOT 的优势包括极快的启动速度、更小的部署体积、更好的隔离性。限制包括不支持反射、动态加载、部分运行时特性。

## 运行时选择

.NET 提供多个运行时选择。.NET Framework 是 Windows 专用的旧版本，已停止更新。.NET Core 是跨平台的新版本，现在统一称为 .NET。Mono 是开源的 .NET 实现，主要用于移动端和 Unity。

```bash
# 查看已安装的运行时
dotnet --list-runtimes

# 查看已安装的 SDK
dotnet --list-sdks
```

不同的运行时在 API、性能、功能上有差异，选择时需要考虑目标平台和功能需求。
