# 异步编程

异步编程是 C# 的核心特性之一，使得程序可以在等待耗时操作完成时执行其他工作，提高应用程序的响应性和吞吐量。C# 5.0 引入了 async 和 await 关键字，彻底改变了异步编程的方式。

## 异步模式演进

.NET 中的异步编程经历了三种模式的演进。早期使用 APM (Asynchronous Programming Model)，基于 IAsyncResult 接口的 Begin/End 模式。后来使用 EAP (Event-based Asynchronous Pattern)，基于事件的异步模式。现在使用 TAP (Task-based Asynchronous Pattern)，基于任务的异步模式。

```csharp
// APM 模式（过时）
Stream stream = File.OpenRead("file.txt");
byte[] buffer = new byte[stream.Length];
IAsyncResult result = stream.BeginRead(buffer, 0, buffer.Length, null, null);
int bytesRead = stream.EndRead(result);

// EAP 模式（过时）
var client = new WebClient();
client.DownloadStringCompleted += (s, e) => {
    Console.WriteLine(e.Result);
};
client.DownloadStringAsync("https://example.com");

// TAP 模式（推荐）
async Task DownloadAsync()
{
    var client = new HttpClient();
    string content = await client.GetStringAsync("https://example.com");
    Console.WriteLine(content);
}
```

## Task 和 `Task<T>`

Task 表示一个异步操作，可以返回值 (`Task<T>`) 或不返回值 (Task)。Task 本质上是对 Future 或 Promise 概念的实现，代表了尚未完成的操作。

```csharp
// 创建 Task
Task task1 = Task.Run(() => Console.WriteLine("Hello"));
Task task2 = new Task(() => Console.WriteLine("World"));
task2.Start();

// Task<T> 获取返回值
Task<int> task = Task.Run(() => {
    Thread.Sleep(1000);
    return 42;
});
int result = await task;

// 组合多个 Task
Task task3 = Task.Run(() => DoWork());
Task task4 = Task.Run(() => DoMoreWork());
await Task.WhenAll(task3, task4);

// 等待任意一个完成
Task<int> task5 = Task.Run(() => Compute1());
Task<int> task6 = Task.Run(() => Compute2());
int firstResult = await await Task.WhenAny(task5, task6);
```

## async 和 await

async 关键字标记一个方法为异步方法，await 关键字等待异步操作完成。async 方法必须有返回类型 `Task、Task<T>、ValueTask、ValueTask<T>` 或 void (仅用于事件处理器)。

```cs
// 异步方法定义
async Task<string> GetDataAsync(string url)
{
    var client = new HttpClient();
    string content = await client.GetStringAsync(url);
    return content.ToUpper();
}

// 调用异步方法
string result = await GetDataAsync("https://example.com");
```

async 方法的执行流程：遇到 await 时，方法会暂停并返回到调用者，await 后面的代码在 await 的操作完成后继续执行。这个过程中，编译器会将 async 方法重写为状态机，自动处理回调的注册和调用。

```csharp
// 编译前
async Task ProcessAsync()
{
    Console.WriteLine("Start");
    await Task.Delay(1000);
    Console.WriteLine("End");
}

// 编译后的伪代码
class ProcessAsyncStateMachine : IAsyncStateMachine
{
    private int state;
    private TaskAwaiter awaiter;

    public void MoveNext()
    {
        switch (state)
        {
            case 0:
                Console.WriteLine("Start");
                awaiter = Task.Delay(1000).GetAwaiter();
                if (!awaiter.IsCompleted)
                {
                    state = 1;
                    awaiter.OnCompleted(this.MoveNext);
                    return;
                }
                break;
            case 1:
                break;
        }
        Console.WriteLine("End");
    }
}
```

## ValueTask

ValueTask 是值类型的 Task，用于避免异步操作已经完成时的内存分配。对于高频调用的异步方法，使用 ValueTask 可以减少 GC 压力。

```csharp
// 使用 ValueTask
async ValueTask<int> GetValueAsync()
{
    // 如果值已经缓存，不需要分配 Task
    if (_cached)
        return new ValueTask<int>(_cachedValue);

    return new ValueTask<int>(LoadValueAsync());
}

// await ValueTask 与 await Task 相同
int value = await GetValueAsync();
```

## 异步流

C# 8.0 引入了异步流，支持 `IAsyncEnumerable<T>` 接口，允许异步地枚举数据序列。

```csharp
// 异步流方法
async IAsyncEnumerable<int> GenerateSequenceAsync()
{
    for (int i = 0; i < 10; i++)
    {
        await Task.Delay(100);
        yield return i;
    }
}

// 消费异步流
await foreach (var item in GenerateSequenceAsync())
{
    Console.WriteLine(item);
}
```

## 取消异步操作

CancellationToken 用于取消异步操作。调用方创建 CancellationTokenSource，将 Token 传递给异步方法，当需要取消时调用 Cancel()。

```csharp
// 可取消的异步方法
async Task DownloadAsync(string url, CancellationToken cancellationToken)
{
    var client = new HttpClient();
    var response = await client.GetAsync(url, cancellationToken);
    var content = await response.Content.ReadAsStringAsync(cancellationToken);
    Console.WriteLine(content);
}

// 调用时支持取消
var cts = new CancellationTokenSource();
var task = DownloadAsync("https://example.com", cts.Token);

// 需要时取消
cts.Cancel();

try
{
    await task;
}
catch (OperationCanceledException)
{
    Console.WriteLine("Operation canceled");
}
```

## 异步最佳实践

异步编程有一些需要注意的地方。避免使用 async void（除非是事件处理器），async void 方法的异常无法被捕获，会导致应用程序崩溃。

```csharp
// 错误：async void
async void Button_Click(object sender, EventArgs e)
{
    // 这里的异常会直接抛出到线程池，无法被捕获
    await Task.Delay(1000);
}

// 正确：async Task
async Task Button_ClickAsync(object sender, EventArgs e)
{
    try
    {
        await Task.Delay(1000);
    }
    catch (Exception ex)
    {
        // 异常可以被捕获
        LogError(ex);
    }
}
```

避免在循环中使用 async lambda，应该使用 Task.WhenAll 或 Parallel.ForEachAsync。

```csharp
// 错误：async lambda
foreach (var url in urls)
{
    tasks.Add(Task.Run(async () => await DownloadAsync(url)));
}

// 正确：使用 WhenAll
var tasks = urls.Select(url => DownloadAsync(url));
await Task.WhenAll(tasks);
```

配置上下文捕获。ConfigureAwait(false) 告诉 await 不需要捕获原始上下文，可以在线程池线程上继续执行，减少死锁风险和提高性能。

```csharp
// 库代码应该使用 ConfigureAwait(false)
async Task ProcessAsync()
{
    var data = await DownloadAsync().ConfigureAwait(false);
    // 后续代码在线程池线程执行
    DoProcessing(data);
}
```

对于应用程序代码（如 ASP.NET Core Controller），不需要 ConfigureAwait(false)，因为 ASP.NET Core 不使用同步上下文。

## 异步 LINQ

System.Interactive.Async 提供了异步 LINQ 操作符，允许对异步流进行类似 LINQ 的操作。

```bash
dotnet add package System.Interactive.Async
```

```csharp
using System.Reactive.Linq;

// 异步 LINQ
var results = await urls
    .Select(url => DownloadAsync(url))
    .Where(content => content.Length > 1000)
    .ToArrayAsync();
```
