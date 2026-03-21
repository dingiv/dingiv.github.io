# C# 基础语法

C# 是一种强类型的静态语言，其语法借鉴了 C、C++ 和 Java，但引入了许多现代语言特性。理解 C# 的基础语法是掌握 .NET 开发的第一步。

## 类型系统

C# 的类型系统分为值类型和引用类型两大类。值类型直接存储数据，分配在栈上或内联在对象中；引用类型存储对象的引用，分配在托管堆上。

### 值类型

值类型包括基本数值类型、结构和枚举。基本数值类型分为整数类型和浮点类型。整数类型有 sbyte/byte（8 位）、short/ushort（16 位）、int/uint（32 位）、long/ulong（64 位）。浮点类型有 float（32 位单精度）、double（64 位双精度）、decimal（128 位高精度，适合财务计算）。

```csharp
// 数值类型
int age = 25;
double price = 19.99;
decimal amount = 1234.56m;  // m 后缀表示 decimal
bool isActive = true;
char grade = 'A';
```

结构是用户定义的值类型，可以包含字段、属性、方法。结构与类的区别在于结构是值类型，赋值时会复制整个数据，而类是引用类型，赋值时只复制引用。

```csharp
// 结构定义
public struct Point
{
    public int X;
    public int Y;

    public Point(int x, int y)
    {
        X = x;
        Y = y;
    }
}

// 使用
Point p1 = new Point(10, 20);
Point p2 = p1;  // 值复制，p2 是 p1 的副本
p2.X = 30;      // 不影响 p1
```

枚举是一组命名常量的集合，基于整数类型。

```csharp
// 枚举定义
enum DayOfWeek
{
    Sunday = 0,
    Monday = 1,
    Tuesday = 2,
    Wednesday = 3,
    Thursday = 4,
    Friday = 5,
    Saturday = 6
}

// 使用
DayOfWeek today = DayOfWeek.Wednesday;
```

### 引用类型

引用类型包括类、接口、委托和数组。类是最常见的引用类型，支持继承、多态等面向对象特性。

```csharp
// 类定义
public class Person
{
    public string Name { get; set; }
    public int Age { get; set; }

    public Person(string name, int age)
    {
        Name = name;
        Age = age;
    }

    public void SayHello()
    {
        Console.WriteLine($"Hello, I'm {Name}");
    }
}

// 使用
Person person1 = new Person("Alice", 25);
Person person2 = person1;  // 引用复制，person2 和 person1 指向同一对象
person2.Name = "Bob";      // person1.Name 也变为 "Bob"
```

字符串是特殊的引用类型，具有值类型的一些特性。字符串是不可变的，任何修改操作都会创建新的字符串对象。

## 变量与常量

C# 使用 var 关键字进行类型推断，编译器会根据初始化表达式推断变量类型。常量使用 const 关键字定义，必须在声明时初始化，且只能是编译时常量。

```csharp
// 显式类型声明
int count = 10;
string name = "Alice";

// 类型推断
var age = 25;              // 推断为 int
var message = "Hello";     // 推断为 string

// 常量
const double Pi = 3.14159;
const int MaxUsers = 100;

// 只读字段
readonly DateTime createdAt = DateTime.Now;
```

## 运算符

C# 的运算符与 C/C++ 类似，但有一些特殊运算符。is 运算符用于类型检查，as 运算符用于安全类型转换，?. 空条件运算符用于避免 NullReferenceException，?? 空合并运算符用于提供默认值。

```csharp
// 类型检查和转换
if (obj is string str)
{
    Console.WriteLine(str.Length);
}

// 安全转换
string result = obj as string;
if (result != null)
{
    Console.WriteLine(result.Length);
}

// 空条件运算符
int? length = text?.Length;

// 空合并运算符
string name = displayName ?? "Anonymous";

// 空合并赋值运算符 (C# 8.0)
name ??= "Default";
```

## 控制流

C# 的控制流语句包括 if-else、switch、for、foreach、while、do-while。switch 表达式在 C# 8.0 中得到增强，支持模式匹配和更简洁的语法。

```csharp
// 传统 switch 语句
switch (value)
{
    case 1:
        Console.WriteLine("One");
        break;
    case 2:
        Console.WriteLine("Two");
        break;
    default:
        Console.WriteLine("Other");
        break;
}

// Switch 表达式 (C# 8.0)
string message = value switch
{
    1 => "One",
    2 => "Two",
    _ => "Other"
};

// 模式匹配的 switch
string result = obj switch
{
    null => "null",
    string s => $"string: {s}",
    int i => $"int: {i}",
    _ => "unknown"
};
```

## 方法

方法使用返回类型、方法名和参数列表定义。参数可以有默认值，方法可以重载。C# 4.0 引入了可选参数和命名参数，C# 12.0 引入了主构造函数参数。

```csharp
// 方法定义
public int Add(int a, int b = 0)
{
    return a + b;
}

// 方法重载
public int Add(int a, int b, int c)
{
    return a + b + c;
}

// 命名参数调用
Add(b: 5, a: 3);

// params 参数
public int Sum(params int[] numbers)
{
    return numbers.Sum();
}

Sum(1, 2, 3, 4, 5);
```

## 属性

属性是字段的智能封装，提供 get 和 set 访问器。自动属性简化了属性的编写。C# 3.0 引入了自动属性，C# 6.0 引入了只读自动属性。

```csharp
// 传统属性
private string _name;
public string Name
{
    get { return _name; }
    set { _name = value; }
}

// 自动属性
public string Name { get; set; }

// 只读自动属性 (C# 6.0)
public string Name { get; }

// 属性初始化器 (C# 6.0)
public string Name { get; set; } = "Default";

// required 属性 (C# 11.0)
public required string Id { get; init; }
```

## 索引器

索引器允许对象像数组一样通过索引访问。

```csharp
public class SampleCollection<T>
{
    private T[] items = new T[100];

    public T this[int index]
    {
        get => items[index];
        set => items[index] = value;
    }
}

// 使用
var collection = new SampleCollection<int>();
collection[0] = 100;
int value = collection[0];
```

## 泛型

泛型允许类型参数化，提高代码复用性和类型安全。C# 的泛型与 C++ 的模板不同，泛型在编译时会被擦除为具体类型，但仍然保留类型信息用于反射。

```csharp
// 泛型类
public class Container<T>
{
    private T _value;

    public Container(T value)
    {
        _value = value;
    }

    public T GetValue() => _value;
}

// 泛型约束
public class Repository<T> where T : class
{
    public void Add(T item) { }
}

// 多个约束
public class DataManager<T>
    where T : class, new()
{
    public T Create() => new T();
}
```

## 命名空间

命名空间用于组织代码，避免命名冲突。using 语句导入命名空间，using static 语句导入类型的静态成员。

```csharp
// 命名空间定义
namespace MyApp.Models
{
    public class Person { }
}

// 导入命名空间
using System;
using System.Collections.Generic;
using static Math;  // 导入静态成员

// 别名
using Project = MyApp.Models.Person;
```
