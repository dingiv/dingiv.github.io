---
title: 对象生命周期
order: 5
---

# 对象生命周期

对象从创建到销毁的整个过程构成了对象的生命周期。理解对象的生命周期对于编写正确、高效的代码至关重要。

## 对象的创建

对象的创建是生命周期的起点，涉及内存分配和初始化两个步骤。在不同编程语言中，对象创建的机制有所差异。

**构造过程**：大多数面向对象语言通过构造函数创建对象。构造函数负责分配内存和初始化对象状态。构造函数可以重载，提供不同的初始化方式；可以调用父类构造函数，确保父类部分正确初始化；可以初始化成员变量，建立对象的初始状态。

```java
class User {
    private String name;
    private int age;

    // 无参构造
    public User() {
        this.name = "";
        this.age = 0;
    }

    // 带参构造
    public User(String name, int age) {
        this.name = name;
        this.age = age;
    }

    // 拷贝构造
    public User(User other) {
        this.name = other.name;
        this.age = other.age;
    }
}
```

**构造时的注意事项**：构造函数中不应该调用可被重写的方法，因为此时子类部分尚未初始化，可能导致子类方法访问未初始化的字段。构造函数应该简单高效，避免执行耗时操作或可能失败的操作。复杂的初始化逻辑应该提取到单独的初始化方法中。

**对象池模式**：对于创建成本高、使用频繁的对象，可以使用对象池来复用对象，避免重复创建销毁的开销。数据库连接池、线程池是典型的应用场景。对象池需要管理对象的借出、归还、清理等逻辑，使用时要注意对象的复位和状态一致性。

## 对象的使用

对象创建后进入使用阶段，这是对象生命周期中最长的部分。在使用阶段，对象的状态会通过方法调用被修改和读取。

**对象状态的管理**：良好的对象设计要求对象始终保持一致的状态。任何公共方法执行后，对象应该处于一个有效的状态。不变量是对象状态必须满足的条件，应该在构造时建立、在方法执行中维护。

```java
class BankAccount {
    private double balance;

    public BankAccount(double initialBalance) {
        if (initialBalance < 0) {
            throw new IllegalArgumentException("初始余额不能为负");
        }
        this.balance = initialBalance;
    }

    public void withdraw(double amount) {
        if (amount <= 0) {
            throw new IllegalArgumentException("取款金额必须为正");
        }
        if (amount > balance) {
            throw new IllegalStateException("余额不足");
        }
        balance -= amount;
        // 方法执行后，对象仍然处于有效状态（余额非负）
    }
}
```

**不可变对象**：不可变对象创建后状态不再改变，具有线程安全、易于缓存、简化理解等优势。设计不可变对象的关键是所有字段都是 final 的，不提供任何修改状态的方法，确保可变组件的防御性拷贝。`String`、`Integer`、`LocalDate` 是典型的不可变对象。

## 对象的销毁

对象的销毁释放对象占用的资源，包括内存和其他系统资源。销毁机制取决于语言的内存管理策略。

**垃圾回收**：Java、Python、C# 等语言使用自动垃圾回收，开发者无需手动释放内存。垃圾回收器通过引用计数、标记清除、复制算法、分代收集等策略识别不再使用的对象并回收其内存。

**垃圾回收的权衡**：自动垃圾回收简化了开发，但也带来了不确定性。垃圾回收的时机不可预测，可能导致性能抖动。对于需要精细控制资源的场景，如实时系统、嵌入式系统，手动内存管理可能更合适。

```java
// 垃圾回收触发时机示例
public class GCDemo {
    public static void main(String[] args) {
        for (int i = 0; i < 1000000; i++) {
            byte[] data = new byte[1024 * 1024]; // 分配 1MB
            // 旧对象逐渐变成垃圾
            if (i % 10000 == 0) {
                System.gc(); // 建议执行垃圾回收（不保证立即执行）
            }
        }
    }
}
```

**手动内存管理**：C++ 等语言需要手动释放内存，使用 `delete` 或 `free` 回收对象。手动管理提供了精确的控制，但也带来了内存泄漏和悬空指针的风险。现代 C++ 倾向于使用智能指针（`shared_ptr`、`unique_ptr`）自动管理对象生命周期。

**资源释放**：除了内存，对象可能持有其他资源如文件句柄、数据库连接、网络连接。这些资源应该及时释放，避免资源耗尽。实现 `AutoCloseable`（Java）、`IDisposable`（C#）接口或使用 `try-with-resources` 语句可以确保资源的正确释放。

```java
// 使用 try-with-resources 确保资源释放
try (Connection conn = dataSource.getConnection();
     PreparedStatement stmt = conn.prepareStatement(sql)) {
    // 使用资源
} // 自动调用 close() 方法释放资源
```

## 作用域与生命周期

作用域决定了对象的可访问范围，生命周期决定了对象的存活时间。理解两者的关系对于正确使用对象至关重要。

**局部变量的生命周期**：局部变量的生命周期限于方法执行期间，存储在栈上，方法返回后自动释放。局部变量应该是短命的，不应该存储到长期存活的对象中，否则会导致意外的内存保留。

```java
// 错误：局部变量逃逸
class UserController {
    private List<String> names; // 实例变量，长期存活

    void processData() {
        List<String> tempNames = new ArrayList<>(); // 局部变量，应该短命
        tempNames.add("Alice");
        tempNames.add("Bob");

        this.names = tempNames; // 局部变量逃逸到实例变量
    }
}
```

**实例变量的生命周期**：实例变量的生命周期与对象绑定，对象存活期间实例变量一直存在。实例变量应该表示对象的固有状态，而不是临时数据或计算结果。

**静态变量的生命周期**：静态变量在类加载时创建，在类卸载时销毁，生命周期贯穿整个应用程序运行期间。静态变量应该谨慎使用，因为它会长期占用内存，并且可能成为并发访问的瓶颈。

**对象作用域与生命周期的不匹配**：当短命的对象被长命的对象引用时，短命对象的生命周期被延长，可能导致内存泄漏。典型的场景包括缓存未设置过期时间、监听器未正确注销、集合中积累了不再使用的对象。

```java
// 内存泄漏示例
class EventManager {
    private List<EventListener> listeners = new ArrayList<>();

    public void addListener(EventListener listener) {
        listeners.add(listener);
    }

    // 问题：没有提供 removeListener 方法
    // 监听器对象无法被回收，导致内存泄漏
}
```

## 生命周期管理的最佳实践

**及时释放资源**：使用完资源后立即释放，不要等待垃圾回收。对于实现了 `AutoCloseable` 或 `IDisposable` 的对象，应该使用 `try-with-resources` 或 `using` 语句确保资源的释放。

**避免过早创建**：对象应该在需要时才创建，而不是预先创建大量可能用不到的对象。延迟初始化是一种有效的优化手段，但对于频繁访问的对象，延迟初始化可能增加同步开销。

**限制作用域**：变量的作用域应该尽可能小，最小的作用域使得代码更易理解、更难出错。局部变量优于实例变量，实例变量优于静态变量。

```java
// 好的做法：作用域最小化
public void processOrder(Order order) {
    // validator 的作用域限于这个方法
    OrderValidator validator = new OrderValidator();
    validator.validate(order);

    // calculator 的作用域限于这个 if 块
    if (order.needsCalculation()) {
        PriceCalculator calculator = new PriceCalculator();
        calculator.calculate(order);
    }
}
```

**使用弱引用**：当需要引用对象但不希望阻止其被垃圾回收时，可以使用弱引用（`WeakReference`）。缓存、监听器注册表等场景适合使用弱引用。

**对象池的使用**：对象池适用于创建成本高、使用频率高的对象。但对象池也有开销，对于轻量对象，直接创建可能比对象池更高效。对象池需要正确处理对象的复位和并发访问。

**生命周期与架构设计**：在分层架构中，不同层的对象有不同的生命周期。Controller 层的对象生命周期限于请求范围，Service 层的对象可能是单例的，Repository 层的对象需要管理数据库连接等资源。理解每层对象的生命周期有助于设计清晰的架构。


//TODO: RAII 资源管理