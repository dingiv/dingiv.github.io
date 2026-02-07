---
title: 类间关系
order: 4
---

# 类间关系
面向对象设计的核心是设计类之间的关系。类与类之间的交互方式决定了系统的可维护性、可扩展性和可测试性。

## 六种基本关系

根据 UML 规范，类之间存在六种基本关系，按照耦合度从强到弱排列：**实现、泛化（继承）、组合、聚合、关联、依赖**。

| 关系类型 | UML 符号 | 耦合强度 | 生命周期绑定 | 含义 |
|---------|---------|---------|-------------|------|
| **实现** | 虚线空心三角 | 最强 | 无关 | 类实现接口 |
| **泛化** | 实线空心三角 | 最强 | 无关 | 继承，is-a 关系 |
| **组合** | 实线实心菱形 | 很强 | 绑定 | 强拥有的部分-整体关系 |
| **聚合** | 实线空心菱形 | 中等 | 独立 | 弱拥有的部分-整体关系 |
| **关联** | 实线箭头 | 较弱 | 独立 | 对象之间的引用关系 |
| **依赖** | 虚线箭头 | 最弱 | 无关 | 使用关系 |

### 依赖关系

依赖是耦合最弱的关系，表示一个类在某个时刻使用到了另一个类。典型场景包括：作为方法的参数、作为方法的返回值、作为局部变量使用。

```java
// Driver 依赖于 Car
class Driver {
    void drive(Car car) {  // Car 是参数，产生依赖
        car.start();
    }
}
```

依赖关系的特点是短暂性和局部性，两个类的生命周期完全独立，一个类的变化对另一个类影响最小。

### 关联关系

关联表示对象之间持久的引用关系，通常表现为类的成员变量。关联可以是单向的，也可以是双向的。

```java
// Teacher 与 Student 是双向关联
class Teacher {
    private List<Student> students;  // 持有 Student 的引用
}

class Student {
    private Teacher teacher;  // 持有 Teacher 的引用
}
```

关联关系的特征是"长期持有"，一个类会长期保存对另一个类的引用，但两个对象仍然可以独立存在。

### 聚合关系

聚合是一种特殊的关联，表示**弱拥有的**部分-整体关系。整体对象持有部分对象的引用，但部分对象可以脱离整体而独立存在。

```java
// 班级聚合学生，学生可以独立于班级存在
class Class {
    private List<Student> students;  // 聚合关系

    public void addStudent(Student student) {
        students.add(student);
    }
}
```

聚合在语义上表达"has-a"关系，但"部分"的生命周期不依赖"整体"。学生可以转班，可以毕业，不影响其作为独立实体的存在。

### 组合关系

组合是一种更强的聚合，表示**强拥有的**部分-整体关系。部分对象的生命周期完全由整体对象控制，整体销毁时部分也随之销毁。

```java
// 人组合心脏，心脏不能脱离人独立存在
class Person {
    private Heart heart;  // 组合关系

    public Person() {
        heart = new Heart();  // 必须在构造时创建
    }
}

// 文档组合段落，段落不能脱离文档存在
class Document {
    private List<Paragraph> paragraphs;

    public Document() {
        paragraphs = new ArrayList<>();
    }
}
```

组合是耦合最强的结构性关系，部分对象通常在整体对象内部创建，不能被外部替换或共享。

### 泛化关系

泛化即继承，表示"是一个"（is-a）的关系。子类继承父类的所有属性和行为，并可以扩展或重写。

```java
// Dog is-a Animal
class Animal {
    void eat() {}
}

class Dog extends Animal {
    void bark() {}
}
```

继承是面向对象的核心机制之一，但过度使用会导致脆弱基类问题，应当遵循里氏替换原则。

### 实现关系

实现表示类实现接口的所有方法，是对行为的抽象。

```java
interface Flyable {
    void fly();
}

class Bird implements Flyable {
    public void fly() {
        // 实现飞行行为
    }
}
```

实现关系是实现多态和依赖倒置的基础，面向接口编程正是基于这一关系。

## 多重性表示

在 UML 类图中，多重性表示一个类与多少个另一个类的对象发生关系。

| 符号 | 含义 |
|------|------|
| `1` | 恰好一个 |
| `0..1` | 零个或一个 |
| `*` 或 `0..*` | 零个或多个 |
| `1..*` | 一个或多个 |
| `n` | 恰好 n 个 |
| `n..m` | n 到 m 个 |

```java
// 一个订单包含多个订单项，一个订单项属于一个订单
class Order {
    private List<OrderItem> items;  // 1 对 *
}

// 一个部门有多个员工，一个员工属于一个部门
class Department {
    private List<Employee> employees;  // 1 对 *
}
```

## 组合优于继承

面向对象设计中有一个重要原则：**组合优于继承**。继承会导致强耦合和脆弱基类问题，而组合提供了更大的灵活性。

### 继承的问题

```java
// 问题场景：继承的脆弱性
class Bird {
    void fly() { }
}

class Penguin extends Bird {
    @Override
    void fly() {
        throw new UnsupportedOperationException("企鹅不会飞");
    }
}
```

企鹅是鸟，但不会飞。这种情况下继承就会产生语义矛盾，违反了里氏替换原则。

### 组合的解决方案

```java
// 使用组合解决
interface FlyBehavior {
    void fly();
}

class Bird {
    private FlyBehavior flyBehavior;

    public Bird(FlyBehavior flyBehavior) {
        this.flyBehavior = flyBehavior;
    }

    void performFly() {
        flyBehavior.fly();
    }
}

class Penguin extends Bird {
    public Penguin() {
        super(() -> { throw new UnsupportedOperationException("企鹅不会飞"); });
    }
}
```

这是策略模式的体现，通过组合不同的行为对象，可以在运行时动态改变对象的行为。

## 依赖倒置与面向接口编程

依赖倒置原则要求高层模块不依赖低层模块，两者都依赖抽象；抽象不依赖细节，细节依赖抽象。这要求我们面向接口编程而不是面向实现编程。

```java
// 错误：高层直接依赖低层实现
class OrderService {
    private MySQLDatabase db;  // 依赖具体实现

    void save(Order order) {
        db.save(order);
    }
}

// 正确：依赖抽象接口
interface Database {
    void save(Order order);
}

class OrderService {
    private Database db;  // 依赖抽象接口

    OrderService(Database db) {
        this.db = db;
    }
}
```

面向接口编程的另一个好处是便于测试，可以轻松注入 mock 对象进行单元测试。

## 里氏替换原则

里氏替换原则是继承关系的黄金法则：子类必须能够替换父类出现在父类能够出现的任何地方，而不破坏程序的正确性。

```java
// 违反里氏替换原则
class Rectangle {
    void setWidth(int width) { }
    void setHeight(int height) { }
    int getArea() { return 0; }
}

class Square extends Rectangle {
    @Override
    void setWidth(int width) {
        super.setWidth(width);
        super.setHeight(width);  // 正方形必须同时设置宽高
    }
}

// 客户端代码假设矩形可以独立设置宽高
void resize(Rectangle rect) {
    rect.setWidth(10);
    rect.setHeight(5);
    assert rect.getArea() == 50;  // 对 Square 会失败
}
```

正方形是矩形的一种数学定义，但在可变对象的设计中，正方形不能替换矩形。这说明继承关系的设计需要非常谨慎，必须考虑行为契约而非仅仅是语义关系。

## 关系设计的实践指南

- **优先使用依赖**：参数传递是最弱的耦合，优先考虑将依赖作为参数传入
- **谨慎使用继承**：只有在明确的"是一个"关系且符合里氏替换原则时才使用继承
- **多用组合**：组合提供了比继承更好的灵活性和可维护性
- **面向接口编程**：依赖抽象而非具体实现，便于替换和测试
- **控制关联数量**：一个类不应关联过多其他类，这通常意味着职责过重
- **区分聚合和组合**：根据语义和生命周期需求选择合适的关系类型
