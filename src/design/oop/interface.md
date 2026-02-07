---
title: 接口与抽象
order: 3
---

# 接口与抽象

接口是面向对象设计的核心工具，它定义了一组行为的契约而不关心具体实现。接口将"是什么"与"怎么做"分离，是解耦、可测试性和可扩展性的基础。

## 接口与抽象类的区别

接口和抽象类都用于定义抽象，但它们在设计意图和使用场景上有本质区别。

| 维度 | 接口 | 抽象类 |
|------|------|--------|
| **继承关系** | 类可以实现多个接口 | 类只能继承一个抽象类（单继承） |
| **状态** | 不能包含实例字段 | 可以包含实例字段和状态 |
| **构造方法** | 没有构造方法 | 可以有构造方法供子类调用 |
| **方法修饰符** | 默认 public，不能有 private/protected | 可以有各种访问修饰符 |
| **设计意图** | 定义"能力"或"行为契约" | 定义"is-a"关系和代码复用 |
| **演变成本** | 添加新方法会破坏所有实现类 | 添加新方法有默认实现不影响子类 |

**使用接口的场景**：定义跨越不同类型层级的行为契约。如 `Comparable` 表示"可比较的"，`Serializable` 表示"可序列化的"，`Flyable` 表示"会飞的"。实现接口的类之间可以没有任何继承关系。

**使用抽象类的场景**：在相关类之间共享代码和状态。如 `AbstractList` 提供了列表操作的基础实现，`ArrayList` 和 `LinkedList` 继承它复用代码。抽象类适合表示"is-a"关系的部分实现。

```java
// 接口：定义行为契约，跨越不同类型层级
interface Flyable {
    void fly();
}

class Bird implements Flyable {
    public void fly() { /* 鸟的飞行方式 */ }
}

class Airplane implements Flyable {
    public void fly() { /* 飞机的飞行方式 */ }
}

// 抽象类：共享代码和状态，表示 is-a 关系
abstract class Animal {
    protected String name;
    protected int age;

    public Animal(String name, int age) {
        this.name = name;
        this.age = age;
    }

    // 抽象方法，子类必须实现
    abstract void makeSound();

    // 具体方法，子类可以直接使用或覆盖
    void sleep() {
        System.out.println(name + " is sleeping");
    }
}

class Dog extends Animal {
    void makeSound() {
        System.out.println("Woof!");
    }
}
```

## 如何设计良好的接口

接口设计需要遵循一些原则，以确保接口既不过于庞大也不过于细碎。

**接口隔离原则**：不要强迫客户端依赖它不需要的接口。一个接口应该只包含一组相关的方法，而不是把所有方法都塞进一个大接口。

```java
// 错误：胖接口
interface UserService {
    void login(String email, String password);
    void register(String email, String password);
    void resetPassword(String email);
    void addPoints(String userId, int points);
    void deductPoints(String userId, int points);
    void getOrders(String userId);
}

// 正确：拆分成多个小接口
interface AuthService {
    void login(String email, String password);
    void register(String email, String password);
    void resetPassword(String email);
}

interface PointsService {
    void addPoints(String userId, int points);
    void deductPoints(String userId, int points);
}

interface OrderService {
    void getOrders(String userId);
}
```

**接口命名**：接口名称应该使用形容词或能力描述，如 `Runnable`、`Comparable`、`Serializable`，或者使用名词形式加上 `I` 前缀（.NET 风格）如 `IUserService`。避免使用过于笼统的名称如 `Manager`、`Handler`。

**方法数量**：一个接口的方法数量应该控制在 5 个以内。过多的方法意味着接口承担了过多职责，违反了单一职责原则。如果接口方法超过 7 个，应该考虑拆分。

**接口演化**：接口一旦发布就不应该轻易修改，因为所有实现类都需要同步更新。如果必须演化，可以创建新接口（如 `Iterable` 演化出 `Collection`），或者使用默认方法（Java 8+）提供向后兼容的实现。

## 接口在解耦和可测试性中的作用

接口是实现依赖倒置的核心工具。高层模块依赖接口而非具体实现，低层模块实现接口，通过依赖注入将两者组装起来。

```java
// 业务逻辑依赖接口
class OrderService {
    private final Database db;
    private final PaymentGateway payment;
    private final EmailService email;

    public OrderService(Database db, PaymentGateway payment, EmailService email) {
        this.db = db;
        this.payment = payment;
        this.email = email;
    }

    public void processOrder(Order order) {
        db.save(order);
        payment.charge(order.getAmount());
        email.sendConfirmation(order.getEmail());
    }
}

// 单元测试时可以注入 mock 实现
class OrderServiceTest {
    void testProcessOrder() {
        MockDatabase mockDb = new MockDatabase();
        MockPaymentGateway mockPayment = new MockPaymentGateway();
        MockEmailService mockEmail = new MockEmailService();

        OrderService service = new OrderService(mockDb, mockPayment, mockEmail);
        service.processOrder(new Order());

        assertTrue(mockDb.wasSaved());
        assertTrue(mockPayment.wasCharged());
        assertTrue(mockEmail.wasSent());
    }
}
```

这种设计的价值在于：可以独立替换任何依赖实现，可以轻松进行单元测试，系统各部分可以并行开发和测试。

## 继承与组合

继承是 OOP 的核心机制，但滥用继承会导致紧耦合和脆弱基类问题。组合优于继承是现代面向对象设计的共识。

### 继承的问题

**脆弱基类问题**：父类的修改会波及所有子类。当修改父类的某个方法时，可能破坏某些子类的行为假设，导致难以预测的 bug。

**继承层次爆炸**：为了复用代码而建立深层继承层次，导致代码难以理解和维护。三层以上的继承层次就需要警惕。

**继承是编译时绑定**：继承关系在编译时确定，无法在运行时改变。如果需要动态改变对象行为，继承就不合适。

```java
// 问题：深层继承层次
class Animal { }
class Mammal extends Animal { }
class Dog extends Mammal { }
class GoldenRetriever extends Dog { }
class LabGoldenRetriever extends GoldenRetriever { }
// 每一层都可能引入脆弱性
```

### 组合的优势

**灵活性**：组合可以在运行时动态改变对象行为，通过替换不同的组件对象。

**低耦合**：组合的对象之间通过接口交互，耦合度远低于继承。

**更好的封装**：组合的内部实现细节对外部不可见，而继承会将父类的实现细节暴露给子类。

```java
// 使用组合替代继承
interface FlyBehavior {
    void fly();
}

class FlyWithWings implements FlyBehavior {
    public void fly() { System.out.println("用翅膀飞行"); }
}

class FlyNoWay implements FlyBehavior {
    public void fly() { System.out.println("不会飞行"); }
}

class Bird {
    private FlyBehavior flyBehavior;

    public Bird(FlyBehavior flyBehavior) {
        this.flyBehavior = flyBehavior;
    }

    public void performFly() {
        flyBehavior.fly();
    }

    public void setFlyBehavior(FlyBehavior flyBehavior) {
        this.flyBehavior = flyBehavior;
    }
}

// 可以在运行时改变行为
Bird duck = new Bird(new FlyWithWings());
duck.performFly();  // 用翅膀飞行
duck.setFlyBehavior(new FlyNoWay());
duck.performFly();  // 不会飞行
```

### 何时使用继承

继承并非一无是处，以下场景是继承的合理使用：

明确的 **is-a** 关系且子类在所有上下文中都可以替换父类（满足里氏替换原则）。如 `Dog` is-a `Animal`，`Circle` is-a `Shape`。

需要共享核心实现和状态。如 `AbstractList` 提供了列表操作的基础实现，子类只需实现几个核心方法。

框架设计的扩展点。如 `Spring MVC` 的 `Controller`、`Servlet` 的 `HttpServlet`，框架通过继承提供模板方法。

```java
// 合理的继承：明确的 is-a 关系
abstract class Shape {
    abstract double area();
    abstract double perimeter();
}

class Circle extends Shape {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    double area() {
        return Math.PI * radius * radius;
    }

    double perimeter() {
        return 2 * Math.PI * radius;
    }
}

// 圆形在任何可以用形状的地方都可以使用
void printShapeInfo(Shape shape) {
    System.out.println("面积: " + shape.area());
    System.out.println("周长: " + shape.perimeter());
}
```

### 多重继承的替代方案

C++ 支持多重继承但带来菱形继承问题，Java 只允许单继承。现代语言提供了多种多重继承的替代方案：

**接口**：可以实现多个接口，获得多重继承的行为定义能力，但无法共享实现。

**默认方法**：Java 8+ 的接口默认方法提供了一定程度的实现共享，但仍然不能包含状态。

**Mixin/Trait**：Scala 的 trait、Ruby 的 module、Rust 的 trait 提供了真正的多重继承能力，可以包含方法和状态，通过线性化解决冲突。

**组合**：最通用的方案，通过组合多个对象获得多重能力，这是组合优于继承原则的体现。

```java
// 使用组合实现多重能力
interface Reader {
    void read();
}

interface Writer {
    void write();
}

class FileReader implements Reader {
    public void read() { System.out.println("从文件读取"); }
}

class FileWriter implements Writer {
    public void write() { System.out.println("写入文件"); }
}

// 通过组合获得读写能力
class FileHandler {
    private Reader reader;
    private Writer writer;

    public FileHandler(Reader reader, Writer writer) {
        this.reader = reader;
        this.writer = writer;
    }

    public void read() { reader.read(); }
    public void write() { writer.write(); }
}
```

## 抽象的层次

抽象不是越抽象越好，需要在抽象和具体之间找到合适的平衡点。

**过度抽象的问题**：为了抽象而抽象，引入不必要的接口和抽象类，增加系统复杂度。当系统中每个类都有对应的接口时，就是过度抽象的信号。

**抽象不足的问题**：具体实现散落在各处，无法统一处理，难以扩展。当发现需要在多处添加相同的 `if-else` 判断类型时，就是抽象不足的信号。

**正确的抽象层次**：抽象应该由需求驱动，而非预先设计。当发现两个或多个类有共同的行为契约时，提取接口；当发现多个类有重复的实现时，提取抽象类或组合对象；当不确定是否需要抽象时，先不抽象，等具体需求出现后再重构。
