---
title: 职责分配
---

# 职责分配原则
软件设计的核心问题之一是职责分配：谁该负责做什么。职责分配不当会导致代码耦合过紧、难以测试、难以维护。本文介绍三组密切相关的职责分配原则：迪米特法则（LoD）、告诉不要问（TDA）和 GRASP 模式。

## 迪米特法则
迪米特法则又称"最少知识原则"（Principle of Least Knowledge），核心观点是：一个对象应该对其他对象保持最少的了解，只与"直接的朋友"通信，不与"陌生人"谈话。

如果一个模块需要深入探索另一个模块的内部结构才能完成工作，那么任何内部细节的变动都会引发链式反应式的故障。通过减少这种横向依赖，系统变得更具容错性和可测试性。

**违反示例**

```java
// 违反迪米特法则
class Customer {
    private Wallet wallet;
    public Wallet getWallet() { return wallet; }
}

class Wallet {
    private Money money;
    public Money getMoney() { return money; }
}

class Money {
    private int amount;
    public int getAmount() { return amount; }
}

// 客户代码需要深入了解对象结构
public void checkout(Customer customer) {
    int amount = customer.getWallet().getMoney().getAmount();
    // 处理金额...
}
```

这种长链调用 `customer.getWallet().getMoney().getAmount()` 违反了迪米特法则，使客户端代码与 Wallet、Money 的内部实现紧密耦合。

**遵循示例**

```java
// 遵循迪米特法则
class Customer {
    private Wallet wallet;

    // 直接在 Customer 内部提供所需功能
    public boolean canPay(int amount) {
        return wallet.hasEnough(amount);
    }

    public void pay(int amount) {
        wallet.withdraw(amount);
    }
}

// 客户代码只与 Customer 交互
public void checkout(Customer customer, int amount) {
    if (customer.canPay(amount)) {
        customer.pay(amount);
    }
}
```

**判断"朋友"的标准**

- 对象本身（this）
- 对象的成员变量
- 对象的方法参数
- 对象创建的对象
- 对象的集合元素

除此之外的对象都是"陌生人"，不应直接交互。

## 告诉不要问
TDA 原则建议开发者不要通过询问对象的内部状态来做出决定，而是直接告诉对象该做什么。

对象的真正威力在于封装：将数据和操作数据的方法绑定在一起。TDA 要求我们将决策逻辑封装在对象内部，而不是暴露给外部代码。

**违反示例**

```java
// 违反 TDA：暴露内部状态，在外部做决策
class Account {
    private int balance;
    public int getBalance() { return balance; }
    public void setBalance(int balance) { this.balance = balance; }
}

// 客户代码需要知道业务规则
public void withdraw(Account account, int amount) {
    if (account.getBalance() >= amount) {
        account.setBalance(account.getBalance() - amount);
    } else {
        throw new InsufficientFundsException();
    }
}
```

这种写法暴露了 Account 的内部状态，并将业务逻辑（余额判断）泄露到客户端代码中。

**遵循示例**

```java
// 遵循 TDA：封装决策逻辑
class Account {
    private int balance;

    public void withdraw(int amount) {
        if (balance < amount) {
            throw new InsufficientFundsException();
        }
        balance -= amount;
    }
}

// 客户代码只负责调用
public void withdraw(Account account, int amount) {
    account.withdraw(amount);
}
```

### TDA 与 LoD 的关系

TDA 和 LoD 相辅相成：TDA 强调"不要问"，LoD 强调"不要深入"。两者共同指向一个目标——减少对象之间的耦合度，提高封装性。

## GRASP 模式

GRASP（General Responsibility Assignment Software Patterns）是一组用于分配职责的指导方针，回答"软件设计中最基本的问题：谁该负责什么"。

### 信息专家（Information Expert）

将职责分配给拥有完成该职责所需信息的类。

```java
// 好的设计：Order 包含计算总价所需的全部信息
class Order {
    private List<OrderItem> items;

    public Money calculateTotal() {
        Money total = new Money(0);
        for (OrderItem item : items) {
            total = total.add(item.getSubtotal());
        }
        return total;
    }
}

// 不好的设计：在别的类中计算
class OrderCalculator {
    public Money calculateTotal(Order order) {
        // 需要暴露 Order 的内部结构
    }
}
```

### 创造者（Creator）

如果类 A 满足以下条件之一，则 A 应该负责创建类 B 的实例：

- A 包含 B
- A 聚合 B
- A 拥有初始化 B 的数据
- A 记录 B 的实例

```java
class Order {
    // Order 包含 OrderItem，由 Order 负责创建
    public OrderItem addItem(Product product, int quantity) {
        OrderItem item = new OrderItem(product, quantity);
        items.add(item);
        return item;
    }
}
```

### 低耦合（Low Coupling）

分配职责时，应尽量减少类之间的耦合度。耦合度越低，系统越容易维护和修改。

```java
// 高耦合：直接依赖具体实现
class OrderProcessor {
    private DatabaseLogger logger = new DatabaseLogger();

    public void process(Order order) {
        logger.log(order);
    }
}

// 低耦合：依赖抽象
class OrderProcessor {
    private Logger logger;

    public OrderProcessor(Logger logger) {
        this.logger = logger;
    }

    public void process(Order order) {
        logger.log(order);
    }
}
```

### 高内聚（High Cohesion）

一个类应该只负责一组相关的功能，保持单一职责。高内聚使类更易于理解、复用和维护。

```java
// 低内聚：一个类承担多种无关职责
class OrderManager {
    public void saveOrder(Order order) { }
    public void sendEmail(Order order) { }
    public void calculateTax(Order order) { }
    public void printInvoice(Order order) { }
}

// 高内聚：职责分离
class OrderRepository {
    public void save(Order order) { }
}

class OrderNotifier {
    public void sendConfirmation(Order order) { }
}

class OrderService {
    private final OrderRepository repository;
    private final OrderNotifier notifier;

    public void placeOrder(Order order) {
        repository.save(order);
        notifier.sendConfirmation(order);
    }
}
```

### 控制器（Controller）

将处理系统事件的职责分配给一个非用户界面的类，这个类代表整个系统或用例。

```java
// 控制器：协调业务流程
class OrderController {
    private OrderService orderService;
    private PaymentService paymentService;
    private InventoryService inventoryService;

    public void placeOrder(PlaceOrderRequest request) {
        // 验证请求
        // 检查库存
        // 创建订单
        // 处理支付
        // 返回结果
    }
}
```

### 多态（Polymorphism）

根据行为类型的不同，将职责分配给各自的行为实现类，通过多态来处理变化。

```java
// 接口定义
interface PaymentStrategy {
    void pay(Money amount);
}

// 多种实现
class CreditCardPayment implements PaymentStrategy {
    public void pay(Money amount) { /* 信用卡支付逻辑 */ }
}

class WeChatPayment implements PaymentStrategy {
    public void pay(Money amount) { /* 微信支付逻辑 */ }
}

// 客户代码不需要知道具体实现
class OrderService {
    public void payOrder(Order order, PaymentStrategy strategy) {
        strategy.pay(order.getTotal());
    }
}
```

### 纯虚构（Pure Fabrication）

当业务对象中找不到合适的位置放某个逻辑时，可以虚构一个类（如 Service 或 Utils）。

```java
// 纯虚构类：不属于任何业务领域，但承担了特定职责
class CurrencyConverter {
    public Money convert(Money amount, Currency from, Currency to) {
        // 汇率转换逻辑
    }
}

class TaxCalculator {
    public Money calculateTax(Order order, TaxRegion region) {
        // 税费计算逻辑
    }
}
```

### 中介者（Indirection）

将职责分配给中介对象，在两个或多个类之间进行中介，减少它们之间的耦合。

```java
// 中介者：解耦多个组件
class Mediator {
    private ComponentA componentA;
    private ComponentB componentB;

    public void notify(Component sender, String event) {
        if (sender == componentA) {
            componentB.handleEventFromA(event);
        } else {
            componentA.handleEventFromB(event);
        }
    }
}
```

### 不变性（Protected Variations）

识别系统中预计会发生变化的部分，创建稳定的接口来封装这些变化。

```java
// 封装变化：数据源可能变化
interface DataSource {
    List<Order> fetchOrders();
}

class DatabaseDataSource implements DataSource {
    public List<Order> fetchOrders() {
        // 从数据库获取
    }
}

class ApiDataSource implements DataSource {
    public List<Order> fetchOrders() {
        // 从 API 获取
    }
}

// 客户代码不受数据源变化影响
class OrderReport {
    private final DataSource dataSource;

    public OrderReport(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    public void generate() {
        List<Order> orders = dataSource.fetchOrders();
        // 生成报告...
    }
}
```

## 三者的关系

LoD、TDA 和 GRASP 不是孤立的原则，而是从不同角度解决同一问题：如何合理分配职责以降低耦合度。

| 原则 | 关注点 | 解决的问题 |
|------|--------|-----------|
| LoD  | 对象之间的交互距离 | 减少不必要的对象依赖 |
| TDA  | 对象封装的边界 | 将决策逻辑保留在对象内部 |
| GRASP| 职责分配的决策框架 | 系统化地决定谁该做什么 |

GRASP 提供了分配职责的通用原则，LoD 和 TDA 则是具体的实现指南。遵循 GRASP 进行职责分配时，自然会满足 LoD 和 TDA 的要求。

## 实践建议

### 设计时的思考顺序

1. 首先应用信息专家原则：谁拥有必要信息就由谁负责
2. 然后考虑低耦合、高内聚：评估分配的影响
3. 如果无法在业务对象中找到合适位置，考虑纯虚构
4. 用控制器模式处理系统事件
5. 用多态封装变化的行为

### 代码审查检查清单

- [ ] 是否存在长链调用（如 `a.getB().getC().doSomething()`）
- [ ] 是否在外部代码中根据对象状态做决策
- [ ] 类的职责是否单一，是否承担了无关功能
- [ ] 是否直接依赖具体实现而非抽象
- [ ] 变化的部分是否被适当封装

职责分配是软件设计的基础，良好的职责分配能够降低系统的复杂度，提高代码的可维护性和可测试性。GRASP 提供了系统的思考框架，LoD 和 TDA 则是具体的实践指南，三者结合使用，能够帮助我们构建出更加健壮的软件系统。
