---
title: DDD
order: 100
---

# 领域驱动设计
领域驱动设计（Domain-Driven Design，DDD）是一种软件开发方法，它将复杂业务领域的知识直接映射到代码结构中。DDD 由 Eric Evans 在 2003 年提出，核心思想是让软件架构与业务领域保持一致，通过统一的语言和模型来连接领域专家和开发者。

## 为什么需要 DDD
传统的开发方式往往以数据库或技术视角为中心，导致业务逻辑散落在各处，代码无法反映业务知识。当业务复杂度增加时，这种割裂会导致维护困难、变更成本高昂。

DDD 通过建立**领域模型**作为软件的核心，让业务规则、约束、流程在代码中得到显式表达。这种设计的价值在于：业务逻辑集中在领域层，技术基础设施被隔离；业务概念与代码概念一一对应，领域专家可以参与代码审查；变更局限在特定子域，不会产生连锁反应。

## 战略设计
战略设计关注系统的整体结构，将大型系统分解为可管理的部分。

### 领域与子域
领域是系统要解决的问题空间，子域是领域的细分部分。根据重要性和复杂度，子域分为三类：

| 子域类型 | 定义 | 示例 | 投入策略 |
|---------|------|------|---------|
| **核心域** | 业务的核心竞争力，差异化所在 | 电商平台的推荐算法、金融公司的风控系统 | 精英团队，持续投入 |
| **支撑域** | 必要但非差异化的业务 | 通用权限管理、消息通知 | 购买或外包，适度投入 |
| **通用域** | 业界已有成熟解决方案 | 用户认证、支付网关 | 直接购买，不做开发 |

认清子域类型可以避免资源错配：在通用域重复造轮子，在核心域使用通用方案。

### 限界上下文
限界上下文是 DDD 中最重要的概念之一。一个限界上下文是一个明确的边界，边界内部，特定领域模型的术语和规则是一致的；边界之外，同样的术语可能有不同的含义。

```java
// 订单上下文中的"商品"
class OrderItem {
    private ProductId productId;  // 商品 ID
    private int quantity;         // 数量
    private Money price;          // 下单时的价格
}

// 库存上下文中的"商品"
class InventoryItem {
    private ProductId productId;  // 商品 ID
    private int availableStock;   // 可用库存
    private int reservedStock;    // 预占库存
}
```

两个上下文都使用"商品"概念，但订单上下文关注下单时的快照，库存上下文关注实时的库存数量。强行统一会导致概念混淆，独立维护各自的模型才是正确做法。

### 上下文映射图
上下文映射图描述不同限界上下文之间的关系。常见的关系类型包括：

- **上游/下游（Upstream/Downstream）**：上游上下文的变化会影响下游上下文
- **防腐层（ACL）**：下游上下文通过适配器隔离上游模型的变化
- **共享内核（Shared Kernel）**：两个上下文共享一小部分模型和代码
- **客户/供应商（Customer/Supplier）**：下游是上游的客户，存在明确的服务关系
- **分开方式（Separate Ways）**：两个上下文完全独立，没有集成关系

## 战术设计
战术设计关注限界上下文内部的建模细节，提供了丰富的模式来表达业务知识。

### 值对象与实体
这是 DDD 中最基础也是最重要的区分。

#### 实体
实体由**身份**标识，而非属性。即使属性完全改变，实体依然是同一个实体。实体具有唯一标识符，如 ID、UUID、业务编号等。

```java
class User {
    private UserId id;  // 唯一标识，不变

    private String name;
    private String email;
    private int age;

    // 即使所有属性都改变，只要 id 相同，就是同一个用户
    public boolean equals(Object other) {
        return other instanceof User && this.id.equals(((User)other).id);
    }

    public int hashCode() {
        return id.hashCode();
    }
}

// 两个属性不同的对象，可能是同一个实体
User user1 = userRepository.findById(new UserId("123"));
user1.setName("新名字");

User user2 = userRepository.findById(new UserId("123"));
// user1 和 user2 是同一个实体，尽管 name 属性不同
```

实体的核心是身份的连续性。用户修改了昵称、换了邮箱，依然是同一个用户；订单的状态从待支付变为已支付，依然是同一个订单。建模时，问自己"这个对象需要连续追踪身份吗"，如果是，它就是实体。

#### 值对象
值对象由**属性**的值决定相等性，没有独立身份。值对象通常是不可变的，创建后状态不再改变。

```java
// 金额值对象
class Money {
    private final BigDecimal amount;
    private final Currency currency;

    public Money(BigDecimal amount, Currency currency) {
        this.amount = amount;
        this.currency = currency;
    }

    // 相等性由所有属性决定
    public boolean equals(Object other) {
        if (!(other instanceof Money)) return false;
        Money that = (Money) other;
        return this.amount.equals(that.amount) &&
               this.currency.equals(that.currency);
    }

    // 返回新对象而非修改自身
    public Money add(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException("货币不同");
        }
        return new Money(this.amount.add(other.amount), this.currency);
    }
}

// 两个属性相同的对象，被认为是相等的
Money price1 = new Money(new BigDecimal("100"), Currency.CNY);
Money price2 = new Money(new BigDecimal("100"), Currency.CNY);
// price1.equals(price2) == true
```

值对象的优势在于简化理解、天然线程安全、避免副作用。典型的值对象包括：金额、日期范围、地址、颜色、坐标等。建模时，问自己"这个对象只是属性的集合吗，替换为另一个相同属性的对象会影响业务吗"，如果是，它就是值对象。

#### 实体与值对象的对比

| 维度 | 实体 | 值对象 |
|------|------|--------|
| **标识** | 有唯一标识符，标识即身份 | 无标识，由属性值决定身份 |
| **可变性** | 可变，属性改变身份不变 | 不可变，改变属性即成为新对象 |
| **生命周期** | 长期存在，有独立生命周期 | 依附于实体，随实体创建销毁 |
| **相等性** | ID 相同即为相等 | 所有属性相同才相等 |
| **示例** | 用户、订单、商品 | 金额、地址、日期范围 |

### 聚合与聚合根

聚合是数据一致性的边界，是一组相关对象的集合。聚合根是聚合的入口，外部只能通过聚合根来访问聚合内部的实体和值对象。

#### 一致性边界

在分布式系统中，跨对象的事务一致性难以保证。聚合将需要强一致性的对象组织在一起，保证聚合内部的数据一致性，聚合之间则接受最终一致性。

```java
// 订单聚合
class Order {
    private OrderId id;              // 聚合根的标识
    private List<OrderItem> items;   // 聚合内部的实体
    private ShippingAddress address; // 聚合内部的值对象
    private OrderStatus status;
    private Money totalAmount;

    // 只能通过聚合根操作订单项
    public void addItem(Product product, int quantity, Money price) {
        if (this.status != OrderStatus.DRAFT) {
            throw new IllegalStateException("只有草稿状态的订单可以添加商品");
        }

        OrderItem item = new OrderItem(product, quantity, price);
        this.items.add(item);
        this.totalAmount = this.totalAmount.add(item.getSubTotal());
    }

    public void removeItem(OrderItemId itemId) {
        if (this.status != OrderStatus.DRAFT) {
            throw new IllegalStateException("只有草稿状态的订单可以移除商品");
        }

        OrderItem item = findItem(itemId);
        this.items.remove(item);
        this.totalAmount = this.totalAmount.subtract(item.getSubTotal());
    }

    // 聚合根负责保证一致性
    public void place() {
        if (this.items.isEmpty()) {
            throw new IllegalStateException("订单不能为空");
        }
        if (this.totalAmount.isZero()) {
            throw new IllegalStateException("订单金额必须大于零");
        }
        this.status = OrderStatus.PLACED;
    }
}

// 外部不能直接访问 OrderItem
// OrderItem orderItem = ...; // 错误
// order.getOrderItems().add(item); // 错误
order.addItem(product, quantity, price); // 正确
```

这个设计中，订单是聚合根，订单项是聚合内部的实体。外部代码不能直接持有订单项的引用，只能通过订单聚合根来操作。这样可以保证添加、删除订单项时，订单总金额的正确更新，维护数据一致性。

#### 聚合设计原则

- **聚合根是唯一入口**：外部只能引用聚合根，不能直接引用聚合内部的对象
- **聚合边界内强一致**：聚合内的对象修改必须在同一事务中完成
- **聚合之间最终一致**：不同聚合的修改可以使用异步事件，不要求强一致
- **保持聚合小巧**：聚合应该尽可能小，大的聚合会影响性能和并发
- **引用用 ID 而非对象引用**：聚合之间通过 ID 引用，避免直接对象引用导致边界模糊

```java
// 订单聚合不应该直接引用库存聚合
class Order {
    // private InventoryItem inventoryItem; // 错误：跨越聚合边界
    private InventoryItemId inventoryItemId; // 正确：使用 ID 引用
}
```

#### 常见聚合设计错误

- **聚合过大**：一个聚合包含过多实体，导致性能问题和并发冲突
- **聚合间直接引用**：聚合根直接持有其他聚合根的对象引用
- **忽略不变量**：聚合根没有维护业务规则，允许非法状态
- **滥用事务**：跨聚合的事务试图实现强一致性，损害系统可伸缩性

### 领域服务

有些业务逻辑不适合放在实体或值对象中，它们不自然地属于任何实体，或者需要协调多个实体。这些逻辑可以放在领域服务中。

```java
// 领域服务：没有自然归属的业务逻辑
class OrderService {
    private OrderRepository orderRepo;
    private InventoryService inventoryService;
    private PaymentService paymentService;

    // 协调多个聚合的业务流程
    public void placeOrder(OrderId orderId) {
        Order order = orderRepo.findById(orderId);

        // 检查库存（跨聚合）
        for (OrderItem item : order.getItems()) {
            if (!inventoryService.checkStock(item.getProductId(), item.getQuantity())) {
                throw new InsufficientStockException();
            }
        }

        // 扣减库存（跨聚合）
        inventoryService.reserveStock(order.getItems());

        // 改变订单状态
        order.place();

        // 发起支付（跨聚合）
        paymentService.charge(order.getTotalAmount());

        orderRepo.save(order);
    }
}
```

领域服务与基础服务的区别在于：领域服务包含业务逻辑，如订单服务、库存服务；基础服务提供技术能力，如邮件服务、短信服务。领域服务是领域层的一部分，基础服务属于基础设施层。

### 领域事件

领域事件是领域内发生的事实，表示过去发生的业务事件。事件发布后，触发其他聚合或限界上下文的反应。

```java
// 领域事件：订单已支付
class OrderPaidEvent {
    private final OrderId orderId;
    private final Money amount;
    private final LocalDateTime occurredAt;

    public OrderPaidEvent(OrderId orderId, Money amount) {
        this.orderId = orderId;
        this.amount = amount;
        this.occurredAt = LocalDateTime.now();
    }
}

// 在聚合根中发布事件
class Order {
    private List<DomainEvent> pendingEvents = new ArrayList<>();

    public void pay(PaymentMethod method) {
        if (this.status != OrderStatus.PLACED) {
            throw new IllegalStateException("订单状态不正确");
        }

        this.status = OrderStatus.PAID;
        this.paymentMethod = method;

        // 记录事件
        this.pendingEvents.add(new OrderPaidEvent(this.id, this.totalAmount));
    }

    public List<DomainEvent> getPendingEvents() {
        return Collections.unmodifiableList(pendingEvents);
    }

    public void clearEvents() {
        this.pendingEvents.clear();
    }
}

// 事件处理器
class OrderPaidEventHandler {
    private InventoryService inventoryService;
    private NotificationService notificationService;

    @EventListener
    public void handle(OrderPaidEvent event) {
        // 扣减库存
        inventoryService.confirmReservation(event.getOrderId());

        // 发送通知
        notificationService.sendPaymentConfirmation(event.getOrderId());
    }
}
```

领域事件的好处在于：解耦聚合，聚合不需要知道谁会对事件做出反应；实现最终一致性，事件处理器可以异步执行；记录业务历史，事件流可以重建系统状态。

### 仓储

仓储封装了对象的持久化和查询逻辑，向领域层提供类似集合的接口。仓储隐藏了数据库细节，让领域层不依赖具体的数据存储技术。

```java
// 仓储接口：领域层定义
interface OrderRepository {
    void save(Order order);
    Optional<Order> findById(OrderId id);
    List<Order> findByCustomer(CustomerId customerId);
}

// 仓储实现：基础设施层实现
class SQLOrderRepository implements OrderRepository {
    private DataSource dataSource;

    public void save(Order order) {
        // 将聚合根序列化到数据库
        String sql = "INSERT INTO orders (id, customer_id, status, total_amount) VALUES (?, ?, ?, ?)";
        // JDBC 操作...
    }

    public Optional<Order> findById(OrderId id) {
        // 从数据库重建聚合根
        String sql = "SELECT * FROM orders WHERE id = ?";
        // JDBC 操作...
        // 需要加载订单项，重建完整的聚合
    }
}
```

仓储只用于聚合根，不用于聚合内部的实体或值对象。查询返回的是完整的聚合，而不是部分数据。复杂的查询可以使用规约模式或专门的查询服务。

## 架构模式

DDD 与特定的架构模式配合使用，可以更好地隔离领域层。

### 分层架构

传统的 DDD 分层架构将系统分为四层：用户界面层、应用层、领域层、基础设施层。

```
┌─────────────────────────────────────┐
│         用户界面层 (UI)              │  处理用户交互，展示信息
├─────────────────────────────────────┤
│          应用层 (Application)        │  编排业务流程，不包含业务逻辑
├─────────────────────────────────────┤
│           领域层 (Domain)            │  核心业务逻辑，与基础设施无关
├─────────────────────────────────────┤
│       基础设施层 (Infrastructure)    │  技术实现，数据持久化，外部服务
└─────────────────────────────────────┘
```

关键规则：依赖方向只能是单向的，上层依赖下层，领域层不依赖任何层。应用层很薄，只负责协调，业务逻辑在领域层。基础设施层实现领域层定义的接口。

### 六边形架构

六边形架构（又称端口和适配器架构）强调通过端口（接口）与外部交互，每个端口有适配器负责具体实现。

```
        ┌───────────────────┐
        │   应用核心        │
        │  (领域层+应用层)  │
        └───────────────────┘
                 │
       ┌─────────┼─────────┐
       │         │         │
    端口      端口      端口
    (接口)    (接口)    (接口)
       │         │         │
    适配器    适配器    适配器
   (Web)    (DB)     (MQ)
```

端口是应用核心提供的接口，如 `OrderRepository`、`PaymentService`。适配器是接口的具体实现，如 `SQLRepository`、`StripePaymentAdapter`。这种架构使得应用核心不依赖具体技术，替换实现只需添加新的适配器。

### CQRS

命令查询责任分离（CQRS）将读操作和写操作分离，使用不同的模型和数据结构。

```java
// 命令端：写模型
class OrderCommand {
    void createOrder(CreateOrderCommand cmd);
    void addItem(AddItemCommand cmd);
    void placeOrder(PlaceOrderCommand cmd);
}

// 查询端：读模型
class OrderQuery {
    OrderDetailView getOrderDetail(OrderId id);
    List<OrderSummaryView> getCustomerOrders(CustomerId customerId);
}

// 命令和查询使用不同的数据库
OrderCommandRepository -> orders_db (写优化)
OrderQueryRepository    -> orders_read_db (读优化)
```

CQRS 适用于读操作和写操作差异很大的场景，如读操作需要复杂的关联查询，写操作需要强一致性保证。代价是系统复杂度增加，最终一致性需要处理。

## 实践指南

**从简单开始**：不要一开始就全面应用 DDD，先在核心子域尝试，逐步积累经验。

**与领域专家合作**：DDD 强调通用语言，需要领域专家和开发者共同参与建模。代码中的术语应该与业务讨论中的术语一致。

**持续重构**：模型不是一次设计完成的，随着对业务理解的加深，不断重构模型，保持代码与业务知识同步。

**警惕过度设计**：DDD 概念多、模式多，容易陷入过度设计。对于简单的 CRUD 应用，传统的分层架构可能更合适。

**工具支持**：考虑使用支持 DDD 的框架，如 Axon Framework（事件驱动）、Spring Data（仓储模式）、JPA（实体映射），减少样板代码。
