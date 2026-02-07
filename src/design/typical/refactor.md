---
title: 重构
order: 2
---

# 重构
重构是在不改变软件外部行为的前提下，改进其内部结构的过程。这个概念由 Martin Fowler 在其经典著作《重构：改善既有代码的设计》中系统阐述。重构不是一次性的大动作，而是日常开发的一部分，是保持代码健康、控制技术债务的关键实践。

## 重构的本质
重构有两个严格的约束：不改变外部行为、改进内部结构。这意味着重构后的代码对外部观察者来说行为完全一致，但内部结构变得更清晰、更易维护、更易扩展。

重构与性能优化的区别在于：重构追求结构改善，性能优化追求执行效率。重构与重写的区别在于：重构是渐进式的改进，重写是推倒重来。重构与添加新功能的区别在于：重构不改变功能，只改变结构。

```java
// 重构前：方法过长，职责不清
public void processOrder(Order order) {
    // 参数校验
    if (order == null) throw new IllegalArgumentException("订单不能为空");
    if (order.getItems() == null || order.getItems().isEmpty()) {
        throw new IllegalArgumentException("订单不能为空");
    }
    if (order.getCustomer() == null) {
        throw new IllegalArgumentException("客户信息不能为空");
    }

    // 计算金额
    double total = 0;
    for (OrderItem item : order.getItems()) {
        if (item.getPrice() == null || item.getPrice() <= 0) {
            throw new IllegalArgumentException("商品价格无效");
        }
        if (item.getQuantity() <= 0) {
            throw new IllegalArgumentException("商品数量无效");
        }
        total += item.getPrice() * item.getQuantity();
    }

    // 保存订单
    order.setTotalAmount(total);
    order.setStatus(OrderStatus.CONFIRMED);
    order.setConfirmTime(LocalDateTime.now());
    orderRepository.save(order);

    // 发送通知
    emailService.sendOrderConfirm(order.getCustomer().getEmail(), order);
    smsService.sendOrderConfirm(order.getCustomer().getPhone(), order);
}

// 重构后：职责分离，结构清晰
public void processOrder(Order order) {
    validateOrder(order);
    Money total = calculateTotalAmount(order);
    confirmOrder(order, total);
    sendNotification(order);
}

private void validateOrder(Order order) {
    if (order == null) throw new IllegalArgumentException("订单不能为空");
    if (order.getItems() == null || order.getItems().isEmpty()) {
        throw new IllegalArgumentException("订单不能为空");
    }
    if (order.getCustomer() == null) {
        throw new IllegalArgumentException("客户信息不能为空");
    }
}

private Money calculateTotalAmount(Order order) {
    Money total = Money.zero();
    for (OrderItem item : order.getItems()) {
        validateOrderItem(item);
        total = total.add(item.getSubTotal());
    }
    return total;
}

private void confirmOrder(Order order, Money total) {
    order.setTotalAmount(total);
    order.setStatus(OrderStatus.CONFIRMED);
    order.setConfirmTime(LocalDateTime.now());
    orderRepository.save(order);
}

private void sendNotification(Order order) {
    Customer customer = order.getCustomer();
    emailService.sendOrderConfirm(customer.getEmail(), order);
    smsService.sendOrderConfirm(customer.getPhone(), order);
}
```

重构的价值不是立即显现的，而是通过降低后续开发的难度、减少 bug 的产生、提高代码的可读性来体现。好的重构让代码"说人话"，让业务逻辑一目了然。

## 何时重构

重构应该成为日常开发习惯，而不是计划中的一项特殊任务。以下是一些明确的重构时机：

**准备性重构**：在添加新功能之前，如果现有代码结构难以支撑新功能，应该先进行重构。这种重构扫清障碍，让新功能的添加变得更加自然。

**理解性重构**：在阅读他人代码或自己过去的代码时，如果发现代码难以理解，可以通过重构来改善结构。这种重构帮助理解代码意图，同时让代码更清晰。

**代码评审后重构**：代码评审中发现的结构性问题，应该及时重构。不要让问题积累，小问题不及时处理会演变成大问题。

**重复代码出现时**：当发现两处或多处代码几乎相同时，应该通过重构提取公共部分。重复代码是维护成本的源头，一处改动需要同步修改多处。

**命名混乱时**：当发现变量、方法、类的命名不能准确表达其意图时，应该通过重构改善命名。好的命名是自解释的，可以减少注释的需要。

**测试失败后重构**：在修复 bug 或添加测试后，如果发现代码结构可以优化，应该进行重构。测试保护了重构的安全性，这是重构的最佳时机。

## 重构的方法论

重构的核心是"小步快跑"。每次只做一个小改动，立即运行测试确认没有破坏行为。这种模式将风险降至最低，即使出错也可以快速回退。

**重构的节奏**：修改代码 → 运行测试 → 确认通过 → 继续下一步。一个完整的重构可能由数十甚至上百个小步骤组成，但每个步骤都是安全的。

**重构的准则**：
- 不要在重构的同时添加新功能
- 不要在重构的同时修复 bug（除非重构引入了 bug）
- 不要在没有测试保护的情况下进行重构
- 不要在时间紧迫时进行大规模重构

**重构的规模**：
- 微重构：几秒钟到几分钟，如提取方法、重命名变量
- 小重构：几分钟到半小时，如提取类、移动方法
- 中重构：半小时到半天，如重新设计类层次
- 大重构：数天到数周，如模块重新划分、架构调整

大多数情况下应该优先进行微重构和小重构，它们积少成多，风险可控。中大规模的重构应该谨慎，最好有专门的重构分支和充分的时间。

## 常见的重构手法
Martin Fowler 在《重构》一书中列举了 70 多种重构手法，以下是最常用和最有价值的几种。

### 提取方法

当一个方法过长或包含多个层次的抽象时，应该将其拆分为多个小方法。每个方法做一件事，有一个清晰的名称。

```java
// 重构前
public String generateReport() {
    StringBuilder sb = new StringBuilder();
    sb.append("销售报告\n");
    sb.append("时间: ").append(LocalDateTime.now()).append("\n");
    sb.append("------------------------\n");

    List<Order> orders = orderRepository.findAll();
    double total = 0;
    for (Order order : orders) {
        sb.append(order.getId()).append(": ")
          .append(order.getAmount()).append("\n");
        total += order.getAmount();
    }

    sb.append("------------------------\n");
    sb.append("总计: ").append(total).append("\n");
    return sb.toString();
}

// 重构后
public String generateReport() {
    StringBuilder sb = new StringBuilder();
    appendHeader(sb);
    appendOrderDetails(sb);
    appendSummary(sb);
    return sb.toString();
}

private void appendHeader(StringBuilder sb) {
    sb.append("销售报告\n");
    sb.append("时间: ").append(LocalDateTime.now()).append("\n");
    sb.append("------------------------\n");
}

private void appendOrderDetails(StringBuilder sb) {
    List<Order> orders = orderRepository.findAll();
    for (Order order : orders) {
        sb.append(order.getId()).append(": ")
          .append(order.getAmount()).append("\n");
    }
}

private void appendSummary(StringBuilder sb) {
    double total = calculateTotal();
    sb.append("------------------------\n");
    sb.append("总计: ").append(total).append("\n");
}
```

### 提取类
当一个类承担过多职责时，应该将部分职责提取到新类中。这是单一职责原则的实践。

```java
// 重构前：订单类承担了计算职责
class Order {
    private List<OrderItem> items;

    public double calculateTotal() {
        double total = 0;
        for (OrderItem item : items) {
            // 会员折扣
            double discount = 0;
            if (item.getProduct().isMemberDiscount() && customer.isMember()) {
                discount = item.getPrice() * 0.1;
            }
            // 数量折扣
            if (item.getQuantity() >= 10) {
                discount += item.getPrice() * 0.05;
            }
            total += item.getPrice() * item.getQuantity() - discount;
        }
        return total;
    }
}

// 重构后：将计算逻辑提取到独立的服务
class Order {
    private List<OrderItem> items;
    private PriceCalculator priceCalculator;

    public Money calculateTotal() {
        return priceCalculator.calculate(items);
    }
}

class PriceCalculator {
    public Money calculate(List<OrderItem> items) {
        Money total = Money.zero();
        for (OrderItem item : items) {
            Money discount = calculateDiscount(item);
            total = total.add(item.getSubTotal().subtract(discount));
        }
        return total;
    }

    private Money calculateDiscount(OrderItem item) {
        Money discount = Money.zero();
        if (isEligibleForMemberDiscount(item)) {
            discount = discount.add(item.getSubTotal().multiply(0.1));
        }
        if (item.getQuantity() >= 10) {
            discount = discount.add(item.getSubTotal().multiply(0.05));
        }
        return discount;
    }
}
```

### 引入参数对象

当一组参数总是同时出现时，应该将它们封装为一个对象。这样可以减少参数数量，提高代码的可读性。

```java
// 重构前
public void createOrder(String customerId, String customerName,
                       String customerEmail, String customerPhone,
                       String address, String city, String zipCode) {
    // 8个参数，难以理解和维护
}

// 重构后
public void createOrder(CreateOrderRequest request) {
    // 清晰简洁
}

class CreateOrderRequest {
    private CustomerInfo customerInfo;
    private Address shippingAddress;
}
```

### 以函数调用取代参数

当一个参数可以通过调用另一个函数获得时，应该删除这个参数，让函数自己调用。这样可以减少参数传递的复杂度。

```java
// 重构前
public void processOrder(Order order, CustomerRepository customerRepo) {
    Customer customer = customerRepo.findById(order.getCustomerId());
    // ...
}

// 重构后
public void processOrder(Order order) {
    Customer customer = customerRepository.findById(order.getCustomerId());
    // ...
}
```

### 以字段取代子类

当子类只是返回不同的常量值时，应该用字段替代子类。

```java
// 重构前：只为性别创建子类
class Person {
    abstract char getCode();
}

class Male extends Person {
    char getCode() { return 'M'; }
}

class Female extends Person {
    char getCode() { return 'F'; }
}

// 重构后：使用字段
class Person {
    private char genderCode;
    public static final Person MALE = new Person('M');
    public static final Person FEMALE = new Person('F');
}
```

### 以多态取代条件式

当出现大量的类型判断和分支时，应该用多态替代条件式。

```java
// 重构前：大量 if-else
class PaymentService {
    public void pay(String type, Money amount) {
        if (type.equals("alipay")) {
            AlipayGateway gateway = new AlipayGateway();
            gateway.pay(amount);
        } else if (type.equals("wechat")) {
            WechatGateway gateway = new WechatGateway();
            gateway.pay(amount);
        } else if (type.equals("card")) {
            CardGateway gateway = new CardGateway();
            gateway.pay(amount);
        }
    }
}

// 重构后：使用多态
interface PaymentGateway {
    void pay(Money amount);
}

class PaymentService {
    private Map<String, PaymentGateway> gateways;

    public void pay(String type, Money amount) {
        PaymentGateway gateway = gateways.get(type);
        if (gateway == null) {
            throw new IllegalArgumentException("不支持的支付方式");
        }
        gateway.pay(amount);
    }
}
```

## 重构与设计原则

重构是实现设计原则的手段。SOLID 原则不是一次设计完成的目标，而是通过持续重构达到的状态。

**单一职责原则（SRP）**：当一个类承担多个职责时，通过提取类、提取方法等重构手法，将职责分离到不同的类中。

**开闭原则（OCP）**：当添加新功能需要修改现有代码时，通过引入抽象、使用多态等重构手法，使代码对扩展开放。

**里氏替换原则（LSP）**：当发现子类不能透明替换父类时，通过提取接口、使用组合等重构手法，消除继承的误用。

**接口隔离原则（ISP）**：当接口过于庞大时，通过拆分接口等重构手法，让客户端只依赖需要的接口。

**依赖倒置原则（DIP）**：当高层模块直接依赖低层实现时，通过引入接口、依赖注入等重构手法，反转依赖关系。

重构是一个动态过程。代码会自然地腐化，就像房间会变乱一样。定期的重构就像打扫房间，保持代码的健康状态。

## 技术债务

技术债务是为了快速交付而做出的妥协，它是有意或无意中积累的设计缺陷和代码问题。技术债务本身不是坏事，关键是要有意识地管理和偿还。

**技术债务的来源**：时间紧迫、需求变化、知识不足、沟通不畅。当为了赶进度而写出"能跑就行"的代码时，就产生了技术债务。

**技术债务的分类**：
- 有意债务：为了赶进度有意为之，计划日后偿还
- 无意债务：由于认知不足产生的，往往更危险
- 轻微债务：局部问题，影响范围小
- 严重债务：架构问题，影响全局

**技术债务的偿还**：小债务立即还，大债务计划还。对于小的代码问题，发现时立即重构。对于大的架构问题，制定重构计划，逐步偿还。

```java
// 技术债务：临时硬编码
public void sendNotification() {
    // TODO: 改为配置
    String email = "admin@company.com";
    // ...
}

// 快速还清：提取为配置
private static final String ADMIN_EMAIL = Config.get("admin.email");
```

技术债务像利息一样会不断累积。不还的债务会越来越重，最终让开发停滞。但也不是所有债务都需要立即还，有些债务可以等到合适的时机再处理。

## 重构的实践建议

**建立重构文化**：鼓励团队将重构作为日常开发的一部分，而不是额外的工作。代码评审时关注代码结构，不只是功能正确性。

**依靠测试保护**：完善的测试是重构的前提。没有测试的重构是盲目的，容易引入 bug。重构前确保测试覆盖充分，重构后测试必须全部通过。

**使用重构工具**：现代 IDE 提供了强大的自动重构功能，如重命名、提取方法、移动类等。这些工具比手工重构更安全、更快速。

**记录重构日志**：对于大规模的重构，记录重构的目的、过程、结果。这有助于团队理解重构的价值，也为未来提供参考。

**避免过度重构**：重构是为了改善结构，不是追求完美。当代码已经足够清晰时，继续重构的边际收益递减，应该适可而止。

**重构与功能开发交替**：不要为了重构而停止功能开发，也不要为了功能开发而忽视重构。好的节奏是开发一点功能，顺便重构一点代码。

## 重构的常见陷阱

**重构与功能修改混杂**：在一次代码改动中既重构又添加功能，当出现问题时难以定位原因。应该先重构，测试通过后再添加功能。

**缺乏测试保护**：在没有足够测试的情况下进行大规模重构，一旦出错难以定位。重构前先补充测试，确保重构的安全性。

**过度设计**：为了重构而重构，引入不必要的抽象和层次。代码应该"刚刚好"，既不过于简单也不过于复杂。

**大爆炸式重构**：试图一次性重构整个系统，风险极高、周期极长。应该将大重构分解为小步骤，渐进式地完成。

**忽视业务价值**：重构的目的是降低后续开发的成本，不是为了重构本身。应该优先重构那些影响最大的部分。

重构是一门平衡的艺术。既要保持代码的健康，又不能无限期地推迟新功能。好的开发者能够在重构与功能开发之间找到合适的节奏，让代码稳步改善，而不是不断腐化。
