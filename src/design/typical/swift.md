# 敏捷开发
敏捷开发不是一种具体的工具，而是一套价值观和原则。它给软件设计的启示是：不要试图在第一天就设计出能支撑未来十年的架构，而要设计出一个"在未来十年里都很容易被修改"的架构。

## 敏捷宣言
2001 年，一群资深开发者在雪鸟会议上提出了《敏捷宣言》，这四句话奠定了敏捷的基调：

| 宣言                     | 解释                             | 实践意义                       |
| ------------------------ | -------------------------------- | ------------------------------ |
| 个体和互动高于流程和工具 | 人才是核心，别被死板的审批流卡死 | 面对面沟通、站立会议、结对编程 |
| 工作的软件高于详尽的文档 | 代码跑起来比设计文档更有说服力   | 可工作的增量、原型优先         |
| 客户合作高于合同谈判     | 把客户当伙伴，而不是对手         | 用户演示、反馈循环             |
| 响应变化高于遵循计划     | 计划赶不上变化，要拥抱变化       | 迭代开发、适应性规划           |

这四大宣言都指向同一个目标：**快速响应变化**。

## 演进式架构
传统架构追求"完美"，试图在第一天就设计出能支撑未来所有需求的系统。敏捷架构则追求"可演进"，设计出一个在未来很容易被修改的系统。

既然需求永远在变，那么架构的目标就不应该是完美，而是易于改变。架构应该像生物一样，能够根据环境的变化而进化。

适应度函数（Fitness Functions）是演进式架构的关键概念，通过自动化测试或监控来确保架构在演变过程中不会违背最初的设计原则。

```java
// 适应度函数示例：架构层面的测试
public class ArchitectureFitnessTests {

    // 测试：模块间不应该有直接依赖，只能通过接口通信
    @Test
    public void modulesShouldOnlyDependOnInterfaces() {
        Set<String> violations = analyzeDependencies();
        assertTrue(violations.isEmpty(),
            "Found direct module dependencies: " + violations);
    }

    // 测试：循环依赖应该被禁止
    @Test
    public void noCircularDependencies() {
        DependencyGraph graph = buildDependencyGraph();
        assertFalse(graph.hasCycles(),
            "Circular dependencies detected");
    }

    // 测试：API 变更必须向后兼容
    @Test
    public void apiChangesMustBeBackwardCompatible() {
        ApiVersion current = getCurrentApi();
        ApiVersion previous = getPreviousApi();
        assertTrue(current.isBackwardCompatibleWith(previous));
    }
}
```

**演进式架构的实践**

| 实践                | 描述                                     |
| ------------------- | ---------------------------------------- |
| 最后责任时刻        | 推迟不可逆的决策，直到必须做的时候       |
| 架构决策记录（ADR） | 记录重要的架构决策及其背景，便于后续审视 |
| 绞杀者模式          | 逐步替换旧系统，而非一次性重写           |
| 功能开关            | 允许在不部署代码的情况下开启/关闭功能    |

## 敏捷的运作方式

传统的"瀑布模型"像造大桥：设计半年、施工一年、最后剪彩。如果桥修到一半发现位置错了，基本无解。

敏捷开发像"进化"：短迭代、小步快跑、快速反馈。

迭代与增量
```
传统瀑布模型：
需求分析 ──→ 设计 ──→ 开发 ──→ 测试 ──→ 发布
   3个月      2个月   6个月   2个月   1个月
    └────────────── 14个月 ────────────┘

敏捷开发：
需求 ──→ 设计 ──→ 开发 ──→ 测试 ──→ 发布 ──┐
  1天      2天      3天      2天    1天   │
                                         │
        ←────────── 2周迭代 ──────────────┘

需求 ──→ 设计 ──→ 开发 ──→ 测试 ──→ 发布 ──┘
  ...
```

每个迭代周期结束都交付一个可运行、可测试的版本，让客户看一眼，不对立刻改。这种快速反馈循环能够：

- 尽早发现方向错误，避免浪费
- 根据实际使用情况调整优先级
- 持续验证技术决策的正确性

## 工程实践

在敏捷环境中，代码会被频繁修改。拥抱重构、测试驱动开发（TDD）、持续集成（CI）、持续交付（CD）都是必不可少的工程实践。只有边界清晰、依赖简单的系统，才能在敏捷的浪潮里快速转身。

### 拥抱重构

敏捷不追求"一次性完美设计"，而是提倡演进式设计。重构（Refactoring）是在不改变功能的前提下优化结构，是敏捷开发的必修课。

```java
// 重构前：难以修改
public class OrderService {
    public void processOrder(Order order) {
        // 订单验证
        if (order == null || order.getItems().isEmpty()) {
            throw new IllegalArgumentException("Invalid order");
        }

        // 库存检查
        for (OrderItem item : order.getItems()) {
            Product product = productRepository.findById(item.getProductId());
            if (product.getStock() < item.getQuantity()) {
                throw new OutOfStockException(product.getName());
            }
        }

        // 价格计算
        Money total = new Money(0);
        for (OrderItem item : order.getItems()) {
            Product product = productRepository.findById(item.getProductId());
            total = total.add(product.getPrice().multiply(item.getQuantity()));
        }

        // 支付处理
        paymentGateway.charge(order.getPaymentMethod(), total);

        // 库存扣减
        for (OrderItem item : order.getItems()) {
            Product product = productRepository.findById(item.getProductId());
            product.setStock(product.getStock() - item.getQuantity());
            productRepository.save(product);
        }
    }
}

// 重构后：职责分离，易于修改
public class OrderService {
    private final OrderValidator validator;
    private final InventoryService inventory;
    private final PricingService pricing;
    private final PaymentService payment;

    public void processOrder(Order order) {
        validator.validate(order);
        inventory.reserve(order.getItems());
        Money total = pricing.calculate(order);
        payment.charge(order.getPaymentMethod(), total);
    }
}
```

### 测试驱动开发（TDD）

先写测试再写代码，确保每次改动都不会破坏原有功能。TDD 的红-绿-重构循环：

1. **红**：写一个失败的测试
2. **绿**：写最简单的代码让测试通过
3. **重构**：优化代码结构，保持测试通过

```java
// 1. 先写测试（红灯）
@Test
public void calculateTotal_withMultipleItems_returnsSum() {
    Order order = new Order();
    order.addItem(new OrderItem("Product A", new Money(100), 2));
    order.addItem(new OrderItem("Product B", new Money(50), 1));

    Money total = order.calculateTotal();

    assertEquals(new Money(250), total);
}

// 2. 写最简单的实现（绿灯）
public class Order {
    private List<OrderItem> items = new ArrayList<>();

    public Money calculateTotal() {
        Money total = new Money(0);
        for (OrderItem item : items) {
            total = total.add(item.getSubtotal());
        }
        return total;
    }
}

// 3. 重构（持续绿灯）
// 如果发现更好的实现方式，可以安全重构
```

### 持续集成与持续交付

| 实践           | 描述                                   | 工具示例                           |
| -------------- | -------------------------------------- | ---------------------------------- |
| CI（持续集成） | 代码一提交就自动触发编译和测试         | Jenkins、GitLab CI、GitHub Actions |
| CD（持续交付） | 通过自动化部署，让软件具备随时发布能力 | Spinnaker、ArgoCD                  |

没有自动化测试和 CI/CD，敏捷就是"快快地写 Bug"。这些工程实践是敏捷的命脉。

## 常见误区

很多团队声称自己"敏捷"，但实际上只是在"快速地乱写代码"。以下是敏捷实践中的常见误区。

+ 只有快，没有质

  忽视了设计原则（如 SOLID），导致技术债堆积如山，后期跑不动。

  ```java
  // "伪敏捷"代码：快速但不可维护
  public class Utils {
      public static void processOrder(Order o, Payment p, User u, Inventory i) {
          // 500 行逻辑混在一起
      }
  }

  // 真正的敏捷：快速且可维护
  public class OrderService {
      public void process(Order order) {
          validator.validate(order);
          inventory.reserve(order);
          payment.charge(order);
      }
  }
  ```

+ 只有站会，没有反馈

  每天开站立会议，但从不听取用户反馈，也不根据反馈调整方向。站会本身不是目的，快速反馈和调整才是。

+ 文档缺失

  敏捷说"工作的软件高于详尽的文档"，不代表"不需要文档"。关键的设计决策、API 契约、运维手册仍然需要文档，只是文档的形式和时机需要调整。

  | 文档类型 | 传统做法            | 敏捷做法                    |
  | -------- | ------------------- | --------------------------- |
  | 设计文档 | 开发前写 100 页文档 | 开发后记录架构决策（ADR）   |
  | API 文档 | 手动维护 Word 文档  | 自动生成（Swagger/OpenAPI） |
  | 运维手册 | 离线 Wiki           | 基础设施即代码（Terraform） |

+ 敏捷 ≠ 无计划

  敏捷不是"想到哪写到哪"，而是"短周期的适应性规划"。每个迭代开始时仍然需要计划，只是计划的范围更小、周期更短。

## 敏捷设计的原则

将敏捷理念应用到软件设计中，需要遵循以下原则：

| 原则         | 描述                                 |
| ------------ | ------------------------------------ |
| 最后责任时刻 | 推迟不可逆的决策，直到必须做的时候   |
| 拥抱变化     | 设计应该易于修改，而非试图预测未来   |
| 小步增量     | 通过小的、持续的改进来实现大的变化   |
| 自动化验证   | 用测试和监控确保每次改动不会破坏系统 |
| 快速反馈     | 尽早获得用户反馈，及时调整方向       |

敏捷设计的本质是承认不确定性，并设计出能够在不确定性中生存和演进的系统。这需要技术和组织两方面的协同配合。
