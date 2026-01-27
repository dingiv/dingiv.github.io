---
title: SOLID
order: 1
---

# SOLID 原则
SOLID 原则是面向对象设计中最经典的五条指导原则，由 Robert C. Martin（Uncle Bob）提出。目的是写出易维护、易扩展、低耦合、高内聚的代码。

| 字母 | 英文全称                        | 中文名       | 一句话核心含义                       | 违反时最典型表现                      |
| ---- | ------------------------------- | ------------ | ------------------------------------ | ------------------------------------- |
| S    | Single Responsibility Principle | 单一职责原则 | 一个类/模块只因单一原因而改变        | 一个类改动会影响多个不相关功能        |
| O    | Open-Closed Principle           | 开闭原则     | 对扩展开放，对修改关闭               | 加新功能必须改已有核心代码            |
| L    | Liskov Substitution Principle   | 里氏替换原则 | 子类能透明替换父类而不破坏程序正确性 | 子类重写方法后行为异常 / 前置条件变弱 |
| I    | Interface Segregation Principle | 接口隔离原则 | 不要强迫客户端依赖它不用的接口       | 实现一个大接口却只用其中两三个方法    |
| D    | Dependency Inversion Principle  | 依赖倒置原则 | 高层模块不依赖低层模块，都依赖抽象   | 业务逻辑直接 new 具体数据库/第三方类  |

## 单一职责原则 (SRP)

单一职责原则不是"一个类只能有一个方法"，而是一个类只服务于一个变化原因或一个业务角色。2026 年最常见的违反场景是一个 Controller 同时干了参数校验、业务逻辑、数据库操作、发邮件、记录日志、格式化响应等工作，应该拆成 Service、Validator、Notifier、Formatter 等独立模块。

```typescript
// 违反示例：一个类承担了太多职责
class UserController {
  register(data: any) {
    // 参数校验
    if (!data.email || !data.password) {
      throw new Error('参数错误');
    }
    // 密码加密
    const hashed = bcrypt.hash(data.password);
    // 数据库操作
    db.users.insert({ email: data.email, password: hashed });
    // 发送欢迎邮件
    emailService.send(data.email, '欢迎注册');
    // 记录日志
    logger.info('用户注册', data.email);
    return { success: true };
  }
}

// 正确做法：职责分离
class Validator {
  validateRegister(data: any) {
    if (!data.email || !data.password) {
      throw new Error('参数错误');
    }
  }
}

class UserRepository {
  insert(user: any) {
    db.users.insert(user);
  }
}

class EmailNotifier {
  sendWelcome(email: string) {
    emailService.send(email, '欢迎注册');
  }
}

class UserService {
  constructor(
    private validator: Validator,
    private repo: UserRepository,
    private notifier: EmailNotifier
  ) {}
  register(data: any) {
    this.validator.validateRegister(data);
    const hashed = bcrypt.hash(data.password);
    this.repo.insert({ email: data.email, password: hashed });
    this.notifier.sendWelcome(data.email);
  }
}
```

## 开闭原则 (OCP)

软件设计的终极目标是让新需求主要靠新增代码实现，而不是改旧代码。现代最常见的做法包括策略模式加插件式扩展、事件总线、规则引擎、配置驱动、装饰器模式和模板方法。典型的违反表现是大量 if-else 判断类型来决定行为，每次加新类型都要改核心类。

```typescript
// 违反示例：每次增加支付方式都要修改核心代码
class PaymentService {
  pay(type: string, amount: number) {
    if (type === 'alipay') {
      console.log('支付宝支付', amount);
    } else if (type === 'wechat') {
      console.log('微信支付', amount);
    } else if (type === 'card') {
      console.log('银行卡支付', amount);
    }
    // 每次新增支付方式都要在这里加 if-else
  }
}

// 正确做法：使用策略模式，对扩展开放
interface PaymentStrategy {
  pay(amount: number): void;
}

class AlipayStrategy implements PaymentStrategy {
  pay(amount: number) {
    console.log('支付宝支付', amount);
  }
}

class WechatStrategy implements PaymentStrategy {
  pay(amount: number) {
    console.log('微信支付', amount);
  }
}

// 新增支付方式只需新增类，不需要修改现有代码
class CardStrategy implements PaymentStrategy {
  pay(amount: number) {
    console.log('银行卡支付', amount);
  }
}

class PaymentService {
  private strategies: Map<string, PaymentStrategy> = new Map();

  register(name: string, strategy: PaymentStrategy) {
    this.strategies.set(name, strategy);
  }

  pay(type: string, amount: number) {
    const strategy = this.strategies.get(type);
    if (!strategy) throw new Error('不支持的支付方式');
    strategy.pay(amount);
  }
}
```

## 里氏替换原则 (LSP)

子类必须遵守父类的契约，前置条件不能变强，后置条件不能变弱，不变量保持不变。Java 和 TypeScript 中最容易踩的坑包括：子类抛出父类没声明的受检异常、子类方法前置条件变严格（父类允许 null 而子类不允许）、子类返回值类型收窄（父类返回 Collection 而子类返回 List）。

```typescript
// 违反示例：子类改变了父类的行为契约
class Rectangle {
  constructor(private width: number, private height: number) {}

  setWidth(width: number) {
    this.width = width;
  }

  setHeight(height: number) {
    this.height = height;
  }

  area(): number {
    return this.width * this.height;
  }
}

class Square extends Rectangle {
  constructor(side: number) {
    super(side, side);
  }

  setWidth(width: number) {
    this.width = width;
    this.height = width; // 强制保持正方形，破坏了父类契约
  }

  setHeight(height: number) {
    this.width = height;
    this.height = height;
  }
}

// 使用父类的地方会被子类破坏
function processRectangle(r: Rectangle) {
  r.setWidth(5);
  r.setHeight(4);
  console.log(r.area()); // Rectangle: 20, Square: 16（错误！）
}

// 正确做法：不要强行继承，使用组合或独立类
class Rectangle {
  constructor(private width: number, private height: number) {}

  setWidth(width: number) {
    this.width = width;
  }

  setHeight(height: number) {
    this.height = height;
  }

  area(): number {
    return this.width * this.height;
  }
}

class Square {
  constructor(private side: number) {}

  setSide(side: number) {
    this.side = side;
  }

  area(): number {
    return this.side * this.side;
  }
}
```

## 接口隔离原则 (ISP)

宁可多个小接口，也不要一个胖接口。现代最典型的反例是一个超级大的 UserService 接口包含登录、注册、重置密码、权限管理、积分操作、订单查询等多个职责，应该拆成 AuthService、ProfileService、PointService 等。这样做的好处是实现类代码量减少，测试时更容易 mock。

```typescript
// 违反示例：胖接口强迫实现类不需要的方法
interface UserService {
  login(email: string, password: string): void;
  register(email: string, password: string): void;
  resetPassword(email: string): void;
  addPoints(userId: string, points: number): void;
  deductPoints(userId: string, points: number): void;
  getOrders(userId: string): Order[];
}

// 只需要积分功能的类被迫实现所有方法
class PointsService implements UserService {
  login(email: string, password: string) {
    throw new Error('不支持');
  }
  register(email: string, password: string) {
    throw new Error('不支持');
  }
  resetPassword(email: string) {
    throw new Error('不支持');
  }
  addPoints(userId: string, points: number) {
    // 实际实现
  }
  deductPoints(userId: string, points: number) {
    // 实际实现
  }
  getOrders(userId: string) {
    throw new Error('不支持');
  }
}

// 正确做法：拆分成多个小接口
interface AuthService {
  login(email: string, password: string): void;
  register(email: string, password: string): void;
  resetPassword(email: string): void;
}

interface PointsService {
  addPoints(userId: string, points: number): void;
  deductPoints(userId: string, points: number): void;
}

interface OrderService {
  getOrders(userId: string): Order[];
}

// 实现类只需要实现自己关心的接口
class PointsServiceImpl implements PointsService {
  addPoints(userId: string, points: number) {
    // 实现
  }
  deductPoints(userId: string, points: number) {
    // 实现
  }
}

class UserServiceImpl implements AuthService, PointsService, OrderService {
  // 实现所有接口
}
```

## 依赖倒置原则 (DIP)

这是最核心的一条原则，其他几条很多时候都是为它服务的。高层（业务）不依赖低层（基础设施），都依赖抽象。现代写法几乎被所有框架默认采用，包括通过构造函数或 Setter 注入接口（依赖注入）、使用 IoC 容器（Spring、NestJS、DI in .NET 等）自动解析依赖。典型的违反表现是在 Service 里直接 new JdbcTemplate、new RedisTemplate 或 new HttpClient。

```typescript
// 违反示例：高层模块直接依赖低层实现
class UserService {
  private db: MySQLDatabase;

  constructor() {
    this.db = new MySQLDatabase(); // 直接依赖具体实现
  }

  getUser(id: string) {
    return this.db.query(`SELECT * FROM users WHERE id = ${id}`);
  }
}

// 问题：无法切换数据库，无法单元测试（mock 困难）

// 正确做法：依赖抽象，通过构造函数注入
interface Database {
  query(sql: string): any;
}

class MySQLDatabase implements Database {
  query(sql: string) {
    // MySQL 实现
  }
}

class PostgreSQLDatabase implements Database {
  query(sql: string) {
    // PostgreSQL 实现
  }
}

class UserService {
  constructor(private db: Database) {} // 依赖抽象

  getUser(id: string) {
    return this.db.query(`SELECT * FROM users WHERE id = ${id}`);
  }
}

// 使用时注入具体实现
const mysqlDb = new MySQLDatabase();
const userService = new UserService(mysqlDb);

// 切换数据库只需改变注入
const pgDb = new PostgreSQLDatabase();
const userService2 = new UserService(pgDb);

// 单元测试时轻松 mock
class MockDatabase implements Database {
  query(sql: string) {
    return { id: '1', name: '测试用户' };
  }
}
const testService = new UserService(new MockDatabase());
```

## 具体做法

数据封装要求对数据进行封装，限制数据的访问权限，特定数据只能由特定方法进行修改和管理。划分模块时将代码根据业务逻辑和功能拆分成一个个小的模块，尽量保证功能的单一性，不将不同功能的代码混合到一起。不同的业务逻辑之间存在依赖，他们的依赖关系应该形成一个单向无环图，最好是一棵树。模块之间进行依赖时应当依赖接口而非实现类，少用继承多用实现。模块的运行时依赖不应当由其自己获取，而应当由外界进行注入。
