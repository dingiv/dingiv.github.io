---
title: 数据完整性
order: 10
---

# 数据完整性

数据完整性是保证数据正确性和一致性的机制。没有完整性的数据是不可靠的，可能导致业务错误、决策失误、系统崩溃。本节从实体完整性、参照完整性、域完整性、用户定义完整性四个方面，介绍数据库如何保证数据完整性。

## 实体完整性

### 主键约束

主键是标识表中每一行的唯一标识，主键约束保证主键列的唯一性和非空性。主键的选择应该满足稳定、唯一、不可为空、不变。自增 ID 是常见的主键选择，UUID 是另一种选择但性能较差。

主键可以是单列主键，也可以是多列复合主键。复合主键适用于多列组合才能唯一标识一行的情况，例如订单明细表的 (order_id, product_id)。

```sql
-- 单列主键
CREATE TABLE users (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL
);

-- 复合主键
CREATE TABLE order_items (
  order_id BIGINT UNSIGNED NOT NULL,
  product_id BIGINT UNSIGNED NOT NULL,
  quantity INT NOT NULL,
  PRIMARY KEY (order_id, product_id)
);
```

主键的索引自动创建，主键查询使用索引，性能高。主键应该尽量短，因为主键会被外键引用，主键过长会加大索引和存储开销。主键应该是无意义的，业务字段可能有变化，不适合作为主键。

### UNIQUE 约束

UNIQUE 约束保证列或列组合的唯一性，与主键的区别是 UNIQUE 约束允许 NULL 值（多个 NULL 被认为是不同的）。一个表只能有一个主键，但可以有多个 UNIQUE 约束。

```sql
-- UNIQUE 约束
CREATE TABLE users (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  email VARCHAR(100) UNIQUE,
  phone VARCHAR(20)
);

CREATE UNIQUE INDEX uk_user_phone ON users(phone);
```

UNIQUE 约束通过唯一索引实现，查询时可以使用索引。UNIQUE 约束的列应该有明确的业务含义，例如用户名、邮箱、手机号应该唯一。UNIQUE 约束可以用于多列，例如 (first_name, last_name) 的组合唯一。

## 参照完整性

### 外键约束

外键约束保证表之间的引用完整性，外键列的值必须在被引用表中存在或为 NULL。外键约束防止出现孤立的记录，例如订单表的用户 ID 必须在用户表中存在。

外键约束的语法是 `FOREIGN KEY (col) REFERENCES ref_table(ref_col) [ON DELETE action] [ON UPDATE action]`。级联操作包括 CASCADE、SET NULL、RESTRICT、NO ACTION、SET DEFAULT。

```sql
-- 外键约束
CREATE TABLE orders (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  user_id BIGINT UNSIGNED NOT NULL,
  order_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT ON UPDATE CASCADE
);
```

ON DELETE CASCADE 表示删除被引用行时，删除引用行。ON DELETE SET NULL 表示删除被引用行时，外键列设为 NULL（需要外键列允许 NULL）。ON DELETE RESTRICT 表示删除被引用行时，阻止删除（这是默认行为）。

ON UPDATE CASCADE 表示更新被引用行时，更新引用行的外键值。ON UPDATE RESTRICT 表示更新被引用行时，阻止更新。

### 外键的性能影响

外键约束需要维护索引，增加了写入开销。外键检查需要查询被引用表，增加了查询延迟。外键约束可能导致锁等待，因为需要锁定被引用行。

外键约束在高并发场景下可能成为瓶颈，某些互联网公司选择在应用层保证参照完整性，而不使用外键约束。应用层保证一致性的优势是性能高、灵活，劣势是需要开发额外的逻辑，可能遗漏某些场景。

### 外键的设计建议

外键的使用应该权衡数据一致性和性能。对于核心业务关系，例如订单和用户、订单和商品，建议使用外键约束。对于非核心关系或高并发场景，可以考虑应用层保证。

外键列必须建立索引，MySQL 要求外键必须有索引，PostgreSQL 不强制但强烈建议。外键索引不仅用于外键检查，也用于关联查询，提高 JOIN 性能。

## 域完整性

### 数据类型

数据类型是域完整性的基础，正确的数据类型可以防止无效数据的插入。例如年龄字段使用 TINYINT 而非 VARCHAR，可以防止插入非数字或超范围的值。

MySQL 的数据类型包括整数类型（TINYINT、SMALLINT、INT、BIGINT）、浮点类型（FLOAT、DOUBLE）、定点类型（DECIMAL）、字符串类型（CHAR、VARCHAR）、文本类型（TEXT）、日期时间类型（DATE、TIME、DATETIME、TIMESTAMP）、二进制类型（BINARY、VARBINARY、BLOB）。

```sql
-- 好的数据类型设计
CREATE TABLE users (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  age TINYINT UNSIGNED NOT NULL COMMENT '0-255',
  balance DECIMAL(12,2) NOT NULL DEFAULT 0.00 COMMENT '精确到分',
  username VARCHAR(50) NOT NULL,
  bio TEXT COMMENT '个人简介',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### NOT NULL 约束

NOT NULL 约束保证列必须有值，不能为 NULL。NULL 表示未知或缺失，与空字符串或 0 不同。NOT NULL 约束应该根据业务语义使用，核心业务字段应该 NOT NULL，可选字段允许 NULL。

NOT NULL 约束的优势是减少 NULL 判断的复杂性，提高查询性能。NULL 判断需要使用 IS NULL 或 IS NOT NULL，不能使用 = 或 !=。聚合函数如 COUNT、SUM 会忽略 NULL 值。

```sql
-- NOT NULL 约束
CREATE TABLE users (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL,
  email VARCHAR(100) NOT NULL,
  phone VARCHAR(20) COMMENT '允许 NULL，表示未填写',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### DEFAULT 约束

DEFAULT 约束为列提供默认值，插入时如果没有指定值，使用默认值。默认值可以是常量、表达式、函数。常见的默认值包括 0、空字符串、CURRENT_TIMESTAMP。

```sql
-- DEFAULT 约束
CREATE TABLE users (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  status TINYINT NOT NULL DEFAULT 0 COMMENT '0-未激活',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

DEFAULT 约束可以简化应用层逻辑，应用层不需要显式设置默认值。DEFAULT 约束与 NOT NULL 配合使用，保证列既有默认值又不允许 NULL。

### CHECK 约束

CHECK 约束定义列值必须满足的条件，例如年龄必须大于 0、余额必须非负。CHECK 约束在 MySQL 8.0.16+ 版本完整支持，之前的版本被解析但被忽略。PostgreSQL 完整支持 CHECK 约束。

```sql
-- CHECK 约束
CREATE TABLE users (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  age TINYINT UNSIGNED NOT NULL CHECK (age >= 0 AND age <= 150),
  balance DECIMAL(12,2) NOT NULL DEFAULT 0.00 CHECK (balance >= 0),
  email VARCHAR(100) NOT NULL CHECK (email LIKE '%@%.%')
);
```

CHECK 约束的优势是将数据验证逻辑放在数据库，应用层不需要重复实现。CHECK 约束的劣势是增加数据库负担，复杂的 CHECK 条件可能影响写入性能。对于简单的条件检查，CHECK 约束是合适的；对于复杂的业务规则，应用层实现更灵活。

## 用户定义完整性

### 触发器

触发器是在 INSERT、UPDATE、DELETE 操作发生时自动执行的存储程序，可以用于实现复杂的业务规则。触发器可以验证数据、自动计算、记录日志。

触发器的优势是在数据库层实现逻辑，应用层无需关心。触发器的劣势是逻辑分散、难以调试、影响性能。触发器的使用应该谨慎，优先考虑约束和默认值，其次考虑触发器。

```sql
-- 触发器：余额不能透支
DELIMITER //
CREATE TRIGGER check_balance_before_update
BEFORE UPDATE ON accounts
FOR EACH ROW
BEGIN
  IF NEW.balance < 0 THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = '余额不能透支';
  END IF;
END //
DELIMITER ;
```

### 存储过程

存储过程可以封装复杂的业务逻辑，在应用层调用存储过程而非直接执行 SQL。存储过程可以保证数据的一致性，因为所有操作在数据库层完成，减少了应用层的错误。

存储过程的争议在于代码可维护性和数据库可移植性。存储过程将业务逻辑放在数据库，导致代码分散，难以测试和版本控制。现代应用倾向于将业务逻辑放在应用层，存储过程只用于简单的数据库操作。

```sql
-- 存储过程：转账
DELIMITER //
CREATE PROCEDURE transfer(
  IN from_account INT,
  IN to_account INT,
  IN amount DECIMAL(10,2),
  OUT result INT
)
BEGIN
  DECLARE from_balance DECIMAL(10,2);
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    ROLLBACK;
    SET result = -1;
  END;

  START TRANSACTION;
  SELECT balance INTO from_balance FROM accounts WHERE id = from_account FOR UPDATE;
  IF from_balance < amount THEN
    SET result = -2;
    ROLLBACK;
  ELSE
    UPDATE accounts SET balance = balance - amount WHERE id = from_account;
    UPDATE accounts SET balance = balance + amount WHERE id = to_account;
    COMMIT;
    SET result = 0;
  END IF;
END //
DELIMITER ;
```

### 应用层验证

应用层验证是最灵活的数据完整性保证方式，可以在应用层实现复杂的业务规则。应用层验证的优势是灵活、可测试、可版本控制。劣势是需要开发额外的逻辑，可能遗漏某些场景，容易出现不一致。

应用层验证应该与数据库约束配合使用，数据库约束保证基本完整性，应用层验证保证业务规则。例如数据库约束保证邮箱格式正确，应用层验证保证邮箱域名有效。

数据完整性是数据库质量的基础，实体完整性保证行的唯一标识，参照完整性保证表之间的关系，域完整性保证列值的有效性，用户定义完整性保证业务规则的满足。完整性需要在设计阶段充分考虑，在开发阶段严格执行，在运维阶段持续监控。
