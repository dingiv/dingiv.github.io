---
title: 数据库规范
order: 1
---

# 数据库规范

数据库规范是团队协作的基础，良好的规范可以提高代码可读性、减少错误、便于维护。规范的制定应该基于实践，避免过度设计。规范一旦制定，需要严格执行和定期审查。

## 命名规范

### 库表命名

数据库名使用小写字母和下划线，不超过 32 个字符。建议使用项目或业务模块名，例如 `ecommerce`、`user_service`。避免使用 MySQL 保留字，避免使用数字开头。

表名使用小写字母和下划线，不超过 32 个字符。建议使用名词复数形式，例如 `users`、`orders`、`order_items`。关联表使用两个表名拼接，例如 `user_roles`、`order_products`。表名应该反映表的含义，避免缩写，如果必须缩写需要在团队文档中说明。

```sql
-- 好的命名
CREATE TABLE users (
  id BIGINT PRIMARY KEY
);

CREATE TABLE order_items (
  id BIGINT PRIMARY KEY,
  order_id BIGINT NOT NULL,
  product_id BIGINT NOT NULL
);

-- 不好的命名
CREATE TABLE usr (
  id BIGINT PRIMARY KEY
);

CREATE TABLE oi (
  id BIGINT PRIMARY KEY
);
```

### 字段命名

字段名使用小写字母和下划线，不超过 32 个字符。字段名应该使用有意义的名称，避免缩写。主键字段统一使用 `id`，外键字段使用 `关联表名_id`，例如 `user_id`、`order_id`。

布尔字段使用 `is_`、`has_`、`can_` 前缀，例如 `is_active`、`has_verified`、`can_delete`。时间字段使用 `_at`、`_on` 后缀，`_at` 表示精确时间，`_on` 表示日期，例如 `created_at`、`updated_at`、`deleted_at`、`birth_on`。

```sql
-- 好的字段命名
CREATE TABLE users (
  id BIGINT PRIMARY KEY,
  username VARCHAR(50) NOT NULL,
  email VARCHAR(100) NOT NULL,
  is_active BOOLEAN DEFAULT TRUE,
  can_login BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 不好的字段命名
CREATE TABLE users (
  id BIGINT PRIMARY KEY,
  uname VARCHAR(50) NOT NULL,
  isact BOOLEAN DEFAULT TRUE,
  ctime TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 索引命名

索引名使用 `idx_表名_字段名` 格式，例如 `idx_users_email`、`idx_orders_user_id`。唯一索引使用 `uk_表名_字段名` 格式，例如 `uk_users_email`。全文索引使用 `ft_表名_字段名` 格式，例如 `ft_articles_content`。

复合索引包含所有字段，字段顺序与索引定义一致，例如 `idx_orders_user_id_status`。如果索引名过长超过 64 个字符，可以省略表名或字段名前缀，但需要保持可读性。

```sql
-- 好的索引命名
CREATE INDEX idx_users_email ON users(email);
CREATE UNIQUE INDEX uk_users_username ON users(username);
CREATE INDEX idx_orders_user_id_status ON orders(user_id, status);

-- 不好的索引命名
CREATE INDEX idx1 ON users(email);
CREATE UNIQUE INDEX unique_username ON users(username);
```

## 字段设计规范

### 数据类型选择

数据类型选择的原则是够用即可，避免过度分配。整数类型根据范围选择：TINYINT 范围 -128 到 127，SMALLINT 范围 -32768 到 32767，INT 范围约 -21 亿到 21 亿，BIGINT 范围约 -9×10^18 到 9×10^18。ID 字段统一使用 BIGINT，为未来扩展留空间。

字符串类型根据长度选择：CHAR 固定长度，适合长度固定的字段如 MD5、UUID。VARCHAR 可变长度，适合长度不定的字段。VARCHAR 的长度应该根据实际需求设置，不是越大越好，因为过长的 VARCHAR 会影响索引性能。

```sql
-- 好的类型选择
CREATE TABLE users (
  id BIGINT PRIMARY KEY,
  age TINYINT UNSIGNED COMMENT '年龄 0-255',
  status TINYINT NOT NULL DEFAULT 0 COMMENT '状态 0-10',
  username VARCHAR(50) NOT NULL COMMENT '用户名',
  email VARCHAR(100) NOT NULL COMMENT '邮箱',
  phone VARCHAR(20) COMMENT '手机号'
);

-- 不好的类型选择
CREATE TABLE users (
  id INT PRIMARY KEY COMMENT 'INT 可能不够',
  age INT COMMENT 'TINYINT 足够',
  status INT COMMENT 'TINYINT 足够',
  username VARCHAR(255) COMMENT '过长',
  email TEXT COMMENT '不应该用 TEXT'
);
```

### 日期时间类型

MySQL 中日期时间类型包括 DATE、TIME、DATETIME、TIMESTAMP。DATE 存储日期，TIME 存储时间，DATETIME 存储日期和时间，不依赖时区。TIMESTAMP 存储时间戳，依赖时区，自动转换。

DATETIME 和 TIMESTAMP 的选择取决于需求。如果需要跨时区，使用 TIMESTAMP。如果不需要时区转换，使用 DATETIME。DATETIME 的范围更大（1000 年到 9999 年），TIMESTAMP 的范围较小（1970 年到 2038 年）。

```sql
-- 好的日期时间设计
CREATE TABLE orders (
  id BIGINT PRIMARY KEY,
  order_date DATE COMMENT '订单日期',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
);
```

### NULL 与默认值

字段默认允许 NULL，但业务关键字段应该设置为 NOT NULL。NULL 的语义是未知，与空字符串或 0 不同。NULL 会导致索引复杂度增加，查询条件需要使用 IS NULL 而不是 = NULL。

默认值应该根据业务逻辑设置，整数字段默认 0，字符串字段默认空字符串，布尔字段默认 FALSE 或 TRUE，时间字段默认 CURRENT_TIMESTAMP。外键字段应该设置默认值或 NOT NULL，避免外键为 NULL。

```sql
-- 好的 NULL 和默认值设计
CREATE TABLE users (
  id BIGINT PRIMARY KEY,
  username VARCHAR(50) NOT NULL,
  email VARCHAR(100) NOT NULL,
  phone VARCHAR(20) COMMENT '允许 NULL，表示未填写',
  status TINYINT NOT NULL DEFAULT 0 COMMENT '状态默认为 0',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 索引设计规范

### 索引原则

索引是性能优化的关键，但索引不是越多越好。每个索引都需要维护成本，写入时需要更新索引，占用存储空间。建议遵循以下原则：为 WHERE、JOIN、ORDER BY、GROUP BY 子句的列创建索引。高选择性列适合索引，低选择性列不适合索引。复合索引遵循最左前缀原则。

避免冗余索引，例如索引 (a) 和索引 (a, b)，前者是后者的冗余。避免重复索引，例如多个索引包含相同的列。定期审查索引使用情况，删除未使用的索引。

### 主键设计

主键使用 BIGINT 自增 ID，避免使用 UUID 或业务字段作为主键。BIGINT 自增 ID 占用空间小、插入性能好、索引效率高。UUID 占用空间大、插入性能差、索引效率低。业务字段可能变化，不适合作为主键。

如果需要全局唯一 ID，可以使用雪花算法生成 BIGINT ID，而不是使用 UUID。雪花算法生成的 ID 有序递增，适合作为主键和索引。

```sql
-- 好的主键设计
CREATE TABLE users (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL,
  email VARCHAR(100) NOT NULL
);

-- 不好的主键设计
CREATE TABLE users (
  id CHAR(36) PRIMARY KEY COMMENT 'UUID 占用空间大'
);

CREATE TABLE users (
  username VARCHAR(50) PRIMARY KEY COMMENT '业务字段可能变化'
);
```

### 外键设计

外键约束保证数据一致性，但会影响性能。外键索引是必须的，MySQL 要求外键必须有索引。如果应用层保证数据一致性，可以不使用外键约束，只使用索引。

外键列的类型和字符集必须与引用列一致，否则无法创建外键。外键的级联操作需要谨慎，CASCADE 级联删除可能导致大量数据被意外删除，SET NULL 可能导致数据不一致。

```sql
-- 好的外键设计
CREATE TABLE orders (
  id BIGINT PRIMARY KEY,
  user_id BIGINT NOT NULL,
  INDEX idx_user_id (user_id),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT
);
```

### 复合索引设计

复合索引的列顺序遵循最左前缀原则，将区分度高的列放在前面。等值查询列放在前面，范围查询列放在后面。覆盖索引包含查询所需的所有列，避免回表。

复合索引不是列越多越好，超过 5 列的复合索引维护成本高，可能不被优化器选择。如果查询需要多个复合索引，考虑是否可以合并或重新设计。

```sql
-- 好的复合索引设计
CREATE TABLE orders (
  id BIGINT PRIMARY KEY,
  user_id BIGINT NOT NULL,
  status TINYINT NOT NULL,
  created_at DATETIME NOT NULL,
  INDEX idx_user_status_created (user_id, status, created_at)
);

-- 可以支持以下查询
SELECT * FROM orders WHERE user_id = 1;
SELECT * FROM orders WHERE user_id = 1 AND status = 2;
SELECT * FROM orders WHERE user_id = 1 AND status = 2 ORDER BY created_at DESC;
```

## SQL 编写规范

### 查询优化

避免 SELECT *，只查询需要的列。SELECT * 会增加网络传输、内存占用、索引失效的可能性。明确列出需要的列，有利于代码阅读和维护。

避免在索引列上使用函数，这会导致索引失效。如果必须使用函数，考虑使用函数索引或重写查询。例如 `WHERE DATE(created_at) = '2023-01-01'` 改为 `WHERE created_at >= '2023-01-01' AND created_at < '2023-01-02'`。

避免隐式类型转换，确保查询条件的类型与字段类型一致。例如字段类型是 VARCHAR，查询条件使用字符串而非整数。

```sql
-- 好的查询
SELECT id, username, email FROM users WHERE id = 12345;
SELECT * FROM orders WHERE user_id = 1 AND status = 2;

-- 不好的查询
SELECT * FROM users WHERE id = 12345;
SELECT * FROM users WHERE DATE(created_at) = '2023-01-01';
SELECT * FROM users WHERE phone = 13800138000;
```

### JOIN 优化

JOIN 使用内连接而非外连接，除非业务需要外连接。内连接性能更好，优化器空间更大。JOIN 确保连接字段有索引，JOIN 的表数量不超过 5 个，过多的 JOIN 会影响性能和可读性。

子查询优先考虑是否可以转换为 JOIN，某些子查询的执行效率低于 JOIN。NOT EXISTS 和 NOT IN 的选择取决于数据量和索引，小数据量使用 NOT IN，大数据量使用 NOT EXISTS。

```sql
-- 好的 JOIN 查询
SELECT u.username, o.order_id
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.status = 2;

-- 好的子查询转 JOIN
SELECT u.username, o.order_id
FROM users u
INNER JOIN orders o ON u.id = o.user_id;
-- 而非
SELECT u.username
FROM users u
WHERE u.id IN (SELECT user_id FROM orders);
```

### 事务规范

事务要尽可能小，减少锁持有时间。长事务会导致锁等待、死锁、undo log 膨胀。事务中避免网络请求、文件操作、复杂计算，这些操作应该在事务外完成。

事务隔离级别根据业务需求选择，READ COMMITTED 是大多数场景的合适选择。SERIALIZABLE 隔离级别性能差，很少使用。显式开启事务，避免隐式事务和自动提交。

```sql
-- 好的事务使用
START TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- 不好的长事务
START TRANSACTION;
SELECT * FROM large_table; -- 避免大查询
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- 网络请求
sleep(10);
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

数据库规范是团队协作的基础，但规范不是一成不变的。随着业务发展和技术演进，规范需要定期审查和更新。规范的制定应该基于共识，执行需要工具支持，例如代码审查、自动化检查、CI 集成。
