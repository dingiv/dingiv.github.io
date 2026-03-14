---
title: SQL 语法基础
order: 9
---

# SQL 语法基础

SQL（Structured Query Language）是关系数据库的标准查询语言，掌握 SQL 语法是数据库开发和运维的基础。本节介绍 DDL、DML、DQL 的核心语法，以及视图、存储过程、触发器的高级特性。

## DDL 数据定义语言

### CREATE 创建

CREATE 语句用于创建数据库对象，包括数据库、表、索引、视图。创建数据库时可以指定字符集和排序规则，例如 `CREATE DATABASE db_name CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci`。

创建表时需要指定列名、数据类型、约束。常见约束包括 PRIMARY KEY 主键、NOT NULL 非空、UNIQUE 唯一、DEFAULT 默认值、AUTO_INCREMENT 自增。外键约束通过 FOREIGN KEY 定义，引用其他表的主键。

```sql
-- 创建表
CREATE TABLE users (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  email VARCHAR(100) NOT NULL,
  age TINYINT UNSIGNED DEFAULT 0,
  status TINYINT NOT NULL DEFAULT 0 COMMENT '0-未激活, 1-正常, 2-禁用',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_email (email),
  INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表';
```

CREATE INDEX 语句用于创建索引，包括普通索引、唯一索引、全文索引、空间索引。索引可以在创建表时定义，也可以通过 CREATE INDEX 添加。复合索引包含多个列，列顺序影响索引使用。

```sql
-- 创建索引
CREATE INDEX idx_user_email ON users(email);
CREATE UNIQUE INDEX uk_user_username ON users(username);
CREATE FULLTEXT INDEX ft_article_content ON articles(content);
CREATE INDEX idx_order_user_status ON orders(user_id, status);
```

### ALTER 修改

ALTER 语句用于修改已有表的结构，包括添加列、删除列、修改列、添加约束、删除约束。ALTER TABLE 是耗时操作，大表的结构变更可能需要数小时，建议在业务低峰期执行。

```sql
-- 添加列
ALTER TABLE users ADD COLUMN phone VARCHAR(20) AFTER email;

-- 修改列
ALTER TABLE users MODIFY COLUMN username VARCHAR(100) NOT NULL;

-- 删除列
ALTER TABLE users DROP COLUMN phone;

-- 添加主键
ALTER TABLE users ADD PRIMARY KEY (id);

-- 删除主键
ALTER TABLE users DROP PRIMARY KEY;
```

MySQL 5.6+ 支持 Online DDL，某些结构变更不锁表、不阻塞读写。Online DDL 通过 ALGORITHM 和 LOCK 子句控制，ALGORITHM=INPLACE 是原地变更，ALGORITHM=COPY 是复制表。LOCK=NONE 不锁表，LOCK=SHARED 共享锁，LOCK=EXCLUSIVE 排他锁。

### DROP 删除

DROP 语句用于删除数据库对象，包括数据库、表、索引、视图。DROP 是危险操作，会永久删除数据和对象，无法直接恢复。执行 DROP 前应该确认，生产环境建议使用备份或删除保护。

DROP DATABASE 删除数据库和所有表，DROP TABLE 删除表和数据，DROP INDEX 删除索引。DROP VIEW 删除视图。DROP TRIGGER 删除触发器。

```sql
-- 删除表
DROP TABLE users;

-- 删除数据库
DROP DATABASE db_name;

-- 删除索引
DROP INDEX idx_user_email ON users;
```

TRUNCATE TABLE 是快速清空表的操作，比 DELETE 快得多，因为 TRUNCATE 不记录逐行删除，直接重置表。TRUNCATE 是 DDL 操作，隐式提交，无法回滚。TRUNCATE 重置自增 ID，DELETE 不影响自增 ID。

## DML 数据操作语言

### INSERT 插入

INSERT 语句用于插入数据，支持单行插入、多行插入、查询插入。单行插入使用 `INSERT INTO table VALUES (value1, value2)` 或 `INSERT INTO table (col1, col2) VALUES (value1, value2)`。多行插入使用 `INSERT INTO table VALUES (...), (...), (...)`。

```sql
-- 单行插入
INSERT INTO users (username, email, age) VALUES ('alice', 'alice@example.com', 25);

-- 多行插入
INSERT INTO users (username, email, age) VALUES
  ('bob', 'bob@example.com', 30),
  ('charlie', 'charlie@example.com', 35);

-- 查询插入
INSERT INTO archived_users
SELECT * FROM users WHERE created_at < '2020-01-01';
```

INSERT IGNORE 忽略唯一键冲突的错误，继续执行。INSERT ON DUPLICATE KEY UPDATE 在唯一键冲突时更新数据。REPLACE INTO 在唯一键冲突时删除旧记录并插入新记录。

```sql
-- INSERT IGNORE
INSERT IGNORE INTO users (id, username, email) VALUES (1, 'alice', 'alice@example.com');

-- ON DUPLICATE KEY UPDATE
INSERT INTO users (id, username, email) VALUES (1, 'alice', 'new@example.com')
ON DUPLICATE KEY UPDATE email = VALUES(email);

-- REPLACE INTO
REPLACE INTO users (id, username, email) VALUES (1, 'alice', 'new@example.com');
```

### UPDATE 更新

UPDATE 语句用于更新已有数据，支持单表更新、多表更新。单表更新使用 `UPDATE table SET col1 = value1 WHERE condition`。多表更新使用 JOIN 关联多个表。

```sql
-- 单表更新
UPDATE users SET status = 1 WHERE id = 1;

-- 多表更新
UPDATE users u
JOIN orders o ON u.id = o.user_id
SET u.last_order_at = o.created_at
WHERE o.status = 'completed';
```

UPDATE 支持 ORDER BY 和 LIMIT，可以控制更新顺序和数量。`UPDATE users SET status = 1 ORDER BY id LIMIT 100` 更新 100 条记录，按 ID 顺序更新。

### DELETE 删除

DELETE 语句用于删除数据，支持单表删除、多表删除。单表删除使用 `DELETE FROM table WHERE condition`。多表删除使用 JOIN 关联多个表。

```sql
-- 单表删除
DELETE FROM users WHERE id = 1;

-- 多表删除
DELETE users, orders
FROM users
JOIN orders ON users.id = orders.user_id
WHERE users.id = 1;
```

DELETE 支持 ORDER BY 和 LIMIT，可以控制删除顺序和数量。`DELETE FROM users ORDER BY id LIMIT 100` 删除 100 条记录，按 ID 顺序删除。

DELETE 是 DML 操作，可以回滚。TRUNCATE 是 DDL 操作，无法回滚。DELETE 触发触发器，TRUNCATE 不触发触发器。DELETE 记录逐行删除，TRUNCATE 直接重置表。

## DQL 数据查询语言

### SELECT 基础

SELECT 语句用于查询数据，基本语法是 `SELECT columns FROM table WHERE condition GROUP BY columns HAVING condition ORDER BY columns LIMIT offset, count`。

SELECT * 返回所有列，生产环境应该明确列出需要的列，避免 SELECT *。SELECT DISTINCT 返回去重后的结果。SELECT 可以使用表达式和函数，例如 `SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM users`。

### WHERE 过滤

WHERE 子句用于过滤行，支持比较运算符、逻辑运算符、范围查询、模糊匹配。比较运算符包括 =、!=、<>、>、<、>=、<=。逻辑运算符包括 AND、OR、NOT。范围查询使用 BETWEEN、IN。模糊匹配使用 LIKE、REGEXP。

```sql
-- 比较运算符
SELECT * FROM users WHERE age > 18;

-- 逻辑运算符
SELECT * FROM users WHERE age > 18 AND status = 1;

-- 范围查询
SELECT * FROM users WHERE age BETWEEN 18 AND 30;
SELECT * FROM users WHERE status IN (1, 2, 3);

-- 模糊匹配
SELECT * FROM users WHERE username LIKE 'alice%';
SELECT * FROM users WHERE email REGEXP '^[a-z]+@example\\.com$';
```

### JOIN 连接

JOIN 用于关联多个表，包括 INNER JOIN 内连接、LEFT JOIN 左连接、RIGHT JOIN 右连接、FULL OUTER JOIN 全连接（MySQL 不支持）。内连接返回匹配的行，左连接返回左表所有行和右表匹配的行，右连接返回右表所有行和左表匹配的行。

```sql
-- 内连接
SELECT u.username, o.order_id
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- 左连接
SELECT u.username, o.order_id
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- 自连接
SELECT e1.name AS employee, e2.name AS manager
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.id;
```

### GROUP BY 分组

GROUP BY 用于分组聚合，常见的聚合函数包括 COUNT、SUM、AVG、MIN、MAX。GROUP BY 后的 SELECT 只能包含分组列和聚合函数，包含其他列会导致错误或非确定结果。

```sql
-- 分组聚合
SELECT status, COUNT(*) AS count
FROM users
GROUP BY status;

-- 多列分组
SELECT DATE(created_at) AS date, status, COUNT(*) AS count
FROM users
GROUP BY DATE(created_at), status;
```

HAVING 用于过滤分组结果，WHERE 用于过滤行。HAVING 在 GROUP BY 之后执行，WHERE 在 GROUP BY 之前执行。HAVING 可以使用聚合函数，WHERE 不能。

```sql
-- HAVING 过滤分组
SELECT status, COUNT(*) AS count
FROM users
GROUP BY status
HAVING count > 100;
```

### ORDER BY 排序

ORDER BY 用于排序结果，支持 ASC 升序和 DESC 降序。可以按多个列排序，优先级从左到右。ORDER BY 可以使用表达式，例如 `ORDER BY score DESC, created_at ASC`。

### LIMIT 分页

LIMIT 用于限制返回行数，`LIMIT count` 返回前 count 行，`LIMIT offset, count` 从第 offset 行开始返回 count 行。分页查询使用 `LIMIT (page-1)*pageSize, pageSize`。

```sql
-- 分页查询
SELECT * FROM users
ORDER BY id
LIMIT 0, 10; -- 第一页
LIMIT 10, 10; -- 第二页
```

## 视图

### CREATE VIEW 创建视图

视图是虚拟表，不存储数据，数据来源于基表的查询结果。视图可以简化复杂查询，提供数据抽象，实现权限控制。创建视图使用 CREATE VIEW 语句。

```sql
-- 创建视图
CREATE VIEW user_orders AS
SELECT u.id, u.username, COUNT(o.id) AS order_count, SUM(o.amount) AS total_amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username;
```

视图可以嵌套，视图可以基于其他视图创建。但嵌套视图会影响性能，查询时需要展开所有视图。视图的限制包括：不能包含子查询、不能使用变量、不能使用临时表、某些视图不支持更新。

### 更新视图

某些视图可以更新，更新视图会更新基表。可更新视图的条件包括：视图只包含一个基表、不包含 DISTINCT、GROUP BY、聚合函数、UNION。复杂视图通常是只读的，不能直接更新。

## 存储过程

### CREATE PROCEDURE 创建存储过程

存储过程是预编译的 SQL 语句集合，可以接收参数、执行复杂逻辑、返回结果。存储过程的优势是减少网络往返、提高性能、封装业务逻辑。创建存储过程使用 CREATE PROCEDURE 语句。

```sql
-- 创建存储过程
DELIMITER //
CREATE PROCEDURE get_user_orders(IN user_id INT)
BEGIN
  SELECT o.id, o.created_at, o.amount
  FROM orders o
  WHERE o.user_id = user_id
  ORDER BY o.created_at DESC;
END //
DELIMITER ;

-- 调用存储过程
CALL get_user_orders(1);
```

存储过程支持三种参数：IN 输入参数、OUT 输出参数、INOUT 输入输出参数。存储过程可以使用变量、条件语句、循环、游标、异常处理。

```sql
-- 复杂存储过程
DELIMITER //
CREATE PROCEDURE transfer_money(
  IN from_account INT,
  IN to_account INT,
  IN amount DECIMAL(10,2),
  OUT result INT
)
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    ROLLBACK;
    SET result = -1;
  END;

  START TRANSACTION;
  UPDATE accounts SET balance = balance - amount WHERE id = from_account;
  UPDATE accounts SET balance = balance + amount WHERE id = to_account;
  COMMIT;
  SET result = 0;
END //
DELIMITER ;
```

存储过程的争议在于代码可维护性和数据库可移植性。存储过程将业务逻辑放在数据库，导致代码分散，难以测试和版本控制。存储过程通常是数据库特定的语法，迁移到其他数据库需要重写。现代应用倾向于将业务逻辑放在应用层，存储过程只用于简单的数据库操作。

## 触发器

### CREATE TRIGGER 创建触发器

触发器是在表上定义的自动执行的存储程序，当 INSERT、UPDATE、DELETE 操作发生时触发。触发器可以用于数据验证、自动更新、审计日志。创建触发器使用 CREATE TRIGGER 语句。

```sql
-- 创建触发器：自动更新 updated_at
CREATE TRIGGER update_users_timestamp
BEFORE UPDATE ON users
FOR EACH ROW
SET NEW.updated_at = CURRENT_TIMESTAMP;

-- 创建触发器：审计日志
CREATE TRIGGER log_users_update
AFTER UPDATE ON users
FOR EACH ROW
INSERT INTO users_audit (user_id, old_value, new_value, action, action_time)
VALUES (OLD.id, OLD.username, NEW.username, 'UPDATE', CURRENT_TIMESTAMP);
```

触发器有六个触发时机：BEFORE INSERT、AFTER INSERT、BEFORE UPDATE、AFTER UPDATE、BEFORE DELETE、AFTER DELETE。BEFORE 触发器可以修改 NEW 值，AFTER 触发器不能修改。

NEW 和 OLD 是触发器的特殊变量，NEW 表示新值（INSERT 和 UPDATE），OLD 表示旧值（UPDATE 和 DELETE）。DELETE 操作没有 NEW，INSERT 操作没有 OLD。

### 触发器限制

触发器的限制包括：触发器不能调用事务控制语句（COMMIT、ROLLBACK）、触发器不能调用动态 SQL、触发器执行时间影响原操作的响应时间、触发器的错误会导致原操作失败。触发器的使用应该谨慎，过度使用会导致逻辑复杂、难以调试。

SQL 语法是数据库开发和运维的基础，DDL 定义结构，DML 操作数据，DQL 查询数据。视图、存储过程、触发器提供了高级特性，但也增加了复杂度。掌握 SQL 语法需要理论知识和实践练习，推荐通过实际项目深入理解。
