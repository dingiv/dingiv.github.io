# SQL
SQL（Structured Query Language）是用于管理关系型数据库的标准语言。本文介绍SQL的基础语法和常用操作。


## 主流数据库对比

### 关系型数据库（RDBMS）
1. **MySQL**
   - 开源关系型数据库
   - 广泛用于Web应用
   - 支持事务和ACID特性
   - 适合OLTP（联机事务处理）
   - 社区版免费，企业版收费

2. **PostgreSQL**
   - 功能强大的开源数据库
   - 支持复杂查询和自定义类型
   - 提供JSON支持
   - 适合复杂的数据分析
   - 完全开源，功能丰富

3. **SQLite**
   - 轻量级嵌入式数据库
   - 零配置
   - 适合移动应用和桌面应用
   - 单文件存储
   - 无需服务器

4. **Oracle**
   - 企业级数据库
   - 高性能和高可用性
   - 丰富的企业特性
   - 适合大型企业应用
   - 商业软件，价格昂贵

### NoSQL数据库
1. **MongoDB**
   - 文档型数据库
   - 灵活的数据模型
   - 适合快速开发
   - 支持复杂查询
   - 社区版免费，企业版收费

2. **Redis**
   - 内存数据库
   - 键值存储
   - 支持多种数据结构
   - 适合缓存和会话存储
   - 开源免费

3. **Cassandra**
   - 列式数据库
   - 高可扩展性
   - 适合大数据应用
   - 分布式架构
   - 开源免费

4. **Neo4j**
   - 图数据库
   - 适合关系型数据
   - 支持复杂图查询
   - 适合社交网络分析
   - 社区版免费，企业版收费

## 数据库选择建议

### 选择关系型数据库的场景
1. 需要强一致性
2. 数据结构相对固定
3. 需要复杂的事务支持
4. 需要复杂的查询和连接操作

### 选择NoSQL数据库的场景
1. 需要高可扩展性
2. 数据结构灵活多变
3. 需要高性能读写
4. 处理大量非结构化数据

## 数据定义语言（DDL）

### 数据库操作
```sql
-- 创建数据库
CREATE DATABASE mydb;

-- 删除数据库
DROP DATABASE mydb;

-- 选择数据库
USE mydb;
```

### 表操作
```sql
-- 创建表
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    age INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 修改表
ALTER TABLE users 
ADD COLUMN phone VARCHAR(20);

-- 删除表
DROP TABLE users;
```

## 数据操作语言（DML）

### 插入数据
```sql
-- 插入单条数据
INSERT INTO users (username, email, age) 
VALUES ('john', 'john@example.com', 25);

-- 插入多条数据
INSERT INTO users (username, email, age) 
VALUES 
    ('alice', 'alice@example.com', 30),
    ('bob', 'bob@example.com', 28);
```

### 更新数据
```sql
-- 更新单条数据
UPDATE users 
SET email = 'new@example.com' 
WHERE id = 1;

-- 更新多条数据
UPDATE users 
SET age = age + 1 
WHERE age < 30;
```

### 删除数据
```sql
-- 删除单条数据
DELETE FROM users 
WHERE id = 1;

-- 删除多条数据
DELETE FROM users 
WHERE age > 50;
```

## 数据查询语言（DQL）

### 基本查询
```sql
-- 查询所有字段
SELECT * FROM users;

-- 查询特定字段
SELECT username, email FROM users;

-- 使用别名
SELECT username AS name, email AS mail FROM users;
```

### 条件查询
```sql
-- 简单条件
SELECT * FROM users WHERE age > 18;

-- 多条件
SELECT * FROM users 
WHERE age > 18 AND email LIKE '%@example.com';

-- IN 操作符
SELECT * FROM users 
WHERE age IN (20, 25, 30);

-- BETWEEN 操作符
SELECT * FROM users 
WHERE age BETWEEN 20 AND 30;
```

### 排序和分组
```sql
-- 排序
SELECT * FROM users 
ORDER BY age DESC, username ASC;

-- 分组
SELECT age, COUNT(*) as count 
FROM users 
GROUP BY age;

-- 分组后筛选
SELECT age, COUNT(*) as count 
FROM users 
GROUP BY age 
HAVING count > 1;
```

### 连接查询
```sql
-- 内连接
SELECT u.username, p.title 
FROM users u 
INNER JOIN posts p ON u.id = p.user_id;

-- 左连接
SELECT u.username, p.title 
FROM users u 
LEFT JOIN posts p ON u.id = p.user_id;

-- 右连接
SELECT u.username, p.title 
FROM users u 
RIGHT JOIN posts p ON u.id = p.user_id;
```

### 子查询
```sql
-- 在WHERE中使用子查询
SELECT * FROM users 
WHERE age > (SELECT AVG(age) FROM users);

-- 在FROM中使用子查询
SELECT u.username, p.post_count 
FROM users u 
JOIN (SELECT user_id, COUNT(*) as post_count FROM posts GROUP BY user_id) p 
ON u.id = p.user_id;
```

## 事务控制

### 事务操作
```sql
-- 开始事务
START TRANSACTION;

-- 执行SQL语句
INSERT INTO users (username, email) VALUES ('test', 'test@example.com');
UPDATE accounts SET balance = balance - 100 WHERE user_id = 1;

-- 提交事务
COMMIT;

-- 回滚事务
ROLLBACK;
```

## 视图

### 视图操作
```sql
-- 创建视图
CREATE VIEW user_posts AS
SELECT u.username, p.title, p.created_at
FROM users u
JOIN posts p ON u.id = p.user_id;

-- 使用视图
SELECT * FROM user_posts WHERE username = 'john';

-- 删除视图
DROP VIEW user_posts;
```

## 存储过程

### 存储过程示例
```sql
-- 创建存储过程
DELIMITER //
CREATE PROCEDURE GetUserPosts(IN userId INT)
BEGIN
    SELECT p.* FROM posts p WHERE p.user_id = userId;
END //
DELIMITER ;

-- 调用存储过程
CALL GetUserPosts(1);
``` 