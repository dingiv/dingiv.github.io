---
title: 数据库设计
order: 8
---

# 数据库设计

数据库设计是应用架构的基础，良好的设计可以支撑业务增长，糟糕的设计会成为技术债务。本节从范式理论、ER 建模、反范式化三个方面，介绍数据库设计的方法和实践。

## 范式理论

### 第一范式 1NF

第一范式要求每个字段都是不可分割的原子值。例如地址字段不应该包含省、市、区，应该拆分为独立的字段。违反第一范式会导致数据难以查询和更新，查询时需要字符串解析，更新时需要部分替换。

```sql
-- 违反 1NF
CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(50),
  address VARCHAR(100) -- "北京市朝阳区xxx街道"
);

-- 符合 1NF
CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(50),
  province VARCHAR(20),
  city VARCHAR(20),
  district VARCHAR(20),
  street VARCHAR(50)
);
```

### 第二范式 2NF

第二范式在 1NF 基础上，要求非主键字段完全依赖于主键，不能存在部分依赖。这主要针对复合主键的场景，如果某个字段只依赖主键的一部分，就不满足 2NF。

例如订单明细表中，主键是 (order_id, product_id)，商品名称只依赖 product_id，不依赖 order_id，这是部分依赖，不符合 2NF。解决方案是将商品信息拆分到独立的商品表。

```sql
-- 违反 2NF
CREATE TABLE order_items (
  order_id INT,
  product_id INT,
  product_name VARCHAR(50), -- 只依赖 product_id
  quantity INT,
  PRIMARY KEY (order_id, product_id)
);

-- 符合 2NF
CREATE TABLE order_items (
  order_id INT,
  product_id INT,
  quantity INT,
  PRIMARY KEY (order_id, product_id)
);

CREATE TABLE products (
  id INT PRIMARY KEY,
  name VARCHAR(50)
);
```

### 第三范式 3NF

第三范式在 2NF 基础上，要求非主键字段直接依赖于主键，不能存在传递依赖。如果字段 A 依赖主键，字段 B 依赖字段 A，这就是传递依赖，不符合 3NF。

例如订单表中的用户名字段依赖用户 ID，用户 ID 依赖订单 ID，这是传递依赖。解决方案是将用户信息拆分到独立的用户表。

```sql
-- 违反 3NF
CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT,
  username VARCHAR(50), -- 传递依赖
  order_time DATETIME
);

-- 符合 3NF
CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT,
  order_time DATETIME
);

CREATE TABLE users (
  id INT PRIMARY KEY,
  username VARCHAR(50)
);
```

### BCNF 范式

BCNF（Boyce-Codd Normal Form）是 3NF 的修正版本，要求每个决定因素都是候选键。在某些情况下，关系满足 3NF 但不满足 BCNF，例如学生、课程、教师的场景：学生选课由 (学生, 课程) 决定，课程教师由课程决定，教师课程由教师决定。这里存在多个重叠的候选键，BCNF 要求进一步分解。

### 范式的权衡

范式化的好处是消除冗余、避免更新异常、保持数据一致性。范式化的代价是查询时需要更多 JOIN，性能可能下降。实际应用中，3NF 是大多数场景的合理选择，既消除了大部分冗余，又不会导致过多的 JOIN。

反范式化是有意的冗余，通过牺牲范式化来换取性能。反范式化需要谨慎，需要在文档中记录冗余的设计原因，确保团队理解这种权衡。

## ER 建模

### 实体关系模型

实体关系模型使用实体、属性、关系描述现实世界。实体代表现实中的对象，如用户、订单、商品。属性描述实体的特征，如用户的姓名、年龄、地址。关系描述实体之间的关联，如用户下订单、订单包含商品。

ER 图使用矩形表示实体，椭圆表示属性，菱形表示关系。实体之间的基数包括一对一 (1:1)、一对多 (1:N)、多对多 (M:N)。例如用户和订单是一对多，订单和商品是多对多。

### 关系的建模

一对一关系可以通过两种方式建模：主键关联和外键关联。主键关联是两个实体共享相同的主键，例如用户和用户详情共享 user_id。外键关联是其中一个实体包含另一个实体的外键，例如用户包含地址表的 address_id。

一对多关系通过"多"端的外键建模，例如订单表包含 user_id 外键。查询某用户的订单时通过 user_id 过滤，查询某订单的用户时通过 user_id JOIN。

多对多关系需要中间表，例如订单和商品的关系通过 order_items 表建模。中间表包含两个实体的外键，还可以包含关系属性如数量、单价。

```sql
-- 一对多：用户-订单
CREATE TABLE users (
  id INT PRIMARY KEY,
  username VARCHAR(50)
);

CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT,
  order_time DATETIME,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 多对多：订单-商品
CREATE TABLE orders (
  id INT PRIMARY KEY,
  order_time DATETIME
);

CREATE TABLE products (
  id INT PRIMARY KEY,
  name VARCHAR(50)
);

CREATE TABLE order_items (
  order_id INT,
  product_id INT,
  quantity INT,
  price DECIMAL(10,2),
  PRIMARY KEY (order_id, product_id),
  FOREIGN KEY (order_id) REFERENCES orders(id),
  FOREIGN KEY (product_id) REFERENCES products(id)
);
```

### 继承的建模

继承是面向对象的概念，关系数据库没有直接的继承支持。继承可以通过三种方式建模：单表继承、类表继承、具体表继承。

单表继承是将所有类的字段放在一个表中，使用类型字段区分子类。这种方式查询简单，但字段多且很多字段为 NULL。

类表继承是父表包含公共字段，子表包含特有字段，子表主键同时是外键指向父表。这种方式避免 NULL 字段，但查询需要 JOIN。

具体表继承是每个子类一个表，父表只是逻辑概念，不存在物理表。这种方式查询简单，但无法查询父类实体。

```sql
-- 类表继承示例
CREATE TABLE people (
  id INT PRIMARY KEY,
  name VARCHAR(50),
  type VARCHAR(20) -- 'student' or 'teacher'
);

CREATE TABLE students (
  id INT PRIMARY KEY,
  grade INT,
  FOREIGN KEY (id) REFERENCES people(id)
);

CREATE TABLE teachers (
  id INT PRIMARY KEY,
  department VARCHAR(50),
  FOREIGN KEY (id) REFERENCES people(id)
);
```

## 反范式化

### 冗余设计

反范式化是通过引入冗余来提高性能。常见的冗余设计包括：在订单表中存储用户名字，避免每次查询订单时 JOIN 用户表。在文章表中存储分类名称，避免每次查询文章时 JOIN 分类表。

冗余设计需要考虑一致性维护。当用户名修改时，需要同步更新所有相关订单。这种一致性可以通过应用层维护，例如修改用户名时触发订单表的更新。也可以使用数据库触发器自动维护。

```sql
-- 反范式化：订单表冗余用户名
CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT,
  username VARCHAR(50), -- 冗余字段
  order_time DATETIME,
  FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### 汇总表

汇总是另一种反范式化，通过预计算汇总数据避免实时计算。例如在订单表之外，创建每日销售额汇总表，存储每天的订单数、销售额。查询销售额时直接读取汇总表，避免扫描大量订单。

汇总是牺牲实时性换取性能，适合不需要精确实时的场景。汇总是定期更新的，例如每小时或每天更新一次。如果实时性要求高，可以考虑缓存而不是汇总。

```sql
-- 汇总表：每日销售额
CREATE TABLE daily_sales (
  sale_date DATE PRIMARY KEY,
  order_count INT,
  total_amount DECIMAL(12,2)
);
```

### 计数表

计数是常见的查询需求，例如统计用户的粉丝数、文章的点赞数。实时计数需要扫描大量数据，性能差。计数表存储预计算的计数，查询时直接读取。

计数的维护可以通过触发器或应用层逻辑。例如用户关注时，触发器更新被关注者的粉丝计数。文章点赞时，应用层更新文章的点赞计数。计数表的一致性要求不高，短时间的不一致可以接受。

```sql
-- 计数表：文章统计
CREATE TABLE article_stats (
  article_id INT PRIMARY KEY,
  view_count INT DEFAULT 0,
  like_count INT DEFAULT 0,
  comment_count INT DEFAULT 0
);
```

### 物化视图

物化视图是预计算的查询结果，存储为物理表。物化视图可以包含复杂的 JOIN、聚合、过滤，查询时直接读取物化视图而不需要重新计算。

MySQL 不支持原生的物化视图，可以通过定期刷新的表模拟。PostgreSQL 支持原生的物化视图，可以使用 REFRESH MATERIALIZED VIEW 命令刷新。物化视图适合查询频繁但更新不频繁的场景。

```sql
-- PostgreSQL 物化视图
CREATE MATERIALIZED VIEW user_order_summary AS
SELECT u.id, u.username, COUNT(o.id) AS order_count, SUM(o.amount) AS total_amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username;

-- 刷新物化视图
REFRESH MATERIALIZED VIEW user_order_summary;
```

## 设计原则

### 适用性原则

数据库设计应该满足业务需求，不过度设计也不欠设计。如果业务简单，不需要过度范式化。如果业务复杂，不要欠设计导致无法扩展。适用性原则要求设计者理解业务，抓住核心需求，避免陷入理论完美主义。

### 演进性原则

数据库设计不是一成不变的，随着业务发展需要调整。初期设计可以简单，快速上线。随着业务增长，逐步重构数据库。演进性原则要求设计者预留扩展空间，例如使用 ID 而非业务主键，预留字段用于未来扩展。

### 性能原则

数据库设计需要考虑性能，包括查询性能、写入性能、存储效率。查询性能可以通过索引、反范式化、汇总表优化。写入性能可以通过批量操作、异步处理、分区表优化。存储效率可以通过合适的数据类型、压缩算法优化。

数据库设计是应用架构的基础，需要理论知识和实践经验。范式化提供了消除冗余的指导，ER 建模提供了描述现实的工具，反范式化提供了性能优化的手段。理解这些工具的权衡，才能做出合理的设计决策。
