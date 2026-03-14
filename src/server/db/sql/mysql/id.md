# 索引
索引是提高数据库查询性能的重要工具。它们通过创建额外的数据结构来加速数据检索，但同时也需要权衡存储空间和更新性能。

## 索引类型

### 1. 主键索引（Primary Key Index）
- 唯一标识表中的每一行
- 自动创建
- 不允许NULL值

```sql
CREATE TABLE users (
    user_id INT PRIMARY KEY,  -- 自动创建主键索引
    username VARCHAR(50),
    email VARCHAR(100)
);
```

### 2. 唯一索引（Unique Index）
- 确保列值的唯一性
- 允许NULL值
- 可以包含多个列

```sql
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_code VARCHAR(20),
    UNIQUE INDEX idx_product_code (product_code)
);
```

### 3. 普通索引（Regular Index）
- 最基本的索引类型
- 不保证唯一性
- 用于加速查询

```sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    INDEX idx_customer (customer_id),
    INDEX idx_date (order_date)
);
```

### 4. 复合索引（Composite Index）
- 包含多个列的索引
- 列的顺序很重要
- 支持前缀匹配

```sql
CREATE TABLE sales (
    sale_id INT PRIMARY KEY,
    product_id INT,
    sale_date DATE,
    amount DECIMAL(10,2),
    INDEX idx_product_date (product_id, sale_date)
);
```

### 5. 全文索引（Full-Text Index）
- 用于文本搜索
- 支持模糊匹配
- 适用于大文本字段

```sql
CREATE TABLE articles (
    article_id INT PRIMARY KEY,
    title VARCHAR(200),
    content TEXT,
    FULLTEXT INDEX idx_content (content)
);
```

## 索引实现方式

### 1. B-Tree索引
- 最常用的索引类型
- 支持等值查询和范围查询
- 适用于大多数场景

### 2. Hash索引
- 只支持等值查询
- 不支持范围查询
- 查询性能极快

### 3. R-Tree索引
- 用于空间数据
- 支持地理信息查询
- 适用于GIS应用

## 索引优化策略

### 1. 选择合适的列
- 高选择性的列
- 频繁用于查询的列
- 用于排序和分组的列

### 2. 避免过度索引
- 每个索引都需要维护成本
- 过多的索引会降低写入性能
- 定期评估索引使用情况

### 3. 复合索引设计
- 最左前缀原则
- 考虑查询模式
- 避免冗余索引

```sql
-- 好的复合索引设计
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    status VARCHAR(20),
    order_date DATE,
    INDEX idx_customer_status_date (customer_id, status, order_date)
);

-- 可以支持以下查询
SELECT * FROM orders WHERE customer_id = 1;
SELECT * FROM orders WHERE customer_id = 1 AND status = 'completed';
SELECT * FROM orders WHERE customer_id = 1 AND status = 'completed' AND order_date > '2023-01-01';
```

## 索引维护

### 1. 定期重建索引
```sql
-- MySQL
ALTER TABLE table_name ENGINE=InnoDB;

-- PostgreSQL
REINDEX TABLE table_name;
```

### 2. 监控索引使用
```sql
-- MySQL
EXPLAIN SELECT * FROM table_name WHERE column = value;

-- PostgreSQL
EXPLAIN ANALYZE SELECT * FROM table_name WHERE column = value;
```

### 3. 删除无用索引
```sql
DROP INDEX index_name ON table_name;
```

## 常见问题与解决方案

### 1. 索引失效
- 使用函数或运算符
- 类型转换
- 使用OR条件

### 2. 索引选择
- 考虑查询频率
- 考虑数据分布
- 考虑更新频率

### 3. 性能优化
- 使用覆盖索引
- 避免索引列上的计算
- 合理使用索引提示

## 最佳实践

1. **索引设计原则**
   - 为常用查询创建索引
   - 保持索引简洁
   - 定期维护索引

2. **监控与调优**
   - 监控索引使用情况
   - 分析慢查询
   - 定期优化索引

3. **文档维护**
   - 记录索引设计决策
   - 说明索引用途
   - 更新索引变更记录 