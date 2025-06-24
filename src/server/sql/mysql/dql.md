# DCL
基础操作：定义数据库和表，增删改数据库和表，查询，权限控制
主要掌握DML和DQL，如果有DBA的工作，则需要再掌握DCL

函数：
聚合函数：主要作用于select之后
其他函数：字符串函数，数值函数，日期函数，流程控制函数
例如：trim,lpad,  rand,  date_diff,data_add

## DDL
```sql
use database/schema
create database [name]
```

## DML
```sql
insert into [table] ([col],[col]) values([val],[val])
insert into [table] values([val],[val]) -- 默认全部列

delete from [table] where [conditions]

update [table] set [col]=[val] where [conditions]
```

## DQL
```sql
select * / col / func()/[child col]
from [table] / [join table] / [child table]
where [conditions]  -- conditions可以是对col和child table的运算
group by [col] having [conditions]
order by [col],[col]
limit [start index],[per pages]
union [select ...]
-- 执行顺序：from where group select order limit
```


## 多表关系：
一对多：在多的字段往往作为一的一个属性，在多中建立外键关联一的主键，on update 设置为级联，on delete设置为set null
多对多：使用第三张表来维护，字段有id作为主键，两张表的主键，作为该表的外键
一对一：在其中的任何一方，将其主键设置成外键关联至另外一个的主键。

## 连接和子查
允许将查询的结果作为新的变量进行使用

## 事务
控制数据的回滚功能，保护数据的完整性和一致性：原子性，一致性，隔离性，持久性。
```sql
begin -- 开启事务
commit -- 提交事务
rollback -- 回滚事务
```

并发事务问题
+ 脏读
+ 不可重复读
+ 幻读

为了解决事务并发问题，可以为事务添加隔离级别。

进阶
存储引擎（表类型）shows engines

索引：高效管理数据，以提高查询速率，降低IO成本，降低排序成本和CPU消耗，但是会使得增删改的效率降低

索引结构：
B+Tree索引
hash索引（精确查询，快速，不支持范围查找）
R-Tree索引（空间索引，主要用于地理空间的数据结构）
Full-Text索引（全文索引，文档匹配，高效）

## mysql调优基础
查看执行频次 show global status like ‘Com______’
慢查询分析 show global variables like ‘slow_query_log’
性能分析 show profiles for [sqlStr]
查询计划 explain

## 索引
+ 对于联合索引，最左前缀原则，中间不能断，不能使用>或<，应该尽可能使用>=或<=
+ 对于字符串查询，要加单引号、模糊匹配不能让字符串开头为模糊符号
+ 对于全部查询，不使用函数作用于列、使用or连接两个字段必须都有否则均失效、值得一提的是，mysql内部对于索引的选择是有优化的，原因是全表扫描在某些情况下更快，这是mysql内部的数据分布评估SQL提示，在SQLStr中显式指定index。
