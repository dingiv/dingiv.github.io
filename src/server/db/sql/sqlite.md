---
title: SQLite
order: 30
---

# SQLite 原理与实现

SQLite 是嵌入式关系数据库，服务端less、零配置，是移动端、桌面端、IoT 设备的首选数据库。理解 SQLite 的实现，有助于在嵌入式场景下做出合适的选择。

## 嵌入式架构

SQLite 不是客户端-服务器架构，而是一个库，直接嵌入应用程序中。应用程序调用 SQLite 的 API，库直接读写数据库文件。这种设计的优势是零配置、无网络开销、部署简单；劣势是并发写入受限、单点故障。

SQLite 的架构包括：核心（SQL 解析器、字节码引擎）、后端（B-tree、pager、操作系统接口）。SQL 语句解析成字节码，由虚拟机执行。后端负责存储、索引、事务管理。

## 数据类型与存储

### 动态类型系统

SQLite 使用动态类型系统，这与大多数数据库不同。SQLite 不强制列的数据类型，任何列可以存储任何类型的数据（除了 INTEGER PRIMARY KEY）。数据类型分为存储类（Storage Class）：NULL、INTEGER、REAL、TEXT、BLOB。类型亲和性（Type Affinity）是列的推荐类型，SQLite 会尝试转换数据到推荐类型，但不保证。

### B-tree 存储

SQLite 使用 B-tree 组织数据，每个 B-tree 节点是一个页面（page），默认 4KB。表数据和索引都存储在 B-tree 中。表 B-tree 的键是 rowid（INTEGER PRIMARY KEY）或主键，值是完整记录。索引 B-tree 的键是索引列，值是 rowid。

rowid 是 64 位整数，每行自动分配。如果表定义了 INTEGER PRIMARY KEY，则 rowid 就是该列的值。没有主键的表，rowid 隐藏且自动递增。

## 事务与锁

### 事务级别

SQLite 支持三种事务模式：DEFERRED（延迟，第一次读时加读锁，第一次写时升级写锁）、IMMEDIATE（立即，立即加保留锁，防止其他事务写）、EXCLUSIVE（排他，立即加写锁，防止其他事务读写）。默认是 DEFERRED。

### 锁机制

SQLite 的锁粒度是数据库级（整个文件），这是它的最大限制。锁状态包括：UNLOCKED（无锁）、SHARED（共享锁，读锁）、RESERVED（保留锁，准备写，允许其他读）、PENDING（待定锁，等待所有读锁释放，准备升级写锁）、EXCLUSIVE（排他锁，写锁）。

锁的升级：UNLOCKED → SHARED（第一次读） → RESERVED（BEGIN IMMEDIATE 或第一次写） → PENDING（等待读锁释放） → EXCLUSIVE（准备写入）。

这种设计允许多个读事务并发，但写事务互斥。多个写事务会串行化，导致写并发受限。SQLite 不适合高并发写入场景。

### 原子提交

SQLite 的原子提交通过回滚日志（Rollback Journal）或预写日志（WAL）实现。回滚日志模式：修改数据前，将原始页写入 journal 文件；修改数据页；提交时删除 journal 文件；崩溃恢复时，如果 journal 存在，回滚修改。WAL 模式：修改写入 WAL 文件；定期 checkpoint 将 WAL 合并到数据库；崩溃恢复时，重放 WAL。WAL 模式支持读写并发，性能更好。

### WAL 模式

WAL 模式是 SQLite 3.7+ 引入的，解决了回滚日志模式的限制：读写并发、更快的提交、更少的磁盘同步。WAL 文件是追加写，多个事务可以并发提交。checkpoint 后台进程定期将 WAL 合并到数据库，并截断 WAL。

WAL 模式的限制：WAL 文件不能在 NFS 等网络文件系统上使用；多个进程写入时，checkpoint 会频繁触发；WAL 文件大小有限制，超过会触发 checkpoint。

## 虚拟机与字节码

SQLite 将 SQL 编译成字节码，由虚拟机执行。字节码是类似汇编的指令集：OpenRead（打开表）、Rewind（定位到第一条记录）、Next（下一条记录）、Column（读取列）、MakeRecord（构造记录）、Insert（插入记录）、Halt（停止）。

虚拟机执行字节码，调用后端 API 操作 B-tree 和 pager。这种设计简化了执行引擎，也易于调试和优化。

## 内存数据库

SQLite 支持内存数据库 `:memory:`，数据存储在内存中，不持久化。内存数据库适合临时计算、缓存、测试场景。内存数据库的 B-tree 和 pager 都在内存中，没有磁盘 I/O，速度极快。

## 性能优化

### PRAGMA 配置

PRAGMA 是 SQLite 的配置指令：synchronous（控制 fsync 频率，FULL 安全但慢，OFF 快但风险）、journal_mode（WAL 或 DELETE）、cache_size（页面缓存大小，越大越好）、temp_store（临时表存储位置，MEMORY 或 FILE）、mmap_size（内存映射 I/O 大小）。

### 批量操作

SQLite 的批量操作可以通过事务包装，批量提交比单条提交快得多。例如插入 1000 行，用事务包装只需一次 fsync，不用事务需要 1000 次 fsync。

### 索引优化

索引选择原则与其他数据库类似：高选择性列、频繁查询的列、范围查询的列。SQLite 的索引是 B-tree，支持范围查询和排序。CREATE INDEX 语句创建索引，DROP INDEX 删除索引。

## 限制与考量

SQLite 的限制主要是并发写入：写事务互斥，不适合高并发写入。单文件存储，单个文件不宜过大（推荐不超过几十 GB）。无用户权限管理，所有用户权限相同。无网络访问，只支持本地访问。

尽管有这些限制，SQLite 的零配置、嵌入式、高性能特点，使其成为移动端、桌面端、IoT 设备的首选数据库。理解它的实现和限制，有助于在合适的场景选择合适的数据库。
