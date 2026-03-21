---
title: 备份与恢复
order: 2
---

# 备份与恢复

数据备份是数据库安全的最重要防线。没有备份的数据一旦丢失就是永久丢失，任何高可用方案都无法替代备份的作用。

## 备份类型

### 逻辑备份与物理备份

逻辑备份导出 SQL 语句或特定格式的数据文件，工具如 mysqldump、pg_dump。优势是跨平台、可读性强、可以选择性恢复部分数据。劣势是恢复速度慢，需要重新执行 SQL 语句，大表恢复耗时可观。

物理备份直接复制数据文件，工具如 Percona XtraBackup、pg_basebackup。优势是恢复速度快，直接复制文件即可。劣势是跨平台性差，不同操作系统、不同 MySQL 版本之间可能不兼容。生产环境推荐物理备份，恢复时间窗口是关键考量。

### 全量备份与增量备份

全量备份包含所有数据，恢复时只需要一个全量备份文件。增量备份只包含自上次备份以来的变更，恢复时需要先恢复全量备份，再依次应用增量备份。WAL 日志可以作为细粒度的增量备份，MySQL 的 binlog、PostgreSQL 的 WAL 都属于这类。

全量备份简单但占用空间大，增量备份节省空间但恢复复杂。常见的策略是每周一次全量备份，每天一次增量备份，保留 4 周数据。根据业务需求调整保留周期，合规要求高的行业需要保留更长时间。

### 冷备份与热备份

冷备份需要停机，保证数据一致性。热备份在运行中备份，需要处理并发写入和一致性问题。现代数据库都支持热备份，这是生产环境的标准要求。

MySQL 的热备份通过 FLUSH TABLES WITH READ LOCK 获取全局读锁，复制非事务表，对于 InnoDB 表通过 XtraBackup 在不锁表的情况下复制。XtraBackup 的原理是监控 redo log，复制过程中记录修改，恢复时重放这些修改使数据文件达到一致状态。

## MySQL 备份实践

### mysqldump 逻辑备份

mysqldump 是 MySQL 官方提供的逻辑备份工具，支持单表、单库、全库备份。基本用法是 `mysqldump -u root -p database_name > backup.sql`。

生产环境推荐使用 `--single-transaction` 参数，该参数在备份开始时启动一个一致性事务，确保备份期间的数据一致性，且不阻塞其他事务。对于大表备份，`--quick` 参数禁止缓存整个查询结果，避免内存溢出。`--routines` 和 `--triggers` 导出存储过程和触发器，`--events` 导出事件调度器。

```bash
# 全库备份（推荐）
mysqldump -u root -p \
  --single-transaction \
  --quick \
  --routines \
  --triggers \
  --events \
  --all-databases > full_backup_$(date +%Y%m%d).sql
```

mysqldump 的输出是 SQL 文本，可以通过 grep 查看特定表，也可以恢复单个表。恢复时使用 `mysql -u root -p < backup.sql`。恢复速度取决于硬件和 SQL 复杂度，100GB 的数据库恢复可能需要数小时。

### XtraBackup 物理备份

Percona XtraBackup 是开源的物理备份工具，支持 InnoDB 的热备份。它无需停机、不锁表、备份速度快，是生产环境的首选工具。

XtraBackup 的备份流程：首先备份 InnoDB 表，通过监控 redo log 记录备份过程中的修改，然后备份非 InnoDB 表时获取读锁，最后释放锁并备份 binlog。恢复流程是将备份文件应用到 redo log，使数据文件达到一致状态，然后启动 MySQL。

```bash
# 全量备份
xtrabackup --backup --target-dir=/data/backup/full

# 增量备份（基于上一次备份）
xtrabackup --backup --target-dir=/data/backup/inc1 \
  --incremental-basedir=/data/backup/full

# 准备恢复（应用 redo log）
xtrabackup --prepare --target-dir=/data/backup/full
xtrabackup --prepare --target-dir=/data/backup/full \
  --incremental-dir=/data/backup/inc1

# 恢复数据
xtrabackup --copy-back --target-dir=/data/backup/full
```

XtraBackup 支持"压缩备份"和"流式备份"，压缩备份节省传输带宽和存储空间，流式备份直接通过网络传输到远程服务器，避免本地存储不足。

### binlog 时间点恢复

binlog 记录了所有数据修改操作，可以实现时间点恢复。定期备份配合 binlog，可以恢复到任意时间点。首先恢复最近的备份，然后使用 mysqlbinlog 工具应用 binlog 中指定时间之后的操作。

```bash
# 查看 binlog 内容
mysqlbinlog --start-datetime="2023-01-01 00:00:00" \
  --stop-datetime="2023-01-01 23:59:59" \
  mysql-bin.000001

# 应用 binlog
mysqlbinlog --start-datetime="2023-01-01 12:00:00" \
  --stop-datetime="2023-01-01 13:00:00" \
  mysql-bin.000001 | mysql -u root -p
```

binlog 有三种格式：statement 记录 SQL 语句，row 记录行变化，mixed 混合模式。推荐使用 row 格式，它最安全且可以准确重现操作，但占用空间较大。statement 格式空间占用小，但可能因上下文差异导致执行结果不一致。

## PostgreSQL 备份实践

### pg_dump 逻辑备份

pg_dump 是 PostgreSQL 官方备份工具，支持多种输出格式。plain 格式输出 SQL 文本，custom 格式输出自定义二进制格式，directory 格式输出目录格式，tar 格式输出 tar 归档。custom 和 directory 格式支持并行恢复和选择性恢复，是推荐的格式。

```bash
# 自定义格式备份（推荐）
pg_dump -Fc -f backup.dump database_name

# 目录格式备份（并行）
pg_dump -j 4 -Fd -f backup_dir database_name

# 恢复
pg_restore -d database_name backup.dump
```

pg_dumpall 备份整个集群，包括全局对象（用户、权限、表空间）。建议定期备份全局对象，日常备份使用 pg_dump 针对单个数据库。

### pg_basebackup 物理备份

pg_basebackup 是 PostgreSQL 的物理备份工具，支持热备份和流式备份。它通过复制数据文件和 WAL 日志实现备份，可以用于搭建备库或时间点恢复。

```bash
# 基础备份
pg_basebackup -D /data/backup -Ft -z -P

# 流式备份到远程
pg_basebackup -D - -Ft -z | ssh user@remote "cat > /data/backup/base.tar"
```

PostgreSQL 的 WAL 归档是实现时间点恢复的关键。配置 `archive_mode = on` 和 `archive_command`，WAL 日志填满后会被归档到指定位置。恢复时先恢复基础备份，然后按顺序应用归档的 WAL 日志。

## 备份策略

### 3-2-1 原则

备份的黄金原则是 3-2-1：保留至少 3 份备份，存储在至少 2 种不同的介质上，至少 1 份异地备份。这个原则确保了单一故障点不会导致数据丢失，例如火灾、地震等灾难不会同时摧毁所有备份。

异地备份可以通过 rsync、scp、云存储（AWS S3、阿里云 OSS）实现。考虑到带宽成本，可以每天只上传增量备份，每周上传全量备份。

### 备份验证

备份的价值在于能够恢复，没有验证的备份是虚假的安全感。定期验证备份的有效性，包括检查备份文件完整性、测试恢复流程、验证恢复后数据的正确性。

生产环境可以在备库上进行恢复测试，避免影响主库。测试频率建议每月一次，自动化测试脚本可以减少人工操作。

### 自动化备份

编写备份脚本并通过 cron 定时执行，是确保备份不遗漏的有效手段。脚本应该包含备份、压缩、传输、清理旧备份、发送通知等功能。重要的一点是指定备份失败时的告警机制，例如邮件、短信、钉钉通知。

```bash
#!/bin/bash
# MySQL 备份脚本示例
BACKUP_DIR="/data/backup"
RETENTION_DAYS=30
DATE=$(date +%Y%m%d)

# 执行备份
mysqldump -u root -p$MYSQL_PASSWORD \
  --single-transaction \
  --quick \
  --all-databases | gzip > $BACKUP_DIR/full_$DATE.sql.gz

# 上传到远程
scp $BACKUP_DIR/full_$DATE.sql.gz user@remote:/backup/

# 清理旧备份
find $BACKUP_DIR -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete

# 检查备份文件
if [ ! -f "$BACKUP_DIR/full_$DATE.sql.gz" ]; then
  echo "备份失败" | mail -s "数据库备份告警" admin@example.com
fi
```

## 灾难恢复

### 主从切换

主库故障时，需要将备库提升为主库。首先确保备库的数据是最新的，然后停止备库的复制进程，将其提升为主库，最后将应用连接切换到新主库。

MySQL 的主从切换步骤：在备库上执行 `STOP SLAVE; RESET SLAVE ALL;`，然后 `SET GLOBAL read_only = OFF;` 解除只读模式。PostgreSQL 的主从切换使用 `pg_ctl promote` 命令或 `SELECT pg_promote()` 函数。

### 数据恢复案例

误删表的恢复是最常见的场景。首先找到误删操作所在的 binlog 位置，使用 mysqlbinlog 工具提取该位置之前的所有操作，恢复到临时库，然后将误删表导出并导入主库。

```bash
# 查找误删操作的 binlog 位置
mysqlbinlog mysql-bin.000001 | grep -i "DROP TABLE"

# 提取误删前的操作
mysqlbinlog --start-position=4 --stop-position=123456 \
  mysql-bin.000001 | mysql -u root -p
```

数据损坏的恢复需要更复杂的过程。如果数据文件损坏，首先要判断是硬件问题还是软件问题。硬件问题需要更换磁盘，软件问题可以通过备份恢复。恢复过程中保持冷静，按照步骤操作，避免二次破坏。

备份是数据安全的最后一道防线，但永远不要等到需要恢复时才意识到备份的重要性。建立完善的备份策略、定期验证备份有效性、制定详细的恢复预案，才能在灾难来临时从容应对。
