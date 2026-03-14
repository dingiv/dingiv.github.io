---
title: 云数据库服务
order: 3
---

# 云数据库服务

云数据库是云计算时代的首选方案，提供弹性、高可用、自动运维等优势。主流云厂商都提供托管数据库服务，包括 AWS、Azure、Google Cloud、阿里云、腾讯云。本节对比主流云数据库服务的特点和选型建议。

## AWS 数据库服务

### Amazon RDS

Amazon RDS 是 AWS 的托管关系数据库服务，支持 MySQL、PostgreSQL、MariaDB、Oracle、SQL Server。RDS 提供自动备份、时间点恢复、只读副本、多可用区部署。RDS 的优势是简单易用、自动运维、按需计费。

RDS 的存储自动扩展可以配置存储阈值，当存储空间不足时自动扩容，避免磁盘满导致的服务不可用。RDS 的只读副本最多支持 15 个，可以分散读压力，实现读写分离。RDS 的多可用区部署在不同可用区创建备库，主库故障时自动切换，实现高可用。

RDS 的实例类型包括通用型、内存优化型、IO 优化型。通用型适合大多数应用，内存优化型适合缓存、内存数据库，IO 优化型适合高 IO 负载。实例选择需要考虑 CPU、内存、存储、网络，建议从小的实例开始，根据实际负载升级。

### Amazon Aurora

Amazon Aurora 是 AWS 的云原生数据库，兼容 MySQL 和 PostgreSQL 协议。Aurora 的存储是分布式和自愈的，数据自动复制到 6 个副本，跨 3 个可用区。Aurora 的计算和存储分离，存储自动扩展，最高支持 128TB。

Aurora 的性能是标准 MySQL 的 5 倍、标准 PostgreSQL 的 3 倍。Aurora 的读写延迟低于 10ms，跨区域复制的延迟低于 1 秒。Aurora Serverless 是无服务器版本，自动扩展计算资源，按使用量计费，适合不稳定负载。

Aurora 的优势是性能高、可用性强、自动运维。劣势是成本较高，比 RDS 贵 50%-100%。Aurora 适合对性能和可用性要求高的业务，例如金融、电商、SaaS。

## Azure 数据库服务

### Azure SQL Database

Azure SQL Database 是微软的托管 SQL Server 服务，提供完全托管的 SQL Server 数据库。Azure SQL Database 支持单库和弹性池，单库是独立的数据库，弹性池是多个数据库共享资源池。

Azure SQL Database 的服务层级包括基本、标准、高级。基本层适合小负载，标准层适合中等负载，高级层适合高负载。每个层级包含多个性能级别，可以根据负载弹性调整。

Azure SQL Database 的优势是与 SQL Server 兼容、与 Azure 生态集成、自动备份和时间点恢复。劣势是功能有限，不支持某些 SQL Server 特性如 CLR 集成、Agent 代理。

### Azure Database for MySQL/PostgreSQL

Azure Database for MySQL/PostgreSQL 是 Azure 的托管开源数据库服务，完全托管 MySQL 和 PostgreSQL。服务类似 AWS RDS，提供自动备份、只读副本、监控告警。

Azure Database 的优势是与 Azure 生态集成、按使用量计费、支持 PITR。劣势是全球可用区较少，某些区域可能不支持。Azure Database 适合已经在使用 Azure 的团队，减少跨云的复杂性。

## Google Cloud 数据库服务

### Cloud SQL

Cloud SQL 是 Google Cloud 的托管关系数据库服务，支持 MySQL、PostgreSQL、SQL Server。Cloud SQL 提供自动备份、只读副本、高可用性配置。Cloud SQL 与 Google Cloud 的其他服务集成良好，例如 Compute Engine、Kubernetes Engine。

Cloud SQL 的优势是与 GCP 生态集成、自动扩展存储、全球部署。劣势是性能不如 AWS Aurora，可用区较少。Cloud SQL 适合已经在使用 GCP 的团队，或者需要全球部署的业务。

### Cloud Spanner

Cloud Spanner 是 Google Cloud 的分布式关系数据库，提供外部一致性和全球分布。Spanner 的架构在"分布式数据库"章节详细讨论，这里不再重复。Spanner 的优势是全球一致、自动扩展、无单点故障。劣势是成本高、学习曲线陡峭。

Spanner 适合全球业务、金融级应用、需要强一致性的场景。对于大多数业务，Cloud SQL 或 Aurora 已经足够，Spanner 是更极端的选择。

## 阿里云数据库服务

### PolarDB

PolarDB 是阿里云的云原生数据库，兼容 MySQL、PostgreSQL、Oracle 协议。PolarDB 采用存储计算分离架构，多个计算节点共享同一存储，可以实现秒级水平扩展。PolarDB 的性能是开源 MySQL 的 6 倍，成本仅为商业数据库的 1/10。

PolarDB 的存储自动扩展，最高支持 100TB。PolarDB 支持一写多读，最多支持 16 个只读节点。PolarDB 的全球数据库功能支持跨地域部署，地域间延迟低于 2 秒。

PolarDB 的优势是性能高、成本低、国内可用区多。劣势是生态不如 AWS Aurora，某些高级功能如 Serverless 还在开发中。PolarDB 适合国内业务，尤其是对成本敏感的中小公司。

### ApsaraDB

ApsaraDB 是阿里云的托管数据库服务，包括 MySQL、PostgreSQL、SQL Server、MongoDB、Redis。ApsaraDB 类似 AWS RDS，提供自动备份、只读副本、监控告警。

ApsaraDB 的优势是国内可用区多、与阿里云生态集成、按需付费。劣势是性能不如 PolarDB，功能不如 AWS Aurora。ApsaraDB 适合中小规模业务，或者作为迁移到云的第一步。

## 腾讯云数据库服务

### TencentDB for MySQL

TencentDB for MySQL 是腾讯云的托管 MySQL 服务，提供高可用、高安全的 MySQL 数据库。TencentDB 支持一主两备、只读副本、秒级备份恢复。TencentDB 的性能优化包括线程池、并行查询、热点更新。

TencentDB 的优势是与腾讯云生态集成、国内可用区多、价格有竞争力。劣势是功能不如 PolarDB 和 Aurora，文档和社区支持不如 AWS。TencentDB 适合已经在使用腾讯云的团队。

### TDSQL

TDSQL 是腾讯云的分布式数据库，支持 MySQL 和 PostgreSQL 协议。TDSQL 采用分布式架构，数据自动分片，支持分布式事务。TDSQL 的金融级一致性、高可用性，适合金融、政务等场景。

TDSQL 的优势是分布式架构、金融级可靠性、国内合规。劣势是成本高、学习曲线陡峭、与开源协议不完全兼容。TDSQL 适合金融级应用、大规模业务、需要分布式事务的场景。

## 选型建议

### 按云厂商选择

云数据库的选择首先取决于云厂商。如果业务已经部署在 AWS，首选 RDS 或 Aurora。如果业务在 Azure，首选 Azure SQL Database。如果业务在阿里云，首选 PolarDB。跨云使用数据库服务会增加网络延迟和复杂性。

跨云部署的场景下，可以考虑多云数据库，例如在 AWS 和阿里云都部署 MySQL，通过数据同步实现容灾。但这种方案复杂度高，需要数据一致性和故障切换的考虑。

### 按业务规模选择

小规模业务（QPS < 1000）可以选择基础版的云数据库，成本低、功能够用。中等规模业务（1000 < QPS < 10000）可以选择标准版，根据负载选择合适的实例规格。大规模业务（QPS > 10000）可以考虑云原生数据库如 Aurora、PolarDB，或者分布式数据库如 TDSQL、Spanner。

业务增长预期也是考虑因素。如果预期业务快速增长，建议选择弹性更好的云原生数据库。如果业务稳定，可以选择成本更低的托管数据库。

### 按功能需求选择

如果需要高可用性，选择支持多可用区部署的服务，例如 Aurora 多主、PolarDB 集群版。如果需要读写分离，选择支持只读副本的服务，所有云数据库都支持。如果需要分布式事务，选择分布式数据库如 TDSQL、Spanner。

如果需要特殊功能，例如时空数据、全文检索、时序数据，选择支持这些功能的云数据库或专用数据库。AWS、Azure、阿里云都提供专用的时序数据库、图数据库、搜索引擎。

### 成本优化

云数据库的成本包括实例费用、存储费用、流量费用、备份费用。成本优化包括选择合适的实例规格、使用预留实例、自动开关实例、清理备份数据。

预留实例相比按需实例可以节省 30%-60% 的成本，但需要承诺使用时长（1 年或 3 年）。对于稳定负载的业务，预留实例是成本优化的有效手段。对于不稳定负载，可以使用 Serverless 版本，按使用量计费。

云数据库是云计算时代的首选方案，提供了自建数据库无法比拟的便利性和可靠性。选择云数据库需要考虑云厂商、业务规模、功能需求、成本预算，综合决策。
