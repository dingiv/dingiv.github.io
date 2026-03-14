---
title: Kubernetes
order: 2
---

# Kubernetes

Kubernetes（K8s）是 Google 开源的容器编排平台，用于自动化部署、扩展和管理容器化应用。Kubernetes 提供服务发现、负载均衡、存储编排、自动扩缩容、自愈等功能，是云原生的事实标准。

## Kubernetes 的架构

### 控制平面

控制平面负责集群的决策和响应，管理集群的状态。

API Server：集群的统一入口，处理 REST 请求，验证和配置数据。

Etcd：高可用的键值存储，存储集群的配置和状态数据。

Scheduler：负责调度 Pod 到合适的节点。

Controller Manager：运行控制器，维护集群的期望状态。

Cloud Controller Manager：与云服务商 API 交互，管理云资源。

### 数据平面

数据平面负责运行容器，由多个节点组成。

Kubelet：节点代理，负责管理 Pod、与 API Server 通信、汇报节点状态。

Kube-proxy：网络代理，负责 Service 的负载均衡和网络代理。

Container Runtime：容器运行时，负责运行容器（containerd、CRI-O）。

## Kubernetes 的核心概念

### Pod

Pod 是 Kubernetes 的最小调度单位，包含一个或多个容器，共享网络和存储。Pod 内的容器可以通过 localhost 通信，共享存储卷。

Pod 的生命周期：Pending（待调度）、Running（运行中）、Succeeded（成功终止）、Failed（失败终止）、Unknown（状态未知）。

Pod 的重启策略：Always（总是重启）、OnFailure（失败时重启）、Never（不重启）。

### Node

Node 是集群的工作节点，运行 Pod。Node 的状态：Ready（就绪）、NotReady（未就绪）、OutOfDisk（磁盘不足）、MemoryPressure（内存压力）、DiskPressure（磁盘压力）、PIDPressure（进程数压力）。

Node 的调度：根据 Pod 的资源请求、节点选择器、亲和性、污点和容忍度，选择合适的节点运行 Pod。

### Namespace

Namespace 是集群的虚拟分区，用于隔离资源。默认命名空间：default（默认）、kube-system（系统组件）、kube-public（公共资源）、kube-node-lease（节点租约）。

### Label 和 Selector

Label 是键值对，用于标识资源。Selector 根据 Label 选择资源，用于 Pod 选择器、Service 选择器。

Label 和 Selector 实现资源间的松耦合，Service 通过 Selector 选择 Pod，Deployment 通过 Selector 管理 Pod。

### Deployment

Deployment 管理 Pod 的副本数和版本，支持滚动更新、回滚。Deployment 创建 ReplicaSet，ReplicaSet 管理 Pod。

Deployment 的更新策略：RollingUpdate（滚动更新）、Recreate（重建）。

滚动更新：逐个替换 Pod，确保服务不中断。可以配置 maxSurge（最大新增 Pod 数）和 maxUnavailable（最大不可用 Pod 数）。

### Service

Service 是 Pod 的稳定网络入口，提供负载均衡和服务发现。Service 通过 Selector 选择 Pod，将请求负载均衡到后端 Pod。

Service 的类型：ClusterIP（集群内部访问）、NodePort（通过节点端口访问）、LoadBalancer（通过负载均衡器访问）、ExternalName（外部名称映射）。

Service 的负载均衡策略：kube-proxy 支持 iptables 和 IPVS 模式，IPVS 性能更好。

### Ingress

Ingress 是 HTTP/HTTPS 路由规则，将外部流量路由到 Service。Ingress 由 Ingress Controller 实现，如 Nginx Ingress Controller、Traefik Ingress Controller。

Ingress 支持基于主机名、路径的路由，支持 TLS 终止、WebSocket、gRPC。

### ConfigMap 和 Secret

ConfigMap 存储配置数据，以环境变量或挂载文件的方式注入 Pod。

Secret 存储敏感数据（如密码、Token），Secret 数据以 Base64 编码存储，可以挂载为文件或环境变量。

### PersistentVolume 和 PersistentVolumeClaim

PersistentVolume（PV）是集群的存储资源，由管理员或 StorageClass 动态创建。

PersistentVolumeClaim（PVC）是对存储的请求，Pod 通过 PVC 挂载存储。

StorageClass：存储类，定义存储的类型和参数，支持动态创建 PV。

PV 的访问模式：ReadWriteOnce（单节点读写）、ReadOnlyMany（多节点只读）、ReadWriteMany（多节点读写）。

## Kubernetes 的调度

### 资源请求和限制

resources.requests：Pod 调度所需的最小资源，用于调度决策。

resources.limits：Pod 可用的最大资源，用于运行时限制。

CPU 资源：单位为 millicore（1 CPU = 1000m），可压缩，超过限制时被限流。

内存资源：单位为字节，不可压缩，超过限制时被 OOM Kill。

### 节点选择器

nodeSelector：通过 Label 选择节点，简单但功能有限。

nodeAffinity：节点亲和性，支持 required（必须满足）和 preferred（优先满足）。

podAffinity 和 podAntiAffinity：Pod 亲和性和反亲和性，控制 Pod 的调度位置。

### 污点和容忍度

Taint（污点）：标记节点，拒绝不匹配的 Pod 调度。

Toleration（容忍度）：Pod 可以容忍污点，即使节点有污点也可以调度。

常见污点：NoSchedule（不调度）、PreferNoSchedule（尽量不调度）、NoExecute（不调度且驱逐现有 Pod）。

## Kubernetes 的自愈

### 健康检查

livenessProbe：存活探针，探测容器是否存活，失败时重启容器。

readinessProbe：就绪探针，探测容器是否就绪，失败时从 Service 移除。

startupProbe：启动探针，探测容器是否启动，用于慢启动容器。

探针类型：HTTP GET、TCP Socket、Exec（执行命令）。

### 自动重启

Kubernetes 通过 ReplicaSet 确保 Pod 的副本数，Pod 故障时自动重启。

### 自动调度

节点故障时，Kubernetes 自动将 Pod 调度到其他节点。

### 自动扩缩容

Horizontal Pod Autoscaler（HPA）：根据 CPU、内存或自定义指标自动调整 Pod 副本数。

Vertical Pod Autoscaler（VPA）：根据资源使用情况自动调整 Pod 的资源请求和限制。

Cluster Autoscaler：根据 Pod 调度失败情况自动调整节点数量。

## Kubernetes 的网络

### Pod 网络

Pod 有独立的 IP 地址，Pod 间可以直接通信，跨节点 Pod 通信通过 Overlay 网络（如 VXLAN）。

### Service 网络

Service 有虚拟 IP（ClusterIP），kube-proxy 通过 iptables 或 IPVS 实现 ClusterIP 的负载均衡。

### Ingress 网络

Ingress 提供 HTTP/HTTPS 路由，Ingress Controller 实现 Ingress 规则。

## Kubernetes 的最佳实践

使用标签：为资源添加 Label，便于选择和管理。

使用命名空间：使用 Namespace 隔离不同环境、不同团队的资源。

设置资源限制：为 Pod 设置资源请求和限制，避免资源争抢。

使用健康检查：配置 livenessProbe 和 readinessProbe，确保 Pod 健康。

使用 ConfigMap 和 Secret：将配置和敏感信息存储在 ConfigMap 和 Secret，不要打包在镜像中。

使用 HPA：配置 HPA 自动扩缩容，提高资源利用率。

使用 RBAC：配置 Role-Based Access Control，限制访问权限。

定期备份 etcd：etcd 存储集群状态，定期备份 etcd 数据。

Kubernetes 是云原生的事实标准，理解 Kubernetes 的核心概念和最佳实践，有助于构建可扩展、高可用的云原生应用。
