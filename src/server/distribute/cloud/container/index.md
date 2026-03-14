---
title: 容器技术
order: 1
---

# 容器技术

容器是一种轻量级的虚拟化技术，将应用和依赖打包在一起，保证应用在任何环境中一致运行。容器相比虚拟机启动快、资源占用少、部署灵活，是云原生的基础。

## 容器的原理

### Linux Namespace

Namespace 实现资源隔离，不同 Namespace 的进程看到不同的资源视图。Linux 提供多种 Namespace：PID Namespace（进程隔离）、Network Namespace（网络隔离）、Mount Namespace（文件系统隔离）、UTS Namespace（主机名隔离）、IPC Namespace（进程间通信隔离）、User Namespace（用户隔离）。

### Linux Cgroup

Cgroup（Control Group）实现资源限制和监控，可以限制进程的 CPU、内存、磁盘 I/O、网络带宽。Cgroup 还可以监控进程的资源使用情况。

### Union File System

UnionFS（联合文件系统）实现分层存储，将多个目录联合挂载到同一个挂载点。Docker 镜像使用 UnionFS，镜像由多个只读层组成，容器启动时添加可写层。

OverlayFS 是 Docker 使用的 UnionFS 实现，支持多层联合挂载。

### 容器 vs 虚拟机

虚拟机：完整的操作系统、Hypervisor 层、资源占用大、启动慢。

容器：共享操作系统内核、资源占用小、启动快。

## Docker

### Docker 的架构

Docker Daemon（dockerd）：Docker 守护进程，管理镜像、容器、网络、存储。

Docker CLI：Docker 命令行工具，与 Docker Daemon 通信。

Docker Registry：Docker 镜像仓库，存储和分发镜像。

Docker 容器：Docker 运行的容器实例。

### Docker 镜像

Docker 镜像是一个只读的模板，包含应用和依赖。镜像由多个层组成，每层代表 Dockerfile 中的一条指令。镜像分层存储，可以复用层，减少存储和传输开销。

Dockerfile：定义镜像构建过程的文本文件。常用指令：FROM（基础镜像）、RUN（执行命令）、COPY（复制文件）、ADD（复制文件并解压）、CMD（容器启动命令）、ENTRYPOINT（容器启动入口）、ENV（环境变量）、EXPOSE（暴露端口）、VOLUME（挂载卷）。

多阶段构建：使用多个 FROM 指令，每个 FROM 开始一个新的构建阶段。多阶段构建可以减少最终镜像的大小，只保留运行时需要的文件。

### Docker 容器

Docker 容器是镜像的运行实例，启动时在镜像最上层添加可写层。容器的修改不会影响镜像，删除容器后可写层也被删除。

容器生命周期：Created（已创建）、Running（运行中）、Paused（暂停）、Stopped（已停止）、Removing（删除中）、Dead（已死亡）。

容器资源限制：通过 Cgroup 限制容器的 CPU、内存、磁盘 I/O。--cpus 限制 CPU 核心数，--memory 限制内存，--pids-limit 限制进程数。

### Docker 网络

Bridge 网络：默认网络模式，容器通过 docker0 网桥通信，容器间可以通信，容器可以访问外部。

Host 网络：容器使用宿主机网络栈，无网络隔离，性能好但安全性差。

None 网络：容器无网络，适用于不需要网络的容器。

Overlay 网络：跨主机容器通信，适用于 Swarm 和 Kubernetes。

Macvlan 网络：容器直接连接物理网络，拥有独立 IP。

### Docker 存储

Volume：Docker 管理的存储，独立于容器生命周期，删除容器后数据不会丢失。

Bind Mount：宿主机目录挂载到容器，直接映射宿主机路径。

Tmpfs：内存文件系统，容器停止后数据丢失。

## 容器安全

### 镜像安全

使用可信镜像：从官方或可信来源拉取镜像，避免使用来历不明的镜像。

扫描镜像漏洞：使用 docker scan 或 Clair 扫描镜像漏洞，及时修复。

最小化镜像：使用精简基础镜像（如 alpine），减少攻击面。

非 root 用户：容器以非 root 用户运行，减少容器逃逸风险。

### 容器隔离

Namespace 隔离：确保容器的 Namespace 正确配置，避免容器逃逸。

Cgroup 限制：限制容器的资源，防止容器占用过多资源影响宿主机。

Seccomp：限制容器可以执行的系统调用，减少攻击面。

AppArmor/SELinux：强制访问控制，限制容器的文件访问。

### 运行时安全

 readOnlyRootFilesystem：只读根文件系统，防止容器被篡改。

Drop capabilities：删除不必要的 Linux Capabilities，减少权限。

No-new-privileges：禁止获取新的权限，防止权限提升。

## 容器的最佳实践

使用多阶段构建：减少最终镜像大小，只保留运行时需要的文件。

使用精简基础镜像：使用 alpine、scratch 等精简基础镜像，减少镜像大小和攻击面。

镜像标签管理：使用语义化版本（如 v1.0.0），避免使用 latest 标签。

资源限制：为容器设置资源限制，避免资源争抢。

健康检查：配置健康检查，自动重启不健康的容器。

日志管理：应用日志输出到标准输出，便于日志收集。

不要在容器中存储重要数据：容器是临时的，数据应该存储在 Volume 或外部存储。

容器是云原生的基础，理解容器的原理和最佳实践，有助于构建可靠的云原生应用。
