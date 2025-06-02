# 常见命令行

## 文本处理
- cat/echo/printf
- vi/vim
- tail/head/less/more
- awk

## 文件操作
- tee：同时输出到文件和屏幕
- tar/unzip/cpio/7z：压缩解压
- find：文件查找

## 磁盘管理
- fdisk：磁盘分区
- fio_libaio：异步 I/O 测试
- dd：数据复制
- lsblk/df/du：块设备/磁盘空间管理

## 内存管理
- free：查看系统内存使用概况
- top / htop：查看实时内存使用
- vmstat：查看内存、CPU 和 IO 状态
- smem：查看进程实际使用的内存
- /proc/meminfo：系统内存状态详细信息
- /proc/[pid]/maps、smaps：查看进程内存映射

## 包管理
- apt：Debian/Ubuntu 包管理
- yum/dnf：RHEL/CentOS 包管理
- rpm：RPM 包管理

## 网络工具
- ip：综合性网络管理工具
- iptables/nftables：netfilter 模块管理
- tc：流量控制
- tcpdump：网络抓包
- nc/socat：Socket 客户端/服务端
- iperf3：网络性能测试
- ss/netstat：Socket 统计
- nmap：网络扫描

## 内核工具
- dmesg：查看内核缓冲区消息
- lspci：PCI 设备信息

## 系统服务
- systemctl：服务管理
- journalctl：日志管理
- top：资源监控