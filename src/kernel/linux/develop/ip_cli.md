---
title: ip 命令
order: 30
---

# ip 命令

`ip` 是 Linux 网络管理的核心工具（iproute2 套件），替代了老旧的 `ifconfig`、`route`、`arp` 命令。它通过 Netlink 类型的 socket 与内核直接通信，走 VIP 通信通道，操作链路层、地址、路由、邻居等几乎所有网络对象。基本语法为 `ip <OBJECT> <COMMAND>`——OBJECT 是操作对象（link/addr/route/rule/neigh...），COMMAND 是动作（add/del/show/set...）。

## 链路层：ip link

`ip link` 管理网络接口的物理和链路层属性。每个接口有一个内核索引号（ifindex）和用户可读的名称。

```bash
# 查看所有接口（含状态、MTU、MAC、速率）
ip link show
ip -s link show          # 带收发统计

# 启用/禁用接口
ip link set eth0 up
ip link set eth0 down

# 修改 MTU
ip link set eth0 mtu 9000

# 修改接口名称
ip link set eth0 name wan0

# 创建虚拟接口
ip link add veth0 type veth peer name veth1    # veth pair（常用于容器）
ip link add br0 type bridge                    # 网桥
ip link add bond0 type bond mode 802.3ad       # 链路聚合
ip link add vlan100 link eth0 type vlan id 100 # VLAN 子接口
```

`ip -s link`（带统计）输出每个接口的收发字节数、包数、drop 计数和错误计数。`RX dropped` 持续增长通常是 ring buffer 太小或 CPU 处理不过来——调大 `ethtool -G eth0 rx 4096`。`RX errors` 通常指示物理链路问题（信号质量差、线缆故障）。

## 地址管理：ip addr

`ip addr` 管理接口上的 IP 地址。一个接口可以绑定多个地址（secondary address）。

```bash
# 查看所有地址
ip addr show
ip -4 addr show eth0     # 只看 IPv4
ip -6 addr show eth0     # 只看 IPv6

# 添加/删除地址
ip addr add 192.168.1.10/24 dev eth0
ip addr add 192.168.1.20/24 dev eth0          # 第二个地址（secondary）
ip addr del 192.168.1.10/24 dev eth0

# 临时地址（ip addr add 的 address lifetime）
ip addr add 10.0.0.1/24 dev eth0 valid_lft 300 preferred_lft 240
```

接口的 `state` 字段显示链路层状态——`UP` 表示接口在管理员和链路层都启用（`ip link set up` + 物理线缆已连接），`DOWN` 表示管理员禁用或链路断开，`UNKNOWN` 表示管理员已启用但链路状态无法判断（如 tun/tap 虚拟设备）。

## 路由管理：ip route

`ip route` 管理内核的路由表。

```bash
# 查看路由表
ip route show              # main 表（默认）
ip route show table all    # 所有表
ip route show table local  # local 表

# 添加/删除路由
ip route add 10.0.0.0/8 via 192.168.1.1        # 通过网关
ip route add 10.0.0.0/8 via 192.168.1.1 dev eth0 # 指定出接口
ip route add 10.0.0.0/8 dev eth0                # 直连网络
ip route add default via 192.168.1.1            # 默认路由
ip route add 10.0.0.0/8 via 192.168.1.1 metric 100 # 带度量值
ip route del 10.0.0.0/8

# 多路径（ECMP）
ip route add default nexthop via 192.168.1.1 weight 1 \
                     nexthop via 192.168.2.1 weight 2

# 查特定目的地的路由（策略路由调试核心工具）
ip route get 8.8.8.8
ip route get 8.8.8.8 from 10.0.1.10 iif eth0 mark 10
```

`ip route show` 的输出从左到右：目的前缀 → 下一跳/出接口 → proto（路由协议来源，kernel/static/bgp/dhcp）→ scope（link/host/global）→ metric。`proto kernel` 是内核自动添加的（如接口地址对应的直连路由），手动 `ip route add` 的为 `proto static`。

## 策略路由：ip rule

```bash
# 查看所有规则
ip rule list

# 添加规则
ip rule add from 192.168.1.0/24 table 100           # 按源地址
ip rule add to 10.0.0.0/8 table 100                 # 按目的地址
ip rule add fwmark 10 table 100                     # 按防火墙标记
ip rule add iif eth0 table 100                      # 按入接口
ip rule add from 192.168.1.0/24 to 10.0.0.0/8 table 100  # 组合条件

# 删除规则
ip rule del from 192.168.1.0/24 table 100
ip rule del priority 100    # 按优先级号删除
```

规则按 `priority` 从小到大匹配。`priority 0` 的 `local` 规则不可删除（内核自动添加）。自定义路由表需要在 `/etc/iproute2/rt_tables` 中声明名称和 ID 的映射。

## 邻居管理：ip neigh

`ip neigh` 操作 ARP（IPv4）和 NDP（IPv6）邻居表。

```bash
# 查看邻居表
ip neigh show              # 当前所有条目
ip neigh show nud reachable # 只看已解析的（活跃）
ip neigh show nud stale    # 只看过期的

# 手动操作
ip neigh add 192.168.1.1 lladdr aa:bb:cc:dd:ee:ff dev eth0  # 静态 ARP
ip neigh del 192.168.1.1 dev eth0                             # 删除条目
ip neigh replace 192.168.1.1 lladdr aa:bb:cc:dd:ee:ff dev eth0 nud permanent # 永久静态
```

输出中的 `state` 字段按 NUD（Neighbour Unreachability Detection）状态机变化：`REACHABLE` 表示最近通信过确认可达，`STALE` 表示超过基址可达时间未通信但仍可使用，`DELAY` 表示正在等待确认（有流量触发），`PROBE` 表示主动发送 ARP 请求探测中，`FAILED` 表示多次探测无响应。`STALE` 条目的数量不反映网络问题——它是正常状态，说明之前通信过但近期没有流量。只有出现 `FAILED` 才需要排查。

## 查看统计与 socket 状态：ip -s 与 ss

```bash
# 接口统计（收发字节、包数、drop、错误）
ip -s link show eth0

# 路由缓存统计
nstat -a | grep -i route

# socket 统计（替代 netstat）
ss -tlnp          # TCP 监听端口
ss -t state established '( dport = :443 or sport = :443 )' # 所有 HTTPS 连接
ss -s             # 汇总统计
```

`ss` 是 socket 状态查询的标准工具。`ss -tlnp` 列出所有 TCP 监听 socket 和对应的进程名（需要 root 才能看到进程信息）。`ss -i` 显示每个 socket 的 TCP 详情（窗口大小、RTT、拥塞算法、重传统计），是 TCP 性能问题的第一手排查工具。
