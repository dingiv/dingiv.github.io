# 文件系统

## IO管理

IO设备拥有独立的控制处理器，现代IO设备通过MMIO方式将自身寄存器地址空间映射到物理内存空间中，让CPU通过直接读写物理地址空间来控制IO设备。

### 设备挂载

1. 设备识别和驱动加载
   - 识别设备：操作系统检测设备并分配设备文件（Linux中通常位于`/dev/`目录下）
   - 加载驱动：操作系统加载适当的驱动程序支持设备操作

2. 设备格式化
   - 存储设备需要经过格式化才能使用
   - 格式化将物理存储空间划分为存储区域
   - 为这些区域建立文件系统
   - 未格式化的存储设备不能直接存储文件和数据

   文件系统格式化：
   - 文件系统是操作系统管理磁盘上文件的方式
   - 不同操作系统使用不同的文件系统格式（ext4、NTFS、FAT32、exFAT等）
   - 分区表（MBR或GPT）定义设备上不同部分的布局和大小

   例如，在Linux中格式化磁盘分区：
   ```bash
   sudo mkfs.ext4 /dev/sda1
   ```

3. 挂载存储设备
   - 格式化后，存储设备的文件系统才可用
   - 挂载操作将设备上的文件系统与操作系统的目录结构连接
   - 用户可以通过路径访问存储设备的内容

   在Linux中挂载设备：
   ```bash
   sudo mount /dev/sda1 /mnt
   ```

4. 文件系统检查与修复
   - 文件系统可能因突然断电或设备损坏而不一致
   - 操作系统执行文件系统检查（fsck）修复问题

   在Linux中手动运行文件系统检查：
   ```bash
   sudo fsck /dev/sda1
   ```

5. 挂载配置（可选）
   - 可以将存储设备配置为系统启动时自动挂载
   - 通过编辑`/etc/fstab`文件完成配置

   例如，添加以下行将设备`/dev/sda1`挂载到`/mnt`：
   ```bash
   /dev/sda1 /mnt ext4 defaults 0 2
   ```

### IOMMU

IOMMU（IO设备内存空间管理单元）：
- 在一些硬件平台上支持IOMMU技术
- 添加IOMMU单元，在CPU访问物理内存地址时添加类似MMU的内存虚拟技术
- 针对IO设备
- 通常伴随DMA一同出现

### Socket套接字

Socket是操作系统提供的跨进程通信底层抽象：
- 基于bind、listen、accept等操作
- 实现对网络通信的封装
- 让上层能够使用C函数方便地调用

服务器端代码示例：
```c
// 创建套接字
int server_fd = socket(AF_INET, SOCK_STREAM, 0);

// 绑定套接字到本机地址
struct sockaddr_in address;
address.sin_family = AF_INET;
address.sin_addr.s_addr = INADDR_ANY; // 监听所有接口
address.sin_port = htons(PORT); // 绑定端口
int ret = bind(server_fd, (struct sockaddr*)&address, sizeof(sockaddr_in));

// 开始监听
listen(server_fd, 3);

// 接收客户端请求
int connect_fd;
while(1) {
    // 阻塞直到有客户端连接
    connect_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addr_len);

    char recv_buf[1024];
    ssize_t bytes_received = 0;
    while(1) {
        // 接收数据
        bytes_received = recv(connect_fd, recv_buf, sizeof(recv_buf), 0);
        // 也可以使用通用的文件描述符操作函数
        // bytes_received = read(connect_fd, recv_buf, sizeof(recv_buf));
    }
}
```

### IO 多路复用

Linux系统通过select、poll、epoll提供系统级别的事件监听机制：
- select：通过fd_set结构维护文件描述符集合
- 每次调用select函数会阻塞
- 内核循环遍历fd_set，监听文件描述符变化
- 找出可以处理的文件描述符进行处理



## 万物皆文件

"一切皆文件"是UNIX的著名哲学理念。在Linux中：
- 具体文件、设备、网络socket等都可以抽象为文件
- 内核通过虚拟文件系统（VFS）提供统一界面
- 程序可以通过文件描述符fd调用IO函数访问文件
- 应用程序可以调用select、poll、epoll等系统调用监听文件变化

常见的IO函数：
- open
- read
- write
- ioctl
- close


## 文件管理

Linux至少需要一个存储设备来建立文件系统：
- 对数据文件进行持久化
- 满足"一切皆文件"的设计哲学
- 最小 Linux 机器实例需要挂载硬盘或基于内存的假文件系统

### VFS（虚拟文件系统）

虚拟文件系统的作用：
- 实现UNIX环境下"一切皆文件"的具体方式
- 为用户空间提供树形结构的文件目录结构
- 让用户通过文件路径访问系统资源
- 所有资源都被抽象成文件
- 用户态程序可以使用统一的操作文件接口

### 文件系统

文件系统用于描述磁盘中数据的组织方式和结构：
- 磁盘在挂载到虚拟文件系统前需要格式化
- 格式化过程就是建立文件系统的过程

> 注意区分文件系统和VFS：
> - 文件系统用于管理和描述块设备中的数据
> - VFS是Linux中的文件结构抽象