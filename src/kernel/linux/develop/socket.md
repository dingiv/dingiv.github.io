# 套接字

### 字节序处理

1. 基本概念
   - 内存和数据流被抽象为字节数组
   - 数据编码需要先转换为 16 进制
   - 内存地址从低到高：0x00000000 -> 0xffffffff

2. 字节序转换
   - 不同平台 CPU 的大小端序不同
   - 网络通信和驱动编写需要固定字节序
   - 使用转换函数处理字节序：
     ```c
     cpu_to_le32()
     cpu_to_le64()
     ```




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

