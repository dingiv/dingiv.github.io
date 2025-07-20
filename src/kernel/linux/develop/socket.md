# 套接字
套接字是实现跨进程通信的标准 API，被各大操作系统广泛支持，是现代网络通信的基础。通过套接字，应用程序可以方便地实现本地或跨主机的进程间数据交换。

## 基本操作流程
典型的网络服务端程序，通常包含以下步骤：
1. 创建套接字（socket）
2. 绑定地址（bind）
3. 监听连接（listen）
4. 接受连接（accept）
5. 数据收发（read/write/recv/send）
6. 关闭连接（close）

示例代码：

```c
// 创建套接字
int server_fd = socket(AF_INET, SOCK_STREAM, 0);

// 绑定套接字到本机地址
struct sockaddr_in address;
address.sin_family = AF_INET;
address.sin_addr.s_addr = INADDR_ANY; // 监听所有接口
address.sin_port = htons(PORT); // 绑定端口
bind(server_fd, (struct sockaddr*)&address, sizeof(address));

// 开始监听
listen(server_fd, 3);

// 接收客户端请求
int connect_fd;
while (1) {
    // 阻塞调用，该函数将阻塞线程，直至有客户端来连接 
    connect_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addr_len);
    char recv_buf[1024];
    ssize_t bytes_received;
    // 阻塞调用，该函数将阻塞线程，直至客户端发送消息
    while ((bytes_received = recv(connect_fd, recv_buf, sizeof(recv_buf), 0)) > 0) {
        // 处理接收到的数据
    }
    close(connect_fd);
}
```

## 套接字类型与协议族
套接字支持多种协议族和类型，常见的有：
- `AF_INET`：IPv4 网络协议
- `AF_INET6`：IPv6 网络协议
- `SOCK_STREAM`：面向连接的字节流（如 TCP）
- `SOCK_DGRAM`：无连接的数据报（如 UDP）
- `SOCK_RAW`：原始套接字，常用于实现自定义协议

```c
// UDP
int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
// 绑定、发送、接收与 TCP 类似，但不需要 listen/accept
```

### 选项与控制
通过 `setsockopt` 和 `getsockopt` 可以灵活控制套接字行为，例如：
- 地址重用：`setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, ...)`
- 设置发送/接收缓冲区大小
- 设置超时：`SO_RCVTIMEO`、`SO_SNDTIMEO`
- 启用/禁用 Nagle 算法：`TCP_NODELAY`
这些选项有助于优化网络程序的性能和健壮性。

## IO 多路复用
默认情况下，套接字操作是阻塞的，当程序想要等待客户端连接和从客户端读写数据的时候，它会陷入休眠，直到有数据到来，它才会被操作系统唤醒，从而开始执行响应。这样导致了一个 server 线程同时只能服务于一个客户。

可以通过如下方式设置为非阻塞模式：
```c
#include <fcntl.h>
int flags = fcntl(fd, F_GETFL, 0);
fcntl(fd, F_SETFL, flags | O_NONBLOCK);
```

IO 多路复用技术允许一个线程同时监听多个套接字和文件描述符，从而使得 server 的并发性和可用性获得提高。Linux 提供了多种 IO 多路复用机制：
- **select**：通过 `fd_set` 结构维护文件描述符集合，适合少量连接，被广大系统支持，移植性好。简单，但是性能受限。不推荐。
- **poll**：支持更多文件描述符，接口更灵活。与 select 类似，连接更多一点，但是没多多少。不推荐。
- **epoll**：Linux 特有，适合大规模并发连接，效率高，常用于高性能服务器。编写难度大，推荐。

```c
// epoll
#include <sys/epoll.h>
int epfd = epoll_create(1024);
struct epoll_event ev, events[10];
ev.events = EPOLLIN;
ev.data.fd = listen_fd;
epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev);
int n = epoll_wait(epfd, events, 10, -1);
```

一般地，对于低并发场景可以使用多线程技术来进行编写，高并发场景推荐 epoll，这个分界线在 100-500 并发左右。更高的并发，需要引入协程和异步架构，实现起来较为复杂。但是，现代的编程语言，例如：Go，Rust 等有丰富的支持，开箱即用。

## 分包与拆包
由于 TCP 是流协议，消息边界不固定，应用层需要根据协议自行处理分包和拆包，确保数据完整性，并实现自己的业务逻辑，业务层的消息也需要自行封装逻辑。

如果不希望自己实现，可以使用已有的逻辑，例如 Http(s) 和 Ftp 等常见的协议，从而快速实现开发。另一方面，如果客户的设备没有相应的客户端协议实现，那么也无法通信，使用通用的协议可以降低沟通成本。但是，Http 的性能较差，在服务集群中，一般会进行自定义协议，例如各种 RPC 协议，例如：gRPC。

### 头体分包
头体分包算法是解决字节流边界，封装自定义协议的通用解决方案。同时也是适用任何基于字节数组和字节流的数据组织和管理的
算法。在基于 TCP 的网络通信中，由于 TCP 是字节流协议，消息边界不固定，常见的做法是采用“包头+包体”分包算法来解决粘包和拆包问题。

每条消息由“包头”和“包体”两部分组成：
- **包头**：通常为固定长度（如 4 字节），用于存放包体长度（或类型、校验等信息）。最常见的是前 4 字节存放包体长度（不含包头本身）。
- **包体**：紧跟在包头之后，长度由包头指定，存放实际业务数据。

**数据格式示例：**
| 字节偏移 | 0~3         | 4~N         |
|----------|-------------|-------------|
| 内容     | 包体长度L   | 包体内容    |

#### 发送方流程
1. 将要发送的数据序列化为字节流。
2. 计算包体长度 L。
3. 构造包头（如 4 字节，存放 L，通常用网络字节序）。
4. 发送“包头+包体”拼接后的完整数据。

```c
uint32_t len = htonl(body_len); // 转为网络字节序
send(fd, &len, 4, 0);
send(fd, body, body_len, 0);
```

#### 接收方流程
1. 先读取固定长度的包头（如 4 字节）。
2. 解析包头，获得包体长度 L。
3. 再读取 L 字节的包体。
4. 解析包体，处理业务。

```c
// 1. 读取包头
uint32_t len;
recv(fd, &len, 4, MSG_WAITALL);
len = ntohl(len); // 转为主机字节序

// 2. 读取包体
char *body = malloc(len);
recv(fd, body, len, MSG_WAITALL);
// 3. 处理 body
```

头体分包方案具有很强的通用性和高效性，适用于各种基于字节流的协议。它只需简单的内存拷贝和长度判断即可完成数据的分包与组包，同时包头还可以灵活扩展，携带类型、校验、序号等元数据。广泛应用于自定义 TCP 协议、各类 RPC 框架（如 gRPC、Thrift 的底层实现）、以及游戏服务器、消息队列等高性能网络服务场景。

实际使用时，需要注意包头和包体的读取都要循环进行，确保读取到完整的数据，防止出现短读问题。同时，包头长度和字节序需要通信双方提前约定一致，包体长度也要做合理限制，避免恶意超大包体导致内存溢出等安全风险。

### 字节序处理
网络通信要求通信的双方使用统一的字节序（网络字节序为大端），因为发包的时候，单个字节和数据的排列可以有两种方向，并且不同平台 CPU 的本地字节序可能不同，所以需要特别注意不管双方的 CPU 字节序如何，都需要在通信的时候统一使用大端序。常用字节序转换函数有：
- `htons()`、`htonl()`：主机字节序转网络字节序
- `ntohs()`、`ntohl()`：网络字节序转主机字节序
- 内核开发常用：`cpu_to_le32()`、`cpu_to_le64()` 等
