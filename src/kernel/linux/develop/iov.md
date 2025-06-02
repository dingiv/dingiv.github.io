# iovec
scatter read/ gather write，散布读/聚集写，聚簇读写，是 C 语言中常用的 IO 批读写的编码技巧，通过 iovec 结构体可以实现多个不连续内存区域的批量读写操作。iovec 结构其实就是一个带长度的数组索引，使用 void 指针来支持任意类型的元素。

```c
struct iovec {
    void  *iov_base;    // 内存起始地址
    size_t iov_len;     // 内存长度
};
```

## 系统调用
Linux 提供了以下系统调用来支持 iovec 操作：

1. readv/writev
```c
#include <sys/uio.h>

ssize_t readv(int fd, const struct iovec *iov, int iovcnt);
ssize_t writev(int fd, const struct iovec *iov, int iovcnt);
```

2. preadv/pwritev
```c
#include <sys/uio.h>

ssize_t preadv(int fd, const struct iovec *iov, int iovcnt, off_t offset);
ssize_t pwritev(int fd, const struct iovec *iov, int iovcnt, off_t offset);
```

## 使用示例

### 1. 基本读写示例
```c
#include <stdio.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    int fd = open("test.txt", O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("open");
        return -1;
    }

    // 准备三个不连续的内存区域
    char buf1[10] = "Hello ";
    char buf2[10] = "World ";
    char buf3[10] = "IOV!";

    // 设置IOV结构
    struct iovec iov[3];
    iov[0].iov_base = buf1;
    iov[0].iov_len = strlen(buf1);
    iov[1].iov_base = buf2;
    iov[1].iov_len = strlen(buf2);
    iov[2].iov_base = buf3;
    iov[2].iov_len = strlen(buf3);

    // 一次性写入所有数据
    ssize_t n = writev(fd, iov, 3);
    if (n < 0) {
        perror("writev");
        return -1;
    }

    // 重置文件偏移量
    lseek(fd, 0, SEEK_SET);

    // 准备读取缓冲区
    char read_buf1[10] = {0};
    char read_buf2[10] = {0};
    char read_buf3[10] = {0};

    // 设置读取IOV结构
    struct iovec read_iov[3];
    read_iov[0].iov_base = read_buf1;
    read_iov[0].iov_len = sizeof(read_buf1);
    read_iov[1].iov_base = read_buf2;
    read_iov[1].iov_len = sizeof(read_buf2);
    read_iov[2].iov_base = read_buf3;
    read_iov[2].iov_len = sizeof(read_buf3);

    // 一次性读取所有数据
    n = readv(fd, read_iov, 3);
    if (n < 0) {
        perror("readv");
        return -1;
    }

    printf("Read: %s%s%s\n", read_buf1, read_buf2, read_buf3);
    close(fd);
    return 0;
}
```

### 2. 网络编程示例
```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/uio.h>

int send_http_response(int sockfd) {
    // HTTP响应头
    const char *header = "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/html\r\n"
                        "Content-Length: 13\r\n"
                        "\r\n";
    
    // HTTP响应体
    const char *body = "Hello, World!";

    // 设置IOV结构
    struct iovec iov[2];
    iov[0].iov_base = (void *)header;
    iov[0].iov_len = strlen(header);
    iov[1].iov_base = (void *)body;
    iov[1].iov_len = strlen(body);

    // 一次性发送所有数据
    return writev(sockfd, iov, 2);
}
```

iovec 结构体提供了显著的性能优势。通过将多个读写操作合并为一次系统调用来减少系统调用次数，从而降低系统调用开销并提高 IO 效率。支持分散/聚集 IO，可以处理不连续的内存区域，避免内存拷贝，提高内存使用效率。提供了原子性操作，保证数据完整性，避免数据竞争，简化错误处理。

在使用 iovec 时需要注意几个关键点。内存对齐方面，需要确保 iov_base 指向的内存是有效的，注意内存边界检查，避免缓冲区溢出。错误处理方面，要检查返回值，处理部分写入/读取的情况，考虑 EINTR 错误。性能方面，需要合理设置 iovcnt 大小，避免过大的单次 IO 操作，考虑使用异步 IO。
