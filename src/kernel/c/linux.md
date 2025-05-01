# linux 编程
linux 系统是基于 C 语言开发的，Linux 系统中的系统编程是指使用 C 语言进行与操作系统内核交互的编程，包括文件管理、进程控制、信号处理、网络编程等。Linux 提供了一套丰富的系统调用接口，允许开发者直接与操作系统交互。

## 系统调用

### 文件操作
open, close, read, write, lseek, unlink 等
```c
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main() {
    int fd = open("file.txt", O_WRONLY | O_CREAT, 0644);
    if (fd == -1) {
        // 处理错误
    }
    write(fd, "Hello, world!", 13);
    close(fd);
    return 0;
}

```

## 进程管理

## 线程调度

## 网络编程

## 内存管理
冯诺依曼机器体系下，内存是一个宽度为 8 比特或者 1 字节的数组，长度为 2<sup>64</sup>。CPU 从内存上读取二进制数据，然后将这些二进制数据以两种方式看待，一种是 **指令**，一种是 **数据**，指令指示 CPU 如何处理数据，如何进行 IO 操作，将数据持久化到内存之外的地方。

一个 64 位的硬件平台，一般 CPU 读取一次内存是读取 8 字节，少于 8 个字节需要读取两次。内存在

## 文件系统

## 硬件驱动

