---
title: IPC
---

# 进程间通信
进程是操作系统提供给上层的虚拟容器，为程序运行提供隔离和保护。容器之间需要交互，交互方式由操作系统提供相应的系统调用实现。

## 文件接口
使用文件系统的接口，使用字节流的读写机制，往往可以参考 socket 编程的机制，使用统一的字符设备读写方案进行编码。

### 套接字（Socket）
满足跨主机的通信、特别适合通过网络或专用线路通信、支持 TCP（可靠）和 UDP（不可靠）协议

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>

// 服务器端
int server_socket = socket(AF_INET, SOCK_STREAM, 0);
struct sockaddr_in server_addr;
server_addr.sin_family = AF_INET;
server_addr.sin_addr.s_addr = INADDR_ANY;
server_addr.sin_port = htons(8080);

bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr));
listen(server_socket, 5);

int client_socket = accept(server_socket, NULL, NULL);
char buffer[1024];
recv(client_socket, buffer, 1024, 0);
printf("Received: %s\n", buffer);
```

### 磁盘文件
简单但效率较低、适合持久化数据交换、需要文件系统支持
```c
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

// 进程A：写入文件
int fd = open("/tmp/ipc_file", O_WRONLY | O_CREAT, 0644);
write(fd, "Hello from process A", 20);
close(fd);

// 进程B：读取文件
int fd = open("/tmp/ipc_file", O_RDONLY);
char buf[100];
read(fd, buf, 100);
close(fd);
```

### eventfd
用于进程或线程间通信、支持用户态和内核态之间通信、轻量级，适合事件通知
```c
#include <sys/eventfd.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    int efd = eventfd(0, 0);  // 初始值为0，阻塞模式

    if (fork() == 0) {
        // 子进程：写入事件
        uint64_t value = 1;
        write(efd, &value, sizeof(value));
        printf("Event written\n");
    } else {
        // 父进程：读取事件
        uint64_t value;
        read(efd, &value, sizeof(value));
        printf("Event received: %lu\n", value);
    }

    close(efd);
    return 0;
}
```

### 管道
分为匿名管道和命名管道，匿名管道的信息保存在当前的进程中，命名管道可以在文件系统中提供文件接口，从而进行跨进程的通信。

匿名管道主要用于父子进程间通信，数据流是单向的；命名管道：支持无父子关系的进程间通信，基于文件系统，支持阻塞和非阻塞模式。

```c
#include <unistd.h>
#include <stdio.h>
#include <sys/wait.h>

int main() {
    int pipefd[2];
    pid_t pid;
    char buf[100];

    if (pipe(pipefd) == -1) {
        perror("pipe");
        return 1;
    }

    pid = fork();
    if (pid == 0) {
        // 子进程：写入数据
        close(pipefd[0]);  // 关闭读端
        write(pipefd[1], "Hello from child", 16);
        close(pipefd[1]);
    } else {
        // 父进程：读取数据
        close(pipefd[1]);  // 关闭写端
        read(pipefd[0], buf, 100);
        printf("Parent received: %s\n", buf);
        close(pipefd[0]);
        wait(NULL);
    }
    return 0;
}
```

```c
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>

// 创建FIFO
mkfifo("/tmp/myfifo", 0666);

// 写入进程
int fd = open("/tmp/myfifo", O_WRONLY);
write(fd, "Hello FIFO", 10);
close(fd);

// 读取进程
int fd = open("/tmp/myfifo", O_RDONLY);
char buf[100];
read(fd, buf, 100);
close(fd);
```

## 独立接口

### 信号
信号是一种软件软中断机制，为进程提供响应外界发送单向数据的机制。进程可以实现对特定信号的响应逻辑，类似于事件的概念。简单、轻量、异步，但是信息量有限、不可靠、难以调试。

```c
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

void signal_handler(int sig) {
    printf("Received signal %d\n", sig);
}

int main() {
    signal(SIGINT, signal_handler);  // 注册信号处理函数
    printf("Waiting for signal...\n");
    while(1) {
        sleep(1);
    }
    return 0;
}
```

操作系统或其他进程发送信号给目标进程，操作系统通知目标进程后，打断当前的进程执行，进程接收信号，执行信号处理程序。
常用信号：
- `SIGTERM`：优雅终止
- `SIGKILL`：强制终止（不可捕获）
- `SIGINT`：中断信号（Ctrl+C）
- `SIGUSR1/SIGUSR2`：用户自定义信号

如果程序没有显示注册，操作系统将为其使用默认行为：
- 终止进程（如 SIGKILL、SIGTERM）
- 暂停进程（如 SIGSTOP）
- 忽略信号（如 SIGCHLD）
- 继续执行（如 SIGCONT）

### 共享内存
基于 linux 内存映射机制，能够满足大数据量的极高性能的数据传输，但是需要手动管理内存同步。一般需要结合其他进程同步通信的机制来一起使用，以保证并发安全。

```c
#include <sys/shm.h>
#include <sys/ipc.h>
#include <stdio.h>
#include <string.h>

int main() {
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);

    // 附加共享内存
    char *str = (char*) shmat(shmid, (void*)0, 0);

    // 写入数据
    strcpy(str, "Hello Shared Memory");
    printf("Data written in memory: %s\n", str);

    // 分离共享内存
    shmdt(str);

    // 删除共享内存
    shmctl(shmid, IPC_RMID, NULL);
    return 0;
}
```

### 消息队列
Linux 系统提供的内核中维护的独特队列，基于内存实现，提供简单的异步提示机制。消息在队列中按顺序排列、支持优先级、提供良好的可靠性和同步机制。

```c
#include <sys/msg.h>
#include <sys/ipc.h>
#include <stdio.h>
#include <string.h>

struct msg_buffer {
    long msg_type;
    char msg_text[100];
};

int main() {
    key_t key = ftok("progfile", 65);
    int msgid = msgget(key, 0666 | IPC_CREAT);

    struct msg_buffer message;
    message.msg_type = 1;
    strcpy(message.msg_text, "Hello Message Queue");

    // 发送消息
    msgsnd(msgid, &message, sizeof(message), 0);

    // 接收消息
    msgrcv(msgid, &message, sizeof(message), 1, 0);
    printf("Received: %s\n", message.msg_text);

    // 删除消息队列
    msgctl(msgid, IPC_RMID, NULL);
    return 0;
}
```

### 信号量（Semaphore）
信号量提供两种类型：
- 二值信号量（互斥锁）：值只有 0 和 1
- 计数信号量：值可以是非负整数，表示资源剩余数量

> + 虽然叫信号量，但是和信号没有关系
> + 本身应该属于同步原语的内容，但是它支持进程间的同步
> + 从中文翻译上来看也不太对，应该翻译成**旗语**

```c
#include <sys/sem.h>
#include <sys/ipc.h>
#include <stdio.h>
#include <unistd.h>

union semun {
    int val;
    struct semid_ds *buf;
    unsigned short *array;
};

int main() {
    key_t key = ftok("semfile", 65);
    int semid = semget(key, 1, 0666 | IPC_CREAT);

    union semun sem_union;
    sem_union.val = 1;  // 初始值为1（二值信号量）
    semctl(semid, 0, SETVAL, sem_union);

    struct sembuf sb;
    sb.sem_num = 0;
    sb.sem_op = -1;  // P操作（获取）
    sb.sem_flg = 0;
    semop(semid, &sb, 1);

    // 临界区代码
    printf("In critical section\n");
    sleep(2);

    sb.sem_op = 1;   // V操作（释放）
    semop(semid, &sb, 1);

    return 0;
}
```

## IPC 机制对比

| 机制     | 类型      | 速度 | 可靠性 | 复杂度 | 适用场景     |
| -------- | --------- | ---- | ------ | ------ | ------------ |
| Socket   | 同步/异步 | 中等 | 高     | 复杂   | 网络通信     |
| 文件     | 同步      | 慢   | 高     | 简单   | 持久化数据   |
| 管道     | 同步      | 中等 | 高     | 简单   | 父子进程通信 |
| 信号     | 异步      | 快   | 低     | 简单   | 事件通知     |
| 共享内存 | 同步      | 最快 | 高     | 复杂   | 大数据交换   |
| 消息队列 | 异步      | 中等 | 高     | 中等   | 结构化消息   |
| 信号量   | 同步      | 快   | 高     | 中等   | 同步控制     |

## 实际应用场景
1. 生产者-消费者模式：使用消息队列或共享内存+信号量实现生产者-消费者模式。
2. 进程池：使用管道或消息队列实现进程池的任务分发。
3. 数据库连接池：使用共享内存存储连接信息，信号量控制并发访问。
4. 日志系统：使用消息队列实现异步日志记录。

选择合适的 IPC 机制需要考虑：
- 数据量大小
- 实时性要求
- 可靠性要求
- 开发复杂度
- 性能要求
