# io_uring
io_uring 是 Linux 内核 5.1（2019 年）引入的一套全新的异步 I/O 框架，由 Jens Axboe 设计并主导开发。它从根本上重新思考了 Linux 异步 I/O 的实现方式，通过共享环形缓冲区（ring buffer）在内核和用户空间之间传递 I/O 请求与完成事件，大幅减少了系统调用次数和内存拷贝开销。在高性能存储、网络服务、数据库等对 I/O 延迟和吞吐量要求极高的场景中，io_uring 已经成为事实标准。

## 设计动机
在 io_uring 出现之前，Linux 的异步 I/O 方案存在明显缺陷。传统的 Linux AIO（`libaio`）虽然提供了异步接口，但存在诸多限制：它只能用于 O_DIRECT 模式的文件 I/O，不支持缓冲 I/O 和普通文件；内部实现仍然需要陷入内核态提交请求，系统调用开销大；且 API 设计复杂，容易出错。另一条路线是通过 epoll 等多路复用机制实现伪异步，本质上仍然是同步非阻塞 I/O，需要应用层维护状态机，编程复杂度高。

epoll 虽然在高并发网络场景下表现优异，但它解决的是"事件通知"问题而非真正的异步 I/O。应用线程仍然需要自己发起 read/write 系统调用，在数据未就绪时返回 EAGAIN 并重新等待。这意味着高负载下每个连接都会产生大量系统调用，CPU 在用户态和内核态之间反复切换，成为性能瓶颈。io_uring 的目标就是提供一个真正异步、高性能、接口简洁统一的 I/O 框架。

## 核心架构
io_uring 的核心思想是**生产者-消费者模型**配合**共享内存**，将 I/O 请求的提交和完成事件的通知都通过环形缓冲区完成，大部分操作无需系统调用。

### 环形缓冲区
io_uring 维护两个环形缓冲区：**提交队列（Submission Queue, SQ）** 和 **完成队列（Completion Queue, CQ）**。SQ 由用户空间向内核提交 I/O 请求，CQ 由内核向用户空间通知 I/O 完成事件。这两个队列通过 `mmap` 映射到用户空间，用户态代码可以直接读写，无需系统调用介入。

SQ 和 CQ 的结构都是 ring buffer，通过头尾索引实现无锁的循环队列。SQ 的尾索引由用户空间推进（表示有新请求），头索引由内核推进（表示请求已被消费）；CQ 的头索引由用户空间推进（表示完成事件已被处理），尾索引由内核推进（表示有新完成事件）。这种设计使得生产者和消费者各自拥有独立的索引，无需加锁即可并发操作。

### 系统调用
io_uring 仅需三个系统调用：

- `io_uring_setup(entries, params)`：创建一个 io_uring 实例，返回文件描述符。`entries` 指定 SQ 和 CQ 的深度，`params` 是一个结构体，用于配置各种特性标志和获取映射所需的偏移量与大小信息。
- `io_uring_enter(fd, to_submit, min_complete, flags, sigset, sz)`：通知内核有新的 SQE 待处理，或者等待至少 `min_complete` 个 CQE 完成。设置了 `IORING_SETUP_SQPOLL` 后，内核会创建一个内核线程轮询 SQ，大部分情况下连这个系统调用都可以省去。
- `io_uring_register(fd, opcode, arg, nr_args)`：注册固定资源（文件描述符、缓冲区、事件fd 等），注册后的资源通过索引引用，内核可以直接使用，避免了每次请求都要解析文件描述符和拷贝缓冲区地址的开销。

### SQE 与 CQE
提交队列中的每个条目称为 **SQE（Submission Queue Entry）**，完成队列中的每个条目称为 **CQE（Completion Queue Entry）**。SQE 包含操作类型（如读、写、接受连接、发送等）、文件描述符、缓冲区地址和长度、用户数据等字段。CQE 包含操作结果（返回值，负值表示错误）和对应的用户数据，用于将完成事件与提交的请求关联起来。

SQE 的结构是 64 字节，CQE 是 16 字节，两者大小固定，便于内核和用户空间高效处理。用户在提交请求时，可以在 SQE 的 `user_data` 字段中填入自定义数据（通常是指向应用层请求结构的指针或请求 ID），当对应的 CQE 出现时，通过该字段找回原始请求上下文。

```c
struct io_uring_sqe {
    __u8  opcode;        // 操作类型：IORING_OP_READ, IORING_OP_WRITE 等
    __u8  flags;         // 标志位
    __u16 ioprio;        // I/O 优先级
    __s32 fd;            // 文件描述符
    union {
        __u64 off;       // 读写的偏移量
        __u64 addr2;
    };
    union {
        __u64 addr;      // 缓冲区地址
        __u64 splice_off_in;
    };
    __u32 len;           // 缓冲区长度
    union {
        __kernel_rwf_t rw_flags;  // 读写标志
        __u32 fsync_flags;
        __u16 poll_events;
        __u32 poll32_events;
        __u32 sync_range_flags;
        __u32 msg_flags;
        __u32 timeout_flags;
        __u32 accept_flags;
        __u32 cancel_flags;
        __u32 open_flags;
        __u32 statx_flags;
        __u32 fadvise_advice;
        __u32 splice_flags;
        __u32 rename_flags;
        __u32 unlink_flags;
        __u32 hardlink_flags;
        __u32 xattr_flags;
        __u32 msg_ring_flags;
        __u32 uring_cmd_flags;
        __u32 waitid_flags;
        __u32 futex_flags;
        __u32 install_fd_flags;
    };
    __u64 user_data;     // 用户自定义数据，原样返回
    union {
        __u16 buf_index;     // 固定缓冲区索引（selective shared buffer）
        __u16 buf_group;
    };
    __u16 personality;   // 凭证选择
    union {
        __s32 splice_fd_in;
        __u32 file_index;
        union {
            __u16 cmd_op;
            __u16 zone;
        };
    };
    __u64 __pad2[3];
};

struct io_uring_cqe {
    __u64 user_data;     // 对应 SQE 中的 user_data
    __s32 res;           // 操作结果
    __u32 flags;         // 完成标志
};
```

## 基本使用

### 基于 liburing
直接操作 io_uring 的系统调用接口较为繁琐，实际开发中通常使用 `liburing` 库，它封装了环形缓冲区的管理、SQE/CQE 的获取与提交、内存映射等细节，提供了简洁易用的 API。

```c
#include <stdio.h>
#include <fcntl.h>
#include <liburing.h>

#define QUEUE_DEPTH 64

int main() {
    struct io_uring ring;
    // 初始化 io_uring 实例
    io_uring_queue_init(QUEUE_DEPTH, &ring, 0);

    int fd = open("test.txt", O_RDONLY);
    char buf[4096];

    // 获取一个空闲的 SQE
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    // 填充 SQE：读操作
    io_uring_prep_read(sqe, fd, buf, sizeof(buf), 0);
    // 设置用户数据，用于在 CQE 中识别请求
    sqe->user_data = 1;

    // 提交请求并等待完成
    io_uring_submit(&ring);

    struct io_uring_cqe *cqe;
    // 等待一个完成事件
    io_uring_wait_cqe(&ring, &cqe);
    printf("read %d bytes\n", cqe->res);
    // 标记 CQE 已消费
    io_uring_cqe_seen(&ring, cqe);

    close(fd);
    io_uring_queue_exit(&ring);
    return 0;
}
```

### 批量提交
io_uring 支持批量提交多个请求，减少系统调用次数。只需连续获取多个 SQE 并填充，然后一次性提交即可。

```c
#define BATCH_SIZE 8

struct io_uring_sqe *sqes[BATCH_SIZE];
char bufs[BATCH_SIZE][4096];

// 批量获取 SQE 并填充
for (int i = 0; i < BATCH_SIZE; i++) {
    sqes[i] = io_uring_get_sqe(&ring);
    io_uring_prep_read(sqes[i], fd, bufs[i], sizeof(bufs[i]), i * 4096);
    sqes[i]->user_data = i;
}
// 一次性提交所有请求
io_uring_submit(&ring);

// 批量收割完成事件
for (int i = 0; i < BATCH_SIZE; i++) {
    io_uring_wait_cqe(&ring, &cqe);
    printf("req %llu: %d bytes\n", cqe->user_data, cqe->res);
    io_uring_cqe_seen(&ring, cqe);
}
```

## 高级特性

### 固定文件与固定缓冲区
io_uring 的 `io_uring_register` 系统调用支持预先注册文件描述符和缓冲区，注册后通过索引引用而非描述符编号，内核可以直接从内部表中获取对应资源，省去了每次请求时解析文件描述符和拷贝用户缓冲区地址的开销。

固定文件通过 `IORING_REGISTER_FILES` 注册，提交请求时使用 `IOSQE_FIXED_FILE` 标志并在 `fd` 字段填写注册索引。固定缓冲区通过 `IORING_REGISTER_BUFFERS` 注册，使用 `IOSQE_FIXED_BUFFER` 标志并通过 `buf_index` 引用。在频繁复用同一组文件或缓冲区（如数据库的 page cache、网络服务器预分配的 buffer pool）的场景下，这两项优化能显著降低 CPU 开销。

```c
// 注册固定文件
int fds[] = {fd1, fd2, fd3};
io_uring_register_files(&ring, fds, 3);

// 使用固定文件提交请求
sqe = io_uring_get_sqe(&ring);
io_uring_prep_read(sqe, 0, buf, sizeof(buf), 0);  // fd=0 表示注册表索引 0
sqe->flags |= IOSQE_FIXED_FILE;
```

### 零拷贝
io_uring 提供了两种零拷贝机制：**固定缓冲区**和 **提供的缓冲区（provided buffers）**。

固定缓冲区注册后由内核直接 DMA 到用户缓冲区，省去了内核态缓冲区到用户态缓冲区的拷贝。provided buffers 则更进一步，用户提前向内核注册一组空缓冲区池（buffer pool），内核在完成读操作时自动选择一个空闲缓冲区填充数据，通过 CQE 告知用户使用了哪个缓冲区。用户处理完数据后，将缓冲区归还给内核池。这种模式使得用户无需提前知道数据会写入哪个缓冲区，内核可以按需分配，非常适合请求-响应模式的网络服务。

```c
// 注册 provided buffer group
struct io_uring_buf_reg reg = {
    .ring_entries = 32,
    .bgid = 1,
};
io_uring_register_buf_ring(&ring, &reg, 0);

// 提交请求时使用 buffer group
sqe = io_uring_get_sqe(&ring);
io_uring_prep_recv(sqe, fd, NULL, 0, 0);
sqe->flags |= IOSQE_BUFFER_SELECT;
sqe->buf_group = 1;

// 在 CQE 中获取实际使用的缓冲区
io_uring_wait_cqe(&ring, &cqe);
if (cqe->flags & IORING_CQE_F_BUFFER) {
    int buf_id = cqe->flags >> 16;
    // buf_id 指向内核选中的缓冲区
}
```

### 链式请求
通过 `IOSQE_IO_LINK` 标志，可以将多个 SQE 串联成链，内核按顺序依次执行。链中前一个请求完成后才执行后一个。如果链中某个请求失败，后续请求会被取消（除非设置了 `IOSQE_IO_HARDLINK`）。

链式请求在需要严格顺序执行的 I/O 流水线中非常有用，例如先读后处理再写，或者先 fsync 再 close。相比用户空间手动管理依赖关系，链式请求将控制流交给内核，减少了用户态与内核态的交互。

```c
// 链式请求：先读后写
struct io_uring_sqe *sqe1 = io_uring_get_sqe(&ring);
io_uring_prep_read(sqe1, fd_in, buf, sizeof(buf), 0);
sqe1->flags |= IOSQE_IO_LINK;  // 链接到下一个请求

struct io_uring_sqe *sqe2 = io_uring_get_sqe(&ring);
io_uring_prep_write(sqe2, fd_out, buf, sizeof(buf), 0);
// sqe2 无需设置 link 标志，链在此结束

io_uring_submit(&ring);
```

### SQ Polling 模式
默认情况下，应用程序通过 `io_uring_enter` 通知内核处理 SQ。开启 `IORING_SETUP_SQPOLL` 后，内核会创建一个专门的内核线程（SQ poll 线程）持续轮询 SQ，发现新请求后立即处理。这种模式下，I/O 提交完全在用户态完成，连 `io_uring_enter` 系统调用都可以省去，实现了真正的零系统调用 I/O 路径。

SQ Polling 模式的代价是 poll 线程会持续占用一个 CPU 核，即使没有 I/O 请求也会空转。因此它适用于 I/O 密集型应用，特别是请求持续不断到达的场景。内核提供了 `IORING_SETUP_SQ_AFF` 选项，可以将 poll 线程绑定到指定 CPU 核上，减少调度干扰。同时可以通过 `sq_thread_idle` 参数设置 poll 线程的空闲超时时间，超过该时间后线程自动休眠，直到被下一次 `io_uring_enter` 唤醒。

```c
struct io_uring_params params = {0};
params.flags = IORING_SETUP_SQPOLL;
params.sq_thread_idle = 2000;  // 空闲 2ms 后休眠
io_uring_queue_init_params(QUEUE_DEPTH, &ring, &params);
```

## 支持的操作类型
io_uring 最初只支持文件读写，但随着版本迭代，操作类型不断扩展，已经覆盖了绝大多数系统 I/O 场景：

- 文件 I/O：`IORING_OP_READ`、`IORING_OP_WRITE`、`IORING_OP_READV`、`IORING_OP_WRITEV`、`IORING_OP_FSYNC`、`IORING_OP_FDATASYNC`
- 网络 I/O：`IORING_OP_RECV`、`IORING_OP_SEND`、`IORING_OP_ACCEPT`、`IORING_OP_CONNECT`、`IORING_OP_RECVMSG`、`IORING_OP_SENDMSG`
- 文件操作：`IORING_OP_OPENAT`、`IORING_OP_CLOSE`、`IORING_OP_STATX`、`IORING_OP_UNLINK`、`IORING_OP_RENAME`
- 缓冲区管理：`IORING_OP_PROVIDE_BUFFERS`、`IORING_OP_REMOVE_BUFFERS`
- 事件通知：`IORING_OP_POLL_ADD`、`IORING_OP_POLL_REMOVE`
- 超时与取消：`IORING_OP_TIMEOUT`、`IORING_OP_TIMEOUT_REMOVE`、`IORING_OP_ASYNC_CANCEL`
- 进程管理：`IORING_OP_WAITID`、`IORING_OP_SIGNAL`
- 高级特性：`IORING_OP_SPLICE`、`IORING_OP_TEE`、`IORING_OP_SHUTDOWN`、`IORING_OP_SOCKET`

这意味着 io_uring 已经不再局限于"异步文件 I/O"，而是一个通用的异步操作框架，理论上任何需要系统调用的操作都可以通过 io_uring 异步提交。

## 与传统方案对比

### 系统调用开销
传统同步 I/O 或 epoll 模型中，每个 I/O 操作至少对应一次系统调用（read/write），在高并发下系统调用次数与 I/O 次数成正比。io_uring 通过共享内存批量提交和收割，可以将数百个 I/O 操作压缩到少数几次系统调用中完成。在 SQ Polling 模式下，甚至可以实现零系统调用。

### 上下文切换
epoll 模型下，每次 read/write 都涉及用户态到内核态的切换。虽然单次切换的开销在微秒级别，但在每秒百万级 I/O 的场景下，累积开销不容忽视。io_uring 的批量处理模式显著减少了上下文切换次数，SQ Polling 模式更是将提交路径完全留在用户态。

### 内核态拷贝
传统 I/O 路径中，数据需要从内核缓冲区拷贝到用户缓冲区（反之亦然）。io_uring 的固定缓冲区和 provided buffers 机制允许内核直接 DMA 到用户预注册的缓冲区，在支持零拷贝的场景下（如网络到存储的转发）可以完全消除内存拷贝。

### 编程复杂度
epoll 的异步是"半异步"的，应用需要维护状态机处理 EAGAIN，在读写未完成时重新注册等待事件。io_uring 是真正的异步，提交请求后只需等待 CQE，无需处理中间状态，编程模型更简洁。对于复杂的 I/O 流水线，链式请求将依赖关系交给内核管理，进一步降低了应用层的复杂度。

## 工程实践

### 性能调优
在实际部署中，队列深度的选择需要根据负载特征调整。过小的队列会导致 SQE 分配失败（需要频繁调用 `io_uring_enter` 刷出已提交的请求），过大的队列则浪费内存。一般建议从 256 或 1024 开始，根据实际 I/O 并发度调整。SQ 和 CQ 的深度可以分别通过 `sq_entries` 和 `cq_entries` 参数独立设置，CQ 深度通常需要大于 SQ 深度（因为链式请求和某些操作可能产生多个 CQE）。

对于延迟敏感型应用，开启 SQ Polling 并绑定到专用 CPU 核是常见的优化手段。对于吞吐量敏感型应用，关闭 SQ Polling 以节省 CPU 资源，依靠批量提交和收割来摊薄系统调用开销。固定文件和固定缓冲区的注册是一次性开销，在文件和缓冲区被反复使用的场景下（如数据库 page I/O、网络 buffer pool）收益显著。

### 错误处理
io_uring 的错误通过 CQE 的 `res` 字段返回，负值表示错误码（与 errno 值相同，如 `-EINVAL`、`-EAGAIN` 等）。需要注意的是，io_uring 在提交阶段几乎不做参数校验，大多数错误在执行阶段才暴露。因此应用代码必须检查每个 CQE 的返回值，特别是对于批量提交的场景，不能假设所有请求都会成功。

对于 `EAGAIN` 类错误（如非阻塞 socket 当前无数据），可以通过注册 `IORING_OP_POLL_ADD` 先等待事件就绪，然后再提交实际的 I/O 操作，避免用户态忙轮询。

### 内核版本兼容
io_uring 从 Linux 5.1 开始引入，核心功能在 5.10 左右趋于稳定。许多高级特性（如 provided buffers、registered ring buffers、socket 操作支持等）在 5.19 和 6.x 系列中才逐步完善。如果需要使用最新特性，建议至少使用 6.1 以上的内核版本。在编写兼容多版本内核的代码时，需要通过 `io_uring_params.features` 检查内核支持的功能标志，按需降级处理。
