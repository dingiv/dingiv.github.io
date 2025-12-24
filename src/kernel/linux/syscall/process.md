# 进程管理
进程管理 API 主要涉及进程的生命周期管理。

## fork 系列
fork 系列创建子进程，复制父进程的虚拟地址空间。复制父进程的内存（代码段、数据段、栈、堆等），但通过**写时复制Copy-on-Write, COW）**优化，仅在修改时复制页面。子进程继承父进程的文件描述符、信号处理程序等。
```c
pid_t fork(void);
int clone(int (*fn)(void *), void *stack, int flags, void *arg, ...);

#include <stdio.h>
#include <unistd.h>

int main() {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    } else if (pid == 0) {
        printf("Child process, PID: %d\n", getpid());
    } else {
        printf("Parent process, Child PID: %d\n", pid);
    }
    return 0;
}
```

## exec 系列
exec 系列在当前进程中加载并执行新程序，替换当前进程的代码段、数据段、栈和堆。

```c
int execl(const char *path, const char *arg, ... /* (char *)NULL */);
int execvp(const char *file, char *const argv[]);
```

## wait 和 waitpid
防止子进程成为僵尸进程（已终止但未回收）。wait 阻塞直到任意子进程终止；waitpid 提供更灵活控制。
```c
pid_t wait(int *status);
pid_t waitpid(pid_t pid, int *status, int options);
```

## exit 和 kill

```c
void exit(int status);

pid_t getpid(void);
pid_t getppid(void);
int kill(pid_t pid, int sig);
```

## mount

```c

```

## 调度管理
获取或设置进程的优先级（nice 值），影响调度。调整进程的调度优先级，影响 CPU 分配。常用于优化实时任务或后台进程。
```c
/**
 * which：PRIO_PROCESS（进程）、PRIO_PGRP（进程组）、PR捕捉PRIO_USER`（用户）。
 * who：进程 ID（0 表示当前进程）。
 * prio：优先级（-20 到 19，值越低优先级越高）。
 */
int getpriority(int which, id_t who);
int setpriority(int which, id_t who, int prio);

// nice 设置新进程的优先级，renice 修改运行中进程的优先级
int nice(int inc);
int renice(int which, id_t who, int nice_value);
```