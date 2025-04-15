# io 操作

## 标准输入输出
```c
#include <stdio.h>

int printf(const char *format, ...);
int scanf(const char *format, ...);
int sprintf(char *str, const char *format, ...);
int sscanf(const char *str, const char *format, ...);
int fprintf(FILE *stream, const char *format, ...);
int fscanf(FILE *stream, const char *format, ...);

/**
 * v 开头的是显示声明的可变参数函数
 * va_list 是一个类型，用于存储格式化字符串和参数列表
 * va_start 是一个宏，用于初始化 va_list
 * va_arg 是一个宏，用于获取参数列表中的下一个参数
 * va_end 是一个宏，用于结束参数列表的遍历
 */
int vprintf(const char *format, va_list ap);
int vscanf(const char *format, va_list ap);
int vsprintf(char *str, const char *format, va_list ap);
int vsscanf(const char *str, const char *format, va_list ap);
int vfprintf(FILE *stream, const char *format, va_list ap);
int vfscanf(FILE *stream, const char *format, va_list ap);

/**
 * 简洁的输入输出函数
 */
int gets(char *s);   // 从标准流读取，不推荐使用
int puts(const char *s);  // 向标准流写入
int fgets(char *s, int size, FILE *stream);   // 从文件流读取
int fputs(const char *s, FILE *stream);   // 向文件流写入

int getchar(void);   // 从标准流读取一个字符
int putchar(int c);    // 向标准流写入一个字符
int fgetc(FILE *stream);  // 从文件流读取一个字符
int fputc(int c, FILE *stream);   // 向文件流写入一个字符
```

## 文件
```c
// 打开文件
FILE *fopen(const char *filename, const char *mode);
FILE *fdopen(int fd, const char *mode);
FILE *freopen(const char *filename, const char *mode, FILE *stream);

// 关闭文件
int fclose(FILE *stream);

// 读写文件
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);

// 文件定位
long ftell(FILE *stream);
int fseek(FILE *stream, long offset, int whence);


```
