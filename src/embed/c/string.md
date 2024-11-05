# 字符串操作
在C语言中，字符串是以'\0'结尾的字符数组。字符串操作函数通常定义在`<string.h>`头文件中。而 `char` 类型在C语言中是一个字符类型，它通常占用1个字节的空间，用于表示ASCII码中的字符，因此，一个字符基本可以认为是一个字节，一个字符可以认为是一个 0-255 的整数。

## 字符串函数

```c
#include <string.h>

/**
 * 字符串长度
 * 通过遍历字符串统计长度，直到遇到'\0'为止
 */
size_t strlen(const char *s);

/**
 * 字符串复制
 * 通过遍历字符串，将源字符串的字符复制到目标字符串，直到遇到'\0'为止
 */
char *strcpy(char *dest, const char *src);
char *strncpy(char *dest, const char *src, size_t n);
// strdup和strndup函数会自动为复制的内容分配内存，并将复制的内容返回，因此需要手动释放内存
char *strdup(const char *s);
char *strndup(const char *s, size_t n);

/**
 * 字符串比较
 * 通过遍历字符串，逐一比较两个字符串的字符，判断是否相等
 */
int strcmp(const char *s1, const char *s2);
int strncmp(const char *s1, const char *s2, size_t n);
int strcasecmp(const char *s1, const char *s2);  // 忽略大小写比较字符串
int strncasecmp(const char *s1, const char *s2, size_t n);  // 忽略大小写比较字符串的前n个字符

/**
 * 字符串查找
 */
char* strchr(const char *s, int c);  // 查找字符c在字符串s中首次出现的位置，返回指向该位置的指针，如果找不到则返回NULL
char* strstr(const char *haystack, const char *needle); // 查找字符串needle在字符串haystack中首次出现的位置，返回指向该位置的指针，如果找不到则返回NULL
char *strpbrk(const char *s, const char *accept); // 查找字符串 s 中第一个出现在 accept 字符串中的字符，并返回指向该字符的指针

/**
 * 字符串拼接
 * 通过遍历字符串，将源字符串的字符复制到目标字符串的末尾
 * 注意：目标字符串必须有足够的空间来容纳源字符串的内容
 */
char *strcat(char *dest, const char *src);
char *strncat(char *dest, const char *src, size_t n);

/**
 * 字符串分割
 * 通过遍历字符串，将字符串按照指定的分隔符进行分割，返回指向分割后的子字符串的指针数组
 * 注意：返回的指针数组需要手动释放内存
 */
char* strtok(char *str, const char *delim);

// eg:
char str[] = "Hello,World,This,Is,C";
const char delim[] = ",";

char *token = strtok(str, delim);  // 第一次调用，传入字符串
while (token != NULL) {
  printf("%s\n", token);          // 打印每个子串
  token = strtok(NULL, delim);    // 后续调用，传入 NULL
}
// 输出：
// Hello
// World
// This
// Is
// C


/**
 * 字符串转换
 * 将字符串转换为整数或浮点数
 */
int atoi(const char *nptr);  // 将字符串转换为 int
long atol(const char *nptr);  // 将字符串转换为 long
long long atoll(const char *nptr);  // 将字符串转换为 long long
double atof(const char *nptr);  // 将字符串转换为 double
long strtol(const char *nptr, char **endptr, int base);  // 将字符串转换为 long，并指定进制
double strtod(const char *nptr, char **endptr);  // 将字符串转换为 double

char* strlwr(char* str);  // 将字符串转换为小写
char* strupr(char* str);  // 将字符串转换为大写
char* strrev(char* str);  // 将字符串反转

// 字符串格式化
char *sprintf(char *str, const char *format, ...);
char *snprintf(char *str, size_t size, const char *format, ...);
char *vsprintf(char *str, const char *format, va_list ap);
char *vsnprintf(char *str, size_t size, const char *format, va_list ap);

// eg:
char str[100];
sprintf(str, "Hello, %s!", "world");  // 将字符串 "Hello, world!" 格式化并存储在 str 中
printf("%s\n", str);  // 输出 "Hello, world!"

/*
 * 字符串统计
 */
size_t strspn(const char *s, const char *accept);  // 统计字符串 s 中出现的字符在 accept 字符串中的个数
size_t strcspn(const char *s, const char *reject);  // 统计字符串 s 中没有出现在 reject 字符串中的字符个数

```

## 字节操作
字节操作是指对单个字节进行操作，例如读取、写入、比较等。在C语言中，可以使用以下类型来表示字节：

- `unsigned char`：无符号字符类型，表示一个字节。
- `signed char`：有符号字符类型，表示一个字节。
- `uint8_t`：无符号8位整数类型，表示一个字节。

## 字节序

字节序是指多字节数据在内存中的存储顺序。常见的字节序有两种：

- 大端序（Big-Endian）：高位字节存储在低地址，低位字节存储在高地址。
- 小端序（Little-Endian）：低位字节存储在低地址，高位字节存储在高地址。

在C语言中，可以使用以下宏来检测和设置字节序：

```c
#include <stdint.h>

// 检测字节序
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
```

## 位运算

在C语言中，可以使用以下位运算符来进行位操作：

- `&`：按位与
- `|`：按位或
- `^`：按位异或
- `~`：按位取反
- `<<`：左移
- `>>`：右移


## 字节操作函数

在C语言中，可以使用以下函数来进行字节操作：
```c
/*
 * 将src指向的内存块复制到dest指向的内存块中，并返回dest指针
 */
void* memcpy(const void* dest, const void* src, size_t n);
void* memmove(void* dest, const void* src, size_t n);  // 与memcpy类似，但可以安全地处理重叠的内存块

/*
 * 将dest指向的内存块中的每个字节都设置为c，并返回dest指针
 */
void* memset(void* dest, int c, size_t n);

/*
 * 比较src1指向的内存块和src2指向的内存块的前n个字节，并返回它们的差值
 */
int memcmp(const void* src1, const void* src2, size_t n);

/*
 * 在src指向的内存块中查找c，并返回指向它的指针
 */
void* memchr(const void* src, int c, size_t n);

```
