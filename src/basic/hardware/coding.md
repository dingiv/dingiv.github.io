# 编码
编码是使用计算机处理数据的前提。数据需要通过计算机的编码方式进行**输入**，**处理**，**输出**，**存储**等。计算机使用二进制数字编码，数据通过转化为二进制数存储在计算机中，并通过二进制的逻辑门电路进行处理。

编码就是通过一张表将一个字符**映射**成为一个二进制数。其核心特点是：二进制、离散、有限。
+ 基于二进制的逻辑门电路进行处理；
+ 使用多个逻辑门电路，形成规模化，从而通过离散数学的理论来处理数据；
+ 规模化是有限的，不能无限制拓宽电路的逻辑门数量，多数计算机使用 32 位或者 64 位的规格来作为处理数据的**位宽**；

编码的本质是一种**协议**。如何理解协议？协议一种约定，一种承诺，一种共识，一种规范……一串数字本没有意义，但是，因为书写和阅读的人都使用了同一个协议，使得它完成了信息存储和传递的目的，它就像自然语言和文字一样，是信息的载体，是沟通的桥梁。其实自然语言也本没有意义，但是，使用语言的人们通过一种共识和协议，让语言和文字完成了承载信息的功能。

## 编码单位 Byte
字节是现代计算机处理数据的基本单位。一个字节代表 8 个 bit 位。为什么不是 4 个 bit，或者是 16 bit 为一个字节？字节作为计算机中数据处理和编码的标准单位，源于多个方面的综合考量。
+ 硬件设计：字节与存储和寻址的便利性，字节太小会导致地址空间膨胀，太大会导致存储空间的浪费；包括内存和磁盘，这些设备的存储单元均采用了 8 bit 的设计；
+ 历史遗留：字节是表示字符的理想单位。早期 ASCII 编码用 7 位表示字符（1 字节留 1 位校验），能表示 128 种字符（字母、数字、符号）；历史上的各种设备的位宽不同，但是可以使用 8 位作为一个基本公约数；

### 多字节数据编码
多字节数据编码用于表示超过一个字节的数据，如整数、浮点数等。主要涉及以下几个方面：

+ 字节序（Endianness），
   - 大端序（Big-Endian）：最高有效字节存储在最低地址
   - 小端序（Little-Endian）：最低有效字节存储在最低地址
   - 网络字节序：采用大端序，用于网络传输

2. 整数编码
   - 有符号整数：使用补码表示
   - 无符号整数：直接二进制表示
   - 常见位宽：8位、16位、32位、64位

3. 浮点数编码
   - IEEE 754标准
   - 单精度（32位）：1位符号位，8位指数，23位尾数
   - 双精度（64位）：1位符号位，11位指数，52位尾数

4. 对齐要求
   - 自然对齐：数据地址是其大小的整数倍
   - 内存访问效率：对齐可提高访问速度
   - 跨平台兼容：不同架构可能有不同对齐要求

## ASCII
现代计算机起源于英语世界，编码英文世界中的字符可以使用很小的比特数7进行编码，而为了最为广泛地使用，ASCII编码采用了8位（1字节）的存储方式。

ASCII 编码包含了
1. 基本ASCII（0-127）
   - 控制字符（0-31）：如换行、回车、制表符等
   - 可打印字符（32-126）：包括字母、数字、标点符号
   - 删除字符（127）：DEL
2. 扩展ASCII（128-255）
   - 不同国家/地区使用不同扩展
   - 包含特殊符号和图形字符
   - 兼容性问题导致使用受限

```
0-31: 控制字符
32-47: 空格和标点符号
48-57: 数字0-9
58-64: 标点符号
65-90: 大写字母A-Z
91-96: 标点符号
97-122: 小写字母a-z
123-126: 标点符号
127: 删除字符
```

### ASCII的局限性
1. 字符集限制
   - 仅支持英文字符
   - 无法表示其他语言
   - 特殊符号支持有限

2. 扩展问题
   - 不同扩展标准不兼容
   - 国际化支持困难
   - 多语言混排困难

3. 现代应用
   - 作为Unicode的基础
   - 在简单系统中仍广泛使用
   - 作为数据传输的基础编码

## 字符集

## 纯数据和指令
在内存中存放的数据本质上都是 01 二进制构建的字节流，但是我们会以不同的视角来看待这些数据，一个纯数据视角，一个是指令视角；纯数据视角使用前文提到的数据