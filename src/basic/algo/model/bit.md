---
title: 位运算
order: 4
---

# 位运算
位运算是一种直接对二进制位进行操作的运算方式，在算法中有着广泛的应用。通过位运算，我们可以实现一些高效的操作，同时节省内存空间。

## 基本位运算操作

### 1. 按位与（&）

**作用**：两个位都为1时，结果才为1

**应用**：
- 判断奇偶：`n & 1` 为1表示奇数，为0表示偶数
- 取最低位：`n & -n` 可以获取n的最低有效位
- 判断是否是2的幂：`n & (n-1) === 0`

```js
// 判断奇偶
function isOdd(n) {
  return (n & 1) === 1;
}

// 判断是否是2的幂
function isPowerOfTwo(n) {
  return n > 0 && (n & (n - 1)) === 0;
}
```

### 2. 按位或（|）

**作用**：两个位有一个为1时，结果就为1

**应用**：
- 设置特定位为1
- 合并两个集合

```js
// 设置特定位为1
function setBit(n, pos) {
  return n | (1 << pos);
}

// 合并两个集合
function mergeSets(set1, set2) {
  return set1 | set2;
}
```

### 3. 按位异或（^）

**作用**：两个位相同时为0，不同时为1

**应用**：
- 交换两个数：`a ^= b; b ^= a; a ^= b;`
- 找出只出现一次的数字
- 判断两个数是否异号

```js
// 交换两个数
function swap(a, b) {
  a ^= b;
  b ^= a;
  a ^= b;
  return [a, b];
}

// 找出只出现一次的数字
function singleNumber(nums) {
  let result = 0;
  for (const num of nums) {
    result ^= num;
  }
  return result;
}
```

### 4. 按位非（~）

**作用**：对每一位取反

**应用**：
- 求补码
- 配合其他位运算使用

```js
// 求补码
function complement(n) {
  return ~n + 1;
}
```

### 5. 左移（<<）

**作用**：将二进制位向左移动，低位补0

**应用**：
- 快速计算2的幂：`1 << n` 等于 2^n
- 创建掩码

```js
// 计算2的n次方
function powerOfTwo(n) {
  return 1 << n;
}

// 创建掩码
function createMask(start, length) {
  return ((1 << length) - 1) << start;
}
```

### 6. 右移（>>）

**作用**：将二进制位向右移动，高位补符号位

**应用**：
- 快速除以2：`n >> 1` 等于 n/2
- 提取特定位

```js
// 快速除以2
function divideByTwo(n) {
  return n >> 1;
}

// 提取特定位
function getBit(n, pos) {
  return (n >> pos) & 1;
}
```

## 位运算技巧

### 1. 位掩码（Bitmask）

位掩码是一种使用二进制位来表示状态的技术，常用于状态压缩。

```js
// 使用位掩码表示状态
const STATE_A = 1 << 0; // 0001
const STATE_B = 1 << 1; // 0010
const STATE_C = 1 << 2; // 0100
const STATE_D = 1 << 3; // 1000

// 设置状态
let state = 0;
state |= STATE_A; // 设置状态A
state |= STATE_C; // 设置状态C

// 检查状态
if (state & STATE_A) {
  // 状态A已设置
}

// 清除状态
state &= ~STATE_A; // 清除状态A
```

### 2. 集合操作

使用位运算可以高效地实现集合的基本操作。

```js
class BitSet {
  constructor(size) {
    this.bits = new Array(Math.ceil(size / 32)).fill(0);
  }
  
  // 添加元素
  add(n) {
    const index = Math.floor(n / 32);
    const pos = n % 32;
    this.bits[index] |= (1 << pos);
  }
  
  // 删除元素
  remove(n) {
    const index = Math.floor(n / 32);
    const pos = n % 32;
    this.bits[index] &= ~(1 << pos);
  }
  
  // 检查元素是否存在
  has(n) {
    const index = Math.floor(n / 32);
    const pos = n % 32;
    return (this.bits[index] & (1 << pos)) !== 0;
  }
}
```

### 3. 位计数

计算一个数的二进制表示中1的个数。

```js
// 方法1：逐位检查
function countBits1(n) {
  let count = 0;
  while (n) {
    count += n & 1;
    n >>= 1;
  }
  return count;
}

// 方法2：利用 n & (n-1) 清除最低位的1
function countBits2(n) {
  let count = 0;
  while (n) {
    n &= n - 1;
    count++;
  }
  return count;
}
```

### 4. 位运算优化

使用位运算可以优化一些常见的数学运算。

```js
// 快速判断是否是2的幂
function isPowerOfTwo(n) {
  return n > 0 && (n & (n - 1)) === 0;
}

// 快速计算绝对值
function abs(n) {
  const mask = n >> 31;
  return (n + mask) ^ mask;
}

// 快速计算两个数的平均值（防止溢出）
function average(a, b) {
  return (a & b) + ((a ^ b) >> 1);
}
```

## 应用场景

### 1. 状态压缩

在解决某些问题时，可以使用位运算来压缩状态，减少内存使用。

```js
// 使用位运算表示棋盘状态
class ChessBoard {
  constructor() {
    this.state = 0;
  }
  
  // 设置棋子
  setPiece(row, col) {
    const pos = row * 8 + col;
    this.state |= (1 << pos);
  }
  
  // 检查是否有棋子
  hasPiece(row, col) {
    const pos = row * 8 + col;
    return (this.state & (1 << pos)) !== 0;
  }
}
```

### 2. 子集生成

使用位运算可以高效地生成集合的所有子集。

```js
function generateSubsets(nums) {
  const n = nums.length;
  const subsets = [];
  
  for (let mask = 0; mask < (1 << n); mask++) {
    const subset = [];
    for (let i = 0; i < n; i++) {
      if (mask & (1 << i)) {
        subset.push(nums[i]);
      }
    }
    subsets.push(subset);
  }
  
  return subsets;
}
```

### 3. 位图算法

使用位运算实现高效的位图算法。

```js
class BitMap {
  constructor(size) {
    this.bits = new Array(Math.ceil(size / 32)).fill(0);
  }
  
  // 设置位
  set(n) {
    const index = Math.floor(n / 32);
    const pos = n % 32;
    this.bits[index] |= (1 << pos);
  }
  
  // 清除位
  clear(n) {
    const index = Math.floor(n / 32);
    const pos = n % 32;
    this.bits[index] &= ~(1 << pos);
  }
  
  // 检查位
  test(n) {
    const index = Math.floor(n / 32);
    const pos = n % 32;
    return (this.bits[index] & (1 << pos)) !== 0;
  }
}
```