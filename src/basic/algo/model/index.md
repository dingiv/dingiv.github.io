---
title: 算法模型
order: 2
---

# 算法模型

算法模型的分类往往基于一定的[**数据结构**](../struct/index)，面向一定的**算法场景**。

而数据结构的类型可以分为：

- 线性结构：数组、链表、栈、队列、堆（优先队列）等；
- 分支结构：各种树、跳表；
- 网络结构：各种图、矩阵；

对应不同的数据结构，有特定的算法模型进行解决。

## 遍历和查找

查找是最为基础和常见的算法模型，其核心在于**遍历**，并且基于一定的数据结构进行遍历。为此，我们需要理解几个重要的问题，即：

1. 遍历的范围，或者说搜索集合；

> 我们需要明确题目的入参和出参，从而确定遍历的范围，或者说**搜索集合**。**入参集合**指的是题目所给的数据结构和数据，**出参集合**指的是由所有可能是答案的候选数据所构成的集合。一些题目中，入参集合和出参集合相同，我们直接以此作为搜索集合，进行遍历即可；如果入参集合和出参集合不同，那么可以先遍历入参集合，构建出出参集合，并以出参集合作为搜索集合，再遍历。亦或者，也可以不使用入参或者出参集合作为搜索集合，基于间接计算算出搜索目标进行验证。总之，出参集合往往作为搜索集合，但不是唯一的选择。

2. 目标的特征，或者说搜索目标；
3. 遍历的方式，即遍历策略；

- 枚举

  枚举法即遍历所有可能的解，一一校验，从而查找答案。枚举法实现简单，时间复杂度一般较高。具体地，枚举法在出参集合中枚举所有可能的备选解，从备选解中找到满足条件的解，或者从入参集合中出发，重复进行计算并寻找所有满足条件的解。枚举是所有查找算法问题的思想基础，后续的算法可以基于枚举法进行优化。枚举的关键就是确定**枚举的范围**或者**搜索集合**。

  从枚举方式出发，枚举法可以基于**循环**实现，也可以基于[**递归**](./recrusion)实现。循环的一般基于线性结构，入参为循环长度 n，递归一般基于树状结构，入参为递归深度 depth 和 分支数 branch。

  题目中关键词：`查找`；

  ```js
  // 一维迭代遍历
  for (let i = 0; i < n; i++) {
    // 遍历出参集合
  }

  // 二维迭代遍历
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      // 遍历出参集合
    }
  }

  // 递归遍历
  function dfs(depth, arg) {
    // 递归终止条件
    if (depth === target) {
      return; // 找到答案
    }
    for (let i = 0; i < m; i++) {
      dfs(depth + 1, arg + i);
    }
    return; // 答案
  }
  ```

- 双指针

  双指针遍历，基于**同时管理两个指针**的遍历方式，可以减少遍历次数，从而降低时间复杂度。双指针遍历可以基于数组、链表、字符串等线性结构，也可以基于哈希表等非线性结构。双指针具体在操作的时候可以有很多不同的方式，例如左右指针、滑动窗口、快慢指针等。其主要特征是题目中具有两个参数。双指针遍历的使用场景往往需要我们找到两个参数之间的关系，从而确定遍历策略。

  题目中关键词：`子数组、两个数组、区间`等等；

  ```js
  let left = 0,
    right = 0;
  while (left < right) {
    // 遍历出参集合
    if (condition) {
      // 处理边界情况
      left++;
    } else {
      // 处理边界情况
      right--;
    }
  }

  for (let left = 0; left < n; left++) {
    let right = left + width;
    // 处理定长窗口
  }

  for (let left = 0, right = 0; left < n && right < m; left++, right++) {
    // 处理双序列
  }
  ```

- 二分法

  二分法是一种**在有序集合中查找目标值**的方法，其时间复杂度为 O(logn)。二分法的基本思想是将集合分成两部分，然后根据目标值的大小，选择其中一部分继续查找，直到找到目标值或者查找范围为空。二分法可以用于实现快速查找，例如查找有序数组中的目标值、查找有序数组中的第 k 小的元素等等。

  题目中关键词：`有序、查找、第 k 小的元素、最大化最小值、最小化最大值、单调`等等；

  ```js
  // 开区间二分遍历
  let left = -1,
    right = n;
  while (left < right) {
    let mid = Math.floor((left + right) / 2);
    if (condition) {
      // 处理边界情况
      left = mid;
    } else {
      // 处理边界情况
      right = mid;
    }
  }

  // 闭区间二分遍历
  let left = 0,
    right = n - 1;
  while (left <= right) {
    let mid = Math.floor((left + right) / 2);
    if (condition) {
      // 处理边界情况
      left = mid + 1;
    } else {
      // 处理边界情况
      right = mid - 1;
    }
  }
  ```

- 贪心法

  贪心法是一种遍历策略，在遍历时，有导向地选择在当前状态下最好或最优的选择进行优先遍历，忽略某些路径，跳过不必要的遍历，并且依然能够找到目标或最优结果的算法。贪心法需要我们提前知道：局部选择是否能导致全局最优。这往往要求题目具有一定的数学性质，例如最优子结构、无后效性、单调性等。

  题目中关键词：`最优、单调性`等等；

  ```js
  for(let i = 0; i < n; i++) {
    if (condition) {
      // 跳过特殊情况
      continue
    }
    for(let j = 0; j < m; j++) {
      // 遍历入参集合
    }
  }


  function dfs(depth, arg) {
    // 递归终止条件
    if (depth === target) {
      return // 找到答案
    }
    if (condition) {
      // 剪枝，跳过特殊情况
      continue
    }
    for (let i = 0; i < m; i++) {
      if (condition) {
        // 剪枝，跳过特殊情况
        continue
      }
      dfs(depth + 1, arg + i);
    }
  }
  ```

- 多趟遍历

  多趟遍历是一种遍历策略，在遍历时，将问题分解为多个子问题，然后依次解决这些子问题，从而得到最终的结果。多趟遍历可以用于解决一些复杂的问题，如果一个问题一次性解决需要较大的时间复杂度，那么就分成多次时间复杂度较低的遍历进行解决；或者一次遍历不能够得到最终结果，那么就分成多次遍历，每次遍历都得到一部分结果，最终得到最终结果。

  题目中关键词：`多趟、多次遍历、多次计算`等等；

  ```js
  const tmp = [];
  for (let i = 0; i < n; i++) {
    // 遍历入参集合
    // 计算中间结果 tmp
  }

  const result = [];
  for (let i = 0; i < n; i++) {
    // 遍历入参集合
  }
  ```

## 数据结构辅助

使用特殊地数据结构保存计算中的中间结果，从而减少重复计算和获取数据的时间，降低时间复杂度，但是这往往会引入额外的空间复杂度。这给我们带来一个启示，就是**空间换时间**，不要重复地去遍历，而是尽量在一次遍历中获知更多的信息，使用合理的数据结构来记录这些信息，然后提高查找速度，从而加快遍历。往往可以使用多趟遍历，先遍历入参，构建好数据结构，再遍历出参，计算出结果。

- 栈

  接口：（1）入栈，将元素从栈顶压入；（2）出栈，将元素将栈顶弹出；

  栈的特点是先进后出，后进先出，可以用于解决一些需要**回溯**的问题，例如括号匹配、表达式求值等等。栈的时间复杂度为 O(1)，空间复杂度为 O(n)。有一类特殊的需求和单调性有关。

- 队列

  接口：（1）入队，将元素添加到队列末尾排队；（2）出队，将队列首个元素移出队列；

  队列的特点是先进先出，后进后出，可以用于解决一些需要**顺序处理**的问题，例如广度优先搜索、生产者消费者模型等等。队列的时间复杂度为 O(1)，空间复杂度为 O(n)。

- 哈希

  接口：（1）插入；（2）查找；（3）删除；（4）调整，更改哈希槽的数量；

  哈希表的特点是**查找速度快**，时间复杂度为 O(1)，空间复杂度为 O(n)。哈希表可以用于解决一些需要快速查找的问题，例如查找重复元素、查找最长无重复子串等等。

- 堆

  接口：（1）插入；（2）弹出堆顶元素；

  堆的特点是**堆顶元素为最大值或最小值**，可以用于解决一些需要快速获取最大值或最小值的问题，例如堆排序、优先队列等等。堆的时间复杂度为 O(logn)，空间复杂度为 O(n)。有一类特殊的需求和单调性有关。

- 树状数组

  接口：（1）单点更新；（2）区间查询；

  树状数组是一种可以高效地进行**单点更新和区间和查询**的数据结构，两种操作的时间复杂度为 O(logn)，可以用于解决一些需要频繁进行区间和查询的问题，例如前缀和、逆序对等等。

  树状数组由两个数组构成，一个是数据数组，一个是树状数组。树状数组中的每个元素表示数据数组中，一段连续区间的和。树状数组可以通过二进制表示法来计算区间的和，从而实现高效的查询和更新。

- 线段树

  接口：（1）单点更新；（2）区间查询；（3）插入元素；（4）删除元素；

  线段树是一种可以高效地进行单点更新和区间查询的数据结构，时间复杂度为 O(logn)，可以用于解决一些需要频繁进行区间查询的问题，例如区间和、区间最大值、区间最小值等等。

  线段树由两个数组构成，一个是数据数组，一个是线段树数组。线段树数组中的每个元素表示数据数组中，一段连续区间的和、最大值、最小值等等。线段树可以通过二分法来计算区间的和、最大值、最小值等等，从而实现高效的查询和更新。

- 并查集

  接口：（1）合并；（2）关系判定；（3）路径压缩；（4）按秩合并；

  并查集是一种可以高效地进行集合合并和关系判定的数据结构，时间复杂度为 O(logn)，可以用于解决一些需要频繁进行合并和查找的问题，例如连通分量、最小生成树等等。

  并查集由两个数组组成，一个是数据数组，一个是父节点数组。父节点数组中的每个元素表示该下标所对应的元素的父元素的下标，一个节点只能有一个父节点，如果需要将一个元素加入到另一个元素集合中，只需要将该元素的父节点设置为该元素集合的根节点即可。

- 字典树

  接口：（1）前缀查找；（2）插入；（3）删除；

  字典树是一种可以高效地进行字符串匹配的数据结构，时间复杂度为 O(m)，可以用于解决一些需要频繁进行字符串匹配的问题，例如最长公共前缀、字符串匹配等等。

## 数学技巧辅助

我们可以使用数学技巧来优化算法，例如使用数学公式、数学定理、数学性质等等。这些技巧可以帮助我们找到更优的算法解决方案，从而降低时间复杂度和空间复杂度。

- 前缀和与差分

  前缀和是指一个数组的前 n 个元素的和。前缀和可以用于快速计算一个数组的子数组和，时间复杂度为 O(1)。前缀和可以用于解决一些需要频繁计算子数组和的问题，例如最大子数组和、子数组和等于 k 的子数组等等。其实，也不一定是和，也可以是最大值、最小值、个数等等基于一个子数组的统计信息，均可以使用前缀和来优化。前缀和将引入**一个额外的数组**来保存中间结果，从而降低时间复杂度。

  ```
  对于数组 a，定义前缀和数组 s，其中 s[i] 表示 a[0] 到 a[i-1] 的和，即：
  s[i] = a[0] + a[1] + a[2] + ... + a[i-1]
  sum(a, b) = s[b+1] - s[a]
  ```

  题目中关键词：子数组、和、最大值、最小值、个数、区间等等；

  差分是指一个数组的每两个数之间的差值所产生的序列，可以理解为前缀和的逆运算。差分数组可以快速地对一个区间内的数值进行区间计算，时间复杂度为 O(1)。差分数组可以用于解决一些需要频繁对区间进行加减操作的问题，例如区间求和、区间修改等等。其实，也不一定是和，也可以是最大值、最小值、个数等等基于一个子数组的统计信息，均可以使用差分数组来优化。差分数组将引入**一个额外的数组**来保存中间结果，从而降低时间复杂度。

  题目中关键词：区间、和、最大值、最小值、个数、加减、修改等等；

  ```
  对于数组 a，定义差分数组 d，其中 d[i] 表示 a[i] 和 a[i-1] 的差值，即：
  d[0] = a[0]
  d[i] = a[i] - a[i-1] (i > 0)

  对 [a, b] 区间内的数值进行加减操作，只需要对差分数组进行修改：
  d[a] += val
  d[b+1] -= val
  ```

- dp 动态规划

  动态规划是一种通过将问题分解为子问题，并将子问题的解存储起来，从而避免重复计算的方法，是典型空间换时间的思想，同时，动态规划刻意地在题目中构造一个数列问题，将题目转化为寻找数列的递推公式的问题，或者说是寻找状态转移方程的问题。动态规划通常用于解决一些具有重叠子问题和最优子结构的问题，例如背包问题、最长公共子序列问题、最长递增子序列问题等等。

  题目中关键词：最优等；（含有重叠子问题）
  具体内容参看 [动态规划](./dp)

- 位运算

  使用位运算可以模拟集合操作，例如集合的并集、交集、差集等等。位运算的时间复杂度为 O(1)，可以用于解决一些需要频繁进行集合操作的问题，例如集合的并集、交集、差集等等，同时降低算法的空间复杂度。但是，位运算需要我们提前知道集合中的元素，并且元素的范围不能太大，否则会超出计算机的表示范围。

- 数学公式

  使用数学公式可以简化算法，一些问题具有鲜明的数学性质，通过归纳数学公式来直接计算答案。数学公式的时间复杂度为 O(1)，可以用于解决一些需要频繁进行计算的问题，例如阶乘、斐波那契数列等等，同时降低算法的空间复杂度。一旦一个问题可以通过推导数学公式来解决，那么该问题的难点就变成了数学推导，而对算法实现的难度大大降低。


## 排序

## 字符串

字符串算法是非常常见的算法问题，包括字符串的遍历、查找、替换、匹配、排序等等，主要是**查找和匹配**。字符串算法的时间复杂度通常较高，需要使用一些特定的技巧来优化。

### 查找

字符串的查找主要目的是在长字符串中找到指定的子串，或者找到满足特定条件的子串。常见的查找算法有：

- 朴素匹配。暴力迭代，比较长字符串中的每个位置处是否能够匹配子串。
- KMP。用于在一个长字符串中寻找一个指定的字符串子串。
- BMH。
- Trie 字典树
- AC 自动机
- RK

* KMP
* Z 算法
* Manacher
*

### 匹配

- 自动机
- Trie

## 图

- 拓扑排序
- 最短路径
- 最小生成树
- 二分图
- 连通分量
- 基环树
- 欧拉回路

## 加密与解密

- 朴素加密。如：凯撒、base64
- 对称加密。AES
- 非对称加密。RSA
- 哈希加密。如：MD5、SHA1