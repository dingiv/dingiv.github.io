---
title: 算法模型
order: 2
---

# 算法模型
算法模型的分类往往基于一定的[**数据结构**](../struct/index)，面向一定的**算法场景**。而数据结构的类型可以分为：
- 线性结构：数组、链表、栈、队列、堆（优先队列）等；
- 分支结构：各种树、跳表；
- 网络结构：各种图、矩阵；
对应不同的数据结构，有特定的算法模型进行解决。

## 遍历和查找
查找是最为基础和常见的算法模型，其核心在于**遍历**，并且基于一定的数据结构进行遍历。为此，需要先理解几个重要的问题，即：
1. 遍历的范围，或者说搜索集合；
   > 我们需要明确题目的入参和出参，从而确定遍历的范围，或者说**搜索集合**。**入参集合**指的是题目所给的数据结构和数据，**出参集合**指的是由所有可能是答案的候选数据所构成的集合。一些，入参集合和出参集合相同，我们直接以此作为搜索集合，进行遍历即可；如果入参集合和出参集合不同，那么可以先遍历入参集合，构建出出参集合，并以出参集合作为搜索集合，再遍历。亦或者，也可以不使用入参或者出参集合作为搜索集合，基于间接计算算出搜索目标进行验证。总之，出参集合往往作为搜索集合，但不是唯一的选择。
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
  多趟遍历是一种遍历策略，在遍历时，将问题分解为多个子问题，然后依次解决这些子问题，从而得到最终的结果。多趟遍历可以用于解决一些复杂的问题，如果一个问题一次性解决需要较大的时间复杂度，那么就分成多次时间复杂度较低的遍历进行解决；或者一次遍历不能够得到最终结果，那么就分成多次遍历，每次遍历都得到一部分结果，然后导出最终结果。

  题目中关键词：`多趟、多次遍历、多次计算`等等；

  ```js
  const tmp = [];
  for (let i = 0; i < n; i++) {
    // 遍历入参集合
    // 计算中间结果 tmp
  }

  const result = [];
  for (let i = 0; i < n; i++) {
    // 遍历出参集合
  }
  ```

## 数据结构辅助

在算法设计中，合理使用数据结构可以显著提高效率。这体现了"空间换时间"的思想：通过使用额外的空间来存储中间结果，避免重复计算，从而降低时间复杂度。更多数据结构参看[数据结构章节](../struct/index)。

### 栈（Stack）
特点：后进先出（LIFO）
- `push`: 入栈，时间复杂度 O(1)
- `pop`: 出栈，时间复杂度 O(1)
- `peek`: 查看栈顶元素，时间复杂度 O(1)

**应用场景**：
- 括号匹配
- 表达式求值
- 函数调用栈
- 浏览器历史记录
- 撤销操作

```js
class Stack {
  constructor() {
    this.items = [];
  }

  push(item) {
    this.items.push(item);
  }

  pop() {
    return this.items.pop();
  }

  peek() {
    return this.items[this.items.length - 1];
  }
}
```

### 队列（Queue）
特点：先进先出（FIFO）
- `enqueue`: 入队，时间复杂度 O(1)
- `dequeue`: 出队，时间复杂度 O(1)
- `front`: 查看队首元素，时间复杂度 O(1)

**应用场景**：
- 广度优先搜索
- 任务调度
- 消息队列
- 缓存实现
- 打印队列

```js
class Queue {
  constructor() {
    this.items = [];
  }

  enqueue(item) {
    this.items.push(item);
  }

  dequeue() {
    return this.items.shift();
  }

  front() {
    return this.items[0];
  }
}
```

### 哈希表（Hash Table）
特点：快速查找，平均时间复杂度 O(1)
- `set`: 插入键值对，平均时间复杂度 O(1)
- `get`: 获取值，平均时间复杂度 O(1)
- `delete`: 删除键值对，平均时间复杂度 O(1)
- `has`: 检查键是否存在，平均时间复杂度 O(1)

**应用场景**：
- 缓存实现
- 字典实现
- 频率统计
- 去重操作
- 快速查找

```js
class HashTable {
  constructor() {
    this.table = new Map();
  }

  set(key, value) {
    this.table.set(key, value);
  }

  get(key) {
    return this.table.get(key);
  }

  delete(key) {
    return this.table.delete(key);
  }

  has(key) {
    return this.table.has(key);
  }
}
```

### 堆（Heap）
特点：快速获取最大/最小值
- `insert`: 插入元素，时间复杂度 O(log n)
- `extract`: 提取最大/最小值，时间复杂度 O(log n)
- `peek`: 查看堆顶元素，时间复杂度 O(1)

**应用场景**：
- 优先队列
- 堆排序
- 任务调度
- 合并K个有序数组
- 中位数查找

```js
class MinHeap {
  constructor() {
    this.heap = [];
  }

  insert(value) {
    this.heap.push(value);
    this.bubbleUp(this.heap.length - 1);
  }

  extractMin() {
    const min = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.bubbleDown(0);
    }
    return min;
  }

  peek() {
    return this.heap[0];
  }

  // 辅助方法
  bubbleUp(index) {
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2);
      if (this.heap[parentIndex] <= this.heap[index]) break;
      [this.heap[parentIndex], this.heap[index]] = [this.heap[index], this.heap[parentIndex]];
      index = parentIndex;
    }
  }

  bubbleDown(index) {
    const lastIndex = this.heap.length - 1;
    while (true) {
      const leftChildIndex = 2 * index + 1;
      const rightChildIndex = 2 * index + 2;
      let smallestIndex = index;

      if (leftChildIndex <= lastIndex && this.heap[leftChildIndex] < this.heap[smallestIndex]) {
        smallestIndex = leftChildIndex;
      }

      if (rightChildIndex <= lastIndex && this.heap[rightChildIndex] < this.heap[smallestIndex]) {
        smallestIndex = rightChildIndex;
      }

      if (smallestIndex === index) break;

      [this.heap[index], this.heap[smallestIndex]] = [this.heap[smallestIndex], this.heap[index]];
      index = smallestIndex;
    }
  }
}
```

### 树状数组（Fenwick Tree）
特点：高效的单点更新和区间查询
- `update`: 单点更新，时间复杂度 O(log n)
- `query`: 区间查询，时间复杂度 O(log n)

**应用场景**：
- 动态前缀和
- 逆序对统计
- 区间和查询
- 频率统计

```js
class FenwickTree {
  constructor(size) {
    this.size = size;
    this.tree = new Array(size + 1).fill(0);
  }

  update(index, delta) {
    while (index <= this.size) {
      this.tree[index] += delta;
      index += index & -index;
    }
  }

  query(index) {
    let sum = 0;
    while (index > 0) {
      sum += this.tree[index];
      index -= index & -index;
    }
    return sum;
  }

  rangeQuery(left, right) {
    return this.query(right) - this.query(left - 1);
  }
}
```

### 线段树（Segment Tree）
特点：支持区间查询和更新
- `update`: 单点/区间更新，时间复杂度 O(log n)
- `query`: 区间查询，时间复杂度 O(log n)
- `build`: 构建线段树，时间复杂度 O(n)

**应用场景**：
- 区间最大值/最小值
- 区间和
- 区间统计
- 动态区间查询

```js
class SegmentTree {
  constructor(data) {
    this.n = data.length;
    this.tree = new Array(4 * this.n);
    this.build(1, 0, this.n - 1, data);
  }

  build(node, start, end, data) {
    if (start === end) {
      this.tree[node] = data[start];
      return;
    }

    const mid = Math.floor((start + end) / 2);
    this.build(2 * node, start, mid, data);
    this.build(2 * node + 1, mid + 1, end, data);
    this.tree[node] = this.tree[2 * node] + this.tree[2 * node + 1];
  }

  update(index, value) {
    this.updateHelper(1, 0, this.n - 1, index, value);
  }

  updateHelper(node, start, end, index, value) {
    if (start === end) {
      this.tree[node] = value;
      return;
    }

    const mid = Math.floor((start + end) / 2);
    if (index <= mid) {
      this.updateHelper(2 * node, start, mid, index, value);
    } else {
      this.updateHelper(2 * node + 1, mid + 1, end, index, value);
    }
    this.tree[node] = this.tree[2 * node] + this.tree[2 * node + 1];
  }

  query(left, right) {
    return this.queryHelper(1, 0, this.n - 1, left, right);
  }

  queryHelper(node, start, end, left, right) {
    if (right < start || left > end) return 0;
    if (left <= start && end <= right) return this.tree[node];

    const mid = Math.floor((start + end) / 2);
    return this.queryHelper(2 * node, start, mid, left, right) +
           this.queryHelper(2 * node + 1, mid + 1, end, left, right);
  }
}
```

### 并查集（Disjoint Set）
特点：高效的集合合并和查询
- `find`: 查找根节点，时间复杂度 O(α(n))
- `union`: 合并集合，时间复杂度 O(α(n))
- `isConnected`: 判断连通性，时间复杂度 O(α(n))

**应用场景**：
- 连通分量
- 最小生成树
- 社交网络分析
- 图的连通性

```js
class DisjointSet {
  constructor(size) {
    this.parent = new Array(size);
    this.rank = new Array(size);
    for (let i = 0; i < size; i++) {
      this.parent[i] = i;
      this.rank[i] = 1;
    }
  }

  find(x) {
    if (this.parent[x] !== x) {
      this.parent[x] = this.find(this.parent[x]); // 路径压缩
    }
    return this.parent[x];
  }

  union(x, y) {
    const rootX = this.find(x);
    const rootY = this.find(y);

    if (rootX === rootY) return;

    // 按秩合并
    if (this.rank[rootX] > this.rank[rootY]) {
      this.parent[rootY] = rootX;
    } else if (this.rank[rootX] < this.rank[rootY]) {
      this.parent[rootX] = rootY;
    } else {
      this.parent[rootY] = rootX;
      this.rank[rootX]++;
    }
  }

  isConnected(x, y) {
    return this.find(x) === this.find(y);
  }
}
```

### 字典树（Trie）
特点：高效的字符串前缀匹配
- `insert`: 插入字符串，时间复杂度 O(m)
- `search`: 查找字符串，时间复杂度 O(m)
- `startsWith`: 查找前缀，时间复杂度 O(m)

**应用场景**：
- 自动补全
- 拼写检查
- 字符串匹配
- 词频统计

```js
class Trie {
  constructor() {
    this.root = {};
  }

  insert(word) {
    let node = this.root;
    for (const char of word) {
      if (!node[char]) {
        node[char] = {};
      }
      node = node[char];
    }
    node.isEnd = true;
  }

  search(word) {
    let node = this.root;
    for (const char of word) {
      if (!node[char]) return false;
      node = node[char];
    }
    return node.isEnd === true;
  }

  startsWith(prefix) {
    let node = this.root;
    for (const char of prefix) {
      if (!node[char]) return false;
      node = node[char];
    }
    return true;
  }
}
```

## 数学技巧辅助
我们可以使用数学技巧来优化算法，例如使用数学公式、数学定理、数学性质等等。这些技巧可以帮助我们找到更优的算法解决方案，从而降低时间复杂度和空间复杂度。

### 前缀和与差分
前缀和是指一个数组的前 n 个元素的和。前缀和可以用于快速计算一个数组的子数组和，时间复杂度为 O(1)。前缀和可以用于解决一些需要频繁计算子数组和的问题，例如最大子数组和、子数组和等于 k 的子数组等等。其实，也不一定是和，也可以是最大值、最小值、个数等等基于一个子数组的统计信息，均可以使用前缀和来优化。前缀和将引入**一个额外的数组**来保存中间结果，从而降低时间复杂度。

```
对于数组 a，定义前缀和数组 s，其中 s[i] 表示 a[0] 到 a[i-1] 的和，即：
s[i] = a[0] + a[1] + a[2] + ... + a[i-1]
sum(a, b) = s[b+1] - s[a]
```

关键词：子数组、和、最大值、最小值、个数、区间等等；

差分是指一个数组的每两个数之间的差值所产生的序列，可以理解为前缀和的逆运算。差分数组可以快速地对一个区间内的数值进行区间计算，时间复杂度为 O(1)。差分数组可以用于解决一些需要频繁对区间进行加减操作的问题，例如区间求和、区间修改等等。其实，也不一定是和，也可以是最大值、最小值、个数等等基于一个子数组的统计信息，均可以使用差分数组来优化。差分数组将引入**一个额外的数组**来保存中间结果，从而降低时间复杂度。

关键词：区间、和、最大值、最小值、个数、加减、修改等等；

```
对于数组 a，定义差分数组 d，其中 d[i] 表示 a[i] 和 a[i-1] 的差值，即：
d[0] = a[0]
d[i] = a[i] - a[i-1] (i > 0)

对 [a, b] 区间内的数值进行加减操作，只需要对差分数组进行修改：
d[a] += val
d[b+1] -= val
```

### dp 动态规划
动态规划是一种通过将问题分解为子问题，并将子问题的解存储起来，从而避免重复计算的方法，是典型**空间换时间**的思想，同时，动态规划刻意地在构造一个数列问题，将题目转化为寻找数列的递推公式的问题，或者说是寻找状态转移方程的问题。动态规划通常用于解决一些具有重叠子问题和最优子结构的问题，例如背包问题、最长公共子序列问题、最长递增子序列问题等等。

关键词：最优等；（含有重叠子问题），具体内容参看 [动态规划](./dp)。

### 位运算
使用位运算可以模拟集合操作，例如集合的并集、交集、差集等等。位运算的时间复杂度为 O(1)，可以用于解决一些需要频繁进行集合操作的问题，例如集合的并集、交集、差集等等，同时降低算法的空间复杂度。但是，位运算需要我们提前知道集合中的元素，并且元素的范围不能太大，否则会超出计算机的表示范围。具体参考[位运算](./bit)

### 数学公式
使用数学公式可以简化算法，一些问题具有鲜明的数学性质，通过归纳数学公式来直接计算答案。数学公式的时间复杂度为 O(1)，可以用于解决一些需要频繁进行计算的问题，例如阶乘、斐波那契数列等等，同时降低算法的空间复杂度。一旦一个问题可以通过推导数学公式来解决，那么该问题的难点就变成了数学推导，而对算法实现的难度大大降低。

## 排序
排序算法是计算机科学中最基础且重要的算法之一。根据不同的应用场景和需求，我们可以选择不同的[排序算法](../sort/index)。

### 排序算法的选择
选择排序算法时，需要考虑以下因素：
1. 数据规模：小规模数据可以使用简单排序，大规模数据需要使用高效排序
2. 数据分布：如果数据分布有特点，可以使用特定排序算法
3. 稳定性：如果需要保持相等元素的相对顺序，选择稳定排序
4. 空间限制：如果空间有限，选择原地排序
5. 数据特点：如果数据基本有序，插入排序可能更高效

### 排序算法的应用
1. 数据预处理：排序是许多算法的基础步骤
2. 查找优化：排序后的数据可以使用二分查找
3. 去重：排序后可以方便地去除重复元素
4. 统计：排序后可以方便地进行各种统计
5. 数据展示：排序后的数据更易于展示和理解

## 字符串
字符串算法是计算机科学中最常见和实用的算法之一，涉及对文本数据的处理和分析。字符串处理在日常开发中有着广泛的应用，从简单的文本搜索到复杂的自然语言处理。
### 字符串算法的主要类别
1. **搜索与匹配**：在文本中查找模式串
   - 暴力匹配（Brute Force）
   - KMP算法（Knuth-Morris-Pratt）
   - Boyer-Moore算法
   - Rabin-Karp算法
   - 后缀树/后缀数组
2. **编辑与比较**：计算字符串间的相似度或距离
   - 最长公共子序列（LCS）
   - 最长公共子串
   - 编辑距离（Levenshtein距离）
   - 字符串对齐
3. **压缩与编码**：减少字符串存储空间
   - Huffman编码
   - 游程编码（Run-length encoding）
   - LZ77/LZ78压缩
4. **文本分析**：从文本中提取信息
   - 正则表达式
   - 词频统计
   - 文本分类

### 常见的字符串数据结构
- **Trie树（前缀树）**：用于高效存储和查找字符串集合
- **后缀树/后缀数组**：用于复杂的字符串匹配问题
- **Bloom过滤器**：用于快速判断一个元素是否在集合中

字符串算法的详细实现和应用可以参考[字符串算法专题](../string/index)。

## 加密与解密
加密和解密是信息安全领域的基础技术，用于保护数据的机密性、完整性和可用性。根据加密方式的不同，可以分为以下几类：
1. **朴素加密**：基于简单的替换或移位操作
   - 凯撒密码：字母表移位
   - Base64：二进制数据编码
   - 其他古典密码
2. **对称加密**：使用相同的密钥进行加密和解密
   - AES（高级加密标准）：最常用的对称加密算法
   - DES（数据加密标准）：较老的对称加密算法
   - 3DES：DES的三重加密版本
3. **非对称加密**：使用不同的密钥进行加密和解密
   - RSA：最广泛使用的非对称加密算法
   - ECC（椭圆曲线加密）：更高效的现代非对称加密
   - DSA（数字签名算法）：主要用于数字签名
4. **哈希加密**：单向加密，不可逆
   - MD5：较老的哈希算法，已不推荐用于安全场景
   - SHA系列：包括SHA-1、SHA-256、SHA-512等
   - HMAC：基于哈希的消息认证码

加密算法的选择需要考虑以下因素：
1. 安全性：算法的抗攻击能力
2. 性能：加密/解密的速度和资源消耗
3. 密钥管理：密钥的生成、存储和分发
4. 应用场景：不同的场景需要不同的加密方案

详细内容请参考[加密与解密专题](../encode/index)。
