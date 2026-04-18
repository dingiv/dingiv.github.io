# 数据结构辅助

在算法设计中，合理使用数据结构可以显著提高效率。这体现了"空间换时间"的思想：通过使用额外的空间来存储中间结果，避免重复计算，从而降低时间复杂度。更多数据结构参看[数据结构章节](../struct/)和[数据结构辅助](./struct)。

## 栈（Stack）
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

## 队列（Queue）
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

## 哈希表（Hash Table）
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

部分题目可能会要求节省空间，并且题目往往是处理一个整数类型的问题，可以考虑采用**原地哈希**的技巧，以节省空间。

## 堆（Heap）
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

## 树状数组（Fenwick Tree）
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

## 线段树（Segment Tree）
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

## 并查集（Disjoint Set）
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

## 字典树（Trie）
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
