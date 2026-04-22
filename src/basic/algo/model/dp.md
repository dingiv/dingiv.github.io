---
title: 动态规划
order: 2
---

# 动态规划
动态规划是一种以空间换时间的算法思想，其本质是利用数学中**递推公式**的思想来解决存在大量重复计算的问题。当一个问题可以递归地分解为若干子问题，且这些子问题存在大量重叠时，动态规划通过缓存中间结果来避免重复计算，从而将指数级的时间复杂度优化为多项式级。

动态规划特别适合求解最值问题（最大值、最小值、方案数等），这类问题通常具备三个核心特征：
- **最优子结构**：问题的最优解包含子问题的最优解，大问题的答案可以从小问题的答案推导出来。
- 重叠子问题：递归求解过程中，相同的子问题会被反复计算，这是动态规划优于普通递归的关键所在。
- 无后效性：一旦某个状态确定之后，后续的决策不会影响到该状态之前的决策结果，这意味着我们可以放心地用子问题的结果来推导更大的问题，而不需要回头修改。

从复杂度的角度看，动态规划可以将 $O(n^k)$ 以上的暴力搜索优化到 $O(n^k)$，其中 $k$ 是问题的维度（状态变量的个数）。常见的一阶和二阶 DP 问题只需要一维或二维数组作为状态表，分别将高于 $O(n)$ 和 $O(n^2)$ 复杂度的问题优化到对应的多项式级别。
## 动态规划解题步骤
解决动态规划问题的核心难点在于两件事：定义状态和推导状态转移方程。其中状态定义是基础，状态定义错了，转移方程就无从谈起。

1. **定义状态**：确定 `dp[i]` 或 `dfs(i)` 表示什么含义。状态定义需要满足两个条件：能描述问题的解，且能从更小的状态推导而来。状态变量的选择直接决定了问题的维度和复杂度。

   常见的状态定义模式包括：
   - 以位置 $i$ 为结尾：`dp[i]` 表示前 $i$ 个元素的某个最优值（如最大子数组和）
   - 以区间 $[i, j]$ 为范围：`dp[i][j]` 表示区间内的最优值（如最长回文子串）
   - 背包类：`dp[i][w]` 表示前 $i$ 个物品、容量为 $w$ 时的最优值
   - 状态机：用多个 dp 数组表示不同状态之间的转换（如股票买卖中的持有/不持有）

2. **初始化状态**：设置边界条件，这是 DP 的启动点。初始化的值必须与状态定义严格一致，否则后续所有结果都会出错。通常初始化的是最小子问题的解，比如 $dp[0]$、$dp[1]$ 或 $dp[0][0]$ 等。

3. **推导状态转移方程**：分析当前状态可以从哪些更小的状态转移而来，考虑所有可能的转移路径，取最优值（最值问题）或累加（计数问题）。这是 DP 的核心，也是最难的一步。

4. **确定遍历顺序和返回值**：遍历顺序必须保证在计算当前状态时，它依赖的所有前置状态已经计算完毕。返回值通常是 dp 数组中的某个特定元素或遍历过程中的极值。

5. **空间优化（可选）**：如果状态转移只依赖有限个前驱状态，可以用滚动变量代替整个数组，将空间复杂度从 $O(n)$ 降到 $O(1)$。

```javascript
function dynamicProgramming(n) {
  // 1. 定义状态数组
  let dp = new Array(n).fill(0);

  // 2. 初始化初始状态和边界条件
  dp[0] = initial_value;

  // 3. 根据状态转移方程计算每个状态
  for (let i = 1; i <= n; i++) {
    dp[i] = dpTransition(dp, i);
  }

  // 4. 返回最终结果
  return getResult(dp);
}

// 状态转移函数，根据具体问题定义
function dpTransition(dp, i) {
  return some_function_of(dp, i);
  // 示例：dp[i] = min(dp[i-1] + cost1, dp[i-2] + cost2);
}

function getResult(dp) {
   return dp[dp.length - 1]
}
```
## 自底向上和自顶向下
动态规划有两种等价的实现方式：dp 数组（自底向上）和记忆化搜索 dfs（自顶向下）。

两者的核心逻辑完全等价——dp 数组的维度 index 与 dfs 函数的参数一一对应，转移方程的形式也类似：

```
// 斐波那契
dp[i] = dp[i-1] + dp[i-2];
dfs(i) = dfs(i-1) + dfs(i-2);
```

选择哪种方式取决于问题的自然思考方向。如果问题的递推关系天然是从 base case 向上构建的（如斐波那契、背包问题），dp 数组更直观；如果问题的思考方向是从最终状态向下分解的（如树形 DP、某些博弈问题），dfs 记忆化搜索更容易理解和编写。

需要注意的是，记忆化搜索通过函数调用栈隐式地处理了遍历顺序的问题，而 dp 数组需要手动确定正确的遍历顺序——这在多维 DP 中是一个常见的出错点。此外，dfs 只会计算实际需要的状态，在状态空间稀疏时可能比 dp 数组更高效。

另外，dp 数组往往是我们知道了推导出最终问题所需要经历的步数，而 dfs 搜索则可以 if 语句灵活判定结束的位置，适合于一开始不知道遍历步数的问题。

## 如何思考状态转移方程
推导状态转移方程的关键在于：对每个状态，考虑"最后一步"做了什么选择。这个思考方式可以系统化为以下步骤：

1. 明确当前状态表示什么（回扣状态定义）
2. 列举到达当前状态的所有可能路径（从哪些前置状态可以转移到当前状态）
3. 对每条路径计算对应的值
4. 根据问题类型，取最优值（最值问题）或求和（计数问题）

以经典的爬楼梯问题为例：每次可以爬 1 或 2 个台阶，问到达第 $n$ 阶有多少种方法。定义 `dp[i]` 为到达第 $i$ 阶的方法数。最后一步要么从第 $i-1$ 阶爬上来，要么从第 $i-2$ 阶爬上来，因此 $dp[i] = dp[i-1] + dp[i-2]$。

这个"看最后一步"的思维方式贯穿了几乎所有的 DP 问题。背包问题中，最后一步是"选或不选当前物品"；编辑距离中，最后一步是"插入、删除或替换"；股票问题中，最后一步是"买入、卖出或不操作"。只要能穷举最后一步的所有选择，状态转移方程就自然浮现出来了。

## 经典问题示例

### 1. 斐波那契数列
这是最简单的 DP 问题，用来理解状态定义和转移方程的基本模式。

```javascript
// 递归解法 O(2^n) —— 存在大量重复计算
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

// 动态规划解法 O(n) —— 自底向上填表
function fibonacciDP(n) {
    if (n <= 1) return n;
    const dp = [0, 1];
    for (let i = 2; i <= n; i++) {
        dp[i] = dp[i-1] + dp[i-2];
    }
    return dp[n];
}

// 空间优化 O(1) —— 只保留前两个状态
function fibonacciOptimized(n) {
    if (n <= 1) return n;
    let prev = 0, curr = 1;
    for (let i = 2; i <= n; i++) {
        [prev, curr] = [curr, prev + curr];
    }
    return curr;
}
```

斐波那契数列揭示了 DP 的核心价值：朴素递归 $fib(n-1) + fib(n-2)$ 会导致 $fib(n-2)$ 被计算两次，$fib(n-3)$ 被计算三次，越往下重复越严重，总复杂度达到 $O(2^n)$。DP 通过缓存消除了这些重复计算，同时保留递推的逻辑不变。

### 2. 最长公共子序列
这是二维 DP 的经典问题。定义 $dp[i][j]$ 为 $text1$ 前 $i$ 个字符和 $text2$ 前 $j$ 个字符的最长公共子序列长度。状态转移考虑最后一个字符是否匹配：匹配则在 $dp[i-1][j-1]$ 基础上加 1，不匹配则取 $dp[i-1][j]$ 和 $dp[i][j-1]$ 的较大值。

```javascript
function longestCommonSubsequence(text1, text2) {
    const m = text1.length;
    const n = text2.length;
    const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (text1[i-1] === text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}
```

这个问题展示了 DP 分析中的一个重要技巧：当两个序列对比时，当前匹配结果取决于上一步的匹配状态，通过二维状态 $dp[i][j]$ 可以完整描述两个序列的匹配进程。不匹配时取 $dp[i-1][j]$ 和 $dp[i][j-1]$ 的较大值，本质上是分别尝试丢弃 $text1$ 的第 $i$ 个字符或 $text2$ 的第 $j$ 个字符，看哪种选择能保留更长的公共子序列。

### 3. 背包问题
背包问题是 DP 中最经典的问题模型，其变种在工程中有广泛应用（资源分配、任务调度等）。

0/1 背包中每个物品只有选或不选两种选择。定义 $dp[i][w]$ 为前 $i$ 个物品、容量为 $w$ 时的最大价值。对第 $i$ 个物品，不选则 $dp[i][w] = dp[i-1][w]$，选则 $dp[i][w] = dp[i-1][w-weight] + value$，取两者较大值。这里有一个关键点：选第 $i$ 个物品时，必须回退到不包含该物品的状态 $dp[i-1][w-weight]$，而不是 $dp[i][w-weight]$，后者会导致物品被重复选择（那就变成了完全背包）。

```javascript
// 0/1背包问题
function knapsack(weights, values, W) {
    const n = weights.length;
    const dp = Array.from({ length: n + 1 }, () => Array(W + 1).fill(0));

    for (let i = 1; i <= n; i++) {
        for (let w = 1; w <= W; w++) {
            if (weights[i-1] <= w) {
                dp[i][w] = Math.max(
                    dp[i-1][w],
                    dp[i-1][w-weights[i-1]] + values[i-1]
                );
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    return dp[n][W];
}

// 完全背包问题 —— 物品可以重复选择
// 与 0/1 背包的关键区别：内层循环的遍历方向
function completeKnapsack(weights, values, W) {
    const n = weights.length;
    const dp = Array(W + 1).fill(0);

    for (let i = 0; i < n; i++) {
        for (let w = weights[i]; w <= W; w++) {
            dp[w] = Math.max(dp[w], dp[w-weights[i]] + values[i]);
        }
    }
    return dp[W];
}
```

0/1 背包和完全背包在滚动数组实现中的唯一区别是内层循环的遍历方向。0/1 背包从大到小遍历，保证 $dp[w]$ 读取的是上一轮（物品 $i-1$）的 $dp[w-weight]$，每个物品只用一次；完全背包从小到大遍历，$dp[w]$ 读取的是本轮（物品 $i$）已更新的 $dp[w-weight]$，物品可以被重复使用。理解这个区别是掌握背包问题的关键。

### 4. 股票买卖问题
股票问题展示了状态机 DP 的思想：通过多个状态变量来描述不同的"持有状态"，在状态之间进行转移。

```javascript
// 买卖股票的最佳时机（一次交易）
function maxProfit(prices) {
    let minPrice = Infinity;
    let maxProfit = 0;

    for (let price of prices) {
        minPrice = Math.min(minPrice, price);
        maxProfit = Math.max(maxProfit, price - minPrice);
    }
    return maxProfit;
}

// 买卖股票的最佳时机（多次交易）
function maxProfitMultiple(prices) {
    let profit = 0;
    for (let i = 1; i < prices.length; i++) {
        if (prices[i] > prices[i-1]) {
            profit += prices[i] - prices[i-1];
        }
    }
    return profit;
}
```

一次交易的做法本质上不是标准的 DP，而是利用了贪心的观察：遍历过程中记录历史最低价，当前价格减去历史最低价就是当天卖出的最大利润。多次交易的贪心策略更巧妙：只要明天涨就今天买明天卖，等价于把所有上涨段都吃满。

股票问题的通用 DP 模型是状态机：定义 $dp[i][0]$ 为第 $i$ 天不持有股票时的最大利润，$dp[i][1]$ 为第 $i$ 天持有股票时的最大利润。这样无论限制几次交易、是否有冷冻期，都可以通过调整状态转移方程来统一建模。例如限制最多 $k$ 次交易时，只需增加一维状态 $dp[i][j][0/1]$，其中 $j$ 表示已完成的交易次数。

## 空间优化技巧
空间优化的核心观察是：如果当前状态只依赖有限个前驱状态，就不需要存储整个 DP 表。

```javascript
// 滚动变量：斐波那契数列，空间 O(n) -> O(1)
function fibonacciCompressed(n) {
    if (n <= 1) return n;
    let prev = 0, curr = 1;
    for (let i = 2; i <= n; i++) {
        [prev, curr] = [curr, prev + curr];
    }
    return curr;
}

// 滚动数组：0/1 背包，空间 O(n*W) -> O(W)
function knapsackRolling(weights, values, W) {
    const n = weights.length;
    const dp = Array(W + 1).fill(0);

    for (let i = 0; i < n; i++) {
        for (let w = W; w >= weights[i]; w--) {
            dp[w] = Math.max(dp[w], dp[w-weights[i]] + values[i]);
        }
    }
    return dp[W];
}

// 记忆化搜索：自顶向下的 DP
function fibonacciMemo(n) {
    const memo = new Map();
    function helper(n) {
        if (n <= 1) return n;
        if (memo.has(n)) return memo.get(n);
        const result = helper(n-1) + helper(n-2);
        memo.set(n, result);
        return result;
    }
    return helper(n);
}
```

滚动数组的关键在于遍历方向必须与依赖方向一致。以 0/1 背包为例，$dp[w]$ 依赖上一轮的 $dp[w-weight]$，所以必须从大到小遍历 $w$，确保读取时 $dp[w-weight]$ 还没被覆盖。如果从小到大遍历，$dp[w-weight]$ 已经被当前轮次更新过了，就退化成了完全背包。

## 常见问题模式
理解 DP 问题的高频模式有助于快速识别问题类型和选择合适的状态定义：

1. **线性 DP**：状态沿一个线性序列推进。代表问题有最长递增子序列（$dp[i]$ 表示以 $i$ 结尾的 LIS 长度）、最大子数组和、编辑距离。
2. **区间 DP**：状态是序列上的一个区间 $[i, j]$，通常枚举区间中的分割点 $k$ 进行合并。代表问题有矩阵链乘法、石子合并、回文子串。区间 DP 的遍历顺序通常是先枚举区间长度，再枚举区间起点。
3. **背包 DP**：在容量约束下选择物品的最优方案。0/1 背包、完全背包、多重背包、分组背包都是其变种，它们之间的区别在于物品的选择次数限制不同。
4. **树形 DP**：在树结构上做 DP，通常通过后序遍历自底向上合并子树的结果。代表问题有二叉树最大路径和、树的最大独立集、树的最小支配集。
5. **状态压缩 DP**：用一个整数的二进制位来表示集合状态，适用于状态空间不大但难以用数组直接索引的场景。代表问题有旅行商问题（TSP）、棋盘覆盖问题。
6. **数位 DP**：处理与数字的各位相关的问题（如统计 $1$ 到 $n$ 中数字 $1$ 出现的次数），核心技巧是将数字的每一位作为状态，同时记录是否触碰到上界。
