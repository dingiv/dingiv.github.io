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

二者区别体现在思考的方向，dp 数组在开展时，从低处往高处逐步填表，每一次递增一次自变量，正向思考；但是 dfs 往往从高处开始搜索，逐步分解题目中的复杂度，在纵深区每一次递减一次自变量，待 dfs 触底时，从回溯区逐步归纳总结各个分支的结果，从而选择出当前节点的最佳状态，并且返回到上一层。

选择策略：
+ 选择哪种方式主要取决于问题的自然思考方向。如果问题的递推关系天然是从 base case 向上构建的（如斐波那契、背包问题），dp 数组更直观；如果问题的思考方向是从最终状态向下分解的（如树形 DP、某些博弈问题），dfs 记忆化搜索更容易理解和编写；
+ 记忆化搜索通过函数调用栈隐式地处理了遍历顺序的问题，而 dp 数组需要手动确定正确的遍历顺序——这在多维 DP 中是一个常见的出错点；
+ 从性能的角度上对比，dp 数组一般比记忆化搜索要好，不过，由于 dfs 可以剪枝，在状态空间稀疏时可能比 dp 数组更高效；
+ dp 数组往往是我们知道了推导出最终问题所需要经历的步数，而 dfs 搜索则可以 if 语句灵活判定结束的位置，适合于一开始不知道遍历步数的问题；

## 如何找到 dp 状态的定义
随着刷题量的增加，我们往往会在接触同类的题目时回忆其类似题目的定义方式，dp 也和其他题目可以通过分析入参是那几个，先初步判定题目中的复杂度规模，从而从复杂度下手，采取降解复杂度的方式，将其定义成 dp 数组的状态，例如入参是一个数组，那么可以预想的一种状态定义方式就是，我们只考虑该数组的前 i 项的时候的子问题作为 dp 数组的状态定义。

dp 数组状态定义对于"结尾"这个限定词，有莫名的情有独钟。例如**最长递增子序列**这道题中，dp 数组的定义不是 **`arr[0..i]` 的最长递增子序列**，而是以 **arr[i] 结尾的最长递增子序列**，通过限定结尾，加强了 `dp[i-1]` 和 `dp[i]` 的关系，从而方便我们进行递推公式的寻找。

## 如何思考状态转移方程
推导状态转移方程的关键在于：对每个状态，考虑"最后一步"做了什么选择。这个思考方式可以系统化为以下步骤：

1. 明确当前状态表示什么（回扣状态定义）
2. 列举到达当前状态的所有可能路径（从哪些已知前置状态可以转移到当前状态，往往就是 `dp[i-1]` 迁移到 `dp[i]`）
3. 对每条路径计算对应的值
4. 根据问题类型，取最优值（最值问题）或求和（计数问题）

以经典的爬楼梯问题为例：每次可以爬 1 或 2 个台阶，问到达第 $n$ 阶有多少种方法。定义 `dp[i]` 为到达第 $i$ 阶的方法数。最后一步要么从第 $i-1$ 阶爬上来，要么从第 $i-2$ 阶爬上来，因此 $dp[i] = dp[i-1] + dp[i-2]$。

这个"看最后一步"的思维方式贯穿了几乎所有的 DP 问题。背包问题中，最后一步是"选或不选当前物品"；编辑距离中，最后一步是"插入、删除或替换"；股票问题中，最后一步是"买入、卖出或不操作"。只要能穷举最后一步的所有选择，状态转移方程就自然浮现出来了。特别地，如果 dp 的自变量是代表入参的规模，那么最后一步的选项往往就是“选或不选”最后一步所新增的物品/参数。

## 典型题目类型
理解 DP 问题的高频模式有助于快速识别问题类型和选择合适的状态定义。每种模式有其典型的状态定义方式和推导技巧，下面逐一分析并给出经典例题。

+ 线性 dp ⭐：一维线性 dp
+ 序列 dp ⭐：基于单一/两个序列的子序列问题
+ 背包 dp ⭐：weight、value、capacity ！组合最优问题
+ 区间 dp
+ 树形 dp
+ 状态机 dp
+ 状态压缩 dp
+ 数位 dp

### 线性 DP
状态沿一个或多个线性序列推进，是最基础的 DP 模式。状态定义通常是"以位置 $i$（或 $i, j$）为结尾/范围的最优值"，推导时只关注当前位置与前驱位置的关系。

斐波那契数列是最简单的线性 DP，用来理解状态定义和转移方程的基本模式：

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

当线性序列从一维扩展到二维时，状态变成 $dp[i][j]$，表示两个序列分别在位置 $i$ 和 $j$ 时的最优值。最长公共子序列（LCS）是典型的二维线性 DP：定义 $dp[i][j]$ 为 $text1$ 前 $i$ 个字符和 $text2$ 前 $j$ 个字符的最长公共子序列长度，状态转移考虑最后一个字符是否匹配，匹配则在 $dp[i-1][j-1]$ 基础上加 1，不匹配则取 $dp[i-1][j]$ 和 $dp[i][j-1]$ 的较大值。

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

不匹配时取 $dp[i-1][j]$ 和 $dp[i][j-1]$ 的较大值，本质上是分别尝试丢弃 $text1$ 的第 $i$ 个字符或 $text2$ 的第 $j$ 个字符，看哪种选择能保留更长的公共子序列。线性 DP 的其他代表问题包括最长递增子序列（$dp[i]$ 表示以 $i$ 结尾的 LIS 长度）、最大子数组和、编辑距离等，核心技巧都是围绕"以某个位置为结尾"来定义状态。

### 序列 DP
基于一个或两个序列求解子序列相关的最优问题。和线性 DP 的区别在于，序列 DP 通常涉及子序列（不要求连续）而非子数组（要求连续），状态转移需要从更广范围的候选位置中寻找最优前驱，而非仅仅依赖相邻的前驱状态。这种"广范围寻优"的特点使得序列 DP 的转移方程中往往包含一个内层循环或二分查找，复杂度通常为 $O(n^2)$，部分问题可以借助贪心 + 二分优化到 $O(n \log n)$。

最长递增子序列（LIS）是序列 DP 的代表问题。定义 $dp[i]$ 为以 $arr[i]$ 结尾的最长递增子序列的长度，状态转移时需要枚举 $i$ 之前所有比 $arr[i]$ 小的位置 $j$，取其中 $dp[j]$ 最大的值加 1：

```javascript
// O(n^2) DP 解法
function lengthOfLIS(nums) {
    const n = nums.length;
    const dp = Array(n).fill(1);
    let max = 1;
    for (let i = 1; i < n; i++) {
        for (let j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = Math.max(dp[i], dp[j] + 1);
            }
        }
        max = Math.max(max, dp[i]);
    }
    return max;
}
```

$O(n^2)$ 解法的瓶颈在于内层循环的线性扫描。观察到 DP 的本质是在维护一组递增的"候选序列末尾值"，可以用一个数组 `tails` 记录长度为 $k$ 的递增子序列的最小末尾值——这个数组一定是严格递增的，因此可以用二分查找来维护，将整体复杂度降到 $O(n \log n)$：

```javascript
// O(n log n) 贪心 + 二分解法
function lengthOfLIS(nums) {
    const tails = [];
    for (const num of nums) {
        let left = 0, right = tails.length;
        while (left < right) {
            const mid = (left + right) >> 1;
            if (tails[mid] < num) left = mid + 1;
            else right = mid;
        }
        tails[left] = num;
    }
    return tails.length;
}
```

注意 `tails` 数组存储的并不是某个合法的 LIS 本身，而是各长度下最小的可能末尾值。末尾值越小，后续能接上的元素就越多，这正是贪心的核心思路。不过这种方法只能求出 LIS 的长度，无法还原具体的序列内容——需要还原时仍需回退到 $O(n^2)$ 的 DP 解法并记录路径。

编辑距离是双序列 DP 的经典问题，定义 $dp[i][j]$ 为将 $word1$ 的前 $i$ 个字符转换为 $word2$ 的前 $j$ 个字符所需的最少操作次数，操作包括插入、删除和替换。最后一步考虑两个序列的末尾字符：相同则无需操作，直接继承 $dp[i-1][j-1]$；不同则分别尝试插入（$dp[i][j-1]+1$）、删除（$dp[i-1][j]+1$）和替换（$dp[i-1][j-1]+1$），取最小值：

```javascript
function minDistance(word1, word2) {
    const m = word1.length, n = word2.length;
    const dp = Array.from({ length: m + 1 }, (_, i) =>
        Array.from({ length: n + 1 }, (_, j) => i === 0 ? j : j === 0 ? i : 0)
    );

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (word1[i-1] === word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1;
            }
        }
    }
    return dp[m][n];
}
```

初始化部分揭示了编辑距离的物理含义：$dp[0][j] = j$ 表示从空串变为 $word2$ 的前 $j$ 个字符需要 $j$ 次插入，$dp[i][0] = i$ 表示从 $word1$ 的前 $i$ 个字符变为空串需要 $i$ 次删除。编辑距离的框架在实际工程中有广泛的应用，如 diff 工具计算文本差异、拼写检查的候选词排序、DNA 序列比对等，是衡量两个序列相似度的基本度量。

### 背包 DP
在容量约束下选择物品组合的最优方案，是工程中应用最广的 DP 模型之一（资源分配、任务调度等）。状态定义为 $dp[i][w]$：前 $i$ 个物品、容量为 $w$ 时的最大价值。

背包 dp 本身又包含多个变种，包括：
+ 0-1 背包：可选项固定，每一个选项可选一次
+ 完全背包：可选项固定，每一个选项可选任意多次
+ 多重背包：可选项固定，每一个选项可选指定若干次
+ 分组背包：可选项先被分成多个组，在每组内部又再选一个

```javascript
// 0/1背包问题 —— 每个物品只能选一次
function knapsack(weights, values, maxWeight) {
    const n = weights.length;
    const dp = Array.from({ length: n + 1 }, () => Array(maxWeight + 1).fill(0));

    for (let i = 1; i <= n; i++) {
        const newOne = weights[i-1]
        for (let w = 1; w <= maxWeight; w++) {
            // 随着 i 的增加，每次对有一个新的物品被纳入我们的考虑范围
            // 如果这个新的物品能够被当前的容量所容纳，则我们有完整的两种方案，选或不选
            if (newOne <= w) {
                dp[i][w] = Math.max(
                    dp[i-1][w], // 不选，dp[i][w] 继承 i-1
                    values[i-1] + dp[i-1][w-newOne] // 选，则当前新物品的价格，加上剩余空间所能够容纳的最大价值（已经被计算过了）
                );
            } else {
                dp[i][w] = dp[i-1][w]; // 没得选
            }
        }
    }
    return dp[n][maxWeight];
}

// 完全背包问题 —— 物品可以重复选择
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

0/1 背包的核心技巧是"选或不选"：不选则 $dp[i][w] = dp[i-1][w]$，选则 $dp[i][w] = dp[i-1][w-weight] + value$。这里必须回退到不包含该物品的状态 $dp[i-1][w-weight]$，而不是 $dp[i][w-weight]$，后者会导致物品被重复选择（变成完全背包）。在滚动数组实现中，0/1 背包和完全背包的唯一区别是内层循环的遍历方向：0/1 背包从大到小遍历，保证 $dp[w]$ 读取的是上一轮的值，每个物品只用一次；完全背包从小到大遍历，$dp[w]$ 读取的是本轮已更新的值，物品可以被重复使用。理解这个区别是掌握背包问题的关键。其他变种包括多重背包（物品有数量上限）和分组背包（每组只能选一个），都是在 0/1 背包基础上的扩展。

### 区间 DP
状态是序列上的一个区间 $[i, j]$，通常枚举区间中的分割点 $k$ 将大区间拆成两个子区间分别求解，再合并结果。遍历顺序是先枚举区间长度（从小到大），再枚举区间起点，这样能保证计算大区间时所有子区间已经计算完毕。状态转移方程通常形如 $dp[i][j] = \min/\max_{i \le k < j} \{dp[i][k] + dp[k+1][j] + cost\}$，关键是找到正确的分割点枚举方式。代表问题有矩阵链乘法（枚举分割点找最小计算代价）、石子合并（枚举最后一次合并的位置）、回文子串（从两端向内收缩判断）。

以石子合并为例，每次只能合并相邻的两堆石子，代价为两堆石子数之和，求将所有石子合并成一堆的最小总代价。定义 $dp[i][j]$ 为合并区间 $[i, j]$ 的最小代价，枚举最后一次合并的分割点 $k$，则 $dp[i][j] = \min_{i \le k < j} \{dp[i][k] + dp[k+1][j] + sum(i,j)\}$，其中 $sum(i,j)$ 是区间内石子的总和。

```javascript
function mergeStones(stones) {
    const n = stones.length;
    const prefix = [0];
    for (let i = 0; i < n; i++) prefix.push(prefix[i] + stones[i]);
    const dp = Array.from({ length: n }, () => Array(n).fill(0));

    for (let len = 2; len <= n; len++) {
        for (let i = 0; i + len - 1 < n; i++) {
            const j = i + len - 1;
            dp[i][j] = Infinity;
            for (let k = i; k < j; k++) {
                dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k+1][j] + prefix[j+1] - prefix[i]);
            }
        }
    }
    return dp[0][n-1];
}
```

遍历顺序是关键：外层枚举区间长度、内层枚举起点，确保计算 $dp[i][j]$ 时所有更小的子区间 $dp[i][k]$ 和 $dp[k+1][j]$ 已经算好。前缀和数组用于 $O(1)$ 计算任意区间的石子总和，避免每次枚举 $k$ 时重新累加。区间 DP 的时间复杂度通常为 $O(n^3)$（三重循环：区间长度、起点、分割点），空间复杂度为 $O(n^2)$。


### 树形 DP
在树结构上做 DP，通常通过后序遍历自底向上合并子树的结果。状态定义一般以"以节点 $u$ 为根的子树"为范围，考虑 $u$ 选或不选（或处于某种状态）时子树的最优值。代表问题有二叉树最大路径和、树的最大独立集（相邻节点不能同时选）、树的最小支配集。树形 DP 的关键技巧是把子树看作独立的子问题，通过递归后序遍历自然地保证计算顺序正确。

以二叉树的最大路径和为例，路径定义为从树中任意节点出发，沿父-子连接到达任意节点的序列，要求找出节点值之和最大的那条路径。dfs 返回以当前节点为端点的最大单向路径和（只取一条分支向上传递），同时在过程中更新全局最大值（考虑两条分支都取的情况）。

```javascript
function maxPathSum(root) {
    let maxSum = -Infinity;

    function dfs(node) {
        if (!node) return 0;
        const left = Math.max(dfs(node.left), 0);
        const right = Math.max(dfs(node.right), 0);
        maxSum = Math.max(maxSum, left + node.val + right);
        return node.val + Math.max(left, right);
    }

    dfs(root);
    return maxSum;
}
```

这里的"取正贡献"是一个实用技巧：如果子树的路径和为负数，不如不选，直接取 0。这种剪枝思维在树形 DP 中很常见，可以简化状态转移的逻辑。后序遍历保证了处理当前节点时左右子树的结果已经就绪，不需要额外的遍历顺序设计。

### 状态机 DP
通过多个状态变量描述不同的"持有状态"，在状态之间进行转移。适用于有明确状态切换规则的场景，比如股票买卖中的持有/不持有、任务调度中的工作/休息。由于状态的出现，dp 数组的维度会相较同等复杂度的其他 dp 上升一个维度。

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

// 买卖股票的最佳时机（多次交易），搜索方法
function maxProfit(prices) {
    const memo = {}
    function useMemo(f) {
        return (a, b) => {
            if (!memo[a])
                memo[a] = {}
            if (memo[a][b])
                return memo[a][b]
            return memo[a][b] = f(a, b)
        }
    }

    // 定义为 dfs(idx, holds)，在规模为 idx 时，如果那时持有股票，则所获得收益是多少；
    // 允许负收益的存在，因为如果只买入，还没有卖，那么此时认为是负收益
    const dfs = useMemo(function (idx, holds) {
        if (idx <= 0)
            return holds ? -prices[0] : 0;
        if (holds)
            return Math.max(dfs(idx - 1, false) - prices[idx], dfs(idx - 1, true))
        else
            return Math.max(dfs(idx - 1, true) + prices[idx], dfs(idx - 1, false))
    })

    return dfs(prices.length - 1, false) // 当数据规模为题目所给定的全部数据时，此时认为我们手上的股票必定要卖出，否则就是倒亏钱
};

// 递推方法
function maxProfit(prices) {
    const len = prices.length
    const dp = Array.from({ length: len }, () => ({ hold: 0, unhold: 0 }))

    dp[0] = { hold: -prices[0], unhold: 0 }

    for (let i = 1; i < len; i++) {
        const curr = dp[i]
        const prev = dp[i - 1]
        curr.hold = Math.max(prev.unhold - prices[i], prev.hold)
        curr.unhold = Math.max(prev.hold + prices[i], prev.unhold)
    }

    return dp[len - 1].unhold
};

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

股票问题的通用 DP 模型是状态机：定义 $dp[i][0]$ 为第 $i$ 天不持有股票时的最大利润，$dp[i][1]$ 为第 $i$ 天持有股票时的最大利润，这样无论限制几次交易、是否有冷冻期，都可以通过调整状态转移方程来统一建模。例如限制最多 $k$ 次交易时，只需增加一维状态 $dp[i][j][0/1]$，其中 $j$ 表示已完成的交易次数。状态机 DP 的核心技巧是画出状态转换图，明确每个状态可以从哪些状态转移而来。并且从中可以看出，状态机引入的状态，会导致 dp 数组的维度增加一维。

### 状态压缩 DP
用一个**整数的二进制位**来表示集合状态，适用于状态空间不大但难以用数组直接索引的场景。例如 $n \le 20$ 时，可以用一个 $n$ 位整数表示哪些元素已被选择。代表问题有旅行商问题（TSP，状态为"已访问城市集合 + 当前所在城市"）、棋盘覆盖问题。关键技巧是将集合映射为整数，用位运算（与、或、异或）来高效地进行状态转移。

以旅行商问题（TSP）为例，给定 $n$ 个城市及两两之间的距离，求从城市 0 出发、经过每个城市恰好一次后返回起点的最短路径长度。定义 $dp[mask][i]$ 为已访问城市集合为 $mask$、当前位于城市 $i$ 时的最短路径长度，其中 $mask$ 是一个 $n$ 位二进制数，第 $j$ 位为 1 表示城市 $j$ 已被访问。

```javascript
function tsp(dist) {
    const n = dist.length;
    const fullMask = (1 << n) - 1;
    const dp = Array.from({ length: 1 << n }, () => Array(n).fill(Infinity));
    dp[1][0] = 0;

    for (let mask = 1; mask <= fullMask; mask++) {
        for (let i = 0; i < n; i++) {
            if (!(mask & (1 << i))) continue;
            for (let j = 0; j < n; j++) {
                if (mask & (1 << j)) continue;
                const nextMask = mask | (1 << j);
                dp[nextMask][j] = Math.min(dp[nextMask][j], dp[mask][i] + dist[i][j]);
            }
        }
    }

    let result = Infinity;
    for (let i = 1; i < n; i++) {
        result = Math.min(result, dp[fullMask][i] + dist[i][0]);
    }
    return result;
}
```

状态压缩的核心在于用位运算代替集合操作：`mask | (1 << j)` 表示将城市 $j$ 加入已访问集合，`mask & (1 << j)` 检查城市 $j$ 是否已被访问。TSP 的时间复杂度为 $O(2^n \cdot n^2)$，空间复杂度为 $O(2^n \cdot n)$，因此只适用于 $n \le 20$ 左右的规模。在工程实践中，状态压缩 DP 常用于配置组合、权限管理等小规模集合枚举场景。

### 数位 DP
处理与数字的各位相关的问题，例如统计 $1$ 到 $n$ 中数字 $1$ 出现的次数。核心技巧是将数字的每一位作为状态，同时用一个布尔值记录是否触碰到上界——一旦触碰上界，当前位的选择范围就被限制在 $[0, digit]$，否则可以自由选择 $[0, 9]$。通常用记忆化搜索实现更方便，因为每一位的推导逻辑是递归的。

以统计 1 到 $n$ 中数字 1 出现的次数为例。逐位处理数字的每一位，用一个布尔值 `tight` 记录当前位是否受上界约束：如果前面的所有位都恰好等于上界对应位的值，当前位只能取 $[0, digit]$，否则可以自由取 $[0, 9]$。

```javascript
function countDigitOne(n) {
    const digits = String(n).split('').map(Number);
    const memo = new Map();

    function dfs(pos, count, tight) {
        if (pos === digits.length) return count;
        const key = `${pos},${count},${tight}`;
        if (memo.has(key)) return memo.get(key);

        const limit = tight ? digits[pos] : 9;
        let result = 0;
        for (let d = 0; d <= limit; d++) {
            result += dfs(pos + 1, count + (d === 1 ? 1 : 0), tight && d === limit);
        }
        memo.set(key, result);
        return result;
    }

    return dfs(0, 0, true);
}
```

`tight` 参数是数位 DP 的精髓：当 `tight` 为 true 时，当前位的选择范围被限制在上界以内；一旦某一位选择了小于上界的值，`tight` 变为 false，后续所有位都可以自由选择 $[0, 9]$。这种设计让记忆化搜索能够统一处理"受约束"和"不受约束"两种情况，避免了分别讨论的复杂性。数位 DP 在实际工程中常用于处理与数字格式相关的统计和校验问题，比如统计满足特定数字模式的数、验证号码格式等。

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
