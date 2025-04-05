---
title: 动态规划
order: 2
---

# 动态规划

动态规划是一种使用空间来换时间的一种典型的算法和思想，具有广泛的使用范围，不过动态规划经常用来解决一些最值问题，这些最值问题存在以下特点：

- 具有可降解规模的重复子结构；
- 子结构之间存在递推关系，一般体现为数列的递推公式的样子；回顾数列问题，递推公式、前 N 项和，在算法中分别称为状态转移方程、前缀和。
- 无后效性，前面的决策不会对后续的决策产生影响；

动态规划一般可以将时间复杂度大于 O(n<sup>k</sup>)的算法，优化为时间复杂度为 O(n<sup>k</sup>)的算法，代价是需要使用额外空间复杂度 O(n<sup>k</sup>)，其中 k 是该动态规划问题的阶数，阶数越高对应的原本的问题的复杂度就越高，而常见的动态规划问题往往是一阶和二阶的，它们可以使用一个或者多个简单的数组序列或者表格作为 dp 数组，以此优化时间复杂度大于 O(n)或 O(n<sup>2</sup>)的问题。

## 动态规划解题步骤
1. **定义状态数组**：确定状态表示，选择合适的数据结构，确定状态维度
2. **初始化状态**：设置初始值，处理边界条件，考虑特殊情况
3. **状态转移方程**：分析状态之间的关系，确定转移条件，考虑所有可能的情况
4. **优化空间复杂度**：使用滚动数组，状态压缩，降维处理

```javascript
function dynamicProgramming(n) {
  // 定义状态数组
  let dp = new Array(n).fill(0);

  // 初始化初始状态和边界条件
  dp[0] = initial_value; // 根据具体问题确定

  // 根据状态转移方程计算每个状态
  for (let i = 1; i <= n; i++) {
    dp[i] = dpTransition(dp, i);
  }

  // 返回最终结果
  return getResult(dp);
}

// 状态转移函数，根据具体问题定义
function dpTransition(dp, i) {
  return some_function_of(dp, i); // 根据具体问题确定
  // 示例转移方程
  // dp[i] = min(dp[i-1] + cost1, dp[i-2] + cost2);
}

function getResult(dp) {
   return dp[dp.length - 1]
}
```

## 经典问题示例

### 1. 斐波那契数列
```javascript
// 递归解法 O(2^n)
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

// 动态规划解法 O(n)
function fibonacciDP(n) {
    if (n <= 1) return n;
    const dp = [0, 1];
    for (let i = 2; i <= n; i++) {
        dp[i] = dp[i-1] + dp[i-2];
    }
    return dp[n];
}

// 空间优化 O(1)
function fibonacciOptimized(n) {
    if (n <= 1) return n;
    let prev = 0, curr = 1;
    for (let i = 2; i <= n; i++) {
        [prev, curr] = [curr, prev + curr];
    }
    return curr;
}
```

### 2. 最长公共子序列
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

### 3. 背包问题
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

// 完全背包问题
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

### 4. 股票买卖问题
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

## 动态规划优化技巧

### 1. 状态压缩
```javascript
// 状态压缩示例：斐波那契数列
function fibonacciCompressed(n) {
    if (n <= 1) return n;
    let prev = 0, curr = 1;
    for (let i = 2; i <= n; i++) {
        [prev, curr] = [curr, prev + curr];
    }
    return curr;
}
```

### 2. 滚动数组
```javascript
// 滚动数组示例：背包问题
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
```

### 3. 记忆化搜索
```javascript
// 记忆化搜索示例：斐波那契数列
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

## 常见问题类型

1. **线性DP**
   - 最长递增子序列
   - 最大子数组和
   - 编辑距离

2. **区间DP**
   - 矩阵链乘法
   - 石子合并
   - 回文子串

3. **树形DP**
   - 二叉树最大路径和
   - 树的最大独立集
   - 树的最小支配集

4. **状态压缩DP**
   - 旅行商问题
   - 数位DP
   - 轮廓线DP

## 解题技巧

1. **确定状态**
   - 明确状态表示
   - 选择合适的状态维度
   - 考虑状态转移的可行性

2. **状态转移**
   - 分析状态之间的关系
   - 考虑所有可能的情况
   - 注意边界条件

3. **优化空间**
   - 使用滚动数组
   - 状态压缩
   - 降维处理

4. **代码实现**
   - 注意数组边界
   - 处理特殊情况
   - 优化空间复杂度
