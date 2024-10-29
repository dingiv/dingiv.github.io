# 动态规划
动态规划是一种使用空间来换时间的一种典型的算法和思想，具有广泛的使用范围，不过动态规划经常用来解决一些最值问题，这些最值问题存在以下特点：
+ 具有可降解规模的重复子结构；
+ 子结构之间存在递推关系，一般体现为数列的递推公式的样子；回顾数列问题，递推公式、前N项和，在算法中分别称为状态转移方程、前缀和。
+ 无后效性，前面的决策不会对后续的决策产生影响；

动态规划一般可以将时间复杂度大于O(n<sup>k</sup>)的算法，优化为时间复杂度为O(n<sup>k</sup>)的算法，代价是需要使用额外空间复杂度O(n<sup>k</sup>)，其中k是该动态规划问题的阶数，阶数越高对应的原本的问题的复杂度就越高，而常见的动态规划问题往往是一阶和二阶的，它们可以使用一个或者多个简单的数组序列或者表格作为dp数组，以此优化时间复杂度大于O(n)或O(n<sup>2</sup>)的问题。

```javascript
function dynamicProgramming(n) {
    // 定义状态数组
    let dp = new Array(n).fill(0);

    // 初始化初始状态和边界条件
    dp[0] = initial_value;  // 根据具体问题确定

    // 根据状态转移方程计算每个状态
    for (let i = 1; i <= n; i++) {
        dp[i] = dpTransition(dp, i);
    }

    // 返回最终结果，最终结果
    return getResult(dp);
}

// 状态转移函数，根据具体问题定义
function dpTransition(dp, i) {
   return some_function_of(dp, i);  // 根据具体问题确定
    // 示例转移方程
    // dp[i] = min(dp[i-1] + cost1, dp[i-2] + cost2);
}
```

下面是一些常见的dp算法题目：
```js
// 最终取值函数，根据题目要求和dp数组的缓存结果，返回题目所需的答案
function dpTransition(dp) {
   return some_function_of(dp);  // 根据具体问题确定
   // 示例取值
   // return dp[dp.length-1];
}

// 示例：斐波那契数列的动态规划实现
function fibonacci(n) {
   if (n <= 1) return n;
   const dp = [1, 1]
   for (let i = 2; i < n; i++) {
      dp[i] = dp[i - 1] + dp[i - 2];
   }
   return dp[n - 1];
}


// 示例：最小路径和
function minPathSum(grid) {
    let m = grid.length;
    let n = grid[0].length;
    const dp = Array.from({ length: m }, () => Array(n).fill(0));

    dp[0][0] = grid[0][0];
    for (let i = 1; i < m; i++) {
        dp[i][0] = dp[i - 1][0] + grid[i][0];
    }
    for (let j = 1; j < n; j++) {
        dp[0][j] = dp[0][j - 1] + grid[0][j];
    }

    for (let i = 1; i < m; i++) {
        for (let j = 1; j < n; j++) {
            dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
        }
    }

    return dp[m - 1][n - 1];
}

// 示例： 0/1 背包问题
// 给定 n 种物品和一个背包。物品 i 的重量为 weight[i]，价值为 value[i]，背包的容量为 W。
// 每种物品只能选择放入或不放入背包。问在不超过背包容量的前提下，如何选择物品，使得背包中的总价值最大。
function knapsack(weights, values, W) {
    let n = weights.length;
    let dp = Array.from({ length: n + 1 }, () => new Array(W + 1).fill(0));

    for (let i = 1; i <= n; i++) {
        for (let w = 1; w <= W; w++) {
            // 状态转移方程：
            // 不选择第 i 件物品：dp[i][w] = dp[i-1][w]
            // 选择第 i 件物品：dp[i][w] = dp[i-1][w-weight[i]] + value[i]（前提是 w >= weight[i]）
            // 综合：dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i])
            if (weights[i - 1] <= w) {
                dp[i][w] = Math.max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }

    return dp[n][W];
}
```
