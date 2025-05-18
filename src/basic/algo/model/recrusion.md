---
title: 递归
order: 1
---

# 递归

递归是一种遍历方式，通过函数调用自身来解决问题。递归函数通常包含两个关键部分：**递归终止条件**和**递归调用**。递归函数会不断调用自身，直到在某种情况下满足终止条件，然后不再调用自身，返回结果，最终完成整个递归过程。

## 递归的特性

递归具有三大特性：

1. 循环能力：递归函数通过不断地调用自身来实现循环，通过终止条件来结束循环，避免了显式的循环语句；
2. 分支能力：递归函数通过条件判断来控制递归的深度和方向，并且通过循环调用自身动态构建分支，实现了灵活而简洁的分支结构遍历能力；
3. 回溯能力：递归函数在递归调用之前和之后会进行一些操作，这些操作可以看作是回溯，用于恢复递归调用前的状态，或者说也可以看做是利用了程序的调用栈自动保存现场的能力，由此，我们可以利用程序的栈区空间存储计算产生的临时数据，而无需手动创建一个数据结构来维护临时数据，但是其缺点也是显而易见的，一方面，程序具有最大的调用栈限制，同时，栈区的空间也具有一定的限制，如果递归深度过大，则可能导致栈溢出 stackoverflow。

循环往往是基于一个线性的结构，其适合遍历一个线性的逻辑结构，并形成一个线性的拓扑结构；而递归则天然适合遍历一个树形结构，并形成一个树形的逻辑拓扑结构，因此，递归在处理树形结构时具有天然的优势。

基于此，循环的一个重要参数是循环结构的**长度 n**，而递归的一个重要参数是树形结构的**深度 d** 和**节点分支数 b**。循环时间复杂度就是长度 `O(n)`，递归时间复杂度就是深度和节点分支数的指数 <code>O(depth<sup>branch</sup>)</code>。

## 递归的结构
递归函数通常具有以下结构：
- 终止条件
- 当前层逻辑处理
- 递归调用
- 回溯区逻辑处理

```javascript
function recursion() {
  // 递归终止条件
  if (condition) {
    return;
  }
  // 纵深区
  // 递归调用
  while (condition) {
    // 处理当前层逻辑
    recursion();
    // 回溯区
  }
  return; // 返回综合答案
}
```

将任意一个递归函数改写为循环结构，通常需要借助一个辅助栈，该辅助栈用于存储递归调用前的状态，包括当前层逻辑处理结果、递归调用参数、回溯区逻辑处理结果等，借用两次循环，一次处理纵深区的内容，一次处理回溯区的内容，从而将一个递归函数改写成循环。

```javascript
function simulate() {
  const stack = [];
  while (condition) {
    // 处理当前层逻辑
    // 递归调用
    stack.push(); // 保存当前层状态
  }
  while (stack.length) {
    // 回溯区
    const state = stack.pop();
    // 处理当前层逻辑
  }
}
```

回溯区处理逻辑是可选的，如果递归函数不需要回溯，则可以省略回溯区，并且该递归函数可以在无需额外的辅助栈的帮助下改写为一个循环结构，但循环结构通常不如递归结构简洁和易读。

```javascript
function simulate() {
  while (condition) {
    // 处理当前层逻辑
    // 递归调用
  }
}
```

## 排列

```javascript
function permute(nums) {
  const result = [];

  function dfs(start) {
    if (start === nums.length - 1) {
      result.push([...nums]);
      return;
    }

    for (let i = start; i < nums.length; i++) {
      [nums[start], nums[i]] = [nums[i], nums[start]]; // 交换
      dfs(start + 1)[(nums[start], nums[i])] = // 递归生成下一个位置的排列
        [nums[i], nums[start]]; // 回溯
    }
  }

  dfs(0);
  return result;
}
```

## 组合
我们可以使用递归来生成组合。同时有两种递归的思路：

- 一种是使用 m 作为递归的深度，使用 nums 数组作为不同分支；
  ```javascript
  function combine(nums, m) {
    const result = [];
    const path = [];

    function dfs(start) {
      if (path.length === m) {
        result.push([...path]);
        return;
      }

      for (let i = start; i < nums.length; i++) {
        path.push(nums[i]);
        dfs(i + 1);
        path.pop();
      }
    }

    dfs(0);
    return result;
  }
  ```
- 另一种是使用 nums 数组作为递归的深度，使用然后使用两个分支，**选**或者**不选** `nums[depth]` 作为分支。而可以发现，该方法存在冗余分支；
  ```js
  function combine(nums, m) {
    const result = [];
    const path = [];

    function dfs(depth) {
      if (path.length === m || depth === nums.length) {
        result.push([...path]);
        return;
      }

      // 分支1：不选 nums[depth]
      dfs(depth + 1);

      // 分支2：选 nums[depth]
      path.push(nums[depth]); // 选 nums[depth]
      dfs(depth + 1);
      path.pop(); // 回溯
    }

    dfs(0);
    return result;
  }
  ```
