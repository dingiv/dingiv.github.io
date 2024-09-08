# 图

图是一种复杂的网状结构，也可以认为是一个二维的平面结构。

## 图的种类

图可以分为以下几种：

- **有向图**：图中边有方向，表示从一个顶点到另一个顶点的关系。
- **无向图**：图中边没有方向，表示两个顶点之间的关系。
- **加权图**：图中边有权值，表示两个顶点之间的关系的强度或距离。
- **无权图**：图中边没有权值，表示两个顶点之间的关系。

## 图的遍历

图的遍历是指按照某种规则访问图中所有顶点的过程。图的遍历有以下两种：

### 深度优先遍历

深度优先遍历（Depth-First Search，DFS）是指从一个顶点开始，沿着图中某条边向下深入到图的最深处，然后回溯到上一个顶点。这种遍历方式使用栈数据结构来实现。

```js
const directions = [
  [0, 1], // 右
  [1, 0], // 下
  [0, -1], // 左
  [-1, 0], // 上
];

function isInBounds(matrix, x, y) {
  return x >= 0 && x < matrix.length && y >= 0 && y < matrix[0].length;
}

function getVisit() {
  return Array.from({ length: 6 }, () => Array(6).fill(false));
}

function dfs(matrix, x, y, visited = getVisit()) {
  if (!isInBounds(matrix, x, y) || visited[x][y]) return;

  visited[x][y] = true;
  console.log(`Visited: (${x}, ${y})`);

  for (const [dx, dy] of directions) {
    dfs(matrix, x + dx, y + dy, visited);
  }
}

// 示例
const matrix = Array.from({ length: 6 }, () => Array(6).fill(0));


dfs(matrix, 0, 0);
```

### 广度优先遍历

广度优先遍历（Breadth-First Search，BFS）是指从一个顶点开始，访问所有与其直接相连的顶点，然后访问与这些顶点直接相连的顶点，依此类推。这种遍历方式使用队列数据结构来实现。

```js
function bfs(matrix, startX, startY) {
  const queue = [[startX, startY]];
  const visited = Array.from({ length: matrix.length }, () => Array(matrix[0].length).fill(false));
  visited[startX][startY] = true;

  while (queue.length > 0) {
    const [x, y] = queue.shift();
    console.log(`Visited: (${x}, ${y})`);

    for (const [dx, dy] of directions) {
      const newX = x + dx,
        newY = y + dy;
      if (isInBounds(matrix, newX, newY) && !visited[newX][newY]) {
        visited[newX][newY] = true;
        queue.push([newX, newY]);
      }
    }
  }
}

// 示例
bfs(matrix, 0, 0);
```
