---
title: 图
order: 2
---

# 图

图是一种复杂的网状结构，由顶点（Vertex）和边（Edge）组成，用于表示对象之间的关系。图在计算机科学中有着广泛的应用，如社交网络、路由算法、任务调度等。

## 图的基本概念

1. **顶点（Vertex）**：图中的基本单位，也称为节点（Node）
2. **边（Edge）**：连接两个顶点的线，表示顶点之间的关系
3. **度（Degree）**：与顶点相连的边的数量
4. **路径（Path）**：从一个顶点到另一个顶点的边的序列
5. **环（Cycle）**：起点和终点相同的路径
6. **连通性（Connectivity）**：图中任意两个顶点之间是否存在路径

## 图的种类

图可以分为以下几种：

- **有向图**：图中边有方向，表示从一个顶点到另一个顶点的关系。
- **无向图**：图中边没有方向，表示两个顶点之间的关系。
- **加权图**：图中边有权值，表示两个顶点之间的关系的强度或距离。
- **无权图**：图中边没有权值，表示两个顶点之间的关系。

## 图的表示方法

### 邻接矩阵

使用二维数组表示图中顶点之间的连接关系。

```js
class Graph {
  constructor(vertices) {
    this.vertices = vertices;
    this.matrix = Array(vertices).fill().map(() => Array(vertices).fill(0));
  }

  addEdge(v1, v2, weight = 1) {
    this.matrix[v1][v2] = weight;
    this.matrix[v2][v1] = weight; // 无向图需要双向设置
  }

  removeEdge(v1, v2) {
    this.matrix[v1][v2] = 0;
    this.matrix[v2][v1] = 0;
  }

  hasEdge(v1, v2) {
    return this.matrix[v1][v2] !== 0;
  }
}
```

### 邻接表

使用数组和链表表示图中顶点之间的连接关系。

```js
class Graph {
  constructor(vertices) {
    this.vertices = vertices;
    this.adjList = new Map();
    for (let i = 0; i < vertices; i++) {
      this.adjList.set(i, []);
    }
  }

  addEdge(v1, v2, weight = 1) {
    this.adjList.get(v1).push({ vertex: v2, weight });
    this.adjList.get(v2).push({ vertex: v1, weight }); // 无向图需要双向设置
  }

  removeEdge(v1, v2) {
    this.adjList.set(v1, this.adjList.get(v1).filter(edge => edge.vertex !== v2));
    this.adjList.set(v2, this.adjList.get(v2).filter(edge => edge.vertex !== v1));
  }

  hasEdge(v1, v2) {
    return this.adjList.get(v1).some(edge => edge.vertex === v2);
  }
}
```

## 图的遍历

图的遍历是指按照某种规则访问图中所有顶点的过程。图的遍历有以下两种：

### 深度优先遍历

深度优先遍历（Depth-First Search，DFS）是指从一个顶点开始，沿着图中某条边向下深入到图的最深处，然后回溯到上一个顶点。这种遍历方式使用栈数据结构来实现。

```js
class Graph {
  // ... 前面的代码保持不变 ...

  dfs(startVertex) {
    const visited = new Set();
    const result = [];

    const dfsHelper = (vertex) => {
      visited.add(vertex);
      result.push(vertex);

      const neighbors = this.adjList.get(vertex);
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor.vertex)) {
          dfsHelper(neighbor.vertex);
        }
      }
    };

    dfsHelper(startVertex);
    return result;
  }
}
```

### 广度优先遍历

广度优先遍历（Breadth-First Search，BFS）是指从一个顶点开始，访问所有与其直接相连的顶点，然后访问与这些顶点直接相连的顶点，依此类推。这种遍历方式使用队列数据结构来实现。

```js
class Graph {
  // ... 前面的代码保持不变 ...

  bfs(startVertex) {
    const visited = new Set();
    const queue = [startVertex];
    const result = [];

    visited.add(startVertex);

    while (queue.length > 0) {
      const vertex = queue.shift();
      result.push(vertex);

      const neighbors = this.adjList.get(vertex);
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor.vertex)) {
          visited.add(neighbor.vertex);
          queue.push(neighbor.vertex);
        }
      }
    }

    return result;
  }
}
```

## 图的最短路径

### Dijkstra算法

用于计算单源最短路径，适用于没有负权边的图。

```js
class Graph {
  // ... 前面的代码保持不变 ...

  dijkstra(startVertex) {
    const distances = new Map();
    const visited = new Set();
    const previous = new Map();

    // 初始化距离
    for (let i = 0; i < this.vertices; i++) {
      distances.set(i, Infinity);
    }
    distances.set(startVertex, 0);

    while (visited.size < this.vertices) {
      // 找到未访问的最小距离顶点
      let minVertex = null;
      let minDistance = Infinity;
      for (let i = 0; i < this.vertices; i++) {
        if (!visited.has(i) && distances.get(i) < minDistance) {
          minVertex = i;
          minDistance = distances.get(i);
        }
      }

      if (minVertex === null) break;

      visited.add(minVertex);

      // 更新相邻顶点的距离
      const neighbors = this.adjList.get(minVertex);
      for (const neighbor of neighbors) {
        const newDistance = distances.get(minVertex) + neighbor.weight;
        if (newDistance < distances.get(neighbor.vertex)) {
          distances.set(neighbor.vertex, newDistance);
          previous.set(neighbor.vertex, minVertex);
        }
      }
    }

    return { distances, previous };
  }
}
```

### Floyd-Warshall算法

用于计算所有顶点之间的最短路径。

```js
class Graph {
  // ... 前面的代码保持不变 ...

  floydWarshall() {
    const dist = Array(this.vertices).fill().map(() => Array(this.vertices).fill(Infinity));

    // 初始化距离矩阵
    for (let i = 0; i < this.vertices; i++) {
      dist[i][i] = 0;
      const neighbors = this.adjList.get(i);
      for (const neighbor of neighbors) {
        dist[i][neighbor.vertex] = neighbor.weight;
      }
    }

    // 动态规划计算最短路径
    for (let k = 0; k < this.vertices; k++) {
      for (let i = 0; i < this.vertices; i++) {
        for (let j = 0; j < this.vertices; j++) {
          if (dist[i][k] + dist[k][j] < dist[i][j]) {
            dist[i][j] = dist[i][k] + dist[k][j];
          }
        }
      }
    }

    return dist;
  }
}
```

## 最小生成树

### Kruskal算法

用于寻找无向图的最小生成树。

```js
class Graph {
  // ... 前面的代码保持不变 ...

  kruskal() {
    const edges = [];
    const result = [];
    const parent = Array(this.vertices).fill().map((_, i) => i);

    // 收集所有边
    for (let i = 0; i < this.vertices; i++) {
      const neighbors = this.adjList.get(i);
      for (const neighbor of neighbors) {
        if (neighbor.vertex > i) { // 避免重复添加边
          edges.push({
            from: i,
            to: neighbor.vertex,
            weight: neighbor.weight
          });
        }
      }
    }

    // 按权重排序
    edges.sort((a, b) => a.weight - b.weight);

    // 查找根节点
    const find = (x) => {
      if (parent[x] !== x) {
        parent[x] = find(parent[x]);
      }
      return parent[x];
    };

    // 合并集合
    const union = (x, y) => {
      const rootX = find(x);
      const rootY = find(y);
      if (rootX !== rootY) {
        parent[rootY] = rootX;
        return true;
      }
      return false;
    };

    // 构建最小生成树
    for (const edge of edges) {
      if (union(edge.from, edge.to)) {
        result.push(edge);
      }
    }

    return result;
  }
}
```

## 图的应用场景

1. **社交网络**：用户之间的关系可以用图表示
2. **路由算法**：网络中的路由选择
3. **任务调度**：任务之间的依赖关系
4. **推荐系统**：用户和物品之间的关系
5. **地图导航**：地点之间的路径规划
6. **电路设计**：电子元件之间的连接关系
7. **编译器优化**：程序中的控制流图
8. **生物信息学**：蛋白质相互作用网络

## 总结

图是一种强大的数据结构，能够表示复杂的关系网络。掌握图的基本概念、表示方法和常用算法，对于解决实际问题非常重要。在实际应用中，需要根据具体场景选择合适的图算法，并注意算法的效率和实现细节。
