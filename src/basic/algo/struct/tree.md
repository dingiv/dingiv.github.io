---
title: 树
order: 1
---

# 树

树是一种典型的分支结构，由节点（Node）和边（Edge）组成，用于表示层次关系。树在计算机科学中有着广泛的应用，如文件系统、数据库索引、编译器语法分析等。

## 树的基本概念

1. **节点（Node）**：树的基本单位，包含数据和指向子节点的引用
2. **根节点（Root）**：树的顶层节点，没有父节点
3. **叶子节点（Leaf）**：没有子节点的节点
4. **内部节点（Internal Node）**：有子节点的节点
5. **度（Degree）**：节点的子节点数量
6. **深度（Depth）**：从根节点到当前节点的路径长度
7. **高度（Height）**：从当前节点到最远叶子节点的路径长度
8. **层（Level）**：根节点为第0层，其子节点为第1层，以此类推

## 树的特性

树具有以下特性：
1. 树中任意两个结点之间存在唯一的路径。
2. 树中没有环。
3. 树中每个结点只有一个父结点（根结点除外）。

## 树的类型

树有很多类型，但是常见的树有：

### 二叉树（Binary Tree）

每个结点最多有两个子结点的树。

```js
class TreeNode {
  constructor(value) {
    this.value = value;
    this.left = null;
    this.right = null;
  }
}

class BinaryTree {
  constructor() {
    this.root = null;
  }

  // 插入节点
  insert(value) {
    const newNode = new TreeNode(value);
    if (!this.root) {
      this.root = newNode;
      return;
    }

    const queue = [this.root];
    while (queue.length > 0) {
      const node = queue.shift();
      if (!node.left) {
        node.left = newNode;
        return;
      }
      if (!node.right) {
        node.right = newNode;
        return;
      }
      queue.push(node.left);
      queue.push(node.right);
    }
  }

  // 删除节点
  delete(value) {
    if (!this.root) return null;
    if (this.root.value === value) {
      this.root = null;
      return;
    }

    const queue = [this.root];
    let targetNode = null;
    let lastNode = null;

    while (queue.length > 0) {
      lastNode = queue.shift();
      if (lastNode.value === value) {
        targetNode = lastNode;
      }
      if (lastNode.left) queue.push(lastNode.left);
      if (lastNode.right) queue.push(lastNode.right);
    }

    if (targetNode) {
      targetNode.value = lastNode.value;
      this.deleteDeepest(lastNode);
    }
  }

  deleteDeepest(node) {
    const queue = [this.root];
    while (queue.length > 0) {
      const current = queue.shift();
      if (current.left === node) {
        current.left = null;
        return;
      }
      if (current.right === node) {
        current.right = null;
        return;
      }
      if (current.left) queue.push(current.left);
      if (current.right) queue.push(current.right);
    }
  }
}
```

### 二叉搜索树（Binary Search Tree）

每个结点的左子树中的所有结点的值都小于该结点的值，右子树中的所有结点的值都大于该结点的值。

```js
class BinarySearchTree extends BinaryTree {
  insert(value) {
    const newNode = new TreeNode(value);
    if (!this.root) {
      this.root = newNode;
      return;
    }

    let current = this.root;
    while (true) {
      if (value < current.value) {
        if (!current.left) {
          current.left = newNode;
          return;
        }
        current = current.left;
      } else {
        if (!current.right) {
          current.right = newNode;
          return;
        }
        current = current.right;
      }
    }
  }

  search(value) {
    let current = this.root;
    while (current) {
      if (value === current.value) return current;
      if (value < current.value) {
        current = current.left;
      } else {
        current = current.right;
      }
    }
    return null;
  }

  delete(value) {
    this.root = this.deleteNode(this.root, value);
  }

  deleteNode(node, value) {
    if (!node) return null;

    if (value < node.value) {
      node.left = this.deleteNode(node.left, value);
    } else if (value > node.value) {
      node.right = this.deleteNode(node.right, value);
    } else {
      if (!node.left) return node.right;
      if (!node.right) return node.left;

      const minRight = this.findMin(node.right);
      node.value = minRight.value;
      node.right = this.deleteNode(node.right, minRight.value);
    }

    return node;
  }

  findMin(node) {
    while (node.left) {
      node = node.left;
    }
    return node;
  }
}
```

### AVL树

一种自平衡的二叉搜索树，每个结点的左子树和右子树的高度差不能超过1。

```js
class AVLNode extends TreeNode {
  constructor(value) {
    super(value);
    this.height = 1;
  }
}

class AVLTree extends BinarySearchTree {
  getHeight(node) {
    return node ? node.height : 0;
  }

  getBalanceFactor(node) {
    return node ? this.getHeight(node.left) - this.getHeight(node.right) : 0;
  }

  updateHeight(node) {
    node.height = Math.max(this.getHeight(node.left), this.getHeight(node.right)) + 1;
  }

  rotateRight(node) {
    const left = node.left;
    const leftRight = left.right;

    left.right = node;
    node.left = leftRight;

    this.updateHeight(node);
    this.updateHeight(left);

    return left;
  }

  rotateLeft(node) {
    const right = node.right;
    const rightLeft = right.left;

    right.left = node;
    node.right = rightLeft;

    this.updateHeight(node);
    this.updateHeight(right);

    return right;
  }

  balance(node) {
    if (!node) return null;

    this.updateHeight(node);
    const balanceFactor = this.getBalanceFactor(node);

    if (balanceFactor > 1) {
      if (this.getBalanceFactor(node.left) < 0) {
        node.left = this.rotateLeft(node.left);
      }
      return this.rotateRight(node);
    }

    if (balanceFactor < -1) {
      if (this.getBalanceFactor(node.right) > 0) {
        node.right = this.rotateRight(node.right);
      }
      return this.rotateLeft(node);
    }

    return node;
  }

  insert(value) {
    this.root = this.insertNode(this.root, value);
  }

  insertNode(node, value) {
    if (!node) return new AVLNode(value);

    if (value < node.value) {
      node.left = this.insertNode(node.left, value);
    } else {
      node.right = this.insertNode(node.right, value);
    }

    return this.balance(node);
  }

  delete(value) {
    this.root = this.deleteNode(this.root, value);
  }

  deleteNode(node, value) {
    if (!node) return null;

    if (value < node.value) {
      node.left = this.deleteNode(node.left, value);
    } else if (value > node.value) {
      node.right = this.deleteNode(node.right, value);
    } else {
      if (!node.left) return node.right;
      if (!node.right) return node.left;

      const minRight = this.findMin(node.right);
      node.value = minRight.value;
      node.right = this.deleteNode(node.right, minRight.value);
    }

    return this.balance(node);
  }
}
```

### 红黑树

一种自平衡的二叉搜索树，通过颜色标记来保持平衡。

```js
class RedBlackNode extends TreeNode {
  constructor(value) {
    super(value);
    this.color = 'RED'; // 新节点默认为红色
  }
}

class RedBlackTree extends BinarySearchTree {
  insert(value) {
    const node = new RedBlackNode(value);
    this.root = this.insertNode(this.root, node);
    this.root.color = 'BLACK'; // 根节点始终为黑色
  }

  insertNode(root, node) {
    if (!root) return node;

    if (node.value < root.value) {
      root.left = this.insertNode(root.left, node);
    } else {
      root.right = this.insertNode(root.right, node);
    }

    // 修复红黑树性质
    if (this.isRed(root.right) && !this.isRed(root.left)) {
      root = this.rotateLeft(root);
    }
    if (this.isRed(root.left) && this.isRed(root.left.left)) {
      root = this.rotateRight(root);
    }
    if (this.isRed(root.left) && this.isRed(root.right)) {
      this.flipColors(root);
    }

    return root;
  }

  isRed(node) {
    return node ? node.color === 'RED' : false;
  }

  rotateLeft(node) {
    const right = node.right;
    node.right = right.left;
    right.left = node;
    right.color = node.color;
    node.color = 'RED';
    return right;
  }

  rotateRight(node) {
    const left = node.left;
    node.left = left.right;
    left.right = node;
    left.color = node.color;
    node.color = 'RED';
    return left;
  }

  flipColors(node) {
    node.color = 'RED';
    node.left.color = 'BLACK';
    node.right.color = 'BLACK';
  }
}
```

### B树和B+树

B树是一种多路平衡搜索树，B+树是B树的变种，常用于数据库索引。

```js
class BTreeNode {
  constructor(degree) {
    this.degree = degree;
    this.keys = [];
    this.children = [];
    this.isLeaf = true;
  }
}

class BTree {
  constructor(degree = 2) {
    this.degree = degree;
    this.root = new BTreeNode(degree);
  }

  insert(key) {
    const root = this.root;
    if (root.keys.length === 2 * this.degree - 1) {
      const newRoot = new BTreeNode(this.degree);
      newRoot.isLeaf = false;
      newRoot.children.push(root);
      this.root = newRoot;
      this.splitChild(newRoot, 0);
    }
    this.insertNonFull(this.root, key);
  }

  insertNonFull(node, key) {
    let i = node.keys.length - 1;
    if (node.isLeaf) {
      while (i >= 0 && key < node.keys[i]) {
        node.keys[i + 1] = node.keys[i];
        i--;
      }
      node.keys[i + 1] = key;
    } else {
      while (i >= 0 && key < node.keys[i]) {
        i--;
      }
      i++;
      if (node.children[i].keys.length === 2 * this.degree - 1) {
        this.splitChild(node, i);
        if (key > node.keys[i]) {
          i++;
        }
      }
      this.insertNonFull(node.children[i], key);
    }
  }

  splitChild(parent, index) {
    const child = parent.children[index];
    const newChild = new BTreeNode(this.degree);
    newChild.isLeaf = child.isLeaf;

    for (let i = 0; i < this.degree - 1; i++) {
      newChild.keys[i] = child.keys[i + this.degree];
    }

    if (!child.isLeaf) {
      for (let i = 0; i < this.degree; i++) {
        newChild.children[i] = child.children[i + this.degree];
      }
    }

    child.keys.length = this.degree - 1;
    if (!child.isLeaf) {
      child.children.length = this.degree;
    }

    for (let i = parent.children.length; i > index + 1; i--) {
      parent.children[i] = parent.children[i - 1];
    }
    parent.children[index + 1] = newChild;

    for (let i = parent.keys.length; i > index; i--) {
      parent.keys[i] = parent.keys[i - 1];
    }
    parent.keys[index] = child.keys[this.degree - 1];
  }
}
```

### 还有一些特殊用途的树
+ **Trie树**。也叫**前缀树**、**单词查找树**、**字典树**。
+ **后缀树**。
+ **表达式树**。
+ **Huffman树**。
+ **线段树**。
+ **并查集**。

## 树的遍历

树的遍历是指访问树中每个结点的过程。常见的树遍历算法包括：

### 前序遍历

先访问根结点，然后访问左子树，最后访问右子树。

  ```js
  function preorderTraversal(node) {
    if (node === null) return;
    console.log(node.value); // 访问根节点
    preorderTraversal(node.left); // 前序遍历左子树
    preorderTraversal(node.right); // 前序遍历右子树
  }
  ```

### 中序遍历

先访问左子树，然后访问根结点，最后访问右子树。

  ```js
  function inorderTraversal(node) {
    if (node === null) return;
    inorderTraversal(node.left); // 中序遍历左子树
    console.log(node.value); // 访问根节点
    inorderTraversal(node.right); // 中序遍历右子树
  }
  ```

### 后序遍历

先访问左子树，然后访问右子树，最后访问根结点。

  ```js
  function postorderTraversal(node) {
    if (node === null) return;
    postorderTraversal(node.left); // 后序遍历左子树
    postorderTraversal(node.right); // 后序遍历右子树
    console.log(node.value); // 访问根节点
  }
  ```

### 层序遍历

从根结点开始，逐层访问树中的结点。

  ```js
  function levelOrderTraversal(root) {
    if (!root) return;
    const queue = [root];
    while (queue.length > 0) {
        const node = queue.shift(); // 出队
        console.log(node.value); // 访问节点
        if (node.left) queue.push(node.left); // 左子节点入队
        if (node.right) queue.push(node.right); // 右子节点入队
    }
  }
  ```

## 树的应用场景

1. **文件系统**：目录结构可以用树表示
2. **数据库索引**：B树和B+树用于数据库索引
3. **编译器**：语法分析树用于编译器
4. **路由算法**：决策树用于路由选择
5. **游戏AI**：决策树用于游戏AI
6. **XML/HTML解析**：DOM树用于解析XML/HTML
7. **机器学习**：决策树用于分类和回归
8. **网络协议**：路由表可以用树表示

## 总结

树是一种重要的数据结构，能够表示层次关系。掌握树的基本概念、类型和操作，对于解决实际问题非常重要。在实际应用中，需要根据具体场景选择合适的树结构，并注意树的平衡性和效率。
