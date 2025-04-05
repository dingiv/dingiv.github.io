---
title: 字符串算法
order: 5
---

# 字符串算法

字符串算法是解决文本处理问题的一系列算法，在各种应用场景中扮演着重要角色，从简单的文本搜索到复杂的自然语言处理。本文将介绍几种常见的字符串算法及其应用。

## 字符串搜索与匹配

字符串搜索是指在一个较长的文本串（主串）中查找一个较短的模式串（子串）的过程。这是字符串算法中最基础也是最常见的问题。

### 暴力匹配（Brute Force）

**核心思想**：从主串的每一个位置出发，逐一比较是否能匹配模式串。

**时间复杂度**：O(n*m)，其中n是主串长度，m是模式串长度

**代码示例**：
```js
function bruteForceSearch(text, pattern) {
  const n = text.length;
  const m = pattern.length;
  
  for (let i = 0; i <= n - m; i++) {
    let j;
    for (j = 0; j < m; j++) {
      if (text[i + j] !== pattern[j]) {
        break;
      }
    }
    if (j === m) {
      return i; // 找到匹配，返回起始索引
    }
  }
  
  return -1; // 未找到匹配
}
```

### KMP算法（Knuth-Morris-Pratt）

**核心思想**：利用已经部分匹配的信息，避免不必要的比较，通过构建"部分匹配表"（next数组）来实现高效匹配。

**时间复杂度**：O(n+m)，其中n是主串长度，m是模式串长度

**代码示例**：
```js
function kmpSearch(text, pattern) {
  if (pattern.length === 0) return 0;
  
  // 构建next数组
  const next = buildNext(pattern);
  
  let i = 0, j = 0;
  while (i < text.length) {
    if (text[i] === pattern[j]) {
      i++;
      j++;
      if (j === pattern.length) {
        return i - j; // 找到完整匹配
      }
    } else if (j > 0) {
      j = next[j - 1]; // 部分匹配失败，回退
    } else {
      i++; // 完全不匹配
    }
  }
  
  return -1; // 未找到匹配
}

function buildNext(pattern) {
  const next = new Array(pattern.length).fill(0);
  let j = 0;
  
  for (let i = 1; i < pattern.length; i++) {
    while (j > 0 && pattern[i] !== pattern[j]) {
      j = next[j - 1]; // 回退
    }
    
    if (pattern[i] === pattern[j]) {
      j++;
    }
    
    next[i] = j;
  }
  
  return next;
}
```

### Boyer-Moore算法

**核心思想**：从模式串的末尾开始比较，并利用"坏字符规则"和"好后缀规则"跳过不必要的比较。

**时间复杂度**：最坏情况 O(n*m)，但在实际应用中通常比KMP更快

**代码示例**：
```js
function boyerMooreSearch(text, pattern) {
  const n = text.length;
  const m = pattern.length;
  
  // 构建坏字符规则表
  const badCharTable = buildBadCharTable(pattern);
  
  let i = m - 1; // 初始位置：模式串末尾与主串对齐
  let j = m - 1; // 从模式串末尾开始比较
  
  while (i < n) {
    if (text[i] === pattern[j]) {
      if (j === 0) {
        return i; // 找到完整匹配
      }
      i--;
      j--;
    } else {
      // 坏字符规则：根据主串当前字符跳过相应距离
      const badCharSkip = badCharTable[text[i].charCodeAt(0)] || m;
      i += m - Math.min(j, 1 + badCharSkip);
      j = m - 1; // 重置j到模式串末尾
    }
  }
  
  return -1; // 未找到匹配
}

function buildBadCharTable(pattern) {
  const table = {};
  const m = pattern.length;
  
  // 对模式串中每个字符，记录其最右出现位置
  for (let i = 0; i < m - 1; i++) {
    table[pattern[i].charCodeAt(0)] = m - 1 - i;
  }
  
  return table;
}
```

### Rabin-Karp算法

**核心思想**：使用哈希函数计算模式串和文本窗口的哈希值，只有哈希值相等时才进行字符逐一比较。

**时间复杂度**：平均 O(n+m)，最坏情况 O(n*m)

**代码示例**：
```js
function rabinKarpSearch(text, pattern) {
  const n = text.length;
  const m = pattern.length;
  const prime = 101; // 哈希函数使用的质数
  
  // 计算幂值
  let h = 1;
  for (let i = 0; i < m - 1; i++) {
    h = (h * 256) % prime;
  }
  
  // 计算初始哈希值
  let patternHash = 0;
  let textHash = 0;
  
  for (let i = 0; i < m; i++) {
    patternHash = (patternHash * 256 + pattern.charCodeAt(i)) % prime;
    textHash = (textHash * 256 + text.charCodeAt(i)) % prime;
  }
  
  // 滚动哈希比较
  for (let i = 0; i <= n - m; i++) {
    if (patternHash === textHash) {
      // 哈希值匹配，进行字符逐一比较
      let j;
      for (j = 0; j < m; j++) {
        if (text[i + j] !== pattern[j]) break;
      }
      if (j === m) return i;
    }
    
    // 计算下一个窗口的哈希值
    if (i < n - m) {
      textHash = (256 * (textHash - text.charCodeAt(i) * h) + text.charCodeAt(i + m)) % prime;
      if (textHash < 0) textHash += prime;
    }
  }
  
  return -1; // 未找到匹配
}
```

## 字符串编辑与比较

### 最长公共子序列（LCS）

**核心思想**：使用动态规划找出两个字符串中的最长公共子序列（不要求连续）。

**时间复杂度**：O(n*m)

**代码示例**：
```js
function longestCommonSubsequence(text1, text2) {
  const n = text1.length;
  const m = text2.length;
  
  // 创建DP表
  const dp = Array(n + 1).fill().map(() => Array(m + 1).fill(0));
  
  // 填充DP表
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (text1[i - 1] === text2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }
  
  // 构造LCS
  let i = n, j = m;
  let lcs = '';
  
  while (i > 0 && j > 0) {
    if (text1[i - 1] === text2[j - 1]) {
      lcs = text1[i - 1] + lcs;
      i--;
      j--;
    } else if (dp[i - 1][j] > dp[i][j - 1]) {
      i--;
    } else {
      j--;
    }
  }
  
  return lcs;
}
```

### 最长公共子串

**核心思想**：使用动态规划找出两个字符串中的最长公共子串（要求连续）。

**时间复杂度**：O(n*m)

**代码示例**：
```js
function longestCommonSubstring(text1, text2) {
  const n = text1.length;
  const m = text2.length;
  
  // 创建DP表
  const dp = Array(n + 1).fill().map(() => Array(m + 1).fill(0));
  
  let maxLength = 0;
  let endIndex = 0;
  
  // 填充DP表
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (text1[i - 1] === text2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
        if (dp[i][j] > maxLength) {
          maxLength = dp[i][j];
          endIndex = i - 1;
        }
      }
    }
  }
  
  return text1.substring(endIndex - maxLength + 1, endIndex + 1);
}
```

### 编辑距离（Levenshtein距离）

**核心思想**：计算将一个字符串转换为另一个字符串所需的最小操作次数（插入、删除、替换）。

**时间复杂度**：O(n*m)

**代码示例**：
```js
function editDistance(word1, word2) {
  const n = word1.length;
  const m = word2.length;
  
  // 创建DP表
  const dp = Array(n + 1).fill().map(() => Array(m + 1).fill(0));
  
  // 初始化
  for (let i = 0; i <= n; i++) {
    dp[i][0] = i;
  }
  
  for (let j = 0; j <= m; j++) {
    dp[0][j] = j;
  }
  
  // 填充DP表
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (word1[i - 1] === word2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(
          dp[i - 1][j],     // 删除
          dp[i][j - 1],     // 插入
          dp[i - 1][j - 1]  // 替换
        );
      }
    }
  }
  
  return dp[n][m];
}
```

## 字符串数据结构

### Trie树（前缀树）

**特点**：高效存储和查找字符串前缀

**应用场景**：自动补全、拼写检查、前缀搜索

**代码示例**：
```js
class TrieNode {
  constructor() {
    this.children = {};
    this.isEndOfWord = false;
  }
}

class Trie {
  constructor() {
    this.root = new TrieNode();
  }
  
  insert(word) {
    let node = this.root;
    for (const char of word) {
      if (!node.children[char]) {
        node.children[char] = new TrieNode();
      }
      node = node.children[char];
    }
    node.isEndOfWord = true;
  }
  
  search(word) {
    let node = this.root;
    for (const char of word) {
      if (!node.children[char]) {
        return false;
      }
      node = node.children[char];
    }
    return node.isEndOfWord;
  }
  
  startsWith(prefix) {
    let node = this.root;
    for (const char of prefix) {
      if (!node.children[char]) {
        return false;
      }
      node = node.children[char];
    }
    return true;
  }
}
```

### 后缀数组

**特点**：存储所有后缀并排序，便于快速查找子串

**应用场景**：全文搜索、字符串匹配

**代码示例**（简化版）：
```js
function buildSuffixArray(text) {
  const n = text.length;
  const suffixes = [];
  
  // 创建所有后缀
  for (let i = 0; i < n; i++) {
    suffixes.push({
      index: i,
      suffix: text.substring(i)
    });
  }
  
  // 按字典序排序后缀
  suffixes.sort((a, b) => a.suffix.localeCompare(b.suffix));
  
  // 返回排序后的后缀索引数组
  return suffixes.map(s => s.index);
}

function searchInSuffixArray(text, suffixArray, pattern) {
  const n = text.length;
  const m = pattern.length;
  
  // 二分查找
  let left = 0, right = n - 1;
  
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    const suffix = text.substring(suffixArray[mid]);
    const cmp = suffix.substring(0, m).localeCompare(pattern);
    
    if (cmp === 0) {
      return suffixArray[mid]; // 找到匹配
    } else if (cmp < 0) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  
  return -1; // 未找到匹配
}
```

## 正则表达式

正则表达式是一种用于字符串匹配的强大工具，在JavaScript中内置支持。

**基本用法**：
```js
// 创建正则表达式
const regex1 = /pattern/;
const regex2 = new RegExp('pattern');

// 测试匹配
regex1.test('string'); // 返回布尔值

// 查找匹配
'string'.match(regex1); // 返回匹配结果数组

// 替换匹配
'string'.replace(regex1, 'replacement');
```

**常用正则模式**：
```js
// 匹配邮箱
const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

// 匹配URL
const urlRegex = /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/;

// 匹配日期 (YYYY-MM-DD)
const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
```

## 应用场景

1. **全文搜索引擎**：结合多种字符串算法实现高效搜索
2. **拼写检查与自动更正**：使用编辑距离算法
3. **DNA序列分析**：使用最长公共子序列、后缀树等算法
4. **数据压缩**：使用LZ77、Huffman编码等算法
5. **自然语言处理**：分词、语法分析、情感分析等

## 总结

字符串算法是信息处理的基础，掌握这些算法可以解决很多实际问题。根据不同的应用场景和性能需求，选择适当的算法至关重要。随着数据量的增加，高效的字符串处理算法变得越来越重要。
