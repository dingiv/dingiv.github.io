---
title: 排序算法
order: 3
---

# 排序算法
排序算法是计算机科学中最基础且重要的算法之一。根据不同的应用场景和需求，我们可以选择不同的排序算法。排序算法主要分为比较排序和非比较排序两大类。

## 排序算法的选择
选择排序算法时，需要考虑以下因素：
1. **数据规模**：小规模数据可以使用简单排序，大规模数据需要使用高效排序
2. **数据分布**：如果数据分布有特点，可以使用特定排序算法
3. **稳定性要求**：如果需要保持相等元素的相对顺序，选择稳定排序
4. **空间限制**：如果空间有限，选择原地排序
5. **数据特点**：如果数据基本有序，插入排序可能更高效

## 比较排序
比较排序是通过比较元素之间的大小关系来进行排序的算法。这类算法的时间复杂度下界为 O(n log n)。

### 冒泡排序（Bubble Sort）
**特点**：
- 稳定排序
- 原地排序
- 时间复杂度：O(n²)
- 空间复杂度：O(1)

**核心思想**：
通过相邻元素的比较和交换，将较大的元素逐渐"冒泡"到数组的末尾。

**应用场景**：
- 教学示例
- 小规模数据排序
- 数据基本有序的情况

```js
function bubbleSort(arr) {
  const n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    let swapped = false;
    for (let j = 0; j < n - i - 1; j++) {
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        swapped = true;
      }
    }
    if (!swapped) break; // 优化：如果一轮比较中没有发生交换，说明已经有序
  }
  return arr;
}
```

### 选择排序（Selection Sort）

**特点**：
- 不稳定排序
- 原地排序
- 时间复杂度：O(n²)
- 空间复杂度：O(1)

**核心思想**：
每次从未排序部分选择最小（或最大）的元素，放到已排序部分的末尾。

**应用场景**：
- 教学示例
- 小规模数据排序
- 交换操作成本较高的情况

```js
function selectionSort(arr) {
  const n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    let minIndex = i;
    for (let j = i + 1; j < n; j++) {
      if (arr[j] < arr[minIndex]) {
        minIndex = j;
      }
    }
    [arr[i], arr[minIndex]] = [arr[minIndex], arr[i]];
  }
  return arr;
}
```

### 插入排序（Insertion Sort）

**特点**：
- 稳定排序
- 原地排序
- 时间复杂度：O(n²)
- 空间复杂度：O(1)

**核心思想**：
将未排序部分的元素逐个插入到已排序部分的适当位置。

**应用场景**：
- 小规模数据排序
- 数据基本有序的情况
- 在线算法（数据流式输入）

```js
function insertionSort(arr) {
  const n = arr.length;
  for (let i = 1; i < n; i++) {
    const key = arr[i];
    let j = i - 1;
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = key;
  }
  return arr;
}
```

### 归并排序（Merge Sort）
**特点**：
- 稳定排序
- 非原地排序
- 时间复杂度：O(n log n)
- 空间复杂度：O(n)

**核心思想**：
将数组分成两半，分别排序后再合并。

**应用场景**：
- 大规模数据排序
- 需要稳定排序的情况
- 外部排序（数据量大于内存）

```js
function mergeSort(arr) {
  if (arr.length <= 1) return arr;
  
  const mid = Math.floor(arr.length / 2);
  const left = mergeSort(arr.slice(0, mid));
  const right = mergeSort(arr.slice(mid));
  
  return merge(left, right);
}

function merge(left, right) {
  const result = [];
  let i = 0, j = 0;
  
  while (i < left.length && j < right.length) {
    if (left[i] <= right[j]) {
      result.push(left[i++]);
    } else {
      result.push(right[j++]);
    }
  }
  
  return result.concat(left.slice(i)).concat(right.slice(j));
}
```

### 快速排序（Quick Sort）

**特点**：
- 不稳定排序
- 原地排序
- 时间复杂度：平均 O(n log n)，最坏 O(n²)
- 空间复杂度：O(log n)

**核心思想**：
选择一个基准元素，将数组分成两部分，一部分小于基准，一部分大于基准，然后递归排序。

**应用场景**：
- 大规模数据排序
- 需要原地排序的情况
- 数据随机分布的情况

```js
function quickSort(arr, left = 0, right = arr.length - 1) {
  if (left >= right) return arr;
  
  const pivotIndex = partition(arr, left, right);
  quickSort(arr, left, pivotIndex - 1);
  quickSort(arr, pivotIndex + 1, right);
  
  return arr;
}

function partition(arr, left, right) {
  const pivot = arr[right];
  let i = left;
  
  for (let j = left; j < right; j++) {
    if (arr[j] < pivot) {
      [arr[i], arr[j]] = [arr[j], arr[i]];
      i++;
    }
  }
  
  [arr[i], arr[right]] = [arr[right], arr[i]];
  return i;
}
```

### 堆排序（Heap Sort）

**特点**：
- 不稳定排序
- 原地排序
- 时间复杂度：O(n log n)
- 空间复杂度：O(1)

**核心思想**：
利用堆这种数据结构进行排序。

**应用场景**：
- 大规模数据排序
- 需要原地排序的情况
- 需要优先队列的情况

```js
function heapSort(arr) {
  const n = arr.length;
  
  // 构建最大堆
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
    heapify(arr, n, i);
  }
  
  // 逐个提取元素
  for (let i = n - 1; i > 0; i--) {
    [arr[0], arr[i]] = [arr[i], arr[0]];
    heapify(arr, i, 0);
  }
  
  return arr;
}

function heapify(arr, n, i) {
  let largest = i;
  const left = 2 * i + 1;
  const right = 2 * i + 2;
  
  if (left < n && arr[left] > arr[largest]) {
    largest = left;
  }
  
  if (right < n && arr[right] > arr[largest]) {
    largest = right;
  }
  
  if (largest !== i) {
    [arr[i], arr[largest]] = [arr[largest], arr[i]];
    heapify(arr, n, largest);
  }
}
```

## 非比较排序

非比较排序不通过比较元素的大小关系来进行排序，通常可以达到线性时间复杂度。

### 计数排序（Counting Sort）

**特点**：
- 稳定排序
- 非原地排序
- 时间复杂度：O(n + k)
- 空间复杂度：O(n + k)

**核心思想**：
统计每个元素出现的次数，然后根据统计结果重构数组。

**应用场景**：
- 数据范围较小的情况
- 需要稳定排序的情况
- 数据分布均匀的情况

```js
function countingSort(arr) {
  const max = Math.max(...arr);
  const count = new Array(max + 1).fill(0);
  const result = new Array(arr.length);
  
  // 统计每个元素出现的次数
  for (const num of arr) {
    count[num]++;
  }
  
  // 计算每个元素的位置
  for (let i = 1; i <= max; i++) {
    count[i] += count[i - 1];
  }
  
  // 构建结果数组
  for (let i = arr.length - 1; i >= 0; i--) {
    result[count[arr[i]] - 1] = arr[i];
    count[arr[i]]--;
  }
  
  return result;
}
```

### 桶排序（Bucket Sort）

**特点**：
- 稳定排序
- 非原地排序
- 时间复杂度：平均 O(n + n²/k)，最坏 O(n²)
- 空间复杂度：O(n + k)

**核心思想**：
将元素分配到有限数量的桶中，对每个桶进行排序，然后按顺序合并。

**应用场景**：
- 数据分布均匀的情况
- 数据范围较大的情况
- 需要稳定排序的情况

```js
function bucketSort(arr, bucketSize = 5) {
  if (arr.length === 0) return arr;
  
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const bucketCount = Math.floor((max - min) / bucketSize) + 1;
  const buckets = new Array(bucketCount);
  
  for (let i = 0; i < bucketCount; i++) {
    buckets[i] = [];
  }
  
  // 将元素分配到桶中
  for (const num of arr) {
    const bucketIndex = Math.floor((num - min) / bucketSize);
    buckets[bucketIndex].push(num);
  }
  
  // 对每个桶进行排序
  const result = [];
  for (const bucket of buckets) {
    insertionSort(bucket);
    result.push(...bucket);
  }
  
  return result;
}
```

### 基数排序（Radix Sort）

**特点**：
- 稳定排序
- 非原地排序
- 时间复杂度：O(d(n + k))
- 空间复杂度：O(n + k)

**核心思想**：
按照位数进行排序，从最低位到最高位依次排序。

**应用场景**：
- 数据位数较少的情况
- 需要稳定排序的情况
- 数据范围较大的情况

```js
function radixSort(arr) {
  const max = Math.max(...arr);
  const maxDigit = String(max).length;
  
  for (let i = 0; i < maxDigit; i++) {
    const buckets = Array.from({ length: 10 }, () => []);
    
    for (const num of arr) {
      const digit = getDigit(num, i);
      buckets[digit].push(num);
    }
    
    arr = [].concat(...buckets);
  }
  
  return arr;
}

function getDigit(num, place) {
  return Math.floor(Math.abs(num) / Math.pow(10, place)) % 10;
}
```

## 排序算法的应用

1. **数据预处理**：排序是许多算法的基础步骤
2. **查找优化**：排序后的数据可以使用二分查找
3. **去重**：排序后可以方便地去除重复元素
4. **统计**：排序后可以方便地进行各种统计
5. **数据展示**：排序后的数据更易于展示和理解

## 排序算法的优化

1. **混合排序**：结合多种排序算法的优点
2. **并行排序**：利用多核处理器进行并行排序
3. **外部排序**：处理大于内存的数据集
4. **自适应排序**：根据数据特点自动选择最优排序算法
5. **稳定化处理**：将不稳定排序算法改造为稳定排序