---
title: 排序算法
order: 2
---

# 排序算法

## 选择排序
最贴近人类思维的排序模式。从左到右遍历，依次寻找最大值，然后放到队列的头部，再在剩下的值中寻找最大值，再把第二大的值找到了，放在队列的第二个位置。
```javascript
function selectionSort(arr) {
    let n = arr.length;
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

## 插入排序
牌佬最喜欢的一集。从形式上类似于打牌的时候，按照牌的面值大小，将牌整理成从左到右，牌面递减的操作。
```javascript
function insertionSort(arr) {
    let n = arr.length;
    for (let i = 1; i < n; i++) {
        let key = arr[i];
        let j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
    return arr;
}
```

## 归并排序
```javascript
function mergeSort(arr) {
    if (arr.length <= 1) return arr;
    const mid = Math.floor(arr.length / 2);
    const left = mergeSort(arr.slice(0, mid));
    const right = mergeSort(arr.slice(mid));
    return merge(left, right);
}

function merge(left, right) {
    let result = [];
    let l = 0, r = 0;
    while (l < left.length && r < right.length) {
        if (left[l] < right[r]) {
            result.push(left[l++]);
        } else {
            result.push(right[r++]);
        }
    }
    return result.concat(left.slice(l)).concat(right.slice(r));
}
```

## 快速排序
```javascript
function quickSort(arr) {
    if (arr.length <= 1) return arr;
    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const right = arr.filter(x => x > pivot);
    const middle = arr.filter(x => x === pivot);
    return quickSort(left).concat(middle).concat(quickSort(right));
}
```

## 堆排序
```javascript
function heapSort(arr) {
    let n = arr.length;
    // 构建大顶堆
    // 从中间开始从后往前，重复下沉
    for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
        sink(arr, i, n);
    }
    // 进行堆排序
    for (let i = n - 1; i > 0; i--) {
        [arr[0], arr[i]] = [arr[i], arr[0]];
        sink(arr, 0, i);
    }
    return arr;
}

/**
 * @param {Array} arr 堆数组，堆的范围不一定包含数组中的全部数据
 * @param {number} i 堆顶索引
 * @param {number} n 堆尾索引
 *
 * i和n提供了当前堆的活动视图
 */
function sink(arr, i, n) {
   let largest = i, left = 2 * i + 1, right = 2 * i + 2;
   if (left < n && arr[left] > arr[largest]) largest = left;
   if (right < n && arr[right] > arr[largest]) largest = right;
   if (largest !== i) {
       [arr[i], arr[largest]] = [arr[largest], arr[i]];
       sink(arr, largest, n);
   }
}
```

## 计数排序
```javascript
function countingSort(arr, maxValue) {
    let count = new Array(maxValue + 1).fill(0);
    let sortedArr = new Array(arr.length);

    // 统计每个值的出现次数
    for (let i = 0; i < arr.length; i++) {
        count[arr[i]]++;
    }

    // 累加计数
    for (let i = 1; i < count.length; i++) {
        count[i] += count[i - 1];
    }

    // 构建排序后的数组
    for (let i = arr.length - 1; i >= 0; i--) {
        sortedArr[--count[arr[i]]] = arr[i];
    }

    return sortedArr;
}
```

### 基数排序
```javascript
function radixSort(arr) {
    const max = Math.max(...arr);
    let exp = 1;
    while (max / exp > 1) {
        countingSortByDigit(arr, exp);
        exp *= 10;
    }
    return arr;
}

function countingSortByDigit(arr, exp) {
    const n = arr.length;
    const output = Array(n).fill(0);
    const count = Array(10).fill(0);

    for (let i = 0; i < n; i++) {
        count[Math.floor(arr[i] / exp) % 10]++;
    }
    for (let i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }
    for (let i = n - 1; i >= 0; i--) {
        output[count[Math.floor(arr[i] / exp) % 10] - 1] = arr[i];
        count[Math.floor(arr[i] / exp) % 10]--;
    }
    for (let i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}
```

## 桶排序
```javascript
function bucketSort(arr, bucketSize = 5) {
    if (arr.length === 0) return arr;

    let i,
        minValue = arr[0],
        maxValue = arr[0];

    // 找到数组中的最大值和最小值
    for (i = 1; i < arr.length; i++) {
        if (arr[i] < minValue) minValue = arr[i];
        else if (arr[i] > maxValue) maxValue = arr[i];
    }

    // 初始化桶
    let bucketCount = Math.floor((maxValue - minValue) / bucketSize) + 1;
    let buckets = Array.from({ length: bucketCount }, () => []);

    // 将数组元素分配到对应的桶中
    for (i = 0; i < arr.length; i++) {
        buckets[Math.floor((arr[i] - minValue) / bucketSize)].push(arr[i]);
    }

    // 对每个桶内的元素进行排序并拼接到结果数组中
    arr.length = 0;
    for (i = 0; i < buckets.length; i++) {
        buckets[i].sort((a, b) => a - b);  // 使用插入排序或其他排序算法
        arr.push(...buckets[i]);
    }

    return arr;
}
```
