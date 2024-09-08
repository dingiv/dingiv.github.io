# 查找算法

## 二分查找

```js
// 注意细节
function binSearch(arr, target) {
   // 左闭右开
   let left = 0, right = arr.length
   let mid = 0
   // 必须取得整<=符号
   while (left <= right) {
      // 与左闭右开对应，使用floor进行取整
      mid = Math.floor((left + right) / 2)
      if (arr[mid] < target) {
         // 不再包含mid
         left = mid + 1
      } else if (arr[mid] > target) {
         // 不再包含mid
         right = mid - 1
      } else break
   }
   return target === arr[mid] ? mid : -1
}
```
