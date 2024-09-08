# 递归


## 排列
```javascript
function permute(nums) {
   const result = []

   function dfs(start) {
      if (start === nums.length - 1) {
         result.push([...nums])
         return
      }

      for (let i = start; i < nums.length; i++) {
         [nums[start], nums[i]] = [nums[i], nums[start]]  // 交换
         dfs(start + 1)  // 递归生成下一个位置的排列
         [nums[start], nums[i]] = [nums[i], nums[start]]  // 回溯
      }
   }

   dfs(0)
   return result
}
```

## 组合
```javascript
function combine(nums, m) {
   const result = []
   const path = []

   function dfs(start) {
      if (path.length === m) {
         result.push([...path])
         return
      }

      for (let i = start; i < nums.length; i++) {
         path.push(nums[i])
         dfs(i + 1)
         path.pop()
      }
   }

   dfs(0)
   return result
}
```
