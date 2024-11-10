# 正则表达式
专门用于高级字符串匹配算法的DSL。属于通用DSL，几乎被所有的语言所内置支持，其理论到实际应用的发展，经历了半个多世纪的演变，从一种数学工具，逐步成为计算机科学中不可或缺的部分，极大地提高了文本处理和数据验证的效率。

## 正则表达式的原理涉及模式匹配（Pattern Match）和有限自动机（Finite Automaton, FA）理论。
模式匹配是指使用一定的语法来代表和描述一个字符串的特征。给定一个**模式**，并通过编译，形成一个**有限自动机**程序，然后使用该模式对目标字符串进行匹配。

## 模式语法
+ 普通字符。26个英文字母、10数字、任意其他语言普通文字……
+ 特殊字符。这些特殊的符号在正则表达式中拥有特殊含义，如果需要表达这些符号本身，那么需要加上一个`\`进行转义。
   - `.`：匹配任意单个字符。
   - 位置匹配。`^`匹配字符开头，`$`匹配字符结尾。
   - 分组捕获。`()`。标记一个子表达式的开始和结束位置。子表达式可以获取供以后使用。在JS中，在匹配结果数组中的1号索引开始的位置是子表达式的结果，如果没有分组，那么该值为空。
   - 数量描述。`*`：匹配前一个字符或者分组0次或多次，`+`：1次或多次，`?`：0次或1次，`{0,3}`：匹配一个字符0到3次，闭区间，只有一个数字就是匹配明确指明匹配几次，如`{4}`；对应有这些字符匹配的非贪婪模式：`*?`:非贪婪模式，尽可能少的匹配，`+?`:非贪婪模式，尽可能少的匹配。
   - `[]`：拾取框。从拾取框中任选一个字符进行匹配，`[^]`：反向拾取，从不在框中的字母中选一个进行匹配 `|`：或者，选择左右两个字符中的一个进行匹配。
   - 非捕获匹配。用于限制单词的边界。
      1. (?:\<pattern\>)，匹配一个子表达式，但是不捕获匹配的文本
      2. (?=\<pattern\>),匹配前一个字符的位置，要求前面的紧挨着的这个位置的字符串能够匹配pattern
      3. (?<=\<pattern\>),匹配前一个字符的位置，要求后面的紧挨着的这个位置的字符串能够匹配pattern
         eg. (?<=abc).+(?=def)，匹配任意以abc开头，def结尾的字符串
      4. (?!\<pattern\>),匹配前一个字符的位置，要求前面的紧挨着的这个位置的字符串不能够匹配pattern
      4. (?<!\<pattern\>),匹配前一个字符的位置，要求前面的紧挨着的这个位置的字符串不能够匹配pattern


+ 转义字符。
   - \d：匹配数字
   - \r：匹配回车键、\n：匹配换行符、\t：换行符、` `：空格
   - \s：匹配空白字符，包括空格、制表符、换页符等等
   - \b：匹配单词边界。单词字母符号：字母（a-z，A-Z）、数字（0-9）和下划线（_）。\b是一个非匹配字符，它不匹配实际的字符，但是匹配一个空位置，这个位置必须是两侧有一个是单词字母字符，另一个不是。

## ps
不同操作系统使用不同的控制字符或组合来表示换行：
+ Windows：使用回车和换行的组合 \r\n 来表示换行。
+ Unix/Linux/macOS：使用换行 \n 来表示换行。
+ 旧版 macOS（Mac OS 9 及以前）：使用回车 \r 来表示换行。

带"<"的表示反向查找，就是以xxx为左边界，不带"<"的表示正向，以xxx为右边界，":"的为双向查找

## 常用例子
1. 单词，`\w、\W(不匹配)、\b(单词边界)`
2. 空字符，`\s、\S`
3. 中文，`[\u4e00-\u9fa5]`
4. 小数，`(-?\d+)(\.\d+)?`
5. 邮箱，`\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,6}\b`
6. 边界，`(?:xxx).*(?:xxx)、(?<=xxx).*(?=xxx)、(?<!xxx).*(?!xxx)`