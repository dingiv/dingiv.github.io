# 常见算法问题
列出常见的算法，按照几个大的模块进行分类。

## 查找算法
查询和读取存储在计算机中的数据是使用计算机处理信息的基本需求。在海量的数据中，如何快速找到需要的数据，是查找算法需要解决的问题。
+ 二分查找。二分查找基于一个***有序***的序列，通过比较中间值和目标值的大小，将查找范围缩小一半，直到找到目标值或者查找范围为空。
+ 查找树。查找树是一种特殊设计的数据结构，用于存储和查找数据。它是一种树形结构，每个节点包含一个键值和指向子节点的指针。查找树可以用于实现各种查找算法，如二分查找、平衡查找树等，从而显著提升读取数据的速度。
   - AVL：空间复杂度高，代码量大
   - 2-3树：代码量大
   - 红黑树：代码精悍晦涩
+ 散列表
   - 软缓存
   - 防碰撞：拉链法、线性探测

## 排序算法
将一组数据按照一定的顺序进行排列，可以加快对数据的处理、查找、删除等操作的速度。排序算法可以是基于比较的，也可以是不基于比较的。

### 基于比较
+ 选择排序
+ 插入排序
+ 归并排序。采用分治思想进行分割式排序，将一个大的问题转化为一个个子问题，然后再将各个子问题统一解决。有自顶向下的递归法，也有自底向上的循环，可以在小的范围时转接其他排序进行小数组优化。优化的手段可以从这几个方面考虑：1、小数组转接；2、归并预判；3、原地归并
+ 快速排序。使用分治思想，并进行大跨度交换，可以看做是插入排序和希尔排序的延续，同时又综合了归并排序的分治思想。优化的手段可以从这几个方面考虑：1、小数组转接；2、三取样决定分割数；3、三向分割，优化出现大部分的相同的数值作为分割数的情况。
+ 堆排序。堆又可以称之为优先队列，可以在数组上构造出分支结构，可以看做是使用分支结构带来的优势，将极端的元素快速选出并放置至合适的位置。是一种基于堆的选择排序。

### 不基于比较
+ 计数排序。计数排序适用于元素值范围较小的数组。它通过计数数组记录每个元素的出现次数，再根据计数数组构建排序后的数组。
+ 基数排序
+ 桶排序



## 字符串匹配算法
+ 朴素匹配。暴力迭代，比较长字符串中的每个位置处是否能够匹配子串。
+ KMP。用于在一个长字符串中寻找一个指定的字符串子串。
+ BMH。
+ Trie字典树
+ AC自动机
+ RK

## 加密算法
+ 朴素加密。如：凯撒、base64
+ 对称加密。AES
+ 非对称加密。RSA
+ 哈希加密。如：MD5、SHA1

## 压缩算法
