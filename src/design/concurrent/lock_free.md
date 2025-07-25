# 无锁并发
无锁并发技术是在不使用锁的情况下解决并发问题的技术，区别于锁机制，往往性能更加高效。使用锁往往容易导致死锁问题。

## 原子操作
依赖于硬件级别的能力，基于原子变量，对单一变量进行并发保证，可以基于此设计无锁队列和无锁环等数据结构，拥有极高的性能，且没有死锁。

## 自旋忙等
实现简单，但是延迟较高。CPU 需要一定得消耗。
```c
while(!is_condition()) {
   sleep(8000); // 8ms
}
```

## RCU
Read-Copy-Update 机制，在 Linux 系统广泛使用，对于读者来说支持共享读，对于写者来说，自己创建一个副本，然后修改，修改完之后再替换原本的数据，高并发读场景中的无锁利器。但是，写慢读快，延迟回收。

## 消息传递
不共享变量，从源头上解决问题

