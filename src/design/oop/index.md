---
title: 面向对象
order: 10
---

# 面向对象编程

面向对象：
对象：封装、继承、多态

## SOLID
+ S: 单一职责原则，对象应该具有一种单一的功能，不要把过多的功能放到同一个类中——将不同的功能进行细分
+ O: 开放封闭原则，对扩展开放，对修改关闭，添加新的功能的时候，应当尽量不要去修改已有的类或者代码——依赖注入
+ L: 里氏替换原则，对象可以在不改变程序正确性的前提下被它的子类所替换的——子类应当忠实地继承父类的所有方法
+ I: 接口隔离原则，多个特定客户端接口要好于一个宽泛用途的接口——接口的粒度应当尽量小
+ D: 依赖倒置原则，依赖于抽象而不是一个实例——依赖接口而不是实现类

## 具体做法
1. 数据封装：对数据进行封装，限制数据的访问权限，特定数据只能由特定方法进行修改和管理
2. 划分模块：将代码根据业务逻辑和功能，拆分成一个个小的模块，并且尽量保证功能的单一性，不将不同功能的代码混合到一起
3. 单向依赖：不同的业务逻辑之间会存在依赖，他们的依赖关系可以形成一个图，这个图应该是一个单向无环图，最好是一棵树
4. 依赖接口：模块之间进行依赖时，依赖于接口，不依赖于实现类，少用继承，多用实现（该做法在动态语言中可以适当放松）
5. 依赖注入：模块的运行时依赖不应当由其自己获取，而应当由外界进行注入
