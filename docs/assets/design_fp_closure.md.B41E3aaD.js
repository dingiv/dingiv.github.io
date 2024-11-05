import{_ as t,c as o,j as e,a,o as l}from"./chunks/framework.CGHvQLJz.js";const m=JSON.parse('{"title":"闭包","description":"","frontmatter":{},"headers":[],"relativePath":"design/fp/closure.md","filePath":"design/fp/closure.md"}'),r={name:"design/fp/closure.md"};function s(p,n,c,i,d,u){return l(),o("div",null,n[0]||(n[0]=[e("h1",{id:"闭包",tabindex:"-1"},[a("闭包 "),e("a",{class:"header-anchor",href:"#闭包","aria-label":'Permalink to "闭包"'},"​")],-1),e("p",null,"函数式： 闭包：组合、函子、钩子",-1),e("p",null,"闭包： 高阶函数：以函数为参数或返回值的函数 一元函数：只有一个参数和一个返回值的函数叫做一元函数 纯函数：运行之后不产生副作用的函数，它具有与传入的参数一一对应的可预期的返回值，甚至更纯——不依赖于外界的变量。 柯里化：将一个多元函数转化为一个返回可连续调用的一元函数 组合：将多个函数转化为一个组合功能的函数。compose/pipeline，compose将函数从右至左调用，pipeline从左至右调用 Point-Free：函数组合。使用各个小功能的函数组合形成更大更复杂的一个函数，直到需要输入之前，我们不关心具体需要处理的数据。 声明式：尽量只声明变量，而不是去修改已有的数据 不可变：数据一但被创建就不应被改变，并且如果是一个复杂类型，那么其内部的属性也应该是不可变的。",-1),e("p",null,[a("函子Functor：一个任意数据类型的包装类，该类有一个方法map，"),e("code",null,"(fn:(data:T)=>E)=>Functor<E>"),a("，该map方法决定了一个外部的函数如何作用于一个被包装类接管的数据，Functor是函数式世界中的数据——数据实体类型、状态、带状态的函数、或者说是非纯函数 Monad：一个拥有flatMap方法的Functor，")],-1),e("p",null,"副作用：一个函数运行时，修改参数的成员、修改外部引用变量的值、与外界产生交互并且不确定是否改变了外部的值。副作用具有传染性，使用了非纯函数的函数也会变成非纯函数。函数只能修改在自己作用域中产生的变量。",-1),e("p",null,"纯函数的优势： （1）确定性：具有可控的行为和确定的输入输出 （2）可测试性：不依赖于外界环境，可以方便地测试 （3）可移植性：不依赖外界环境，可以轻松移植 （4）可替换性：只要签名相同 （5）可并行性：不会改变外部状态，可以随意并行，简化了并行编程的难度 （6）可缓存性：可以轻松进行缓存，主要用于耗时任务",-1),e("p",null,"数据流动：数据进入函数中，然后函数返回输出，可以认为数据从函数的入口处流并发生了类型转化，从函数的出口处传出。",-1)]))}const _=t(r,[["render",s]]);export{m as __pageData,_ as default};
