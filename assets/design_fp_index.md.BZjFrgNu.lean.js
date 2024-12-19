import{_ as l,c as n,a0 as s,o as t}from"./chunks/framework.p2VkXzrt.js";const e="/assets/%E5%87%BD%E6%95%B0%E5%BC%8F.BIb_8_xG.png",i="/assets/%E5%87%BD%E6%95%B0%E5%BC%8F2.2PA_2Nkh.png",b=JSON.parse('{"title":"函数式编程","description":"","frontmatter":{},"headers":[],"relativePath":"design/fp/index.md","filePath":"design/fp/index.md"}'),p={name:"design/fp/index.md"};function o(r,a,c,d,h,g){return t(),n("div",null,a[0]||(a[0]=[s('<h1 id="函数式编程" tabindex="-1">函数式编程 <a class="header-anchor" href="#函数式编程" aria-label="Permalink to &quot;函数式编程&quot;">​</a></h1><p>函数式编程是一种编程范式，它将程序视为数学函数对输入数据的评估从而产生输出的过程。并强调使用纯函数、不可变数据、高阶函数和函数组合等概念。</p><h2 id="函数式编程的四个概念" tabindex="-1">函数式编程的四个概念 <a class="header-anchor" href="#函数式编程的四个概念" aria-label="Permalink to &quot;函数式编程的四个概念&quot;">​</a></h2><p>函数式的世界中，有四个基本概念：常量、纯函数、变量、脏函数。要区分这四个概念，需要先理解生命周期和作用域。</p><p>生命周期：一个对象或者数据从创建到死亡的全过程。创建 -&gt; 更新* n -&gt; 销毁，（6 个 hook）。常量只会经历创建和销毁，而变量会经历更新。 作用域：全局作用域和局部作用域，全局作用域对程序的所有部分可见，局部作用域对某个部分的程序和代码可见。全局作用域的生命周期与常量类似，局部作用域的生命周期与变量类似。 副作用：一个函数在运行的时候，与外界发生了交互，对外界产生了影响。</p><blockquote><p>函数式要求，对变量做出限制：</p><ol><li>尽量多地使用常量，而不是变量；</li><li>不要使用全局变量，只有常量才能全局；</li><li>函数应当是纯的；</li></ol></blockquote><p><img src="'+e+'" alt="alt text"><img src="'+i+`" alt="alt text"></p><ul><li>常量</li></ul><p>常量是指一经创建就不再改变值的量。常量是函数式编程的基础，常量是函数式编程中的最小单位，常量是不可变的，常量是纯的，常量是可组合的，常量是可复用的，常量是可测试的。常量具有较高的可维护性，因为其可控性，所以往往可以被放置在全局作用域中，拥有广泛的可见性和较长的生命周期。</p><ul><li>纯函数</li></ul><p>纯函数是指接受一些输入，并在进行一些计算处理后，返回一定的输出的这一程序过程。相较于普通的函数，纯函数的输入和输出是一一对应的，纯函数的输出只依赖于输入，纯函数的输出不会受到外部状态的影响。纯函数是函数式编程的核心，具有纯洁性、可复用性、可组合性、可测试性、较高的可维护性，因此它往往可以被放置在全局作用域中，拥有广泛的可见性和较长的生命周期。</p><p>纯函数的优势：</p><ul><li><p>确定性：具有可控的行为和确定的输入输出</p></li><li><p>可测试性：不依赖于外界环境，可以方便地测试</p></li><li><p>可移植性：不依赖外界环境，可以轻松移植</p></li><li><p>可替换性：只要签名相同</p></li><li><p>可并行性：不会改变外部状态，可以随意并行，简化了并行编程的难度</p></li><li><p>可缓存性：可以轻松进行缓存，主要用于耗时任务</p></li><li><p>变量</p></li></ul><p>变量是相对于常量而言的。变量拥有可变性，变量拥有 update 生命周期。在大型的应用程序中，程序的复杂性往往因为变量的不可控而大大增加。因此，应当让变量的生命周期尽量减短，限制其作用域，从而降低程序复杂度，保证程序的可维护性。在函数式编程中，变量也被称为状态。</p><ul><li>脏函数</li></ul><p>脏函数是相对于纯函数而言的。如果一个函数与外界发生了交互，造成了一些意想不到的影响，那么这个函数就是一个脏函数，也可以被称为<strong>副作用</strong>。一个函数可能因为以下途径产生副作用：</p><ol><li>函数通过闭包引用了外部作用域中的变量；$r、$w</li><li>函数通过传入的指针修改了指针指向的数据；$m</li><li>函数执行了 IO 操作，如读写文件、网络请求等；$i、$o</li><li>调用了其他脏函数（脏函数的传染性）；</li></ol><blockquote><h3 id="js-中的隐性副作用" tabindex="-1">JS 中的隐性副作用 <a class="header-anchor" href="#js-中的隐性副作用" aria-label="Permalink to &quot;JS 中的隐性副作用&quot;">​</a></h3><p>除去一些常见的副作用，JS 中有一些隐性副作用可能会被忽略，如：</p><ol><li><code>async/await</code> 和 Promise 的使用；</li><li><code>setTimeout</code>、<code>setInterval</code> 的使用；</li><li><code>Math.random()</code> 和 <code>Date.now()</code> 的使用；</li><li><code>console.log()</code> 的使用；</li><li>多数 Web API 的使用；</li><li>......</li></ol></blockquote><p>总结下来：主要由三种脏函数：1、闭包函数；2、写参函数；3、IO 函数。</p><h2 id="纯函数和函数组合" tabindex="-1">纯函数和函数组合 <a class="header-anchor" href="#纯函数和函数组合" aria-label="Permalink to &quot;纯函数和函数组合&quot;">​</a></h2><p><strong>函数组合</strong>是指将多个函数组合成一个函数，从而获得更复杂的功能。语言支持<strong>高阶函数</strong>是实现组合的前提。函数组合是函数式编程中的核心概念，组合可以对多个函数进行创建、增强、pipe、柯里化等，将多个简单的函数组合成一个更大的函数，从而实现更复杂的逻辑。</p><p>组合往往是针对于纯函数而言的，对于脏函数而言，组合往往没有意义，因为脏函数的副作用会限制其通用性。</p><blockquote><p>函数组合的概念区分于面向对象中的组合。面向对象中的组合是指一个类实现了多个原子化的接口，从而组合了多种<strong>能力</strong>，以达到程序的复用性。同时，解决了多继承的菱形继承问题和普通继承的代码粒度问题。</p></blockquote><h3 id="高阶函数" tabindex="-1">高阶函数 <a class="header-anchor" href="#高阶函数" aria-label="Permalink to &quot;高阶函数&quot;">​</a></h3><p>高阶函数是指至少满足以下条件之一的函数：</p><ol><li>接受一个或多个函数作为参数；</li><li>返回一个函数；</li></ol><p>高阶函数是一种强大的语言特性，在支持高阶函数的语言中，它允许我们以函数作为一等公民，从而实现函数的抽象、复用、组合等操作。高阶函数是函数式编程的核心概念，它使得函数式编程具有更高的抽象层次和更强的表达能力。</p><h3 id="多元函数" tabindex="-1">多元函数 <a class="header-anchor" href="#多元函数" aria-label="Permalink to &quot;多元函数&quot;">​</a></h3><p>函数根据参数的个数可以分成一元函数、高阶一元函数、二元函数、高阶二元函数、中阶二元函数等等。对于一元函数，我们可以简单地将一个参数的结果传入另一个函数中，这样依次调用，从而形成一个像流水线一样的结构，让程序的逻辑变得有条理。二元函数可以通过面向对象的代码来简化书写。组合或者 pipe 的写法由链式调用和使用高阶多元函数 compose 或者 pipeline 来完成，compose 从右往左调用，pipeline 从左往右调用。</p><h2 id="变量和状态管理" tabindex="-1">变量和状态管理 <a class="header-anchor" href="#变量和状态管理" aria-label="Permalink to &quot;变量和状态管理&quot;">​</a></h2><p>对于变量而言，我们无法避免其副作用，但是我们可以通过一些手段来减少副作用的影响范围，从而降低程序复杂度，保证程序的可维护性。状态管理就是致力于解决这个问题。</p><h3 id="状态容器" tabindex="-1">状态容器 <a class="header-anchor" href="#状态容器" aria-label="Permalink to &quot;状态容器&quot;">​</a></h3><p>为了隔离状态的影响范围和封装对于状态的操作，往往将状态封装到一个容器中，这个容器定义了对一类状态 T 的一种操作或者一系列操作，此处的操作可以是一个一元函数或者二元函数，可纯可脏。为了提高操作定义的复用性和控制粒度，容器分为不同的类型，每种容器只定义一种操作，往往是原子化的。状态容器可以参考面向对象编程中的对象。</p><p>在 TS 中，一个容器往往被定义为一个实现了某个泛型接口（在 Haskell 中称为高阶类型 HKT），接口中规定一个或多个接口方法，一个或多个静态方法，这些方法只能用于某一类状态 T，T 由容器的泛型指定。</p><p>如图，上面的 x，y，z 表示常量，f，g，h 表示纯函数，F 表示容器，<code>F&lt;X&gt;</code>表示封装了 X 类型的状态的容器。 常见的容器有，Option、Either、Array、IO、Task、Curry</p><p>常见的泛型接口有：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Eq：</span></span>
<span class="line"><span>    equals: &lt;T&gt;(this: T, another:T) =&gt; boolean</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Ord：</span></span>
<span class="line"><span>    compare: &lt;T&gt;(this: T, another:T) =&gt; -1 | 0 | 1</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Semi：</span></span>
<span class="line"><span>    * : &lt;T&gt;(this: T, another:T) =&gt; T</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Functor:  代表了一个映射操作，往往是从不可变区到可变区的映射</span></span>
<span class="line"><span>    map :  &lt;A, B&gt; ( this: F&lt;A&gt; , f: (a:  A) =&gt; B )  =&gt;  F&lt;B&gt;</span></span>
<span class="line"><span>    of: &lt;T&gt;(data: T) =&gt; F&lt;T&gt;</span></span>
<span class="line"><span>    lift: &lt;A, B&gt;(f: (a: A) =&gt; B) =&gt; &lt;A, B&gt;(a: F&lt;A&gt;) =&gt; F&lt;B&gt;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Apply extends Functor:  当T为一元函数时，可以有一个新的操作</span></span>
<span class="line"><span>     apply: &lt;T extends (a: A) =&gt; B, R&gt;( this: F&lt;T&gt;, a: F&lt;A&gt; ) =&gt; F&lt;B&gt;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Monad&lt;T&gt; extends Functor&lt;T&gt;:</span></span>
<span class="line"><span>     chain:  &lt;A, B&gt;( this: F&lt;A&gt; , f: (b: A)=&gt; F&lt;B&gt; ) =&gt; F&lt;B&gt;</span></span>
<span class="line"><span>     flatten: (a: F&lt;T&gt;) =&gt; T</span></span>
<span class="line"><span>     flatMap: &lt;A, B&gt;  ( a: F&lt;A&gt;, f: (a: A) =&gt; F&lt;B&gt; )   =&gt;  F&lt;B&gt;</span></span></code></pre></div><h2 id="脏函数和副作用" tabindex="-1">脏函数和副作用 <a class="header-anchor" href="#脏函数和副作用" aria-label="Permalink to &quot;脏函数和副作用&quot;">​</a></h2><p>在函数式编程中，我们无法完全避免副作用，但是我们可以通过一些手段来减少副作用的影响范围，从而降低程序复杂度，保证程序的可维护性。</p><h3 id="脏函数的延迟执行" tabindex="-1">脏函数的延迟执行 <a class="header-anchor" href="#脏函数的延迟执行" aria-label="Permalink to &quot;脏函数的延迟执行&quot;">​</a></h3><p>脏函数在不被调用之前都可以认为是无害的，但是直到它被调用之后，那么副作用产生，对于函数式的编程而言，不是要完全没有副作用，而是要把副作用局限在一个很小的范围内，或者将副作用的发生延迟到尽可能后期，最迟是在用户输入数据（前端）或者外界进行访问（后端）的时候，这个时候发生的副作用是无可避免的。</p><h3 id="脏函数的可原谅性" tabindex="-1">脏函数的可原谅性 <a class="header-anchor" href="#脏函数的可原谅性" aria-label="Permalink to &quot;脏函数的可原谅性&quot;">​</a></h3><p>在实际的开发中，我们无法完全避免副作用，但是我们可以选择原谅一些副作用，从而获得更高的开发自由度，如：</p><ol><li><strong>私有闭包</strong>，一个<strong>闭包函数</strong>独享了一个外部变量，这个外部变量除了这个函数，其他函数都不能访问。可以认为这个外部变量是函数私有的，这个副作用是可原谅的；典型例子：随机数生成器、节流函数、单例函数、函数缓存；在 JS 中，由于是单线程的，所以私有闭包不需要考虑并发问题；</li><li><strong>局部变量</strong>，如果一个函数在内部创建了一个变量，即拥有一个变量的所有权，那么在函数的内部修改，调用<strong>写参函数</strong>不会被传染；</li><li><strong>独立 IO</strong>，在函数中，调用了一些独立性较强的 IO 模块，这些模块对程序的主体功能没有直接影响，并且不会抛出错误，如日志打印、调试信息等，这些方法往往独立于程序的主体逻辑，不容易增加程序的逻辑复杂度，这些副作用是可原谅的；</li><li></li></ol><h2 id="其他概念" tabindex="-1">其他概念 <a class="header-anchor" href="#其他概念" aria-label="Permalink to &quot;其他概念&quot;">​</a></h2><ul><li>高阶函数：以函数为参数或返回值的函数</li><li>一元函数：只有一个参数和一个返回值的函数叫做一元函数</li><li>纯函数：运行之后不产生副作用的函数，它具有与传入的参数一一对应的可预期的返回值，甚至更纯——不依赖于外界的变量。</li><li>柯里化：将一个多元函数转化为一个返回可连续调用的一元函数</li><li>组合：将多个函数转化为一个组合功能的函数。compose/pipeline，compose将函数从右至左调用，pipeline从左至右调用</li><li>Point-Free：函数组合。使用各个小功能的函数组合形成更大更复杂的一个函数，直到需要输入之前，我们不关心具体需要处理的数据。</li><li>声明式：尽量只声明变量，而不是去修改已有的数据</li><li>不可变：数据一但被创建就不应被改变，并且如果是一个复杂类型，那么其内部的属性也应该是不可变的。</li></ul>`,46)]))}const m=l(p,[["render",o]]);export{b as __pageData,m as default};