import{_ as a,c as n,a0 as t,o as p}from"./chunks/framework.CGHvQLJz.js";const l="/assets/%E5%87%BD%E6%95%B0%E5%BC%8F.BIb_8_xG.png",e="/assets/%E5%87%BD%E6%95%B0%E5%BC%8F2.2PA_2Nkh.png",B=JSON.parse('{"title":"函数式编程","description":"","frontmatter":{},"headers":[],"relativePath":"design/fp/index.md","filePath":"design/fp/index.md"}'),i={name:"design/fp/index.md"};function g(o,s,c,r,d,h){return p(),n("div",null,s[0]||(s[0]=[t('<h1 id="函数式编程" tabindex="-1">函数式编程 <a class="header-anchor" href="#函数式编程" aria-label="Permalink to &quot;函数式编程&quot;">​</a></h1><p>函数式的世界中，有四个概念常量、纯函数、变量、脏函数。</p><p>生命周期：一个对象或者数据从创建到死亡的全过程。创建-&gt;更新*n-&gt;销毁，（6个hook）。常量只会经历创建和销毁，而变量会经历更新。 作用域：全局作用域和局部作用域，全局作用域对程序的所有部分可见，局部作用域对某个部分的程序和代码可见。全局作用域的生命周期与常量类似，局部作用域的生命周期与变量类似。 副作用：一个函数在运行的时候，与外界发生了交互，对外界产生了影响。 函数式要求，对变量做出限制：</p><blockquote><ol><li>尽量多地使用常量，而不是变量；</li><li>不要使用全局变量，只有常量才能全局；</li><li>函数应当是纯的；</li></ol></blockquote><p><img src="'+l+'" alt="alt text"><img src="'+e+`" alt="alt text"></p><p>脏函数在不被调用之前都可以认为是无害的，但是直到它被调用之后，那么副作用产生，对于函数式的编程而言，不是要完全没有副作用，而是要把副作用局限在一个很小的范围内，或者将副作用的发生延迟到尽可能后期，最迟是在用户输入数据（前端）或者外界进行访问（后端）的时候，这个时候发生的副作用是无可避免的。 脏函数无可避免，那么可以向其他范式里面那样来手动管理即可。而对于只读脏函数、幂等写脏函数这些函数式可以在一定程度上无视的。</p><p>对于函数和脏函数：可以通过高阶函数和闭包，进行：创建、增强、组合、pipe、柯里化，从而形成一个更大更复杂的函数。一元函数、高阶一元函数、二元函数、高阶二元函数、中阶二元函数，讨论二元函数是因为二元函数可以通过面向对象的代码来简化书写。组合或者pipe的写法有链式调用和使用高阶多元函数compose或者pipeline来完成，compose从右往左调用，pipeline从左往右调用。</p><p>对于变量和脏函数：统称为状态。为了隔离状态的影响范围，往往将状态封装到一个容器中，这个容器定义了对一类状态T的一种操作或者一系列操作，该操作可以是一个一元函数或者二元函数，可纯可脏。为了提高操作定义的复用性和控制粒度，容器分为不同的类型，每种容器只定义一种操作，往往是原子化的。</p><p>在TS中，一个容器往往被定义为一个实现了某个泛型接口（在Haskell中称为高阶类型HKT），接口中规定一个或多个接口方法，一个或多个静态方法，这些方法只能用于某一类状态T，T由容器的泛型指定。</p><p>如图，上面的x，y，z表示常量，f，g，h表示纯函数，F表示容器，<code>F&lt;X&gt;</code>表示封装了X类型的状态的容器。 常见的容器有，Option、Either、Array、IO、Task、Curry</p><p>常见的泛型接口有：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Eq：</span></span>
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
<span class="line"><span>     flatMap: &lt;A, B&gt;  ( a: F&lt;A&gt;, f: (a: A) =&gt; F&lt;B&gt; )   =&gt;  F&lt;B&gt;</span></span></code></pre></div>`,12)]))}const F=a(i,[["render",g]]);export{B as __pageData,F as default};
