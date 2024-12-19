import{_ as e,c as t,a0 as i,o as l}from"./chunks/framework.p2VkXzrt.js";const u=JSON.parse('{"title":"状态管理","description":"","frontmatter":{},"headers":[],"relativePath":"design/fp/state.md","filePath":"design/fp/state.md"}'),r={name:"design/fp/state.md"};function o(n,a,s,d,p,h){return l(),t("div",null,a[0]||(a[0]=[i('<h1 id="状态管理" tabindex="-1">状态管理 <a class="header-anchor" href="#状态管理" aria-label="Permalink to &quot;状态管理&quot;">​</a></h1><p>在函数的世界中，为了能够达到纯洁，我们引入了纯函数，纯函数没有副作用，没有状态的变化，但是纯函数并不能解决所有的问题，比如我们需要在某个时刻记录一些状态，比如用户登录状态，购物车状态等等，这些状态的变化需要被记录下来，但是又不能影响纯函数的执行，所以我们需要引入状态管理。</p><p>状态管理就是将状态的变化记录下来，并且能够通过纯函数来访问和修改状态，同时保证状态的变化不会影响到纯函数的执行。</p><h2 id="状态管理的基本概念" tabindex="-1">状态管理的基本概念 <a class="header-anchor" href="#状态管理的基本概念" aria-label="Permalink to &quot;状态管理的基本概念&quot;">​</a></h2><p>状态管理的基本概念包括：</p><ul><li>状态：程序运行时的数据，包括变量、对象、数组等。</li><li>状态变化：程序运行时状态的变化，包括变量的赋值、对象的修改、数组的添加等。</li><li>状态管理：将状态的变化记录下来，并且能够通过纯函数来访问和修改状态，同时保证状态的变化不会影响到纯函数的执行。</li></ul><h2 id="状态管理的实现" tabindex="-1">状态管理的实现 <a class="header-anchor" href="#状态管理的实现" aria-label="Permalink to &quot;状态管理的实现&quot;">​</a></h2><p>状态管理的实现可以通过以下几种方式：</p><ul><li>全局变量：将状态保存在全局变量中，通过脏函数来访问和修改状态；</li><li>闭包：将状态保存在闭包中，通过闭包来访问和修改状态；</li><li>状态管理库：使用状态管理库等模块在程序中进行集中化管理；</li></ul><h2 id="状态的生命周期" tabindex="-1">状态的生命周期 <a class="header-anchor" href="#状态的生命周期" aria-label="Permalink to &quot;状态的生命周期&quot;">​</a></h2><p>状态的生命周期包括：</p><ul><li>创建：init，创建状态，包括变量的初始化、对象的创建、数组的初始化等。</li><li>访问：get，通过纯函数来访问状态。</li><li>修改：set，通过纯函数来修改状态。</li><li>销毁：drop，销毁状态，包括变量的销毁、对象的销毁、数组的销毁等。</li></ul><p>一个操作对应 2 个切点，共 8 个切点</p><h2 id="状态订阅" tabindex="-1">状态订阅 <a class="header-anchor" href="#状态订阅" aria-label="Permalink to &quot;状态订阅&quot;">​</a></h2><p>状态订阅是指当状态发生变化时，通知相关的副作用函数，使得副作用在用户进行输入的时候发生。而不是我们的程序主动发出的副作用。 订阅主要是订阅 set 和 drop，因为这两个函数会改变状态，所以需要通知相关的副作用函数和衍生状态重新执行。</p><p>自动订阅和手动订阅，手动订阅需要用户手动调用 API 来指定订阅的数据，</p><h2 id="衍生状态" tabindex="-1">衍生状态 <a class="header-anchor" href="#衍生状态" aria-label="Permalink to &quot;衍生状态&quot;">​</a></h2><p>衍生状态是指通过纯函数计算出来的状态，这些状态依赖于其他状态，当其他状态发生变化时，衍生状态也会发生变化。衍生状态可以通过纯函数来计算。</p>',18)]))}const f=e(r,[["render",o]]);export{u as __pageData,f as default};
