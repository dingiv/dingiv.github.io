import{_ as a,c as e,o as i,a1 as t}from"./chunks/framework.CceCxLSN.js";const l="/assets/aop.DDpJo9KN.png",u=JSON.parse('{"title":"AOP (Java篇)","description":"","frontmatter":{},"headers":[],"relativePath":"design/aop/aop-java.md","filePath":"design/aop/aop-java.md"}'),o={name:"design/aop/aop-java.md"},r=t('<h1 id="aop-java篇" tabindex="-1">AOP (Java篇) <a class="header-anchor" href="#aop-java篇" aria-label="Permalink to &quot;AOP (Java篇)&quot;">​</a></h1><p>AOP（Aspect Oriented Programming）是一种编程范式，它将横切关注点（如日志、事务管理等）与业务逻辑分离，通过预编译方式和运行期动态代理实现程序功能的统一维护的一种技术。AOP的核心思想是将业务逻辑和横切关注点分离，使得业务逻辑更加清晰和简洁。</p><p>AOP的主要优点包括：</p><ul><li>提高代码的可读性和可维护性：通过将横切关注点分离到独立的切面中，可以减少代码的重复，提高代码的可读性和可维护性。</li><li>提高代码的复用性：切面可以独立地被声明和使用，可以在不同的地方重复使用，提高代码的复用性。</li><li>提高代码的模块化：通过将横切关注点分离到独立的切面中，可以更好地实现代码的模块化，使得代码结构更加清晰和合理。</li><li>降低代码的耦合度：通过将横切关注点分离到独立的切面中，可以降低代码之间的耦合度，使得代码更加独立和可替换。</li></ul><p>spring AOP底层使用JDK动态代理和CGlib第三方库的动态代理来实现AOP，使用了代理模式。</p><p><img src="'+l+'" alt="alt text"></p><ul><li>切点：指已经存在的业务逻辑和方法</li><li>连接点：指切点之间的间隙，在连接点的位置可以插入其他额外代码，以到达增强原本代码的功能</li><li>通知：增强的额外代码称为通知代码。通知的类型有三种： 前置通知：在切点的前方加入的通知，即在切点方法被调用之前需要增加的代码，可以处理切点方法的参数。 后置通知：在切点的后方加入的通知，即在切点方法被调用之前需要增加的代码，可以处理切点方法的返回值。 环绕通知：同时在切点的方法的前方和后方加入的通知，可以处理传入的参数和返回值，甚至控制原切点方法的调用，以达到偷梁换柱的能力。</li><li>切面：可以看到增加的代码以切点为锚点，在一个切点的前后增加通知。那么，一个切点和他的通知可以被称为一个切面，并且切面与原本的方法之间是松耦合的，原本的方法对切面是无感知的。</li></ul><p>切面可以单独地被声明成一个对象类型。切面的定义包括：（1）切点表达式，用于匹配哪些已有的方法会被增强（在哪些切点的前后连接点加入通知）；（2）通知代码，需要增强的功能代码逻辑，可选通知类型。</p><h2 id="事务" tabindex="-1">事务 <a class="header-anchor" href="#事务" aria-label="Permalink to &quot;事务&quot;">​</a></h2><p>基于AOP来进一步封装的DAO层提供的事务能力，实现了spring的事务接口的DAO层方法将可以加入spring事务。目前spring JDBC template、mybaits、Herbanate都实现了spring的事务接口。不过值得注意的是，事务的使用和声明是在service层的。</p><h3 id="事务传播属性" tabindex="-1">事务传播属性 <a class="header-anchor" href="#事务传播属性" aria-label="Permalink to &quot;事务传播属性&quot;">​</a></h3><p>一个已经声明了事务的service method在调用另一个声明了事务的method的时候，嵌套的事务之间的关系如何确定？这通过内层的method的事务传播属性来决定，外层的method通过try catch内层method，来界定如何响应事务的回滚。</p><ul><li>REQUIRED，作为内层事务时，加入外层事务</li><li>REQUIRES_NEW，作为内层事务时，创建新的事务，不加入外层事务</li><li>SUPPORTS，作为内层事务时，加入外层事务，如果外层没有事务，则不创建新的事务</li><li>NOT_SUPPORTED，作为内层事务时，不加入外层事务，如果外层有事务，则挂起外层事务</li><li>NEVER，作为内层事务时，不加入外层事务，如果外层有事务，则抛出异常</li></ul><p>事务隔离级别： 本质是并发编程的内容，处理并发事务之间的关系</p>',14),p=[r];function s(n,c,d,_,h,m){return i(),e("div",null,p)}const O=a(o,[["render",s]]);export{u as __pageData,O as default};
