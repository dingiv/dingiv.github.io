import{_ as s,c as i,o as a,a1 as n}from"./chunks/framework.CceCxLSN.js";const l="/assets/di.DiEDqS_l.png",p="/assets/bean.dio.DOIb__av.svg",F=JSON.parse('{"title":"IOC (Java篇)","description":"","frontmatter":{},"headers":[],"relativePath":"design/oop/ioc-java.md","filePath":"design/oop/ioc-java.md"}'),e={name:"design/oop/ioc-java.md"},h=n('<h1 id="ioc-java篇" tabindex="-1">IOC (Java篇) <a class="header-anchor" href="#ioc-java篇" aria-label="Permalink to &quot;IOC (Java篇)&quot;">​</a></h1><p>控制反转是一种设计原则，用于将对象的创建和依赖关系的管理从应用程序代码中分离出来，从而提高代码的可维护性和可测试性。</p><p>在传统的面向对象编程中，对象的创建和依赖关系的管理通常由应用程序代码直接负责。例如，一个类可能需要另一个类的实例作为其依赖项，应用程序代码需要负责创建和配置这些依赖项。这种直接依赖关系可能会导致代码难以测试和维护，因为应用程序代码需要了解和依赖其他类的实现细节。控制反转可以用于将对象的创建和依赖关系的管理从应用程序代码中分离出来，应用程序代码不再直接创建和配置依赖项，而是通过依赖注入（Dependency Injection）的方式将依赖项注入到应用程序代码中。依赖注入可以通过构造函数注入、属性注入或方法注入等方式实现。</p><p><img src="'+l+`" alt="alt text"></p><h2 id="控制反转的优点" tabindex="-1">控制反转的优点 <a class="header-anchor" href="#控制反转的优点" aria-label="Permalink to &quot;控制反转的优点&quot;">​</a></h2><ul><li>降低耦合度：通过将对象的创建和依赖关系的管理从应用程序代码中分离出来，可以降低应用程序代码与其他类的耦合度，从而提高代码的可维护性和可扩展性。</li><li>提高代码的可测试性：通过依赖注入，可以将依赖项替换为模拟对象或测试替身，从而更容易编写单元测试和集成测试。</li><li>提高代码的可重用性：通过将对象的创建和依赖关系的管理从应用程序代码中分离出来，可以更容易地重用和共享代码。</li><li>实现开闭原则：在后续的维护当中，如果需要修改依赖项的实现，只需要修改依赖项的实现，而不需要修改应用程序代码，从而实现开闭原则。</li></ul><h2 id="控制反转的缺点" tabindex="-1">控制反转的缺点 <a class="header-anchor" href="#控制反转的缺点" aria-label="Permalink to &quot;控制反转的缺点&quot;">​</a></h2><ul><li>增加了复杂性：通过依赖注入，需要引入额外的依赖注入容器或框架，这可能会增加系统的复杂性。</li><li>学习成本：对于新手来说，理解和使用依赖注入可能需要一定的学习成本。</li><li>性能开销：依赖注入容器或框架可能会引入一定的性能开销，特别是在频繁创建和销毁对象的情况下。</li></ul><h2 id="控制反转的示例" tabindex="-1">控制反转的示例 <a class="header-anchor" href="#控制反转的示例" aria-label="Permalink to &quot;控制反转的示例&quot;">​</a></h2><div class="language-java vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">java</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> class</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> UserService</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    private</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> UserRepository userRepository;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    public</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> UserService</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(UserRepository </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">userRepository</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        this</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.userRepository </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> userRepository;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    }</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> void</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> createUser</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(User </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">user</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        userRepository.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">save</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(user);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> class</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> UserRepository</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> void</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> save</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(User </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">user</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        // 保存用户到数据库</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> class</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> Main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> static</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> void</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">String</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[] </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">args</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        UserRepository userRepository </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> UserRepository</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">();</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        UserService userService </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> UserService</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(userRepository);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        userService.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">createUser</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> User</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">());</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><h2 id="spring" tabindex="-1">Spring <a class="header-anchor" href="#spring" aria-label="Permalink to &quot;Spring&quot;">​</a></h2><p>Spring 是一个流行的 Java 框架，它提供了强大的依赖注入功能，可以简化应用程序的开发和维护。Spring 的依赖注入可以通过 XML 配置、注解配置或 Java 配置来实现。</p><p>Spring 有多个模块，包括</p><ul><li>Spring Core：提供了核心功能，包括依赖注入、事件处理、资源管理等。</li><li>Spring AOP：提供了面向切面编程（AOP）功能，可以用于实现横切关注点，如事务管理、日志记录等。</li><li>Spring MVC：提供了模型-视图-控制器（MVC）架构的支持，可以用于构建 Web 应用程序。</li><li>Spring Context：提供了应用程序上下文，包括 Bean 工厂、事件发布、国际化等。</li><li>Spring Data：提供了数据访问对象（DAO、ORM）支持，可以简化数据库访问。</li><li>......</li></ul><h2 id="spring-bean" tabindex="-1">Spring Bean <a class="header-anchor" href="#spring-bean" aria-label="Permalink to &quot;Spring Bean&quot;">​</a></h2><p>Spring Bean 是 Spring 框架中的一个核心概念，它是一个由 Spring 容器管理的对象。Spring 容器负责创建、配置和管理 Bean 的生命周期。</p><p>Spring Bean 的生命周期包括以下步骤：</p><ol><li>实例化：Spring 容器通过反射机制创建 Bean 的实例。</li><li>属性注入：Spring 容器将 Bean 的依赖项注入到 Bean 中。</li><li>初始化：Spring 容器调用 Bean 的初始化方法，通常是通过实现 InitializingBean 接口或使用 @PostConstruct 注解。</li><li>使用：应用程序代码可以使用 Bean。</li><li>销毁：当应用程序上下文关闭时，Spring 容器调用 Bean 的销毁方法，通常是通过实现 DisposableBean 接口或使用 @PreDestroy 注解。 <img src="`+p+`" alt=""></li></ol><h2 id="spring-依赖注入" tabindex="-1">Spring 依赖注入 <a class="header-anchor" href="#spring-依赖注入" aria-label="Permalink to &quot;Spring 依赖注入&quot;">​</a></h2><p>bean 实例化方法</p><ul><li>构造函数（直接与类名绑定的方式）</li><li>工厂函数（定义一个工厂类，并指定工厂方法）</li><li>工厂接口（实现 spring 提供的工厂接口和工厂方法）</li><li>动态注册（使用容器的动态注册的函数，添加已经实例化的对象）</li></ul><p>bean 在使用时依赖注入可以通过</p><ul><li>构造函数注入</li><li>属性注入</li><li>方法注入等</li></ul><p>bean 的作用域</p><ul><li>单例（Singleton）：Spring 容器中只有一个 Bean 实例，默认的作用域。</li><li>原型（Prototype）：每次请求时，Spring 容器都会创建一个新的 Bean 实例。</li><li>请求（Request）：每次 HTTP 请求时，Spring 容器都会创建一个新的 Bean 实例。</li><li>会话（Session）：每次 HTTP 会话时，Spring 容器都会创建一个新的 Bean 实例。</li><li>全局会话（Global Session）：在 Portlet 应用中，每个全局会话都会创建一个新的 Bean 实例。</li></ul><p>bean 的循环依赖</p><ul><li>构造函数注入：通过构造函数注入，可以避免循环依赖的问题，因为构造函数注入要求在创建 Bean 实例时，所有依赖项都已经存在。</li><li>属性注入：如果两个 Bean 之间存在循环依赖，可以通过属性注入来解决循环依赖的问题，但是需要确保至少有一个 Bean 是单例作用域，并且可以在应用程序上下文启动时创建。</li></ul><h2 id="spring-注解配置" tabindex="-1">Spring 注解配置 <a class="header-anchor" href="#spring-注解配置" aria-label="Permalink to &quot;Spring 注解配置&quot;">​</a></h2><p>Spring 提供了多种注解来简化配置，包括</p><p>定义 bean</p><ul><li>@Configuration：用于定义配置类，相当于 XML 配置文件。</li><li>@Bean：用于定义 Bean，相当于 XML 配置文件中的 <code>&lt;bean&gt;</code> 标签。</li><li>@Component：用于定义组件，相当于 XML 配置文件中的 <code>&lt;bean&gt;</code> 标签。</li><li>@Service：用于定义服务，相当于 XML 配置文件中的 <code>&lt;bean&gt;</code> 标签。</li><li>@Repository：用于定义数据访问对象，相当于 XML 配置文件中的 <code>&lt;bean&gt;</code> 标签。</li></ul><p>注入依赖</p><ul><li>@Autowired：用于自动注入依赖项，相当于 XML 配置文件中的 <code>&lt;property&gt;</code> 标签。</li><li>@Qualifier：用于指定要注入的 Bean，相当于 XML 配置文件中的 <code>&lt;qualifier&gt;</code> 标签。</li><li>@Value：用于注入属性值，相当于 XML 配置文件中的 <code>&lt;value&gt;</code> 标签。</li><li>@Scope：用于指定 Bean 的作用域，相当于 XML 配置文件中的 <code>&lt;scope&gt;</code> 标签。</li><li>@Lazy：用于延迟初始化 Bean，相当于 XML 配置文件中的 <code>&lt;lazy-init&gt;</code> 标签。</li></ul><div class="language-java vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">java</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">@</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">Configuration</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> class</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> AppConfig</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    @</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">Bean</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    public</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> UserRepository </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">userRepository</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> UserRepository</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">();</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    }</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    @</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">Bean</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    public</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> UserService </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">userService</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(UserRepository </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">userRepository</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> UserService</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(userRepository);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> class</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> Main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> static</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> void</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">String</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[] </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">args</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ApplicationContext context </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> AnnotationConfigApplicationContext</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AppConfig.class);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        UserService userService </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> context.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">getBean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(UserService.class);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        userService.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">createUser</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> User</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">());</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 使用注解配置</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">@</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">Service</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> class</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> UserService</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    private</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> UserRepository userRepository;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    @</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">Autowired</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    public</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> UserService</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(UserRepository </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">userRepository</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        this</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.userRepository </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> userRepository;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    }</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    public</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> void</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> createUser</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(User </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">user</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        userRepository.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">save</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(user);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div>`,34),t=[h];function k(r,E,d,g,y,o){return a(),i("div",null,t)}const u=s(e,[["render",k]]);export{F as __pageData,u as default};
