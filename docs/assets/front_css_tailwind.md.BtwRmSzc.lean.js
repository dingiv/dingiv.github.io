import{_ as e,c as t,a0 as i,o}from"./chunks/framework.CGHvQLJz.js";const h=JSON.parse('{"title":"tailwindcss","description":"","frontmatter":{},"headers":[],"relativePath":"front/css/tailwind.md","filePath":"front/css/tailwind.md"}'),s={name:"front/css/tailwind.md"};function l(n,a,d,c,r,p){return o(),t("div",null,a[0]||(a[0]=[i('<h1 id="tailwindcss" tabindex="-1">tailwindcss <a class="header-anchor" href="#tailwindcss" aria-label="Permalink to &quot;tailwindcss&quot;">​</a></h1><p>tailwind是css原子化框架，以css后处理器的形式实现，为css提供了一组开箱即用的原子类型、一套优雅规范的设计系统、一个简单易用的样式生成器，为开发者提供便利和约束。它显著提高开发效率的同时，提高了css的复用性和组合性。</p><h2 id="主题系统" tabindex="-1">主题系统 <a class="header-anchor" href="#主题系统" aria-label="Permalink to &quot;主题系统&quot;">​</a></h2><p>强大的预定义变体，可以灵活地定义原子级的样式，从而轻松更换，包含三个子系统 颜色系统colors、尺寸系统spacing、响应系统screens（responsive），三个子系统精确地抽象了平时开发中最常用的内容，并将其可操作化</p><h2 id="指令" tabindex="-1">指令 <a class="header-anchor" href="#指令" aria-label="Permalink to &quot;指令&quot;">​</a></h2><p><code>@tailwind</code>指令可以让用户无须关心路径，对tailwind的内置资源进行导入。 <code>@apply</code>指令让用户在指定区域内展开一组tailwind类的内容。</p><h2 id="内置资源" tabindex="-1">内置资源 <a class="header-anchor" href="#内置资源" aria-label="Permalink to &quot;内置资源&quot;">​</a></h2><p>包括在三个layer中</p><ol><li>基础规则Base，在layer base层中，直接作用于普通元素的全局样式；</li><li>工具集合Utilities，在layer utilities中，代表一个css样式的一系列预定义的原子值，表现为一个类名；</li><li>组件Components，在layer components中，代表一个预定预定义的css class和样式，它可以被tailwind识别并加入tailwind的组合系统当中，表现为一组类名</li></ol><ul><li>注：Utilities更加松散，为一些基础类，Components是成块的逻辑块，并且Component中的类一般依赖于已经定义好的Utilities，通过@apply衍生基础类</li></ul><h2 id="基本语法" tabindex="-1">基本语法 <a class="header-anchor" href="#基本语法" aria-label="Permalink to &quot;基本语法&quot;">​</a></h2><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Modifier:class</span></span></code></pre></div><p><code>Modifier</code>可以有多个，最后一个是class，这两个位置也可以是方括号中加上一个&quot;任意值&quot;，表示临时属性。</p><h2 id="变体" tabindex="-1">变体 <a class="header-anchor" href="#变体" aria-label="Permalink to &quot;变体&quot;">​</a></h2><p>变体Variant与修饰符Modifier，tailwind给定一个修饰符代表一个类别产生作用时需要的条件，它被抽象成一个Variant，同时具象为一个css选择器加上一个花括号，被修饰的component将被限定在其中，这使得tailwind可以简单抽象一种状态、一类元素、一个通用属性，并能够方便快捷地与其他类别进行组合。 这个操作相当于条件判断，只有满足了该条件再声明相应的样式。例如：<code>hover:</code>、<code>xl:</code>、<code>after:</code>等等。</p><h2 id="任意值" tabindex="-1">任意值 <a class="header-anchor" href="#任意值" aria-label="Permalink to &quot;任意值&quot;">​</a></h2><p>任意值Arbitrary和声明式函数调用。</p><ul><li>类名位置处，<code>class1-[p1]</code>相当于调用动态Utilities &quot;class1&quot;或者动态Components &quot;class1&quot;函数，并传入参数&quot;p1&quot;，并产生动态样式，<code>[styleName:styleValue]</code>相当于退化为完全的行内样式，与style属性中填写的行内样式类似；</li><li>修饰符位置处，<code>modi1-[p1]:</code>相当于调用动态Variant &quot;modi1&quot;，并出入参数&quot;p1&quot;，并产生动态条件，匹配动态的条件，<code>[atrr1=value1]</code>或者<code>[&amp;:hover]</code>相当于退化为手动写选择器，需要有&quot;&amp;&quot;站位,不占位则默认在最前面,&quot;[&quot;后面一个的位置处。</li></ul>',18)]))}const q=e(s,[["render",l]]);export{h as __pageData,q as default};
