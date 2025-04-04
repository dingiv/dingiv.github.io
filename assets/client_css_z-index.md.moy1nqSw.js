import{_ as e,c as l,o as a,ae as t}from"./chunks/framework.Dh1jimFm.js";const n="/assets/z-index.CSRIOjeU.png",h=JSON.parse('{"title":"层叠上下文与z-index","description":"","frontmatter":{},"headers":[],"relativePath":"client/css/z-index.md","filePath":"client/css/z-index.md"}'),o={name:"client/css/z-index.md"};function s(r,i,d,c,p,x){return a(),l("div",null,i[0]||(i[0]=[t('<h1 id="层叠上下文与z-index" tabindex="-1">层叠上下文与z-index <a class="header-anchor" href="#层叠上下文与z-index" aria-label="Permalink to &quot;层叠上下文与z-index&quot;">​</a></h1><p>两个元素的图层决定着元素显示的上下关系，上层的元素将覆盖下层的元素进行显示。决定图层关系的CSS属性为z-index，z-index越大，元素的显示优先级越高。</p><h2 id="z-index比较方法" tabindex="-1">z-index比较方法 <a class="header-anchor" href="#z-index比较方法" aria-label="Permalink to &quot;z-index比较方法&quot;">​</a></h2><ul><li>首先先看要比较的两个元素是否处于同一个SC中，如果是，谁的层叠等级大，谁在上面；</li><li>如果两个元素不在同一SC中，先比较他们的父SC，当两个元素层叠水平相同、层叠顺序相同时，在 DOM 结构中后面的元素层叠等级在前面元素之上；</li><li>如果一个元素拥有CS则在普通元素之上</li></ul><p>普通的元素层级如图示： <img src="'+n+'" alt="image"></p><h2 id="层叠上下文的产生" tabindex="-1">层叠上下文的产生 <a class="header-anchor" href="#层叠上下文的产生" aria-label="Permalink to &quot;层叠上下文的产生&quot;">​</a></h2><p>层叠上下文的创建有两种情况，一种是强制创建，一种是可选创建。</p><ul><li><p>强制创建：</p><ol><li>文档的根元素，如html；</li><li>position设置为fix、sticky；</li><li>opacity&lt;1；</li><li>mix-blend-mode不为normal；</li><li>isolation=isolate；</li><li>使用了CSS3的动画特性，例如：filter、transform、will-change、clip-path……</li></ol></li><li><p>可选创建：指定z-index不为auto；在此基础上，z-index的</p><ol><li>position设置为relative、absolute；</li><li>flexbox和grid的子元素；</li></ol><blockquote><p>另外，position不为none 的时候同时会创建一个偏移上下文。</p></blockquote></li></ul>',8)]))}const m=e(o,[["render",s]]);export{h as __pageData,m as default};
