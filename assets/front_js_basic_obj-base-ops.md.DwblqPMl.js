import{_ as e,c as l,o as a,ae as i}from"./chunks/framework.Dh1jimFm.js";const j=JSON.parse('{"title":"js 对象基本操作","description":"","frontmatter":{},"headers":[],"relativePath":"front/js/basic/obj-base-ops.md","filePath":"front/js/basic/obj-base-ops.md"}'),p={name:"front/js/basic/obj-base-ops.md"};function r(o,t,s,d,b,n){return a(),l("div",null,t[0]||(t[0]=[i('<h1 id="js-对象基本操作" tabindex="-1">js 对象基本操作 <a class="header-anchor" href="#js-对象基本操作" aria-label="Permalink to &quot;js 对象基本操作&quot;">​</a></h1><p>在 js 中我们可以对一个对象进行一些语言层面上的基本操作，这些操作是潜藏在语言表面语法下的底层执行逻辑。</p><h2 id="属性操作" tabindex="-1">属性操作 <a class="header-anchor" href="#属性操作" aria-label="Permalink to &quot;属性操作&quot;">​</a></h2><p>属性操作就是指对象上携带的数值，js 中的对象从表现上就是其他语言中的 HashMap 数据结构，用于快速获取散列数据。一个对象上可以有多个属性，每个属性需要使用要给 key 来索引，key 只能是 string 或者 symbol 这两种基本类型（注意数字不是，数字会被转化为 string）。一个属性往往有以下的定义和描述符。</p><table tabindex="0"><thead><tr><th>描述符</th><th>定义</th></tr></thead><tbody><tr><td>value</td><td>代表了该属性的值</td></tr><tr><td>writeale</td><td>代表了属性当前是否能被“修改”</td></tr><tr><td>enumeratable</td><td>代表了该属性当前是否能够被“枚举”</td></tr><tr><td>configurable</td><td>代笔了当前属性是否能够被“配置”</td></tr></tbody></table><ul><li><p>属性读取：obj.key</p></li><li><p>属性赋值：obj.key = value</p></li><li><p>属性定义：Object.defineProperty</p></li><li><p>属性删除：delete obj.key</p></li><li><p>检查属性：&quot;key&quot; in obj</p></li><li><p>获取描述符：Object.getOwnPropertyDescriptor</p></li><li><p>冻结对象：Object.freeze</p></li><li><p>密封对象：Object.seal</p></li><li><p>获取原型：Object.getPrototypeOf</p></li><li><p>设置原型：Object.setPrototypeOf</p></li><li><p>获取属性名：Object.keys / Object.getOwnPropertyNames</p></li><li><p>获取符号属性：Object.getOwnPropertySymbols</p></li><li><p>[[GET]] 该操作用于在对象的一个属性上，并获取一个值，一个 slot 对外，，然后再用来获取对象的 slot 上的值。</p></li><li><p>[[SET]] 该操作作用于对象 slot 上的值，如果该 slot 上没有属性定义则施加定义，如果有定义则修改 value 值。</p></li><li><p>[[DELETE]] 该操作作用于</p></li><li><p>[[Define Property]]</p></li><li><p>[[ENUMERATE]] 该操作代表了一个属性是否能够被枚举，</p></li></ul>',6)]))}const _=e(p,[["render",r]]);export{j as __pageData,_ as default};
