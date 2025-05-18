# web component
原生的 web 元素组件化能力，使用全局 API `customElements.define()` 来进行注册，单个元素的声明使用 class 语法，继承 HTMLElement 进行书写，该类实例化之后，成为 DOM 元素实例，需要在元素的身上定义几个关键的 hooks 函数，然后来对元素的逻辑进行管理。

## hooks
```ts
[propKey:string] : any // 任意属性，可以暴露在元素实例的数据
constructor()  :  void
// 返回这个元素的可观测动态属性，这些属性将会被观察，并在发生变化时调用回调函数attributeChangedCallback
static get observedAttributes() :  string[]
attributeChangedCallback(property, oldValue, newValue) : void
connectedCallback() :void  // 在这个元素被挂载到document时调用
disconnectedCallback() : void  // 在这个元素从document卸载时调用
adoptedCallback() : void
```

## 影子节点的 css 解决方案
- 局部声明，在影子节点内部声明的 style 标签只会作用于该影子节点，连同类都没有办法影响，因此这种样式将会被重复声明；
- 局部引用，使用 link 标签引入 css 文件，这种方式一个页面只会申请一次网络请求，同时能够使用 css 文件隔离逻辑
- CSS 变量，影子节点的能够访问自定义元素的 CSS 变量，这是一种外部 CSS 变量向影子节点传递的简陋方式
- 样式穿透，在组件的内部使用自定义元素的类型，加上::part('tag')伪元素选择器可以选择影子节点内部 part 属性等于'tag'的元素，另一方面，在影子节点组件的内部使用::slotted('selector')将可以在组件内修改未来插槽中填充的元素的样式
- 样式表构造，使用 API CSSStyleSheet，手动创建一个 CSSOM 树，创建好后添加到 adoptedStyleSheets 属性，该属性位于 document 上和 shadowRoot 上，即可改变 CSS 属性，该方式性能最好，并且可以实现多个影子节点实例之间共用 CSS
