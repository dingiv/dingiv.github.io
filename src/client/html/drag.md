# HTML 拖放 API
在拖放的过程中涉及两个元素，一个是拖动元素（或者说拖动源），一个是放置元素（或放置目标）。

- 拖动元素是被拖动的元素，在这个元素身上可以提供原始的被拖动的数据，而被拖动的元素也有可能是从非浏览器中来的，它可能是从操作系统中拖动过来的，所以拖动元素不一定存在，所以考虑这点，更准确的说法应该是拖动源。当一个 html 元素被设置了 draggable 后，那么它就可以被拖动。
- 放置元素是拖动的过程中，在拖动路径上所遇到的其他元素，不能是拖动元素自己。一个元素想要成为放置元素，需要为改元素绑定 ondrop 事件处理器。同样的，放置的位置也可能不在浏览器中，而在其他的应用里，为了能让其他的应用也能识别拖动的内容，可以按照一定的约定进行拖动数据的定义。

## 拖动元素
当一个 html 元素被设置了 draggable 后，那么它就可以被拖动。

### 事件
拖动元素：如果一个拖动源在浏览器，那么我们可以在拖动元素的身上可以提供一些事件，这些事件如下：

- dragstart：当用户开始拖动元素的时候触发一次
- drag：当用户开始拖动元素之后每隔一段时间触发一次，只要用户不松手，就重复触发
- dragend：当用户结束拖动元素之后触发一次，也就是用户松手的时候

放置元素：如果一个元素被拖动路径所经过或者在次结束拖动，那么会有如下事件：

- dragenter：当拖动元素进入放置元素的时候触发一次
- dragover: 当拖动元素进入放置元素的过程中，如果拖动不结束，就会重复触发
- drop：当拖动元素在改放置元素的身上结束拖动时触发，用于处理拖动之后的数据接收问题

### 定义拖拽数据
拖拽数据是由数据源所提供的数据，所以定义拖拽数据在 dragstart 事件中进行定义。

```javascript
function onDragStart(ev) {
  // 添加拖拽数据
  ev.dataTransfer.setData("text/plain", ev.target.innerText);
  ev.dataTransfer.setData("text/html", ev.target.outerHTML);
  ev.dataTransfer.setData("text/uri-list", ev.target.ownerDocument.location.href);
  // 在dataTransfer上可以定义多个数据，数据的结构是一个平对象，string类型的键和任意类型的值
  // 在定义的时候，一般会使用mime类型的名称作为键值，这种作法在各种程序中都很通用，可以保证当拖动目标不在浏览器中时的正常工作。
}
```

### 定义拖拽图像或效果
拖拽效果涉及了两个人之间的交互。

#### 拖拽元素和放置元素之间的交互
在拖拽元素的 dragstart 上设置拖动的效果类型

```javascript
function onDragStart(ev) {
  event.dataTransfer.effectAllowed = "move";
  event.dataTransfer.dropEffect = "move";
}
```

在放置元素的 dragover 中阻止默认行为

```javascript
function onDragOver(ev) {
  ev.preventDefault();
}
```
