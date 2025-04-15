# svg
svg雪碧图，将多个svg合并到一个文件当中，然后在使用时通过svg的`<use />`标签进行引用，实现减少网络请求的目的。可以使用专门的构建工具来进行操作。该方案提供了解决在一个项目中管理大量本地svg图标的方法。

## vite-plugin-svg-icons
自动将项目中某个目录下的所有svg文件进行打包，并将svg雪碧图内容注入到**index.html**中，然后再使用时通过如下语法使用
```html
<svg aria-hidden="true">
   <use xlink:href="#targetId" fill="red" />
</svg>
```
需要在**main.js**中引入必要逻辑
```js
import 'virtual:svg-icons-register';
```