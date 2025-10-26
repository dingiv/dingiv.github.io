# Web 优化


## 浏览器性能优化

### 关键渲染路径优化

1. **减少关键资源**
   - 减少阻塞渲染的CSS和JavaScript
   - 内联关键CSS
   - 异步加载非关键JavaScript

2. **减少资源大小**
   - 压缩HTML、CSS、JavaScript
   - 使用Gzip/Brotli压缩
   - 图片优化

3. **减少请求数量**
   - 合并CSS和JavaScript文件
   - 使用CSS Sprite
   - 使用字体图标或SVG

4. **优化加载顺序**
   - CSS放在head中
   - JavaScript放在body底部
   - 使用async/defer属性

### 渲染性能优化

1. **减少回流（Reflow）**
   - 批量修改DOM
   - 使用document fragment
   - 避免频繁读取布局信息

2. **减少重绘（Repaint）**
   - 使用CSS transform和opacity代替修改位置和可见性
   - 使用will-change提示浏览器
   - 合理使用GPU加速

3. **帧率优化**
   - 使用requestAnimationFrame
   - 避免长任务阻塞主线程
   - 使用Web Workers分担计算密集型任务

### 网络优化

1. **资源预加载**
   - preload关键资源
   - prefetch可能需要的资源
   - preconnect提前建立连接

2. **HTTP优化**
   - 使用HTTP/2多路复用
   - 使用HTTP/3 QUIC协议
   - 合理设置缓存策略

3. **CDN加速**
   - 使用CDN分发静态资源
   - 选择离用户最近的节点
   - 使用多CDN提供冗余

## 浏览器开发者工具

现代浏览器提供了强大的开发者工具，帮助开发者调试和优化Web应用。

### Elements（元素）

- 检查和修改DOM结构
- 实时编辑CSS样式
- 查看事件监听器
- 断点调试DOM变化

### Console（控制台）

- 输出调试信息
- 执行JavaScript代码
- 查看错误和警告
- 使用console API

### Network（网络）

- 监控网络请求
- 分析资源加载时间
- 查看HTTP头信息
- 模拟网络条件

### Performance（性能）

- 记录和分析页面性能
- 查看CPU和内存使用情况
- 识别性能瓶颈
- 分析帧率和渲染时间

### Memory（内存）

- 分析内存使用情况
- 查找内存泄漏
- 查看内存分配
- 生成堆快照

### Application（应用）

- 管理本地存储
- 查看和修改Cookie
- 管理Service Worker
- 查看Web应用清单

### Security（安全）

- 检查HTTPS证书
- 识别混合内容问题
- 查看内容安全策略
- 分析安全漏洞

## 浏览器兼容性

### 检测和解决兼容性问题

1. **特性检测**
   - 检测浏览器是否支持特定功能
   - 根据支持情况提供不同实现
   - 避免使用用户代理检测

2. **Polyfill**
   - 为旧浏览器提供新功能的模拟实现
   - 只在需要时加载
   - 使用现代工具自动添加

3. **渐进增强**
   - 从基本功能开始构建
   - 逐步添加高级特性
   - 确保核心功能在所有浏览器中可用

4. **工具支持**
   - Babel转译现代JavaScript
   - PostCSS处理CSS兼容性
   - Autoprefixer自动添加厂商前缀
   - Browserslist定义目标浏览器

### 常见兼容性资源

- **Can I use**：查询特性兼容性数据
- **MDN Web Docs**：详细的API兼容性信息
- **Modernizr**：特性检测库
- **core-js**：JavaScript标准库polyfill

