# 存储


## 浏览器存储

浏览器提供了多种存储机制，用于在客户端保存数据。

### Cookie

- 最早的浏览器存储机制
- 容量限制：通常为4KB
- 会随HTTP请求一起发送
- 可设置过期时间、域名范围等
- 通过设置HttpOnly和Secure提高安全性

### LocalStorage

- 永久存储机制，除非手动清除
- 容量限制：通常为5MB
- 不会随HTTP请求发送
- 仅支持字符串存储
- 同源访问限制

### SessionStorage

- 会话级存储，关闭标签页后清除
- 容量限制：通常为5MB
- 不会随HTTP请求发送
- 仅支持字符串存储
- 同源且同标签页访问限制

### IndexedDB

- 结构化存储机制，支持复杂数据类型
- 容量大，通常>50MB
- 异步API，不阻塞主线程
- 支持事务和索引
- 同源访问限制

### Web Storage API

- localStorage和sessionStorage的统一接口
- 提供setItem、getItem、removeItem、clear等方法
- 支持storage事件监听变化
