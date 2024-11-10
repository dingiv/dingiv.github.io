# io
在前端应用中，涉及的io对象主要有如下内容，在实际的使用中，这些API往往以命令式的代码进行操作，这与函数编程所使用的声明式编程风格不符，因此需要使用高级框架对其进行封装。同时，这些io操作一般会涉及异步操作，因此，为了让API变得优雅，必须使用响应式编程的思想来优化API风格，提高框架的使用体验。
1. 有持久化能力的Web API，例如：
   + Cookies
   + SessionStorage
   + LocalStorage
   + Indexed DB
   + History API
   + Clipboard API
   + File API
2. Web Worker API
3. Http和WebSockets
纯粹的手动请求还远远不能满足实际的开发需要，一些高级特性也需要支持
   + 声明式编程风格
   + 响应式能力、响应式数据封装
   + io抽象、数据预处理、校验、拼装、多请求封装
   + 缓存、持久化
   + 限流、防抖
   + Mock
   + 请求拦截、重定向、逻辑代理
   + SSR支持
   + UI框架无关、IO目标无关或多重支持