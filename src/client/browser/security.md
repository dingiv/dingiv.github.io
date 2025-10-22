# 安全
浏览器安全是 Web 开发和用户隐私保护的关键领域，涉及多种机制和技术，用于防止恶意代码执行、数据泄露和各类攻击。

## 浏览器安全模型

### 同源策略（Same-Origin Policy）

同源策略是浏览器安全的基石，限制一个源（origin）的文档或脚本如何与另一个源的资源进行交互。

- **同源定义**：协议、域名和端口号都必须相同
- **限制范围**：

  - DOM 访问受限
  - Cookie、LocalStorage 和 IndexedDB 访问受限
  - XMLHttpRequest 和 Fetch API 请求受限

- **示例**：
  ```
  // 这些URL与 https://example.com/page.html 比较
  https://example.com/other.html       // 同源 - 只有路径不同
  https://sub.example.com/page.html    // 不同源 - 子域名不同
  http://example.com/page.html         // 不同源 - 协议不同
  https://example.com:8080/page.html   // 不同源 - 端口不同
  ```

### 沙箱隔离（Sandboxing）

浏览器通过沙箱机制隔离渲染进程，限制其访问系统资源的能力。

- **进程级沙箱**：将渲染进程与系统和其他进程隔离
- **iframe 沙箱**：通过 sandbox 属性限制 iframe 内容的权限
- **站点隔离**：将不同站点放在不同的渲染进程中

### 内容安全策略（Content Security Policy，CSP）

CSP 是一种额外的安全层，用于防止 XSS 和数据注入攻击。

- **实现方式**：通过 HTTP 头部或 meta 标签配置
- **策略示例**：

  ```html
  <!-- 通过meta标签配置CSP -->
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' https://trusted.cdn.com" />
  ```

- **常用指令**：
  - `default-src`：默认资源加载策略
  - `script-src`：JavaScript 资源加载策略
  - `style-src`：CSS 资源加载策略
  - `img-src`：图片资源加载策略
  - `connect-src`：连接请求策略（如 XHR、WebSockets）

## 常见安全威胁与防护

### 跨站脚本攻击（Cross-Site Scripting，XSS）

攻击者通过在网页中注入恶意脚本，窃取用户数据或执行恶意操作。

- **类型**：

  1. **存储型 XSS**：恶意脚本存储在服务器数据库中
  2. **反射型 XSS**：恶意脚本通过 URL 参数反射到页面
  3. **DOM 型 XSS**：恶意脚本通过 DOM 操作插入页面

- **防护措施**：

  1. 输入验证和输出编码
  2. 实施内容安全策略（CSP）
  3. 使用 HttpOnly 和 Secure 标志保护 Cookie
  4. 使用现代框架的自动转义机制

- **防护代码示例**：

  ```javascript
  // 不安全的代码
  element.innerHTML = userInput;

  // 安全的代码
  element.textContent = userInput;
  // 或使用DOMPurify等库
  element.innerHTML = DOMPurify.sanitize(userInput);
  ```

### 跨站请求伪造（Cross-Site Request Forgery，CSRF）

攻击者诱导用户在已认证的网站上执行非预期操作。

- **攻击原理**：利用浏览器会自动发送目标站点 Cookie 的特性
- **防护措施**：

  1. 使用 CSRF 令牌
  2. 验证 Referer/Origin 头
  3. 使用 SameSite Cookie 属性
  4. 实施双重认证

- **防护代码示例**：
  ```html
  <!-- 表单中添加CSRF令牌 -->
  <form action="/transfer" method="post">
    <input type="hidden" name="csrf_token" value="random_token_value" />
    <!-- 其他表单字段 -->
  </form>
  ```

### 点击劫持（Clickjacking）

攻击者通过透明 iframe 覆盖在目标网站上，诱导用户点击看不见的元素。

- **防护措施**：

  1. 设置 X-Frame-Options 头
  2. 使用 CSP 的 frame-ancestors 指令
  3. 使用 JavaScript 帧破解防护

- **防护代码示例**：
  ```html
  <!-- 使用JavaScript防护 -->
  <style>
    body {
      display: none;
    }
  </style>
  <script>
    if (self === top) {
      document.body.style.display = "block";
    } else {
      top.location = self.location;
    }
  </script>
  ```

### 中间人攻击（Man-in-the-Middle，MITM）

攻击者拦截并可能修改浏览器与服务器之间的通信。

- **防护措施**：
  1. 使用 HTTPS 协议
  2. 实施 HTTP 严格传输安全（HSTS）
  3. 使用公钥固定（Public Key Pinning）
  4. 检查证书有效性

### 浏览器扩展安全

浏览器扩展可能访问敏感 API 和数据，需要特别关注其安全性。

- **安全注意事项**：
  1. 最小权限原则
  2. 内容安全策略
  3. 安全的消息传递
  4. 定期审核扩展更新

## 跨域解决方案

### 跨域资源共享（Cross-Origin Resource Sharing，CORS）

CORS 是一种标准机制，允许服务器声明哪些源可以访问其资源。

- **简单请求和预检请求**：

  1. 简单请求：直接发送，带 Origin 头
  2. 预检请求：先发送 OPTIONS 请求，获取权限

- **服务器配置示例**：
  ```
  Access-Control-Allow-Origin: https://example.com
  Access-Control-Allow-Methods: GET, POST, PUT
  Access-Control-Allow-Headers: Content-Type, Authorization
  Access-Control-Allow-Credentials: true
  ```

### JSONP（JSON with Padding）

利用 script 标签不受同源策略限制的特性实现跨域。

- **实现示例**：

  ```javascript
  function handleResponse(data) {
    console.log(data);
  }

  // 创建script标签加载跨域数据
  const script = document.createElement("script");
  script.src = "https://api.example.com/data?callback=handleResponse";
  document.body.appendChild(script);
  ```

- **安全风险**：可能导致 XSS 攻击，只应与可信来源使用

### 代理服务器

通过同源的服务器中转请求，规避同源策略限制。

- **实现方式**：
  1. 服务端代理
  2. 开发环境代理（如 webpack-dev-server）

### PostMessage API

允许不同源的窗口之间安全通信。

- **安全使用示例**：

  ```javascript
  // 发送消息
  targetWindow.postMessage(message, "https://trusted-receiver.com");

  // 接收消息
  window.addEventListener("message", (event) => {
    // 验证消息源
    if (event.origin !== "https://trusted-sender.com") return;

    // 处理消息
    console.log(event.data);
  });
  ```

## 现代浏览器安全特性

### 子资源完整性（Subresource Integrity，SRI）

通过加密哈希验证加载的资源是否被篡改。

- **实现示例**：
  ```html
  <script src="https://cdn.example.com/library.js" integrity="sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8wC" crossorigin="anonymous"></script>
  ```

### HTTPS 升级

- **HTTP 严格传输安全（HSTS）**：强制使用 HTTPS 连接

  ```
  Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
  ```

- **升级不安全请求**：通过 CSP 自动升级 HTTP 请求到 HTTPS
  ```
  Content-Security-Policy: upgrade-insecure-requests
  ```

### 特性策略（Feature Policy）/ 权限策略（Permissions Policy）

限制网站可以使用的浏览器功能和 API。

- **实现示例**：
  ```
  Permissions-Policy: camera=(), microphone=(self "https://trusted.com")
  ```

### 隐私保护功能

- **第三方 Cookie 限制**
- **指纹识别防护**
- **加密 SNI**
- **COOP/COEP/CORP 跨源隔离策略**

## 浏览器安全开发最佳实践
1. 始终使用 HTTPS：保护数据传输安全
2. 实施内容安全策略：防止 XSS 攻击
3. 使用现代安全 HTTP 头：HSTS, X-Content-Type-Options 等
4. 安全管理 Cookie：使用 HttpOnly, Secure 和 SameSite 属性
5. 输入验证与输出编码：防止注入攻击
6. 实施 CSRF 保护：防止跨站请求伪造
7. 定期更新依赖项：修复已知安全漏洞
8. 使用安全的会话管理：防止会话劫持
9. 实施安全的跨域通信：正确配置 CORS
10. 进行安全测试和审计：发现和修复安全漏洞
