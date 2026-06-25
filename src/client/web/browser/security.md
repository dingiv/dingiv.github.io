# 浏览器安全
浏览器作为用户与互联网之间的中介，需要在不破坏 Web 开放性的前提下尽可能保护用户安全。这篇文章梳理浏览器安全的核心机制——同源策略、常见攻击手段的原理与防护、跨域解决方案，以及现代浏览器提供的安全特性。

## 同源策略
同源策略（Same-Origin Policy）是浏览器安全的基石。所谓"同源"，要求协议、域名和端口号三者完全相同。`https://example.com/page1.html` 和 `https://example.com/page2.html` 同源（只有路径不同），但和 `https://sub.example.com/page.html` 不同源（子域名不同），和 `http://example.com/page.html` 也不同源（协议不同）。

同源策略的限制体现在三个层面：DOM 层面禁止跨源访问其他窗口的 DOM 节点；数据存储层面禁止跨源读写 Cookie、LocalStorage、IndexedDB；网络层面限制跨源的 XMLHttpRequest 和 Fetch 请求（但浏览器允许跨源发送请求，只是阻止 JavaScript 读取响应内容）。这些限制使得恶意网站无法通过脚本读取用户在其他网站上的数据。

浏览器通过沙箱（Sandbox）机制在进程层面提供额外的隔离。现代 Chrome 的站点隔离（Site Isolation）策略会将不同站点的页面分配到不同的渲染进程中，防止 Spectre 等 CPU 侧信道攻击利用共享进程内存读取其他站点的敏感数据。`<iframe>` 的 sandbox 属性允许开发者对嵌入的第三方内容施加细粒度的权限控制——可以限制脚本执行、表单提交、弹窗等能力，只开放必要的权限。

## 跨站脚本攻击（XSS）
XSS 是最常见的 Web 攻击之一，核心原理是让浏览器执行了攻击者注入的恶意脚本。根据注入方式的不同，XSS 分为三种类型。

存储型 XSS 中，恶意脚本被持久化存储在服务器端（如数据库中的评论内容），当其他用户浏览包含恶意内容的页面时脚本被执行。这种类型的危害最大，因为影响范围不限于攻击者自己。反射型 XSS 中，恶意脚本通过 URL 参数传递到服务器，服务器将其嵌入响应页面返回给用户，攻击者需要通过社工手段诱导用户点击带恶意参数的链接。DOM 型 XSS 不经过服务器，完全在客户端发生——JavaScript 直接读取不可信的输入（如 URL 的 hash 部分）并将其插入到 DOM 中。

XSS 防护的核心原则是：永远不要将不可信的数据直接插入到 HTML 中。`element.innerHTML = userInput` 是最典型的危险操作，应该用 `element.textContent` 替代，或者使用 DOMPurify 等库对 HTML 内容进行消毒处理后再插入。现代前端框架（React、Vue、Angular）默认对插值内容进行转义，大幅降低了 XSS 的风险，但在使用 `dangerouslySetInnerHTML`（React）或 `v-html`（Vue）时仍然需要注意。

```
// 危险：直接插入不可信内容
element.innerHTML = userInput

// 安全：文本节点不会执行 HTML
element.textContent = userInput

// 如必须插入 HTML，使用 DOMPurify 消毒
element.innerHTML = DOMPurify.sanitize(userInput)
```

内容安全策略（CSP）是防护 XSS 的另一道防线。CSP 通过 HTTP 响应头（或 `<meta>` 标签）声明页面允许加载哪些来源的资源，浏览器会拦截违反策略的请求。例如设置 `script-src 'self'` 后，浏览器会阻止所有来自外部域名的 JavaScript 执行，即使攻击者成功注入了 `<script src="https://evil.com/attack.js">`，浏览器也会拒绝加载。

```
Content-Security-Policy: default-src 'self'; script-src 'self' https://trusted.cdn.com; style-src 'self' 'unsafe-inline'
```

CSP 的 `default-src` 是默认策略，`script-src`、`style-src`、`img-src` 等分别控制不同类型资源的加载来源。实际部署时，可以先开启 `Content-Security-Policy-Report-Only` 模式，让浏览器只报告违规而不拦截，观察一段时间确认无误后再切换为强制模式。需要注意的是，`'unsafe-inline'` 会削弱 CSP 对内联脚本和内联样式的防护能力，`'unsafe-eval'` 则允许 `eval()` 等动态代码执行，两者都应尽量避免。

## 跨站请求伪造（CSRF）
CSRF 利用浏览器会自动携带目标站点 Cookie 的特性，诱导用户在已认证的状态下向目标站点发送非预期请求。例如用户已登录银行网站，此时访问了一个恶意页面，恶意页面中隐藏着一个指向银行转账接口的表单，浏览器会自动携带银行的 Cookie 完成转账——用户并不知情。

CSRF 和 XSS 的本质区别在于：XSS 是让浏览器执行攻击者的脚本，CSRF 是让浏览器以用户身份发送攻击者的请求。防护 CSRF 的核心思路是让服务器能够区分"合法请求"和"伪造请求"。

SameSite Cookie 属性是最现代的 CSRF 防护手段。设置 `SameSite=Strict` 后，Cookie 只在同站请求中发送，跨站请求完全不会携带；`SameSite=Lax` 允许从外部链接点击进入时携带 Cookie（这样用户从搜索引擎点击进入时仍然保持登录状态），但阻止跨站的 POST 请求。大多数现代浏览器默认将未指定 SameSite 属性的 Cookie 视为 `Lax`。

在 SameSite 之前，CSRF Token 是主流的防护方案。服务器为每个表单或每个会话生成一个随机令牌，嵌入表单的隐藏字段中。提交表单时服务器校验这个令牌——攻击者的恶意页面无法获取到目标站点的 CSRF Token（受同源策略保护），所以伪造的请求不会携带有效的 Token。这个方案至今仍在使用，特别是需要兼容旧浏览器的场景。

```html
<form action="/transfer" method="post">
  <input type="hidden" name="csrf_token" value="a1b2c3d4">
  <!-- 其他表单字段 -->
</form>
```

此外，服务器可以通过检查 `Origin` 或 `Referer` 请求头来判断请求来源是否合法。但 `Referer` 可能被浏览器或用户代理屏蔽，`Origin` 在某些场景下（如 302 跨域重定向）也可能丢失，所以这些手段只能作为辅助防护而非唯一防线。

## 跨域资源共享（CORS）
同源策略限制了跨源的网络请求，但现代 Web 应用经常需要调用第三方 API、加载 CDN 资源，浏览器通过 CORS（Cross-Origin Resource Sharing）机制提供了受控的跨源访问方式。

CORS 将跨源请求分为两类：简单请求和预检请求。简单请求满足一定条件（GET/HEAD/POST 方法、Content-Type 为 text/plain 等），浏览器直接发送并在请求头中附带 `Origin` 字段，服务器通过响应头 `Access-Control-Allow-Origin` 声明是否允许。预检请求（Preflight）在请求不满足简单请求条件时触发（如使用 PUT/DELETE 方法、Content-Type 为 application/json 等），浏览器先发送一个 OPTIONS 请求询问服务器是否允许，收到允许的响应后才发送实际请求。

```
// 预检请求的响应头示例
Access-Control-Allow-Origin: https://example.com
Access-Control-Allow-Methods: GET, POST, PUT, DELETE
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Allow-Credentials: true
Access-Control-Max-Age: 86400
```

`Access-Control-Allow-Credentials: true` 允许跨源请求携带 Cookie，此时 `Access-Control-Allow-Origin` 不能设为 `*`，必须指定具体的源。`Access-Control-Max-Age` 指定预检结果的有效期，避免每次复杂请求都发送 OPTIONS 预检。

在 CORS 出现之前，JSONP 是实现跨源数据获取的主要手段。它利用 `<script>` 标签不受同源策略限制的特性，将数据包装在一个回调函数中返回。JSONP 只支持 GET 请求，且存在安全风险（任意域名的脚本都能在页面中执行），现在已经不推荐使用，但在对接一些老旧的第三方 API 时可能还会遇到。

对于无法修改响应头的场景，可以通过同源的服务端代理转发请求来绕过浏览器的同源限制。前端请求自己的服务器，服务器再请求目标 API 并将结果返回给前端。开发环境中 webpack-dev-server 的 proxy 配置就是这种方案的便捷实现。

## 其他安全特性

### 子资源完整性（SRI）
从 CDN 加载第三方 JavaScript 库时，如果 CDN 被入侵或响应被中间人篡改，恶意脚本就会在用户浏览器中执行。SRI 通过在 `<script>` 标签上声明资源内容的加密哈希值，让浏览器在加载资源后校验完整性，不匹配则拒绝执行。

```html
<script src="https://cdn.example.com/library.js"
  integrity="sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8wC"
  crossorigin="anonymous"></script>
```

`crossorigin="anonymous"` 是 SRI 的前置条件——只有设置了 `crossorigin` 属性，浏览器才会在加载跨源资源时获取足够的错误信息来执行完整性校验。

### HTTPS 与 HSTS
HTTPS 通过 TLS 加密浏览器与服务器之间的通信内容，防止中间人窃听和篡改。HSTS（HTTP Strict Transport Security）通过响应头 `Strict-Transport-Security: max-age=31536000; includeSubDomains; preload` 告诉浏览器在指定时间内对该站点只使用 HTTPS 连接，即使用户在地址栏输入了 `http://`，浏览器也会自动升级为 HTTPS。加入 HSTS Preload List 后，浏览器内置的预加载列表会在首次访问前就强制使用 HTTPS，彻底消除了首次 HTTP 请求被劫持的可能。

### 权限策略（Permissions Policy）
权限策略（原名 Feature Policy）允许服务器声明页面可以使用哪些浏览器功能和 API。例如 `Permissions-Policy: camera=(), microphone=(self)` 表示禁止所有来源使用摄像头，麦克风只允许同源页面使用。这可以防止恶意第三方脚本在用户不知情的情况下调用敏感 API。

### 跨源隔离
COOP（Cross-Origin-Opener-Policy）、COEP（Cross-Origin-Embedder-Policy）和 CORP（Cross-Origin-Resource-Policy）三个响应头组合使用可以实现跨源隔离。跨源隔离后的页面可以使用 `SharedArrayBuffer` 和高精度定时器等高性能 API，同时防止跨源攻击者通过 `window.opener` 等途径干扰页面。这三个头的作用分别是：COOP 防止跨源页面通过 `window.opener` 访问当前窗口，COEP 要求页面加载的所有跨源资源都显式授权（通过 CORS 或 CORP），CORP 限制跨源请求如何使用响应。

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

开启跨源隔离后，页面中的所有跨源资源（图片、脚本、样式表等）都需要正确配置 CORS 或 CORP，否则会被浏览器拦截，这也是为什么大多数网站还没有启用跨源隔离的原因——迁移成本较高，需要确保所有第三方资源都能正确处理跨源请求。
