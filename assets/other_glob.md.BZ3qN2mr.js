import{_ as o,c,a0 as d,o as a}from"./chunks/framework.p2VkXzrt.js";const g=JSON.parse('{"title":"glob","description":"","frontmatter":{},"headers":[],"relativePath":"other/glob.md","filePath":"other/glob.md"}'),t={name:"other/glob.md"};function s(l,e,r,i,n,h){return a(),c("div",null,e[0]||(e[0]=[d('<h1 id="glob" tabindex="-1">glob <a class="header-anchor" href="#glob" aria-label="Permalink to &quot;glob&quot;">​</a></h1><p>glob是广泛内置于各个操作系统的，专门用于文件路径匹配的匹配语法，它比正则表达式更加简单和轻量，更加适用于文件系统路径的匹配。</p><ul><li>基础语法：<code>/</code>、<code>*</code>、<code>?</code>、<code>[]</code></li><li>拓展语法：<code>**</code>、<code>{}</code>、<code>()</code></li></ul><h2 id="分隔符和片段" tabindex="-1">分隔符和片段 <a class="header-anchor" href="#分隔符和片段" aria-label="Permalink to &quot;分隔符和片段&quot;">​</a></h2><p><strong>概念</strong>：分隔符是<code>/</code>，通过<code>split(&#39;/&#39;)</code> 得到的数组每一项是片段。</p><p><strong>示例：</strong></p><ul><li><code>src/index.js</code> 有两个片段，分别是 <code>src</code> 和 <code>index.js</code></li><li><code>src/**/*.js</code> 有三个片段，分别是 <code>src</code>、<code>**</code> 和 <code>*.js</code></li></ul><h2 id="单个星号" tabindex="-1">单个星号 <a class="header-anchor" href="#单个星号" aria-label="Permalink to &quot;单个星号&quot;">​</a></h2><p><strong>概念</strong>：单个星号<code>*</code> 用于匹配单个片段中的零个或多个字符。</p><p><strong>示例</strong>：</p><ul><li><code>src/*.js</code> 表示 <code>src</code> 目录下所有以 <code>js</code> 结尾的文件，但是不能匹配 <code>src</code> 子目录中的文件，例如 <code>src/login/login.js</code></li><li><code>/home/*/.bashrc</code> 匹配所有用户的 <code>.bashrc</code> 文件</li></ul><p>需要注意的是，<code>*</code> 不能匹配分隔符<code>/</code>，也就是说不能跨片段匹配字符。</p><h2 id="问号" tabindex="-1">问号 <a class="header-anchor" href="#问号" aria-label="Permalink to &quot;问号&quot;">​</a></h2><p><strong>概念</strong>：问号 <code>?</code>匹配单个片段中的单个字符。</p><p><strong>示例</strong>：</p><ul><li><code>test/?at.js</code> 匹配形如 <code>test/cat.js</code>、<code>test/bat.js</code> 等所有3个字符且后两位是 <code>at</code> 的 js 文件，但是不能匹配 <code>test/flat.js</code></li><li><code>src/index.??</code> 匹配 src 目录下以 <code>index</code> 打头，后缀名是两个字符的文件，例如可以匹配 <code>src/index.js</code> 和 <code>src/index.md</code>，但不能匹配 <code>src/index.jsx</code></li></ul><h2 id="中括号" tabindex="-1">中括号 <a class="header-anchor" href="#中括号" aria-label="Permalink to &quot;中括号&quot;">​</a></h2><p>**概念:**同样是匹配单个片段中的单个字符，但是字符集只能从括号内选择，如果字符集内有<code>-</code>，表示范围。</p><p><strong>示例：</strong></p><ul><li><code>test/[bc]at.js</code> 只能匹配<code>test/bat.js</code> 和 <code>test/cat.js</code></li><li><code>test/[c-f]at.js</code> 能匹配 <code>test/cat.js</code>、<code>test/dat.js</code>、<code>test/eat.js</code> 和<code>test/fat.js</code></li></ul><h2 id="惊叹号" tabindex="-1">惊叹号 <a class="header-anchor" href="#惊叹号" aria-label="Permalink to &quot;惊叹号&quot;">​</a></h2><p>**概念：**表示取反，即排除那些去掉惊叹号之后能够匹配到的文件。 <strong>示例：</strong></p><ul><li><code>test/[!bc]at.js</code>不能匹配 <code>test/bat.js</code> 和 <code>test/cat.js</code>，但是可以匹配 <code>test/fat.js</code></li><li><code>!test/tmp/**&#39;</code> 排除 <code>test/tmp</code> 目录下的所有目录和文件</li></ul><h2 id="扩展语法" tabindex="-1">扩展语法 <a class="header-anchor" href="#扩展语法" aria-label="Permalink to &quot;扩展语法&quot;">​</a></h2><p>基础语法非常简单好记，但是功能非常局限，为了丰富 glob 的功能，衍生了下面三种扩展语法：</p><h2 id="两个星号" tabindex="-1">两个星号 <a class="header-anchor" href="#两个星号" aria-label="Permalink to &quot;两个星号&quot;">​</a></h2><p>**概念：**两个星号<code>**</code> 可以跨片段匹配零个或多个字符，也就是说<code>**</code>是递归匹配所有文件和目录的，如果后面有分隔符，即 <code>**/</code> 的话，则表示只递归匹配所有目录（不含隐藏目录）。</p><p><strong>示例：</strong></p><ul><li><code>/var/log/**</code> 匹配 <code>/var/log</code> 目录下所有文件和文件夹，以及文件夹里面所有子文件和子文件夹</li><li><code>/var/log/**/*.log</code> 匹配 <code>/var/log</code> 及其子目录下的所有以 <code>.log</code> 结尾的文件</li><li><code>/home/*/.ssh/**/*.key</code> 匹配所有用户的 <code>.ssh</code> 目录及其子目录内的以<code>.key</code> 结尾的文件</li></ul><h2 id="大括号" tabindex="-1">大括号 <a class="header-anchor" href="#大括号" aria-label="Permalink to &quot;大括号&quot;">​</a></h2><p>**概念：**匹配大括号内的所有模式，模式之间用逗号进行分隔，支持大括号嵌套，支持用<code>..</code> 匹配连续的字符，即<code>{start..end}</code> 语法。</p><p><strong>示例：</strong></p><ul><li><code>a.{png,jp{,e}g}</code> 匹配 <code>a.png</code>、<code>a.jpg</code>、<code>a.jpeg</code></li><li><code>{a..c}{1..2}</code> 匹配 <code>a1 a2 b1 b2 c1 c2</code></li></ul><p>注意：<code>{}</code> 与 <code>[]</code> 有一个很重要的区别：如果匹配的文件不存在，<code>[]</code>会失去模式的功能，变成一个单纯的字符串，而 <code>{}</code> 依然可以展开。</p><h2 id="小括号" tabindex="-1">小括号 <a class="header-anchor" href="#小括号" aria-label="Permalink to &quot;小括号&quot;">​</a></h2><p>**概念：**小括号必须跟在 <code>?</code>、<code>*</code>、<code>+</code>、<code>@</code>、<code>!</code> 后面使用，且小括号里面的内容是一组以 <code>|</code> 分隔符的模式集合，例如：<code>abc|a?c|ac*</code>。</p>',36)]))}const b=o(t,[["render",s]]);export{g as __pageData,b as default};
