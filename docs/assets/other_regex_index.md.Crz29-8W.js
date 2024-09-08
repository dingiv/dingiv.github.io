import{_ as e,c as t,o,a1 as a}from"./chunks/framework.CceCxLSN.js";const _=JSON.parse('{"title":"正则表达式","description":"","frontmatter":{},"headers":[],"relativePath":"other/regex/index.md","filePath":"other/regex/index.md"}'),l={name:"other/regex/index.md"},i=a('<h1 id="正则表达式" tabindex="-1">正则表达式 <a class="header-anchor" href="#正则表达式" aria-label="Permalink to &quot;正则表达式&quot;">​</a></h1><p>专门用于高级字符串匹配算法的DSL。属于通用DSL，几乎被所有的语言所内置支持，其理论到实际应用的发展，经历了半个多世纪的演变，从一种数学工具，逐步成为计算机科学中不可或缺的部分，极大地提高了文本处理和数据验证的效率。</p><h2 id="正则表达式的原理涉及模式匹配-pattern-match-和有限自动机-finite-automaton-fa-理论。" tabindex="-1">正则表达式的原理涉及模式匹配（Pattern Match）和有限自动机（Finite Automaton, FA）理论。 <a class="header-anchor" href="#正则表达式的原理涉及模式匹配-pattern-match-和有限自动机-finite-automaton-fa-理论。" aria-label="Permalink to &quot;正则表达式的原理涉及模式匹配（Pattern Match）和有限自动机（Finite Automaton, FA）理论。&quot;">​</a></h2><p>模式匹配是指使用一定的语法来代表和描述一个字符串的特征。给定一个<strong>模式</strong>，并通过编译，形成一个<strong>有限自动机</strong>程序，然后使用该模式对目标字符串进行匹配。</p><h2 id="模式语法" tabindex="-1">模式语法 <a class="header-anchor" href="#模式语法" aria-label="Permalink to &quot;模式语法&quot;">​</a></h2><ul><li><p>普通字符。26个英文字母、10数字、任意其他语言普通文字……</p></li><li><p>特殊字符。这些特殊的符号在正则表达式中拥有特殊含义，如果需要表达这些符号本身，那么需要加上一个<code>\\</code>进行转义。</p><ul><li><code>.</code>：匹配任意单个字符。</li><li>位置匹配。<code>^</code>匹配字符开头，<code>$</code>匹配字符结尾。</li><li>分组捕获。<code>()</code>。标记一个子表达式的开始和结束位置。子表达式可以获取供以后使用。在JS中，在匹配结果数组中的1号索引开始的位置是子表达式的结果，如果没有分组，那么该值为空。</li><li>数量描述。<code>*</code>：匹配前一个字符或者分组0次或多次，<code>+</code>：1次或多次，<code>?</code>：0次或1次，<code>{0,3}</code>：匹配一个字符0到3次，闭区间，只有一个数字就是匹配明确指明匹配几次，如<code>{4}</code>；对应有这些字符匹配的非贪婪模式：<code>*?</code>:非贪婪模式，尽可能少的匹配，<code>+?</code>:非贪婪模式，尽可能少的匹配。</li><li><code>[]</code>：拾取框。从拾取框中任选一个字符进行匹配，<code>[^]</code>：反向拾取，从不在框中的字母中选一个进行匹配 <code>|</code>：或者，选择左右两个字符中的一个进行匹配。</li><li>非捕获匹配。用于限制单词的边界。 <ol><li>(?:&lt;pattern&gt;)，匹配一个子表达式，但是不捕获匹配的文本</li><li>(?=&lt;pattern&gt;),匹配前一个字符的位置，要求前面的紧挨着的这个位置的字符串能够匹配pattern</li><li>(?&lt;=&lt;pattern&gt;),匹配前一个字符的位置，要求后面的紧挨着的这个位置的字符串能够匹配pattern eg. (?&lt;=abc).+(?=def)，匹配任意以abc开头，def结尾的字符串</li><li>(?!&lt;pattern&gt;),匹配前一个字符的位置，要求前面的紧挨着的这个位置的字符串不能够匹配pattern</li><li>(?&lt;!&lt;pattern&gt;),匹配前一个字符的位置，要求前面的紧挨着的这个位置的字符串不能够匹配pattern</li></ol></li></ul></li><li><p>转义字符。</p><ul><li>\\d：匹配数字</li><li>\\r：匹配回车键、\\n：匹配换行符、\\t：换行符、<code></code>：空格</li><li>\\s：匹配空白字符，包括空格、制表符、换页符等等</li><li>\\b：匹配单词边界。单词字母符号：字母（a-z，A-Z）、数字（0-9）和下划线（_）。\\b是一个非匹配字符，它不匹配实际的字符，但是匹配一个空位置，这个位置必须是两侧有一个是单词字母字符，另一个不是。</li></ul></li></ul><h2 id="ps" tabindex="-1">ps <a class="header-anchor" href="#ps" aria-label="Permalink to &quot;ps&quot;">​</a></h2><p>不同操作系统使用不同的控制字符或组合来表示换行：</p><ul><li>Windows：使用回车和换行的组合 \\r\\n 来表示换行。</li><li>Unix/Linux/macOS：使用换行 \\n 来表示换行。</li><li>旧版 macOS（Mac OS 9 及以前）：使用回车 \\r 来表示换行。</li></ul><p>带&quot;&lt;&quot;的表示反向查找，就是以xxx为左边界，不带&quot;&lt;&quot;的表示正向，以xxx为右边界，&quot;:&quot;的为双向查找</p><h2 id="常用例子" tabindex="-1">常用例子 <a class="header-anchor" href="#常用例子" aria-label="Permalink to &quot;常用例子&quot;">​</a></h2><ol><li>单词，<code>\\w、\\W(不匹配)、\\b(单词边界)</code></li><li>空字符，<code>\\s、\\S</code></li><li>中文，<code>[\\u4e00-\\u9fa5]</code></li><li>小数，<code>(-?\\d+)(\\.\\d+)?</code></li><li>邮箱，<code>\\b[\\w.%+-]+@[\\w.-]+\\.[a-zA-Z]{2,6}\\b</code></li><li>边界，<code>(?:xxx).*(?:xxx)、(?&lt;=xxx).*(?=xxx)、(?&lt;!xxx).*(?!xxx)</code></li></ol>',12),c=[i];function d(r,n,s,x,p,h){return o(),t("div",null,c)}const m=e(l,[["render",d]]);export{_ as __pageData,m as default};
