import{_ as a,o as n,c as e,ah as i}from"./chunks/framework.BwbIerCg.js";const k=JSON.parse('{"title":"网络协议","description":"","frontmatter":{"title":"网络协议","order":10},"headers":[],"relativePath":"basic/network/index.md","filePath":"basic/network/index.md"}'),p={name:"basic/network/index.md"};function t(l,s,c,r,d,o){return n(),e("div",null,[...s[0]||(s[0]=[i(`<h1 id="计算机网络" tabindex="-1">计算机网络 <a class="header-anchor" href="#计算机网络" aria-label="Permalink to “计算机网络”">​</a></h1><p>计算机网络是一组约定成规的协议。正如人与人之间使用一套共同使用的语言来交换信息，计算机之间的交流也需要一定的约定和语言——网络协议。</p><p>计算机网络的内容主要围绕计算网络分层模型来展开，不同的层次解决一部分问题，同一层之间互相通信需要遵循当前层的协议。在 ISO 标准中定义的计算机网络模型分为 7 层。然而在实际的实现中，这个层次结构是不准确的，并且显得比较冗余。</p><div class="language-"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e;" tabindex="0" dir="ltr"><code><span class="line"><span>应用层</span></span>
<span class="line"><span>表示层</span></span>
<span class="line"><span>会话层</span></span>
<span class="line"><span>传输层</span></span>
<span class="line"><span>网络层</span></span>
<span class="line"><span>数据链路层</span></span>
<span class="line"><span>物理层</span></span></code></pre></div><p>因此，更多的时候我们采用的 TCP/IP 4 层模型来描述我们实践工程实践中的情况。其中，普通应用只需实现应用层的协议即可，对于下层的三层由操作系统来实现，不同的操作系统对网络模型的实现是不同的，但对上层不感知。</p><div class="language-"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e;" tabindex="0" dir="ltr"><code><span class="line"><span>应用层</span></span>
<span class="line"><span>-----</span></span>
<span class="line"><span>传输层</span></span>
<span class="line"><span>网络层</span></span>
<span class="line"><span>链路层</span></span></code></pre></div>`,6)])])}const g=a(p,[["render",t]]);export{k as __pageData,g as default};
