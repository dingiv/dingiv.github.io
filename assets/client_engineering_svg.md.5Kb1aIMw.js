import{_ as i,c as a,o as t,ae as e}from"./chunks/framework.Dh1jimFm.js";const o=JSON.parse('{"title":"svg","description":"","frontmatter":{},"headers":[],"relativePath":"client/engineering/svg.md","filePath":"client/engineering/svg.md"}'),n={name:"client/engineering/svg.md"};function l(h,s,p,g,k,r){return t(),a("div",null,s[0]||(s[0]=[e(`<h1 id="svg" tabindex="-1">svg <a class="header-anchor" href="#svg" aria-label="Permalink to &quot;svg&quot;">​</a></h1><p>svg雪碧图，将多个svg合并到一个文件当中，然后在使用时通过svg的<code>&lt;use /&gt;</code>标签进行引用，实现减少网络请求的目的。可以使用专门的构建工具来进行操作。该方案提供了解决在一个项目中管理大量本地svg图标的方法。</p><h2 id="vite-plugin-svg-icons" tabindex="-1">vite-plugin-svg-icons <a class="header-anchor" href="#vite-plugin-svg-icons" aria-label="Permalink to &quot;vite-plugin-svg-icons&quot;">​</a></h2><p>自动将项目中某个目录下的所有svg文件进行打包，并将svg雪碧图内容注入到<strong>index.html</strong>中，然后再使用时通过如下语法使用</p><div class="language-html vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">html</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&lt;</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">svg</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> aria-hidden</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;true&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">   &lt;</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">use</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> xlink:href</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;#targetId&quot;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> fill</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;red&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> /&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&lt;/</span><span style="--shiki-light:#22863A;--shiki-dark:#85E89D;">svg</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;</span></span></code></pre></div><p>需要在<strong>main.js</strong>中引入必要逻辑</p><div class="language-js vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">js</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &#39;virtual:svg-icons-register&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span></code></pre></div>`,7)]))}const E=i(n,[["render",l]]);export{o as __pageData,E as default};
