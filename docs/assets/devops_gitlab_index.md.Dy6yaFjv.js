import{_ as a,c as t,o as e,a1 as i}from"./chunks/framework.CceCxLSN.js";const g=JSON.parse('{"title":"Gitlab","description":"","frontmatter":{},"headers":[],"relativePath":"devops/gitlab/index.md","filePath":"devops/gitlab/index.md"}'),r={name:"devops/gitlab/index.md"},l=i('<h1 id="gitlab" tabindex="-1">Gitlab <a class="header-anchor" href="#gitlab" aria-label="Permalink to &quot;Gitlab&quot;">​</a></h1><p>gitlab为普通企业提供了大多数开发所需要的功能，集成了代码托管和CICD于一体。</p><h2 id="代码托管" tabindex="-1">代码托管 <a class="header-anchor" href="#代码托管" aria-label="Permalink to &quot;代码托管&quot;">​</a></h2><p>使用git进行代码管理，可以基于分支、tag进行推送和构建</p><h2 id="gitlab流水线" tabindex="-1">gitlab流水线 <a class="header-anchor" href="#gitlab流水线" aria-label="Permalink to &quot;gitlab流水线&quot;">​</a></h2><p>集成gitlab流水线，根据代码的推送自动安排gitlab runner进行流水线任务，gitlab runner需要部署在一台或者多台与gitlab自身不在同一台的机器上。gitlab runner部署时，需要向gitlab服务器进行注册，从而获悉gitlab分发的流水线任务。</p><p>runner可以使用多种执行器，最多使用的shell和docker，一般使用docker即可，可以快速方便地安装应用，减少环境问题。</p>',7),n=[l];function o(d,s,c,_,b,h){return e(),t("div",null,n)}const u=a(r,[["render",o]]);export{g as __pageData,u as default};
