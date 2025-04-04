import{_ as e,c as t,o as r,ae as i}from"./chunks/framework.Dh1jimFm.js";const l="/assets/image.6k2T9StR.png",m=JSON.parse('{"title":"linux 内核","description":"","frontmatter":{},"headers":[],"relativePath":"kernel/linux/index.md","filePath":"kernel/linux/index.md"}'),n={name:"kernel/linux/index.md"};function o(h,a,d,s,c,u){return r(),t("div",null,a[0]||(a[0]=[i('<h1 id="linux-内核" tabindex="-1">linux 内核 <a class="header-anchor" href="#linux-内核" aria-label="Permalink to &quot;linux 内核&quot;">​</a></h1><p><img src="'+l+'" alt="alt text"></p><h2 id="内核体系结构" tabindex="-1">内核体系结构 <a class="header-anchor" href="#内核体系结构" aria-label="Permalink to &quot;内核体系结构&quot;">​</a></h2><h3 id="内核源码结构" tabindex="-1">内核源码结构 <a class="header-anchor" href="#内核源码结构" aria-label="Permalink to &quot;内核源码结构&quot;">​</a></h3><h3 id="中断机制" tabindex="-1">中断机制 <a class="header-anchor" href="#中断机制" aria-label="Permalink to &quot;中断机制&quot;">​</a></h3><p>中断主要有软件中断和硬件中断两种。</p><h2 id="内核引导" tabindex="-1">内核引导 <a class="header-anchor" href="#内核引导" aria-label="Permalink to &quot;内核引导&quot;">​</a></h2><h2 id="进程管理" tabindex="-1">进程管理 <a class="header-anchor" href="#进程管理" aria-label="Permalink to &quot;进程管理&quot;">​</a></h2><p>进程是操作系统对一个独立执行的任务的管理单元。其代表了一个应用层程序的实例化对象，执行一个应用层的自定义程序。在程序的执行中，系统将为进程分配 CPU 资源，并为程序许诺一个独立而广阔的虚拟内存空间，供程序使用。</p><h3 id="进程内存空间" tabindex="-1">进程内存空间 <a class="header-anchor" href="#进程内存空间" aria-label="Permalink to &quot;进程内存空间&quot;">​</a></h3><p>进程的内存是操作系统许诺的一个虚假空间，分为两个部分，一部分是内核空间，一部分是用户空间，内核空间预先存放了操作系统为用户提供过的一些数据和信息，用户的代码和数据存放在用户空间，用户可以在自己的空间中执行程序，同时也可以调用内核空间中的函数，从而进入内核态，使用内核提供的 API 执行更加底层的操作，包括：IO 操作，硬件控制，系统控制等等。</p><h2 id="内存管理" tabindex="-1">内存管理 <a class="header-anchor" href="#内存管理" aria-label="Permalink to &quot;内存管理&quot;">​</a></h2><h2 id="文件系统" tabindex="-1">文件系统 <a class="header-anchor" href="#文件系统" aria-label="Permalink to &quot;文件系统&quot;">​</a></h2><h2 id="驱动系统" tabindex="-1">驱动系统 <a class="header-anchor" href="#驱动系统" aria-label="Permalink to &quot;驱动系统&quot;">​</a></h2><h2 id="网络协议栈" tabindex="-1">网络协议栈 <a class="header-anchor" href="#网络协议栈" aria-label="Permalink to &quot;网络协议栈&quot;">​</a></h2>',15)]))}const _=e(n,[["render",o]]);export{m as __pageData,_ as default};
