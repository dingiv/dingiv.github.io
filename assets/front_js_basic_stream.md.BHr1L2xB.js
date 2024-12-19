import{_ as a,c as r,a0 as t,o as s}from"./chunks/framework.p2VkXzrt.js";const i="/assets/stream.DRXwVqa-.png",h=JSON.parse('{"title":"Stream API","description":"","frontmatter":{},"headers":[],"relativePath":"front/js/basic/stream.md","filePath":"front/js/basic/stream.md"}'),l={name:"front/js/basic/stream.md"};function d(n,e,o,p,m,b){return s(),r("div",null,e[0]||(e[0]=[t('<h1 id="stream-api" tabindex="-1">Stream API <a class="header-anchor" href="#stream-api" aria-label="Permalink to &quot;Stream API&quot;">​</a></h1><h2 id="readablestream" tabindex="-1">ReadableStream <a class="header-anchor" href="#readablestream" aria-label="Permalink to &quot;ReadableStream&quot;">​</a></h2><p>获得生产者 （1）new ReadableStream({ start?(ctrl){ ctrl.enqueue(data) }, pull?(ctrl){} }) （2）从其他API获得</p><p>使用生产者 （1）对接至WriteableStream，rs.pipeTo(ws) （2）调用reader，手动消费， reader=rs.getReader() reader.read().then((res)=&gt;process(res))</p><h2 id="writeablestream-消费者" tabindex="-1">WriteableStream：消费者 <a class="header-anchor" href="#writeablestream-消费者" aria-label="Permalink to &quot;WriteableStream：消费者&quot;">​</a></h2><p>获得消费者 （1）new WriteableStream({ write(data){} })</p><p>使用消费者 （1）让Readable对接 （2）手动调用writer， writer=ws.getWriter() writer.ready.write()</p><h2 id="node-stream-readable" tabindex="-1">node:stream.Readable <a class="header-anchor" href="#node-stream-readable" aria-label="Permalink to &quot;node:stream.Readable&quot;">​</a></h2><p>获得生产者 （1）new Readable({ read(){ this.push(chunk) } }) （2）从其他API获得，文件流，标准流，网络流，压缩流 （3）从可迭代对象获取，Readable.from(iterable)</p><p>使用生产者 （1）对接至Writeable，rs.pipe(ws) （2）监听data事件 （3）监听readable事件，并且手动调用rs.read()方法消费值，这种情况下，流式静止的，必须循环调用read() （4）使用node:stream/promise的pipeline()，流编排函数</p><p>生产者有静止状态、流动状态和关闭状态，初始为静止状态，在调用了pipe和被监听了data事件后变为流动状态</p><h2 id="node-stream-writeable" tabindex="-1">node:stream.Writeable <a class="header-anchor" href="#node-stream-writeable" aria-label="Permalink to &quot;node:stream.Writeable&quot;">​</a></h2><p>获得消费者 （1）new Writeable({ write(chunk,encoding,cb){ process(chunk) cb() } }) （2）从其他API获得，文件流，标准流，网络流，压缩流</p><p>使用消费者 （1）让Readable对接 （2）监听drain事件 （3）监听writeable事件，手动调用ws.write()生产值 （4）pipeline</p><p>消费者有静止状态、流动状态和关闭状态，初始为静止状态，被对接pipe和被监听了drain事件后变为流动状态 <img src="'+i+'" alt="alt text"></p>',15)]))}const u=a(l,[["render",d]]);export{h as __pageData,u as default};