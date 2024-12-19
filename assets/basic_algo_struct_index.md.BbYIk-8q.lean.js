import{_ as l,c as i,a0 as e,o as r}from"./chunks/framework.p2VkXzrt.js";const b=JSON.parse('{"title":"数据结构","description":"","frontmatter":{"title":"数据结构","order":1},"headers":[],"relativePath":"basic/algo/struct/index.md","filePath":"basic/algo/struct/index.md"}'),t={name:"basic/algo/struct/index.md"};function o(n,a,d,h,s,u){return r(),i("div",null,a[0]||(a[0]=[e('<h1 id="数据结构" tabindex="-1">数据结构 <a class="header-anchor" href="#数据结构" aria-label="Permalink to &quot;数据结构&quot;">​</a></h1><p>常见的数据结构可以按照逻辑结构、物理结构、功能用途来进行分类。</p><h2 id="按物理结构分" tabindex="-1">按物理结构分 <a class="header-anchor" href="#按物理结构分" aria-label="Permalink to &quot;按物理结构分&quot;">​</a></h2><ul><li>连续结构/数组/Array，在物理内存上，其数据的存储是物理连续的。</li><li>非连续结构/链表/Link+Node，在物理内存上，其数据的存储是物理不连续的。</li></ul><h2 id="按逻辑结构分" tabindex="-1">按逻辑结构分 <a class="header-anchor" href="#按逻辑结构分" aria-label="Permalink to &quot;按逻辑结构分&quot;">​</a></h2><h3 id="线性结构" tabindex="-1">线性结构 <a class="header-anchor" href="#线性结构" aria-label="Permalink to &quot;线性结构&quot;">​</a></h3><ul><li>列表</li><li>栈</li><li>队列</li></ul><h3 id="分支结构-半平面结构" tabindex="-1">分支结构（半平面结构） <a class="header-anchor" href="#分支结构-半平面结构" aria-label="Permalink to &quot;分支结构（半平面结构）&quot;">​</a></h3><ul><li>哈希表</li><li>堆</li><li>红黑树</li><li>跳表</li><li>B树</li></ul><h3 id="网状结构-平面结构" tabindex="-1">网状结构（平面结构） <a class="header-anchor" href="#网状结构-平面结构" aria-label="Permalink to &quot;网状结构（平面结构）&quot;">​</a></h3><p>复杂的网状结构，其构成有两个要素，一个是节点，一个是边，图可以用来抽象物体之间的关系。</p><h4 id="图的性质" tabindex="-1">图的性质 <a class="header-anchor" href="#图的性质" aria-label="Permalink to &quot;图的性质&quot;">​</a></h4><p>图具有很多性质和概念，不同的概念强调了图的某个方面的有用特性，是将图用于生产和解决实际问题的重要前提。</p><ul><li>密度。根据图中节点之间的关系连接数量，可以将图分为稀疏图和稠密图。</li><li>方向性。根据图中的节点之间的关系的方向性，可以将图分为有向图和无向图</li><li>权重。为不同节点之间的关系添加一个代表重要性的具体数值，将图分为了加权图和无权图。</li><li>环。在图中，节点之间的关系连线形成了环路，用以描述图是否是有环图或者无环图。</li></ul><h4 id="图的遍历" tabindex="-1">图的遍历 <a class="header-anchor" href="#图的遍历" aria-label="Permalink to &quot;图的遍历&quot;">​</a></h4><ul><li>dfs。深度优先。在遍历图上的节点的时候，总是先处理完当前的节点上的数据，然后再进入下一跳节点，处理下一跳数据。</li><li>bfs。广度优先。在遍历图上的节点的时候，总是进入下一个节点，直到所有的下一跳节点都遍历完成之后，再处理本节点上的数据。</li></ul><h4 id="图的应用" tabindex="-1">图的应用 <a class="header-anchor" href="#图的应用" aria-label="Permalink to &quot;图的应用&quot;">​</a></h4><ul><li>可达性：路径存在问题</li><li>拓扑排序：调度问题</li><li>强连通分量-&gt;Kosaraju算法</li><li>最小生成树-&gt;Prim算法/Kruskal算法</li><li>最短路径问题-&gt;dijkstra算法/Bellmon-Ford算法/Floyd算法</li></ul>',18)]))}const m=l(t,[["render",o]]);export{b as __pageData,m as default};