import{_ as s,c as i,o as a,a1 as n}from"./chunks/framework.CceCxLSN.js";const l="/assets/binary-tree.DCl37hVx.png",t="/assets/avl-tree.eKIe-clW.png",h="/assets/rbtree.BA5OAyiR.png",e="/assets/b_.BkhuB2LS.png",A=JSON.parse('{"title":"树","description":"","frontmatter":{},"headers":[],"relativePath":"basic/algo/tree.md","filePath":"basic/algo/tree.md"}'),k={name:"basic/algo/tree.md"},p=n('<h1 id="树" tabindex="-1">树 <a class="header-anchor" href="#树" aria-label="Permalink to &quot;树&quot;">​</a></h1><p>树是一种典型的分支结构，亦可以看做是一种简单的网状结构，这种结构具有单向性，没有环。</p><h2 id="树的特性" tabindex="-1">树的特性 <a class="header-anchor" href="#树的特性" aria-label="Permalink to &quot;树的特性&quot;">​</a></h2><p>树具有以下特性：</p><ol><li>树中任意两个结点之间存在唯一的路径。</li><li>树中没有环。</li><li>树中每个结点只有一个父结点（根结点除外）。</li></ol><h2 id="树的类型" tabindex="-1">树的类型 <a class="header-anchor" href="#树的类型" aria-label="Permalink to &quot;树的类型&quot;">​</a></h2><p>树有很多类型，但是常见的树有：</p><ul><li><strong>二叉树</strong>：每个结点最多有两个子结点的树。但是在平常的使用中，二叉树的可用性质还不够，经常使用的是<strong>二叉搜索树</strong>。每个结点的左子树中的所有结点的值都小于该结点的值。每个结点的右子树中的所有结点的值都大于该结点的值。每个结点的值都是唯一的。 <img src="'+l+'" alt="binary-tree"></li><li><strong>AVL树</strong>：一种自平衡的二叉搜索树。每个结点的左子树和右子树的高度差不能超过1。每个结点的左子树和右子树都是AVL树。根结点的高度是树的高度。 <img src="'+t+'" alt="avl-tree"></li><li><strong>红黑树</strong>：红黑树是一种自平衡的二叉搜索树。每个结点要么是红色，要么是黑色。根结点是黑色。每个叶子结点是黑色。每个红色结点的两个子结点都是黑色。从任意一个结点到其所有叶子结点的路径中，黑色结点的个数相同。为了强化对于结构的要求，有时还会要求，红色节点只能是其父节点的左子节点，称为<strong>左倾红黑树</strong>。 <img src="'+h+'" alt="rbtree"></li><li><strong>B树</strong>：一种多叉搜索树。每个结点最多有m个子结点（m&gt;=2）。每个结点的子结点个数在[m/2]和m之间（根结点除外）。根结点的子结点个数在2和m之间。每个结点的所有子结点的高度相同。每个结点的所有子结点的值都大于该结点的值。每个结点的值都是唯一的。为了能够方便对B树上的结构进行遍历，还出现了<strong>B+树</strong>，它是B树的变种，它的所有的数据均位于叶子结点，当然中间结点也存放着边界节点的数据。 <img src="'+e+`" alt="b+ tree"></li></ul><p>还有一些特殊用途的树：</p><ul><li><strong>Trie树</strong>。也叫<strong>前缀树</strong>、<strong>单词查找树</strong>、<strong>字典树</strong>。</li><li><strong>后缀树</strong>。</li><li><strong>表达式树</strong>。</li><li><strong>Huffman树</strong>。</li><li><strong>线段树</strong>。</li><li><strong>并查集</strong></li></ul><h2 id="树的建构" tabindex="-1">树的建构 <a class="header-anchor" href="#树的建构" aria-label="Permalink to &quot;树的建构&quot;">​</a></h2><p>树的建构可以通过以下步骤来实现：</p><ol><li>选择一个根结点作为树的根。</li><li>将其他结点与根结点连接，形成树的分支结构。</li><li>确保每个结点只有一个父结点，避免形成环。</li></ol><h2 id="树的遍历" tabindex="-1">树的遍历 <a class="header-anchor" href="#树的遍历" aria-label="Permalink to &quot;树的遍历&quot;">​</a></h2><p>树的遍历是指访问树中每个结点的过程。常见的树遍历算法包括：以二叉树为例</p><ul><li>前序遍历：先访问根结点，然后访问左子树，最后访问右子树。<div class="language-js vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">js</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> preorderTraversal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">node</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">  if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (node </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">===</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> null</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">  console.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.value); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 访问根节点</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">  preorderTraversal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.left); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 前序遍历左子树</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">  preorderTraversal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.right); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 前序遍历右子树</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div></li><li>中序遍历：先访问左子树，然后访问根结点，最后访问右子树。<div class="language-js vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">js</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> inorderTraversal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">node</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">  if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (node </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">===</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> null</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">  inorderTraversal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.left); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 中序遍历左子树</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">  console.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.value); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 访问根节点</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">  inorderTraversal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.right); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 中序遍历右子树</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div></li><li>后序遍历：先访问左子树，然后访问右子树，最后访问根结点。<div class="language-js vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">js</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> postorderTraversal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">node</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">  if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (node </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">===</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> null</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">  postorderTraversal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.left); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 后序遍历左子树</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">  postorderTraversal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.right); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 后序遍历右子树</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">  console.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.value); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 访问根节点</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div></li><li>层序遍历：从根结点开始，逐层访问树中的结点<div class="language-js vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">js</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> levelOrderTraversal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">root</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">  if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">root) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">  const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> queue</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [root];</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">  while</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (queue.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">      const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> node</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> queue.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">shift</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 出队</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      console.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.value); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 访问节点</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">      if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (node.left) queue.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">push</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.left); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 左子节点入队</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">      if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (node.right) queue.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">push</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(node.right); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 右子节点入队</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div></li></ul><ul><li>前三种是dfs，后一种是bfs。</li></ul><h2 id="树的操作" tabindex="-1">树的操作 <a class="header-anchor" href="#树的操作" aria-label="Permalink to &quot;树的操作&quot;">​</a></h2><p>树的操作包括：</p><ul><li>插入：向树中插入一个新结点。</li><li>删除：从树中删除一个结点。</li><li>查询：在树中查找一个结点。</li></ul>`,20),r=[p];function d(E,g,o,y,c,F){return a(),i("div",null,r)}const D=s(k,[["render",d]]);export{A as __pageData,D as default};