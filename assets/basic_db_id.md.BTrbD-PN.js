import{_ as i,c as a,o as l,ae as n}from"./chunks/framework.Dh1jimFm.js";const g=JSON.parse('{"title":"索引","description":"","frontmatter":{},"headers":[],"relativePath":"basic/db/id.md","filePath":"basic/db/id.md"}'),h={name:"basic/db/id.md"};function t(e,s,k,p,d,r){return l(),a("div",null,s[0]||(s[0]=[n(`<h1 id="索引" tabindex="-1">索引 <a class="header-anchor" href="#索引" aria-label="Permalink to &quot;索引&quot;">​</a></h1><p>索引是提高数据库查询性能的重要工具。它们通过创建额外的数据结构来加速数据检索，但同时也需要权衡存储空间和更新性能。</p><h2 id="索引类型" tabindex="-1">索引类型 <a class="header-anchor" href="#索引类型" aria-label="Permalink to &quot;索引类型&quot;">​</a></h2><h3 id="_1-主键索引-primary-key-index" tabindex="-1">1. 主键索引（Primary Key Index） <a class="header-anchor" href="#_1-主键索引-primary-key-index" aria-label="Permalink to &quot;1. 主键索引（Primary Key Index）&quot;">​</a></h3><ul><li>唯一标识表中的每一行</li><li>自动创建</li><li>不允许NULL值</li></ul><div class="language-sql vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">sql</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">CREATE</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> TABLE</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> users</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    user_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">INT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> PRIMARY KEY</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,  </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">-- 自动创建主键索引</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    username </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">VARCHAR</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    email </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">VARCHAR</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span></code></pre></div><h3 id="_2-唯一索引-unique-index" tabindex="-1">2. 唯一索引（Unique Index） <a class="header-anchor" href="#_2-唯一索引-unique-index" aria-label="Permalink to &quot;2. 唯一索引（Unique Index）&quot;">​</a></h3><ul><li>确保列值的唯一性</li><li>允许NULL值</li><li>可以包含多个列</li></ul><div class="language-sql vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">sql</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">CREATE</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> TABLE</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> products</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    product_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">INT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> PRIMARY KEY</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    product_code </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">VARCHAR</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    UNIQUE</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> INDEX</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> idx_product_code (product_code)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span></code></pre></div><h3 id="_3-普通索引-regular-index" tabindex="-1">3. 普通索引（Regular Index） <a class="header-anchor" href="#_3-普通索引-regular-index" aria-label="Permalink to &quot;3. 普通索引（Regular Index）&quot;">​</a></h3><ul><li>最基本的索引类型</li><li>不保证唯一性</li><li>用于加速查询</li></ul><div class="language-sql vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">sql</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">CREATE</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> TABLE</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> orders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    order_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">INT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> PRIMARY KEY</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    customer_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">INT</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    order_date </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">DATE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    INDEX</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> idx_customer (customer_id),</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    INDEX</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> idx_date (order_date)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span></code></pre></div><h3 id="_4-复合索引-composite-index" tabindex="-1">4. 复合索引（Composite Index） <a class="header-anchor" href="#_4-复合索引-composite-index" aria-label="Permalink to &quot;4. 复合索引（Composite Index）&quot;">​</a></h3><ul><li>包含多个列的索引</li><li>列的顺序很重要</li><li>支持前缀匹配</li></ul><div class="language-sql vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">sql</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">CREATE</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> TABLE</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> sales</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    sale_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">INT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> PRIMARY KEY</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    product_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">INT</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    sale_date </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">DATE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    amount </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">DECIMAL</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    INDEX</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> idx_product_date (product_id, sale_date)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span></code></pre></div><h3 id="_5-全文索引-full-text-index" tabindex="-1">5. 全文索引（Full-Text Index） <a class="header-anchor" href="#_5-全文索引-full-text-index" aria-label="Permalink to &quot;5. 全文索引（Full-Text Index）&quot;">​</a></h3><ul><li>用于文本搜索</li><li>支持模糊匹配</li><li>适用于大文本字段</li></ul><div class="language-sql vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">sql</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">CREATE</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> TABLE</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> articles</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    article_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">INT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> PRIMARY KEY</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    title </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">VARCHAR</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">200</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    content </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">TEXT</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    FULLTEXT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> INDEX</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> idx_content (content)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span></code></pre></div><h2 id="索引实现方式" tabindex="-1">索引实现方式 <a class="header-anchor" href="#索引实现方式" aria-label="Permalink to &quot;索引实现方式&quot;">​</a></h2><h3 id="_1-b-tree索引" tabindex="-1">1. B-Tree索引 <a class="header-anchor" href="#_1-b-tree索引" aria-label="Permalink to &quot;1. B-Tree索引&quot;">​</a></h3><ul><li>最常用的索引类型</li><li>支持等值查询和范围查询</li><li>适用于大多数场景</li></ul><h3 id="_2-hash索引" tabindex="-1">2. Hash索引 <a class="header-anchor" href="#_2-hash索引" aria-label="Permalink to &quot;2. Hash索引&quot;">​</a></h3><ul><li>只支持等值查询</li><li>不支持范围查询</li><li>查询性能极快</li></ul><h3 id="_3-r-tree索引" tabindex="-1">3. R-Tree索引 <a class="header-anchor" href="#_3-r-tree索引" aria-label="Permalink to &quot;3. R-Tree索引&quot;">​</a></h3><ul><li>用于空间数据</li><li>支持地理信息查询</li><li>适用于GIS应用</li></ul><h2 id="索引优化策略" tabindex="-1">索引优化策略 <a class="header-anchor" href="#索引优化策略" aria-label="Permalink to &quot;索引优化策略&quot;">​</a></h2><h3 id="_1-选择合适的列" tabindex="-1">1. 选择合适的列 <a class="header-anchor" href="#_1-选择合适的列" aria-label="Permalink to &quot;1. 选择合适的列&quot;">​</a></h3><ul><li>高选择性的列</li><li>频繁用于查询的列</li><li>用于排序和分组的列</li></ul><h3 id="_2-避免过度索引" tabindex="-1">2. 避免过度索引 <a class="header-anchor" href="#_2-避免过度索引" aria-label="Permalink to &quot;2. 避免过度索引&quot;">​</a></h3><ul><li>每个索引都需要维护成本</li><li>过多的索引会降低写入性能</li><li>定期评估索引使用情况</li></ul><h3 id="_3-复合索引设计" tabindex="-1">3. 复合索引设计 <a class="header-anchor" href="#_3-复合索引设计" aria-label="Permalink to &quot;3. 复合索引设计&quot;">​</a></h3><ul><li>最左前缀原则</li><li>考虑查询模式</li><li>避免冗余索引</li></ul><div class="language-sql vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">sql</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">-- 好的复合索引设计</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">CREATE</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> TABLE</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> orders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    order_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">INT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> PRIMARY KEY</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    customer_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">INT</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    status</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> VARCHAR</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    order_date </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">DATE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    INDEX</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> idx_customer_status_date (customer_id, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">status</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, order_date)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">-- 可以支持以下查询</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">SELECT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> FROM</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> orders </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">WHERE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> customer_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">SELECT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> FROM</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> orders </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">WHERE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> customer_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> AND</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> status</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &#39;completed&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">SELECT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> FROM</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> orders </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">WHERE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> customer_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> AND</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> status</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &#39;completed&#39;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> AND</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> order_date </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &#39;2023-01-01&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span></code></pre></div><h2 id="索引维护" tabindex="-1">索引维护 <a class="header-anchor" href="#索引维护" aria-label="Permalink to &quot;索引维护&quot;">​</a></h2><h3 id="_1-定期重建索引" tabindex="-1">1. 定期重建索引 <a class="header-anchor" href="#_1-定期重建索引" aria-label="Permalink to &quot;1. 定期重建索引&quot;">​</a></h3><div class="language-sql vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">sql</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">-- MySQL</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">ALTER</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> TABLE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> table_name ENGINE</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InnoDB;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">-- PostgreSQL</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">REINDEX </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">TABLE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> table_name;</span></span></code></pre></div><h3 id="_2-监控索引使用" tabindex="-1">2. 监控索引使用 <a class="header-anchor" href="#_2-监控索引使用" aria-label="Permalink to &quot;2. 监控索引使用&quot;">​</a></h3><div class="language-sql vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">sql</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">-- MySQL</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">EXPLAIN </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">SELECT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> FROM</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> table_name </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">WHERE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> column </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> value</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">-- PostgreSQL</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">EXPLAIN ANALYZE </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">SELECT</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> FROM</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> table_name </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">WHERE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> column </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> value</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span></code></pre></div><h3 id="_3-删除无用索引" tabindex="-1">3. 删除无用索引 <a class="header-anchor" href="#_3-删除无用索引" aria-label="Permalink to &quot;3. 删除无用索引&quot;">​</a></h3><div class="language-sql vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">sql</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">DROP</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> INDEX</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> index_name </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">ON</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> table_name;</span></span></code></pre></div><h2 id="常见问题与解决方案" tabindex="-1">常见问题与解决方案 <a class="header-anchor" href="#常见问题与解决方案" aria-label="Permalink to &quot;常见问题与解决方案&quot;">​</a></h2><h3 id="_1-索引失效" tabindex="-1">1. 索引失效 <a class="header-anchor" href="#_1-索引失效" aria-label="Permalink to &quot;1. 索引失效&quot;">​</a></h3><ul><li>使用函数或运算符</li><li>类型转换</li><li>使用OR条件</li></ul><h3 id="_2-索引选择" tabindex="-1">2. 索引选择 <a class="header-anchor" href="#_2-索引选择" aria-label="Permalink to &quot;2. 索引选择&quot;">​</a></h3><ul><li>考虑查询频率</li><li>考虑数据分布</li><li>考虑更新频率</li></ul><h3 id="_3-性能优化" tabindex="-1">3. 性能优化 <a class="header-anchor" href="#_3-性能优化" aria-label="Permalink to &quot;3. 性能优化&quot;">​</a></h3><ul><li>使用覆盖索引</li><li>避免索引列上的计算</li><li>合理使用索引提示</li></ul><h2 id="最佳实践" tabindex="-1">最佳实践 <a class="header-anchor" href="#最佳实践" aria-label="Permalink to &quot;最佳实践&quot;">​</a></h2><ol><li><p><strong>索引设计原则</strong></p><ul><li>为常用查询创建索引</li><li>保持索引简洁</li><li>定期维护索引</li></ul></li><li><p><strong>监控与调优</strong></p><ul><li>监控索引使用情况</li><li>分析慢查询</li><li>定期优化索引</li></ul></li><li><p><strong>文档维护</strong></p><ul><li>记录索引设计决策</li><li>说明索引用途</li><li>更新索引变更记录</li></ul></li></ol>`,49)]))}const o=i(h,[["render",t]]);export{g as __pageData,o as default};
