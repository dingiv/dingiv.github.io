import{_ as s,c as a,o as i,a1 as n}from"./chunks/framework.CceCxLSN.js";const g=JSON.parse('{"title":"nginx","description":"","frontmatter":{},"headers":[],"relativePath":"front/nginx/index.md","filePath":"front/nginx/index.md"}'),p={name:"front/nginx/index.md"},l=n(`<h1 id="nginx" tabindex="-1">nginx <a class="header-anchor" href="#nginx" aria-label="Permalink to &quot;nginx&quot;">​</a></h1><h2 id="使用ubuntu安装nginx" tabindex="-1">使用Ubuntu安装nginx <a class="header-anchor" href="#使用ubuntu安装nginx" aria-label="Permalink to &quot;使用Ubuntu安装nginx&quot;">​</a></h2><ul><li><p>apt安装 在 /usr/share/nginx 文件下有默认的html文件结构，但是在ubuntu上用不了。我们也不一定使用它，我们可以自己自定服务器的目录。配置文件位于**/etc/nginx/**下,配置基础配置即可使用nginx。</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">apt</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> install</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> nginx</span></span></code></pre></div></li><li><p>apt命令</p></li></ul><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">apt</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> install</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">packagenam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">e</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">versio</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">apt</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --purge</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> remove</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">nam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">e</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">add-apt-repository</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">lsb_release</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -cs</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">) stable&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> </span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 添加阿里云的Ubuntu docker镜像</span></span></code></pre></div><ul><li>配置apt的镜像源</li></ul><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">vim</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> /etc/apt/sources.list</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 添加阿里云的镜像源</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">deb</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://mirrors.aliyun.com/ubuntu/</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> bionic</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> main</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> restricted</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> universe</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> multiverse</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">deb</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://mirrors.aliyun.com/ubuntu/</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> bionic-security</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> main</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> restricted</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> universe</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> multiverse</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">deb</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://mirrors.aliyun.com/ubuntu/</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> bionic-updates</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> main</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> restricted</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> universe</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> multiverse</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">deb</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://mirrors.aliyun.com/ubuntu/</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> bionic-proposed</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> main</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> restricted</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> universe</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> multiverse</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">deb</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://mirrors.aliyun.com/ubuntu/</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> bionic-backports</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> main</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> restricted</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> universe</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> multiverse</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">deb-src</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://mirrors.aliyun.com/ubuntu/</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> bionic</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> main</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> restricted</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> universe</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> multiverse</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">deb-src</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://mirrors.aliyun.com/ubuntu/</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> bionic-security</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> main</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> restricted</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> universe</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> multiverse</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">deb-src</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://mirrors.aliyun.com/ubuntu/</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> bionic-updates</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> main</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> restricted</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> universe</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> multiverse</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">deb-src</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://mirrors.aliyun.com/ubuntu/</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> bionic-proposed</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> main</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> restricted</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> universe</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> multiverse</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">deb-src</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> http://mirrors.aliyun.com/ubuntu/</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> bionic-backports</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> main</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> restricted</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> universe</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> multiverse</span></span></code></pre></div><ul><li>apt安装 nginx 常用命令</li></ul><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">apt</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> install</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> nginx</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> //版本太老了</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">apt</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --purge</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> autoremove</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> nginx</span></span></code></pre></div><ul><li>安装包安装</li></ul><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>apt install gcc</span></span>
<span class="line"><span>apt install libpcre3 libpcre3-dev #正则表达式依赖</span></span>
<span class="line"><span>apt install openssl libssl-dev #ssl依赖</span></span>
<span class="line"><span>apt install zlib1g zlib1g-dev #zlib依赖</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 不可以直接抄要看好路径</span></span>
<span class="line"><span></span></span>
<span class="line"><span>//下载安装包，建议手动下载用XTrem上传</span></span>
<span class="line"><span>wget http://nginx.org/download/nginx-1.20.2.tar.gz</span></span>
<span class="line"><span>//linux解压命令</span></span>
<span class="line"><span>tar -xvf nginx-1.20.2.tar.gz</span></span>
<span class="line"><span>//进入解压后的文件夹</span></span>
<span class="line"><span>cd /nginx-1.20.2</span></span>
<span class="line"><span>//运行c语言安转脚本</span></span>
<span class="line"><span>./configure --prefix=/usr/local/nginx</span></span>
<span class="line"><span>//编译安装</span></span>
<span class="line"><span>make</span></span>
<span class="line"><span>make install</span></span>
<span class="line"><span>//进入程序目录中，运行nginx，注意这个没有全局注册命令行，要在前面加上./</span></span>
<span class="line"><span>cd /sbin</span></span>
<span class="line"><span>./nginx</span></span></code></pre></div><h3 id="nginx常用命令" tabindex="-1">nginx常用命令 <a class="header-anchor" href="#nginx常用命令" aria-label="Permalink to &quot;nginx常用命令&quot;">​</a></h3><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>./nginx</span></span>
<span class="line"><span>./nginx -s stop //关闭</span></span>
<span class="line"><span>./nginx -s reload //重新加载</span></span></code></pre></div><h3 id="keepalived" tabindex="-1">keepalived <a class="header-anchor" href="#keepalived" aria-label="Permalink to &quot;keepalived&quot;">​</a></h3><p>高可用软件包，可以与nginx，redis等服务进行并用，以防止单点故障，实现高可用效果</p><p>配置文件位于 /etc/keepalived/keepalived.conf</p><h3 id="ubuntu防火墙" tabindex="-1">Ubuntu防火墙 <a class="header-anchor" href="#ubuntu防火墙" aria-label="Permalink to &quot;Ubuntu防火墙&quot;">​</a></h3><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>ufw status</span></span>
<span class="line"><span>ufw enable</span></span>
<span class="line"><span>ufw disable</span></span>
<span class="line"><span>ufw default allow </span></span>
<span class="line"><span>ufw default deny</span></span>
<span class="line"><span>ufw allow 80</span></span>
<span class="line"><span>ufw deny 80  #允许/禁止外部访问80端口</span></span>
<span class="line"><span>ufw allow 80/tcp     #80后面加/tcp或/udp，表示tcp或udp封包</span></span>
<span class="line"><span>ufw deny smtp        #禁止外部访问smtp服务</span></span>
<span class="line"><span>ufw allow from 192.168.100.38    #允许此IP访问本机所有端口</span></span>
<span class="line"><span>ufw allow serviceName</span></span></code></pre></div><h3 id="进程查看" tabindex="-1">进程查看 <a class="header-anchor" href="#进程查看" aria-label="Permalink to &quot;进程查看&quot;">​</a></h3><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>ps -ef | grep &lt;进程的名字&gt;</span></span>
<span class="line"><span>kill &lt;进程id&gt;</span></span></code></pre></div><h1 id="docker" tabindex="-1">Docker <a class="header-anchor" href="#docker" aria-label="Permalink to &quot;Docker&quot;">​</a></h1><p>docker常用命令</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>docker run &lt;images&gt;</span></span></code></pre></div><h2 id="ubuntu安装docker" tabindex="-1">ubuntu安装docker <a class="header-anchor" href="#ubuntu安装docker" aria-label="Permalink to &quot;ubuntu安装docker&quot;">​</a></h2><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>apt update</span></span>
<span class="line"><span>apt install ca-certificates curl gnupg lsb-release</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>#添加阿里云镜像</span></span>
<span class="line"><span>add-apt-repository &quot;deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>#正式安装</span></span>
<span class="line"><span>apt update</span></span>
<span class="line"><span>apt install docker-ce docker-ce-cli containerd.io</span></span>
<span class="line"><span></span></span>
<span class="line"><span>#要添加docker的阿里云镜像服务</span></span>
<span class="line"><span>mkdir -p /etc/docker</span></span>
<span class="line"><span>tee /etc/docker/daemon.json &lt;&lt;-&#39;EOF&#39;</span></span>
<span class="line"><span>{</span></span>
<span class="line"><span>  &quot;registry-mirrors&quot;: [&quot;https://81il0r78.mirror.aliyuncs.com&quot;]</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>EOF</span></span>
<span class="line"><span>systemctl daemon-reload</span></span>
<span class="line"><span>systemctl restart docker</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>docker search &lt;image&gt;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>docker pull &lt;image&gt;[:&lt;tag&gt;]</span></span></code></pre></div><h2 id="portainer-安装" tabindex="-1">portainer 安装 <a class="header-anchor" href="#portainer-安装" aria-label="Permalink to &quot;portainer 安装&quot;">​</a></h2><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>docker run -d -p 8000:8000 -p 9000:9000 --name portainer \\</span></span>
<span class="line"><span>--restart=always \\</span></span>
<span class="line"><span>-v /var/run/docker.sock:/var/run/docker.sock \\</span></span>
<span class="line"><span>-v portainer_data:/data portainer/portainer-ce:latest</span></span></code></pre></div>`,27),e=[l];function t(h,k,r,d,c,F){return i(),a("div",null,e)}const u=s(p,[["render",t]]);export{g as __pageData,u as default};
