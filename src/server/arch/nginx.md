# Nginx

## 使用Ubuntu安装nginx
+ apt安装
   在 /usr/share/nginx 文件下有默认的html文件结构，但是在ubuntu上用不了。我们也不一定使用它，我们可以自己自定服务器的目录。配置文件位于**/etc/nginx/**下,配置基础配置即可使用nginx。
   ```shell
   apt install nginx
   ```

+ apt命令
```shell
apt install <packagename>=<version>
apt --purge remove <name>
add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
# 添加阿里云的Ubuntu docker镜像
```

* 配置apt的镜像源
```shell
vim /etc/apt/sources.list

# 添加阿里云的镜像源
deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
```

* apt安装 nginx 常用命令
```shell
apt install nginx //版本太老了
apt --purge autoremove nginx
```


+ 安装包安装

```
apt install gcc
apt install libpcre3 libpcre3-dev #正则表达式依赖
apt install openssl libssl-dev #ssl依赖
apt install zlib1g zlib1g-dev #zlib依赖

# 不可以直接抄要看好路径

//下载安装包，建议手动下载用XTrem上传
wget http://nginx.org/download/nginx-1.20.2.tar.gz
//linux解压命令
tar -xvf nginx-1.20.2.tar.gz
//进入解压后的文件夹
cd /nginx-1.20.2
//运行c语言安转脚本
./configure --prefix=/usr/local/nginx
//编译安装
make
make install
//进入程序目录中，运行nginx，注意这个没有全局注册命令行，要在前面加上./
cd /sbin
./nginx
```

### nginx常用命令

```
./nginx
./nginx -s stop //关闭
./nginx -s reload //重新加载
```

### keepalived

高可用软件包，可以与nginx，redis等服务进行并用，以防止单点故障，实现高可用效果

配置文件位于 /etc/keepalived/keepalived.conf



### Ubuntu防火墙

```
ufw status
ufw enable
ufw disable
ufw default allow
ufw default deny
ufw allow 80
ufw deny 80  #允许/禁止外部访问80端口
ufw allow 80/tcp     #80后面加/tcp或/udp，表示tcp或udp封包
ufw deny smtp        #禁止外部访问smtp服务
ufw allow from 192.168.100.38    #允许此IP访问本机所有端口
ufw allow serviceName
```

### 进程查看

```
ps -ef | grep <进程的名字>
kill <进程id>

```

# Docker

docker常用命令

```
docker run <images>

```

## ubuntu安装docker

```
apt update
apt install ca-certificates curl gnupg lsb-release

#添加阿里云镜像
add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"

#正式安装
apt update
apt install docker-ce docker-ce-cli containerd.io

#要添加docker的阿里云镜像服务
mkdir -p /etc/docker
tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://81il0r78.mirror.aliyuncs.com"]
}
EOF
systemctl daemon-reload
systemctl restart docker
```

```
docker search <image>

docker pull <image>[:<tag>]
```

## portainer 安装

```
docker run -d -p 8000:8000 -p 9000:9000 --name portainer \
--restart=always \
-v /var/run/docker.sock:/var/run/docker.sock \
-v portainer_data:/data portainer/portainer-ce:latest
```

