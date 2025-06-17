# Nginx

## 

## 安装和使用
+ apt 安装
  在 /usr/share/nginx 文件下有默认的 html 文件结构，但是在 ubuntu 上用不了。我们也不一定使用它，我们可以自己自定服务器的目录。配置文件位于**/etc/nginx/**下,配置基础配置即可使用 nginx。

  ```shell
  apt install nginx
  ```
+ 编译安装

```sh
apt install gcc
apt install libpcre3 libpcre3-dev # 正则表达式依赖
apt install openssl libssl-dev # ssl 依赖
apt install zlib1g zlib1g-dev # zlib 依赖

# 下载安装包
wget http://nginx.org/download/nginx-1.20.2.tar.gz
tar -xvf nginx-1.20.2.tar.gz
cd /nginx-1.20.2
# 运行c语言安转脚本
./configure --prefix=/usr/local/nginx
# 编译安装
make
make install
# 进入程序目录中，运行 nginx
cd /sbin
./nginx
```

### nginx 常用命令
```
./nginx
./nginx -s stop //关闭
./nginx -s reload //重新加载
```

### keepalived
高可用软件包，可以与 nginx，redis 等服务进行并用，以防止单点故障，实现高可用效果
配置文件位于 /etc/keepalived/keepalived.conf