# Gitlab
gitlab为普通企业提供了大多数开发所需要的功能，集成了代码托管和CICD于一体。

## 代码托管
使用git进行代码管理，可以基于分支、tag进行推送和构建

## gitlab流水线
集成gitlab流水线，根据代码的推送自动安排gitlab runner进行流水线任务，gitlab runner需要部署在一台或者多台与gitlab自身不在同一台的机器上。gitlab runner部署时，需要向gitlab服务器进行注册，从而获悉gitlab分发的流水线任务。

runner可以使用多种执行器，最多使用的shell和docker，一般使用docker即可，可以快速方便地安装应用，减少环境问题。
