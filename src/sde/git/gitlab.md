# Gitlab
gitlab 为普通企业提供了大多数开发所需要的功能，集成了代码托管和 CICD 于一体。

## 代码托管
使用 git 进行代码管理，可以基于分支、tag 进行推送和构建

## gitlab 流水线
集成 gitlab 流水线，根据代码的推送自动安排 gitlab runner 进行流水线任务，gitlab runner 需要部署在一台或者多台与 gitlab 自身不在同一台的机器上。gitlab runner 部署时，需要向 gitlab 服务器进行注册，从而获悉 gitlab 分发的流水线任务。

runner 可以使用多种执行器，最多使用的 shell 和 docker，一般使用 docker 即可，可以快速方便地安装应用，减少环境问题。
