# 伪文件系统
操作系统内核


设备模型可以挂载到 `/sys/devices` 目录下


在用户层使用 udev 服务监听内核的设备抽象层的设备事件 uevent，然后做出响应，维护临时文件系统 /dev 目录，向用户空间暴露设备文件。设备树

