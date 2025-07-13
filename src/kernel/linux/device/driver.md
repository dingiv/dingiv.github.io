# 驱动接口
linux 驱动程序的 SPI 接口由各个硬件厂商实现。

> SPI 和 API。在第三方库的调用场景中，一个第三方库往往涉及两个角色，一个是库的使用者/调用者，一个是库的实现者/编写者，而二者之间的交互通过一个双方约定的接口来进行。在这个过程中，如果是由库的实现者来定义接口，那么这个接口就叫 API，如果是由库的使用者来定义，那么这个接口就叫做 SPI。

## 注册



## 文件操作




## 中断处理
驱动框架：
注册设备和驱动：
c

Collapse

Unwrap

Copy
register_chrdev(major, "my_device", &fops);
文件操作：实现 open、read、write、ioctl 等。
中断处理：注册 IRQ 处理程序：
c

Collapse

Unwrap

Copy
request_irq(irq, handler, IRQF_SHARED, "my_device", NULL);