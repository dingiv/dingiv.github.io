# 驱动接口
Linux 驱动程序是操作系统与硬件之间的桥梁。它们通过一套标准接口（SPI，Service Provider Interface）与内核交互，由硬件厂商或开发者实现，保证了内核对各种硬件的统一管理和访问。

> 在软件开发中，API（Application Programming Interface）通常由库的实现者定义，供调用者使用；而 SPI（Service Provider Interface）则由库的使用者定义，要求第三方实现者按照规范实现接口。Linux 驱动开发中的 SPI 由内核定义，硬件厂商实现，保证了驱动的可插拔和标准化。
> 
> 驱动也是内核软件开发中，最为常见的需求，硬件厂商可以为自己的硬件实现驱动，内核开发者也可以通过

驱动根据设备类型进行分类，字符设备驱动、块设备驱动、网络设备，大多数时候开发者需要开发的是字符设备驱动。因此，本文将主要介绍字符设备驱动的编写。
| 设备类型 | 字符设备驱动                           | 块设备驱动                 | 网络设备驱动                 |
| -------- | -------------------------------------- | -------------------------- | ---------------------------- |
| 典型设备 | 串口、按键、LED 等                     | 硬盘、SD 卡                | 网卡                         |
| 数据模型 | 顺序读写，适合初学者和大多数的定制硬件 | 支持随机访问和缓冲         | 面向数据包，与协议栈紧密结合 |
| 复杂度   | 最常见，开发简单，需求广泛             | 接口复杂，复杂度高，较少见 | 最少见，专业性强             |
| 开发者   | 多为自定义硬件开发者                   | 多由存储厂商开发           | 通常由网卡厂商主导           |
| 上层     | VFS                                    | 块设备子系统               | 网络协议栈                   |

驱动注册是驱动开发的第一步。以字符设备为例，驱动需要实现一组 `file_operations` 操作函数，并通过 `register_chrdev()` 或 `cdev_add()` 注册到内核。块设备和网络设备也有类似的注册流程，分别通过 `register_blkdev()`、`register_netdev()` 等接口完成注册。

## 字符设备驱动
1. 实现 file_operations
   ```c
   #include <linux/module.h>
   #include <linux/fs.h>
   #include <linux/uaccess.h>

   static int my_open(struct inode *inode, struct file *file) {
       return 0;
   }

   static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *ppos) {
       // 示例：返回一个字符
       char data = 'A';
       if (copy_to_user(buf, &data, 1)) return -EFAULT;
       return 1;
   }

   static struct file_operations my_fops = {
       .owner = THIS_MODULE,
       .open = my_open,
       .read = my_read,
   };
   ```
   file_operations 是供 vfs 层的 inode 来调用的接口，开发者需要实现几个基本的文件操作，然后当用户访问字符设备的时候就会触发相应的处理函数。

2. 注册和卸载驱动
   ```c
   #define MY_MAJOR 240

   static int __init my_init(void) {
       return register_chrdev(MY_MAJOR, "mychardev", &my_fops);
   }

   static void __exit my_exit(void) {
       unregister_chrdev(MY_MAJOR, "mychardev");
   }

   module_init(my_init);
   module_exit(my_exit);
   MODULE_LICENSE("GPL");
   ```
   使用 `module_init`，`module_exit` 等宏来注册设备驱动的初始化函数和卸载函数，当驱动被内核加载的时候会调用初始化函数，开发者可以在这个钩子中完成组件的初始化工作，同理卸载函数完成资源清理和注销。

3. 编译与加载
   - 编译：`make`
   - 加载模块：`sudo insmod mychardev.ko`
   - 卸载模块：`sudo rmmod mychardev`
   - 创建设备节点：`sudo mknod /dev/mychardev c 240 0`
   - 读测试：`cat /dev/mychardev`

   这样就完成了一个最基础的 Linux 字符设备驱动的编写和测试流程。实际开发中可根据硬件需求扩展 open、read、write、ioctl 等接口。

## 生命周期钩子
内核为驱动提供了丰富的接口函数和生命周期钩子，帮助开发者管理驱动的加载、运行和卸载过程。

### 模块加载与卸载
- `module_init(init_func)`：指定驱动模块加载时调用的初始化函数。
- `module_exit(exit_func)`：指定驱动模块卸载时调用的清理函数。
- `MODULE_LICENSE("GPL")`：声明模块许可证，避免内核加载警告。

### 设备注册与注销
- `register_chrdev()` / `unregister_chrdev()`：注册/注销字符设备。
- `cdev_init()` / `cdev_add()` / `cdev_del()`：初始化、添加、删除 cdev 结构体。
- `register_blkdev()` / `unregister_blkdev()`：注册/注销块设备。
- `register_netdev()` / `unregister_netdev()`：注册/注销网络设备。

### 文件操作钩子（file_operations）
驱动通过实现 `struct file_operations` 结构体，定义 open、read、write、ioctl 等操作方法。这些方法为用户空间提供了一种以访问文件的方式来访问硬件的接口。
- `.open`：打开设备文件时调用。
- `.release`：关闭设备文件时调用。
- `.read`：从设备读取数据时调用。
- `.write`：向设备写入数据时调用。
- `.ioctl` / `.unlocked_ioctl`：设备控制命令处理。
- `.mmap`：内存映射支持。
- `.poll` / `.llseek` 等：其他高级操作。

```c
struct file_operations my_fops = {
    .open = my_open,
    .read = my_read,
    .write = my_write,
    .release = my_release,
    // 还可以有 ioctl、mmap、poll 等
};
```

### 设备探测与移除
- `probe(struct platform_device *pdev)`：设备被发现时自动调用，常用于平台设备和总线设备。
- `remove(struct platform_device *pdev)`：设备被移除时调用。

### 中断与定时器
- `request_irq()` / `free_irq()`：注册/释放中断处理函数。
- `tasklet_init()` / `tasklet_schedule()`：软中断与下半部处理。
- `timer_setup()` / `add_timer()` / `del_timer()`：定时器相关。

许多硬件事件通过中断通知驱动。驱动通过 `request_irq()` 注册中断处理函数，中断处理函数负责响应硬件事件，完成数据采集、状态更新等任务。

```c
request_irq(irq, handler, IRQF_SHARED, "my_device", NULL);
```

### 电源管理钩子
- `suspend()` / `resume()`：设备挂起与唤醒时调用，常用于节能和休眠。
