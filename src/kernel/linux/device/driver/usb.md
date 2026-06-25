# USB 驱动
USB（Universal Serial Bus）是计算机与外设连接的最通用总线，支持热插拔和即插即用，广泛应用于键盘、鼠标、摄像头、存储设备、网络适配器等场景。Linux 内核的 USB 子系统采用分层架构，将主机控制器硬件细节、通用 USB 协议逻辑和具体设备驱动分离开来，使得 USB 驱动开发只需关注设备本身的通信协议，而无需处理底层总线传输。

## USB 子系统架构
Linux USB 子系统分为三层。底层是**主机控制器驱动（HCD, Host Controller Driver）**，负责与具体的 USB 主机控制器硬件通信，如 xHCI（USB 3.0）、EHCI（USB 2.0）、OHCI/UHCI（USB 1.1）。中间层是 **USB 核心（USB Core）**，提供设备枚举、配置管理、请求分发、 urb（USB Request Block）管理等通用功能。上层是 **USB 设备驱动（USB Gadget/Device Driver）**，即开发者编写的针对具体 USB 设备的驱动程序。

这种分层架构意味着 USB 驱动开发者不需要关心数据在总线上是如何传输的（是 USB 2.0 的批量传输还是 USB 3.0 的超高速流传输），也不需要关心主机控制器是 xHCI 还是 EHCI。USB 核心层屏蔽了这些差异，向上提供统一的 API。

USB 设备的逻辑组织为**配置（Configuration）→ 接口（Interface）→ 端点（Endpoint）**的层次结构。一个设备可以有一个或多个配置，每个配置包含一个或多个接口，每个接口包含一个或多个端点。驱动程序绑定的是接口而非整个设备，这意味着一个复合设备（Composite Device，如带麦克风的摄像头）的不同接口可以由不同的驱动分别管理。

## 设备识别与匹配
USB 设备通过**厂商 ID（Vendor ID）**、**产品 ID（Product ID）** 和**设备类（bDeviceClass）**来标识。与 PCI 类似，驱动通过匹配表声明自己支持的设备范围。USB 额外引入了设备类的概念，标准设备类（如大容量存储 0x08、HID 0x03、音频 0x01、网络 0x0e）可以被通用驱动接管，而特定厂商的自定义设备则需要厂商提供专用驱动。

```c
static const struct usb_device_id my_usb_ids[] = {
    // 匹配特定厂商和产品
    { USB_DEVICE(0x1234, 0x5678) },
    // 匹配特定设备类
    { USB_INTERFACE_INFO(USB_CLASS_VENDOR_SPEC, 0, 0) },
    // 使用设备类匹配（bDeviceClass 在设备级别指定）
    { USB_DEVICE_AND_INTERFACE_INFO(0x1234, 0x5678,
        USB_CLASS_VENDOR_SPEC, 0, 0) },
    { 0 }  // 结束标记
};
MODULE_DEVICE_TABLE(usb, my_usb_ids);
```

`USB_DEVICE` 是最常用的匹配宏，仅按厂商 ID 和产品 ID 匹配。`USB_INTERFACE_INFO` 按接口类、子类、协议匹配，适用于驱动某一类标准设备。`USB_DEVICE_AND_INTERFACE_INFO` 同时匹配设备 ID 和接口信息，最精确但约束也最多。

USB 驱动中一个容易混淆的概念是：匹配的是接口而非设备。一个 USB 设备可能有多个接口，`MODULE_DEVICE_TABLE` 中的匹配表是按接口级别的标准来匹配的。内核在枚举设备时，会为每个接口单独查找匹配的驱动，所以一个复合设备的不同接口可以绑定到不同的驱动。

## 驱动注册与生命周期
USB 驱动通过 `struct usb_driver` 结构体定义，核心回调同样是 probe 和 remove，但这里的 probe 参数是 `struct usb_interface` 而非设备本身。

```c
static int my_usb_probe(struct usb_interface *intf,
                        const struct usb_device_id *id) {
    struct usb_device *udev = interface_to_usbdev(intf);
    struct usb_host_interface *alt = intf->cur_altsetting;

    // 1. 检查端点数量和类型
    if (alt->desc.bNumEndpoints < 2) return -ENODEV;

    struct usb_endpoint_descriptor *ep_in = NULL, *ep_out = NULL;
    for (int i = 0; i < alt->desc.bNumEndpoints; i++) {
        struct usb_endpoint_descriptor *ep = &alt->endpoint[i].desc;
        if (usb_endpoint_is_bulk_in(ep))  ep_in = ep;
        if (usb_endpoint_is_bulk_out(ep)) ep_out = ep;
    }
    if (!ep_in || !ep_out) return -ENODEV;

    // 2. 分配私有数据
    struct my_usb_device *dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev) return -ENOMEM;

    // 3. 保存端点地址，后续提交 urb 时使用
    dev->ep_in = ep_in->bEndpointAddress;
    dev->ep_out = ep_out->bEndpointAddress;
    dev->udev = udev;

    // 4. 设置接口，将设备切换到目标配置（某些设备有多个 altsetting）
    int ret = usb_set_interface(udev, alt->desc.bInterfaceNumber, 0);
    if (ret) goto free;

    // 5. 挂载私有数据到接口
    usb_set_intfdata(intf, dev);

    // 6. 注册 USB 设备文件（可选，让用户空间通过 /dev 访问）
    dev->minor = usb_register_dev(intf, &my_usb_class);
    if (dev->minor < 0) {
        ret = dev->minor;
        goto free;
    }

    return 0;

free:
    kfree(dev);
    return ret;
}

static void my_usb_disconnect(struct usb_interface *intf) {
    struct my_usb_device *dev = usb_get_intfdata(intf);

    // 取消所有进行中的 urb
    usb_kill_urb(dev->urb_in);
    usb_kill_urb(dev->urb_out);

    // 注销设备文件
    usb_deregister_dev(intf, &my_usb_class);

    // 释放资源
    usb_set_intfdata(intf, NULL);
    kfree(dev);
}

static struct usb_driver my_usb_driver = {
    .name = "my_usb_device",
    .id_table = my_usb_ids,
    .probe = my_usb_probe,
    .disconnect = my_usb_disconnect,
};

module_usb_driver(my_usb_driver);
MODULE_LICENSE("GPL");
```

`module_usb_driver` 与 `module_pci_driver` 类似，是一个简化宏，自动生成 init/exit 函数。`usb_set_intfdata` 和 `usb_get_intfdata` 是一对常用的数据挂载接口，将驱动私有数据关联到 USB 接口上，在 disconnect 时取回进行清理。

### 端点类型
USB 定义了四种端点传输类型，每种适用于不同的数据传输场景：

控制传输（Control）用于设备配置和命令，所有设备都必须有默认控制端点（EP0），USB 核心层自动管理，驱动一般不需要直接操作。中断传输（Interrupt）用于低延迟、小批量的数据交换，如键盘、鼠标的输入报告，USB 规范保证中断传输的服务间隔。批量传输（Bulk）用于大批量、无实时性要求的数据传输，如 U 盘的数据读写，总线空闲时优先使用带宽。等时传输（Isochronous）用于对实时性要求高的数据流，如音视频采集，保证固定带宽但不保证可靠性（无重传机制）。

驱动在 probe 时需要检查接口的端点描述符，确认端点的类型、方向（IN/OUT）和最大包长，这些信息决定了后续 urb 的提交方式。

## URB 机制
URB（USB Request Block）是 USB 驱动与 USB 核心层之间的核心数据结构，代表一次 USB 传输请求。驱动通过分配和提交 URB 来发起数据传输，传输完成后通过回调函数通知驱动。

```c
// 分配 urb
struct urb *urb = usb_alloc_urb(0, GFP_KERNEL);

// 填充 bulk out urb
unsigned char *buf = kmalloc(512, GFP_KERNEL);
memcpy(buf, data, 512);
usb_fill_bulk_urb(urb, udev, dev->ep_out, buf, 512,
                  my_write_callback, dev);
urb->transfer_flags |= URB_FREE_BUFFER;  // 传输完成后自动释放 buf

// 提交 urb
int ret = usb_submit_urb(urb, GFP_KERNEL);
if (ret) {
    usb_free_urb(urb);
    return ret;
}

// 完成回调函数
static void my_write_callback(struct urb *urb) {
    struct my_usb_device *dev = urb->context;
    if (urb->status == 0) {
        // 传输成功，urb->actual_length 是实际传输的字节数
    } else {
        // 传输失败，urb->status 包含错误码
    }
}
```

`usb_fill_bulk_urb` 是一个辅助函数，用于快速填充批量传输的 urb。类似的还有 `usb_fill_int_urb`（中断传输）和 `usb_fill_control_urb`（控制传输）。`urb->context` 是用户自定义指针，会在回调函数中原样返回，通常指向驱动私有数据结构。

URB 的生命周期是：分配（`usb_alloc_urb`）→ 填充 → 提交（`usb_submit_urb`）→ 完成（回调函数被调用）→ 释放（`usb_free_urb`）。提交后的 urb 由 USB 核心层和主机控制器管理，驱动不应再直接修改它。如果需要在设备断开前取消传输，调用 `usb_kill_urb`（同步等待 urb 完成）或 `usb_unlink_urb`（异步取消）。

### 同步 API
除了异步的 urb 机制，USB 核心层还提供了同步 API，封装了 urb 的分配、提交和等待过程，适用于简单的单次传输场景。

```c
// 同步批量读取
int ret, actual;
unsigned char buf[512];
ret = usb_bulk_msg(udev, dev->ep_in, buf, 512, &actual, 2000);
// 超时 2000ms，actual 返回实际读取的字节数

// 同步控制传输
ret = usb_control_msg(udev, usb_sndctrlpipe(udev, 0),
    0x01,  // request
    USB_TYPE_VENDOR | USB_RECIP_DEVICE | USB_DIR_OUT,
    0, 0, buf, sizeof(buf), 2000);
```

`usb_bulk_msg` 和 `usb_control_msg` 在内部会阻塞当前线程直到传输完成或超时，使用简单但无法实现高性能的流水线传输。对于需要同时处理多个传输的场景（如网络适配器、存储设备），应使用异步 urb 并在回调中重新提交，形成传输流水线。

## 电源管理与热插拔
USB 总线天生支持热插拔，设备随时可能被拔出。驱动的 disconnect 函数必须确保在设备拔出时正确释放所有资源，包括取消进行中的 urb、释放缓冲区、注销设备文件等。一个常见的错误是在 disconnect 时对已拔出的设备提交新的 urb，这会导致内核崩溃。使用 `usb_get_intfdata` 获取私有数据时，需要检查指针是否有效。

USB 设备的电源管理比 PCI 更复杂，因为 USB 总线本身有电源管理能力。当总线空闲一段时间后，主机可以暂停（Suspend）整个总线以节能。驱动需要通过 `pm` 回调支持这一机制。

```c
static int my_usb_suspend(struct usb_interface *intf, pm_message_t message) {
    struct my_usb_device *dev = usb_get_intfdata(intf);
    usb_kill_urb(dev->urb_in);
    usb_kill_urb(dev->urb_out);
    return 0;
}

static int my_usb_resume(struct usb_interface *intf) {
    struct my_usb_device *dev = usb_get_intfdata(intf);
    // 重新提交 urb，恢复数据传输
    usb_submit_urb(dev->urb_in, GFP_KERNEL);
    return 0;
}

static struct usb_driver my_usb_driver = {
    .suspend = my_usb_suspend,
    .resume  = my_usb_resume,
    // ...
};
```

USB 设备的远程唤醒（Remote Wakeup）也是一个需要注意的特性。如果设备需要在总线挂起期间主动唤醒主机（如键盘检测到按键），驱动需要调用 `usb_enable_autosuspend` 启用自动挂起，并通过 `usb_set_interface` 的配置告知 USB 核心。`usb_disable_autosuspend` 则用于禁用自动挂起。

## 用户空间接口
USB 驱动通常需要向用户空间暴露接口，让应用程序能够与设备通信。最常见的方式是注册 USB 设备类，让 udev 自动创建 `/dev/usb/` 下的设备节点。

```c
// 定义 USB 设备类
static struct usb_class_driver my_usb_class = {
    .name = "my_usb%d",  // %d 会被替换为 minor 号，生成 /dev/my_usb0 等
    .fops = &my_usb_fops,
    .minor_base = MY_USB_MINOR_BASE,
};

// file_operations 实现与字符设备驱动一致
static const struct file_operations my_usb_fops = {
    .owner = THIS_MODULE,
    .open = my_usb_open,
    .release = my_usb_release,
    .read = my_usb_read,
    .write = my_usb_write,
    .unlocked_ioctl = my_usb_ioctl,
};
```

USB 设备类的 `fops` 与普通字符设备驱动的 `file_operations` 完全一致。在 `read` 和 `write` 中，驱动通过提交 urb（同步或异步）完成与 USB 设备的数据交换。这种模式将 USB 总线细节封装在驱动内部，用户空间只需像读写普通文件一样操作 `/dev/my_usb0` 即可。

另一种方式是使用 `usbfs`（也称为 `usbdevfs`），内核将 USB 设备的控制权直接暴露给用户空间，由用户空间程序（如 libusb）直接提交 ioctl 来控制设备。这种方式常用于不需要内核驱动的场景，如固件升级工具、设备调试工具等。

## 调试与工具
USB 驱动调试常用的工具和方法：

- `lsusb`：列出系统中所有 USB 设备及其厂商 ID、产品 ID、设备类等基本信息，配合 `-v` 参数可以查看完整的设备描述符和配置描述符。
- `usbmon`：内核自带的 USB 流量抓包工具，通过 `debugfs` 接口导出，可以捕获总线上所有 USB 传输的细节，类似于网络抓包的 tcpdump。
- `dmesg`：查看内核 USB 子系统的日志输出，包括设备枚举过程、驱动绑定事件、传输错误等。
- `CONFIG_USB_DEBUG`：内核配置选项，开启 USB 核心层的调试信息输出。
- `usbdevfs`：通过 `ioctl` 从用户空间直接与 USB 设备交互，常配合 libusb 使用，适合不编写内核驱动的场景。

设备枚举失败时，首先用 `lsusb -v` 确认设备是否被识别以及描述符是否正确。驱动 probe 未触发时，检查 `id_table` 的匹配条件是否与设备描述符一致。传输超时或错误时，使用 `usbmon` 抓包分析总线上的实际传输情况。设备热插拔后驱动未自动加载，检查 `MODULE_DEVICE_TABLE` 是否正确导出以及 udev 规则是否配置了 modalias 匹配。
