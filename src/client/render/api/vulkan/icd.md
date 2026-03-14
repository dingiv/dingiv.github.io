# ICD

libvulkan.so 实际上是 Vulkan Loader（由 Khronos 维护）。它本身不包含任何渲染逻辑，而是一个“集线器”。如果一个 GPU 厂商（比如华为、高通或你自己写一个远程 GPU 驱动）想让 Loader 找到自己，必须遵循 ICD（Installable Client Driver） 加载机制。

1. 核心机制：JSON 清单文件 (Manifest File)
Vulkan Loader 不会去扫描所有的 .so 文件（那样太慢且不安全），它扫描的是特定系统目录下的 JSON 配置文件。

扫描路径： Loader 会在以下 Linux 标准路径中寻找 JSON 文件：

/usr/share/vulkan/icd.d/

/etc/vulkan/icd.d/

在 Android 上，通常是在 /vendor/etc/vulkan/icd.d/ 或通过系统属性直接指定。

2. JSON 文件的内容格式
厂商需要编写一个 JSON 文件（例如 my_gpu_icd.json），内容大致如下：

JSON
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "/vendor/lib64/libvulkan_mygpu.so",
        "api_version": "1.3.0"
    }
}
library_path： 这是关键。它告诉 Loader 真正的驱动二进制文件在哪里。

3. 驱动程序（.so）必须实现的导出函数
即便 Loader 找到了你的 .so，如果你的库里没有特定的入口点，它依然无法加载。作为 C 程序员，你的驱动库必须导出以下函数：

A. vk_icdGetInstanceProcAddr
这是 Loader 与驱动通信的第一个“握手”函数。Loader 会通过 dlsym 找到它。

B. vk_icdNegotiateLoaderICDInterfaceVersion
这是一个版本协商函数。驱动需要告诉 Loader：“我支持的 ICD 接口版本是多少”（目前主流是版本 5 或 6）。

4. 完整的加载流水线 (Loader Workflow)
当你调用 vkCreateInstance 时，底层发生了这些事：

扫描： Loader 遍历上述 /etc/vulkan/icd.d/ 目录。

解析： 读取 JSON，获取所有可用 GPU 驱动的路径。

加载： 对每个路径执行 dlopen()。

握手： 调用 vk_icdNegotiateLoaderICDInterfaceVersion 确认兼容性。

枚举： Loader 聚合所有驱动返回的物理设备（Physical Devices），最终应用层调用 vkEnumeratePhysicalDevices 就能看到你的 GPU 了。

5. 在 Android / 容器环境下的特殊处理
在你关注的 Android Docker 或 远程 GPU 场景中，你可以利用这个机制实现“无感知替换”：

欺骗 Loader： 你可以修改容器内的 /vendor/etc/vulkan/icd.d/ 下的 JSON 文件，将 library_path 指向你编写的一个“转发驱动”（Forwarding Driver）。

环境变量干预： 在标准 Linux 下，你可以设置环境变量 VK_ICD_FILENAMES 直接指定 JSON 路径，绕过系统默认扫描。

Bash
export VK_ICD_FILENAMES=/path/to/your/custom_icd.json
这在调试远程 GPU 方案时非常有用，不需要修改系统只读分区。