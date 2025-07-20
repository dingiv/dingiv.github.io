# 虚拟文件系统
Linux 继承了 Unix 系统的经典哲学——**一切皆文件**的思想。虚拟文件系统（Virtual File System，VFS）是 Linux 内核中的一个重要抽象层，它为不同的文件系统提供了统一的接口，使得应用程序可以用相同的方式访问不同类型的文件系统。

VFS 的主要作用是：
1. 统一文件操作接口：为各种文件系统提供统一的 API 接口，包括 open、read、write、close 等系统调用，抹平不同的文件系统（如 ext4、NTFS、NFS 等）的差异；
2. 系统信息和硬件信息抽象：将系统内部的信息以虚拟文件的形式挂载到 VFS 中的**临时文件** `devtmpfs` 节点，用户进程可以通过访问文件的方式来获取系统的信息，包括管道、套接字、/proc、/sys等；将硬件设备（如磁盘、网络设备、字符设备）抽象为文件；
3. 作为系统调用接口的补充：系统调用接口的数量有限，且长期保持稳定，新的内核模块可以通过暴露虚拟文件接口，让用户程序以操作文件的方式与内核模块交互；

## 层次结构
VFS 是一个从上到下的层次结构
+ 用户态文件句柄 file
+ 内核态文件树节点 dentry
+ 物理数据映射节点 inode

特别地，对于磁盘管理，还包括了进一步对接下层的超级块
+ 文件系统结构 super_block

## 用户态文件句柄
用户态进程需要访问一个文件的时候，需要先打开一个文件，打开文件的过程就是创建一个文件句柄的过程。文件句柄的内核态数据结构是 `struct file`，用户态通过 open 函数返回的**文件描述符**（一个 int 类型的文件 id），每一个进程所打开的文件都会保存在进程的结构体中，从而存储进程所持有的文件资源。

```c
struct file {
    struct path f_path;             // 文件路径
    struct inode *f_inode;          // 关联的inode
    const struct file_operations *f_op; // 文件操作函数
    spinlock_t f_lock;              // 文件锁
    atomic_long_t f_count;          // 引用计数
    unsigned int f_flags;           // 文件标志
    fmode_t f_mode;                 // 文件模式
    struct mutex f_pos_lock;        // 位置锁
    // ...
};
```

## 内核态文件树节点
在内核的内存中驻留着一个全局的虚拟文件树，用于向用户空间提供结构化的文件目录系统，用户可以通过文件的形式访问系统的各个资源和文件，这层结构是向上提供的抽象。

树中的每一个节点都是使用一个 dentry 结构来表示，这是一个树节点。通过该节点可以访问父目录和该目录中内容，或者说它是一个文件。
```c
struct dentry {
    unsigned int d_flags;           // 目录项标志
    struct dentry *d_parent;        // 父目录项
    struct qstr d_name;             // 文件名
    struct inode *d_inode;          // 关联的 inode
    const struct dentry_operations *d_op; // 目录项操作函数
    struct super_block *d_sb;       // 所属超级块
    struct list_head d_child;       // 子目录项链表
    struct list_head d_subdirs;     // 子目录链表
    // ...
};
```

每一个 dentry 都必须关联一个 inode，用于指向这个文件代表的真实物理资源。

## 物理数据映射节点
Linux 中的设备主要就是分为三种，字符设备、块设备、网络设备，访问虚拟文件树时，一个文件必须关联到一个真实的物理资源上，也就是这三种设备中的一种。

```c
struct inode {
    umode_t i_mode;                 // 文件类型和权限
    uid_t i_uid;                    // 用户ID
    gid_t i_gid;                    // 组ID
    const struct inode_operations *i_op; // inode 操作函数
    const struct file_operations *i_fop; // 文件操作函数
    struct super_block *i_sb;       // 所属超级块
    struct address_space *i_mapping; // 地址空间
    unsigned long i_ino;            // inode号
    atomic_t i_count;               // 引用计数
    unsigned int i_nlink;           // 硬链接数
    dev_t i_rdev;                   // 设备号
    loff_t i_size;                  // 文件大小
    // ...
};

struct inode_operations {
    struct dentry * (*lookup) (struct inode *,struct dentry *, unsigned int);
    void * (*follow_link) (struct dentry *, struct nameidata *);
    int (*permission) (struct inode *, int);
    int (*readlink) (struct dentry *, char __user *,int);
    void (*put_link) (struct dentry *, struct nameidata *, void *);
    int (*create) (struct inode *,struct dentry *, umode_t, bool);
    int (*link) (struct dentry *,struct inode *,struct dentry *);
    int (*unlink) (struct inode *,struct dentry *);
    int (*symlink) (struct inode *,struct dentry *,const char *);
    int (*mkdir) (struct inode *,struct dentry *,umode_t);
    int (*rmdir) (struct inode *,struct dentry *);
    int (*mknod) (struct inode *,struct dentry *,umode_t,dev_t);
    int (*rename) (struct inode *, struct dentry *, struct inode *, struct dentry *);
    // ...
};

struct file_operations {
    struct module *owner;
    loff_t (*llseek) (struct file *, loff_t, int);
    ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
    unsigned int (*poll) (struct file *, struct poll_table_struct *);
    long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
    long (*compat_ioctl) (struct file *, unsigned int, unsigned long);
    int (*mmap) (struct file *, struct vm_area_struct *);
    int (*open) (struct inode *, struct file *);
    int (*flush) (struct file *, fl_owner_t id);
    int (*flock) (struct file *, int, struct file_lock *);
    // ...
};
```

### 字符设备对接
一个字符设备被看作一个 inode，使用一个 inode 进行抽象，字符设备的 inode 表示 /dev 下的设备文件（如 /dev/tty）。
+ inode->i_cdev 指向 struct cdev，关联字符设备驱动。
+ inode->i_rdev 存储主次设备号（major/minor），用于查找驱动。

当打开一个字符设备的时候，VFS 找到该设备的驱动，并直接调用相应的方法进行设备操作。

### [块设备对接](./block)
块设备的 inode 表示 /dev 下的设备文件（如 /dev/sda）或文件系统中的文件。
+ inode->i_bdev 指向 struct block_device，关联块设备驱动。
+ inode->i_rdev 存储主次设备号，链接到 struct gendisk（表示磁盘）。

但是，对于块设备而言，用户通常不是通过访问它的虚拟设备文件来访问数据的，而是通过将其挂载到某个目录中，然后 VFS 会构造出这个块设备中的文件目录结构，用户通过直接访问这些文件节点，即 `struct dentry` 节点，从而区读写文件。对于一个普通的文件，其 dentry 指向的 inode 中的 `struct super_block` 指向了该文件所在的磁盘块，通过中间文件系统来对一个文件进行读写。

### 超级块
超级块抽象的是一个磁盘目录系统，且已经挂载到 VFS 上的某个目录上。当用户访问该目录下的某个文件节点的时候，VFS 通过文件的 inode 中的 super_block，从而寻找到该文件的文件系统和所属的磁盘分区，又通过 s_bdev 得知该分区归属的真实物理磁盘。

```c
struct super_block {
    dev_t s_dev;                    // 设备标识符
    struct block_device *s_bdev;    // 块设备磁盘分区
    struct file_system_type *s_type; // 文件系统类型
    struct super_operations *s_op;  // 超级块操作函数
    struct dquot_operations *dq_op; // 配额操作函数
    struct quotactl_ops *s_qcop;    // 配额控制操作
    struct export_operations *s_export_op; // 导出操作
    unsigned long s_flags;          // 挂载标志
    struct dentry *s_root;          // 根目录项
    char s_id[32];                  // 文件系统ID
    u8 s_uuid[16];                  // UUID
    fmode_t s_mode;                 // 挂载模式
    struct list_head s_list;        // 超级块链表
    // ...
};
```

对于不同的超级块，通过访问其文件系统，由文件系统来控制数据在磁盘中的文件目录组织形式，不过，文件系统也不会直接访问磁盘驱动，而是通过更下层的模块——块设备子系统来访问。

## 文件操作

```c
// 简化的文件打开流程
int do_open(struct file *file, const char *pathname, int flags, umode_t mode)
{
    struct path path;
    struct inode *inode;
    int error;

    // 1. 路径查找
    error = path_lookup(pathname, flags, &path);
    if (error)
        return error;

    // 2. 获取inode
    inode = path.dentry->d_inode;

    // 3. 权限检查
    error = inode_permission(inode, flags);
    if (error)
        goto out;

    // 4. 调用文件系统的open方法
    if (inode->i_fop && inode->i_fop->open) {
        error = inode->i_fop->open(inode, file);
        if (error)
            goto out;
    }

    // 5. 设置文件对象
    file->f_path = path;
    file->f_inode = inode;
    file->f_op = inode->i_fop;

out:
    return error;
}
```

```c
// 简化的文件读取流程
ssize_t do_read(struct file *file, char __user *buf, size_t count, loff_t *pos)
{
    ssize_t ret;

    // 1. 参数检查
    if (!file->f_op || !file->f_op->read)
        return -EINVAL;

    // 2. 调用文件系统的read方法
    ret = file->f_op->read(file, buf, count, pos);

    // 3. 更新访问时间
    if (ret > 0)
        file_accessed(file);

    return ret;
}
```

## 文件缓存
虚拟文件子系统为了减少磁盘的操作，提高文件读写的速度，提供了文件的高速缓存机制。其主要的缓存对象是文件数据和文件元数据。文件数据通过**页面缓存**机制实现，而文件元数据通过文件系统提供的**文件系统元数据缓冲区**实现。

### 页面缓存
页面缓存是 VFS 的主要缓存机制，基于 `struct address_space（<linux/fs.h>）`，将文件数据映射到内存页面
+ 读操作：用户调用 read("/mnt/file.txt")，VFS 通过 inode->i_mapping 检查页面缓存。缓存命中：直接从 struct page 拷贝数据到用户空间（copy_to_user）。缓存未命中：调用 address_space->a_ops->readpage（如 ext4_readpage），生成 struct bio 提交到块设备子系统，加载数据到页面缓存。
+ 写操作：用户调用 write("/mnt/file.txt")，数据写入页面缓存（copy_from_user），标记为脏页（SetPageDirty）。脏页由 writeback 机制（kworker 线程或 fsync）触发，通过 a_ops->writepages（如 ext4_writepages）生成 bio，提交到块设备子系统。

## 文件系统
文件系统是一个工作在虚拟文件系统和块设备子系统之间的中间层，它是 Linux 为了实现支持多个不同的文件系统而设置的一个可灵活替换的中间协议层，可以由第三方实现。文件系统的主要工作是负责在磁盘这个字节数组中，组织文件目录的逻辑结构。

以 ext4 文件系统的文件读写为例，介绍文件系统在上下层之间如何工作：
+ 用户调用 `open("/mnt/file.txt")`，VFS 找到 dentry 和 inode，通过 inode->i_sb 访问 super_block，得知为 ext4 文件系统，接下来的文件操作逻辑由 ext4 完成；
+ ext4 通过 super_block->s_op 触发文件系统操作，定位文件数据块；
+ ext4 生成 bio 请求，通过 super_block->s_bdev 提交到块设备子系统；
+ 块设备子系统接受 bio，并处理请求，完成任务队列处理；
+ 块设备子系统调用磁盘硬件，与硬件交互，完成读写；

### 缓冲区
缓冲区是文件系统附带的辅助缓存机制，基于 `struct buffer_head（<linux/buffer_head.h>）`，主要用于文件系统元数据（如 ext4 的超级块、inode 表）。

+ 元数据读写：
  文件系统（如 ext4）通过 getblk 或 sb_bread 获取 buffer_head，缓存元数据。  
  读：检查 buffer_head->b_state，若 BH_Uptodate，直接返回；否则，触发 bio 读取块设备。  
  写：更新 buffer_head，标记 BH_Dirty，通过 submit_bh 提交到块设备子系统。
+ 与页面缓存的关系：缓冲区常作为页面缓存的子集，元数据块绑定到页面（page->private 指向 buffer_head）。页面缓存优先处理文件数据，缓冲区专注于元数据或小块 I/O。