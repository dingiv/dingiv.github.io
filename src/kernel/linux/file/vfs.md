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

特别地，对于磁盘管理
+ 文件系统结构 super_block
+ 文件系统磁盘元数据

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
    struct timespec i_atime;        // 访问时间
    struct timespec i_mtime;        // 修改时间
    struct timespec i_ctime;        // 创建时间
    spinlock_t i_lock;              // inode锁
    struct mutex i_mutex;           // inode互斥锁
    // ...
};
```

### 字符设备
一个字符设备被看作一个 inode，使用一个 inode 进行抽象，


### 块设备

### 网络设备


### VFS 的核心数据结构

#### 1. super_block（超级块）

```c
struct super_block {
    struct list_head s_list;        // 超级块链表
    dev_t s_dev;                    // 设备标识符
    unsigned long s_blocksize;      // 块大小
    unsigned char s_blocksize_bits; // 块大小位数
    unsigned char s_dirt;           // 脏标志
    unsigned long long s_maxbytes;  // 最大文件大小
    struct file_system_type *s_type; // 文件系统类型
    struct super_operations *s_op;  // 超级块操作函数
    struct dquot_operations *dq_op; // 配额操作函数
    struct quotactl_ops *s_qcop;    // 配额控制操作
    struct export_operations *s_export_op; // 导出操作
    unsigned long s_flags;          // 挂载标志
    unsigned long s_magic;          // 魔数
    struct dentry *s_root;          // 根目录项
    struct rw_semaphore s_umount;   // 卸载信号量
    struct mutex s_lock;            // 超级块锁
    int s_count;                    // 引用计数
    atomic_t s_active;              // 活跃引用计数
    void *s_security;               // 安全信息
    struct xattr_handler **s_xattr; // 扩展属性处理器
    struct list_head s_inodes;      // inode链表
    struct hlist_bl_head s_anon;    // 匿名inode链表
    struct list_head s_mounts;      // 挂载点链表
    struct block_device *s_bdev;    // 块设备
    struct backing_dev_info *s_bdi; // 后备设备信息
    struct mtd_info *s_mtd;         // MTD设备信息
    struct hlist_node s_instances;  // 实例链表
    struct quota_info s_dquot;      // 配额信息
    struct sb_writers s_writers;    // 写者信息
    char s_id[32];                  // 文件系统ID
    u8 s_uuid[16];                  // UUID
    void *s_fs_info;                // 文件系统私有信息
    unsigned int s_max_links;       // 最大硬链接数
    fmode_t s_mode;                 // 挂载模式
    u32 s_time_gran;                // 时间粒度
    struct mutex s_vfs_rename_mutex; // 重命名互斥锁
    char *s_subtype;                // 子类型
    char *s_options;                // 挂载选项
};
```

#### 2. inode（索引节点）

```c
struct inode {
    umode_t i_mode;                 // 文件类型和权限
    uid_t i_uid;                    // 用户ID
    gid_t i_gid;                    // 组ID
    const struct inode_operations *i_op; // inode操作函数
    const struct file_operations *i_fop; // 文件操作函数
    struct super_block *i_sb;       // 所属超级块
    struct address_space *i_mapping; // 地址空间
    unsigned long i_ino;            // inode号
    atomic_t i_count;               // 引用计数
    unsigned int i_nlink;           // 硬链接数
    dev_t i_rdev;                   // 设备号
    loff_t i_size;                  // 文件大小
    struct timespec i_atime;        // 访问时间
    struct timespec i_mtime;        // 修改时间
    struct timespec i_ctime;        // 创建时间
    spinlock_t i_lock;              // inode锁
    struct mutex i_mutex;           // inode互斥锁
    struct rw_semaphore i_alloc_sem; // 分配信号量
    const struct file_operations *i_fop; // 文件操作函数
    struct list_head i_devices;     // 设备链表
    union {
        struct pipe_inode_info *i_pipe; // 管道信息
        struct block_device *i_bdev;    // 块设备
        struct cdev *i_cdev;            // 字符设备
    };
    __u32 i_generation;             // 生成号
    void *i_private;                // 私有数据
};
```

#### 3. dentry（目录项）

```c
struct dentry {
    unsigned int d_flags;           // 目录项标志
    seqcount_t d_seq;               // 序列计数
    struct hlist_bl_node d_hash;    // 哈希链表节点
    struct dentry *d_parent;        // 父目录项
    struct qstr d_name;             // 文件名
    struct inode *d_inode;          // 关联的inode
    unsigned char d_iname[DNAME_INLINE_LEN]; // 内联文件名
    struct lockref d_lockref;       // 锁引用
    const struct dentry_operations *d_op; // 目录项操作函数
    struct super_block *d_sb;       // 所属超级块
    void *d_fsdata;                 // 文件系统私有数据
    struct list_head d_lru;         // LRU链表
    struct list_head d_child;       // 子目录项链表
    struct list_head d_subdirs;     // 子目录链表
    union {
        struct hlist_node d_alias;  // 别名链表
        struct hlist_bl_node d_in_lookup_hash; // 查找哈希链表
        struct rcu_head d_rcu;      // RCU头
    } d_u;
};
```

#### 4. file（文件对象）

```c
struct file {
    union {
        struct llist_node fu_llist; // 链表节点
        struct rcu_head fu_rcuhead; // RCU头
    } f_u;
    struct path f_path;             // 文件路径
    struct inode *f_inode;          // 关联的inode
    const struct file_operations *f_op; // 文件操作函数
    spinlock_t f_lock;              // 文件锁
    atomic_long_t f_count;          // 引用计数
    unsigned int f_flags;           // 文件标志
    fmode_t f_mode;                 // 文件模式
    struct mutex f_pos_lock;        // 位置锁
    loff_t f_pos;                   // 文件位置
    struct fown_struct f_owner;     // 文件所有者
    const struct cred *f_cred;      // 凭证
    struct file_ra_state f_ra;      // 预读状态
    u64 f_version;                  // 版本号
    void *private_data;             // 私有数据
    struct list_head f_ep_links;    // epoll链接
    struct address_space *f_mapping; // 地址空间
};
```

### VFS 的操作接口

#### 1. 文件操作接口（file_operations）

```c
struct file_operations {
    struct module *owner;
    loff_t (*llseek) (struct file *, loff_t, int);
    ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
    ssize_t (*read_iter) (struct kiocb *, struct iov_iter *);
    ssize_t (*write_iter) (struct kiocb *, struct iov_iter *);
    int (*iterate) (struct file *, struct dir_context *);
    unsigned int (*poll) (struct file *, struct poll_table_struct *);
    long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
    long (*compat_ioctl) (struct file *, unsigned int, unsigned long);
    int (*mmap) (struct file *, struct vm_area_struct *);
    int (*open) (struct inode *, struct file *);
    int (*flush) (struct file *, fl_owner_t id);
    int (*release) (struct inode *, struct file *);
    int (*fsync) (struct file *, loff_t, loff_t, int datasync);
    int (*aio_fsync) (struct kiocb *, int datasync);
    int (*fasync) (int, struct file *, int);
    int (*lock) (struct file *, int, struct file_lock *);
    ssize_t (*sendpage) (struct file *, struct page *, int, size_t, loff_t *, int);
    unsigned long (*get_unmapped_area)(struct file *, unsigned long, unsigned long, unsigned long, unsigned long);
    int (*check_flags)(int);
    int (*flock) (struct file *, int, struct file_lock *);
    ssize_t (*splice_write)(struct pipe_inode_info *, struct file *, loff_t *, size_t, unsigned int);
    ssize_t (*splice_read)(struct file *, loff_t *, struct pipe_inode_info *, size_t, unsigned int);
    int (*setlease)(struct file *, long, struct file_lock **);
    long (*fallocate)(struct file *file, int mode, loff_t offset, loff_t len);
    int (*show_fdinfo)(struct seq_file *m, struct file *f);
};
```

#### 2. inode 操作接口（inode_operations）

```c
struct inode_operations {
    struct dentry * (*lookup) (struct inode *,struct dentry *, unsigned int);
    void * (*follow_link) (struct dentry *, struct nameidata *);
    int (*permission) (struct inode *, int);
    struct posix_acl * (*get_acl)(struct inode *, int);
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
    int (*setattr) (struct dentry *, struct iattr *);
    int (*getattr) (const struct path *, struct kstat *, u32, unsigned int);
    ssize_t (*listxattr) (struct dentry *, char *, size_t);
    void (*truncate) (struct inode *);
    int (*fiemap)(struct inode *, struct fiemap_extent_info *, u64 start, u64 len);
    int (*update_time)(struct inode *, struct timespec *, int);
    int (*atomic_open)(struct inode *, struct dentry *, struct file *, unsigned open_flag, umode_t create_mode);
    int (*tmpfile) (struct inode *, struct dentry *, umode_t);
    int (*set_acl)(struct inode *, struct posix_acl *, int);
};
```

### VFS 的工作流程

#### 1. 文件打开流程

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

#### 2. 文件读写流程

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

### VFS 的应用场景

#### 1. 传统文件系统

- **ext4、xfs、btrfs**：本地磁盘文件系统
- **NFS、CIFS**：网络文件系统
- **FAT32、NTFS**：Windows 兼容文件系统

#### 2. 特殊文件系统

- **procfs**：进程信息文件系统（/proc）
- **sysfs**：系统信息文件系统（/sys）
- **tmpfs**：临时文件系统（/tmp）
- **devtmpfs**：设备文件系统（/dev）

#### 3. 设备抽象

```c
// 字符设备示例
static const struct file_operations my_char_fops = {
    .owner = THIS_MODULE,
    .read = my_char_read,
    .write = my_char_write,
    .open = my_char_open,
    .release = my_char_release,
    .unlocked_ioctl = my_char_ioctl,
};

// 块设备示例
static const struct block_device_operations my_block_fops = {
    .owner = THIS_MODULE,
    .open = my_block_open,
    .release = my_block_release,
    .ioctl = my_block_ioctl,
};
```

#### 4. 网络抽象

```c
// 套接字文件操作
static const struct file_operations socket_file_ops = {
    .owner = THIS_MODULE,
    .read = sock_read,
    .write = sock_write,
    .poll = sock_poll,
    .unlocked_ioctl = sock_ioctl,
    .mmap = sock_mmap,
    .release = sock_close,
};
```
