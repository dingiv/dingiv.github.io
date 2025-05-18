---
title: Shell
order: 10
---

# Shell
shell 是 linux 系统提供内置脚本系统，用户可以通过 shell 命令来与系统进行交互。

shell 的种类有很多，比如：Bourne Shell（sh）、C Shell（csh）、Korn Shell（ksh）、Bourne Again Shell（bash）等。目前最常用的 shell 是 bash，它是 GNU 计划的一部分，Linux 大多数发行版都包含 bash。同时在 MacOS 中，bash 也被广泛支持。

## 基础
shell 是基于命令式编程的，在众多方面特性的处理上，与高级语言不同。shell 的每一行都可以看作是一个命令，每行的第一个单词代表了命令的名称，

shell 的语法内容包括三个部分：

- 基础语法和命令，这个是所有 shell 都共有的部分，比如变量、条件判断、循环等。该部分可以跨 shell，比如 bash、zsh 等。
- 内置语法和命令，这个是各个 shell 中自定义的部分，属于 shell 的方言，依赖于不同的 shell 环境。
- 外部命令，这个是 shell 调用系统命令的部分，比如 ls、cd、mv 等。这些命令依赖于外部的程序实现，主要依赖于操作系统和系统全局快捷命令。

### 常用语法

1. 注释

   ```bash
   #! /bin/bash # shebang，在脚本的第一行，用于指定解释器，例如：。
   #：单行注释；
   ```

2. 变量

   **shell 的变量是基于字符串的，变量没有类型，都是字符串**。变量名和等号之间不能有空格，变量名只能包含字母、数字和下划线，且不能以数字开头。如果需要空格需要使用引号。

   ```bash
   var=12 # 直接赋值
   var1="hello $var" # 双引号赋值，双引号可以展开变量并保留特殊符号的转义。
   var='hello' # 单引号赋值，单引号不能展开变量，也不能保留特殊符号的转义。
   var=`echo "hello"` # 反引号赋值，反引号可以执行命令并返回结果。
   var=$(echo "hello")
   var1=${var}
   ```

   由于 shell 的变量是基于字符串的，因此，在进行数值计算的时候，需要使用一些内置或者第三方命令，如 `expr` 命令或者 `$(())` 来进行数值计算。

3. 变量声明
   shell 的变量声明有两种方式，一种是直接赋值，另一种是使用 `declare` 命令。

   ```bash
   var=12
   declare -i var=12
   ```

   `declare` 命令可以用来声明变量的类型，例如：`-i` 表示整数，`-r` 表示只读，`-a` 表示数组，`-f` 表示函数，`-x` 表示环境变量。

4. 字符串

   字符串的拼接，可以直接使用加号进行拼接，也可以使用双引号进行拼接。

   ```bash
   str1="hello"
   str2="world"
   str3=$str1$str2
   str4="$str1 $str2"
   echo $str3
   echo $str4
   ```

   字符串操作是 shell 中非常重要的部分，包括字符串的截取、替换、查找、删除等操作。

   常用字符串操作命令：

   - `echo`：输出字符串
   - `printf`：格式化输出字符串
   - `cut`：截取字符串
   - `sed`：替换字符串
   - `grep`：查找字符串
   - `awk`：处理字符串
   - `tr`：删除字符串中的字符

   包括插值符号自身也对字符串处理提供了强大的内置动能。

   ```bash
   str="hello world"
   echo $str
   printf "%s\n" $str
   echo ${str:0:5}
   echo ${str:6:5}
   echo ${str#hello}
   echo ${str%world}
   echo ${str//o/O}
   echo ${str/o/O}
   echo ${str//o}
   echo ${str/o/}
   ```

5. 条件判断
   test 命令，用于检查某个条件是否成立，如果条件成立则返回 true，否则返回 false。test 命令可以接受多种类型的参数，包括字符串、数字和文件。

   `[` 命令，是 test 命令的语法糖，需要用 `]` 结尾。`[[` 是 bash 内置的命令，需要以 `]]` 结尾，功能比 test 更加强大，支持正则表达式。语义更好。

   ```bash
   test -e file
   [ -e file ]
   [[ -e file ]]
   ```

   条件判断语句，if、elif、else、fi。条件判断语句可以嵌套，也可以使用 `&&` 和 `||` 来连接多个条件判断语句。

   ```bash
   if [ -e file ]; then
       echo "file exists"
   elif [ -d file ]; then
       echo "file is a directory"
   else
       echo "file does not exist"
   fi
   ```

6. 循环

   循环语句，for、while、until。循环语句可以嵌套，也可以使用 `break` 和 `continue` 来控制循环的执行。

   ```bash
   for i in {1..10}; do
       echo $i
   done

   while [ $i -lt 10 ]; do
       echo $i
       i=$((i+1))
   done

   until [ $i -gt 10 ]; do
       echo $i
       i=$((i+1))
   done
   ```

7. 数组和字典
   shell 的数组是基于字符串的，数组没有类型，都是字符串。数组名和等号之间不能有空格，数组名只能包含字母、数字和下划线，且不能以数字开头。如果需要空格需要使用引号。

   ```bash
   arr=(1 2 3 4 5)
   arr[0]=1
   arr[1]=2
   arr[2]=3
   arr[3]=4
   arr[4]=5
   echo ${arr[0]}
   echo ${arr[1]}
   echo ${arr[2]}
   echo ${arr[3]}
   echo ${arr[4]}
   echo ${arr[@]}
   echo ${#arr[@]}
   echo ${!arr[@]}
   ```

   字典，在 shell 中的键值对数据容器，其声明必须使用 `declare -A` 命令，并且键值对之间使用空格隔开，键和值之间使用等号隔开。

   ```bash
   declare -A dict
   dict["key1"]="value1"
   dict["key2"]="value2"
   dict["key3"]="value3"
   echo ${dict["key1"]}
   echo ${dict["key2"]}
   echo ${dict["key3"]}
   echo ${dict[@]}
   echo ${#dict[@]}
   echo ${!dict[@]}
   ```

8. 正则表达式
   shell 中的正则表达式和 Perl 中的正则表达式语法基本一致，包括元字符、转义字符、字符类、量词、断言等。常用的正则表达式命令有 grep、sed、awk 等。

   ```bash
   echo "hello world" | grep "hello"
   echo "hello world" | sed "s/hello/HELLO/g"
   echo "hello world" | awk '{print $1}'
   echo "hello world" | tr 'o' 'O'
   ```

9. 文件操作
   shell 中的文件操作包括文件的创建、删除、重命名、复制、移动、查看等操作。常用的文件操作命令有 touch、rm、mv、cp、ls、cat、more、less、head、tail、grep 等。

   ```bash
   touch file
   rm file
   mv file newfile
   cp file newfile
   ls
   cat file
   more file
   less file
   head file
   tail file
   grep "hello" file
   ```

10. 函数

    函数定义，function、()、{}。函数可以接受参数，参数可以通过 `$1`、`$2` 等来获取，也可以通过 `$@` 来获取所有参数。函数的状态值可以通过 return 语句进行返回，也可以通过函数体中最后一条语句的执行状态进行返回。函数的状态值可以通过 $? 来获取，$? 表示上一条命令的执行状态，0 表示成功，非 0 表示失败。区分于其他高级语言，在 shell 中，函数的“返回值”是一个整数，表示函数的执行状态，0 表示成功，非 0 表示失败。而想要达到像其他高级语言中的返回一个值和变量的效果需要借助 `echo` 命令，通过标准输入和标准输出来完成。echo 返回一个字符串，通过管道或者重定向来获取，代表这个程序的标准输出，该字符串以换行符结束，在同一个程序中可以输出多个字符串，字符串之间使用换行符隔开，通过捕获`$()`和重定向`>`输出来获取函数的返回值。

    ```bash
    function myfunc() {
        echo "hello $1"
    }

    myfunc "world"
    ```

11. 管道与重定向

    在 shell 中，标准输入、标准输出和标准错误输出是三个重要的概念。标准输入是程序从外部获取数据的通道，标准输出是程序向外部输出数据的通道，标准错误输出是程序向外部输出错误信息的通道。而在 shell 中，标准输入、标准输出和标准错误输出默认都是终端，也就是命令行窗口。一个函数、命令、程序的执行结果，默认都会输出到标准输出。

    可以通过 `>` 来重定向标准输出，`2>` 来重定向标准错误输出，`&>` 来重定向标准输出和标准错误输出。重定向的文件可以是一个文件，也可以是一个设备，比如 `/dev/null`。`/dev/null` 是一个特殊的设备，它将所有的数据都丢弃，相当于一个黑洞。

### 命令与面向过程
在 shell 中，需要通过输入命令来执行相应的逻辑，一般命令是一行语句，以换行符结束。shell 会逐行读取用户输入的命令，并把它传递给内核去执行。命令的格式是一行语句，以一个命令名称开始，后面跟一串参数，参数之间使用空格隔开。

如何获取命令？命令就是一个程序，**shell 读取机器上快速路径 path 中的全局程序，或者从当前脚本中读取声明的函数，或者从文件系统中指定程序**。

执行一行命令的时候，shell 会创建一个子进程来执行命令，子进程会继承父进程的文件描述符，然后子进程会执行命令，当命令执行完毕后，子进程会退出，然后 shell 会继续等待用户输入命令。这也可以认为是面向过程的编程方式。

在 shell 中，可以使用 shell 函数的方式来拆分逻辑，一个函数就相当于一个命令，一个过程，一个小型的程序。可以通过定义函数，来定义一个命令，函数体是多行子命令，函数体中的代码可以调用其他命令，也可以调用其他函数。

### 函数与命令行参数
每条命令行语句都由命令和参数组成，命令是程序的名字，参数是传递给程序的选项，参数可以是一个，也可以是多个。每行的第一个单词是命令，后面的单词是参数，参数会在命令执行时被传递给命令，以一个数组的形式进行传递，数组的索引下标 0 为命令，下标 1 开始为参数。

函数的状态值，函数的状态值是一个整数，表示函数的执行状态，0 表示成功，非 0 表示失败。函数的状态值可以通过 return 语句进行返回，也可以通过函数体中最后一条语句的执行状态进行返回。函数的状态值可以通过 $? 来获取，$? 表示上一条命令的执行状态，0 表示成功，非 0 表示失败。

### 环境变量
shell 在进行执行的时候，会将系统中定义的环境变量加载到 shell 中，作为 shell 的变量来供给 shell 脚本中的逻辑进行使用，可以通过`echo`命令来查看环境变量，也可以通过`export`命令来设置环境变量，环境变量是全局变量，可以在整个 shell 中访问。
