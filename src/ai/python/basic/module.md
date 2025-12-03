# 模块系统
模块系统是现代语言的基本能力，提供了代码组织、封装、分享和复用的强大机制。

## 模块层级
python 模块（module）是一个包含 Python 代码（函数、类、变量等）的文件，通常以 .py 扩展名存储。模块允许将代码分解为逻辑单元，支持复用和命名空间隔离。文件名就是模块名。

python 包，多个模块可以组织成包（package），即包含 `__init__.py` 文件的目录。该文件就是这个包的入口文件，同时在该目录下的所有的文件或者模块构成了这个包。目录名就是包名。一个包内部可以继续包含文件夹，被看作是一个子包。

```
project/
├── src/                    # 源码包 src
│   ├── __init__.py         # 源码包 src 的入口
│   ├── core/               # 核心包 core
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── models.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py
├── tests/
│   ├── test_database.py
├── main.py                 # 程序执行的入口
├── requirements.txt        # pip install 目录的安装列表，用于快速安装
├── pyproject.toml          # pip -m build 模块的打包逻辑，用于发布和构建 py 项目
```

## 模块导入
python 库包括标准库和三方库，标准库在 python 解释器的文件夹中的 `Lib` 目录中，第三库位于 `Lib/site-packages` 目录下。

模块导入有两种方式，一个是绝对导入，一个是相对导入。绝对导入从 **sys 模块中的 `sys.path` 变量**中的路径进行导入。`sys.path` 该变量是一个字符串数组，保存着全局包的保存路径，默认的 `Lib` 和 `Lib/site-packages` 默认位于该变量中。更全面地，在该变量中包含，有顺序之分，排在前的优先级更高。
1. 当前程序执行的 cwd
2. `PYTHONPATH` 环境变量中定义的路径
3. 当前项目的虚拟环境中的 `Lib` 和 `Lib/site-packages`
4. 全局包路径 `Lib` 和 `Lib/site-packages`

相对导入以 `.` 开头路径使用相对路径，相对路径指的是**相对于当前的执行脚本**的路径。这个点和绝对导入的 cwd 不同，cwd 会因为程序的执行 cwd 而发生变化，这导致脚本依赖于特定的执行路径，而相对路径依赖的是脚本之间的相对位置。

```py
# 全局导入 math_utils
import math_utils
# 简化引用
import math_utils as mu
# 只导入所需内容
from math_utils import add, PI
# 将 math_utils 中的所有符号，导入到当前脚本中的命名空间中
from math_utils import *
# 相对导入
from .dir1.mod1 import add
```

可以通过在执行 import 之前，修改 sys.path 的值，从而使得绝对导入可以导入自定义路径下的包。
```py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入 custom_pkg
import custom_pkg
```

## 虚拟环境 venv
python 虚拟化环境是 python 标准库中自带的一个多项目环境隔离的能力，为了防止全局包的版本冲突问题，你可以选择在当个 python 项目中使用虚拟环境来进行开发，从而隔离同一个机器上的不同项目之间的第三包的版本不同问题，同时也可以让 python 项目的可迁移性变得更好。它就相当于一个全新的 python 解释器环境，但是其二进制共用本机上的原本 python 环境，但它包含自己独立的第三方包依赖。

```bash
# 执行模块 venv，参数为 .venv，这将在当前的目录下创建一个目录 .venv 作为一个虚拟环境
python -m venv .venv

# 在执行项目前，在当前的终端中使用“激活脚本”，从而使用该 venv
source .venv/bin/activate

# 执行项目
python main.py
```

## 项目管理
Python 项目管理包括，构建、打包、发布、依赖管理等。发布是从开发环境部署到生产环境前的准备工作，或者将自己编写的包发布到远程的 pypi 服务器中。

### setup.py
旧版本 python 项目管理技术，通过 requirements.txt 来管理的依赖。

### pyproject.toml
`pyproject.toml` 是 Python 现代项目的标准配置文件，取代了 **setup.py + requirements.txt** 的老方案。它是 PEP 518/517/621 规范定义的统一入口文件，包含项目元数据 + 构建配置 + 依赖管理。在项目构建发布的时候使用。出于安全考虑，使用了 pyproject.toml，取消了 setup.py 的动态执行的能力。不过，由于 requirements.txt 配置简单，使用广泛，pyproject.toml 依然可以与 requirements.txt 混用。

通过配置一个 `pyproject.toml` 文件，定义项目的依赖和数据，从而通过 `build` 模块进行打包和构建。构建完成后，在 dist 目录下将会出现一个压缩包，便是构建产物。
```bash
# 安装 build 构建模块
pip install build
# 启动 构建
python -m build
```

### 二进制包
使用 pyinstaller 可以将一个 python 项目打包成为一个可独立运行的二进制可执行文件，用户可以直接使用无需安装依赖，对于 toC 产品、网络不佳环境、不方便安装 python 环境的机器来说非常重要。

```bash
python -m pip install pyinstaller
python -m pip install -r requirements.txt
pyinstaller --onefile main.py
```