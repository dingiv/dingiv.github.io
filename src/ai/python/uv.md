# uv
uv（2023 年起逐步流行）是围绕 Python 项目管理与依赖解决方案发展的工具集。它以 `pyproject.toml` 为配置中心、基于虚拟环境（venv）进行项目级环境隔离，目标是保留 `pip`/`venv` 的简单性，同时补齐现代项目管理、可重复构建和 CI 集成的需求。

## 特性简介
- 目标：提供一种轻量、可复现的项目管理方式，降低对全局 Python 配置的依赖。
- 配置文件：以 `pyproject.toml` 为主，兼容从 `requirements.txt` 导入/导出依赖清单。
- 环境：在项目目录下使用独立虚拟环境（典型为 `.venv/` 或项目专有目录），避免全局包污染。
- 可用场景：个人项目、团队协作、CI/CD 管道以及与科学计算仓库（如 conda）互操作的项目。

> pyproject.toml 是 python 官方推行的项目管理方案，可以通过 pip + venv 的方式来导入和导出，uv 是 pip 的社区升级版本，实现官方的规范来进行管理，使用 venv 作为自己的虚拟环境管理。

## 基本使用
uv 基于 venv 在项目级别管理 python 环境，推荐在项目根目录维护一个 `pyproject.toml`，其中包含项目元信息与依赖声明。下面是一个简化示例：

```toml
[project]
name = "myproject"
version = "0.1.0"
description = "示例项目"

[project.dependencies]
numpy = "^1.25"
requests = "^2.31"

[tool.uv]
venv = ".venv" # 可选：指定项目虚拟环境目录
lock-file = "uv.lock" # 可选：锁文件位置
```

如果项目需要兼容旧工具链，可以通过导出 `requirements.txt`：

```text
# uv 将 pyproject 的依赖导出为 requirements.txt 的示例输出
numpy==1.25.0
requests==2.31.0
```

### 依赖与虚拟环境管理

- 创建/激活项目虚拟环境：工具通常会在项目目录下生成 `.venv/`（或指定目录），并在本地作用域管理解释器与已安装包。
- 安装依赖：在项目配置文件中声明依赖后，使用工具的安装命令将依赖解析并写入锁文件，保证可重复安装。
- 升级/移除依赖：修改 `pyproject.toml` 或使用工具提供的子命令更新依赖，同时更新锁文件。
- 导出/兼容：提供将锁文件或配置导出为 `requirements.txt` 的能力，以兼容某些 CI 或部署环境。

示例典型工作流（伪命令与步骤，仅作说明）：

1. 初始化项目并创建虚拟环境：`uv init`（或 `uv env create`）
2. 安装当前配置的依赖并生成锁文件：`uv install` -> 生成 `uv.lock`
3. 添加新依赖（示例）：`uv add flask`
4. 导出兼容 `requirements.txt`：`uv export -f requirements.txt`

（实际子命令名称请参考 `uv` 版本文档）

## 与其它工具的对比
python 生态中，包仓库被分成了两座山头，一个是官方的 pypi，一个是 anoconda 公司维护的 conda 仓库，两个仓库在一定程度上存在隔阂，各自使用相应的包管理器。

```
pip(pypi) + venv -----> poetry(pypi) + venv --> uv(pypi) + venv --
                                                                 |-----> pixi
conda(conda-forge) ---> mamba(conda-forge)-----------------------
```

- `pip + venv`：这是 Python 官方最基础的组合，简单但在依赖锁定、可复现构建、团队协作方面能力有限。`uv` 在保持这种简洁的同时补充锁文件和项目级配置支持。
- `poetry`：poetry 也使用 `pyproject.toml` 并包含依赖解析、打包与发布功能。`uv` 的设计目标更偏向与现有 `pip` 生态兼容，并降低学习曲线（视具体实现而定）。
- `conda` / `mamba`：针对科学计算与二进制包提供了强大的仓库与环境管理能力，但其全局和 shell 级配置有时难以和项目级管理模式对齐。社区出现了像 `pixi` 这样的工具，旨在结合 `uv` 的项目级管理与 `conda` 仓库的二进制包能力。
- `pixi`：定位为下一代包与构建工具，借鉴了 nodejs 的 `package.json` 与 `node_modules` 思路，将包与解释器管理与项目紧密耦合，常与 `uv` 的依赖解析或仓库访问能力互补。
