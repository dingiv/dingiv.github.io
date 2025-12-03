# uv
uv 是于 2023 年发布的 python 社区新一代项目管理工具，作为目前的主流使用方案。

## pyproject.toml + requirements.txt
pyproject.toml 是 python 官方推行的项目管理方案，可以通过 pip + venv 的方式来导入和导出，但是这个文件的功能仅限依赖管理，uv 是 pip 的社区升级版本，实现官方的规范来进行管理，使用 venv 作为自己的虚拟环境管理。

## 其他包管理


### pip
pip + venv 是 python 自带的包管理和虚拟环境隔离工具，但是，由于功能有限，社区开发出更加强大的管理工具。

### poetry

### conda
conda 是科学计算社区搞的一个 python 仓库源，可以下在很多 pypi 仓库中没有的库，特别是和科学计算、人工智能领域相关的，并且附带了虚拟环境管理的能力，但是，由于 conda 开始收费，并且，另一方面，conda 基于 shell 环境变量来管理环境，在一定程度上和现代语言的基于单文件夹项目的管理风格脱节了。所以社区开始探索免费的方案，即 pixi。pixi 在下载方面集成 uv 的能力，在环境管理方面集成 anaconda 的仓库，支持从 pypi 和 anaconda 仓库下载包，同时，受到 Rust 社区的 cargo 管理器的启发，使用项目级别和文件夹级别的环境管理模式，简化了 conda 的繁琐全局配置。

### pixi
pixi 是下一代的 python 包管理、项目管理、构建工具。

```
pip(pypi) + venv -----> poetry(pypi) + venv --> uv(pypi) + venv --
                                                                 |-----> pixi
conda(conda-forge) ---> mamba(conda-forge)-----------------------
```

pixi.toml 是 pixi 的项目工程文件，基本上可以认为是 nodejs 的 package.json 的翻版，同时，pixi 将会把当前项目使用的包放到 .pixi 目录下，同时，在全局维护下载的包的缓存文件，减少重复下载依赖包，对比 nodejs 的 node_modules 文件夹；并且，更加一步地，pixi 支持管理 python 解释器，这个点相当于 nodejs 生态的 nvm 管理器。下载了 pixi 之后，无需也不建议再在电脑上安装其他的 python。
