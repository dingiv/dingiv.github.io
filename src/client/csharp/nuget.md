# NuGet 包管理

NuGet 是 .NET 的包管理器，类似于 Python 的 pip、Node.js 的 npm。NuGet 使得开发者可以轻松地发现、安装、更新和移除 .NET 项目中的依赖库。NuGet 包是编译好的代码 (DLL)、相关文件和元数据的压缩包，文件扩展名为 .nupkg。

## 包结构

NuGet 包由多个文件组成，其中最重要的是 .nuspec 文件（包清单）和编译后的程序集。包清单文件描述了包的元数据，包括 ID、版本、作者、依赖关系等信息。

```
package.nupkg
├── [Content_Types].xml
├── _rels/
├── package/
│   ├── lib/
│   │   └── net6.0/
│   │       └── MyLibrary.dll
│   └── MyLibrary.nuspec
```

lib 文件夹包含针对不同框架版本的程序集，如 net461 (.NET Framework 4.6.1)、netstandard2.0 (.NET Standard)、net6.0 (.NET 6)。NuGet 会根据项目的目标框架选择合适的程序集版本。

## dotnet CLI

dotnet CLI 是 .NET 的命令行工具，提供了创建、构建、运行、测试、发布项目等功能，同时也包含了包管理命令。

```bash
# 添加包
dotnet add package Newtonsoft.Json

# 添加特定版本
dotnet add package Newtonsoft.Json --version 13.0.1

# 添加项目引用
dotnet add reference ../MyProject/MyProject.csproj

# 移除包
dotnet remove package Newtonsoft.Json

# 列出所有包
dotnet list package

# 还原依赖
dotnet restore

# 发布为 NuGet 包
dotnet pack
```

## 项目文件

.NET 项目使用 .csproj 文件描述项目配置，包括包引用、框架版本、编译选项等。现代 .NET Core/.NET 5+ 使用 SDK 风格的项目文件，大大简化了配置。

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net6.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />
    <PackageReference Include="Microsoft.Extensions.Logging" Version="6.0.0" />
  </ItemGroup>
</Project>
```

ItemGroup 中的 PackageReference 定义了项目依赖的 NuGet 包，Version 属性指定版本范围。版本号遵循语义化版本规范，如 1.2.3 表示主版本.次版本.补丁版本。

## 版本范围

NuGet 支持多种版本范围指定方式：

```
1.0.0       精确版本
1.0.0-*     兼容版本，主版本和次版本相同
1.0.*       同 1.0.0-*
[1.0.0]      精确版本，方括号表示包含
(1.0.0,)     大于 1.0.0，圆括号表示不包含
[1.0.0,2.0.0] 1.0.0 到 2.0.0 之间的任何版本
```

最常用的是浮动版本 `*`，它会自动获取补丁版本或次版本更新。例如 `1.0.*` 会安装 1.0.x 的最新版本，`1.*` 会安装 1.x.x 的最新版本。

## 全局工具

NuGet 支持全局安装 .NET 工具，这些工具可以在系统的任何位置运行。全局工具类似于 Node.js 的全局 npm 包。

```bash
# 安装全局工具
dotnet tool install --global dotnet-format

# 更新全局工具
dotnet tool update --global dotnet-format

# 卸载全局工具
dotnet tool uninstall --global dotnet-format

# 列出已安装的全局工具
dotnet tool list --global
```

## 本地工具

.NET Core 3.0 引入了本地工具，工具安装到项目目录而非全局。本地工具通过 manifest 文件管理，每个项目可以有自己的一组工具。

```bash
# 初始化工具 manifest
dotnet new tool-manifest

# 安装本地工具
dotnet tool install dotnet-format

# 运行本地工具
dotnet dotnet-format
```

## 私有源

除了官方的 nuget.org，企业可以搭建私有 NuGet 源来管理内部包。Azure Artifacts、MyGet、Sonatype Nexus 都提供了私有 NuGet 源的功能。

```bash
# 添加源
dotnet nuget add source https://api.nuget.org/v3/index.json -n nuget.org

# 添加私有源
dotnet nuget add source https://company.com/nuget/v3/index.json -n CompanyFeed

# 列出所有源
dotnet nuget list source

# 设置默认源
dotnet nuget disable source nuget.org
```

## 包发布

发布 NuGet 包需要先创建 .nuspec 文件或使用项目文件中的属性，然后使用 dotnet pack 命令打包。发布可以使用 dotnet nuget push 或 NuGet.exe。

```bash
# 打包
dotnet pack -c Release

# 发布到 nuget.org
dotnet nuget push MyPackage.1.0.0.nupkg --api-key YOUR_API_KEY --source https://api.nuget.org/v3/index.json

# 发布到私有源
dotnet nuget push MyPackage.1.0.0.nupkg --source https://company.com/nuget/v3/index.json
```

## 多目标框架

现代项目可以同时支持多个目标框架，如 .NET Framework 和 .NET Core/.NET 5+。项目文件中使用 TargetFrameworks 属性（复数）指定多个目标。

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFrameworks>net6.0;net48</TargetFrameworks>
  </PropertyGroup>

  <ItemGroup Condition=" '$(TargetFramework)' == 'net48' ">
    <PackageReference Include="System.Net.Http" Version="4.3.4" />
  </ItemGroup>
</Project>
```

条件引用可以为不同目标框架选择不同的依赖，充分利用 #if 预处理指令编写平台特定代码。
