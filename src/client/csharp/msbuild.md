# MSBuild 构建系统

MSBuild 是 Microsoft 的构建平台，用于编译、测试和部署 .NET 项目。.csproj 项目文件本质上是 MSBuild 的 XML 脚本，定义了构建过程的各个阶段和目标。现代 .NET SDK 风格的项目大大简化了 MSBuild 配置，但理解 MSBuild 的工作原理对于高级场景仍然必要。

## 项目文件格式

.NET 项目文件使用 XML 格式，根元素是 Project。传统项目文件使用复杂的 XML 结构，现代 SDK 风格项目只需要几个属性。

```xml
<!-- 传统项目文件 (复杂) -->
<Project ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <Import Project="$(MSBuildToolsPath)\Microsoft.CSharp.targets" />
  <PropertyGroup>
    <Configuration Condition=" '$(Configuration)' == '' ">Debug</Configuration>
    <Platform Condition=" '$(Platform)' == '' ">AnyCPU</Platform>
    <ProjectGuid>{12345678-1234-1234-1234-123456789ABC}</ProjectGuid>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include="System" />
    <Reference Include="System.Core" />
    <Compile Include="Program.cs" />
  </ItemGroup>
</Project>

<!-- 现代 SDK 风格项目 (简洁) -->
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>
</Project>
```

SDK 风格项目隐式引入了常用的目标和属性，大大简化了配置。Sdk 属性指定了使用的 SDK，Microsoft.NET.Sdk 是最常用的 SDK，适用于大多数项目类型。

## 目标和任务

MSBuild 的核心概念是目标 (Target) 和任务 (Task)。目标是一组任务的集合，任务是执行特定操作的单元。

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <!-- 自定义目标 -->
  <Target Name="PrintInfo" BeforeTargets="Build">
    <Message Text="Building $(ProjectName)" Importance="high" />
  </Target>

  <!-- 任务调用 -->
  <Target Name="CustomTask" AfterTargets="Build">
    <Copy SourceFiles="$(OutputPath)$(AssemblyName).dll"
          DestinationFolder="../Bin/" />
  </Target>
</Project>
```

常用的内置任务包括 Copy（复制文件）、Delete（删除文件）、Message（输出消息）、Exec（执行命令）。目标之间可以定义依赖关系，BeforeTargets 和 AfterTargets 属性指定目标的执行顺序。

## 属性和项

MSBuild 使用属性 (Property) 和项 (Item) 来管理配置和数据。属性是键值对，项是一组文件的集合。

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net6.0</TargetFramework>
    <Version>1.0.0</Version>
    <Authors>John Doe</Authors>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />
  </ItemGroup>

  <ItemGroup>
    <Compile Include="**/*.cs" />
    <EmbeddedResource Include="**/*.resx" />
    <None Include="**/*.txt" CopyToOutputDirectory="PreserveNewest" />
  </ItemGroup>
</Project>
```

属性可以使用 `$(PropertyName)` 语法引用，支持属性组合和条件判断。项可以使用 Include、Exclude、Update、Remove 等操作进行过滤和修改。

## 构建配置

MSBuild 支持多个构建配置，常见的有 Debug 和 Release。配置决定了编译器的优化选项、调试信息的生成方式等。

```bash
# Debug 构建
dotnet build -c Debug

# Release 构建
dotnet build -c Release

# 自定义配置
dotnet build -c:CustomConfig
```

Debug 配置禁用优化、生成完整的调试信息、定义 DEBUG 符号。Release 配置启用优化、减少调试信息、不定义 DEBUG 符号。

## 多目标框架

项目可以同时支持多个目标框架，使用 TargetFrameworks (复数) 属性指定。

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFrameworks>net6.0;net48</TargetFrameworks>
  </PropertyGroup>

  <!-- 条件引用 -->
  <ItemGroup Condition=" '$(TargetFramework)' == 'net48' ">
    <Reference Include="System.Net.Http" />
  </ItemGroup>
</Project>
```

多目标项目会针对每个框架编译一次，生成多个输出目录。

## 条件编译

可以使用条件编译符号在编译时选择性地包含或排除代码。

```xml
<PropertyGroup Condition=" '$(Configuration)' == 'Release' ">
    <DefineConstants>RELEASE</DefineConstants>
</PropertyGroup>
```

```csharp
#if DEBUG
    Console.WriteLine("Debug mode");
#elif RELEASE
    Console.WriteLine("Release mode");
#else
    Console.WriteLine("Other mode");
#endif
```

## 依赖关系

项目之间可以定义依赖关系，被依赖的项目会先编译。

```xml
<ItemGroup>
  <!-- 项目引用 -->
  <ProjectReference Include="..\MyProject\MyProject.csproj" />

  <!-- 程序集引用 -->
  <Reference Include="System.Data.SqlClient" />
</ItemGroup>
```

项目引用会自动处理依赖传递，如果被引用的项目引用了其他项目，这些依赖也会被解析。

## 自定义目标

开发者可以编写自定义目标和任务来扩展构建过程。Task 是继承自 Task 类的 C# 类，使用 Output 属性定义输出。

```csharp
// 自定义任务
public class CustomTask : Task
{
    [Required]
    public string InputFile { get; set; }

    [Output]
    public string OutputFile { get; set; }

    public override bool Execute()
    {
        // 执行任务逻辑
        OutputFile = ProcessFile(InputFile);
        return true;
    }
}
```

```xml
<!-- 使用自定义任务 -->
<Project>
  <UsingTask TaskName="CustomTask" AssemblyFile="CustomTasks.dll" />

  <Target Name="UseCustomTask">
    <CustomTask InputFile="input.txt">
      <Output TaskParameter="OutputFile" PropertyName="Result" />
    </CustomTask>
    <Message Text="Result: $(Result)" />
  </Target>
</Project>
```

## dotnet build 命令

dotnet build 是常用的构建命令，支持多种选项。

```bash
# 基本构建
dotnet build

# 指定配置
dotnet build -c Release

# 指定目标框架
dotnet build -f net6.0

# 指定输出目录
dotnet build -o ./bin/Release

# 不还原依赖
dotnet build --no-restore

# 生成详细日志
dotnet build -v detailed

# 并行构建
dotnet build -m
```

dotnet build 内部调用 MSBuild，大部分 MSBuild 命令也可以通过 dotnet msbuild 直接调用。

## 目录.build.props

自定义的 Directory.Build.props 文件可以统一配置多个项目的构建选项，避免在每个项目中重复配置。

```xml
<!-- Directory.Build.props (放在解决方案根目录) -->
<Project>
  <PropertyGroup>
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.CodeAnalysis.NetAnalyzers" Version="6.0.0">
      <PrivateAssets>all</PrivateAssets>
      <IncludeAssets>runtime; build; native; contentfiles; analyzers; buildtransitive</IncludeAssets>
    </PackageReference>
  </ItemGroup>
</Project>
```

Directory.Build.props 会被所有项目自动导入，适合配置代码分析、版本号、公司信息等全局设置。
