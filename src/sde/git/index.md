---
title: Git
order: 20
---

# Git

Git 是分布式版本控制系统，是现代软件工程的基础设施。Git 记录代码的所有变更历史，支持多人协作开发，提供分支管理、代码审查、版本回退等功能。

## 约定式提交

约定式提交（Conventional Commits）规范提交信息格式，使提交历史清晰易读。

### 提交类型

feat：新功能、新特性

fix：修改 bug

perf：更改代码，以提高性能（在不影响代码内部行为的前提下，对程序性能进行优化）

refactor：代码重构（重构，在不影响代码内部行为、功能下的代码修改）

docs：文档修改

style：代码格式修改，注意不是 css 修改（例如分号修改）

test：测试用例新增、修改

build：影响项目构建或依赖项修改

revert：恢复上一次提交

ci：持续集成相关文件修改

chore：其他修改（不在上述类型中的修改）

release：发布新版本

workflow：工作流相关文件修改

### 提交格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

type 是提交类型，scope 是影响范围（可选），subject 是简短描述（不超过 50 字符）。

示例：
```
feat(auth): add JWT authentication

Implement JWT-based authentication with access token and refresh token.

- Add login endpoint
- Add token refresh endpoint
- Add middleware for token validation

Closes #123
```

## Git 分支管理

### Git Flow

Git Flow 是经典的分支管理策略，适合有明确发布周期的项目。

main：主分支，只包含稳定发布版本

develop：开发分支，集成所有功能开发

feature：功能分支，从 develop 分支，完成后合并回 develop

release：发布分支，从 develop 分支，准备发布时使用

hotfix：修复分支，从 main 分支，紧急修复使用

### GitHub Flow

GitHub Flow 是简化的分支管理策略，适合持续部署项目。

main：主分支，始终可部署

feature：功能分支，从 main 分支，通过 PR 合并到 main

### Trunk Based Development

基于主干的开发，所有开发在主分支进行，频繁发布。

## Git Hook

Git Hook 是 Git 在特定事件触发的脚本，可用于代码检查、自动测试、格式化等。

### 常用 Hook

pre-commit：提交前触发，可用于代码格式检查、运行测试

commit-msg：提交信息编辑后触发，可用于验证提交格式

pre-push：推送前触发，可用于运行完整测试套件

### Husky

Husky 是 Git Hook 管理工具，简化 Hook 配置。

```json
{
  "husky": {
    "hooks": {
      "pre-commit": "npm run lint",
      "commit-msg": "commitlint -E HUSKY_GIT_PARAMS"
    }
  }
}
```

## Git 最佳实践

频繁提交：小步快跑，每次提交包含一个完整的逻辑单元

清晰提交：提交信息准确描述变更内容，便于历史追溯

原子提交：每次提交只做一件事，不包含无关变更

审查合并：代码审查通过后再合并到主分支

保持干净：定期清理已合并的分支，保持分支列表简洁

Git 是工程化的基础，理解 Git 的工作原理和最佳实践，有助于高效协作开发。
