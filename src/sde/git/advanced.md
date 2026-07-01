# Git 高阶用法

日常开发中，`add`、`commit`、`push`、`pull`、`branch` 这些基础命令已经覆盖了大部分工作场景，但 Git 的真正威力远不止于此。当我们需要整理提交历史、在大仓库中只拉取部分文件、同时维护多个开发环境时，掌握 Git 的高阶命令能让工作效率成倍提升。这些命令在开源协同和大型项目工程化中尤其常见，理解其原理和适用场景是工程师从会用 Git 到精通 Git 的分水岭。

## 交互式变基 (rebase -i)

交互式变基是 Git 最强大的提交历史编辑工具，它允许我们对已有提交进行合并、修改、重排、删除等操作。在日常开发中，`rebase -i` 最常见的场景是在将 feature 分支合入主线之前，把多个零散的"修复 typo""补充注释"提交整理成逻辑清晰的几个提交，让项目历史保持干净。

```
git rebase -i HEAD~5
```

执行后会打开编辑器，列出最近 5 次提交，每行代表一个提交，前面是操作指令。常用指令包括 `pick`（保留该提交）、`reword`（修改提交信息）、`squash`（合并到上一个提交并保留信息）、`fixup`（合并且丢弃信息）、`drop`（删除该提交）、`edit`（暂停以便修改提交内容）。各指令的排列顺序决定了最终提交的顺序，你可以通过调整行序来重新组织提交。

假设你正在进行一项功能开发，产生了 5 个提交，其中第一和第三个是核心逻辑，其余都是随手的修复和临时改动。执行 `git rebase -i HEAD~5`，将提示的 5 行指令调整为：

```
pick abc1234 feat: 实现用户认证核心逻辑
fixup def5678 fix: 修复拼写错误
squash ghi9012 feat: 补充认证异常处理
drop jkl3456 debug: 临时打印日志
reword mno7890 docs: 更新相关文档说明
```

这个例子展示了 `rebase -i` 的典型使用模式：`fixup` 清理无意义的修复提交，`squash` 将紧密相关的提交合并，`drop` 删除临时调试代码，`reword` 修正不够清晰的提交信息。最终，杂乱无章的 5 个提交被整理成逻辑分明的 3 个提交。

需要注意的是，`rebase -i` 会改写提交历史，因此只应该对尚未推送到远端共享分支的本地提交使用。如果已经推送，rebase 后需要 `git push --force-with-lease` 来覆盖远端，此时要确保没有其他人基于旧的提交进行开发。`--force-with-lease` 比 `--force` 更安全，它会在推送前检查远端是否被其他人更新过，避免覆盖他人的提交。

在生产实践中，建议每个 feature 分支在合入之前都做一次交互式变基，这会显著提高 `git log` 的可读性，让代码审查者能够按提交顺序逐个理解变更意图，而不是在一堆"fix""update"中迷失方向。

## 部分检出 (sparse-checkout)

当你面对一个大型 monorepo 时，`clone` 整个仓库可能需要几分钟甚至更久，拉回几 GB 的历史和文件。如果只需要其中一部分目录进行开发，`sparse-checkout` 可以只检出指定的文件和目录，大幅减少网络传输和本地磁盘占用。

Git 2.25 引入的新接口让 sparse-checkout 的配置变得简单。以下命令让工作区只保留 `src/client` 和 `docs` 两个目录的内容：

```
git clone --filter=blob:none --sparse <repo-url>
cd <repo>
git sparse-checkout set src/client docs
```

其中 `--filter=blob:none` 是部分克隆（partial clone）的参数，它指示 Git 先不下载任何文件内容（blob），仅下载提交历史和树结构，文件内容在需要时才按需获取。配合 sparse-checkout，只有 `set` 指定的目录中的文件才会被实际拉取。这种组合非常适合 CI/CD 环境中只构建特定子项目、或者跨团队协作中只关注自己负责的模块的场景。

`sparse-checkout` 还支持添加和移除路径。`git sparse-checkout add <path>` 在不影响已有配置的情况下追加新路径，`git sparse-checkout disable` 则恢复完整工作区。当需求发生变化时，可以灵活调整工作区包含的内容。

sparse-checkout 的一个常见工程实践是在大型后端仓库中，前端团队只检出 `frontend/` 目录，后端团队只检出 `backend/` 目录，各自在轻量的工作区中开发，互不干扰。对于需要跨模块修改的场景，再临时添加需要的路径即可。

## 工作树 (worktree)

`git worktree` 允许你在同一个仓库中同时管理多个工作目录，每个工作目录对应不同的分支或提交。这对以下几类场景尤其有用：正在一个分支上开发大功能，需要紧急修复一个线上 bug，不想 stash 或 commit 当前半成品；需要同时在不同版本上运行测试或构建；跨分支对比文件差异而不频繁切换分支。

```
# 为 hotfix 分支创建一个新的工作树
git worktree add ../repo-hotfix hotfix

# 查看所有工作树
git worktree list
```

上述命令在 `../repo-hotfix` 目录下创建了一个全新的工作区，该目录 checkout 到了 `hotfix` 分支。两个工作树共享同一个 `.git` 仓库，因此不会重复存储提交历史，磁盘占用很小。你可以在原目录继续开发功能，在新目录修复 bug，两者互不影响。修复完成后，切换到 `../repo-hotfix` 提交并推送，然后用 `git worktree remove ../repo-hotfix` 清理这个临时工作树。

worktree 的另一个典型场景是与 `rebase -i` 配合使用。当你需要在分支 B 上验证分支 A 的编译情况时，与其 stash 当前修改再切换分支，不如创建一个临时的 worktree 指向分支 A，编译验证后直接删除，整个过程不动一根毛发。

需要注意的是，同一个分支不能同时在两个工作树中 checkout（Git 会拒绝这种操作）；每个工作树都可以独立地执行 add、commit 等操作，只是不能有分支冲突。在需要长时间并行的任务之间切换时，worktree 比 stash 更优雅——不需要担心 stash 列表越积越多，也不用担心半成品修改丢失。

## 补丁文件 (format-patch & am)

对于传统的邮件列表协作方式（Linux 内核社区至今仍在使用），补丁文件是最重要的代码交换方式。`format-patch` 将一系列提交导出为邮件风格的补丁文件（.patch），每封"邮件"包含提交信息、作者、时间、以及完整的 diff 内容；`am`（apply mailbox）则将这些补丁文件应用到目标仓库。

```
# 将最近 3 次提交导出为补丁文件
git format-patch HEAD~3

# 将指定范围的提交导出
git format-patch main..feature

# 应用补丁到目标仓库
git am *.patch
```

`format-patch` 生成的每个文件都是一封格式完整的邮件，主题行是提交信息的第一行，正文是完整提交信息，附件是 diff。这保证了补丁文件应用到任何仓库后，不仅代码变更一致，连原始作者、提交时间和描述信息也完整保留。这与 `diff` 命令不同，`diff` 只产生代码差异，不携带提交的元信息。

`am` 支持很多实用参数：`--signoff` 在补丁末尾追加 Signed-off-by 行（满足某些项目的合规要求），`--3way` 启用三方合并以降低冲突几率，`--continue`、`--skip`、`--abort` 提供了冲突处理流程。

在工程实践中，当两个团队使用不同的 Git 托管平台、无法直接 pull 和 merge 时（例如一方在内网无法访问外网仓库），用 `format-patch` 和 `am` 进行离线同步是非常可靠的方式。另一个场景是将上游开源项目的多个提交 cherry-pick 到自己维护的 fork 时，patch 文件比逐个 cherry-pick 更不容易遗漏或出错。

## Cherry-pick：精选提交

`cherry-pick` 可以将任意一个或多个提交的变更应用到当前分支，而不需要合并整个分支。这对于跨分支迁移 bug 修复非常有用——你修复了一个线上 bug 并提交到 release 分支，现在希望这个修复也出现在 main 和 develop 上，但不想把它们整个合并。

```
git cherry-pick <commit-hash>
```

一行命令，提交的变更就会被应用到当前分支并生成一个新的提交（内容相同、hash 不同）。`cherry-pick` 支持范围操作：`git cherry-pick A..B` 应用从 A（不包含）到 B（包含）之间的所有提交，`git cherry-pick A^..B` 则包含 A 本身。

多个提交连续 cherry-pick 时发生冲突是比较常见的现象。解决方式与 rebase 类似：解决冲突后 `git add`，然后 `git cherry-pick --continue` 继续；或者 `git cherry-pick --abort` 放弃整个操作。在工程中，cherry-pick 配合 `-x` 参数可以在生成的提交信息中自动附加原始 commit hash 的引用，方便后续追溯修复来源。

在实际应用中，cherry-pick 最适合处理"同一个 bug 在多个 release 分支上都需要修复"的场景。将修复合入主分支后，cherry-pick 到各个维护分支，简单直接且不容易引入无关变更。

## 二分查找 (bisect)

当引入了一个 bug 却不确定是哪个提交造成的，`git bisect` 可以通过二分法自动定位问题提交。这在大型项目中尤为实用——面对几十上百个候补提交，逐个手动检查显然不现实。

```
# 启动 bisect
git bisect start

# 标记当前版本有问题
git bisect bad

# 标记某个较早的版本没问题
git bisect good <known-good-commit>

# Git 自动 checkout 中间版本，测试后标记
git bisect good   # 或 git bisect bad

# 重复直到定位到问题提交
# 完成后退出
git bisect reset
```

`bisect` 的工作原理是二分查找：每次在待搜索区间选中中间提交，你测试后告诉 Git 这个提交是好是坏，Git 自动缩小范围。对于 N 个提交，最多只需 `log2(N)` 步就能精确定位。如果你可以编写自动化脚本来判断问题是否存在，还能进一步省去手动测试：`git bisect run ./test-script.sh`，全自动跑完整个流程。

bisect 在工程中的一个重要原则是每个提交应该都能正常工作（原子提交）。如果提交历史中夹杂了"编译失败""测试不通过"的中间提交，bisect 会失效——落入这些提交后无法判断功能好坏。这也是为什么保持干净的提交历史不仅仅是为了可读性，更是为了 bisect 能可靠地工作。

## 引用日志 (reflog)

`reflog` 是 Git 的"时间机器"，记录了本地仓库中 HEAD 和分支引用的所有变更历史。一旦你误操作——比如 rebase 错了、reset 丢了提交、或者 merge 了不想合并的分支——reflog 几乎总是能救你回来。

```
git reflog
```

输出会列出 HEAD 每次移动的时间和操作，每条记录都有对应的引用（`HEAD@{0}`、`HEAD@{1}` 等）。要恢复一个被 reset 掉的提交，只需找到它在 reflog 中的位置，然后 `git reset --hard HEAD@{n}` 或 `git checkout HEAD@{n} -- .` 即可。

需要注意的是，reflog 是一个纯本地机制，不会被推送到远端。它的默认保留期是 90 天（对不可达条目为 30 天），足够覆盖绝大多数误操作场景。对于关键仓库，可以通过 `gc.reflogExpire` 配置项延长保留时间。

reflog 是理解"Git 中几乎一切操作都可逆"这个事实的基础——只要你没有强制垃圾回收，数据就不会真正丢失。在 `git reset --hard` 之前看一眼 reflog 的当前指向，是很多资深工程师的习惯动作。

## 部分暂存 (git add -p)

`git add -p`（patch mode）允许交互式地将文件中的部分修改暂存，而非整个文件。这在开发中非常实用：你对一个文件做了多组修改，但它们应该分属不同的逻辑提交，用 `-p` 可以逐个选择哪些修改块加入暂存区。

执行后 Git 会在终端逐个展示修改块，并询问操作。常用选择：`y` 暂存当前块，`n` 跳过当前块，`s` 将当前块进一步拆分为更小的块，`e` 手动编辑。这一流程让每个提交都严格只包含一个逻辑单元，完美契合原子提交原则。

在工程中，养成使用 `add -p` 自我审查修改的习惯，可以减少很多事后用 `rebase -i` 重新整理的工作量。每块修改在进入暂存区之前都被重新审视过一次，很多碎片的代码问题和调试遗留能够在提交之前就被发现。

## 引用日志过滤与清理 (filter-branch & filter-repo)

当仓库中不小心提交了大文件、敏感信息或需要全局修改作者信息时，普通的提交级操作无能为力——需要遍历全部历史来重写。

`git filter-branch` 是传统工具，但已不推荐使用，它的性能差且容易出错。现代替代方案是 `git-filter-repo`（需单独安装），它不仅快得多，还提供了更清晰的参数接口和更完整的安全检查。

常见应用：清理历史上误提交的大文件以缩减仓库体积、从所有历史中移除敏感信息（密钥、密码）、统一修改所有提交的邮箱地址。这些操作都会彻底重写历史，执行前必须确保所有协作者知晓并重新 clone 仓库。

```
# 从所有历史中删除某个文件
git filter-repo --path unwanted-file --invert-paths

# 修改所有提交的作者邮箱
git filter-repo --email-callback 'return email.replace(b"old@example.com", b"new@example.com")'
```

在工程实践中，最需要 filter-repo 的场景往往是仓库体积膨胀——某次不小心提交了一个几百兆的二进制文件，虽然后来删除了，但 Git 历史中永远保留着它，导致每个 clone 都要下载这个大文件。用 filter-repo 清理后仓库体积可能缩小几十倍，效果立竿见影。

## 总结

Git 的高阶命令不是用来炫耀技巧的，每个都有它对应的工程问题与最佳场景。`rebase -i` 整理提交历史让仓库叙事清晰，`sparse-checkout` 解决大仓库的性能瓶颈，`worktree` 提供灵活的多任务并行能力，`format-patch` 和 `am` 打通离线协作的最后一公里，`bisect` 精准定位问题，`reflog` 防止灾难性误操作。这些工具组合使用，构成了一个工程师驾驭 Git 的完整技能树。

在实践中，关键是遇到问题时能意识到 Git 有对应的解决工具，然后查阅文档、理解原理、安全应用。不要盲目记忆参数，理解每个命令为什么存在、解决什么问题，才能在需要时信手拈来。
