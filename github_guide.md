# GitHub 远程仓库操作指引

本文档为您提供了从初始化、处理冲突到推送代码至 GitHub 远程仓库 `https://github.com/sassisyphe/similar_segment.git` 的完整操作指南。

## 1. 首次初始化并推送到远程仓库

如果您的本地项目还未成功与远程仓库建立连接，请按照以下步骤进行：

### 1.1 初始化并添加文件
如果您之前 `git init` 并且撤销过，我们可以重新走一遍标准流程。
在项目根目录 `/disk0/tanghaoran/myproj` 下执行：

```bash
# 初始化 Git 仓库（如果尚未初始化）
git init

# 将所有文件添加到暂存区
git add .

# 提交第一次修改
git commit -m "Initial commit: Add module1-4 and analysis scripts"
```

### 1.2 关联远程仓库并推送
由于您已经有一个现成的远程仓库，执行以下命令将其关联：

```bash
# 添加远程仓库地址 (origin 是默认的远程仓库名称)
git remote add origin https://github.com/sassisyphe/similar_segment.git

# 确保当前分支名称为 main (GitHub 现在的默认主分支名)
git branch -M main

# 推送到远程仓库
git push -u origin main
```

**⚠️ 注意：** 
如果远程仓库 `https://github.com/sassisyphe/similar_segment.git` 中已经存在文件（例如 README.md 或 LICENSE），直接 `git push` 会报错（提示 updates were rejected）。
**解决方法**：先拉取远程文件并合并，然后再推送：
```bash
# 拉取远程 main 分支并允许合并不相关的历史记录
git pull origin main --allow-unrelated-histories

# 解决可能出现的冲突后，再次推送
git push -u origin main
```

---

## 2. 日常更新与推送流程

在日常开发中，当您修改了代码或生成了新的分析报告后，请按照以下“三步曲”将更新推送到 GitHub：

### 第一步：查看状态与添加文件
```bash
# 查看哪些文件被修改了
git status

# 将所有改动（包括新增和删除）添加到暂存区
git add .
# 或者只添加特定文件： git add module1_download.py
```

### 第二步：提交修改
```bash
# 编写清晰的提交信息，说明本次修改了什么
git commit -m "Update: Fix baostock API string format in module1 and add MFI indicator"
```

### 第三步：推送到 GitHub
```bash
# 因为首次推送时已经使用了 -u 绑定了上游分支，后续只需直接 push 即可
git push
```

---

## 3. 常见问题与解决办法

### 3.1 遇到代码冲突 (Merge Conflict)
如果您在多台设备上开发，或者有其他人修改了远程仓库，在 `git push` 时可能会失败。
**解决流程：**
1. 先拉取最新代码：`git pull`
2. Git 会提示哪些文件存在冲突，打开这些文件，您会看到类似 `<<<<<<< HEAD` 和 `>>>>>>>` 的标记。
3. 手动修改这些文件，保留您想要的代码，删除标记符。
4. 重新提交：
   ```bash
   git add .
   git commit -m "Merge conflict resolved"
   git push
   ```

### 3.2 误添加了不想上传的大文件
如果您的项目里生成了非常大的数据文件（如 `data/daily/*.parquet`）或大量的图片缓存，直接 `git add .` 会导致仓库过于庞大，甚至超过 GitHub 的限制。

**解决方法：使用 `.gitignore`**
在项目根目录创建一个名为 `.gitignore` 的文件，写入以下内容：

```text
# 忽略虚拟环境
venv/

# 忽略数据文件
data/
*.parquet
*.csv

# 忽略缓存和日志
__pycache__/
*.log
run_full.log

# 忽略生成的图片 (如果您不想上传图片的话)
# plots/
```
创建好后，执行 `git add .gitignore` 并提交。这样 Git 就会自动忽略这些不需要追踪的文件。

### 3.3 GitHub 认证失败 (Authentication failed)
自 2021 年起，GitHub 不再支持密码验证。如果您在推送时遇到权限问题：
- **推荐方法**：使用 [GitHub Personal Access Token (PAT)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) 作为密码输入。
- **配置 SSH**：如果您觉得每次输入 Token 麻烦，可以在宿主机生成 SSH Key 并添加到 GitHub 账号中，然后将远程地址改为 SSH 格式：
  `git remote set-url origin git@github.com:sassisyphe/similar_segment.git`