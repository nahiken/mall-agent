#Requires -Version 5.1
<#
.SYNOPSIS
    把 mall-agent 整理成可安全上传 GitHub 的状态。

.DESCRIPTION
    做四件事：
      1. 预检：旧历史里是否出现过 .env / docs / 进展记录 / 密钥；
      2. 把旧 .git 移到仓库外备份（不删除你的任何源码）；
      3. 按 .gitignore 重新 git init，只保留一次干净提交；
      4. 可选：配置远程并推送。

    为什么不能只靠 .gitignore：
      已经 commit 过的文件会永远留在历史里，GitHub 上仍可翻到。
      要让"上次的内容"消失，只能换掉历史（本脚本的做法）。

.PARAMETER RemoteUrl
    GitHub 仓库地址，例：https://github.com/<you>/mall-agent.git
    不填则只做本地整理，不推送。

.PARAMETER Branch
    主分支名，默认 main。

.PARAMETER UserName / UserEmail
    新提交的作者信息；不填则沿用当前 git config。
    建议用 GitHub 的 noreply 邮箱，避免暴露真实邮箱。

.PARAMETER DryRun
    只做预检并列出将被提交的文件，不改动任何东西。

.EXAMPLE
    pwsh -File scripts\publish_clean.ps1 -DryRun

.EXAMPLE
    pwsh -File scripts\publish_clean.ps1 `
        -RemoteUrl https://github.com/yourname/mall-agent.git `
        -UserName "yourname" -UserEmail "12345678+yourname@users.noreply.github.com"
#>
[CmdletBinding()]
param(
    [string]$RemoteUrl = '',
    [string]$Branch = 'main',
    [string]$UserName = '',
    [string]$UserEmail = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Info($m) { Write-Host "  $m" }
function Ok($m)   { Write-Host "[OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[!]  $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "[x]  $m" -ForegroundColor Red; exit 1 }
function Section($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }

# ---------------------------------------------------------------- 0. 环境检查
Section '0. 环境检查'
if (-not (Test-Path 'pyproject.toml')) { Die "当前目录不像 mall-agent 根目录：$Root" }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Die '找不到 git，请先安装 Git for Windows' }
if (-not (Test-Path '.gitignore')) { Die '缺少 .gitignore，先补上再运行' }
Ok "仓库根目录：$Root"
Info ((git --version) -join '')

$hasGit = Test-Path '.git'

# ---------------------------------------------------------------- 1. 旧历史预检
Section '1. 旧历史预检（只读）'
if ($hasGit) {
    $commits = @(git rev-list --all 2>$null)
    Info "旧提交数：$($commits.Count)"

    $sensitivePaths = @('.env', 'docs', 'progress.md', 'task_plan.md', 'findings.md', 'plan_exec')
    foreach ($p in $sensitivePaths) {
        $hit = @(git log --all --oneline -- $p 2>$null)
        if ($hit.Count -gt 0) {
            Warn "历史中出现过：$p（$($hit.Count) 个提交）—— 必须重写历史"
        }
    }

    $envHistory = @(git log --all --oneline -- .env 2>$null)
    if ($envHistory.Count -gt 0) {
        Warn '.env 曾经被提交过，里面的 API Key 视为已泄露，请立刻去平台吊销并重新生成。'
    }

    $tracked = @(git ls-files 2>$null)
    Info "旧索引跟踪文件数：$($tracked.Count)"
}
else {
    Info '当前目录还没有 .git（首次初始化），跳过旧历史检查。'
}

# ---------------------------------------------------------------- 2. 敏感内容扫描
Section '2. 工作区敏感内容扫描'
$patterns = @(
    'sk-[A-Za-z0-9]{20,}',
    'AIza[0-9A-Za-z_\-]{20,}',
    'ghp_[0-9A-Za-z]{20,}',
    'AKIA[0-9A-Z]{16}'
)
$textExt = @('.py', '.md', '.yaml', '.yml', '.toml', '.txt', '.json', '.ps1', '.example', '.conf', '.ini')
$scanFiles = @()
foreach ($d in @('src', 'scripts', 'tests', 'config', 'deploy')) {
    if (Test-Path $d) {
        $scanFiles += Get-ChildItem -Path $d -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch '__pycache__' -and $textExt -contains $_.Extension }
    }
}
$scanFiles += Get-ChildItem -Path . -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -in @('README.md', 'README-project-struct.md', 'pyproject.toml', '.env.example') }

$found = $false
foreach ($pat in $patterns) {
    $hits = $scanFiles | Select-String -Pattern $pat -ErrorAction SilentlyContinue
    if ($hits) {
        $found = $true
        $hits | ForEach-Object { Warn "疑似密钥：$($_.Path):$($_.LineNumber)" }
    }
}
if (-not $found) { Ok '待提交范围内未发现硬编码密钥。' }
Info '注意：.env 本身不参与扫描（它不会被提交），但仍建议定期轮换 Key。'

if ($DryRun) {
    Section 'DryRun：以下文件将被提交（依据当前 .gitignore）'
    if ($hasGit) {
        # --dry-run 只列出结果，不会真的暂存
        git add -A --dry-run 2>$null | ForEach-Object { Info $_ }
    }
    else {
        Warn '还没有 .git，无法用 git 预览；请先 git init 或直接不带 -DryRun 运行。'
    }
    Ok 'DryRun 结束，未改动仓库。'
    exit 0
}

# ---------------------------------------------------------------- 3. 备份旧 .git
Section '3. 备份旧 .git（移到仓库外）'
if ($hasGit) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $backup = Join-Path (Split-Path -Parent $Root) ".git-backup-mall-agent-$stamp"
    Move-Item -Path (Join-Path $Root '.git') -Destination $backup
    Ok "旧历史已移到：$backup"
    Warn '确认新仓库没问题后，可以手动删除这个备份。'
}

# ---------------------------------------------------------------- 4. 重新初始化
Section '4. 重建干净仓库'
git init -q -b $Branch | Out-Null
if ($UserName)  { git config user.name  $UserName }
if ($UserEmail) { git config user.email $UserEmail }

$name  = git config user.name
$email = git config user.email
if (-not $name -or -not $email) {
    Die '缺少 git 作者信息。请加 -UserName / -UserEmail 参数，或先执行 git config --global user.name/user.email'
}
Info "作者：$name <$email>"
if ($email -match '@(qq|163|126|gmail|outlook)\.com$') {
    Warn '当前邮箱是真实邮箱，会出现在公开提交里。建议换成 GitHub noreply 邮箱。'
}

git add -A

# 提交前最后一道闸
Section '5. 提交前校验'
$blocked = @(
    '^\.env$',
    '^docs/',
    '^progress\.md$',
    '^task_plan\.md$',
    '^findings\.md$',
    '^plan_exec/',
    '^data/sessions/',
    '^data/reports/'
)
$staged = @(git diff --cached --name-only)
Info "待提交文件数：$($staged.Count)"
$violations = $staged | Where-Object { $p = $_; $blocked | Where-Object { $p -match $_ } }
if ($violations) {
    git rm -r --cached . --quiet 2>$null
    Die "以下本应排除的文件仍被暂存：`n$($violations -join "`n")`n请检查 .gitignore"
}

$secretHits = git grep --cached -n -I -E 'sk-[A-Za-z0-9]{20,}|ghp_[0-9A-Za-z]{20,}|AKIA[0-9A-Z]{16}' 2>$null
if ($secretHits) {
    git rm -r --cached . --quiet 2>$null
    Die "暂存区发现疑似密钥：`n$secretHits`n请先删除后再运行"
}

$bigFiles = $staged | ForEach-Object {
    $p = Join-Path $Root $_
    if (Test-Path $p) {
        $size = (Get-Item $p).Length
        if ($size -gt 10MB) { "{0}  {1:N1} MB" -f $_, ($size / 1MB) }
    }
} | Where-Object { $_ }
if ($bigFiles) { Warn "以下文件超过 10 MB，GitHub 单文件上限 100 MB：`n$($bigFiles -join "`n")" }

Ok '校验通过。'

# ---------------------------------------------------------------- 6. 提交
Section '6. 建立首次提交'
git commit -q -m 'init: mall-agent AI 客服/导购 Agent 三方案对比与评测' | Out-Null
Ok "新历史：$(git rev-parse --short HEAD)  （仅 1 个提交）"

# ---------------------------------------------------------------- 7. 推送
if ($RemoteUrl) {
    Section '7. 配置远程并推送'
    git remote remove origin 2>$null
    git remote add origin $RemoteUrl
    Info "origin = $RemoteUrl"
    Warn '这是强推覆盖远端主分支。如果远端已有内容，会被替换。'
    $answer = Read-Host '确认推送？输入 y 继续'
    if ($answer -eq 'y') {
        git push -u origin $Branch --force
        Ok '推送完成。去 GitHub 检查：Commits 只有 1 条。'
    }
    else {
        Warn '已跳过推送。稍后可手动：git push -u origin <branch> --force'
    }
}
else {
    Section '7. 下一步'
    Info '本地已整理好。上传时执行：'
    Info '  git remote add origin https://github.com/<you>/mall-agent.git'
    Info "  git push -u origin $Branch"
}

Section '完成'
Info '自检清单：'
Info '  1. GitHub → Commits：应只有 1 条；'
Info '  2. 打开仓库根目录：不应有 .env / docs / progress.md / task_plan.md / findings.md；'
Info '  3. 代码页搜索 sk- ：应无结果；'
Info '  4. 如 Key 曾进过旧历史：立即去平台吊销并换新。'
