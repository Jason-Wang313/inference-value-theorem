param(
    [switch]$Clean,
    [switch]$Package
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $Root)
$UserProfile = [Environment]::GetFolderPath("UserProfile")
$Downloads = Join-Path $UserProfile "Downloads"
$OneDriveDesktop = Join-Path $UserProfile "OneDrive\Desktop"
$Desktop = if (Test-Path $OneDriveDesktop) { $OneDriveDesktop } else { [Environment]::GetFolderPath("Desktop") }
$PdfOut = Join-Path $Desktop "best-of-n-llm-v4.pdf"
$RepoPdfOut = Join-Path $RepoRoot "paper\final\best-of-n-llm-v4.pdf"
$ZipOut = Join-Path $Downloads "best-of-n-llm-v4-source.zip"

Push-Location $Root
try {
    function Invoke-Checked {
        param([string]$Command, [string[]]$Arguments)
        & $Command @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Command failed with exit code $LASTEXITCODE"
        }
    }

    if ($Clean) {
        $patterns = @("*.aux", "*.bbl", "*.blg", "*.fls", "*.fdb_latexmk", "*.log", "*.out", "*.synctex.gz", "main.pdf")
        foreach ($pattern in $patterns) {
            Get-ChildItem -Path . -Filter $pattern -File -ErrorAction SilentlyContinue | Remove-Item -Force
        }
    }

    Invoke-Checked "python" @((Join-Path $RepoRoot "experiments\18_v4_protocol_evidence.py"))

    $latexmkWorks = $false
    $latexmkCmd = Get-Command latexmk -ErrorAction SilentlyContinue
    $perlCmd = Get-Command perl -ErrorAction SilentlyContinue
    if ($latexmkCmd -and $perlCmd) {
        latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
        $latexmkWorks = ($LASTEXITCODE -eq 0)
    }
    if (-not $latexmkWorks) {
        Invoke-Checked "pdflatex" @("-interaction=nonstopmode", "-halt-on-error", "main.tex")
        $AuxPath = Join-Path $Root "main.aux"
        $NeedsBibtex = (Test-Path $AuxPath) -and (Select-String -LiteralPath $AuxPath -Pattern "\\bibdata" -Quiet)
        if ($NeedsBibtex) {
            Invoke-Checked "bibtex" @("main")
        }
        Invoke-Checked "pdflatex" @("-interaction=nonstopmode", "-halt-on-error", "main.tex")
        Invoke-Checked "pdflatex" @("-interaction=nonstopmode", "-halt-on-error", "main.tex")
    }
    $RepoPdfDir = Split-Path -Parent $RepoPdfOut
    if (-not (Test-Path $RepoPdfDir)) {
        New-Item -ItemType Directory -Path $RepoPdfDir | Out-Null
    }
    Copy-Item -LiteralPath (Join-Path $Root "main.pdf") -Destination $PdfOut -Force
    Copy-Item -LiteralPath (Join-Path $Root "main.pdf") -Destination $RepoPdfOut -Force
    Remove-Item -LiteralPath (Join-Path $Root "main.pdf") -Force -ErrorAction SilentlyContinue

    if ($Package) {
        if (Test-Path $ZipOut) {
            Remove-Item -LiteralPath $ZipOut -Force
        }
        $sourceItems = @(
            "main.tex",
            "appendix.tex",
            "references.bib",
            "README.md",
            "submission_metadata.json",
            "build.ps1",
            "iclr2027_conference.sty",
            "iclr2026_conference.sty",
            "iclr2026_conference.bst",
            "natbib.sty",
            "fancyhdr.sty",
            "math_commands.tex",
            "figures"
        )
        Compress-Archive -Path $sourceItems -DestinationPath $ZipOut
    }

    Write-Host "PDF: $PdfOut"
    Write-Host "Repo PDF: $RepoPdfOut"
    if ($Package) {
        Write-Host "Source ZIP: $ZipOut"
    }
}
finally {
    Pop-Location
}
