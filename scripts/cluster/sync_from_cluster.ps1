# Download cluster checkpoints locally with unique archive names + metadata.
#
# Run from Assignment3 (Windows PowerShell, VPN if needed):
#   powershell -ExecutionPolicy Bypass -File scripts/cluster/sync_from_cluster.ps1 -ClusterUser atml_team044
#
# Pull specific models only:
#   ... -Tags pgd_at,trades_r18,erm_r34
#
# By default each pull gets a UNIQUE file (never overwrites older epochs):
#   results/archive/20260614_erm_r34_resnet34_ep80_u....pt
#   results/checkpoints/baseline_erm_r34_resnet34_ep80.pt
#
# Optional: also overwrite the cluster "latest" alias (same name as on cluster):
#   ... -UpdateLatest
#
param(
    [string]$ClusterUser = "atml_team044",
    [string]$ClusterHost = "conduit2.hpc.uni-saarland.de",
    [string]$RemoteBase = "~/tml26_task3",
    [string]$ArchiveDir = "results/archive",
    [string]$CheckpointsDir = "results/checkpoints",
    [string[]]$Tags = @("pgd_at", "trades_r18", "erm_r34", "fgsm_at", "erm_r18"),
    [switch]$Force,
    [switch]$UpdateLatest
)

$ErrorActionPreference = "Stop"
$Local = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not (Test-Path "$Local\scripts\eval_model.py")) {
    throw "Run from Assignment3 repo (expected scripts/eval_model.py)"
}

$Remote = "${ClusterUser}@${ClusterHost}"
$Archive = Join-Path $Local $ArchiveDir
New-Item -ItemType Directory -Force -Path $Archive | Out-Null

$ManifestPath = Join-Path $Archive "manifest.json"
$Manifest = @()
if (Test-Path $ManifestPath) {
    $Manifest = @(Get-Content $ManifestPath -Raw | ConvertFrom-Json)
}

# tag -> remote checkpoint filename, architecture, progress run folder tag
$Registry = @{
    pgd_at     = @{ RemotePt = "pgd_at_resnet18.pt"; Arch = "resnet18"; ProgressTag = "pgd_at" }
    trades_r18 = @{ RemotePt = "trades_r18_resnet18.pt"; Arch = "resnet18"; ProgressTag = "trades_r18" }
    erm_r34    = @{ RemotePt = "baseline_erm_r34_resnet34.pt"; Arch = "resnet34"; ProgressTag = "erm_r34" }
    erm_r18    = @{ RemotePt = "baseline_erm_resnet18.pt"; Arch = "resnet18"; ProgressTag = "erm_resnet18" }
    fgsm_at    = @{ RemotePt = "fgsm_at_resnet18.pt"; Arch = "resnet18"; ProgressTag = "fgsm_at" }
}

function Format-Unified([object]$Value) {
    if ($null -eq $Value -or "$Value" -eq "?") { return "na" }
    try {
        return ("{0:F4}" -f [double]$Value).Replace(".", "p")
    } catch {
        return "na"
    }
}

function Test-RemoteFile {
    param([string]$RemotePath)
    $cmd = 'test -f ' + $RemotePath + ' && echo yes || echo no'
    $result = & ssh $Remote $cmd 2>$null
    if ($null -eq $result) { return $false }
    return ($result.ToString().Trim() -eq "yes")
}

Write-Host "Pulling from ${Remote}:${RemoteBase} -> $Archive"
Write-Host "Tags: $($Tags -join ', ')"
Write-Host ""

$Stamp = Get-Date -Format "yyyyMMdd"
$Pulled = 0

foreach ($Tag in $Tags) {
    if (-not $Registry.ContainsKey($Tag)) {
        Write-Warning "Unknown tag '$Tag' - skip (known: $($Registry.Keys -join ', '))"
        continue
    }

    $Info = $Registry[$Tag]
    $Arch = $Info.Arch
    $ProgressTag = $Info.ProgressTag
    $RemotePt = "$RemoteBase/results/checkpoints/$($Info.RemotePt)"
    $RemoteProgress = "$RemoteBase/results/runs/${ProgressTag}_${Arch}/progress.json"

    if (-not (Test-RemoteFile -RemotePath $RemotePt)) {
        Write-Warning "[$Tag] missing on cluster: $($Info.RemotePt) - skip"
        continue
    }

    $Progress = $null
    $Epoch = "unk"
    $Unified = "na"
    $TmpProgress = Join-Path $env:TEMP "tml3_progress_${Tag}.json"

    if (Test-RemoteFile -RemotePath $RemoteProgress) {
        & scp -q "${Remote}:${RemoteProgress}" $TmpProgress
        if (Test-Path $TmpProgress) {
            $Progress = Get-Content $TmpProgress -Raw | ConvertFrom-Json
            if ($Progress.last_epoch) {
                $Epoch = $Progress.last_epoch
            } elseif ($Progress.epochs_completed) {
                $Epoch = $Progress.epochs_completed
            }
            $Unified = Format-Unified $Progress.best_unified_score
            if ($Unified -eq "na") {
                $Unified = Format-Unified $Progress.final_unified_score
            }
        }
    }

    $BaseName = "${Stamp}_${Tag}_${Arch}_ep${Epoch}_u${Unified}"
    $DestPt = Join-Path $Archive "$BaseName.pt"
    $DestMeta = Join-Path $Archive "$BaseName.meta.json"

    if ((Test-Path $DestPt) -and (-not $Force)) {
        Write-Host "[$Tag] already archived today: $(Split-Path $DestPt -Leaf) - use -Force to re-pull"
        continue
    }

    Write-Host "[$Tag] downloading -> $(Split-Path $DestPt -Leaf)"
    & scp -q "${Remote}:${RemotePt}" $DestPt

    # Versioned copy for GUI — one file per epoch, never overwrites ep80 when pulling ep100
    $CkptRoot = Join-Path $Local $CheckpointsDir
    New-Item -ItemType Directory -Force -Path $CkptRoot | Out-Null
    $Stem = [System.IO.Path]::GetFileNameWithoutExtension($Info.RemotePt)
    $VersionedPt = Join-Path $CkptRoot "${Stem}_ep${Epoch}.pt"
    Copy-Item -Path $DestPt -Destination $VersionedPt -Force
    Write-Host "[$Tag] versioned -> ${CheckpointsDir}/${Stem}_ep${Epoch}.pt"

    if ($UpdateLatest) {
        $CanonicalPt = Join-Path $CkptRoot $Info.RemotePt
        Copy-Item -Path $DestPt -Destination $CanonicalPt -Force
        Write-Host "[$Tag] latest alias -> ${CheckpointsDir}/$($Info.RemotePt) (-UpdateLatest)"
    }

    $RelPath = (Resolve-Path $DestPt).Path
    if ($RelPath.StartsWith($Local)) {
        $RelPath = $RelPath.Substring($Local.Length + 1).Replace("\", "/")
    }

    $clusterUnified = $null
    if ($Progress) {
        if ($null -ne $Progress.final_unified_score) {
            $clusterUnified = $Progress.final_unified_score
        } else {
            $clusterUnified = $Progress.best_unified_score
        }
    }

    $Meta = [ordered]@{
        archived_at       = (Get-Date).ToString("o")
        source            = "cluster:${Remote}"
        tag               = $Tag
        architecture      = $Arch
        remote_checkpoint = $RemotePt
        local_checkpoint  = $RelPath
        versioned_checkpoint = "${CheckpointsDir}/${Stem}_ep${Epoch}.pt".Replace("\", "/")
        last_epoch        = $Epoch
        cluster_unified   = $clusterUnified
        progress          = $Progress
    }
    $Meta | ConvertTo-Json -Depth 10 | Set-Content -Path $DestMeta -Encoding UTF8

    $Manifest = @($Manifest | Where-Object { $_.tag -ne $Tag -or $_.local_checkpoint -ne $Meta.local_checkpoint })
    $Manifest += $Meta
    $Pulled++
}

if ($Manifest.Count -gt 0) {
    $Manifest | ConvertTo-Json -Depth 10 | Set-Content -Path $ManifestPath -Encoding UTF8
}

Write-Host ""
Write-Host "Pulled $Pulled checkpoint(s) -> $ArchiveDir"
Write-Host "Manifest: $ArchiveDir/manifest.json"
Write-Host ""
Write-Host "Evaluate all archived models locally:"
Write-Host '  .\.venv\Scripts\Activate.ps1'
Write-Host '  python scripts/eval_archive.py'
