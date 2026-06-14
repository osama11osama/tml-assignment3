# Upload code + checkpoints to Saarland HPC (run on Windows PowerShell).
#
# Run from Assignment3 folder:
#   powershell -File scripts/cluster/sync_to_cluster.ps1 -ClusterUser atml_team044
#
param(
    [string]$ClusterUser = "atml_team044",
    [string]$ClusterHost = "conduit2.hpc.uni-saarland.de",
    [string]$RemoteBase = "~/tml26_task3"
)

$ErrorActionPreference = "Stop"
$Local = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not (Test-Path "$Local\scripts\train_pgd_at.py")) {
    throw "Run from Assignment3 repo (expected scripts/train_pgd_at.py)"
}

$Remote = "${ClusterUser}@${ClusterHost}"
Write-Host "Syncing $Local -> ${Remote}:${RemoteBase}"

ssh $Remote "mkdir -p ${RemoteBase}/{data,results/checkpoints,results/runs,runlogs,scripts/cluster/condor,src,configs}"

scp -r "$Local\src" "${Remote}:${RemoteBase}/"
scp -r "$Local\scripts" "${Remote}:${RemoteBase}/"
scp -r "$Local\configs" "${Remote}:${RemoteBase}/"
scp "$Local\requirements.txt" "${Remote}:${RemoteBase}/"
scp "$Local\task_template.py" "${Remote}:${RemoteBase}/"

$Fgsm = "$Local\results\checkpoints\fgsm_at_resnet18.pt"
if (Test-Path $Fgsm) {
    Write-Host "Uploading FGSM init for PGD-AT (~43 MB)..."
    scp $Fgsm "${Remote}:${RemoteBase}/results/checkpoints/"
} else {
    Write-Warning "Missing $Fgsm - upload manually before Step 3."
}

$Erm = "$Local\results\checkpoints\baseline_erm_resnet18.pt"
if (Test-Path $Erm) {
    Write-Host "Uploading ERM checkpoint (~43 MB)..."
    scp $Erm "${Remote}:${RemoteBase}/results/checkpoints/"
}

Write-Host "Fixing CRLF line endings on cluster..."
ssh $Remote 'cd ~/tml26_task3 && find scripts/cluster -type f \( -name "*.sh" -o -name "*.sub" \) -exec sed -i "s/\r$//" {} + && chmod +x scripts/cluster/*.sh && echo CRLF fixed'

Write-Host ""
Write-Host "Week 1 on cluster:"
Write-Host "  bash scripts/cluster/submit_week1.sh"
Write-Host "  docs/CLUSTER_WEEK1.md"
