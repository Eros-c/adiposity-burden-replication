if (-not $env:BMI_ROOT) { $env:BMI_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path }  # auto-detect repo root; override with $env:BMI_ROOT

param(
    [ValidateSet('validation','full','all')]
    [string]$Mode = 'all'
)

$ErrorActionPreference = 'Stop'
$Root = '$env:BMI_ROOT\results\future_scenario_ascii\04_future_capable_mrp\phase4b_d1_reduced_demography_v1_20260831'
$D1Script = Join-Path $Root 'scripts\phase4b_d1_mrp_task_v1.R'
$ComparatorScript = Join-Path $Root 'scripts\phase4b_source_comparator_task_v1.R'
$D1Stage = Join-Path $Root 'model_run_v1'
$ComparatorStage = Join-Path $Root 'source_comparator_v1'
$LogRoot = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$Internal = 1..5 | ForEach-Object { "internal_fold_$_" }
$Country = 1..5 | ForEach-Object { "country_fold_$_" }
$Source = @(
    'source_holdout_dhs', 'source_holdout_knhanes',
    'source_holdout_ensanut', 'source_holdout_shes',
    'source_holdout_steps', 'source_holdout_pns'
)

function Test-TaskComplete {
    param([string]$Stage, [string]$Task)
    $Metadata = Join-Path $Stage "models\checkpoints\${Task}_metadata_v1.json"
    if (-not (Test-Path -LiteralPath $Metadata)) { return $false }
    try { $Value = Get-Content -LiteralPath $Metadata -Raw | ConvertFrom-Json } catch { return $false }
    if ($Value.status -ne 'complete') { return $false }
    if (-not (Test-Path -LiteralPath $Value.fit_path)) { return $false }
    if (-not (Test-Path -LiteralPath $Value.recipe_path)) { return $false }
    if ($Task -ne 'full_model' -and -not (Test-Path -LiteralPath $Value.prediction_path)) { return $false }
    return $true
}

function Invoke-Task {
    param([string]$Label, [string]$Script, [string]$Stage, [string]$Task)
    if (Test-TaskComplete -Stage $Stage -Task $Task) {
        Write-Host "SKIP $Label $Task"
        return
    }
    $Log = Join-Path $LogRoot "${Label}_${Task}_v1.log"
    if (Test-Path -LiteralPath $Log) { throw "Incomplete task has an existing log: $Log" }
    Write-Host "RUN  $Label $Task"
    & Rscript $Script $Task *>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) { throw "Task failed: $Label $Task, exit=$LASTEXITCODE" }
    if (-not (Test-TaskComplete -Stage $Stage -Task $Task)) { throw "Task did not freeze complete metadata: $Label $Task" }
}

if ($Mode -in @('validation','all')) {
    foreach ($Task in @($Internal + $Country + $Source)) {
        Invoke-Task -Label 'd1' -Script $D1Script -Stage $D1Stage -Task $Task
    }
    foreach ($Task in $Source) {
        Invoke-Task -Label 'source_comparator' -Script $ComparatorScript -Stage $ComparatorStage -Task $Task
    }
    $Qa = Join-Path $D1Stage 'qa\all_eligible20_qa_summary_v1.json'
    if (-not (Test-Path -LiteralPath $Qa)) {
        & Rscript $D1Script finalize *>&1 | Tee-Object -FilePath (Join-Path $LogRoot 'd1_finalize_v1.log')
        if ($LASTEXITCODE -ne 0) { throw "D1 finalize failed, exit=$LASTEXITCODE" }
    } else {
        Write-Host 'SKIP d1 finalize'
    }
}

if ($Mode -in @('full','all')) {
    Invoke-Task -Label 'd1' -Script $D1Script -Stage $D1Stage -Task 'full_model'
}
