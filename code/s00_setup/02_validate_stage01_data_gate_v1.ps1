if (-not $env:BMI_ROOT) { $env:BMI_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path }  # auto-detect repo root; override with $env:BMI_ROOT

[CmdletBinding()]
param(
    [string]$ConfigPath = "$env:BMI_ROOT\results\未来情景预测\01_input_inventory\future_input_paths_template_v1.json",
    [string]$OutputRoot = "$env:BMI_ROOT\results\未来情景预测\01_input_inventory\data_gate_checks"
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-PathAudit([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return [pscustomobject]@{ exists = $false; kind = ''; file_count = 0; size_bytes = 0; sample_files = '' }
    }
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $item = Get-Item -LiteralPath $Path
        return [pscustomobject]@{
            exists = $true
            kind = 'file'
            file_count = 1
            size_bytes = [int64]$item.Length
            sample_files = $item.Name
        }
    }
    if (Test-Path -LiteralPath $Path -PathType Container) {
        $files = @(Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction Stop)
        return [pscustomobject]@{
            exists = $true
            kind = 'directory'
            file_count = $files.Count
            size_bytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
            sample_files = (($files | Select-Object -First 5 -ExpandProperty Name) -join '; ')
        }
    }
    return [pscustomobject]@{ exists = $false; kind = ''; file_count = 0; size_bytes = 0; sample_files = '' }
}

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "Config not found: $ConfigPath"
}

$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
if ($config.schema_version -ne 'future_input_paths_v1') {
    throw "Unsupported schema_version: $($config.schema_version)"
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runDir = Join-Path $OutputRoot $timestamp
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$rows = foreach ($dataset in $config.datasets) {
    $audit = Get-PathAudit ([string]$dataset.local_path)
    [pscustomobject]@{
        dataset_id = [string]$dataset.dataset_id
        required = [bool]$dataset.required
        local_path = [string]$dataset.local_path
        exists = [bool]$audit.exists
        kind = [string]$audit.kind
        file_count = [int]$audit.file_count
        size_bytes = [int64]$audit.size_bytes
        sample_files = [string]$audit.sample_files
        expected = [string]$dataset.expected
        gate_status = if ($audit.exists -and $audit.file_count -gt 0) { 'path_present_pending_content_audit' } else { 'missing' }
    }
}

$csvPath = Join-Path $runDir 'stage01_data_gate_path_audit_v1.csv'
$rows | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding utf8BOM

$missingRequired = @($rows | Where-Object { $_.required -and $_.gate_status -eq 'missing' })
$summary = [ordered]@{
    generated_at = (Get-Date).ToString('o')
    config_path = $ConfigPath
    config_sha256 = Get-Sha256 $ConfigPath
    datasets = $rows.Count
    required_missing = $missingRequired.Count
    stage01_path_gate_passed = ($missingRequired.Count -eq 0)
    content_audit_still_required = $true
    note = '路径通过仅表示文件存在。年份、情景、年龄、性别、单位、UI和病因口径仍需逐表审计后才能进入第2步。'
}
$jsonPath = Join-Path $runDir 'stage01_data_gate_summary_v1.json'
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$hashRows = @($csvPath, $jsonPath) | ForEach-Object {
    $item = Get-Item -LiteralPath $_
    [pscustomobject]@{ path = $item.FullName; size_bytes = $item.Length; sha256 = Get-Sha256 $item.FullName }
}
$hashRows | Export-Csv -LiteralPath (Join-Path $runDir 'SHA256SUMS_v1.csv') -NoTypeInformation -Encoding utf8BOM

$summary | ConvertTo-Json -Depth 6
if ($missingRequired.Count -gt 0) { exit 2 }
exit 0
