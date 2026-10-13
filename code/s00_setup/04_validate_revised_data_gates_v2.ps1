#requires -Version 7.0
if (-not $env:BMI_ROOT) { $env:BMI_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path }  # auto-detect repo root; override with $env:BMI_ROOT

[CmdletBinding()]
param(
    [ValidateSet(
        'audit',
        'gate_A1_historical_crc_prototype',
        'gate_A2_future_bmi_anchor',
        'gate_A3_future_crc_outcomes',
        'gate_B_future_national_scenarios',
        'gate_C_future_spatial_prediction',
        'gate_D_historical_spatial_reconstruction'
    )]
    [string]$Gate = 'audit',
    [string]$ConfigPath = "$env:BMI_ROOT\results\未来情景预测\01_input_inventory\future_input_paths_template_v2.json",
    [string]$OutputRoot = "$env:BMI_ROOT\results\未来情景预测\01_input_inventory\data_gate_checks_v2"
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
        $size = if ($files.Count -gt 0) { [int64](($files | Measure-Object -Property Length -Sum).Sum) } else { [int64]0 }
        return [pscustomobject]@{
            exists = $true
            kind = 'directory'
            file_count = $files.Count
            size_bytes = $size
            sample_files = (($files | Select-Object -First 5 -ExpandProperty Name) -join '; ')
        }
    }
    return [pscustomobject]@{ exists = $false; kind = ''; file_count = 0; size_bytes = 0; sample_files = '' }
}

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "Config not found: $ConfigPath"
}

$config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($config.schema_version -ne 'future_input_paths_v2') {
    throw "Unsupported schema_version: $($config.schema_version)"
}

$datasetIds = @($config.datasets | ForEach-Object { [string]$_.dataset_id })
if (($datasetIds | Sort-Object -Unique).Count -ne $datasetIds.Count) {
    throw 'Duplicate dataset_id values in config.'
}

foreach ($gateSpec in $config.gates) {
    foreach ($requiredId in @($gateSpec.required_dataset_ids)) {
        if ($requiredId -notin $datasetIds) {
            throw "Gate $($gateSpec.gate_id) references unknown dataset_id: $requiredId"
        }
    }
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runDir = Join-Path $OutputRoot $timestamp
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$rows = foreach ($dataset in $config.datasets) {
    $dataAudit = Get-PathAudit ([string]$dataset.local_path)
    $documentationRequired = [bool]$dataset.documentation_required
    $documentationPath = [string]$dataset.documentation_path
    $documentationAudit = Get-PathAudit $documentationPath

    $pathReady = $dataAudit.exists -and $dataAudit.file_count -gt 0
    $documentationReady = (-not $documentationRequired) -or ($documentationAudit.exists -and $documentationAudit.file_count -gt 0)
    $knownContentGap = $false
    if ($dataset.PSObject.Properties.Name -contains 'known_content_gap') {
        $knownContentGap = [bool]$dataset.known_content_gap
    }
    $status = if ($knownContentGap) {
        'known_content_gap'
    }
    elseif (-not $pathReady) {
        'missing'
    }
    elseif (-not $documentationReady) {
        'documentation_missing'
    }
    elseif ($documentationRequired) {
        'path_and_documentation_present_pending_content_audit'
    }
    else {
        'path_present_pending_content_audit'
    }

    [pscustomobject]@{
        dataset_id = [string]$dataset.dataset_id
        requirement_level = [string]$dataset.requirement_level
        local_path = [string]$dataset.local_path
        data_exists = [bool]$dataAudit.exists
        data_kind = [string]$dataAudit.kind
        data_file_count = [int]$dataAudit.file_count
        data_size_bytes = [int64]$dataAudit.size_bytes
        sample_files = [string]$dataAudit.sample_files
        documentation_required = $documentationRequired
        documentation_path = $documentationPath
        documentation_exists = [bool]$documentationAudit.exists
        known_content_gap = $knownContentGap
        expected = [string]$dataset.expected
        path_gate_status = $status
        path_ready = ($pathReady -and $documentationReady -and -not $knownContentGap)
    }
}

$datasetAuditPath = Join-Path $runDir 'stage01_dataset_path_audit_v2.csv'
$rows | Export-Csv -LiteralPath $datasetAuditPath -NoTypeInformation -Encoding utf8BOM

$gateRows = foreach ($gateSpec in $config.gates) {
    $requiredIds = @($gateSpec.required_dataset_ids | ForEach-Object { [string]$_ })
    $requiredRows = @($rows | Where-Object { $_.dataset_id -in $requiredIds })
    $missing = @($requiredRows | Where-Object { -not $_.path_ready })
    [pscustomobject]@{
        gate_id = [string]$gateSpec.gate_id
        required_dataset_count = $requiredIds.Count
        ready_dataset_count = @($requiredRows | Where-Object { $_.path_ready }).Count
        missing_or_undocumented_count = $missing.Count
        missing_or_undocumented_dataset_ids = ((@($missing | ForEach-Object { [string]$_.dataset_id })) -join ';')
        path_gate_passed = ($missing.Count -eq 0 -and $requiredRows.Count -eq $requiredIds.Count)
        content_audit_required = $true
        gate_status = if ($missing.Count -eq 0) { 'path_gate_passed_content_audit_required' } else { 'blocked_missing_or_undocumented_inputs' }
    }
}

$gateAuditPath = Join-Path $runDir 'stage01_revised_gate_status_v2.csv'
$gateRows | Export-Csv -LiteralPath $gateAuditPath -NoTypeInformation -Encoding utf8BOM

$selectedGate = if ($Gate -eq 'audit') { $null } else { $gateRows | Where-Object { $_.gate_id -eq $Gate } | Select-Object -First 1 }
if ($Gate -ne 'audit' -and $null -eq $selectedGate) {
    throw "Requested gate not found in config: $Gate"
}

$summary = [ordered]@{
    generated_at = (Get-Date).ToString('o')
    powershell_version = $PSVersionTable.PSVersion.ToString()
    config_path = $ConfigPath
    config_sha256 = Get-Sha256 $ConfigPath
    requested_gate = $Gate
    datasets = $rows.Count
    gates = $gateRows.Count
    gates_path_ready = @($gateRows | Where-Object { $_.path_gate_passed }).Count
    selected_gate_path_passed = if ($Gate -eq 'audit') { $null } else { [bool]$selectedGate.path_gate_passed }
    content_audit_still_required = $true
    note = '路径门仅检查文件和必需文档是否存在。年份、情景、年龄、性别、单位、UI、病因层级、overweight定义和数据闭合必须另行审计。audit模式只盘点，不因缺失返回错误。'
}
$summaryPath = Join-Path $runDir 'stage01_revised_gate_summary_v2.json'
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding utf8

$hashInputs = @($datasetAuditPath, $gateAuditPath, $summaryPath)
$hashRows = foreach ($path in $hashInputs) {
    $item = Get-Item -LiteralPath $path
    [pscustomobject]@{
        path = $item.FullName
        size_bytes = [int64]$item.Length
        sha256 = Get-Sha256 $item.FullName
    }
}
$hashRows | Export-Csv -LiteralPath (Join-Path $runDir 'SHA256SUMS_v2.csv') -NoTypeInformation -Encoding utf8BOM

$summary | ConvertTo-Json -Depth 8
if ($Gate -ne 'audit' -and -not [bool]$selectedGate.path_gate_passed) {
    exit 2
}
exit 0
