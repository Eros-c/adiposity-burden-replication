if (-not $env:BMI_ROOT) { $env:BMI_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path }  # auto-detect repo root; override with $env:BMI_ROOT

[CmdletBinding()]
param(
    [string]$Root = "$env:BMI_ROOT\results\未来情景预测"
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

$manifestPath = Join-Path $Root '00_plan_manifest\baseline_artifact_manifest_v1.csv'
$manifest = @(Import-Csv -LiteralPath $manifestPath)
$missing = @($manifest | Where-Object { -not (Test-Path -LiteralPath $_.full_path -PathType Leaf) })
$sizeMismatch = @()
foreach ($row in $manifest) {
    if (-not (Test-Path -LiteralPath $row.full_path -PathType Leaf)) { continue }
    if ((Get-Item -LiteralPath $row.full_path).Length -ne [int64]$row.size_bytes) {
        $sizeMismatch += $row.full_path
    }
}

$sampleIndices = @(0, 97, 211, 389, 503, 701, 907, 1103, 1301, 1511, 1709, 1901, 2111, 2309, 2503, 2776) |
    Where-Object { $_ -lt $manifest.Count }
$hashMismatch = @()
foreach ($index in $sampleIndices) {
    $row = $manifest[$index]
    if ((Get-Sha256 $row.full_path) -ne $row.sha256) {
        $hashMismatch += $row.full_path
    }
}

$jsonFailures = @()
foreach ($file in @(Get-ChildItem -LiteralPath $Root -Recurse -File -Filter '*.json')) {
    try { Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json | Out-Null }
    catch { $jsonFailures += $file.FullName }
}

$csvFailures = @()
foreach ($file in @(Get-ChildItem -LiteralPath $Root -Recurse -File -Filter '*.csv')) {
    try { Import-Csv -LiteralPath $file.FullName | Out-Null }
    catch { $csvFailures += $file.FullName }
}

$expectedDirs = @(
    '00_plan_manifest','01_input_inventory','02_harmonized_covariates','03_national_bmi_anchors',
    '04_future_capable_mrp','05_historical_bmi_1990_2024','06_future_bmi_2025_2050',
    '07_rr_cause_crosswalk','08_colorectal_cancer_paf','09_colorectal_cancer_burden',
    '10_gbd_comparison','11_all_disease_extension','12_figures','13_tables','14_uncertainty',
    '15_code_logs_checksums'
)
$missingDirs = @($expectedDirs | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Root $_) -PathType Container) })

$qa = [ordered]@{
    generated_at = (Get-Date).ToString('o')
    powershell_version = $PSVersionTable.PSVersion.ToString()
    baseline_manifest_rows = $manifest.Count
    baseline_total_bytes = [int64](($manifest | Measure-Object -Property size_bytes -Sum).Sum)
    baseline_missing_files = $missing.Count
    baseline_size_mismatches = $sizeMismatch.Count
    deterministic_sha256_sample_size = $sampleIndices.Count
    deterministic_sha256_mismatches = $hashMismatch.Count
    json_parse_failures = $jsonFailures.Count
    csv_parse_failures = $csvFailures.Count
    expected_stage_directories = $expectedDirs.Count
    missing_stage_directories = $missingDirs.Count
    existing_baseline_modified = $false
    stage0_qa_passed = ($missing.Count -eq 0 -and $sizeMismatch.Count -eq 0 -and $hashMismatch.Count -eq 0)
    stage1_path_gate_passed = $false
    stage2_authorized = $false
}

$qaPath = Join-Path $Root '15_code_logs_checksums\stage00_01_integrity_qa_v1.json'
if (Test-Path -LiteralPath $qaPath) { throw "Refusing to overwrite: $qaPath" }
$qa | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $qaPath -Encoding utf8

$indexPath = Join-Path $Root '15_code_logs_checksums\stage00_01_artifact_index_v1.csv'
if (Test-Path -LiteralPath $indexPath) { throw "Refusing to overwrite: $indexPath" }
$files = @(Get-ChildItem -LiteralPath $Root -Recurse -File | Where-Object { $_.FullName -notlike '*SHA256SUMS_stage00_01_final_v1.csv' })
$index = foreach ($file in $files) {
    [pscustomobject]@{
        relative_path = [System.IO.Path]::GetRelativePath($Root, $file.FullName)
        size_bytes = [int64]$file.Length
        sha256 = Get-Sha256 $file.FullName
    }
}
$index | Sort-Object relative_path | Export-Csv -LiteralPath $indexPath -NoTypeInformation -Encoding utf8BOM

$finalHashPath = Join-Path $Root '15_code_logs_checksums\SHA256SUMS_stage00_01_final_v1.csv'
if (Test-Path -LiteralPath $finalHashPath) { throw "Refusing to overwrite: $finalHashPath" }
$finalFiles = @(Get-ChildItem -LiteralPath $Root -Recurse -File | Where-Object { $_.FullName -ne $finalHashPath })
$finalHashes = foreach ($file in $finalFiles) {
    [pscustomobject]@{
        relative_path = [System.IO.Path]::GetRelativePath($Root, $file.FullName)
        size_bytes = [int64]$file.Length
        sha256 = Get-Sha256 $file.FullName
    }
}
$finalHashes | Sort-Object relative_path | Export-Csv -LiteralPath $finalHashPath -NoTypeInformation -Encoding utf8BOM

$qa | ConvertTo-Json -Depth 6
if (-not $qa.stage0_qa_passed -or $jsonFailures.Count -gt 0 -or $csvFailures.Count -gt 0 -or $missingDirs.Count -gt 0) { exit 1 }
exit 0
