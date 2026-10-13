if (-not $env:BMI_ROOT) { $env:BMI_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path }  # auto-detect repo root; override with $env:BMI_ROOT

param(
    [string]$RunId = "gate_b_A4_remediated_v1_20260827"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "PowerShell 7 or newer is required."
}

$RawRoot = "$env:BMI_ROOT\data\未来情景预测\01_raw"
$AsciiRoot = "$env:BMI_ROOT\gate_b_raw_link"
$ScriptRoot = "$env:BMI_ROOT\scripts\未来情景预测\pipeline\01_gate_b"
$OutputParent = "$env:BMI_ROOT\results\未来情景预测\02_future_covariate_gate"
$RunDir = Join-Path $OutputParent $RunId
$WorkDir = Join-Path "$env:BMI_ROOT\gate_b_work" $RunId
$A4Archive = Join-Path $RawRoot "ssp_grid_gdp\WangSun_0p25deg\original\GDP_025d (2000-2100).7z"
$A4RasterDir = Join-Path $RawRoot "ssp_grid_gdp\WangSun_0p25deg\extracted\025d"
$A1Population = Join-Path $AsciiRoot "ssp_country\IIASA_Basic_Drivers_2025\population\original\iamc_data-c28b8f39-d6f2-4a5e-8de7-0569891f8874.csv"
$A2Recode = Join-Path $AsciiRoot "ssp_country\IIASA_Basic_Drivers_2025\WIC_age_sex\V14\Recode dictionary.csv"
$Cmip6Metadata = "$env:BMI_ROOT\data\未来情景预测\00_metadata\cmip6_stage4_v3"
$Python = (Get-Command py -ErrorAction Stop).Source
$RasterPython = "python"
$Rscript = "Rscript"

foreach ($path in @($RawRoot, $ScriptRoot, $Cmip6Metadata, $A1Population, $A2Recode, $A4Archive, $A4RasterDir, $RasterPython, $Rscript)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required input missing: $path" }
}
if (Test-Path -LiteralPath $RunDir) { throw "Refusing to overwrite formal run: $RunDir" }
if (Test-Path -LiteralPath $WorkDir) { throw "Refusing to overwrite work directory: $WorkDir" }

$junction = Get-Item -LiteralPath $AsciiRoot -ErrorAction SilentlyContinue
if (-not $junction) {
    New-Item -ItemType Junction -Path $AsciiRoot -Target $RawRoot | Out-Null
    $junction = Get-Item -LiteralPath $AsciiRoot
}
$resolvedTarget = [System.IO.Path]::GetFullPath([string]$junction.Target)
$resolvedRaw = [System.IO.Path]::GetFullPath($RawRoot)
if ($junction.LinkType -ne "Junction" -or $resolvedTarget.TrimEnd('\') -ne $resolvedRaw.TrimEnd('\')) {
    throw "ASCII junction does not target the immutable raw root: $AsciiRoot -> $($junction.Target)"
}

New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
New-Item -ItemType Directory -Path $OutputParent -Force | Out-Null
$LogPath = Join-Path $WorkDir "run_gate_b_console.log"
$A4AuditDir = Join-Path $WorkDir "a4_audit"
$OldReport = Join-Path $OutputParent "Gate_B_report.md"
$OldPassFail = Join-Path $OutputParent "Gate_B_pass_fail.csv"
$OldReportHash = if (Test-Path -LiteralPath $OldReport) { (Get-FileHash -LiteralPath $OldReport -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$OldPassFailHash = if (Test-Path -LiteralPath $OldPassFail) { (Get-FileHash -LiteralPath $OldPassFail -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
Start-Transcript -LiteralPath $LogPath | Out-Null
$validatorExit = 99
try {
    $NetcdfJson = Join-Path $WorkDir "Gate_B_netcdf_inspection.json"
    $RasterJson = Join-Path $WorkDir "Gate_B_raster_inspection.json"
    $Crosswalk = Join-Path $WorkDir "Gate_B_country_crosswalk.csv"

    & $RasterPython (Join-Path $ScriptRoot "audit_a4_provenance_v1.py") `
        --archive $A4Archive `
        --raster-dir $A4RasterDir `
        --output-dir $A4AuditDir
    if ($LASTEXITCODE -ne 0) { throw "A4 provenance audit failed: exit $LASTEXITCODE" }
    $A4AuditCsv = Join-Path $A4AuditDir "A4_provenance_audit.csv"

    & $Python (Join-Path $ScriptRoot "inspect_gate_b_netcdf_v1.py") | Set-Content -LiteralPath $NetcdfJson -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "NetCDF inspection failed: exit $LASTEXITCODE" }

    & $RasterPython (Join-Path $ScriptRoot "inspect_gate_b_rasters_v1.py") | Set-Content -LiteralPath $RasterJson -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Raster inspection failed: exit $LASTEXITCODE" }

    & $Rscript (Join-Path $ScriptRoot "build_gate_b_country_crosswalk_v1.R") $A1Population $A2Recode $Crosswalk
    if ($LASTEXITCODE -ne 0) { throw "Country crosswalk failed: exit $LASTEXITCODE" }

    & $Python (Join-Path $ScriptRoot "validate_gate_b_A4_remediated_v2.py") `
        --root $AsciiRoot `
        --netcdf-json $NetcdfJson `
        --raster-json $RasterJson `
        --a4-audit $A4AuditCsv `
        --crosswalk $Crosswalk `
        --cmip6-dir $Cmip6Metadata `
        --output-dir $RunDir
    $validatorExit = $LASTEXITCODE
    if ($validatorExit -notin @(0, 2)) { throw "Gate B validator failed unexpectedly: exit $validatorExit" }

    foreach ($source in @($NetcdfJson, $RasterJson, $Crosswalk)) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $RunDir (Split-Path $source -Leaf))
    }
    Get-ChildItem -LiteralPath $A4AuditDir -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $RunDir $_.Name)
    }
    Copy-Item -LiteralPath (Join-Path $ScriptRoot "audit_a4_provenance_v1.py") -Destination $RunDir
    Copy-Item -LiteralPath (Join-Path $ScriptRoot "validate_gate_b_A4_remediated_v2.py") -Destination $RunDir
    Copy-Item -LiteralPath (Join-Path $ScriptRoot "inspect_gate_b_netcdf_v1.py") -Destination $RunDir
    Copy-Item -LiteralPath (Join-Path $ScriptRoot "inspect_gate_b_rasters_v1.py") -Destination $RunDir
    Copy-Item -LiteralPath (Join-Path $ScriptRoot "build_gate_b_country_crosswalk_v1.R") -Destination $RunDir
    Copy-Item -LiteralPath $PSCommandPath -Destination $RunDir
}
finally {
    Stop-Transcript | Out-Null
}

Copy-Item -LiteralPath $LogPath -Destination (Join-Path $RunDir "run_gate_b_console.log")

Copy-Item -LiteralPath (Join-Path $RunDir "Gate_B_report.md") -Destination (Join-Path $RunDir "Gate_B_report_A4_remediated.md")
Copy-Item -LiteralPath (Join-Path $RunDir "Gate_B_pass_fail.csv") -Destination (Join-Path $RunDir "Gate_B_pass_fail_A4_remediated.csv")
Copy-Item -LiteralPath (Join-Path $RunDir "Gate_B_dataset_contracts.csv") -Destination (Join-Path $RunDir "Gate_B_dataset_contracts_A4_remediated.csv")

$summary = [ordered]@{
    phase = "Phase 1 Gate B A4 provenance remediation"
    validator_exit_code = $validatorExit
    overall = if ($validatorExit -eq 0) { "PASS" } else { "FAIL" }
    run_directory = $RunDir
    raw_root = $RawRoot
    raw_root_modified = $false
    regridding_performed = $false
    variable_harmonization_performed = $false
    model_fitting_performed = $false
    disease_burden_analysis_performed = $false
    old_gate_b_report_sha256_before = $OldReportHash
    old_gate_b_pass_fail_sha256_before = $OldPassFailHash
}
$summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDir "Gate_B_runner_summary.json") -Encoding utf8

$hashRows = Get-ChildItem -LiteralPath $RunDir -File | Where-Object Name -ne "Gate_B_artifact_sha256.csv" | Sort-Object Name | ForEach-Object {
    [pscustomobject]@{
        path = $_.FullName
        size_bytes = $_.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
    }
}
$hashRows | Export-Csv -LiteralPath (Join-Path $RunDir "Gate_B_artifact_sha256.csv") -NoTypeInformation -Encoding utf8

Copy-Item -LiteralPath (Join-Path $RunDir "Gate_B_report_A4_remediated.md") -Destination (Join-Path $OutputParent "Gate_B_report_A4_remediated.md")
Copy-Item -LiteralPath (Join-Path $RunDir "Gate_B_pass_fail_A4_remediated.csv") -Destination (Join-Path $OutputParent "Gate_B_pass_fail_A4_remediated.csv")
Copy-Item -LiteralPath (Join-Path $RunDir "Gate_B_dataset_contracts_A4_remediated.csv") -Destination (Join-Path $OutputParent "Gate_B_dataset_contracts_A4_remediated.csv")
Copy-Item -LiteralPath (Join-Path $RunDir "A4_provenance_audit.csv") -Destination (Join-Path $OutputParent "A4_provenance_audit.csv")

$OldReportHashAfter = if (Test-Path -LiteralPath $OldReport) { (Get-FileHash -LiteralPath $OldReport -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$OldPassFailHashAfter = if (Test-Path -LiteralPath $OldPassFail) { (Get-FileHash -LiteralPath $OldPassFail -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
if ($OldReportHash -ne $OldReportHashAfter -or $OldPassFailHash -ne $OldPassFailHashAfter) {
    throw "Existing Gate B audit-history files changed unexpectedly."
}

Write-Host ($summary | ConvertTo-Json)
exit $validatorExit
