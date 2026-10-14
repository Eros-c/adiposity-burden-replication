if (-not $env:BMI_ROOT) { $env:BMI_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path }  # auto-detect repo root; override with $env:BMI_ROOT

[CmdletBinding()]
param([string]$OutputRoot = '$env:BMI_ROOT\results\未来情景预测')

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-CsvUtf8Bom($Data, [string]$Path) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite: $Path" }
    [System.IO.File]::WriteAllLines(
        $Path,
        ($Data | ConvertTo-Csv -NoTypeInformation),
        [System.Text.UTF8Encoding]::new($true)
    )
}

function Write-TextUtf8([string]$Text, [string]$Path) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite: $Path" }
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

function Get-PathSummary([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject]@{exists=$false; files=0; bytes=[int64]0}
    }
    $files = @(Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue)
    return [pscustomobject]@{
        exists = $true
        files = $files.Count
        bytes = [int64](($files | Measure-Object Length -Sum).Sum)
    }
}

$logPath = Join-Path $OutputRoot '15_code_logs_checksums\logs\stage00_01_initialization_v1.log'
if (Test-Path -LiteralPath $logPath) { throw "Refusing to overwrite log: $logPath" }
Start-Transcript -LiteralPath $logPath -Force | Out-Null
try {
    $requiredBaseline = @(
        '00_plan_manifest\frozen_study_spec_v1.json',
        '00_plan_manifest\stage_gate_manifest_v1.csv',
        '00_plan_manifest\baseline_roots_v1.csv',
        '00_plan_manifest\baseline_artifact_manifest_v1.csv',
        '00_plan_manifest\baseline_summary_v1.json'
    )
    foreach ($relative in $requiredBaseline) {
        $path = Join-Path $OutputRoot $relative
        if (-not (Test-Path -LiteralPath $path)) { throw "Missing partial Stage 0 artifact: $path" }
    }

    $inventoryDefinitions = @(
        [pscustomobject]@{id='baseline_mrp';name='区域STEPS增强MRP与2015-2024 BMI7结果';path='$env:BMI_ROOT\results\降尺度\experiments\BMI7_MRP_STEPS_regional_augmentation_20260812';stage='0';status='available';years='2015-2024';spatial='0.1°, 0.5°, ADM1';scenario='historical';age_sex='age × sex';unit='BMI category proportions';notes='Frozen comparison baseline'},
        [pscustomobject]@{id='ncd_risc_bmi7';name='NCD-RisC国家-年份-年龄-性别BMI组成';path='$env:BMI_ROOT\data\肥胖\全球\NCD_RisC\国家-年份-年龄-性别肥胖';stage='3';status='available';years='1980-2024';spatial='country';scenario='historical';age_sex='age × sex';unit='proportion';notes='Historical seven-category anchor'},
        [pscustomobject]@{id='gbd_adult_bmi_forecast';name='GBD成人超重和肥胖预测机器可读表';path='';stage='3';status='missing_blocking';years='2022-2050';spatial='country';scenario='GBD reference';age_sex='25+ age × sex';unit='prevalence and 95% UI';notes='Article available; machine-readable estimates absent locally'},
        [pscustomobject]@{id='gbd_crc_total_forecast';name='GBD结直肠癌总发病、死亡和DALY';path='';stage='9';status='missing_blocking';years='1990-2050';spatial='country';scenario='historical and forecast';age_sex='age × sex';unit='counts, rates and 95% UI';notes='Detailed CRC extract required'},
        [pscustomobject]@{id='gbd_crc_high_bmi';name='GBD结直肠癌高BMI归因结果';path='';stage='10-11';status='missing_blocking';years='historical and forecast';spatial='country';scenario='GBD';age_sex='age × sex';unit='attributable counts/rates and 95% UI';notes='Comparison dataset'},
        [pscustomobject]@{id='local_gbd_2020';name='现有本地GBD高BMI聚合结果';path='$env:BMI_ROOT\data\GBD数据';stage='audit';status='available_not_sufficient';years='2020 only';spatial='country';scenario='historical';age_sex='aggregated';unit='attributable burden';notes='Only three aggregate files; insufficient for CRC 1990-2050'},
        [pscustomobject]@{id='ssp_population_grid';name='SSP网格人口';path='';stage='2,4,6';status='missing_blocking';years='2010-2100 decadal';spatial='1/8° or native';scenario='SSP1,SSP2,SSP3,SSP5';age_sex='total and urban/rural';unit='persons';notes='Future population surface'},
        [pscustomobject]@{id='ssp_population_country';name='SSP国家年龄性别人口与城市化';path='';stage='2,3,6';status='missing_blocking';years='2020-2050';spatial='country';scenario='SSP1,SSP2,SSP3,SSP5';age_sex='age × sex';unit='persons/shares';notes='IIASA SSP database'},
        [pscustomobject]@{id='ssp_gdp_grid';name='SSP网格GDP';path='';stage='2,4,6';status='missing_blocking';years='2030-2050';spatial='0.25° preferred';scenario='SSP1,SSP2,SSP3,SSP5';age_sex='none';unit='PPP per capita';notes='Future GDP surface'},
        [pscustomobject]@{id='cmip6_climate';name='NEX-GDDP-CMIP6温度和湿度';path='';stage='2,4,6';status='missing_blocking';years='2015-2050';spatial='0.25°';scenario='SSP126,SSP245,SSP370,SSP585';age_sex='none';unit='temperature/humidity';notes='At least five GCMs'},
        [pscustomobject]@{id='worldpop_historical';name='WorldPop年龄性别人口网格';path='$env:BMI_ROOT\data\人口结构网格数据';stage='2,5,6';status='available_partial';years='2006,2012-2025';spatial='1 km';scenario='historical';age_sex='age × sex';unit='persons';notes='Does not cover 1990-2005'},
        [pscustomobject]@{id='historical_gdp_grid';name='历史人均GDP网格';path='$env:BMI_ROOT\data\GDP网格数据';stage='2,4,5';status='available';years='1990-2024';spatial='5 arc-min';scenario='historical';age_sex='none';unit='PPP per capita';notes='Kummu et al. historical source'},
        [pscustomobject]@{id='historical_pm25';name='ACAG SatPM2.5';path='$env:BMI_ROOT\data\PM2.5网格数据';stage='sensitivity';status='available_sensitivity_only';years='historical';spatial='native';scenario='historical';age_sex='none';unit='µg/m³';notes='Excluded from future main model'},
        [pscustomobject]@{id='era5_temperature';name='ERA5 2m temperature';path='$env:BMI_ROOT\data\协变量数据\ERA5：温度';stage='2,4,5';status='available';years='historical';spatial='native';scenario='historical';age_sex='none';unit='K/°C';notes='CMIP6 bias-correction reference'},
        [pscustomobject]@{id='era5_humidity';name='ERA5 temperature/dewpoint humidity source';path='$env:BMI_ROOT\data\协变量数据\ERA5：湿度';stage='2,4,5';status='available';years='historical';spatial='native';scenario='historical';age_sex='none';unit='K and derived RH %';notes='CMIP6 bias-correction reference'},
        [pscustomobject]@{id='ghs_smod';name='GHS-SMOD城市化分类';path='$env:BMI_ROOT\data\城市化网格数据_GHS_SMOD';stage='2,4,5';status='available_partial';years='2005,2010,2015,2020';spatial='1 km';scenario='historical';age_sex='none';unit='settlement class';notes='Future requires SSP-compatible source'},
        [pscustomobject]@{id='rr_workbook';name='BMI-疾病RR资料表';path='$env:BMI_ROOT\data\肥胖-疾病剂量反应关系\BMI_疾病关联.xlsx';stage='7-8';status='available_requires_crc_freeze';years='literature';spatial='none';scenario='none';age_sex='study-specific';unit='RR per 5 kg/m²';notes='CRC parameter still requires explicit freeze'},
        [pscustomobject]@{id='static_elevation';name='GMTED2010高程';path='$env:BMI_ROOT\data\协变量数据\GMTED2010：高程数据';stage='2,4,6';status='available';years='static';spatial='native';scenario='static';age_sex='none';unit='metres';notes='Static predictor'},
        [pscustomobject]@{id='static_accessibility';name='2015城市和港口交通时间';path='$env:BMI_ROOT\data\Travel time to cities and ports in the year 2015';stage='2,4,6';status='available';years='2015 static';spatial='~1 km';scenario='static';age_sex='none';unit='minutes';notes='Static predictor'}
    )

    $inventory = foreach ($d in $inventoryDefinitions) {
        $summary = Get-PathSummary $d.path
        [pscustomobject]@{
            dataset_id=$d.id; dataset=$d.name; required_stage=$d.stage; status=$d.status
            local_path=$d.path; path_exists=$summary.exists; file_count=$summary.files; size_bytes=$summary.bytes
            years=$d.years; spatial_resolution=$d.spatial; scenario_coverage=$d.scenario
            age_sex_coverage=$d.age_sex; unit=$d.unit; notes=$d.notes
        }
    }
    Write-CsvUtf8Bom $inventory (Join-Path $OutputRoot '01_input_inventory\input_inventory_v1.csv')

    $contract = @(
        [pscustomobject]@{dataset_id='gbd_adult_bmi_forecast';required_fields='location_id/name, iso3, year, age_group_id/name, sex_id/name, overweight mean/lower/upper, obesity mean/lower/upper';required_years='2022-2050';required_scenarios='GBD reference';required_ui='yes';acceptance='25+; stable IDs; definitions documented'},
        [pscustomobject]@{dataset_id='gbd_crc_total_forecast';required_fields='cause_id, location_id, year, age_group_id, sex_id, measure_id, metric_id, mean, lower, upper';required_years='1990-2050';required_scenarios='historical and forecast';required_ui='yes';acceptance='CRC one-to-one cause mapping; incidence/death/DALY separable'},
        [pscustomobject]@{dataset_id='gbd_crc_high_bmi';required_fields='cause_id, rei_id, location_id, year, age_group_id, sex_id, measure_id, metric_id, mean, lower, upper';required_years='historical and forecast';required_scenarios='GBD';required_ui='yes';acceptance='rei=high BMI; no parent-child double counting'},
        [pscustomobject]@{dataset_id='ssp_population_grid';required_fields='scenario, year, georeference, total population, urban/rural if available';required_years='2020-2050 at least decadal';required_scenarios='SSP1,SSP2,SSP3,SSP5';required_ui='preferred';acceptance='global; non-negative; totals documented'},
        [pscustomobject]@{dataset_id='ssp_population_country';required_fields='iso3, scenario, year, age, sex, population, urbanization';required_years='2020-2050';required_scenarios='SSP1,SSP2,SSP3,SSP5';required_ui='preferred';acceptance='crosswalkable to MRP poststrata'},
        [pscustomobject]@{dataset_id='ssp_gdp_grid';required_fields='scenario, year, georeference, GDP/GDPpc PPP, price base';required_years='2030-2050';required_scenarios='SSP1,SSP2,SSP3,SSP5';required_ui='no';acceptance='0.25° preferred; PPP base documented'},
        [pscustomobject]@{dataset_id='cmip6_climate';required_fields='GCM, scenario, date/year, tas, hurs or humidity inputs, georeference';required_years='2015-2050';required_scenarios='SSP126,SSP245,SSP370,SSP585';required_ui='ensemble spread';acceptance='at least 5 GCMs; calendar and units documented'}
    )
    Write-CsvUtf8Bom $contract (Join-Path $OutputRoot '01_input_inventory\future_data_contract_v1.csv')

    $gaps = @(
        [pscustomobject]@{priority=1;dataset_id='gbd_adult_bmi_forecast';blocking_stage='3';problem='机器可读的国家×年份×年龄×性别预测缺失';required_action='从论文补充材料或IHME工具下载均值和95% UI';status='open'},
        [pscustomobject]@{priority=2;dataset_id='gbd_crc_total_forecast';blocking_stage='9';problem='结直肠癌详细总负担缺失';required_action='导出1990-2050发病、死亡、DALY数量和率';status='open'},
        [pscustomobject]@{priority=3;dataset_id='gbd_crc_high_bmi';blocking_stage='10-11';problem='结直肠癌高BMI归因结果缺失';required_action='导出一致病因、风险、年龄、性别口径的数据';status='open'},
        [pscustomobject]@{priority=4;dataset_id='ssp_population_grid';blocking_stage='2';problem='SSP人口网格缺失';required_action='下载SSP1/2/3/5至2050';status='open'},
        [pscustomobject]@{priority=5;dataset_id='ssp_population_country';blocking_stage='2-3';problem='SSP国家年龄性别人口和城市化缺失';required_action='下载并冻结IIASA版本';status='open'},
        [pscustomobject]@{priority=6;dataset_id='ssp_gdp_grid';blocking_stage='2';problem='SSP网格GDP缺失';required_action='下载0.25° 2030-2050产品';status='open'},
        [pscustomobject]@{priority=7;dataset_id='cmip6_climate';blocking_stage='2';problem='CMIP6未来温湿度集合缺失';required_action='下载至少5个GCM和4个情景';status='open'},
        [pscustomobject]@{priority=8;dataset_id='worldpop_historical';blocking_stage='5-6';problem='1990-2005年龄性别人口网格缺口';required_action='选择历史回推来源或调整空间重建起点';status='open'}
    )
    Write-CsvUtf8Bom $gaps (Join-Path $OutputRoot '01_input_inventory\blocking_data_gaps_v1.csv')

    $sources = @(
        [pscustomobject]@{dataset_id='gbd_adult_bmi_forecast';title='GBD adult overweight and obesity forecasts to 2050';url='https://pmc.ncbi.nlm.nih.gov/articles/PMC11920007/';note='机器可读数据可能在补充材料或IHME工具'},
        [pscustomobject]@{dataset_id='ssp_population_country';title='IIASA SSP database';url='https://iiasa.ac.at/models-tools-data/ssp';note='记录发行版本和访问日期'},
        [pscustomobject]@{dataset_id='ssp_population_grid';title='NASA SSP population grids';url='https://data.nasa.gov/dataset/global-population-projection-grids-based-on-shared-socioeconomic-pathways-ssps-2010-2100';note='下载SSP1/2/3/5'},
        [pscustomobject]@{dataset_id='ssp_gdp_grid';title='Global gridded GDP projections under SSPs';url='https://www.nature.com/articles/s41597-022-01300-x';note='优先0.25°产品'},
        [pscustomobject]@{dataset_id='ssp_gdp_grid';title='Gridded GDP repository';url='https://doi.org/10.5281/zenodo.7898409';note='保存文件哈希和单位'},
        [pscustomobject]@{dataset_id='cmip6_climate';title='NASA NEX-GDDP-CMIP6';url='https://www.nasa.gov/nasa-earth-exchange-nex/gddp/downscaled-climate-projections-nex-gddp-cmip6/';note='至少5个GCM'},
        [pscustomobject]@{dataset_id='ghs_wup';title='GHS-WUP 2025';url='https://human-settlement.emergency.copernicus.eu/ghs_wup2025.php';note='用于一致性检查'}
    )
    Write-CsvUtf8Bom $sources (Join-Path $OutputRoot '01_input_inventory\authoritative_download_sources_v1.csv')

    $readiness = [ordered]@{
        generated_at=(Get-Date).ToString('o'); stage0='completed'; stage1='blocked_partial'
        datasets_total=$inventory.Count
        available_or_partial=@($inventory | Where-Object status -Match '^available').Count
        missing_blocking=@($inventory | Where-Object status -eq 'missing_blocking').Count
        can_start_stage2=$false; can_fit_future_mrp=$false; can_generate_2050_bmi=$false
        can_compute_crc_paf_prototype=$false; can_compute_crc_burden=$false
        reason='GBD未来BMI锚点、SSP人口/GDP、CMIP6气候及详细结直肠癌GBD数据缺失。'
    }
    Write-TextUtf8 ($readiness | ConvertTo-Json -Depth 5) (Join-Path $OutputRoot '01_input_inventory\local_data_readiness_v1.json')

    $stage00 = @'
# 第0步：冻结现有成果

第0步已完成。现有区域STEPS增强MRP、验证、2015–2024年0.1°/0.5°/ADM1结果和更早的Stage 9基线均保持只读；没有复制数据本体，也没有重新拟合或修改任何正式结果。

`baseline_artifact_manifest_v1.csv`记录所有冻结文件的完整路径、大小、修改时间和SHA256。当前增强MRP使用53,777条训练记录，现有结果覆盖2015–2024年。旧模型仅作为未来模型的比较基线，不能直接当作2050年预测器。

20国稳定聚合验证（有效样本量≥20）中，内部加权R²为0.860，国家留出加权R²为0.788；国家留出的国家锚点内肥胖偏差相关为0.335。因此，未来模型必须重新验证时间外推和国家内部迁移能力。
'@
    Write-TextUtf8 $stage00 (Join-Path $OutputRoot '00_plan_manifest\stage00_freeze_report_CN_v1.md')

    $stage01 = @'
# 第1步：关键数据审计

第1步仅部分完成，数据门未通过。现有历史BMI、历史GDP、ERA5、WorldPop、GHS-SMOD、高程、可达性和RR资料可支持方法准备，但不能支持1990–2050年正式预测。

当前缺少：GBD成人超重/肥胖未来锚点、结直肠癌详细总负担和高BMI归因结果、SSP人口与年龄性别人口、SSP网格GDP、四情景CMIP6温湿度集合。现有`$env:BMI_ROOT\data\GBD数据`仅含2020年聚合结果，不能替代上述输入。

在阻断项完成前，不启动未来MRP、不生成2050地图、不计算正式结直肠癌PAF/负担，也不扩展到全部疾病。下载后应先按`future_data_contract_v1.csv`核验字段、年份、单位、情景和95% UI。
'@
    Write-TextUtf8 $stage01 (Join-Path $OutputRoot '01_input_inventory\stage01_inventory_report_CN_v1.md')

    $workflow = @'
# 执行流程

```text
冻结历史基线
  → 补齐GBD、SSP和CMIP6数据
  → 数据门
  → 统一国家/年龄/性别/年份/情景和空间网格
  → 建立国家七分类BMI锚点
  → 拟合未来可用MRP
  → 非劣性与时间回测门
  → 生成0.5°、ADM1和0.1° BMI结果
  → 冻结结直肠癌RR与GBD病因映射
  → 计算PAF和归因负担
  → 不确定性传播与国家比较
  → 结直肠癌样板通过后扩展疾病
```

当前停点：第0步已完成；第1步因关键未来数据缺失而停在数据门前。
'@
    Write-TextUtf8 $workflow (Join-Path $OutputRoot '00_plan_manifest\implementation_workflow_CN_v1.md')
}
finally {
    Stop-Transcript | Out-Null
}

$shaPath = Join-Path $OutputRoot '15_code_logs_checksums\SHA256SUMS_stage00_01_v1.csv'
if (Test-Path -LiteralPath $shaPath) { throw "Refusing to overwrite: $shaPath" }
$shaRows = foreach ($file in (Get-ChildItem -LiteralPath $OutputRoot -Recurse -File)) {
    [pscustomobject]@{
        relative_path=[System.IO.Path]::GetRelativePath($OutputRoot,$file.FullName)
        size_bytes=[int64]$file.Length
        sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
Write-CsvUtf8Bom $shaRows $shaPath

[pscustomobject]@{
    status='stage00_completed_stage01_blocked_partial'
    inventory_rows=$inventory.Count
    missing_blocking=@($inventory | Where-Object status -eq 'missing_blocking').Count
    output_root=$OutputRoot
} | ConvertTo-Json
