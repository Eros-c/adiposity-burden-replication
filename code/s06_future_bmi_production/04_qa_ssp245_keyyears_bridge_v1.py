from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import csv, json
from pathlib import Path
import numpy as np
import rasterio

ROOT=Path(_cfg.ROOT / "future_scenario_ascii" / "06_future_bmi_2025_2050" / "bridge_ncd2024_gbdtrend_sensitivity_v1_20260909")
YEARS=(2025,2030,2035,2040,2045,2050)
COUNTS={"probability_overall":7,"probability_sex":14,"probability_age":105,"probability_age_sex":210}
rows=[]
for year in YEARS:
    summary=json.loads((ROOT/'qa'/f'ssp245_{year}_bridge_summary_v1.json').read_text(encoding='utf-8'))
    if summary.get('status')!='PASS_BRIDGE_PILOT': raise RuntimeError(f'{year} bridge summary not PASS')
    for family,count in COUNTS.items():
        path=ROOT/'outputs'/'ssp245'/str(year)/f'{family}_ssp245_{year}_0p1deg_bridge_v1.tif'
        with rasterio.open(path) as ds:
            if (ds.count,ds.width,ds.height,ds.crs.to_string(),ds.nodata)!=(count,3600,1800,'EPSG:4326',-9999.0): raise RuntimeError(f'Invalid structure {path}')
            sample=ds.read(window=rasterio.windows.Window(1000,600,512,512)).reshape(count,-1)
            valid=np.all(np.isfinite(sample)&(sample!=ds.nodata),axis=0)
            if valid.any():
                if family=='probability_overall': groups=sample[:,valid].T.reshape(-1,7)
                elif family=='probability_sex': groups=sample[:,valid].T.reshape(-1,2,7).reshape(-1,7)
                elif family=='probability_age': groups=sample[:,valid].T.reshape(-1,15,7).reshape(-1,7)
                else: groups=sample[:,valid].T.reshape(-1,30,7).reshape(-1,7)
                closure=float(np.max(np.abs(groups.sum(axis=1)-1)))
            else: closure=float('nan')
            rows.append({'year':year,'family':family,'path':str(path),'bands':ds.count,'bytes':path.stat().st_size,'sample_group_closure_max_error':closure,'summary_country_target_error':summary['maximum_country_target_error'],'pass':bool(np.isfinite(closure) and closure<1e-5 and summary['maximum_country_target_error']<=1e-8)})
if not all(row['pass'] for row in rows): raise RuntimeError('Bridge key-year QA failed')
qa=ROOT/'qa'; qa.mkdir(exist_ok=True)
with (qa/'ssp245_keyyears_bridge_full_qa_v1.csv').open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(qa/'ssp245_keyyears_bridge_full_qa_v1.json').write_text(json.dumps({'status':'PASS','years':YEARS,'files':len(rows),'max_sample_closure_error':max(row['sample_group_closure_max_error'] for row in rows),'max_country_target_error':max(row['summary_country_target_error'] for row in rows)},indent=2)+'\n',encoding='utf-8')
print('BRIDGE_KEYYEARS_QA_PASS')
