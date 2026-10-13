"""Derive closure-preserving aggregate bridge probabilities from age-sex COGs."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))
import paths as _cfg  # noqa: E402  # central path config (env-overridable)
import csv, json, os
from pathlib import Path
import numpy as np
import rasterio

ROOT=Path(_cfg.ROOT / "future_scenario_ascii" / "06_future_bmi_2025_2050" / "bridge_ncd2024_gbdtrend_sensitivity_v1_20260909")
FORMAL=Path(_cfg.ROOT / "future_scenario_ascii" / "06_future_bmi_2025_2050" / "full_covariate_annual_0p1deg_production_v1_20260906")
YEAR=int(os.environ.get("BRIDGE_YEAR","2025")); CODE="ssp245"; NODATA=-9999.0
BMI=("bmi_lt_18_5","bmi_18_5_20","bmi_20_25","bmi_25_30","bmi_30_35","bmi_35_40","bmi_ge_40")
SEXES=("Female","Male"); AGES=("25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85-89","90-94","95+")

def desc(family):
    if family=='probability_age': return [f"{a.replace('-','_').replace('+','plus')}__{b}" for a in AGES for b in BMI]
    if family=='probability_sex': return [f"{s}__{b}" for s in SEXES for b in BMI]
    if family=='probability_overall': return list(BMI)
    raise KeyError(family)

def out_path(family): return ROOT/'outputs'/CODE/str(YEAR)/f'{family}_{CODE}_{YEAR}_0p1deg_bridge_v2.tif'

def create(path,source,count,descriptions):
    if path.exists(): raise FileExistsError(path)
    profile=source.profile.copy();profile.update(driver='GTiff',count=count,dtype='float32',nodata=NODATA,tiled=True,blockxsize=256,blockysize=256,compress='ZSTD',zstd_level=1,predictor=3,BIGTIFF='IF_SAFER')
    dst=rasterio.open(path, 'w', **profile)
    for i,d in enumerate(descriptions,1):dst.set_band_description(i,d)
    dst.update_tags(product_role='closure-preserving aggregate from bridge age-sex v1',year=str(YEAR),scenario_code=CODE)
    return dst

def main():
    base=ROOT/'outputs'/CODE/str(YEAR)
    age_sex=base/f'probability_age_sex_{CODE}_{YEAR}_0p1deg_bridge_v1.tif'
    population=FORMAL/'outputs'/CODE/str(YEAR)/f'population_age_sex_{CODE}_{YEAR}_0p1deg_v1.tif'
    outputs={f:out_path(f) for f in ('probability_age','probability_sex','probability_overall')}
    with rasterio.open(age_sex) as ps, rasterio.open(population) as qs:
        dst={f:create(outputs[f],ps,{'probability_age':105,'probability_sex':14,'probability_overall':7}[f],desc(f)) for f in outputs}
        try:
            for row in range(0,ps.height,40):
                h=min(40,ps.height-row);w=rasterio.windows.Window(0,row,ps.width,h)
                p=ps.read(window=w).reshape(2,15,7,h,ps.width).astype(np.float64)
                pop=qs.read(window=w).reshape(2,15,h,ps.width).astype(np.float64)
                valid=(pop>0)&np.isfinite(pop)&np.all(np.isfinite(p)&(p!=ps.nodata),axis=2)
                pop=np.where(valid,pop,0.0);p=np.where(valid[:,:,None],p,0.0)
                tiny=np.finfo(np.float64).tiny
                age_pop=pop.sum(axis=0); age_prob=(p*pop[:,:,None]).sum(axis=0)/np.maximum(age_pop[:,None],tiny)
                sex_pop=pop.sum(axis=1); sex_prob=(p*pop[:,:,None]).sum(axis=1)/np.maximum(sex_pop[:,None],tiny)
                total_pop=sex_pop.sum(axis=0); total_prob=(sex_prob*sex_pop[:,None]).sum(axis=0)/np.maximum(total_pop[None],tiny)
                for a in range(15): dst['probability_age'].write(np.where(age_pop[a,None]>0,age_prob[a],NODATA).astype(np.float32),indexes=list(range(a*7+1,a*7+8)),window=w)
                for s in range(2): dst['probability_sex'].write(np.where(sex_pop[s,None]>0,sex_prob[s],NODATA).astype(np.float32),indexes=list(range(s*7+1,s*7+8)),window=w)
                dst['probability_overall'].write(np.where(total_pop[None]>0,total_prob,NODATA).astype(np.float32),window=w)
        finally:
            for d in dst.values():d.close()
    rows=[]
    for f,p in outputs.items():
        with rasterio.open(p) as ds:
            x=ds.read(window=rasterio.windows.Window(1000,600,512,512)).reshape(ds.count,-1);v=np.all(np.isfinite(x)&(x!=ds.nodata),axis=0);cl=float(np.max(np.abs(x[:,v].T.reshape(-1,7).sum(1)-1))) if v.any() else float('nan')
            rows.append({'year':YEAR,'family':f,'path':str(p),'bands':ds.count,'bytes':p.stat().st_size,'sample_closure_error':cl,'pass':bool(np.isfinite(cl) and cl<1e-5)})
    if not all(r['pass'] for r in rows):raise RuntimeError('Aggregate bridge QA failed')
    q=ROOT/'qa';q.mkdir(exist_ok=True)
    with (q/f'{CODE}_{YEAR}_bridge_aggregate_v2_qa.csv').open('w',encoding='utf-8-sig',newline='') as h:
        wr=csv.DictWriter(h,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print('BRIDGE_AGGREGATE_V2_PASS')

if __name__=='__main__':main()
