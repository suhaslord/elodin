#!/usr/bin/env python3
"""Compare the delivered reduced-order InSight entry model with reconstructed SPICE.

The reduced model has only scalar altitude and speed, not a 3-D state.  We therefore
align t=0 to the reconstructed descending 128 km altitude crossing and compare only
altitude/speed histories; no 3-D trajectory-accuracy claim is made.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import spiceypy as spice
from spiceypy.utils.support_types import SPICEDOUBLE_CELL

INSIGHT_ID=-189
MARS_ID=499
ENTRY_ALT_KM=128.0

def coverage(spk: Path):
    window=SPICEDOUBLE_CELL(2000)
    spice.spkcov(str(spk), INSIGHT_ID, window)
    intervals=[spice.wnfetd(window,i) for i in range(spice.wncard(window))]
    if not intervals: raise RuntimeError('No InSight (-189) coverage found in SPK')
    return min(x[0] for x in intervals), max(x[1] for x in intervals)

def mars_mean_radius():
    try: return float(np.asarray(spice.bodvrd('MARS','RADII',3)[1],float).mean())
    except Exception: return 3389.5

def state_at(et):
    return np.asarray(spice.spkgeo(INSIGHT_ID,float(et),'J2000',MARS_ID)[0],float)

def altitude_km(et):
    return float(np.linalg.norm(state_at(et)[:3])-mars_mean_radius())

def find_entry_crossing(start, stop):
    # Locate the first descending 128 km crossing, then linearly refine it.
    grid=np.linspace(start,stop,max(3,int(np.ceil((stop-start)/0.5))+1))
    alt=np.asarray([altitude_km(et) for et in grid])
    idx=np.where((alt[:-1] > ENTRY_ALT_KM) & (alt[1:] <= ENTRY_ALT_KM))[0]
    if idx.size == 0:
        raise RuntimeError('No descending 128 km entry crossing found in reconstructed SPK coverage')
    i=int(idx[0]); a0,a1=alt[i],alt[i+1]
    frac=(a0-ENTRY_ALT_KM)/(a0-a1)
    return float(grid[i]+frac*(grid[i+1]-grid[i]))

def truth_at(ets):
    states=np.asarray([state_at(et) for et in ets],float)
    rmean=mars_mean_radius()
    return np.linalg.norm(states[:,:3],axis=1)-rmean, np.linalg.norm(states[:,3:],axis=1)*1000.0

def main():
    p=argparse.ArgumentParser(); p.add_argument('--spk',required=True,type=Path); p.add_argument('--kernel',action='append',default=[],type=Path); p.add_argument('--reduced-csv',type=Path,required=True); p.add_argument('--out',type=Path,default=Path('insight_edl_scalar_comparison.csv')); a=p.parse_args()
    spice.kclear()
    try:
        for k in a.kernel: spice.furnsh(str(k))
        spice.furnsh(str(a.spk)); start,stop=coverage(a.spk); entry=find_entry_crossing(start,stop)
        rows=list(csv.DictReader(a.reduced_csv.open(encoding='utf-8')))
        t=np.asarray([float(r['time_s']) for r in rows]); model_alt=np.asarray([float(r['altitude_km']) for r in rows]); model_speed=np.asarray([float(r['velocity_m_s']) for r in rows])
        ets=entry+t; keep=ets<=stop; t=t[keep]; ets=ets[keep]; model_alt=model_alt[keep]; model_speed=model_speed[keep]
        truth_alt,truth_speed=truth_at(ets); alt_err=model_alt-truth_alt; speed_err=model_speed-truth_speed
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['utc','elapsed_s','model_altitude_km','spice_altitude_km','altitude_error_km','model_speed_m_s','spice_speed_m_s','speed_error_m_s'])
            for i,et in enumerate(ets): w.writerow([spice.et2utc(float(et),'ISOC',3),t[i],model_alt[i],truth_alt[i],alt_err[i],model_speed[i],truth_speed[i],speed_err[i]])
        print('SPK coverage:',spice.et2utc(start,'ISOC',3),'to',spice.et2utc(stop,'ISOC',3))
        print('128 km entry alignment:',spice.et2utc(entry,'ISOC',3)); print('samples compared:',len(t)); print('comparison CSV:',a.out)
        print('altitude RMSE km:',float(np.sqrt(np.mean(alt_err**2)))); print('altitude max abs km:',float(np.max(np.abs(alt_err))))
        print('speed RMSE m/s:',float(np.sqrt(np.mean(speed_err**2)))); print('speed max abs m/s:',float(np.max(np.abs(speed_err))))
        print('NOTE: scalar altitude/speed validation only; the reduced model contains no 3-D state.')
    finally: spice.kclear()
if __name__=='__main__': main()
