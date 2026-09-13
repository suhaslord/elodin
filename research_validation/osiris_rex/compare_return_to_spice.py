#!/usr/bin/env python3
"""Compare a Sun-only OSIRIS-REx return propagation with reconstructed NAIF SPICE."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp
from spiceypy.utils.support_types import SPICEDOUBLE_CELL
ORX_ID=-64; SUN_ID=10

def coverage(spk):
    w=SPICEDOUBLE_CELL(20000); spice.spkcov(str(spk),ORX_ID,w)
    xs=[spice.wnfetd(w,i) for i in range(spice.wncard(w))]
    if not xs: raise RuntimeError('No OSIRIS-REx (-64) coverage found in SPK')
    return min(x[0] for x in xs),max(x[1] for x in xs)

def rhs(mu):
    def f(_t,y):
        r=y[:3]; rn=np.linalg.norm(r); return np.hstack((y[3:],-mu*r/rn**3))
    return f

def main():
    p=argparse.ArgumentParser(); p.add_argument('--spk',required=True,type=Path); p.add_argument('--kernel',action='append',default=[],type=Path); p.add_argument('--start-utc',default='2021-05-10T00:00:00'); p.add_argument('--stop-utc',default='2023-09-17T00:00:00'); p.add_argument('--step-hours',type=float,default=24.0); p.add_argument('--mu-sun',type=float,default=1.3271244004193938e11); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    spice.kclear()
    try:
        for k in a.kernel: spice.furnsh(str(k))
        spice.furnsh(str(a.spk)); cs,ce=coverage(a.spk); t0=spice.str2et(a.start_utc); t1=spice.str2et(a.stop_utc)
        print('SPK coverage:',spice.et2utc(cs,'ISOC',3),'to',spice.et2utc(ce,'ISOC',3))
        if t0<cs or t1>ce or t1<=t0: raise ValueError('Requested window is outside reconstructed SPK coverage')
        y0=np.asarray(spice.spkgeo(ORX_ID,t0,'J2000',SUN_ID)[0],float); step=a.step_hours*3600.0; et=np.arange(t0,t1+0.5*step,step); tau=et-t0
        sol=solve_ivp(rhs(a.mu_sun),(0,float(tau[-1])),y0,t_eval=tau,rtol=1e-11,atol=1e-12,method='DOP853')
        if not sol.success: raise RuntimeError(sol.message)
        reduced=sol.y.T; truth=np.asarray([spice.spkgeo(ORX_ID,float(e),'J2000',SUN_ID)[0] for e in et],float)
        pe=np.linalg.norm(reduced[:,:3]-truth[:,:3],axis=1); ve=1000.0*np.linalg.norm(reduced[:,3:]-truth[:,3:],axis=1)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['utc','elapsed_days','truth_x_km','truth_y_km','truth_z_km','truth_vx_km_s','truth_vy_km_s','truth_vz_km_s','twobody_x_km','twobody_y_km','twobody_z_km','twobody_vx_km_s','twobody_vy_km_s','twobody_vz_km_s','position_error_km','velocity_error_m_s'])
            for i,e in enumerate(et): w.writerow([spice.et2utc(float(e),'ISOC',3),tau[i]/86400.0,*truth[i].tolist(),*reduced[i].tolist(),pe[i],ve[i]])
        print('rows:',len(et)); print('output:',a.out); print('final position error km:',float(pe[-1])); print('max position error km:',float(pe.max())); print('final velocity error m/s:',float(ve[-1])); print('max velocity error m/s:',float(ve.max()))
        print('NOTE: errors include all omitted effects: maneuvers, third bodies, SRP, and other modeled/estimated effects.')
    finally: spice.kclear()
if __name__=='__main__': main()
