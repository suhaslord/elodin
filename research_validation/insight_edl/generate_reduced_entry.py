#!/usr/bin/env python3
"""Regenerate the delivered reduced-order InSight scalar entry trajectory."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

H0=128000.0; V0=5500.0; H_PAR=11100.0; V_PAR=385.0; T_PAR=218.0
RHO0=0.02; SCALE_H=11100.0; G=3.71

def integrate(beta,gamma_deg,t_eval=None):
    gamma=np.deg2rad(gamma_deg)
    def rhs(_t,y):
        h,v=y; rho=RHO0*np.exp(-h/SCALE_H)
        return [-v*np.sin(gamma), G*np.sin(gamma)-0.5*rho*v*v/beta]
    return solve_ivp(rhs,(0,T_PAR),(H0,V0),t_eval=t_eval,max_step=0.2,rtol=1e-8,atol=1e-10)

def milestone(beta,gamma_deg):
    gamma=np.deg2rad(gamma_deg)
    def rhs(_t,y):
        h,v=y; rho=RHO0*np.exp(-h/SCALE_H)
        return [-v*np.sin(gamma), G*np.sin(gamma)-0.5*rho*v*v/beta]
    def event(_t,y): return y[0]-H_PAR
    event.terminal=True; event.direction=-1
    sol=solve_ivp(rhs,(0,800),(H0,V0),events=event,max_step=0.2,rtol=1e-8,atol=1e-10)
    return float(sol.t_events[0][0]),float(sol.y_events[0][0][1])

def residual(x):
    t,v=milestone(x[0],x[1]); return [t-T_PAR,v-V_PAR]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--samples',type=int,default=600); a=p.parse_args()
    fit=root(residual,[95.0,9.0]);
    if not fit.success: raise RuntimeError(fit.message)
    beta,gamma=fit.x; t=np.linspace(0,T_PAR,a.samples); sol=integrate(beta,gamma,t)
    if not sol.success: raise RuntimeError(sol.message)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['time_s','altitude_km','velocity_m_s'])
        for ti,h,v in zip(sol.t,sol.y[0],sol.y[1]): w.writerow([ti,h/1000.0,v])
    print(f'effective beta kg/m^2: {beta:.6f}'); print(f'effective fixed gamma deg: {gamma:.6f}'); print(f'output: {a.out}')
if __name__=='__main__': main()
