#!/usr/bin/env python3
"""Su-Olson epsilon=1 half-space diffusion solution by Laplace inversion.

Dimensionless PDE: u_tau=u_XX+v-u, v_tau=u-v; initially u=v=0.
Marshak BC u(0,tau)-(2/sqrt(3))*u_X(0,tau)=1, u(infinity,tau)=0.
U(X,s)=exp(-kX)/(s*(1+2*k/sqrt(3))), k=sqrt(s*(1+1/(s+1))).
V=U/(s+1). This is an independent semianalytic evaluation, not a fit to MC.
IMC solves transport: disagreement with diffusion at tau=1 is expected.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def solution(x, tau, material=False, dps=22):
    with mp.workdps(dps):
        def transform(s):
            k=mp.sqrt(s*(1+1/(s+1)))
            val=mp.exp(-k*x)/(s*(1+2*k/mp.sqrt(3)))
            return val/(s+1) if material else val
        return float(mp.invertlaplace(transform,tau,method='dehoog'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',nargs='?',type=Path,default=Path(__file__).parent/'output')
    p.add_argument('--check',action='store_true')
    args=p.parse_args(); out=args.output
    fig,panels=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    axes=panels[0]
    metrics={}
    for ax,tau in zip(axes,[1,10,100]):
        data=np.loadtxt(out/f'profile_tau{tau}.txt'); x,u,v,_=data.T
        assert np.all(np.isfinite(data)) and np.all(data[:,1:]>=0)
        xmax={1:4,10:10,100:25}[tau]
        xr=np.linspace(0,xmax,121)
        cache=out/f'reference_tau{tau}.npz'
        if cache.exists():
            ref=np.load(cache); xr,ur,vr=ref['x'],ref['u'],ref['v']
        else:
            ur=np.array([solution(float(xx),tau) for xx in xr])
            vr=np.array([solution(float(xx),tau,True) for xx in xr])
            np.savez(cache,x=xr,u=ur,v=vr)
        # Independent precision repeat at the surface, inside wave and tail.
        for j in (0,30,90):
            assert abs(solution(float(xr[j]),tau,dps=32)-ur[j])<1e-10
        inside=x<xmax
        ru,rv=np.interp(x[inside],xr,ur),np.interp(x[inside],xr,vr)
        metrics[str(tau)]={'radiation_relative_L1':float(np.sum(abs(u[inside]-ru))/np.sum(ru)),
                           'material_relative_L1':float(np.sum(abs(v[inside]-rv))/np.sum(rv)),
                           'material_temperature_relative_L1':float(np.sum(abs(v[inside]**.25-np.maximum(rv,0)**.25))/np.sum(np.maximum(rv,0)**.25))}
        ax.plot(xr,ur,'k-',label='Diffusion radiation');ax.plot(xr,vr,'k--',label='Diffusion material')
        ax.plot(x[inside],u[inside],color='#0072B2',label='IMC radiation',lw=1)
        ax.plot(x[inside],v[inside],color='#D55E00',label='IMC material',lw=1)
        ax.set(xlim=(0,xmax),ylim=(0,1),xlabel=r'$X=\sqrt{3}\,\sigma x$',title=rf'$\tau={tau}$')
        ax.grid(alpha=.2)
        tax=panels[1,list(axes).index(ax)]
        tax.plot(xr,np.maximum(vr,0)**.25,'k--',label='Diffusion material temperature')
        tax.plot(x[inside],v[inside]**.25,color='#D55E00',lw=1,label='IMC material temperature')
        tax.set(xlim=(0,xmax),ylim=(0,1),xlabel=r'$X=\sqrt{3}\,\sigma x$')
        tax.grid(alpha=.2)
    axes[0].set_ylabel(r'$u=E/(aT_b^4),\quad v=(T/T_b)^4$');axes[2].legend(fontsize=8)
    panels[1,0].set_ylabel(r'$T/T_b$');panels[1,2].legend(fontsize=8)
    fig.suptitle('Su–Olson (1996), ε=1 — transport versus semianalytic diffusion')
    fig.savefig(out/'comparison.png',dpi=180)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
    if args.check:
        assert metrics['100']['radiation_relative_L1']<0.06, 'Late radiation discrepancy exceeds 6%'
        assert metrics['100']['material_relative_L1']<0.06, 'Late material discrepancy exceeds 6%'
        print('PASS: late-time diffusion-limit comparison')

if __name__=='__main__': main()
