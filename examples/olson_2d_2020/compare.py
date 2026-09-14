#!/usr/bin/env python3
"""Compare diagonal profiles with Olson PN markers extracted from Fig 6.
Plot axes follow the paper: log10(T_keV^4), log10(E/(a TkeV^4)).
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_field(directory, tag):
    path=directory/f'field_ct{tag}.txt'
    return np.loadtxt(path if path.exists() else path.with_suffix('.txt.gz'))

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',nargs='?',type=Path,default=Path(__file__).parent/'output')
p.add_argument('--check',action='store_true')
p.add_argument('--coarse',type=Path,help='Optional coarser run for a mesh-sensitivity comparison')
args=p.parse_args();out=args.output;here=Path(__file__).parent
fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained');metrics={}
for ct,tag,color in [(2,'2','#222222'),(2.5,'2p5','#0072B2'),(3,'3','#D55E00')]:
    data=np.loadtxt(out/f'profile_ct{tag}.txt');r,T,E,al=data.T
    assert np.all(np.isfinite(data)) and np.all(T>0) and np.all(E>=0)
    metrics[str(ct)]={}
    for ax,quantity,values in zip(axes,['material_T4','radiation'],[4*np.log10(T),np.log10(np.maximum(E,1e-300))]):
        ref=np.loadtxt(here/'reference'/f'{quantity}.csv',delimiter=',',skiprows=1)
        ref=ref[ref[:,0]==ct];rr,yy=ref[:,1:].T
        valid=(rr>=r[0])&(rr<=r[-1])
        # No interpolation across the discontinuities at r=sqrt(2)*0.5,1.5,2.5.
        dx=np.sqrt(2)*3.8/np.sqrt(load_field(out,tag).shape[0])
        for edge in np.sqrt(2)*np.array([.5,1.5,2.5]):valid&=abs(rr-edge)>dx
        prediction=np.interp(rr,r,values)
        finite=valid&(prediction>-20)
        metrics[str(ct)][quantity]={'log10_RMSE':float(np.sqrt(np.mean((prediction[finite]-yy[finite])**2))),
            'compared_markers':int(finite.sum()),'excluded_interface_radius_cm':float(dx)}
        core=finite&(rr<np.sqrt(.5)-dx)
        metrics[str(ct)][quantity]['foam_log10_RMSE']=float(np.sqrt(np.mean((prediction[core]-yy[core])**2)))
        metrics[str(ct)][quantity]['foam_markers']=int(core.sum())
        ax.plot(r,values,color=color,lw=1,label=f'IMC ct={ct:g}')
        ax.scatter(rr,yy,facecolors='none',edgecolors=color,s=22,label=f'Published PN ct={ct:g}')
        ax.grid(alpha=.2)
    published=np.loadtxt(here/'reference'/'published_imc_material_T4.csv',delimiter=',',skiprows=1)
    published=published[published[:,0]==ct]
    axes[0].plot(published[:,1],published[:,2],ls='--',color=color,lw=.9,alpha=.75)
    sample=np.linspace(max(r[0],published[0,1]),1.4,150)
    mask=abs(sample-np.sqrt(.5))>dx
    simlog=np.interp(sample,r,4*np.log10(T))
    reflog=np.interp(sample,published[:,1],published[:,2])
    metrics[str(ct)]['published_IMC_material']={
        'log10_T4_RMSE':float(np.sqrt(np.mean((simlog[mask]-reflog[mask])**2))),
        'temperature_relative_L1':float(np.sum(abs(10**(simlog[mask]/4)-10**(reflog[mask]/4)))/np.sum(10**(reflog[mask]/4)))}
    axes[0].set(xlim=(0,1.8),ylim=(-8,-3.8),xlabel='Diagonal radius r (cm)',ylabel=r'$\log_{10}[(T/\mathrm{keV})^4]$')
    axes[1].set(xlim=(0,2.25),ylim=(-8,-1.25),xlabel='Diagonal radius r (cm)',ylabel=r'$\log_{10}[E/(aT_{\mathrm{keV}}^4)]$')
axes[1].legend(loc='lower left',fontsize=7,ncol=2)
axes[0].text(.03,.05,'Dashed lines: published IMC',transform=axes[0].transAxes,fontsize=8)
fig.suptitle('Olson 2020 2D — continuous-frequency IMC, published PN markers and IMC curves')
fig.savefig(out/'comparison.png',dpi=180)
field=load_field(out,'3');n=round(np.sqrt(len(field)))
assert n*n==len(field), 'Incomplete Cartesian field'
metrics['aluminum_area_cm2']=float(field[:,4].sum()*(3.8/n)**2)
assert abs(metrics['aluminum_area_cm2']-2.5)<1e-10, 'Incorrect block geometry'
fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
m=ax.imshow(np.log10(np.maximum(field[:,3].reshape(n,n).T,1e-8)),origin='lower',extent=(0,3.8,0,3.8),vmin=-8,vmax=-1.5,cmap='turbo',interpolation='nearest')
for x,y,w,h in [(.5,.5,1,1),(1.5,1.5,1,1),(1.5,0,1,.5)]:ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='white',lw=.8,alpha=.7))
ax.set(xlabel='x (cm)',ylabel='y (cm)',title=f'Olson 2020 — ct=3 cm, {n} × {n} cells')
fig.colorbar(m,ax=ax,label=r'$\log_{10}[E/(aT_{\mathrm{keV}}^4)]$')
fig.savefig(out/'radiation_map.png',dpi=180)
if (out/'energy_budget.txt').exists():
    budget=np.atleast_2d(np.loadtxt(out/'energy_budget.txt'))
    metrics['maximum_energy_budget_residual']=float(np.max(abs(budget[:,-1])))
    # Analytic quadrant integral of exp(-18.7*r^3). The part outside the
    # 3.8 cm square is smaller than exp(-1000), negligible in double precision.
    arad=4*5.670374419e-5/2.99792458e10
    kevK=1.602176634e-9/1.380649e-16
    area=math.pi*math.gamma(2/3)/(6*18.7**(2/3))
    expected=budget[:,0]*arad*(.5*kevK)**4*area
    metrics['source_normalization_relative_error']=float(np.max(abs(budget[:,2]/expected-1)))
if args.coarse:
    coarse=load_field(args.coarse,'3');nc=round(np.sqrt(len(coarse)))
    assert n%nc==0 and nc<n and nc*nc==len(coarse)
    factor=n//nc
    averaged=field[:,2:4].reshape(nc,factor,nc,factor,2).mean(axis=(1,3)).reshape(-1,2)
    heated=averaged[:,0]>.011
    metrics['mesh_sensitivity']={
        'coarse_n':nc,'fine_n':n,
        'heated_material_temperature_relative_L1':float(np.sum(abs(averaged[heated,0]-coarse[heated,2]))/np.sum(averaged[heated,0])),
        'radiation_relative_L1':float(np.sum(abs(averaged[:,1]-coarse[:,3]))/np.sum(averaged[:,1])),
        'note':'Includes Monte Carlo sampling differences; fine fields volume-averaged to coarse cells'}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for directory,label,color in [(args.coarse,f'IMC {nc} × {nc}','#0072B2'),(out,f'IMC {n} × {n}','#D55E00')]:
        line=np.loadtxt(directory/'profile_ct3.txt')
        axes[0].plot(line[:,0],4*np.log10(line[:,1]),color=color,lw=1,label=label)
        axes[1].plot(line[:,0],np.log10(np.maximum(line[:,2],1e-300)),color=color,lw=1,label=label)
    ref=np.loadtxt(here/'reference'/'published_imc_material_T4.csv',delimiter=',',skiprows=1)
    ref=ref[ref[:,0]==3];axes[0].plot(ref[:,1],ref[:,2],'k--',lw=.9,label='Published IMC')
    for ax,q,xmax in zip(axes,['material_T4','radiation'],[1.8,2.25]):
        ref=np.loadtxt(here/'reference'/f'{q}.csv',delimiter=',',skiprows=1);ref=ref[ref[:,0]==3]
        ax.scatter(ref[:,1],ref[:,2],facecolors='none',edgecolors='#777777',s=20,label='Published PN')
        ax.set(xlim=(0,xmax),ylim=(-8,-3.8 if q=='material_T4' else -1.25),xlabel='Diagonal radius r (cm)')
        ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=7)
    axes[0].set_ylabel(r'$\log_{10}[(T/\mathrm{keV})^4]$')
    axes[1].set_ylabel(r'$\log_{10}[E/(aT_{\mathrm{keV}}^4)]$')
    fig.suptitle('Olson 2020 — mesh sensitivity at ct=3 cm (independent MC histories)')
    fig.savefig(out/'mesh_comparison.png',dpi=180)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))

if args.check:
    assert metrics.get('maximum_energy_budget_residual',1)<1e-8, 'Energy budget failed'
    assert metrics.get('source_normalization_relative_error',1)<.02, 'Volume-source normalization failed'
    for ct in ['2','2.5','3']:
        # Sec. IV explicitly reports PN/MC agreement in the foam before
        # r=sqrt(0.5); beyond it, their interface and radiation tails differ.
        # Report full PN discrepancies, but test the common smooth region
        # and independently test the full material profile against IMC.
        for q in ['material_T4','radiation']:
            assert metrics[ct][q]['foam_markers']>=4, 'Insufficient foam-reference coverage'
            assert metrics[ct][q]['foam_log10_RMSE']<.1, (ct,q,metrics[ct][q])
        assert metrics[ct]['published_IMC_material']['temperature_relative_L1']<.08, (ct,metrics[ct])
    print('PASS: energy conservation and published-reference comparison')
