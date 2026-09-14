#!/usr/bin/env python3
"""Independent quadrature/thermodynamic checks of the C++ model output."""
import sys
import numpy as np
from scipy.integrate import quad

def sigma(E,T,al):
    if al:
        if E<.01:return .001*min(1e7,1e8*T)
        base=1e3/E**2/(1+20*T**1.5)
        if E<.1:return .001*base
        if E<1.5:return .001*(base+1e4/E**2/(1+200*T*T))
        return .001*(base*(1.5/E)**.5+1e5*(1.5/E)**2.5/(1+1000*T*T))
    if E<.008:return .001*min(1e7,1e9*T*T)
    base=192/E**2/(1+200*T**1.5)
    if E<.3:return .001*base
    return .001*(base*(.3/E)**.5+4e4*(.3/E)**2.5/(1+8000*T*T))

kevk=1.602176634e-9/1.380649e-16
a=4*5.670374419e-5/2.99792458e10

def energy(T,al):
    chi,H=(.3,.5) if al else (.1,.1)
    alpha=2/(1+np.sqrt(1+4*np.exp(chi/T)))
    return a*kevk**4*H*(T+(T+chi)*alpha)

maxerr=0
maxcdf=0
for al,T,planck,mean,e,cv,back,q10,q50,q90 in np.loadtxt(sys.argv[1]):
    def integrand(logE,moment):
        E=np.exp(logE);x=E/T
        return 0 if x>700 else 15/np.pi**4*x**4/np.expm1(x)*sigma(E,T,al)*E**moment
    edges=np.log([1e-7,.008,.01,.1,.3,1.5,100])
    integ=[sum(quad(lambda x:integrand(x,m),lo,hi,epsabs=1e-10,epsrel=1e-10)[0] for lo,hi in zip(edges[:-1],edges[1:])) for m in (0,1)]
    errs=[abs(planck/integ[0]-1),abs(mean/(integ[1]/integ[0])-1)]
    maxerr=max(maxerr,*errs)
    assert max(errs)<3e-4,(T,al,errs)
    for probability,quantile in zip([.1,.5,.9],[q10,q50,q90]):
        cut=np.log(quantile)
        cumulative=sum(quad(lambda x:integrand(x,0),lo,min(hi,cut),epsabs=1e-10,epsrel=1e-10)[0]
                       for lo,hi in zip(edges[:-1],edges[1:]) if lo<cut)/integ[0]
        maxcdf=max(maxcdf,abs(cumulative-probability))
        assert abs(cumulative-probability)<1e-3,(T,al,probability,cumulative)
    assert abs(e/energy(T,al)-1)<1e-12
    delta=T*1e-5
    derivative=(energy(T+delta,al)-energy(T-delta,al))/(2*delta)/kevk
    assert abs(cv/derivative-1)<1e-8
    assert abs(back/T-1)<1e-9
print(f'PASS: EOS inversion, Cv derivative, Planck integral, emission quantiles and mean; maximum spectral relative error {maxerr:.3g}')

print(f'Maximum absolute emission CDF discrepancy: {maxcdf:.3g} (tolerance 1e-3)')
