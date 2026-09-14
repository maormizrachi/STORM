# Su–Olson (1996) Marshak benchmark

B. Su and G. L. Olson, *Benchmark Results for the Non-Equilibrium Marshak Diffusion Problem*, JQSRT **56**, 337–351 (1996), [DOI](https://doi.org/10.1016/0022-4073(96)84524-9).

This example solves **gray radiative transport with IMC**, and compares it with
that paper's **half-space diffusion** solution. They are different equations:
the transport/diffusion discrepancy at early time is physical and is not an
MC convergence error. The agreement improves at late time. This is the 1996
boundary-driven problem, not the 1997 finite-volume-source Su–Olson problem.

## Exact problem and code conventions

- Absorption and total macroscopic opacity: 1 cm⁻¹; no physical scattering.
- Density: 1 g cm⁻³. Material energy density: `a*T^4`; volumetric heat
  capacity: `4*a*T^3`, giving epsilon = 1.
- Constant incident blackbody bath `Tb = 1 keV` at x=0, with incoming flux
  `a*c*Tb^4/4`, cosine-weighted hemispheric directions, and freely escaping
  outgoing packets. The material temperature is evolved, not prescribed.
- Initially no radiation packets; `T/Tb = 1e-6` gives negligible material
  energy `v0 = 1e-24` while avoiding zero heat capacity in the IMC interface.
- Dimensionless `X = sqrt(3)*sigma*x`, `tau = c*sigma*t`.
- Domain `0 <= X <= 40`, reflecting distant right boundary. The reference
  is semi-infinite; at tau=100 the right wall is deep in the negligible tail.
- One transverse cell in each direction with reflecting walls. Slab transport
  folds transverse trajectories but preserves the full three-dimensional
  angular distribution. DDMC and random walk are disabled.
- Snapshots at tau = 1, 10, 100. Radiation energy is the **instantaneous census
  sum divided by cell volume**, not an interval-averaged estimator.

`main.cpp` explicitly declares the reflecting faces needed by STORM's slab
transport. `SuOlsonEOS::dT2cv` returns volumetric heat capacity as required by
the current Fleck expression, and `de2T` converts specific internal energy.
With rho=1 the specific and volumetric numerical heat capacities coincide.

## Build and run

From this directory, `./build.sh` builds `su_olson` (MPI) and
`su_olson_serial` using the same main and `#ifdef STORM_WITH_MPI`.
The CMMC dependency supplies units/Planck headers; Compton **physics is off**.
The manager explicitly uses two-sided MPI, requiring no RDMA hardware.

```
./su_olson_serial 512 8 10000 0.05 output_serial
mpirun -np 4 ./su_olson 512 8 10000 0.05 output
sbatch --ntasks-per-node=4 --export=ALL,NEW=8,BOUNDARY=10000,DTAU=0.05 run.slurm
python3 compare.py output
```

Arguments: `Nx new_packet_parameter boundary_packets delta_tau output_dir`.
STORM's thermal allocation uses a global nominal budget of
`10*Ncells*new_packet_parameter`, with per-cell minimum `new_packet_parameter`
and maximum `20*new_packet_parameter`; the parameter is not an exact count.
The boundary packet count is exact per illuminated face and step. Comb
population control uses a 400-packet cell minimum, a nominal global budget
of `5*Ncells*400`, and a `20*400` per-cell maximum.
The main and SLURM defaults match the validated 512-cell run above.

The supplied SLURM script uses partition `bigrun`. Submit from this directory,
or pass `sbatch --chdir=/absolute/path/to/this/directory`.
Use a fresh output directory for separate runs.

## Independent comparison

Python dependencies: NumPy, Matplotlib, mpmath. `compare.py` derives the
Laplace-space solution directly from the coupled linear diffusion equations:

```
u_tau = u_XX + v - u,  v_tau = u - v
u(0,tau) - 2/sqrt(3)*u_X(0,tau) = 1
k(s) = sqrt(s*(1 + 1/(s+1)))
U(X,s) = exp(-k(s)*X)/(s*(1+2*k(s)/sqrt(3)))
V(X,s) = U(X,s)/(s+1)
```

The script uses de Hoog inversion at 22 decimal digits, independently repeats
selected points at 32 digits, and requires agreement within 1e-10. It writes
`comparison.png` and `metrics.json`, reporting relative L1 discrepancies for
radiation energy, material energy, and material temperature over the plotted
ranges (X<4, 10, 25 at tau=1, 10, 100). The far reflecting boundary is excluded
from these half-space error norms. References are cached as `.npz` files.
Delete those caches to regenerate the reference computation.
The raw profiles also include `T/Tb = v^(1/4)`.

SLURM launches use a unique executable copy in `.job_binaries/` for each job,
so a subsequent rebuild cannot invalidate an executable mapped over NFS.

See [RESULTS.md](RESULTS.md) for recorded runs, numerical discrepancies,
and retained profiles and figures.
