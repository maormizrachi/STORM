/*
 * Moving slab benchmark — gray IMC adaptation
 *
 * Based on the McClarren & Gentile (2021) moving radiating slab problem
 * from RICH (regression_tests/cases/moving_slab_mc).
 *
 * A uniform slab of material (rho = 0.1 g/cm^3, T = 1 keV, L = 0.4 cm)
 * radiates into vacuum.  In RICH this slab moves at v = 0.5994 cm/ns with
 * 124-group frequency-dependent transport and Doppler shifts.
 *
 * STORM adaptation (gray, static slab):
 *   - The 124-group opacity table is collapsed to a single gray Planck-mean.
 *   - The slab is placed at its t=t_O/2 midpoint position so the observer
 *     distance is representative of the time-averaged geometry.
 *   - No Doppler effects.  This means the output is the frequency-integrated
 *     radiation energy density profile — not the Doppler-shifted spectrum.
 *   - Very large cv prevents the slab from cooling (mimics noHydroFeedback).
 *
 * The benchmark verifies: photon emission from a hot slab, free-streaming
 * through vacuum, transparent + reflecting boundary conditions, and gray
 * IMC energy conservation.
 *
 * Usage:
 *   ./moving_slab [Nx] [new_per_cell]
 *
 *   Nx:           total x-cells (default 80)
 *   new_per_cell: volume emission photons per cell per step (default 100)
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <string>
#include <numeric>
#include <algorithm>
#include "examples/Vector3D.hpp"
#include "MadCart/CartesianMesh3D.hpp"
#include "PhysicalConstants.hpp"
#include "radiation/RadiationIMC.hpp"
#include "radiation/RadiationCell.hpp"
#include "population/CombPopulationControl.hpp"
#include "manager/MonteCarloManagerSerial.hpp"
#include "MovingSlabOpacity.hpp"
#include "MovingSlabBoundary.hpp"

using Grid = MadCart::CartesianMesh3D<Vector3D>;

class MovingSlabEOS
{
public:
    MovingSlabEOS(double cvSlab, double cvVac, double rhoSlab, double rhoVac)
        : cvPerMassSlab_(cvSlab / rhoSlab), cvPerMassVac_(cvVac / rhoVac),
          rhoSlab_(rhoSlab)
    {}

    double dT2cv(double density, double /*temperature*/,
                 const std::vector<double> &, const std::vector<std::string> &) const
    {
        return (density > 0.5 * rhoSlab_) ? cvPerMassSlab_ : cvPerMassVac_;
    }

    double de2T(double density, double specificEnergy,
                const std::vector<double> &tracers, const std::vector<std::string> &tracerNames) const
    {
        double cvPerMass = dT2cv(density, 0.0, tracers, tracerNames);
        return (cvPerMass > 0.0) ? specificEnergy / cvPerMass : 0.0;
    }

private:
    double cvPerMassSlab_;
    double cvPerMassVac_;
    double rhoSlab_;
};

int main(int argc, char *argv[])
{
    using IMC = STORM::RadiationIMC<Vector3D, Grid, STORM::RadiationCell, STORM::SimpleExtensives,
                                    MovingSlabEOS, 1>;

    size_t Nx = (argc >= 2) ? std::stoul(argv[1]) : 80;
    size_t newPhotonsPerCell = (argc >= 3) ? std::stoul(argv[2]) : 100;

    double const rhoSlab   = 0.1;
    double const L_slab    = 0.4;
    double const T_slab_keV = 1.0;
    double const T_slab    = T_slab_keV * STORM::constants::kev_kelvin;
    double const rhoVacuum = 1e-10;
    double const vSlab     = 0.5994e9;
    double const tO        = 10e-9;
    double const zO        = 12.0;

    double const slabMidShift = vSlab * tO * 0.5;
    double const xSlabStart   = slabMidShift;
    double const xSlabEnd     = xSlabStart + L_slab;
    double const xMax         = zO + 0.2;
    double const dy           = xMax / Nx;

    Vector3D lower(0, 0, 0);
    Vector3D upper(xMax, dy, dy);

    Grid grid(lower, upper, Nx, 1, 1);
    size_t Ncells = grid.GetPointNo();

    std::cout << "Moving slab benchmark (gray, static): "
              << Ncells << " cells, domain [0, " << xMax << "] cm" << std::endl;
    std::cout << "Slab position [" << xSlabStart << ", " << xSlabEnd << "] cm"
              << "  T_slab=" << T_slab_keV << " keV"
              << "  rho=" << rhoSlab << " g/cm^3"
              << "  observer z_O=" << zO << " cm" << std::endl;

    double const cvPerVolumeSlab = 1e23 / STORM::constants::kev_kelvin;
    double const cvPerVolumeVac  = 1e10;
    double const T_vac   = 100.0;

    std::vector<STORM::RadiationCell> cells(Ncells);
    std::vector<STORM::SimpleExtensives> extensives(Ncells);
    std::vector<double> densities(Ncells);

    for(size_t i = 0; i < Ncells; i++)
    {
        double x = grid.GetCellCM(i).x;
        double volume = grid.GetVolume(i);
        bool inSlab = (x >= xSlabStart and x <= xSlabEnd);
        double rho = inSlab ? rhoSlab : rhoVacuum;
        double cvPerVolume = inSlab ? cvPerVolumeSlab : cvPerVolumeVac;
        double T = inSlab ? T_slab : T_vac;

        densities[i] = rho;
        cells[i].temperature = T;
        cells[i].internalEnergy = cvPerVolume * T * volume;
        extensives[i].mass = rho * volume;
        extensives[i].internal_energy = cells[i].internalEnergy;
    }

    STORM::RadiationIMCParameters<1> imcParams;
    imcParams.newPhotonsPerCell = newPhotonsPerCell;
    imcParams.noHydroFeedback = true;
    imcParams.energyBoundaries = {0.0, 1e30};
    imcParams.energyBoundariesProvided = true;

    std::shared_ptr<MovingSlabEOS> eos =
        std::make_shared<MovingSlabEOS>(cvPerVolumeSlab, cvPerVolumeVac, rhoSlab, rhoVacuum);
    std::shared_ptr<STORM::examples::MovingSlabOpacity<Vector3D, Grid>> opacityModel =
        std::make_shared<STORM::examples::MovingSlabOpacity<Vector3D, Grid>>(rhoSlab, densities, cells);
    std::shared_ptr<STORM::examples::MovingSlabBoundary<Vector3D, Grid>> boundary =
        std::make_shared<STORM::examples::MovingSlabBoundary<Vector3D, Grid>>(grid);
    std::shared_ptr<IMC> physics =
        std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacityModel, imcParams);
    std::shared_ptr<STORM::CombPopulationControl<Vector3D, Grid>> popControl =
        std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, 15, 6.0);

    STORM::MonteCarloManagerSerial<Vector3D, Grid> manager(grid, physics, popControl, boundary);
    std::vector<STORM::Particle<Vector3D, Grid>> particles;

    double dt = 1e-12;
    double const dtMax = 1e-10;
    double const dtRamp = 1.1;
    double simTime = 0;
    size_t cycle = 0;
    double const tEnd = tO;

    std::cout << "new_per_cell=" << newPhotonsPerCell
              << "  t_end=" << tEnd * 1e9 << " ns" << std::endl;
    std::cout << std::endl;

    while(simTime < tEnd)
    {
        double thisDt = std::min(dt, tEnd - simTime);
        if(thisDt <= 0)
        {
            break;
        }

        particles = manager.step(std::move(particles), thisDt);

        simTime += thisDt;
        cycle++;
        dt = std::min(dt * dtRamp, dtMax);

        if(cycle % 20 == 0 or simTime >= tEnd)
        {
            double maxErad = 0;
            double totalErad = 0;
            for(size_t i = 0; i < Ncells; i++)
            {
                double eradTotal = extensives[i].Erad;
                double eradDensity = eradTotal / grid.GetVolume(i);
                maxErad = std::max(maxErad, eradDensity);
                totalErad += eradTotal;
            }
            double maxT_keV = 0;
            for(size_t i = 0; i < Ncells; i++)
            {
                maxT_keV = std::max(maxT_keV, cells[i].temperature / STORM::constants::kev_kelvin);
            }
            std::cout << "Cycle " << cycle
                      << "  t=" << simTime * 1e9 << " ns"
                      << "  dt=" << thisDt * 1e9 << " ns"
                      << "  particles=" << particles.size()
                      << "  maxErad=" << maxErad
                      << "  totalErad=" << totalErad
                      << "  maxT=" << maxT_keV << " keV" << std::endl;
        }
    }

    std::vector<double> simX(Ncells);
    std::vector<double> simErad(Ncells);
    std::vector<double> simT(Ncells);
    std::vector<size_t> idx(Ncells);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return grid.GetMeshPoint(a).x < grid.GetMeshPoint(b).x;
    });

    for(size_t i = 0; i < Ncells; i++)
    {
        size_t k = idx[i];
        simX[i] = grid.GetMeshPoint(k).x;
        simErad[i] = extensives[k].Erad / grid.GetVolume(k);
        simT[i] = cells[k].temperature;
    }

    std::string profilePath = "moving_slab_profile.txt";
    {
        std::ofstream out(profilePath);
        out << "# x_cm  Erad_erg_per_cm3  T_kelvin\n";
        out << std::scientific << std::setprecision(12);
        for(size_t i = 0; i < Ncells; i++)
        {
            out << simX[i] << " " << simErad[i] << " " << simT[i] << "\n";
        }
        std::cout << "\nWrote " << profilePath << std::endl;
    }

    double observerErad = 0;
    double minDist = std::numeric_limits<double>::max();
    size_t observerIdx = 0;
    for(size_t i = 0; i < Ncells; i++)
    {
        double d = std::abs(simX[i] - zO);
        if(d < minDist)
        {
            minDist = d;
            observerIdx = i;
            observerErad = simErad[i];
        }
    }
    std::cout << "Observer at x=" << simX[observerIdx] << " cm: Erad=" << observerErad
              << " erg/cm^3" << std::endl;

    double slabEnergy = 0;
    double vacEnergy = 0;
    for(size_t i = 0; i < Ncells; i++)
    {
        size_t k = idx[i];
        if(densities[k] > 0.5 * rhoSlab)
        {
            slabEnergy += extensives[k].Erad;
        }
        else
        {
            vacEnergy += extensives[k].Erad;
        }
    }
    std::cout << "Erad in slab: " << slabEnergy << " erg"
              << "  Erad in vacuum: " << vacEnergy << " erg" << std::endl;

    std::cout << "\nNote: This is a gray, static-slab approximation. The original"
              << " benchmark uses 124-group\nfrequency-dependent transport with"
              << " Doppler shifts (v = 0.5994 cm/ns). The spectral\nshape at the"
              << " observer cannot be reproduced without multigroup transport.\n"
              << std::endl;

    std::cout << "Done." << std::endl;
    return 0;
}
