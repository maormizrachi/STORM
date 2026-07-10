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
#include <units/units.hpp>
#include "radiation/RadiationIMC.hpp"
#include "radiation/RadiationCell.hpp"
#include "population/CombPopulationControl.hpp"
#include "manager/MonteCarloManagerSerial.hpp"
#include "DensmoreOpacity.hpp"
#include "DensmoreBoundary.hpp"

/*
 * Densmore et al. (2012), Figure 4: heterogeneous step-opacity benchmark.
 *
 * 1D slab, x in [0, 3] cm.
 * Left BC:  Planck source at T = 1 keV;  particles escape.
 * Right BC: reflecting.
 * All y/z BCs: reflecting.
 *
 * Opacity: sigma(E) = sigma0 / (sqrt(kT) * E^3)
 *   sigma0 = 10  keV^{3.5}/cm   for x < 2 cm
 *   sigma0 = 1000 keV^{3.5}/cm  for x >= 2 cm
 *
 * EOS: Cv = 1e15 / kev_kelvin  erg/K/cm^3   (constant, linear EOS)
 * Initial T = 1 eV.   density = 1.
 * dt = 5e-12 s,  t_final = 1e-9 s  (200 steps).
 *
 * 30-group frequency-dependent transport with opacity-weighted Planck
 * emission sampling.
 *
 * Usage:
 *   ./densmore2012 [Nx] [new_per_cell] [boundary_per_cell]
 */

using Grid = MadCart::CartesianMesh3D<Vector3D>;

class DensmoreEOS
{
public:
    DensmoreEOS(double cvPerVolume, double density)
        : cvPerMass_(cvPerVolume / density)
    {}

    double dT2cv(double /*density*/, double /*temperature*/,
                 const std::vector<double> &, const std::vector<std::string> &) const
    {
        return cvPerMass_;
    }

    double de2T(double /*density*/, double specificEnergy,
                const std::vector<double> &, const std::vector<std::string> &) const
    {
        return (cvPerMass_ > 0.0) ? specificEnergy / cvPerMass_ : 0.0;
    }

private:
    double cvPerMass_;
};

struct RefPoint
{
    double x, T_keV;
};

static std::vector<RefPoint> LoadCSV(const std::string &path)
{
    std::vector<RefPoint> ref;
    std::ifstream in(path);
    if(!in.is_open())
    {
        return ref;
    }
    std::string line;
    while(std::getline(in, line))
    {
        if(line.empty() or line[0] == '#')
        {
            continue;
        }
        double x, T;
        char comma;
        std::istringstream iss(line);
        if(iss >> x >> comma >> T)
        {
            ref.push_back({x, T});
        }
    }
    return ref;
}

int main(int argc, char *argv[])
{
    constexpr size_t G = STORM::examples::N_DENSMORE_GROUPS;
    using IMC = STORM::RadiationIMC<Vector3D, Grid, STORM::RadiationCell, STORM::SimpleExtensives,
                                    DensmoreEOS, G>;

    size_t Nx = (argc >= 2) ? std::stoul(argv[1]) : 512;
    size_t newPhotonsPerCell = (argc >= 3) ? std::stoul(argv[2]) : 50;
    size_t boundaryPhotonsPerCell = (argc >= 4) ? std::stoul(argv[3]) : 100;

    double keV_K = units::kev_kelvin;
    double eV_K = keV_K / 1000.0;

    double domainLength = 3.0;
    double xStep = 2.0;
    double T_init = eV_K;
    double T_boundary = keV_K;
    double density = 1.0;
    double cvPerVolume = 1e15 / keV_K;
    double tf = 1e-9;
    double dt = 5e-12;
    size_t iterations = static_cast<size_t>(tf / dt);

    double Emin = units::kev * 1e-4;
    double Emax = units::kev * 1e2;
    std::array<double, G + 1> energyBoundaries{};
    energyBoundaries[0] = Emin;
    double ratio = std::pow(Emax / Emin, 1.0 / G);
    for(size_t g = 0; g < G; ++g)
    {
        energyBoundaries[g + 1] = energyBoundaries[g] * ratio;
    }

    double dy = domainLength / Nx;
    Vector3D lower(0, 0, 0);
    Vector3D upper(domainLength, dy, dy);

    Grid grid(lower, upper, Nx, 1, 1);
    size_t Ncells = grid.GetPointNo();

    std::cout << "Densmore 2012 heterogeneous step-opacity (" << G << "-group MC)" << std::endl;
    std::cout << "Nx=" << Ncells << ", domain=[0, " << domainLength << "] cm" << std::endl;
    std::cout << "new_per_cell=" << newPhotonsPerCell << ", boundary_per_cell=" << boundaryPhotonsPerCell << std::endl;
    std::cout << "dt=" << dt << " s, t_final=" << tf << " s, iterations=" << iterations << std::endl;

    std::vector<STORM::RadiationCell> cells(Ncells);
    std::vector<STORM::SimpleExtensives> extensives(Ncells);
    std::vector<int> regionFlags(Ncells, 0);

    for(size_t i = 0; i < Ncells; i++)
    {
        double x = grid.GetCellCM(i).x;
        double volume = grid.GetVolume(i);
        regionFlags[i] = (x < xStep) ? 1 : 0;
        cells[i].temperature = T_init;
        cells[i].internalEnergy = cvPerVolume * T_init * volume;
        extensives[i].mass = density * volume;
        extensives[i].internal_energy = cells[i].internalEnergy;
    }

    STORM::RadiationIMCParameters<G> imcParams;
    imcParams.newPhotonsPerCell = newPhotonsPerCell;
    imcParams.withRandomWalk = true;
    imcParams.withMultigroupOpacity = true;
    imcParams.energyBoundaries = energyBoundaries;
    imcParams.energyBoundariesProvided = true;

    std::shared_ptr<DensmoreEOS> eos = std::make_shared<DensmoreEOS>(cvPerVolume, density);
    std::shared_ptr<STORM::examples::DensmoreOpacity<Vector3D, Grid>> opacityModel =
        std::make_shared<STORM::examples::DensmoreOpacity<Vector3D, Grid>>(regionFlags, cells);
    opacityModel->setGroupBoundaries(energyBoundaries);
    std::shared_ptr<STORM::examples::DensmoreBoundary<Vector3D, Grid>> boundary =
        std::make_shared<STORM::examples::DensmoreBoundary<Vector3D, Grid>>(grid, T_boundary, boundaryPhotonsPerCell, energyBoundaries);
    std::shared_ptr<IMC> physics =
        std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacityModel, imcParams);
    std::shared_ptr<STORM::CombPopulationControl<Vector3D, Grid>> popControl =
        std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, 200, 5.0);

    STORM::MonteCarloManagerSerial<Vector3D, Grid> manager(grid, physics, popControl, boundary);
    std::vector<STORM::Particle<Vector3D, Grid>> particles;

    for(size_t step = 0; step < iterations; step++)
    {
        particles = manager.step(std::move(particles), dt);

        if(step % 20 == 0 or step + 1 == iterations)
        {
            double maxT = 0;
            for(size_t i = 0; i < Ncells; i++)
            {
                maxT = std::max(maxT, cells[i].temperature);
            }
            double fraction = double(step + 1) / iterations;
            int pct = static_cast<int>(fraction * 100);
            std::cout << "Step " << step + 1 << "/" << iterations
                      << " (" << pct << "%)"
                      << "  particles=" << particles.size()
                      << "  maxT=" << maxT / keV_K << " keV" << std::endl;
        }
    }

    std::vector<double> simX(Ncells), simT(Ncells);
    std::vector<size_t> idx(Ncells);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return grid.GetMeshPoint(a).x < grid.GetMeshPoint(b).x;
    });
    for(size_t i = 0; i < Ncells; i++)
    {
        size_t k = idx[i];
        simX[i] = grid.GetMeshPoint(k).x;
        simT[i] = cells[k].temperature;
    }

    std::string profilePath = "densmore2012_profile.txt";
    {
        std::ofstream out(profilePath);
        out << "# Densmore2012 gray MC  t=" << tf << "  Nx=" << Nx << "\n";
        out << "# x(cm)  T(K)\n";
        for(size_t i = 0; i < Ncells; i++)
        {
            out << simX[i] << " " << simT[i] << "\n";
        }
        std::cout << "\nWrote " << profilePath << std::endl;
    }

    std::string refPath = std::string(STORM_DATA_DIR) + "/data/densmore2012_fig4_mc.csv";
    std::vector<RefPoint> ref = LoadCSV(refPath);
    if(!ref.empty())
    {
        double l1sum = 0;
        size_t count = 0;
        for(const RefPoint &rp : ref)
        {
            size_t j = 0;
            while(j + 1 < Ncells and simX[j + 1] < rp.x)
            {
                j++;
            }
            if(j + 1 >= Ncells)
            {
                continue;
            }
            double frac = (rp.x - simX[j]) / (simX[j + 1] - simX[j]);
            double T_interp = simT[j] + frac * (simT[j + 1] - simT[j]);
            double T_keV = T_interp / keV_K;
            l1sum += std::abs(T_keV - rp.T_keV);
            count++;
        }
        double l1 = (count > 0) ? l1sum / count : -1;
        std::cout << "DENSMORE2012_TGAS_L1 = " << std::scientific << l1 << " keV" << std::endl;
        if(l1 >= 0 and l1 < 0.10)
        {
            std::cout << "PASS (L1 < 0.10 keV)" << std::endl;
        }
        else
        {
            std::cout << "WARN: gray approximation may differ from multigroup reference" << std::endl;
        }
    }
    else
    {
        std::cout << "No reference data at " << refPath << " — skipping comparison" << std::endl;
    }

    {
        std::string scriptDir = __FILE__;
        scriptDir = scriptDir.substr(0, scriptDir.rfind('/'));
        std::string cmd = "python3 " + scriptDir + "/plot_densmore.py";
        std::cout << "Running: " << cmd << std::endl;
        std::system(cmd.c_str());
    }

    std::cout << "\nDone." << std::endl;
    return 0;
}
