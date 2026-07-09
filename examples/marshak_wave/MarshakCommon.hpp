#ifndef STORM_MARSHAK_COMMON_HPP
#define STORM_MARSHAK_COMMON_HPP

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
#include "MarshakOpacity.hpp"
#include "MarshakBoundary.hpp"

namespace STORM {
namespace examples {

using MarshakGrid = MadCart::CartesianMesh3D<Vector3D>;

struct ProblemParams
{
    int number;
    double domainLength;
    double xOffset;

    double kappaP0, kappaR0, alpha, betaRho;

    double f_eos;
    double eosBeta;
    double eosMu;

    double T_bath_coeff;
    double T_bath_exponent;

    double tf;
    double initialDt;
};

inline ProblemParams GetProblemParams(int problem)
{
    ProblemParams p;
    p.number = problem;
    p.xOffset = 0;
    p.tf = 1e-9;
    p.initialDt = 1e-15;

    double keV_K = constants::kev_kelvin;

    switch(problem)
    {
        case 1:
            p.domainLength = 0.2;
            p.kappaP0 = 0.1;
            p.kappaR0 = 100.0;
            p.alpha = 3.0;
            p.betaRho = 0.0;
            p.f_eos = 6.860085e14 / std::pow(keV_K, 4.0);
            p.eosBeta = 4.0;
            p.eosMu = 0.0;
            p.T_bath_coeff = 1.008038;
            p.T_bath_exponent = 1.0 / 3.0;
            break;
        case 2:
            p.domainLength = 0.2;
            p.kappaP0 = 100.0;
            p.kappaR0 = 100.0;
            p.alpha = 3.0;
            p.betaRho = 0.0;
            p.f_eos = 6.860085e14 / std::pow(keV_K, 4.0);
            p.eosBeta = 4.0;
            p.eosMu = 0.0;
            p.T_bath_coeff = 1.014565;
            p.T_bath_exponent = 1.0 / 3.0;
            break;
        case 3:
            p.domainLength = 1.0;
            p.kappaP0 = 0.1;
            p.kappaR0 = 40.0;
            p.alpha = 1.5;
            p.betaRho = 1.2;
            p.f_eos = 1e14 / std::pow(keV_K, 3.4);
            p.eosBeta = 3.4;
            p.eosMu = 0.14;
            p.T_bath_coeff = 1.0470478;
            p.T_bath_exponent = 86.0 / 57.0;
            break;
        case 4:
            p.domainLength = 1.0;
            p.xOffset = 1e-5;
            p.kappaP0 = 0.001;
            p.kappaR0 = 2.0;
            p.alpha = 4.5;
            p.betaRho = 1.9;
            p.f_eos = 1e14 / std::pow(keV_K, 6.0);
            p.eosBeta = 6.0;
            p.eosMu = 0.3;
            p.T_bath_coeff = 1.01008116;
            p.T_bath_exponent = 14.0 / 139.0;
            p.initialDt = 1e-17;
            break;
        default:
            std::cerr << "Unknown problem number: " << problem << " (must be 1-4)" << std::endl;
            exit(1);
    }
    return p;
}

inline double ComputeDensity(int problem, double x)
{
    switch(problem)
    {
        case 1: case 2: return 1.0;
        case 3: return std::pow(std::max(x, 1e-30), 20.0 / 19.0);
        case 4: return std::pow(std::max(x, 1e-30), -40.0 / 139.0);
        default: return 1.0;
    }
}

inline double EOS_E_from_T(const ProblemParams &p, double T, double rho)
{
    return p.f_eos * std::pow(T, p.eosBeta) * std::pow(rho, 1.0 - p.eosMu);
}

inline double EOS_T_from_E(const ProblemParams &p, double E, double rho)
{
    double rhoFactor = std::pow(rho, 1.0 - p.eosMu);
    double ratio = std::max(E, 0.0) / (p.f_eos * rhoFactor);
    return std::pow(std::max(ratio, 0.0), 1.0 / p.eosBeta);
}

inline double EOS_cv(const ProblemParams &p, double T, double rho)
{
    return p.f_eos * p.eosBeta * std::pow(std::max(T, 1.0), p.eosBeta - 1.0) * std::pow(rho, 1.0 - p.eosMu);
}

inline double BathTemperature(const ProblemParams &p, double t)
{
    double t_ns = std::max(t, 1e-20) * 1e9;
    return p.T_bath_coeff * std::pow(t_ns, p.T_bath_exponent) * constants::kev_kelvin;
}

class MarshakEOS
{
public:
    MarshakEOS(const ProblemParams &params) : params_(params) {}

    double dT2cv(double density, double temperature,
                 const std::vector<double> &, const std::vector<std::string> &) const
    {
        double cvPerVolume = EOS_cv(params_, temperature, density);
        return cvPerVolume / density;
    }

    double de2T(double density, double specificEnergy,
                const std::vector<double> &, const std::vector<std::string> &) const
    {
        double energyPerVolume = specificEnergy * density;
        return EOS_T_from_E(params_, energyPerVolume, density);
    }

private:
    ProblemParams params_;
};

struct ReferencePoint
{
    double x, Tgas, Trad;
};

inline std::vector<ReferencePoint> LoadReference(const std::string &path)
{
    std::vector<ReferencePoint> ref;
    std::ifstream in(path);
    if(!in.is_open())
    {
        return ref;
    }
    double x, Tg, Tr;
    while(in >> x >> Tg >> Tr)
    {
        ref.push_back({x, Tg, Tr});
    }
    return ref;
}

inline double ComputeL1(const std::vector<double> &simX, const std::vector<double> &simT,
                         const std::vector<ReferencePoint> &ref, bool useGas)
{
    if(ref.empty() or simX.empty())
    {
        return -1.0;
    }
    double keV_K = constants::kev_kelvin;
    double l1sum = 0;
    size_t count = 0;
    size_t j = 0;
    for(size_t i = 0; i < ref.size(); i++)
    {
        double refX = ref[i].x;
        double refT = useGas ? ref[i].Tgas : ref[i].Trad;
        while(j + 1 < simX.size() and simX[j + 1] < refX)
        {
            j++;
        }
        if(j + 1 >= simX.size())
        {
            break;
        }
        double frac = (refX - simX[j]) / (simX[j + 1] - simX[j]);
        double simTinterp = simT[j] + frac * (simT[j + 1] - simT[j]);
        double refT_keV = refT / keV_K;
        double simT_keV = simTinterp / keV_K;
        if(refT_keV > 1e-4)
        {
            l1sum += std::abs(simT_keV - refT_keV) / refT_keV;
            count++;
        }
    }
    return count > 0 ? l1sum / count : -1.0;
}

inline int RunMarshakWave(int problem, int argc, char *argv[])
{
    using IMC = RadiationIMC<Vector3D, MarshakGrid, RadiationCell, SimpleExtensives, MarshakEOS, 1>;

    size_t Nx = (argc >= 2) ? std::stoul(argv[1]) : 256;
    size_t newPhotonsPerCell = (argc >= 3) ? std::stoul(argv[2]) : 15;
    size_t boundaryPhotonsPerCell = (argc >= 4) ? std::stoul(argv[3]) : 100;

    ProblemParams params = GetProblemParams(problem);

    double xMax = params.xOffset + params.domainLength;
    double dy = xMax / Nx;
    Vector3D lower(0, 0, 0);
    Vector3D upper(xMax, dy, dy);

    MarshakGrid grid(lower, upper, Nx, 1, 1);
    size_t Ncells = grid.GetPointNo();

    double keV_K = constants::kev_kelvin;
    double T_init = 1e-3 * keV_K;

    std::cout << "Marshak wave problem " << problem << ": " << Ncells << " cells, domain [0, " << xMax << "] cm" << std::endl;

    std::vector<RadiationCell> cells(Ncells);
    std::vector<SimpleExtensives> extensives(Ncells);
    std::vector<double> densities(Ncells);

    for(size_t i = 0; i < Ncells; i++)
    {
        double x = grid.GetCellCM(i).x;
        double volume = grid.GetVolume(i);
        double rho = ComputeDensity(problem, x);
        densities[i] = rho;
        cells[i].temperature = T_init;
        cells[i].internalEnergy = EOS_E_from_T(params, T_init, rho) * volume;
        extensives[i].mass = rho * volume;
        extensives[i].internal_energy = cells[i].internalEnergy;
    }

    double T_bath_init = BathTemperature(params, params.initialDt);

    RadiationIMCParameters<1> imcParams;
    imcParams.newPhotonsPerCell = newPhotonsPerCell;
    imcParams.withRandomWalk = true;
    imcParams.energyBoundaries = {0.0, 1e30};
    imcParams.energyBoundariesProvided = true;

    std::shared_ptr<MarshakEOS> eos = std::make_shared<MarshakEOS>(params);
    std::shared_ptr<MarshakOpacity<Vector3D, MarshakGrid>> opacityModel =
        std::make_shared<MarshakOpacity<Vector3D, MarshakGrid>>(params.kappaP0, params.kappaR0, params.alpha, params.betaRho, densities, cells);
    std::shared_ptr<MarshakBoundary<Vector3D, MarshakGrid>> boundary =
        std::make_shared<MarshakBoundary<Vector3D, MarshakGrid>>(grid, T_bath_init, boundaryPhotonsPerCell);
    std::shared_ptr<IMC> physics =
        std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacityModel, imcParams);
    std::shared_ptr<CombPopulationControl<Vector3D, MarshakGrid>> popControl =
        std::make_shared<CombPopulationControl<Vector3D, MarshakGrid>>(grid, 15, 6.0);

    MonteCarloManagerSerial<Vector3D, MarshakGrid> manager(grid, physics, popControl, boundary);
    std::vector<Particle<Vector3D, MarshakGrid>> particles;

    double dt = params.initialDt;
    double simTime = 0;
    size_t cycle = 0;

    std::cout << "T_bath(t_final) = " << BathTemperature(params, params.tf) / keV_K << " keV" << std::endl;
    std::cout << "new_per_cell=" << newPhotonsPerCell << ", boundary_per_cell=" << boundaryPhotonsPerCell << std::endl;
    std::cout << std::endl;

    while(simTime < params.tf)
    {
        double t_now = std::max(simTime, params.initialDt);
        double T_bath = BathTemperature(params, t_now);
        boundary->SetTemperature(T_bath);

        dt = std::min(dt, params.tf - simTime);
        particles = manager.step(std::move(particles), dt);

        simTime += dt;
        cycle++;

        double newDt = std::max(params.initialDt, simTime * 1e-3);
        dt = std::min(newDt, 5e-11);

        if(cycle % 50 == 0 or simTime >= params.tf)
        {
            double maxT_keV = 0;
            for(size_t i = 0; i < Ncells; i++)
            {
                maxT_keV = std::max(maxT_keV, cells[i].temperature / keV_K);
            }
            int pct = static_cast<int>(simTime / params.tf * 100);
            std::cout << "Cycle " << cycle << " (" << pct << "%)  t=" << simTime * 1e9 << "/" << params.tf * 1e9 << " ns  dt=" << dt
                      << "  particles=" << particles.size() << "  maxT=" << maxT_keV << " keV"
                      << "  T_bath=" << T_bath / keV_K << " keV" << std::endl;
        }
    }

    const std::vector<double> &EradTimeAvg = physics->getEradTimeAvg();

    std::vector<double> simX(Ncells), simT(Ncells), simTrad(Ncells);
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
        double Erad = std::max(EradTimeAvg[k], 0.0);
        simTrad[i] = std::pow(Erad / constants::arad, 0.25);
    }

    std::string profilePath = "marshak_wave_" + std::to_string(problem) + "_profile.txt";
    {
        std::ofstream out(profilePath);
        out << std::scientific << std::setprecision(12);
        for(size_t i = 0; i < Ncells; i++)
        {
            out << simX[i] << " " << simT[i] << " " << simTrad[i] << "\n";
        }
        std::cout << "\nWrote " << profilePath << std::endl;
    }

    std::string refPath = std::string(STORM_DATA_DIR) + "/reference.txt";
    std::vector<ReferencePoint> ref = LoadReference(refPath);
    if(!ref.empty())
    {
        double l1 = ComputeL1(simX, simT, ref, true);
        std::cout << "TGAS_REL_L1 = " << std::scientific << l1 << std::endl;
        if(l1 >= 0 and l1 < 0.10)
        {
            std::cout << "PASS (L1 < 0.10)" << std::endl;
        }
        else if(l1 >= 0)
        {
            std::cout << "WARN: L1 = " << l1 << " >= 0.10 (coarse MC resolution may differ from diffusion)" << std::endl;
        }
    }
    else
    {
        std::cout << "No reference data found at " << refPath << " - skipping comparison" << std::endl;
    }

    {
        std::string scriptDir = __FILE__;
        scriptDir = scriptDir.substr(0, scriptDir.rfind('/'));
        std::string cmd = "python3 " + scriptDir + "/plot_marshak.py " + std::to_string(problem);
        std::cout << "Running: " << cmd << std::endl;
        std::system(cmd.c_str());
    }

    std::cout << "\nDone." << std::endl;
    return 0;
}

} // namespace examples
} // namespace STORM

#endif // STORM_MARSHAK_COMMON_HPP
