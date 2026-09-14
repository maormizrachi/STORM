#include <cstdlib>
// Su & Olson, JQSRT 56, 337 (1996), DOI 10.1016/0022-4073(96)84524-9.
// Full gray IMC transport compared with the HALF-SPACE DIFFUSION solution.
// No volume source, no scattering, no imposed material surface temperature.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#ifdef STORM_WITH_MPI
#include <mpi.h>
#include "manager/MonteCarloManagerFactory.hpp"
#else
#include "manager/MonteCarloManager.hpp"
#endif
#include "examples/Vector3D.hpp"
#include "MadCart/CartesianMesh3D.hpp"
#include "radiation/RadiationIMC.hpp"
#include "population/CombPopulationControl.hpp"
#include "../marshak_wave/MarshakBoundary.hpp"
#include "SuOlsonPhysics.hpp"
using Grid = MadCart::CartesianMesh3D<Vector3D>;
class SuOlsonBoundary : public STORM::examples::MarshakBoundary<Vector3D, Grid>
{
  public:
    using STORM::examples::MarshakBoundary<Vector3D, Grid>::MarshakBoundary;
    STORM::DeviceBoundaryFaceBehavior getDeviceBoundaryFaceBehavior(size_t face, size_t, size_t) const override
    {
        return std::abs(this->grid.FaceCM(face).x) < 1e-12
                   ? STORM::DeviceBoundaryFaceBehavior::HostOnly
                   : STORM::DeviceBoundaryFaceBehavior::ReflectingRigid;
    }
    STORM::DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(size_t face, size_t, size_t) const override
    {
        return std::abs(this->grid.FaceCM(face).x) < 1e-12
                   ? STORM::DDMCBoundaryFaceBehavior::ThermalSource
                   : STORM::DDMCBoundaryFaceBehavior::ReflectingRigid;
    }
    bool isEscape(STORM::ParticleStatus s) const override
    {
        return s == STORM::ParticleStatus::REMOVE;
    }
};
using Cell = STORM::RadiationCell;
using Opacity = SuOlsonOpacity<Vector3D, Grid>;
using IMC = STORM::RadiationIMC<Vector3D, Grid, Cell, STORM::SimpleExtensives, SuOlsonEOS, 1, Opacity>;

int main(int argc, char **argv)
{
    int rank = 0;
#ifdef STORM_WITH_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    int result = 0;
    try
    {
        // argv: Nx, new packets/cell/step, boundary packets/step, delta_tau, output_dir.
        const size_t nx = argc > 1 ? std::stoul(argv[1]) : 512;
        const size_t np = argc > 2 ? std::stoul(argv[2]) : 8;
        const size_t nb = argc > 3 ? std::stoul(argv[3]) : 10000;
        const double dtau = argc > 4 ? std::stod(argv[4]) : 0.05;
        const std::string out = argc > 5 ? argv[5] : "output";
        if(nx < 2 || !np || !nb || !(dtau > 0 && dtau <= 1000))
        {
            throw std::runtime_error("Invalid arguments");
        }
        std::filesystem::create_directories(out);
        // Xmax=40 makes the far wall negligible through tau=100. Transverse
        // reflecting walls and slab transport retain the full 3D angular law.
        const double length = 40 / std::sqrt(3.), Tb = units::kev_kelvin;
        Grid grid(Vector3D(0, 0, 0), Vector3D(length, 1, 1), nx, 1, 1);
#ifdef STORM_WITH_MPI
        grid.BuildParallel(std::vector<double>(nx, 1.));
#endif
        const size_t n = grid.GetPointNo();
        std::vector<Cell> cells(n);
        std::vector<STORM::SimpleExtensives> ext(n);
        for(size_t i = 0; i < n; ++i)
        {
            // Positive 1e-6 Tb regularization gives v0=1e-24. No initial photons.
            cells[i].temperature = 1e-6 * Tb;
            cells[i].internalEnergy = units::arad * std::pow(cells[i].temperature, 4) * grid.GetVolume(i);
            ext[i].mass = grid.GetVolume(i); // rho=1 g/cm^3
            ext[i].internal_energy = cells[i].internalEnergy;
        }
        STORM::RadiationIMCParameters<1> p;
        p.newPhotonsPerCell = np;
        p.withSlabTransport = true;
        p.withHydro = false;
        p.withCompton = false;
        p.withDDMC = false;
        p.withRandomWalk = false;
        p.withMultigroupOpacity = false;
        p.withEgTimeAvg = false;
        p.energyBoundaries = {0., 100 * units::kev};
        p.energyBoundariesProvided = true;
        std::shared_ptr<SuOlsonEOS> eos = std::make_shared<SuOlsonEOS>();
        std::shared_ptr<Opacity> op = std::make_shared<Opacity>();
        std::shared_ptr<SuOlsonBoundary> bc = std::make_shared<SuOlsonBoundary>(grid, Tb, nb);
        // MarshakBoundary emits ac Tb^4/4 with p(mu)=2mu, absorbs outgoing
        // left-face rays, and reflects the distant right and transverse faces.
        std::shared_ptr<IMC> physics = std::make_shared<IMC>(grid, bc, cells, ext, eos, op, p);
        std::shared_ptr<STORM::CombPopulationControl<Vector3D, Grid>> pop =
            std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, 400, 5.);
#ifdef STORM_WITH_MPI
        STORM::MonteCarloManager<Vector3D, Grid, IMC> manager =
            STORM::CreateMonteCarloManager<Vector3D, Grid>(grid, physics, pop, bc, STORM::ManagerType::P2P);
#else
        STORM::MonteCarloManager<Vector3D, Grid, IMC> manager(grid, physics, pop, bc);
#endif
        if(rank == 0)
        {
            std::cout << "Su-Olson epsilon=1 sigma=1/cm Xmax=40 Nx=" << nx << " delta_tau=" << dtau << " slab=" << p.withSlabTransport << " ddmc=" << p.withDDMC << " (min tau " << p.ddmcMinCellOpticalDepth << ") rw=" << p.withRandomWalk
                      << " thermal_parameter=" << np << " boundary_packets=" << nb << '\n'
                      << p << std::flush;
        }
        double tau = 0;
        size_t step = 0;
        for(double target : {1., 10., 100.})
        {
            while(tau < target - 1e-12)
            {
                double h = std::min(dtau, target - tau);
                manager.step(h / units::clight);
                tau += h;
                if(rank == 0 && ++step % 100 == 0)
                {
                    std::cout << "tau=" << tau << std::endl;
                }
            }
            // Census packet sums: instantaneous radiation energy, not a step average.
            std::vector<double> local(nx * 3, 0.), global(nx * 3, 0.), rad(n, 0.);
            for(const STORM::Particle<Vector3D> &packet : manager.getParticles())
            {
                rad.at(packet.cellIndex) += packet.weight;
            }
            for(size_t i = 0; i < n; ++i)
            {
                const size_t j = std::min(nx - 1, size_t(grid.GetCellCM(i).x / length * nx));
                local[3 * j] = rad[i] / grid.GetVolume(i) / (units::arad * std::pow(Tb, 4));
                local[3 * j + 1] = std::pow(cells[i].temperature / Tb, 4);
                local[3 * j + 2] = 1;
            }
#ifdef STORM_WITH_MPI
            MPI_Reduce(local.data(), global.data(), int(global.size()), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
            global = local;
#endif
            if(rank == 0)
            {
                std::ofstream f(out + "/profile_tau" + std::to_string(int(target)) + ".txt");
                f << std::setprecision(16) << "# tau=" << target << " epsilon=1 Xmax=40 Nx=" << nx << "\n# X u_radiation v_material T_material_over_Tbath\n";
                for(size_t j = 0; j < nx; ++j)
                {
                    if(global[3 * j + 2] != 1 || !std::isfinite(global[3 * j]) || !std::isfinite(global[3 * j + 1]))
                    {
                        throw std::runtime_error("Invalid profile/ownership");
                    }
                    f << 40 * (j + 0.5) / nx << ' ' << global[3 * j] << ' ' << global[3 * j + 1] << ' ' << std::pow(global[3 * j + 1], .25) << '\n';
                }
                std::cout << "Wrote profile at tau=" << target << " ddmc_steps=" << physics->getDDMCStepCount()
                          << " ddmc_leaks=" << physics->getDDMCLeakCount() << " rw_steps=" << physics->getRandomWalkStepCount() << std::endl;
            }
        }
    }
    catch(const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        result = 1;
#ifdef STORM_WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    }
#ifdef STORM_WITH_MPI
    MPI_Finalize();
#endif
    return result;
}
