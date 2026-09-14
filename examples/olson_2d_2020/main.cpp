// Olson (2020), DOI 10.1080/23324309.2020.1800745, complex 2D test.
// Parameters and reference curves also in Steinberg & Heizler (2023), Sec IV.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
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
#include "OlsonPhysics.hpp"
#include "OlsonSource.hpp"
using Grid = MadCart::CartesianMesh3D<Vector3D>;
using Opacity = Olson::Opacity<Vector3D, Grid>;
using IMC = STORM::RadiationIMC<Vector3D, Grid, Olson::Cell, STORM::SimpleExtensives, Olson::EOS, 1, Opacity>;
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
        // argv: Nxy, thermal packets/cell/step, source sampling density,
        //       dt(seconds), output_dir, census target/cell.
        const size_t nx = argc > 1 ? std::stoul(argv[1]) : 380;
        const size_t np = argc > 2 ? std::stoul(argv[2]) : 4;
        const size_t ns = argc > 3 ? std::stoul(argv[3]) : 512;
        const double dt = argc > 4 ? std::stod(argv[4]) : 1e-13;
        const std::string out = argc > 5 ? argv[5] : "output";
        const size_t census = argc > 6 ? std::stoul(argv[6]) : 80;
        // Multiples of 38 put all 0.5,1.5,2.5 cm material faces on mesh faces.
        if(nx % 38 || nx < 38 || !np || !ns || !census || !(dt > 0 && dt <= 1e-12))
        {
            throw std::runtime_error("Invalid arguments; Nxy must be a multiple of 38");
        }
        std::filesystem::create_directories(out);
        // Extrude by 1 cm with reflecting z faces: spatially 2D, full sphere
        // of transport directions (NOT directions confined to the xy plane).
        const double dx = 3.8 / nx, scale = units::arad * std::pow(units::kev_kelvin, 4);
        Grid grid(Vector3D(0, 0, 0), Vector3D(3.8, 3.8, 1), nx, nx, 1);
#ifdef STORM_WITH_MPI
        // A uniform cell partition puts most source packets on one rank.
        // Static cost weights distribute the source region across ranks;
        // these weights change ownership only, never physical cell volumes.
        std::vector<double> costs(nx * nx);
        for(size_t ix = 0; ix < nx; ++ix)
        {
            for(size_t iy = 0; iy < nx; ++iy)
            {
                double r = std::hypot((ix + .5) * dx, (iy + .5) * dx);
                costs[ix * nx + iy] = 1 + 20 * std::exp(-18.7 * r * r * r) + std::exp(-r * r / 2);
            }
        }
        grid.BuildParallel(costs);
#endif
        size_t n = grid.GetPointNo();
        std::vector<Olson::Cell> cells(n);
        std::vector<STORM::SimpleExtensives> ext(n);
        double initialLocal = 0;
        for(size_t i = 0; i < n; ++i)
        {
            Vector3D pos = grid.GetCellCM(i);
            bool al = Olson::IsAluminum(pos.x, pos.y);
            cells[i].tracers[0] = al ? 1 : 0;
            cells[i].temperature = .01 * units::kev_kelvin;
            cells[i].internalEnergy = Olson::Energy(.01, al) * grid.GetVolume(i);
            cells[i].Erad = scale * 1e-8 / Olson::rho;
            ext[i].mass = Olson::rho * grid.GetVolume(i);
            ext[i].internal_energy = cells[i].internalEnergy;
            ext[i].Erad = scale * 1e-8 * grid.GetVolume(i);
            initialLocal += ext[i].internal_energy + ext[i].Erad;
        }
        STORM::RadiationIMCParameters<1> p;
        p.newPhotonsPerCell = np;
        // This flag is essential even with one bookkeeping group: frequencies
        // remain continuous and absorption is evaluated at each packet energy.
        p.withMultigroupOpacity = true;
        p.withRandomWalk = false;
        p.withHydro = false;
        p.withCompton = false;
        p.withDDMC = false;
        p.withSlabTransport = false;
        p.withEgTimeAvg = false;
        p.energyBoundaries = {1e-7 * units::kev, 100 * units::kev};
        p.energyBoundariesProvided = true;
        std::shared_ptr<Olson::EOS> eos = std::make_shared<Olson::EOS>();
        std::shared_ptr<Opacity> op = std::make_shared<Opacity>();
        std::shared_ptr<OlsonSource<Vector3D, Grid>> bc = std::make_shared<OlsonSource<Vector3D, Grid>>(grid, ns, dx, rank);
        std::shared_ptr<IMC> physics = std::make_shared<IMC>(grid, bc, cells, ext, eos, op, p);
        std::shared_ptr<STORM::CombPopulationControl<Vector3D, Grid>> pop =
            std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, census, 5.);
#ifdef STORM_WITH_MPI
        STORM::MonteCarloManager<Vector3D, Grid, IMC> manager =
            STORM::CreateMonteCarloManager<Vector3D, Grid>(grid, physics, pop, bc, STORM::ManagerType::P2P);
#else
        STORM::MonteCarloManager<Vector3D, Grid, IMC> manager(grid, physics, pop, bc);
#endif
        // The initial radiation is a 0.01 keV Planck spectrum, not the
        // opacity-weighted spectrum used for thermal material emission.
        manager.getParticles() = physics->generateInitialParticles(4);
        for(STORM::Particle<Vector3D> &packet : manager.getParticles())
        {
            packet.frequency = bc->BlackbodyEnergy(.01);
        }
        if(rank == 0)
        {
            std::cout << "Olson 2020: rho=0.001 Nxy=" << nx << " dt=" << dt
                      << " thermal_parameter=" << np << " source_density=" << ns << " census=" << census
                      << " continuous-frequency IMC; reflecting boundaries; volume source\n"
                      << p << std::flush;
        }
        double ct = 0;
        size_t step = 0;
        for(double target : {2., 2.5, 3.})
        {
            while(ct < target - 1e-12)
            {
                double h = std::min(dt, (target - ct) / units::clight);
                manager.step(h);
                ct += h * units::clight;
                if(rank == 0 && ++step % 50 == 0)
                {
                    std::cout << "ct=" << ct << std::endl;
                }
            }
            std::vector<double> rad(n, 0.), local(nx * nx * 4, 0.), global(local.size());
            for(const STORM::Particle<Vector3D> &p : manager.getParticles())
            {
                rad.at(p.cellIndex) += p.weight;
            }
            double totalLocal = 0;
            for(size_t i = 0; i < n; ++i)
            {
                Vector3D pos = grid.GetCellCM(i);
                size_t ix = std::min(nx - 1, size_t(pos.x / dx)), iy = std::min(nx - 1, size_t(pos.y / dx)), j = 4 * (ix * nx + iy);
                local[j] = cells[i].temperature / units::kev_kelvin;
                local[j + 1] = rad[i] / grid.GetVolume(i) / scale;
                local[j + 2] = cells[i].tracers[0];
                local[j + 3] = 1;
                totalLocal += ext[i].internal_energy + rad[i];
            }
#ifdef STORM_WITH_MPI
            MPI_Reduce(local.data(), global.data(), int(global.size()), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
            global = local;
#endif
            double budgetLocal[3] = {initialLocal, bc->GetInjectedEnergy(), totalLocal}, budget[3] = {};
#ifdef STORM_WITH_MPI
            MPI_Reduce(budgetLocal, budget, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
            std::copy(budgetLocal, budgetLocal + 3, budget);
#endif
            if(rank == 0)
            {
                double error = (budget[2] - budget[0] - budget[1]) / (budget[0] + budget[1]);
                std::ofstream balance(out + "/energy_budget.txt", std::ios::app);
                balance << std::setprecision(16) << target << ' ' << budget[0] << ' ' << budget[1] << ' ' << budget[2] << ' ' << error << '\n';
                std::cout << "Energy budget relative residual=" << error << std::endl;
                std::string tag = target == 2 ? "2" : target == 2.5 ? "2p5"
                                                                    : "3";
                std::ofstream field(out + "/field_ct" + tag + ".txt"), profile(out + "/profile_ct" + tag + ".txt");
                field << std::setprecision(16) << "# ct=" << target << " Nx=" << nx << "\n# x_cm y_cm T_keV E_over_aTkeV4 aluminum\n";
                profile << std::setprecision(16) << "# ct=" << target << " Nx=" << nx << "\n# r_cm T_keV E_over_aTkeV4 aluminum\n";
                for(size_t ix = 0; ix < nx; ++ix)
                {
                    for(size_t iy = 0; iy < nx; ++iy)
                    {
                        size_t j = 4 * (ix * nx + iy);
                        if(global[j + 3] != 1 || !std::isfinite(global[j]) || !std::isfinite(global[j + 1]) || global[j] <= 0 || global[j + 1] < 0)
                        {
                            throw std::runtime_error("Invalid profile/ownership");
                        }
                        field << (ix + .5) * dx << ' ' << (iy + .5) * dx << ' ' << global[j] << ' ' << global[j + 1] << ' ' << global[j + 2] << '\n';
                        if(ix == iy)
                        {
                            profile << std::sqrt(2.) * (ix + .5) * dx << ' ' << global[j] << ' ' << global[j + 1] << ' ' << global[j + 2] << '\n';
                        }
                    }
                }
                std::cout << "Wrote field and diagonal profile at ct=" << target << std::endl;
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
