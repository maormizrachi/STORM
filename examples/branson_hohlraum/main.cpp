// Branson's 3D hohlraum performance problem (lanl/branson inputs/3D_hohlraum_multi_node.xml)
// run in STORM, so the two codes can be timed on identical physics and an identical mesh.
//
// Geometry, materials, timestep and packet budget come from BransonDeck.hpp, which is
// generated directly from the XML by generate_deck_header.py -- nothing is transcribed.
//
// Both codes: gray IMC, Fleck-Cummings, sigma_a = opacA + opacB T^opacC, no scattering,
// no hydro, no population control, REFLECT on x=0 and y=0, VACUUM on the other four faces.
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef STORM_WITH_MPI
#include <mpi.h>
#include "manager/MonteCarloManagerFactory.hpp"
#else
#include "manager/MonteCarloManager.hpp"
#endif

#include <units/units.hpp>

#include "examples/Vector3D.hpp"
#include "MadCart/CartesianMesh3D.hpp"
#include "population/NoPopulationControl.hpp"
#include "radiation/RadiationIMC.hpp"

#include "BransonDeck.hpp"
#include "BransonHohlraumPhysics.hpp"

using Grid = MadCart::CartesianMesh3D<Vector3D>;
using Cell = STORM::RadiationCell;
using Opacity = BransonHohlraum::Opacity<Vector3D, Grid>;
using Boundary = BransonHohlraum::Boundary<Vector3D, Grid>;
using IMC = STORM::RadiationIMC<Vector3D, Grid, Cell, STORM::SimpleExtensives,
                                BransonHohlraum::EOS, 1, Opacity>;

namespace
{

//! Which division of an axis a coordinate falls in (divisions are contiguous).
template<typename DivArray>
std::size_t divisionIndex(const DivArray &divisions, double coordinate)
{
    for(std::size_t d = 0; d + 1 < divisions.size(); ++d)
    {
        if(coordinate < divisions[d].hi)
        {
            return d;
        }
    }
    return divisions.size() - 1;
}

bool envFlag(const char *name)
{
    const char *v = std::getenv(name);
    return v && std::string(v) != "0" && std::string(v) != "false";
}

} // namespace

int main(int argc, char **argv)
{
    int rank = 0, nRanks = 1;
#ifdef STORM_WITH_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#endif
    int result = 0;
    try
    {
        namespace D = BransonDeck;
        constexpr std::size_t nCells = D::nx * D::ny * D::nz;

        // argv: new packets per cell (default = deck photon budget / cells), steps.
        // Branson spreads its budget over cells by emission energy with a floor of one
        // packet per cell; STORM does the same through emissionFloorPhotonsPerCell.
        const std::size_t newPerCell =
            argc > 1 ? std::stoul(argv[1])
                     : static_cast<std::size_t>(D::photons) / nCells;
        const std::size_t steps =
            argc > 2 ? std::stoul(argv[2])
                     : static_cast<std::size_t>(std::llround((D::tStopShakes - D::tStartShakes) / D::dtShakes));

        // One EOS constant serves the mesh only if every region shares CV and density.
        for(const D::Region &r : D::regions)
        {
            if(r.CV != D::regions[0].CV || r.density != D::regions[0].density)
            {
                throw std::runtime_error("Deck has per-region CV/density; EOS assumes one value");
            }
        }
        const double density = D::regions[0].density;
        const double specificHeat = BransonHohlraum::specificHeatCgs(D::regions[0].CV);
        const double dt = D::dtShakes * BransonHohlraum::shake;

        Grid grid(Vector3D(D::xLo, D::yLo, D::zLo), Vector3D(D::xHi, D::yHi, D::zHi),
                  D::nx, D::ny, D::nz);
#ifdef STORM_WITH_MPI
        // Branson feeds METIS per-cell vertex weights of max(1, (ln sigma_a)^3)
        // (decompose_mesh.h:135), so STORM partitions on the same information --
        // otherwise the comparison would be against a partitioner the other code
        // does not have.  BH_WEIGHTS=work uses the real cost proxy instead
        // (packets x events ~ packets / fleck), and BH_WEIGHTS=uniform disables both.
        const std::string weightMode = std::getenv("BH_WEIGHTS") ? std::getenv("BH_WEIGHTS") : "branson";
        std::vector<double> weights(nCells, 1.0);
        if(weightMode != "uniform")
        {
            const double cvVol = density * specificHeat;
            for(std::size_t ix = 0; ix < D::nx; ++ix)
            {
                for(std::size_t iy = 0; iy < D::ny; ++iy)
                {
                    for(std::size_t iz = 0; iz < D::nz; ++iz)
                    {
                        // Index order must match CartesianMesh3D::BuildParallel.
                        const std::size_t idx = (ix * D::ny + iy) * D::nz + iz;
                        const Vector3D c(D::xLo + (ix + 0.5) * D::cellWidth,
                                         D::yLo + (iy + 0.5) * D::cellWidth,
                                         D::zLo + (iz + 0.5) * D::cellWidth);
                        const D::Region &reg = D::regions[D::regionIndexAt(
                            divisionIndex(D::xDiv, c.x), divisionIndex(D::yDiv, c.y),
                            divisionIndex(D::zDiv, c.z))];
                        const double sigma = reg.opacA;
                        if(weightMode == "work")
                        {
                            // packets(cell) x events(packet); events ~ ln(1/cutoff)/fleck.
                            const double T = reg.T_e * units::kev_kelvin;
                            const double fleck = 1.0 / (1.0 + 4.0 * units::arad * T * T * T
                                                                  * sigma * units::clight * dt / cvVol);
                            weights[idx] = std::max(1.0, (1.0 / fleck) * std::pow(sigma, 0.0));
                        }
                        else
                        {
                            weights[idx] = std::max(1.0, double(int(std::pow(std::log(sigma), 3))));
                        }
                    }
                }
            }
        }
        grid.BuildParallel(weights);
#endif
        const std::size_t n = grid.GetPointNo();

        std::vector<int> regionOfCell(n);
        std::vector<Cell> cells(n);
        std::vector<STORM::SimpleExtensives> ext(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            const Vector3D c = grid.GetCellCM(i);
            const int r = D::regionIndexAt(divisionIndex(D::xDiv, c.x),
                                           divisionIndex(D::yDiv, c.y),
                                           divisionIndex(D::zDiv, c.z));
            regionOfCell[i] = r;
            cells[i].temperature = D::regions[r].T_e * units::kev_kelvin;
            ext[i].mass = density * grid.GetVolume(i);
            ext[i].internal_energy = ext[i].mass * specificHeat * cells[i].temperature;
            cells[i].internalEnergy = ext[i].internal_energy;
        }

        std::vector<double> opacA, opacB, opacC, opacS;
        for(const D::Region &r : D::regions)
        {
            opacA.push_back(r.opacA);
            opacB.push_back(r.opacB);
            opacC.push_back(r.opacC);
            opacS.push_back(r.opacS);
        }

        STORM::RadiationIMCParameters<1> p;
        p.newPhotonsPerCell = newPerCell;
        p.emissionFloorPhotonsPerCell = 1; // Branson's per-cell floor
        p.withHydro = false;
        p.withCompton = false;
        p.withSlabTransport = false;
        p.withMultigroupOpacity = false;
        p.withEgTimeAvg = false;
        p.energyBoundaries = {0., 100 * units::kev};
        p.energyBoundariesProvided = true;
        // Branson has neither, so both default off; switch on to price the acceleration.
        p.withDDMC = envFlag("BH_DDMC");
        p.ddmcMinCellOpticalDepth = std::getenv("BH_DDMC_MIN_TAU") ? std::stod(std::getenv("BH_DDMC_MIN_TAU")) : 3.0;
        p.withRandomWalk = envFlag("BH_RW");
        // Branson kills a packet at 1% of its creation weight (Constants::cutoff_fraction);
        // STORM's default is 0.1%, which costs ln(1000)/ln(100) = 1.5x more events per
        // packet.  Match Branson by default here so the timing compares like with like.
        p.weightCutoffFraction = std::getenv("BH_CUTOFF") ? std::stod(std::getenv("BH_CUTOFF")) : 1e-2;

        std::shared_ptr<BransonHohlraum::EOS> eos = std::make_shared<BransonHohlraum::EOS>(specificHeat);
        std::shared_ptr<Opacity> op = std::make_shared<Opacity>(regionOfCell, cells, opacA, opacB, opacC, opacS);
        std::shared_ptr<Boundary> bc = std::make_shared<Boundary>(grid);
        std::shared_ptr<IMC> physics = std::make_shared<IMC>(grid, bc, cells, ext, eos, op, p);
        // Branson never calls comb_photons(), so STORM must not comb either.
        std::shared_ptr<STORM::NoPopulationControl<Vector3D, Grid>> pop =
            std::make_shared<STORM::NoPopulationControl<Vector3D, Grid>>(grid);

#ifdef STORM_WITH_MPI
        const std::string managerName = std::getenv("BH_MANAGER") ? std::getenv("BH_MANAGER") : "p2p";
        const STORM::ManagerType managerType =
            managerName == "rdma" ? STORM::ManagerType::RDMA : STORM::ManagerType::P2P;
        STORM::MonteCarloManager<Vector3D, Grid, IMC> manager =
            STORM::CreateMonteCarloManager<Vector3D, Grid>(grid, physics, pop, bc, managerType);
#else
        STORM::MonteCarloManager<Vector3D, Grid, IMC> manager(grid, physics, pop, bc);
#endif

        if(rank == 0)
        {
            std::cout << "Branson 3D hohlraum in STORM: " << D::nx << 'x' << D::ny << 'x' << D::nz
                      << " = " << nCells << " cells on " << nRanks << " ranks\n"
                      << "dt=" << D::dtShakes << " sh (" << dt << " s), steps=" << steps
                      << ", new packets/cell=" << newPerCell
                      << " (budget " << newPerCell * nCells << ")"
                      << ", ddmc=" << p.withDDMC << ", rw=" << p.withRandomWalk
                      << ", weight_cutoff=" << p.weightCutoffFraction
#ifdef STORM_WITH_MPI
                      << ", manager=" << managerName << ", weights=" << weightMode
#endif
                      << '\n';
            for(std::size_t r = 0; r < D::regions.size(); ++r)
            {
                const double T = D::regions[r].T_e;
                const double cvVol = density * specificHeat;
                const double f = 1.0 / (1.0 + 4.0 * units::arad
                                                  * std::pow(T * units::kev_kelvin, 3)
                                                  * D::regions[r].opacA * units::clight * dt / cvVol);
                std::cout << "  region " << D::regions[r].id << ": sigma_a=" << D::regions[r].opacA
                          << " /cm, T=" << T << " keV, fleck=" << std::setprecision(6) << f
                          << " (expect ~" << std::llround(std::log(100.0) / std::max(f, 1e-30))
                          << " events/packet)\n";
            }
            std::cout << std::flush;
        }

        for(std::size_t step = 0; step < steps; ++step)
        {
            manager.step(dt);
            if(rank == 0)
            {
                std::cout << "step " << step + 1 << " / " << steps << " done" << std::endl;
            }
        }

        if(rank == 0)
        {
            std::cout << "Done. ddmc_steps=" << physics->getDDMCStepCount()
                      << " ddmc_leaks=" << physics->getDDMCLeakCount()
                      << " rw_steps=" << physics->getRandomWalkStepCount() << std::endl;
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
