/*
 * Moving slab benchmark -- 124-group IMC with DDMC
 *
 * McClarren & Gentile (2021) moving radiating slab benchmark.
 *
 * A uniform slab of material (rho = 0.1 g/cm^3, T = 1 keV, L = 0.4 cm)
 * moves at v = 0.5994 cm/ns and radiates into vacuum.  124-group
 * frequency-dependent transport with Doppler shifts, DDMC acceleration.
 *
 * Usage:
 *   mpirun -np N ./moving_slab [newPhotonsPerCell]
 */

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <algorithm>
#include <mpi.h>
#include "examples/Vector3D.hpp"
#include "MadVoro/Voronoi3D.hpp"
#include <units/units.hpp>
#include <planck_integral/planck_integral.hpp>
#include <mpi_utils/mpi_collectives.hpp>

#include "radiation/RadiationIMC.hpp"
#include "radiation/RadiationIMCParameters.hpp"
#include "population/NoPopulationControl.hpp"
#include "manager/MonteCarloManagerFactory.hpp"
#include "utils/MpiExchangeGrid.hpp"
#include "examples/MPI_ParticleDtype.hpp"
#include "mesh_movement/VoronoiMeshMovement.hpp"
#include "MovingSlabOpacity.hpp"
#include "MovingSlabBoundary.hpp"

// ============================================================
// Cell and extensives types with multigroup + velocity support
// ============================================================

constexpr size_t G = 124;

struct MovingSlabCell
{
    size_t ID = 0;
    double temperature = 0;
    double internalEnergy = 0;
    double Erad = 0;
    double density = 0;
    Vector3D velocity;
    std::array<double, G> Eg{};
};

struct MovingSlabExtensives
{
    double internal_energy = 0;
    double mass = 0;
    double Erad = 0;
    std::array<double, G> Eg{};
};

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
        double cv = dT2cv(density, 0.0, tracers, tracerNames);
        return (cv > 0.0) ? specificEnergy / cv : 0.0;
    }

private:
    double cvPerMassSlab_;
    double cvPerMassVac_;
    double rhoSlab_;
};

// ============================================================
// Mesh point generation (matches RICH's layout)
// ============================================================

using Grid = MadVoro::Voronoi3D<Vector3D>;

static constexpr size_t N_SLAB_PTS = 20;
static constexpr size_t N_VAC_PTS = 60;
static constexpr size_t N_X_PTS = N_SLAB_PTS + N_VAC_PTS;
static constexpr size_t NYZ = 3;
static constexpr size_t N_TOTAL_PTS = N_X_PTS * NYZ * NYZ;

static std::vector<Vector3D> BuildAllPoints(double t, double vSlab, double L_slab, double xSym, double cellHalfYZ)
{
    double slabPtsX[N_SLAB_PTS];
    for(size_t i = 0; i < N_SLAB_PTS; ++i)
    {
        slabPtsX[i] = L_slab * (static_cast<double>(i) + 0.5) / N_SLAB_PTS;
    }

    std::array<double, NYZ> yzCenters;
    for(size_t k = 0; k < NYZ; ++k)
    {
        yzCenters[k] = -cellHalfYZ + cellHalfYZ * (2.0 * k + 1.0) / NYZ;
    }

    std::vector<Vector3D> xPts(N_X_PTS);
    double shift = vSlab * t;
    for(size_t i = 0; i < N_SLAB_PTS; ++i)
    {
        xPts[i] = Vector3D(slabPtsX[i] + shift, 0, 0);
    }
    double vacStart = L_slab + shift;
    size_t nLeft = N_VAC_PTS - 2;
    double h = (xSym - vacStart) / nLeft;
    double d = h / 2.0;
    for(size_t i = 0; i < nLeft; ++i)
    {
        double x = vacStart + h * (static_cast<double>(i) + 0.5);
        xPts[N_SLAB_PTS + i] = Vector3D(x, 0, 0);
    }
    xPts[N_SLAB_PTS + nLeft]     = Vector3D(xSym, 0, 0);
    xPts[N_SLAB_PTS + nLeft + 1] = Vector3D(xSym + d, 0, 0);

    std::vector<Vector3D> pts;
    pts.reserve(N_TOTAL_PTS);
    for(size_t i = 0; i < N_X_PTS; ++i)
    {
        for(size_t jy = 0; jy < NYZ; ++jy)
        {
            for(size_t jz = 0; jz < NYZ; ++jz)
            {
                pts.emplace_back(xPts[i].x, yzCenters[jy], yzCenters[jz]);
            }
        }
    }
    return pts;
}

// ============================================================
// SyncParticleCellIDs: keep particle.cellID in sync with
// the current cell's ID (like RICH's MonteCarloManager3D
// wrappers and Voronoi3DMovement.cpp post-UNC sync).
// ============================================================

static void SyncParticleCellIDs(const std::vector<MovingSlabCell> &cells,
                                std::vector<STORM::Particle<Vector3D, Grid>> &particles)
{
    size_t N = cells.size();
    for(auto &p : particles)
    {
        if(p.cellIndex < N)
        {
            p.cellID = cells[p.cellIndex].ID;
        }
    }
}

// ============================================================
// Rebalance: redistribute cells across MPI ranks
// ============================================================

static bool Rebalance(Grid &grid,
                      STORM::MonteCarloManager<Vector3D, Grid> &manager,
                      std::vector<MovingSlabCell> &cells,
                      std::vector<MovingSlabExtensives> &extensives,
                      std::vector<STORM::Particle<Vector3D, Grid>> &particles,
                      int rank)
{
    size_t N = grid.GetPointNo();
    std::vector<double> weights(N, 1);

    const std::vector<size_t> &counters = manager.GetCellsStepsCounters();
    const std::vector<size_t> &particleCounts = manager.GetBeginningParticleCount();

    double maxFactor = 20, minFactor = 0.1;
    double sumOfWeights = 0;
    for(size_t i = 0; i < N; i++)
    {
        if(counters.size() == N)
        {
            weights[i] += 0.005 * static_cast<double>(counters[i]);
            sumOfWeights += weights[i];
        }
//        if(particleCounts.size() == N)
//        {
//            weights[i] += 1 * static_cast<double>(particleCounts[i]);
//        }
    }

    size_t Ntotal = N;
    MPI_Allreduce(MPI_IN_PLACE, &sumOfWeights, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &Ntotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    double averageWeight = sumOfWeights / Ntotal;
    double minWeight = averageWeight * minFactor;
    double maxWeight = averageWeight * maxFactor;
    for(double &w : weights)
    {
        w = std::clamp(w, minWeight, maxWeight);
    }

    if(!grid.ShouldRebalance(weights))
    {
        return false;
    }

    if(rank == 0)
    {
        std::cout << "Rebalancing..." << std::endl;
    }

    grid.Rebalance(weights);

    STORM::MPI_exchange_data(grid, cells, false);
    STORM::MPI_exchange_data(grid, extensives, false);

    // Migrate MC cost counters along with cells (like RICH's addMigrationBuffer)
    STORM::MPI_exchange_data(grid, manager.GetCellsStepsCounters(), false);
    STORM::MPI_exchange_data(grid, manager.GetBeginningParticleCount(), false);

    STORM::UpdateNewCellsAfterExchange<Vector3D>(grid, particles);

    size_t newN = grid.GetPointNo();
    cells.resize(newN);
    extensives.resize(newN);
    manager.GetCellsStepsCounters().resize(newN, 0);
    manager.GetBeginningParticleCount().resize(newN, 0);

    if(rank == 0)
    {
        std::cout << "Rebalance done, local cells: " << newN << std::endl;
    }

    return true;
}

// ============================================================
// Remesh: move points with the slab, rebuild tessellation
// ============================================================

static void Remesh(Grid &grid, double vSlab, double L_slab, double xSym,
                   double prevTime, double nowTime,
                   std::vector<MovingSlabCell> &cells,
                   std::vector<MovingSlabExtensives> &extensives,
                   std::vector<STORM::Particle<Vector3D, Grid>> &particles,
                   STORM::MonteCarloManager<Vector3D, Grid> &manager)
{
    double slabFrontOld = L_slab + vSlab * prevTime;
    double slabFrontNew = L_slab + vSlab * nowTime;
    double dx = vSlab * (nowTime - prevTime);

    size_t localN = grid.GetPointNo();
    std::vector<Vector3D> localPoints(localN);
    for(size_t i = 0; i < localN; ++i)
    {
        Vector3D p = grid.GetMeshPoint(i);
        if(p.x <= slabFrontOld)
        {
            p.x += dx;
        }
        else
        {
            double f = (p.x - slabFrontOld) / (xSym - slabFrontOld);
            p.x = slabFrontNew + f * (xSym - slabFrontNew);
        }
        localPoints[i] = p;
    }

    Vector3D newLL(vSlab * nowTime, grid.GetBoxCoordinates().first.y, grid.GetBoxCoordinates().first.z);
    Vector3D ur = grid.GetBoxCoordinates().second;
    grid.SetBox(newLL, ur);
    grid.BuildParallel(localPoints);

    STORM::MPI_exchange_data(grid, cells, false);
    STORM::MPI_exchange_data(grid, extensives, false);
    STORM::MPI_exchange_data(grid, manager.GetCellsStepsCounters(), false);
    STORM::MPI_exchange_data(grid, manager.GetBeginningParticleCount(), false);

    size_t newN = grid.GetPointNo();
    manager.GetCellsStepsCounters().resize(newN, 0);
    manager.GetBeginningParticleCount().resize(newN, 0);
    for(size_t i = cells.size(); i < newN; ++i)
    {
        cells.push_back(MovingSlabCell{});
    }
    for(size_t i = extensives.size(); i < newN; ++i)
    {
        extensives.push_back(MovingSlabExtensives{});
    }
    cells.resize(newN);
    extensives.resize(newN);

    for(size_t i = 0; i < newN; ++i)
    {
        extensives[i].internal_energy = cells[i].internalEnergy;
        extensives[i].mass = cells[i].density * grid.GetVolume(i);
    }

    auto boxCoords = grid.GetBoxCoordinates();
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [&boxCoords](const STORM::Particle<Vector3D, Grid> &p)
            {
                return p.location.x < boxCoords.first.x || p.location.x > boxCoords.second.x;
            }),
        particles.end());
}

// ============================================================
// RunSimulation: main time-stepping loop
// ============================================================

struct SimulationResult
{
    size_t cycles;
    double wallTimeSeconds;
};

static SimulationResult RunSimulation(
    Grid &grid,
    STORM::MonteCarloManager<Vector3D, Grid> &manager,
    std::vector<MovingSlabCell> &cells,
    std::vector<MovingSlabExtensives> &extensives,
    std::vector<STORM::Particle<Vector3D, Grid>> &particles,
    double vSlab, double L_slab, double xSym, double tO,
    int rank, int nprocs)
{
    size_t Ncells = grid.GetPointNo();
    double dt = 1e-3 * 1e-9;
    double const dtMax = 0.1 * 1e-9;
    double const dtRamp = 1.1;
    double const tEnd = tO + dtMax / 2.0;
    double simTime = 0;
    size_t cycle = 0;

    auto wallStart = std::chrono::high_resolution_clock::now();
    double prevTime = 0.0;

    // Dual load balancers (like RICH's loads["remesh"] and loads["radiation-mc"]).
    std::shared_ptr<LoadBalancer<Vector3D>> remeshLB = (nprocs > 1) ? grid.GetLoadBalancer() : nullptr;
    std::shared_ptr<LoadBalancer<Vector3D>> mcLB = (nprocs > 1) ? grid.GetLoadBalancer() : nullptr;

    auto switchLoadBalancer = [&](const std::shared_ptr<LoadBalancer<Vector3D>> &lb, const char *label)
    {
        grid.SetLoadBalancer(lb);
        STORM::MPI_exchange_data(grid, cells, false);
        STORM::MPI_exchange_data(grid, extensives, false);
        STORM::MPI_exchange_data(grid, manager.GetCellsStepsCounters(), false);
        STORM::MPI_exchange_data(grid, manager.GetBeginningParticleCount(), false);
        Ncells = grid.GetPointNo();
        if(rank == 0)
        {
            std::cout << "  LB switch (" << label << "). Ncells in rank 0: " << Ncells << std::endl;
        }
    };

    while(simTime < tEnd)
    {
        double thisDt = std::min(dt, tEnd - simTime);
        if(thisDt <= 0)
        {
            break;
        }

        if(rank == 0)
        {
            std::cout << "=== Cycle " << cycle << ", t=" << simTime << ", dt=" << thisDt << " ===" << std::endl;
        }

        if(cycle > 0)
        {
            if(nprocs > 1)
            {
                switchLoadBalancer(remeshLB, "switch-to-remesh");
            }

            Remesh(grid, vSlab, L_slab, xSym, prevTime, simTime, cells, extensives, particles, manager);
            Ncells = grid.GetPointNo();

            if(nprocs > 1)
            {
                remeshLB = grid.GetLoadBalancer();
            }

            if(nprocs > 1)
            {
                switchLoadBalancer(mcLB, "switch-to-mc");
            }

            bool forceRebalance = cycle < 4;
            if(nprocs > 1 and (forceRebalance or (cycle % 5 == 0)))
            {
                {
                    std::vector<size_t> cellIDs(Ncells);
                    for(size_t i = 0; i < Ncells; ++i)
                    {
                        cellIDs[i] = cells[i].ID;
                    }
                    STORM::MeshMovement<Vector3D, Grid>::UpdateNewCells(grid, particles, cellIDs);
                    SyncParticleCellIDs(cells, particles);
                }

                if(Rebalance(grid, manager, cells, extensives, particles, rank))
                {
                    Ncells = grid.GetPointNo();
                }
            }

            if(nprocs > 1)
            {
                mcLB = grid.GetLoadBalancer();
            }
        }

        {
            std::vector<size_t> cellIDs(Ncells);
            for(size_t i = 0; i < Ncells; ++i)
            {
                cellIDs[i] = cells[i].ID;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            auto uncStart = std::chrono::high_resolution_clock::now();
            STORM::MeshMovement<Vector3D, Grid>::UpdateNewCells(grid, particles, cellIDs);
            MPI_Barrier(MPI_COMM_WORLD);
            double uncTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - uncStart).count();
            if(rank == 0)
            {
                std::cout << "  UNC (pre-transport): " << uncTime << "s, particles=" << particles.size() << std::endl;
            }
            SyncParticleCellIDs(cells, particles);
        }

        prevTime = simTime;
        particles = manager.step(std::move(particles), thisDt);
        SyncParticleCellIDs(cells, particles);

        simTime += thisDt;
        cycle++;
        dt = std::min(dt * dtRamp, dtMax);

        if(cycle % 5 == 0 and rank == 0)
        {
            double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - wallStart).count();
            double slabBackNow  = vSlab * simTime;
            double slabFrontNow = L_slab + vSlab * simTime;
            std::cout << "Step " << cycle
                      << "  t=" << simTime * 1e9 << " ns"
                      << "  dt=" << thisDt * 1e9 << " ns"
                      << "  slab=[" << slabBackNow << ", " << slabFrontNow << "]"
                      << "  elapsed=" << elapsed << "s" << std::endl;
        }
    }

    double wallTotal = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - wallStart).count();
    if(rank == 0)
    {
        std::cout << "Done. " << cycle << " steps, wall time: " << wallTotal << "s" << std::endl;
    }

    return {cycle, wallTotal};
}

// ============================================================
// Main
// ============================================================

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    using IMC = STORM::RadiationIMC<Vector3D, Grid, MovingSlabCell, MovingSlabExtensives,
                                    MovingSlabEOS, G>;

    size_t newPhotonsPerCell = (argc >= 2) ? std::stoul(argv[1]) : 30000 / (NYZ * NYZ);

    double const rhoSlab     = 0.1;
    double const L_slab      = 0.4;
    double const T_slab_keV  = 1.0;
    double const T_slab      = T_slab_keV * units::kev_kelvin;
    double const vSlab       = 0.5994e9;
    double const zO          = 12.0;
    double const xSym        = 12.0;
    double const tO           = 10e-9;
    double const rhoVacuum   = 1e-10;
    double const cellHalfYZ  = 1.0;

    double const xMax = zO + 0.2;
    Vector3D ll(0, -cellHalfYZ, -cellHalfYZ);
    Vector3D ur(xMax, cellHalfYZ, cellHalfYZ);

    // --- Build initial mesh ---
    std::vector<Vector3D> points;
    if(nprocs == 1)
    {
        points = BuildAllPoints(0.0, vSlab, L_slab, xSym, cellHalfYZ);
    }
    else
    {
        if(rank == 0)
        {
            points = BuildAllPoints(0.0, vSlab, L_slab, xSym, cellHalfYZ);
        }
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    Grid grid(ll, ur);
    if(nprocs == 1)
    {
        grid.Build(points);
    }
    else
    {
        grid.BuildParallel(points);
    }

    // --- EOS ---
    double const cvPerVolumeSlab = 1e23 / units::kev_kelvin;
    double const cvPerVolumeVac  = 1e23 / units::kev_kelvin;
    double const T_vac = 1e5;

    // --- Initialize cells ---
    size_t Ncells = grid.GetPointNo();
    std::vector<MovingSlabCell> cells(Ncells);
    std::vector<MovingSlabExtensives> extensives(Ncells);

    std::array<double, G + 1> energyBoundaries{};
    for(size_t g = 0; g < G; ++g)
    {
        energyBoundaries[g] = STORM::examples::OPACITY_TABLE_124[g].nu_min * units::kev;
    }
    energyBoundaries[G] = STORM::examples::OPACITY_TABLE_124[G - 1].nu_max * units::kev;

    size_t globalCellOffset = 0;
    MPI_Exscan(&Ncells, &globalCellOffset, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    for(size_t i = 0; i < Ncells; ++i)
    {
        cells[i].ID = globalCellOffset + i;
        double x = grid.GetMeshPoint(i).x;
        double volume = grid.GetVolume(i);
        bool inSlab = (x >= 0.0 and x <= L_slab);

        double rho = inSlab ? rhoSlab : rhoVacuum;
        double cvPerVolume = inSlab ? cvPerVolumeSlab : cvPerVolumeVac;
        double T = inSlab ? T_slab : T_vac;

        cells[i].density = rho;
        cells[i].temperature = T;
        cells[i].internalEnergy = cvPerVolume * T * volume;
        cells[i].velocity = inSlab ? Vector3D(vSlab, 0, 0) : Vector3D(0, 0, 0);

        for(size_t g = 0; g < G; ++g)
        {
            if(inSlab)
            {
                cells[i].Eg[g] = planck_integral::planck_energy_density_group_integral(
                    energyBoundaries[g], energyBoundaries[g + 1], T_slab) / rho;
            }
            else
            {
                cells[i].Eg[g] = 0.0;
            }
        }
        cells[i].Erad = std::accumulate(cells[i].Eg.begin(), cells[i].Eg.end(), 0.0);

        extensives[i].mass = rho * volume;
        extensives[i].internal_energy = cells[i].internalEnergy;
        extensives[i].Erad = cells[i].Erad * extensives[i].mass;
        for(size_t g = 0; g < G; ++g)
        {
            extensives[i].Eg[g] = cells[i].Eg[g] * extensives[i].mass;
        }
    }

    // --- IMC parameters ---
    STORM::RadiationIMCParameters<G> imcParams;
    imcParams.newPhotonsPerCell = newPhotonsPerCell;
    imcParams.withHydro = true;
    imcParams.withMultigroupOpacity = true;
    imcParams.withDDMC = true;
    imcParams.noHydroFeedback = true;
    imcParams.withEgTimeAvg = true;
    imcParams.energyBoundariesProvided = true;
    imcParams.energyBoundaries = energyBoundaries;

    auto eos = std::make_shared<MovingSlabEOS>(cvPerVolumeSlab, cvPerVolumeVac, rhoSlab, rhoVacuum);
    auto opacityModel = std::make_shared<STORM::examples::MovingSlabOpacity<Vector3D, Grid, MovingSlabCell>>(rhoSlab, cells);
    auto boundary = std::make_shared<STORM::examples::MovingSlabBoundary<Vector3D, Grid>>(grid);
    auto physics = std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacityModel, imcParams);
    auto popControl = std::make_shared<STORM::NoPopulationControl<Vector3D, Grid>>(grid);

    STORM::MonteCarloManager<Vector3D, Grid> manager = STORM::CreateMonteCarloManager<Vector3D, Grid>(
        grid, physics, popControl, boundary);
    std::vector<STORM::Particle<Vector3D, Grid>> particles;

    if(rank == 0)
    {
        std::cout << "Moving slab MC benchmark: "
                  << G << " groups, "
                  << N_TOTAL_PTS << " mesh points (" << N_X_PTS << "x * " << NYZ << "y * " << NYZ << "z), "
                  << "newPhotonsPerCell=" << newPhotonsPerCell
                  << ", v_slab=" << vSlab << " cm/s"
                  << ", L=" << L_slab << " cm"
                  << ", T=" << T_slab_keV << " keV"
                  << ", z_O=" << zO << " cm"
                  << ", t_O=" << tO * 1e9 << " ns"
                  << ", MPI ranks=" << nprocs
                  << std::endl;
    }

    SimulationResult result = RunSimulation(grid, manager, cells, extensives, particles,
                                            vSlab, L_slab, xSym, tO, rank, nprocs);

    // --- Find observer cells at x closest to z_O ---
    const auto &EgTA = physics->getEgTimeAvg();
    size_t localN = grid.GetPointNo();

    double bestX = 0.0;
    double minDist = std::numeric_limits<double>::max();
    for(size_t i = 0; i < localN; ++i)
    {
        double d = std::abs(grid.GetMeshPoint(i).x - zO);
        if(d < minDist)
        {
            minDist = d;
            bestX = grid.GetMeshPoint(i).x;
        }
    }

    int writerRank = 0;
    {
        struct { double dist; int rank; } localBest{minDist, rank}, globalBest;
        MPI_Allreduce(&localBest, &globalBest, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        writerRank = globalBest.rank;
        MPI_Bcast(&bestX, 1, MPI_DOUBLE, writerRank, MPI_COMM_WORLD);
    }

    constexpr double xTol = 1e-8;
    std::array<double, G> localEgSum{};
    size_t localObsCount = 0;
    for(size_t i = 0; i < localN; ++i)
    {
        if(std::abs(grid.GetMeshPoint(i).x - bestX) < xTol)
        {
            const auto &egta = EgTA[i];
            for(size_t g = 0; g < G; ++g)
            {
                localEgSum[g] += egta[g];
            }
            ++localObsCount;
        }
    }

    std::array<double, G> globalEgSum{};
    size_t globalObsCount = 0;
    MPI_Reduce(localEgSum.data(), globalEgSum.data(), G, MPI_DOUBLE, MPI_SUM, writerRank, MPI_COMM_WORLD);
    MPI_Reduce(&localObsCount, &globalObsCount, 1, MPI_UNSIGNED_LONG, MPI_SUM, writerRank, MPI_COMM_WORLD);

    if(rank == writerRank)
    {
        std::cout << "Observer cells: " << globalObsCount << " cells at x=" << bestX
                  << " (target z_O=" << zO << ", dist=" << minDist << ")" << std::endl;

        double invCount = (globalObsCount > 0) ? 1.0 / static_cast<double>(globalObsCount) : 0.0;

        std::string specPath = "moving_slab_mc_spectrum.txt";
        std::ofstream out(specPath);
        out << std::scientific << std::setprecision(12);
        out << "# Moving slab MC benchmark\n";
        out << "# v_slab_cm_per_ns " << vSlab / 1e9 << "\n";
        out << "# L_slab_cm " << L_slab << "\n";
        out << "# T_slab_keV " << T_slab_keV << "\n";
        out << "# rho_slab " << rhoSlab << "\n";
        out << "# z_O_cm " << zO << "\n";
        out << "# t_O_ns " << tO * 1e9 << "\n";
        out << "# observer_x_cm " << bestX << "\n";
        out << "# observer_yz_cells " << globalObsCount << "\n";
        out << "# steps " << result.cycles << "\n";
        out << "# wall_time_s " << result.wallTimeSeconds << "\n";
        out << "# mpi_ranks " << nprocs << "\n";
        out << "# columns: group nu_min_keV nu_max_keV Eg_time_avg_erg_per_cm3\n";

        for(size_t g = 0; g < G; ++g)
        {
            out << g
                << " " << STORM::examples::OPACITY_TABLE_124[g].nu_min
                << " " << STORM::examples::OPACITY_TABLE_124[g].nu_max
                << " " << globalEgSum[g] * invCount
                << "\n";
        }
        out.close();
        std::cout << "Wrote " << specPath << std::endl;
    }

    if(rank == 0)
    {
        std::string scriptDir = __FILE__;
        scriptDir = scriptDir.substr(0, scriptDir.rfind('/'));
        std::string cmd = "python3 " + scriptDir + "/check_spectrum.py";
        std::cout << "Running: " << cmd << std::endl;
        std::system(cmd.c_str());
    }

    MPI_Finalize();
    return 0;
}
