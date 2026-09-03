#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <mpi.h>

#include "examples/Vector3D.hpp"
#include "MadVoro/Voronoi3D.hpp"
#include <planck_integral/planck_integral.hpp>
#include <units/units.hpp>
#include <mpi_utils/mpi_collectives.hpp>

#include "boundary/BoundaryCondition.hpp"
#include "manager/MonteCarloManagerFactory.hpp"
#include "mesh_movement/VoronoiMeshMovement.hpp"
#include "population/NoPopulationControl.hpp"
#include "radiation/RadiationIMC.hpp"
#include "radiation/RadiationIMCParameters.hpp"
#include "radiation/RadiationOpacityModel.hpp"
#include "utils/MpiExchangeGrid.hpp"
#include "examples/moving_slab/MovingSlabBoundary.hpp"
#include "examples/moving_slab/MovingSlabOpacity.hpp"

namespace
{

constexpr size_t G = 32;
constexpr size_t N_FINE_GROUPS = 124;
constexpr size_t N_COARSE_GROUPS = 32;
constexpr double COARSE_EMIN_KEV = 1.0e-3;
constexpr double COARSE_EMAX_KEV = 3.0e+1;
constexpr size_t N_SLAB_PTS = 20;
constexpr size_t N_VAC_PTS = 60;
constexpr size_t N_X_PTS = N_SLAB_PTS + N_VAC_PTS;
constexpr size_t NYZ = 3;
constexpr size_t N_TOTAL_PTS = N_X_PTS * NYZ * NYZ;

using Grid = MadVoro::Voronoi3D<Vector3D>;
using Particle = STORM::Particle<Vector3D>;

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
                const std::vector<double> &tracers,
                const std::vector<std::string> &tracerNames) const
    {
        double cv = dT2cv(density, 0.0, tracers, tracerNames);
        return (cv > 0.0) ? specificEnergy / cv : 0.0;
    }

private:
    double cvPerMassSlab_;
    double cvPerMassVac_;
    double rhoSlab_;
};

struct CollapsedOpacity
{
    std::array<double, N_COARSE_GROUPS + 1> boundaryKeV{};
    std::array<double, N_COARSE_GROUPS> kappa{};
};

CollapsedOpacity CollapseOpacityPlanck(double temperature)
{
    CollapsedOpacity result;
    double ratio = std::pow(COARSE_EMAX_KEV / COARSE_EMIN_KEV,
                            1.0 / static_cast<double>(N_COARSE_GROUPS));
    result.boundaryKeV[0] = COARSE_EMIN_KEV;
    for(size_t g = 0; g < N_COARSE_GROUPS; ++g)
    {
        result.boundaryKeV[g + 1] = result.boundaryKeV[g] * ratio;
    }
    result.boundaryKeV[N_COARSE_GROUPS] = COARSE_EMAX_KEV;

    for(size_t coarseGroup = 0; coarseGroup < N_COARSE_GROUPS; ++coarseGroup)
    {
        double newLo = result.boundaryKeV[coarseGroup] * units::kev;
        double newHi = result.boundaryKeV[coarseGroup + 1] * units::kev;
        double numerator = 0.0;
        double denominator = 0.0;
        for(size_t fineGroup = 0; fineGroup < N_FINE_GROUPS; ++fineGroup)
        {
            double oldLo =
                STORM::examples::OPACITY_TABLE_124[fineGroup].nu_min * units::kev;
            double oldHi =
                STORM::examples::OPACITY_TABLE_124[fineGroup].nu_max * units::kev;
            double overlapLo = std::max(newLo, oldLo);
            double overlapHi = std::min(newHi, oldHi);
            if(overlapHi <= overlapLo)
            {
                continue;
            }
            double planck = planck_integral::planck_energy_density_group_integral(
                overlapLo, overlapHi, temperature);
            numerator += STORM::examples::OPACITY_TABLE_124[fineGroup].kappa * planck;
            denominator += planck;
        }
        result.kappa[coarseGroup] =
            (denominator > 0.0) ? numerator / denominator : 0.0;
    }
    return result;
}

class MovingSlabOpacity32 final
    : public STORM::RadiationOpacityModel<Vector3D, Grid, MovingSlabCell, G>
{
public:
    using Base = STORM::RadiationOpacityModel<Vector3D, Grid, MovingSlabCell, G>;
    using GroupArray = typename Base::GroupArray;
    using GroupBoundaries = typename Base::GroupBoundaries;
    using GroupCdf = std::array<double, G + 1>;

    MovingSlabOpacity32(double rhoSlab,
                        const std::array<double, G> &sigma,
                        const GroupBoundaries &boundaries)
        : rhoSlab_(rhoSlab), sigma_(sigma), boundaries_(boundaries)
    {}

    double CalcPlanckOpacity(const MovingSlabCell &cell) override
    {
        if(IsVacuum(cell))
        {
            return 1e-12;
        }
        double numerator = 0.0;
        double denominator = 0.0;
        for(size_t g = 0; g < G; ++g)
        {
            double planck = planck_integral::planck_energy_density_group_integral(
                boundaries_[g], boundaries_[g + 1], cell.temperature);
            numerator += sigma_[g] * planck;
            denominator += planck;
        }
        return (denominator > 0.0) ? numerator / denominator : 0.0;
    }

    double CalcAbsorptionOpacity(const MovingSlabCell &cell, double energy) override
    {
        if(IsVacuum(cell))
        {
            return 1e-12;
        }
        size_t group = findGroup(energy, boundaries_);
        return (group < G) ? sigma_[group] : 1e-100;
    }

    double CalcScatteringOpacity(const MovingSlabCell &) override
    {
        return 0.0;
    }

    double CalcScatteringOpacity(const MovingSlabCell &, double) override
    {
        return 0.0;
    }

    double GetThermalEnergy(const MovingSlabCell &cell, double random,
                            const GroupBoundaries &boundaries) const override
    {
        GroupCdf cumulative = ComputeCumulativePlanck(cell, boundaries);
        double total = cumulative[G];
        if(!(total > 0.0) || !std::isfinite(total))
        {
            return Base::GetThermalEnergy(cell, random, boundaries);
        }
        double r = std::clamp(random, 0.0, std::nextafter(1.0, 0.0));
        return STORM::LinearInterpolation(cumulative, boundaries, r * total);
    }

    double SampleThermalEnergyInGroup(
        const MovingSlabCell &cell, size_t group, double random,
        const GroupBoundaries &boundaries) const override
    {
        group = std::min(group, G - 1);
        GroupCdf cumulative = ComputeCumulativePlanck(cell, boundaries);
        double c0 = cumulative[group];
        double c1 = cumulative[group + 1];
        if(c1 <= c0 || !std::isfinite(c1 - c0))
        {
            return 0.5 * (boundaries[group] + boundaries[group + 1]);
        }
        double r = std::clamp(random, 0.0, std::nextafter(1.0, 0.0));
        return STORM::LinearInterpolation(cumulative, boundaries, c0 + r * (c1 - c0));
    }

    GroupArray GetThermalGroupPdf(const MovingSlabCell &cell,
                                  const GroupBoundaries &boundaries) const override
    {
        GroupArray pdf{};
        GroupCdf cumulative = ComputeCumulativePlanck(cell, boundaries);
        double total = cumulative[G];
        if(!(total > 0.0) || !std::isfinite(total))
        {
            return pdf;
        }
        for(size_t g = 0; g < G; ++g)
        {
            double weight = cumulative[g + 1] - cumulative[g];
            pdf[g] = (weight > 0.0 && std::isfinite(weight)) ? weight / total : 0.0;
        }
        return pdf;
    }

    GroupArray GetCumulativeOpacity(const MovingSlabCell &cell,
                                    const GroupBoundaries &boundaries) const override
    {
        GroupArray cumulativeUpper{};
        GroupCdf cumulative = ComputeCumulativePlanck(cell, boundaries);
        for(size_t g = 0; g < G; ++g)
        {
            cumulativeUpper[g] = cumulative[g + 1];
        }
        return cumulativeUpper;
    }

    GroupArray getEnergyCenters(const GroupBoundaries &boundaries) const override
    {
        GroupArray centers{};
        for(size_t g = 0; g < G; ++g)
        {
            centers[g] = 0.5 * (boundaries[g] + boundaries[g + 1]);
        }
        return centers;
    }

private:
    bool IsVacuum(const MovingSlabCell &cell) const
    {
        return cell.density < 0.5 * rhoSlab_;
    }

    GroupCdf ComputeCumulativePlanck(
        const MovingSlabCell &cell, const GroupBoundaries &boundaries) const
    {
        GroupCdf cumulative{};
        bool vacuum = IsVacuum(cell);
        double kT = units::k_boltz * cell.temperature;
        for(size_t g = 0; g < G; ++g)
        {
            double a = boundaries[g] / kT;
            double b = boundaries[g + 1] / kT;
            double planck = (a > 0.0 && b > a)
                ? planck_integral::planck_integral(a, b) : 0.0;
            double weight = (vacuum ? 1e-12 : sigma_[g]) * planck;
            cumulative[g + 1] = cumulative[g] + weight;
        }
        return cumulative;
    }

    double rhoSlab_;
    std::array<double, G> sigma_;
    GroupBoundaries boundaries_;
};

void SyncParticleCellIDs(const std::vector<MovingSlabCell> &cells,
                         std::vector<Particle> &particles)
{
    for(Particle &particle : particles)
    {
        if(particle.cellIndex < cells.size())
        {
            particle.cellID = cells[particle.cellIndex].ID;
        }
    }
}

std::vector<Vector3D> BuildAllPoints(double time, double slabVelocity,
                                     double slabLength, double symmetryPoint,
                                     double cellHalfYZ)
{
    std::array<double, N_SLAB_PTS> slabPointsX{};
    for(size_t i = 0; i < N_SLAB_PTS; ++i)
    {
        slabPointsX[i] =
            slabLength * (static_cast<double>(i) + 0.5) / N_SLAB_PTS;
    }

    std::array<double, NYZ> yzCenters{};
    for(size_t k = 0; k < NYZ; ++k)
    {
        yzCenters[k] = -cellHalfYZ +
            cellHalfYZ * (2.0 * k + 1.0) / NYZ;
    }

    std::array<double, N_X_PTS> xPoints{};
    double shift = slabVelocity * time;
    for(size_t i = 0; i < N_SLAB_PTS; ++i)
    {
        xPoints[i] = slabPointsX[i] + shift;
    }
    double vacuumStart = slabLength + shift;
    size_t leftPoints = N_VAC_PTS - 2;
    double h = (symmetryPoint - vacuumStart) / leftPoints;
    double d = h / 2.0;
    for(size_t i = 0; i < leftPoints; ++i)
    {
        xPoints[N_SLAB_PTS + i] =
            vacuumStart + h * (static_cast<double>(i) + 0.5);
    }
    xPoints[N_SLAB_PTS + leftPoints] = symmetryPoint;
    xPoints[N_SLAB_PTS + leftPoints + 1] = symmetryPoint + d;

    std::vector<Vector3D> points;
    points.reserve(N_TOTAL_PTS);
    for(size_t i = 0; i < N_X_PTS; ++i)
    {
        for(size_t jy = 0; jy < NYZ; ++jy)
        {
            for(size_t jz = 0; jz < NYZ; ++jz)
            {
                points.emplace_back(xPoints[i], yzCenters[jy], yzCenters[jz]);
            }
        }
    }
    return points;
}

void ExchangeCellData(
    Grid &grid, std::vector<MovingSlabCell> &cells,
    std::vector<MovingSlabExtensives> &extensives,
    STORM::MonteCarloManager<Vector3D, Grid> &manager)
{
    STORM::MPI_exchange_data(grid, cells, false);
    STORM::MPI_exchange_data(grid, extensives, false);
    STORM::MPI_exchange_data(grid, manager.GetCellsStepsCounters(), false);
    STORM::MPI_exchange_data(grid, manager.GetBeginningParticleCount(), false);
}

void Remesh(
    Grid &grid, double slabVelocity, double slabLength, double symmetryPoint,
    double previousTime, double currentTime,
    std::vector<MovingSlabCell> &cells,
    std::vector<MovingSlabExtensives> &extensives,
    std::vector<Particle> &particles,
    STORM::MonteCarloManager<Vector3D, Grid> &manager)
{
    double oldFront = slabLength + slabVelocity * previousTime;
    double newFront = slabLength + slabVelocity * currentTime;
    double dx = slabVelocity * (currentTime - previousTime);
    size_t localN = grid.GetPointNo();
    std::vector<Vector3D> localPoints(localN);
    for(size_t i = 0; i < localN; ++i)
    {
        Vector3D point = grid.GetMeshPoint(i);
        if(point.x <= oldFront)
        {
            point.x += dx;
        }
        else
        {
            double fraction = (point.x - oldFront) / (symmetryPoint - oldFront);
            point.x = newFront + fraction * (symmetryPoint - newFront);
        }
        localPoints[i] = point;
    }

    std::pair<Vector3D, Vector3D> box = grid.GetBoxCoordinates();
    grid.SetBox(Vector3D(slabVelocity * currentTime, box.first.y, box.first.z),
                box.second);
    grid.BuildParallel(localPoints);
    ExchangeCellData(grid, cells, extensives, manager);

    size_t newN = grid.GetPointNo();
    manager.GetCellsStepsCounters().resize(newN, 0);
    manager.GetBeginningParticleCount().resize(newN, 0);
    cells.resize(newN);
    extensives.resize(newN);
    for(size_t i = 0; i < newN; ++i)
    {
        extensives[i].internal_energy = cells[i].internalEnergy;
        extensives[i].mass = cells[i].density * grid.GetVolume(i);
    }

    std::pair<Vector3D, Vector3D> newBox = grid.GetBoxCoordinates();
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [&newBox](const Particle &particle)
            {
                return particle.location.x < newBox.first.x ||
                       particle.location.x > newBox.second.x;
            }),
        particles.end());
}

bool Rebalance(
    Grid &grid, STORM::MonteCarloManager<Vector3D, Grid> &manager,
    std::vector<MovingSlabCell> &cells,
    std::vector<MovingSlabExtensives> &extensives,
    std::vector<Particle> &particles, int rank)
{
    size_t n = grid.GetPointNo();
    std::vector<double> weights(n, 1.0);
    const std::vector<size_t> &counters = manager.GetCellsStepsCounters();
    const std::vector<size_t> &particleCounts =
        manager.GetBeginningParticleCount();
    double localWeight = 0.0;
    for(size_t i = 0; i < n; ++i)
    {
        if(counters.size() == n)
        {
            weights[i] += 0.005 * static_cast<double>(counters[i]);
        }
        if(particleCounts.size() == n)
        {
            weights[i] += 10.0 * static_cast<double>(particleCounts[i]);
        }
        localWeight += weights[i];
    }
    size_t totalCells = n;
    double totalWeight = localWeight;
    MPI_Allreduce(MPI_IN_PLACE, &totalWeight, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &totalCells, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    double averageWeight = totalWeight / static_cast<double>(totalCells);
    for(double &weight : weights)
    {
        weight = std::clamp(weight, 0.1 * averageWeight, 20.0 * averageWeight);
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
    ExchangeCellData(grid, cells, extensives, manager);
    STORM::UpdateNewCellsAfterExchange<Vector3D>(grid, particles);
    size_t newN = grid.GetPointNo();
    cells.resize(newN);
    extensives.resize(newN);
    manager.GetCellsStepsCounters().resize(newN, 0);
    manager.GetBeginningParticleCount().resize(newN, 0);
    return true;
}

} // anonymous namespace

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    constexpr double rhoSlab = 0.1;
    constexpr double slabLength = 0.4;
    constexpr double slabTemperatureKeV = 1.0;
    constexpr double slabVelocity = 0.5994e9;
    constexpr double observerPosition = 12.0;
    constexpr double observationTime = 10e-9;
    constexpr double rhoVacuum = 1e-10;
    constexpr double cellHalfYZ = 0.5;
    constexpr double symmetryPoint = 12.0;
    constexpr double xMax = observerPosition + 0.2;

    CollapsedOpacity collapsed =
        CollapseOpacityPlanck(slabTemperatureKeV * units::kev_kelvin);
    std::array<double, G + 1> energyBoundaries{};
    std::array<double, G> energyCenters{};
    std::array<double, G> sigma{};
    for(size_t g = 0; g <= G; ++g)
    {
        energyBoundaries[g] = collapsed.boundaryKeV[g] * units::kev;
    }
    for(size_t g = 0; g < G; ++g)
    {
        energyCenters[g] = 0.5 * (energyBoundaries[g] + energyBoundaries[g + 1]);
        sigma[g] = rhoSlab * collapsed.kappa[g];
    }

    Vector3D lowerLeft(0, -cellHalfYZ, -cellHalfYZ);
    Vector3D upperRight(xMax, cellHalfYZ, cellHalfYZ);
    std::vector<Vector3D> points = BuildAllPoints(
        0.0, slabVelocity, slabLength, symmetryPoint, cellHalfYZ);
    if(nprocs > 1)
    {
        if(rank != 0)
        {
            points.clear();
        }
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    }

    Grid grid(lowerLeft, upperRight);
    if(nprocs == 1)
    {
        grid.Build(points);
    }
    else
    {
        grid.BuildParallel(points);
    }

    double const cvPerVolume = 1e23 / units::kev_kelvin;
    double const vacuumTemperature = 1e5;
    size_t nCells = grid.GetPointNo();
    std::vector<MovingSlabCell> cells(nCells);
    std::vector<MovingSlabExtensives> extensives(nCells);
    size_t globalCellOffset = 0;
    MPI_Exscan(&nCells, &globalCellOffset, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, MPI_COMM_WORLD);
    for(size_t i = 0; i < nCells; ++i)
    {
        cells[i].ID = globalCellOffset + i;
        double x = grid.GetMeshPoint(i).x;
        bool inSlab = x >= 0.0 && x <= slabLength;
        double density = inSlab ? rhoSlab : rhoVacuum;
        double temperature = inSlab
            ? slabTemperatureKeV * units::kev_kelvin : vacuumTemperature;
        double volume = grid.GetVolume(i);
        cells[i].density = density;
        cells[i].temperature = temperature;
        cells[i].internalEnergy = cvPerVolume * temperature * volume;
        cells[i].velocity = inSlab
            ? Vector3D(slabVelocity, 0, 0) : Vector3D(0, 0, 0);
        for(size_t g = 0; g < G; ++g)
        {
            cells[i].Eg[g] = inSlab
                ? planck_integral::planck_energy_density_group_integral(
                    energyBoundaries[g], energyBoundaries[g + 1], temperature) /
                    density
                : 0.0;
        }
        cells[i].Erad = std::accumulate(
            cells[i].Eg.begin(), cells[i].Eg.end(), 0.0);
        extensives[i].mass = density * volume;
        extensives[i].internal_energy = cells[i].internalEnergy;
        extensives[i].Erad = cells[i].Erad * extensives[i].mass;
        for(size_t g = 0; g < G; ++g)
        {
            extensives[i].Eg[g] = cells[i].Eg[g] * extensives[i].mass;
        }
    }

    MovingSlabEOS eosModel(cvPerVolume, cvPerVolume, rhoSlab, rhoVacuum);
    std::shared_ptr<MovingSlabEOS> eos =
        std::make_shared<MovingSlabEOS>(eosModel);
    std::shared_ptr<MovingSlabOpacity32> opacity =
        std::make_shared<MovingSlabOpacity32>(
            rhoSlab, sigma, energyBoundaries);
    std::shared_ptr<STORM::examples::MovingSlabBoundary<Vector3D, Grid>> boundary =
        std::make_shared<STORM::examples::MovingSlabBoundary<Vector3D, Grid>>(grid);
    using IMC = STORM::RadiationIMC<
        Vector3D, Grid, MovingSlabCell, MovingSlabExtensives,
        MovingSlabEOS, G>;
    STORM::RadiationIMCParameters<G> parameters;
    parameters.newPhotonsPerCell = 10000 / (NYZ * NYZ);
    parameters.withHydro = true;
    parameters.withMultigroupOpacity = true;
    parameters.withRandomWalk = false;
    parameters.withDDMC = false;
    parameters.noHydroFeedback = true;
    parameters.withEgTimeAvg = true;
    parameters.energyBoundaries = energyBoundaries;
    parameters.energyBoundariesProvided = true;
    std::shared_ptr<IMC> physics = std::make_shared<IMC>(
        grid, boundary, cells, extensives, eos, opacity, parameters);
    std::shared_ptr<STORM::NoPopulationControl<Vector3D, Grid>> population =
        std::make_shared<STORM::NoPopulationControl<Vector3D, Grid>>(grid);
    STORM::MonteCarloManager<Vector3D, Grid> manager =
        STORM::CreateMonteCarloManager<Vector3D, Grid>(
            grid, physics, population, boundary);
    std::vector<Particle> particles;

    if(rank == 0)
    {
        std::cout << "Moving slab MC benchmark (32-group, original vacuum): "
                  << G << " groups, " << N_TOTAL_PTS
                  << " mesh points (" << N_X_PTS << "x * " << NYZ
                  << "y * " << NYZ << "z), newPhotonsPerCell="
                  << parameters.newPhotonsPerCell << ", v_slab="
                  << slabVelocity << " cm/s, L=" << slabLength
                  << " cm, T=" << slabTemperatureKeV << " keV, z_O="
                  << observerPosition << " cm, t_O=" << observationTime * 1e9
                  << " ns, MPI ranks=" << nprocs << std::endl;
    }

    double dt = 1e-3 * 1e-9;
    constexpr double dtMax = 0.1e-9;
    constexpr double dtRamp = 1.1;
    double simTime = 0.0;
    double previousTime = 0.0;
    double tEnd = observationTime + dtMax / 2.0;
    size_t stepCount = 0;
    std::chrono::high_resolution_clock::time_point wallStart =
        std::chrono::high_resolution_clock::now();
    std::shared_ptr<LoadBalancer<Vector3D>> remeshLB =
        nprocs > 1 ? grid.GetLoadBalancer() : nullptr;
    std::shared_ptr<LoadBalancer<Vector3D>> mcLB =
        nprocs > 1 ? grid.GetLoadBalancer() : nullptr;

    while(simTime < tEnd)
    {
        double thisDt = std::min(dt, tEnd - simTime);
        if(thisDt <= 0.0)
        {
            break;
        }
        if(stepCount > 0)
        {
            if(nprocs > 1)
            {
                grid.SetLoadBalancer(remeshLB);
                ExchangeCellData(grid, cells, extensives, manager);
            }
            Remesh(grid, slabVelocity, slabLength, symmetryPoint,
                   previousTime, simTime, cells, extensives, particles, manager);
            if(nprocs > 1)
            {
                remeshLB = grid.GetLoadBalancer();
                grid.SetLoadBalancer(mcLB);
                ExchangeCellData(grid, cells, extensives, manager);
            }
            nCells = grid.GetPointNo();
            std::vector<size_t> cellIDs(nCells);
            for(size_t i = 0; i < nCells; ++i)
            {
                cellIDs[i] = cells[i].ID;
            }
            STORM::MeshMovement<Vector3D, Grid>::UpdateNewCells(
                grid, particles, cellIDs);
            if(nprocs > 1 && (stepCount < 4 || stepCount % 5 == 0))
            {
                if(Rebalance(grid, manager, cells, extensives, particles, rank))
                {
                    nCells = grid.GetPointNo();
                }
                mcLB = grid.GetLoadBalancer();
            }
        }

        std::vector<size_t> cellIDs(nCells);
        for(size_t i = 0; i < nCells; ++i)
        {
            cellIDs[i] = cells[i].ID;
        }
        STORM::MeshMovement<Vector3D, Grid>::UpdateNewCells(
            grid, particles, cellIDs);
        previousTime = simTime;
        particles = manager.step(std::move(particles), thisDt);
        SyncParticleCellIDs(cells, particles);
        simTime += thisDt;
        ++stepCount;
        dt = std::min(dt * dtRamp, dtMax);

        if(stepCount % 5 == 0 && rank == 0)
        {
            double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - wallStart).count();
            std::cout << "Step " << stepCount << "  t=" << simTime * 1e9
                      << " ns  dt=" << thisDt * 1e9 << " ns  slab=["
                      << slabVelocity * simTime << ", "
                      << slabLength + slabVelocity * simTime
                      << "]  elapsed=" << elapsed << "s" << std::endl;
        }
    }

    const std::vector<std::array<double, G>> &egTimeAvg =
        physics->getEgTimeAvg();
    size_t localN = grid.GetPointNo();
    double bestX = 0.0;
    double minDistance = std::numeric_limits<double>::max();
    for(size_t i = 0; i < localN; ++i)
    {
        double distance = std::abs(grid.GetMeshPoint(i).x - observerPosition);
        if(distance < minDistance)
        {
            minDistance = distance;
            bestX = grid.GetMeshPoint(i).x;
        }
    }
    struct
    {
        double distance;
        int rank;
    } localBest{minDistance, rank}, globalBest{};
    MPI_Allreduce(&localBest, &globalBest, 1, MPI_DOUBLE_INT, MPI_MINLOC,
                  MPI_COMM_WORLD);
    int writerRank = globalBest.rank;
    MPI_Bcast(&bestX, 1, MPI_DOUBLE, writerRank, MPI_COMM_WORLD);

    constexpr double xTolerance = 1e-8;
    std::array<double, G> localSum{};
    size_t localCount = 0;
    for(size_t i = 0; i < localN; ++i)
    {
        if(std::abs(grid.GetMeshPoint(i).x - bestX) < xTolerance)
        {
            for(size_t g = 0; g < G; ++g)
            {
                localSum[g] += egTimeAvg[i][g];
            }
            ++localCount;
        }
    }
    std::array<double, G> globalSum{};
    size_t globalCount = 0;
    MPI_Reduce(localSum.data(), globalSum.data(), G, MPI_DOUBLE, MPI_SUM,
               writerRank, MPI_COMM_WORLD);
    MPI_Reduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG, MPI_SUM,
               writerRank, MPI_COMM_WORLD);

    if(rank == writerRank)
    {
        std::ofstream output("moving_slab_mc_32_spectrum.txt");
        output << std::scientific << std::setprecision(12);
        output << "# Moving slab MC benchmark 32-group (original_vacuum)\n";
        output << "# v_slab_cm_per_ns " << slabVelocity / 1e9 << "\n";
        output << "# L_slab_cm " << slabLength << "\n";
        output << "# T_slab_keV " << slabTemperatureKeV << "\n";
        output << "# rho_slab " << rhoSlab << "\n";
        output << "# z_O_cm " << observerPosition << "\n";
        output << "# t_O_ns " << observationTime * 1e9 << "\n";
        output << "# observer_x_cm " << bestX << "\n";
        output << "# observer_yz_cells " << globalCount << "\n";
        output << "# steps " << stepCount << "\n";
        output << "# mpi_ranks " << nprocs << "\n";
        output << "# columns: group nu_min_keV nu_max_keV kappa_cm2_per_g "
                   "Eg_time_avg_erg_per_cm3\n";
        double inverseCount = globalCount > 0
            ? 1.0 / static_cast<double>(globalCount) : 0.0;
        for(size_t g = 0; g < G; ++g)
        {
            output << g << " " << collapsed.boundaryKeV[g] << " "
                   << collapsed.boundaryKeV[g + 1] << " "
                   << collapsed.kappa[g] << " "
                   << globalSum[g] * inverseCount << "\n";
        }
        std::cout << "Wrote moving_slab_mc_32_spectrum.txt" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
