#ifndef STORM_MONTE_CARLO_PHYSICS_HPP
#define STORM_MONTE_CARLO_PHYSICS_HPP

#ifdef STORM_WITH_GPU
#include <cstdint>
#endif
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "../particle/Particle.hpp"
#include "../particle/StepResult.hpp"
#include "../boundary/BoundaryCondition.hpp"
#include "../elementary/PointOps.hpp"

namespace STORM {

using namespace STORM::fallback;

template<typename T, typename Grid>
class MonteCarloPhysics
{
public:
    using MCParticle = Particle<T>;

    MonteCarloPhysics(const Grid &grid, const std::shared_ptr<BoundaryCondition<T, Grid>> &boundary);

    virtual ~MonteCarloPhysics() = default;

    virtual void updateGridData(void);

    virtual std::vector<MCParticle> preStep(double fullDt) = 0;

    virtual StepResult step(MCParticle &particle, std::vector<MCParticle> &particlesToAdd) = 0;

    virtual void onBoundaryResult(const MCParticle &, ParticleStatus, bool)
    {}

    virtual void postStep(const std::vector<MCParticle> &particles, double fullDt) = 0;

    virtual size_t getRandomWalkStepCount() const { return 0; }
    virtual size_t getDDMCStepCount() const { return 0; }
    virtual size_t getDDMCLeakCount() const { return 0; }
    virtual size_t getDDMCCensusCount() const { return 0; }
    virtual size_t getDDMCUpscatterCount() const { return 0; }
    virtual size_t getDDMCFallbackCount() const { return 0; }
    virtual std::string getAccelerationDebugInfo(size_t, double) const { return std::string(); }

protected:
    const Grid &grid;
    std::shared_ptr<BoundaryCondition<T, Grid>> boundary;

    std::tuple<size_t, dt_t, cell_index_t> getIntersectionDetails(MCParticle &particle);

    struct FlatGridData
    {
        std::vector<std::size_t> cellFaceOffsets;
        std::vector<T> cellCenters;
        std::vector<std::size_t> faceIndices;
        std::vector<T> normals;
        std::vector<T> pointsOnFaces;
        std::vector<double> facePlaneOffsets;
        std::vector<cell_index_t> nextCellIndices;
        std::vector<std::uint8_t> boundaryCrossings;
        std::vector<std::uint8_t> deviceBoundaryBehaviors;
    } gridData;
    std::size_t gridDataBuildGeneration_ =
        std::numeric_limits<std::size_t>::max();
};

template<typename T, typename Grid>
MonteCarloPhysics<T, Grid>::MonteCarloPhysics(const Grid &grid, const std::shared_ptr<BoundaryCondition<T, Grid>> &boundary)
    : grid(grid), boundary(boundary)
{}

template<typename T, typename Grid>
void MonteCarloPhysics<T, Grid>::updateGridData(void)
{
    size_t Ncells = this->grid.GetPointNo();
    const std::size_t buildGeneration = this->grid.GetBuildGeneration();
    if(buildGeneration == this->gridDataBuildGeneration_ &&
       this->gridData.cellFaceOffsets.size() == Ncells + 1)
    {
        return;
    }
    this->gridDataBuildGeneration_ = buildGeneration;

    this->gridData.cellFaceOffsets.assign(Ncells + 1, 0);
    this->gridData.cellCenters.resize(Ncells);
    std::size_t directedFaceCount = 0;
    for(std::size_t i = 0; i < Ncells; ++i)
    {
        directedFaceCount += this->grid.GetCellFaces(i).size();
        this->gridData.cellFaceOffsets[i + 1] = directedFaceCount;
    }
    this->gridData.faceIndices.resize(directedFaceCount);
    this->gridData.normals.resize(directedFaceCount);
    this->gridData.pointsOnFaces.resize(directedFaceCount);
    this->gridData.facePlaneOffsets.resize(directedFaceCount);
    this->gridData.nextCellIndices.resize(directedFaceCount);
    this->gridData.boundaryCrossings.resize(directedFaceCount);
    this->gridData.deviceBoundaryBehaviors.resize(directedFaceCount);

    for(std::size_t i = 0; i < Ncells; ++i)
    {
        this->gridData.cellCenters[i] = this->grid.GetMeshPoint(i);
        const auto &faces = this->grid.GetCellFaces(i);
        std::size_t directedFace = this->gridData.cellFaceOffsets[i];
        for(const std::size_t faceIdx : faces)
        {
            T normalTowardsCenterOfCell = this->grid.Normal(faceIdx);
            if(ScalarProd(normalTowardsCenterOfCell, this->grid.GetMeshPoint(i) - this->grid.FaceCM(faceIdx)) < 0)
            {
                normalTowardsCenterOfCell *= -1;
            }
            const auto &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            const std::size_t nextCell = neighbors.first == i ? neighbors.second : neighbors.first;
            this->gridData.faceIndices[directedFace] = faceIdx;
            const T normal = normalize(normalTowardsCenterOfCell);
            const T pointOnFace = this->grid.FaceCM(faceIdx);
            this->gridData.normals[directedFace] = normal;
            this->gridData.pointsOnFaces[directedFace] = pointOnFace;
            this->gridData.facePlaneOffsets[directedFace] =
                ScalarProd(pointOnFace, normal);
            this->gridData.nextCellIndices[directedFace] = static_cast<cell_index_t>(nextCell);
            const bool boundaryCrossing = this->grid.IsPointOutsideBox(nextCell);
            this->gridData.boundaryCrossings[directedFace] = boundaryCrossing;
#ifdef STORM_WITH_TRACING_HISTORY
            this->gridData.deviceBoundaryBehaviors[directedFace] =
                static_cast<std::uint8_t>(DeviceBoundaryFaceBehavior::HostOnly);
#else
            this->gridData.deviceBoundaryBehaviors[directedFace] =
                static_cast<std::uint8_t>(
                    boundaryCrossing && this->boundary
                        ? this->boundary->getDeviceBoundaryFaceBehavior(
                              faceIdx, i, nextCell)
                        : DeviceBoundaryFaceBehavior::HostOnly);
#endif
            ++directedFace;
        }
    }
}

template<typename T, typename Grid>
inline std::tuple<size_t, dt_t, cell_index_t> MonteCarloPhysics<T, Grid>::getIntersectionDetails(MCParticle &particle)
{
    using coord_t = typename T::coord_type;
    const std::size_t cellIndex = static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex + 1 >= this->gridData.cellFaceOffsets.size())
    {
        StormError eo("MonteCarloPhysics::getIntersectionDetails: invalid cell index");
        eo.addEntry("Cell index", cellIndex);
        eo.addEntry("Particle", particle);
        throw eo;
    }

    std::size_t bestDirectedFace = std::numeric_limits<std::size_t>::max();
    coord_t bestTime = std::numeric_limits<coord_t>::max();
    const double velocityTolerance = EPSILON * fastabs(particle.velocity);
    const std::size_t begin = this->gridData.cellFaceOffsets[cellIndex];
    const std::size_t end = this->gridData.cellFaceOffsets[cellIndex + 1];
    for(std::size_t directedFace = begin; directedFace < end; ++directedFace)
    {
        const T &normal = this->gridData.normals[directedFace];
        const double normalVelocity = ScalarProd(normal, particle.velocity);
        if(normalVelocity >= -velocityTolerance)
            continue;
        const coord_t time =
            (this->gridData.facePlaneOffsets[directedFace] -
             ScalarProd(particle.location, normal)) /
            normalVelocity;
        if(time > 0 && time < bestTime)
        {
            bestTime = time;
            bestDirectedFace = directedFace;
        }
    }

    if(bestDirectedFace == std::numeric_limits<std::size_t>::max())
    {
        StormError eo("MonteCarloPhysics::getIntersectionDetails: no face intersection found");
        eo.addEntry("Particle", particle);
        eo.addEntry("Cell index", cellIndex);
        if(this->grid.IsPointOutsideBox(particle.location))
            eo.addEntry("Particle outside domain", true);
        else
            eo.addEntry("Real containing cell", this->grid.GetContainingCell(particle.location));
#ifdef STORM_WITH_TRACING_HISTORY
        particle.addTracingHistoryToError(eo);
#endif
        throw eo;
    }

    const std::size_t face = this->gridData.faceIndices[bestDirectedFace];
    return std::make_tuple(
        face, static_cast<dt_t>(bestTime),
        this->gridData.nextCellIndices[bestDirectedFace]);
}

} // namespace STORM

#endif // STORM_MONTE_CARLO_PHYSICS_HPP
