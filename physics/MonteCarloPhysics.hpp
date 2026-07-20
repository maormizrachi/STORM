#ifndef STORM_MONTE_CARLO_PHYSICS_HPP
#define STORM_MONTE_CARLO_PHYSICS_HPP

#include <memory>
#include <string>
#include <tuple>
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
    using MCParticle = Particle<T, Grid>;

    MonteCarloPhysics(const Grid &grid, const std::shared_ptr<BoundaryCondition<T, Grid>> &boundary);

    virtual ~MonteCarloPhysics() = default;

    virtual void updateGridData(void);

    virtual std::vector<MCParticle> preStep(double fullDt) = 0;

    virtual StepResult<T, Grid> step(MCParticle &particle, std::vector<MCParticle> &particlesToAdd) = 0;

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

    std::tuple<size_t, dt_t, size_t> getIntersectionDetails(MCParticle &particle);

    struct
    {
        std::vector<std::vector<T>> normalsOfCells;
        std::vector<std::vector<T>> pointsOnFaces;
    } gridData;
};

template<typename T, typename Grid>
MonteCarloPhysics<T, Grid>::MonteCarloPhysics(const Grid &grid, const std::shared_ptr<BoundaryCondition<T, Grid>> &boundary)
    : grid(grid), boundary(boundary)
{}

template<typename T, typename Grid>
void MonteCarloPhysics<T, Grid>::updateGridData(void)
{
    size_t Ncells = this->grid.GetPointNo();

    this->gridData.normalsOfCells = std::vector<std::vector<T>>(Ncells);
    this->gridData.pointsOnFaces = std::vector<std::vector<T>>(Ncells);

    for(size_t i = 0; i < Ncells; i++)
    {
        std::vector<T> &normals = this->gridData.normalsOfCells[i];
        std::vector<T> &onFaces = this->gridData.pointsOnFaces[i];
        const auto &faces = this->grid.GetCellFaces(i);
        for(const size_t &faceIdx : faces)
        {
            T normalTowardsCenterOfCell = this->grid.Normal(faceIdx);
            if(ScalarProd(normalTowardsCenterOfCell, this->grid.GetMeshPoint(i) - this->grid.FaceCM(faceIdx)) < 0)
            {
                normalTowardsCenterOfCell *= -1;
            }
            normals.push_back(normalize(normalTowardsCenterOfCell));
            onFaces.push_back(this->grid.FaceCM(faceIdx));
        }
    }
}

template<typename T, typename Grid>
inline std::tuple<size_t, dt_t, size_t> MonteCarloPhysics<T, Grid>::getIntersectionDetails(MCParticle &particle)
{
    size_t cellIndex = particle.cellIndex;
    const std::vector<T> &normalsOfFaces = this->gridData.normalsOfCells[cellIndex];
    const std::vector<T> &pointsOnFaces = this->gridData.pointsOnFaces[cellIndex];
    auto [faceIntersect, timeIntersect] = particle.distanceToNearestFace(this->grid, normalsOfFaces, pointsOnFaces);
    assert(faceIntersect < this->grid.GetTotalFacesNumber());
    assert(timeIntersect >= 0);
    const std::pair<size_t, size_t> &cellNeighbors = this->grid.GetFaceNeighbors(faceIntersect);
    assert(particle.cellIndex == cellNeighbors.first or particle.cellIndex == cellNeighbors.second);
    size_t nextCellIndex = (cellNeighbors.first == particle.cellIndex) ? cellNeighbors.second : cellNeighbors.first;
    return std::make_tuple(faceIntersect, timeIntersect, nextCellIndex);
}

} // namespace STORM

#endif // STORM_MONTE_CARLO_PHYSICS_HPP
