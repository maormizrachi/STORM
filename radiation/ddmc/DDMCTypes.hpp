#ifndef STORM_DDMC_TYPES_HPP
#define STORM_DDMC_TYPES_HPP

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace STORM::ddmc {

enum class FaceKind : unsigned char
{
    Internal,
    InterfaceToIMC,
    ReflectingBoundary,
    ThermalizingBoundary
};

enum class EligibilityReason : unsigned char
{
    Uncomputed,
    Eligible,
    InvalidGeometry,
    InvalidThermalState,
    OpticallyThin,
    NoDiffusionCoefficient,
    BoundaryExcluded,
    NoLeakage
};

template<typename PointT>
struct FaceLeak
{
    std::size_t faceIndex = std::numeric_limits<std::size_t>::max();
    std::size_t nextCellIndex = std::numeric_limits<std::size_t>::max();
    FaceKind kind = FaceKind::InterfaceToIMC;
    double rate = 0.0;
    double internalRate = 0.0;
    double boundaryRate = 0.0;
    double ddmcRate = 0.0;
    double transportRate = 0.0;
    double ddmcFraction = 0.0;
    double area = 0.0;
    double sourceDistanceToFace = 0.0;
    double targetDistanceToFace = 0.0;
    double conductance = 0.0;
    double sourceBandMass = 1.0;
    double commonBandMass = 1.0;
    std::size_t targetGroupCutoff = 0;
    bool targetDDMCEligible = false;
    PointT outwardNormal{};
};

template<typename PointT>
struct CellData
{
    bool eligible = false;
    EligibilityReason eligibilityReason = EligibilityReason::Uncomputed;
    bool boundaryExcluded = false;
    bool observerExcluded = false;
    double sigmaA = 0.0;
    double sigmaT = 0.0;
    double sigmaEnergyAbs = 0.0;
    double sigmaMomentum = 0.0;
    double sigmaDiffusion = 0.0;
    double sigmaParticleGate = 0.0;
    double sigmaGroupExit = 0.0;
    double diffusionCoefficient = 0.0;
    double gamma = 1.0;
    double velocityDivergence = 0.0;
    double maxFaceVelocityJumpOverC = 0.0;
    double totalLeakRate = 0.0;
    double faceAreaSum = 0.0;
    std::size_t groupCutoff = 0;
    std::vector<FaceLeak<PointT>> faceLeaks;
    // Symmetric 3-D face-normal moment matrix, stored as xx, xy, xz, yy,
    // yz, zz.  It is optional for the core transport path but gives the
    // momentum-coupling phase a stable place to accumulate geometry.
    std::array<double, 6> fluxMatrix{};
};

} // namespace STORM::ddmc

#endif // STORM_DDMC_TYPES_HPP
