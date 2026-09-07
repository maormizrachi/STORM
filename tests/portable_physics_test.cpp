#include "gpu/DeviceParticle.hpp"
#include "gpu/GreyIMCKernel.hpp"
#include "radiation/source/SourceCore.hpp"
#ifdef STORM_WITH_GPU
#include "gpu/SourceDeviceEmit.hpp"
#endif
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace STORM;
using namespace STORM::gpu;

struct PhysicsResult
{
    double time, survivingWeight, frequency, tableOpacity, analyticOpacity;
    double labGroup0, labGroup1;
};

// Local arrays deliberately live inside the kernel: the same checks execute
// on a CPU and in CUDA/HIP without capturing any host pointers.
STORM_GPU_INLINE_FUNCTION
PhysicsResult EvaluatePhysics(const std::size_t index)
{
    DeviceVec3 center[] = {{0,0,0}};
    double boundaries[] = {1e-10, 1e-9, 2e-9};
    double cdf[] = {0,1,2};
    double kT[] = {1e-9};
    source::SampleViews<DeviceVec3> sourceViews;
    sourceViews.cellCenters = center;
    sourceViews.cellCount = 1;
    sourceViews.groupCount = 2;
    sourceViews.energyBoundaries = boundaries;
    sourceViews.thermalEmissionCdf = cdf;
    sourceViews.thermalKT = kT;
    sourceViews.thermalFrequencyLaw = ThermalFrequencyLaw::BoseEinstein0;
    sourceViews.fullDt = 2;
    sourceViews.speedOfLight = 1;
    sourceViews.sampleFrequency = 1;
    source::EmittedScalars scalars;
    DeviceVec3 location, velocity;
    source::EmitThermalPacket(sourceViews, 0, source::MakeSourceRngKey(17,0,index),
                              1, location, velocity, scalars);
    PhysicsResult result{};
    result.time = scalars.timeLeft;
    result.survivingWeight = transport::Exp(-3 * scalars.timeLeft);
    result.frequency = scalars.frequency;

    GreyIMCViews<DeviceVec3> views;
    double absorption[] = {20}, scattering[] = {0}, fleck[] = {1};
    double scale[] = {20e-30}, table[] = {20,100};
    double groups[] = {0,0};
    views.absorptionOpacities = absorption;
    views.scatteringOpacities = scattering;
    views.fleckFactors = fleck;
    views.energyBoundaries = boundaries;
    views.spectralAbsorptionScale = scale;
    views.groupCount = 2;
    views.pendingGroupRadiationEnergy = groups;
    DeviceParticle p;
    p.frequency = 1.1e-9;
    transport::SpectralTableOpacityPolicy policy;
    result.analyticOpacity = policy.Evaluate(p,views,0,2e-10).absorption;
    views.groupAbsorptionOpacities = table;
    auto opacity = policy.Evaluate(p,views,0,9e-10);
    result.tableOpacity = opacity.absorption;
    // The interaction uses comoving group 0, but the lab diagnostic is group 1.
    policy.TallyGroupRadiation(p,views,0,opacity,3);
    result.labGroup0 = groups[0];
    result.labGroup1 = groups[1];
    return result;
}

void Require(bool condition, const char *message)
{
    if(!condition) throw std::runtime_error(message);
}

int main(int argc, char **argv)
{
    constexpr std::size_t count = 65536;
#ifdef STORM_WITH_GPU
    Kokkos::initialize(argc,argv);
    {
    // Exercise actual host-to-device snapshot uploads, including replacement
    // of a previously synchronized allocation (not just stack-backed views).
    GreyIMCData data;
    Kokkos::View<double*> uploaded("uploaded_opacity",1);
    for(int pass=0; pass<2; ++pass)
    {
        data.UploadGrid(std::vector<std::size_t>{0,0},
            std::vector<DeviceVec3>{{0,0,0}}, std::vector<cell_id_t>{0},
            std::vector<DeviceVec3>{}, std::vector<double>{},
            std::vector<cell_index_t>{}, std::vector<std::uint8_t>{},
            std::vector<std::uint8_t>{});
        data.UploadTables({20}, {0}, {1});
        data.UploadSpectral({1e-10,1e-9,2e-9}, {20e-30}, {0,1,2},
            {20.0+pass,100}, {1e-9}, ThermalFrequencyLaw::BoseEinstein0);
        const auto views = data.Views(1,false,false,false,false);
        Kokkos::parallel_for("check_uploaded_opacity",1,KOKKOS_LAMBDA(int) {
            DeviceParticle p;
            uploaded(0) = transport::SpectralTableOpacityPolicy{}.Evaluate(p,views,0,2e-10).absorption;
        });
        const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},uploaded);
        Require(values(0)==20+pass,"Device opacity snapshot upload changed the table");
    }
    Kokkos::View<PhysicsResult*> device("physics_results",count);
    Kokkos::parallel_for("portable_physics",count,KOKKOS_LAMBDA(std::size_t i) {
        device(i) = EvaluatePhysics(i);
    });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},device);
#else
    (void)argc; (void)argv;
#endif
    double meanTime=0, survival=0, lowQuarter=0;
    std::size_t lowGroup=0;
    const double lowBounds[] = {1e-10,1e-9};
    const double quarterEnergy = ddmc::SampleBoseEinstein0FrequencyInGroup(
        lowBounds,1,0,1e-9,0.25);
    for(std::size_t i=0;i<count;++i)
    {
        const auto expected = EvaluatePhysics(i);
#ifdef STORM_WITH_GPU
        const auto actual = host(i);
        Require(std::abs(actual.time-expected.time)<1e-12,"CPU/GPU emission times differ");
        Require(std::abs(actual.frequency-expected.frequency)<1e-20,"CPU/GPU source spectra differ");
#else
        const auto actual = expected;
#endif
        Require(actual.time>0 && actual.time<2,"Source time is not inside timestep");
        Require(std::abs(actual.tableOpacity-20)<1e-12,"Table opacity was replaced by inverse cube");
        Require(std::abs(actual.analyticOpacity-2.5)<1e-12,"Inverse-cube opacity changed");
        Require(actual.labGroup0==0 && actual.labGroup1==3,"Group tally is not in lab frame");
        meanTime += actual.time/count;
        survival += actual.survivingWeight/count;
        if(actual.frequency<1e-9){++lowGroup; if(actual.frequency<quarterEnergy) ++lowQuarter;}
    }
    Require(std::abs(meanTime-1)<0.012,"Thermal emission time is biased");
    Require(std::abs(survival-(1-std::exp(-6))/6)<0.004,"Uniform emitting absorber solution changed");
    Require(std::abs(double(lowGroup)/count-0.5)<0.012,"Emission group probabilities changed");
    Require(std::abs(lowQuarter/lowGroup-0.25)<0.015,"Within-group source distribution changed");
#ifdef STORM_WITH_GPU
    }
    Kokkos::finalize();
#endif
    std::cout << "Portable source, absorption, and lab-frame tally checks passed\n";
}
