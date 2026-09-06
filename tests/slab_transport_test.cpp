#include "gpu/DeviceParticle.hpp"
#include "gpu/GreyIMCKernel.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace STORM;
using namespace STORM::gpu;

struct Result
{
    DeviceParticle particle;
    double deposited = 0.0, integrated = 0.0;
    int events = 0, error = 0;
};

STORM_GPU_INLINE_FUNCTION
Result Run(const bool slab, const double vx)
{
    // Absorbing, nonscattering material admits a closed-form solution.
    // The narrow box causes many explicit y/z reflections.
    DeviceVec3 center[] = {{0.5, 0.005, 0.005}};
    DeviceVec3 normals[] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    std::size_t offsets[] = {0,6};
    double planes[] = {0,-1,0,-0.01,0,-0.01};
    cell_index_t neighbors[] = {1,2,3,4,5,6};
    std::uint8_t boundary[] = {1,1,1,1,1,1};
    std::uint8_t behavior[6];
    for(int i=0;i<6;++i) behavior[i] = static_cast<std::uint8_t>(DeviceBoundaryFaceBehavior::ReflectingRigid);
    double absorption[] = {0.02}, scattering[] = {0}, fleck[] = {1};
    Result output;
    GreyIMCViews<DeviceVec3> views;
    views.grid.cellCount = 1;
    views.grid.cellCenters = center;
    views.grid.cellFaceOffsets = offsets;
    views.grid.normals = normals;
    views.grid.facePlaneOffsets = planes;
    views.grid.nextCellIndices = neighbors;
    views.grid.boundaryCrossings = boundary;
    views.grid.deviceBoundaryBehaviors = behavior;
    views.grid.slabTransport = slab;
    views.grid.slabUpperY = views.grid.slabUpperZ = 0.01;
    views.absorptionOpacities = absorption;
    views.scatteringOpacities = scattering;
    views.fleckFactors = fleck;
    views.speedOfLight = 1;
    views.pendingMaterialEnergy = &output.deposited;
    views.pendingRadiationEnergy = &output.integrated;
    auto &p = output.particle;
    p.location = {0.5,0.002,0.004};
    p.velocity = {vx,-0.4,transport::Sqrt(0.84-vx*vx)};
    p.weight = p.initialWeight = 2;
    p.timeLeft = 0.9;
    p.rngKey = 17;
    for(int i=0;i<1000;++i)
    {
        const auto result = transport::AdvanceIMC(p, views);
        ++output.events;
        if(result.error != TransportError::None) { output.error = 1; break; }
        if(result.step.change == ParticleStatus::DONE) break;
    }
    return output;
}

void Check(const Result &slab, const Result &explicitWalls, const double vx)
{
    const double weight = 2*std::exp(-0.02*0.9);
    auto close = [](double a,double b,double tolerance) {
        if(std::abs(a-b)>tolerance) throw std::runtime_error("Slab analytic/differential comparison failed");
    };
    if(slab.error || explicitWalls.error || slab.events != 1 || explicitWalls.events < 30)
        throw std::runtime_error("Slab did not bypass transverse events");
    close(slab.particle.weight,weight,1e-12);
    close(slab.deposited+slab.particle.weight,2,1e-12);
    close(slab.integrated,(2-weight)/0.02,1e-12);
    close(slab.particle.location.x,0.5+vx*0.9,1e-12);
    close(slab.particle.weight,explicitWalls.particle.weight,1e-12);
    close(slab.deposited,explicitWalls.deposited,1e-12);
    close(slab.integrated,explicitWalls.integrated,1e-12);
    close(slab.particle.location.y,explicitWalls.particle.location.y,1e-6);
    close(slab.particle.location.z,explicitWalls.particle.location.z,1e-6);
    close(slab.particle.velocity.y,explicitWalls.particle.velocity.y,1e-12);
    close(slab.particle.velocity.z,explicitWalls.particle.velocity.z,1e-12);
}

int main(int argc,char **argv)
{
#ifdef STORM_WITH_GPU
    Kokkos::initialize(argc,argv);
    {
        Kokkos::View<Result*> results("slab_test",4);
        Kokkos::parallel_for("slab_analytic_test",4,KOKKOS_LAMBDA(int i) {
            results(i)=Run(i%2==0,i<2 ? 0.3 : 0.0);
        });
        const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},results);
        Check(host(0),host(1),0.3);
        Check(host(2),host(3),0.0);
    }
    Kokkos::finalize();
#else
    (void)argc; (void)argv;
    Check(Run(true,0.3),Run(false,0.3),0.3);
    Check(Run(true,0),Run(false,0),0);
#endif
    // Multiple periods, negative positions, and exact-wall directions.
    double p=-0.03,v=-1;
    FoldSlabCoordinate(p,v,0,0.01);
    if(std::abs(p-0.01)>1e-14 || v>0) throw std::runtime_error("Negative fold failed");
    p=0;v=-1;FoldSlabCoordinate(p,v,0,0.01);
    if(p!=0 || v!=1) throw std::runtime_error("Wall fold failed");
    FlatGridView<DeviceVec3> emptyGrid;
    emptyGrid.slabTransport=1;
    if(FindIntersection(DeviceParticle{},emptyGrid,1).valid)
        throw std::runtime_error("Slab accepted an invalid cell");
    std::cout << "Slab analytic conservation and explicit-wall comparisons passed\n";
}
