#include "gpu/KokkosLocalTransportExecutor.hpp"
#include <iostream>
#include <initializer_list>

using namespace STORM;
using namespace STORM::gpu;

template<typename T>
Kokkos::View<T*> Upload(std::initializer_list<T> values)
{
    Kokkos::View<T*> result("overlap_fixture",values.size());
    auto host=Kokkos::create_mirror_view(result);
    std::size_t i=0;
    for(const auto &value:values) host(i++)=value;
    Kokkos::deep_copy(result,host);
    return result;
}

int main(int argc,char **argv)
{
    Kokkos::initialize(argc,argv);
    {
        auto centers=Upload<DeviceVec3>({{0.5,0.5,0.5}});
        auto normals=Upload<DeviceVec3>({{1,0,0},{-1,0,0}});
        auto offsets=Upload<std::size_t>({0,2});
        auto planes=Upload<double>({0,-1});
        auto neighbors=Upload<cell_index_t>({1,2});
        auto boundary=Upload<std::uint8_t>({0,0});
        auto zeros=Upload<double>({0});
        auto ones=Upload<double>({1});
        auto tallies=Upload<double>({0});
        GreyIMCViews<DeviceVec3> views;
        views.grid.cellCount=1;
        views.grid.cellCenters=centers.data();
        views.grid.cellFaceOffsets=offsets.data();
        views.grid.normals=normals.data();
        views.grid.facePlaneOffsets=planes.data();
        views.grid.nextCellIndices=neighbors.data();
        views.grid.boundaryCrossings=boundary.data();
        views.absorptionOpacities=zeros.data();
        views.scatteringOpacities=zeros.data();
        views.fleckFactors=ones.data();
        views.speedOfLight=1;
        views.depositMaterialEnergy=0;
        views.pendingRadiationEnergy=tallies.data();
        KokkosLocalTransportExecutor executor(64,true);
        std::vector<bool> seen;
        std::size_t consumed=0;
        auto consume=[&](CompletedBatch &batch) {
            for(const auto &remote:batch.remotes)
            {
                const auto id=remote.cold.id;
                if(id>=seen.size() || seen[id] || remote.particle.location.x!=1 ||
                   remote.particle.timeLeft!=0.5 || remote.result.step.nextCellIndex!=2)
                    throw std::runtime_error("Corrupt or duplicate pipelined packet");
                seen[id]=true;
                ++consumed;
            }
        };
        for(std::size_t batch=0;batch<12;++batch)
        {
            const std::size_t count=batch%3==0 ? 4096*(batch+1) : 7;
            const std::size_t first=seen.size();
            seen.resize(first+count,false);
            const auto offset=executor.AllocateActiveSlots(count);
            auto packets=executor.ActivePackets();
            auto cold=executor.ActiveColdPackets();
            Kokkos::parallel_for("overlap_test_packets",count,KOKKOS_LAMBDA(std::size_t i) {
                DeviceParticle p;
                p.location={0.5,0.5,0.5};p.velocity={1,0,0};
                p.timeLeft=1;p.weight=p.initialWeight=1;p.rngKey=i+1;
                packets(offset+i)=p;
                DeviceParticleCold c;c.id=first+i;cold(offset+i)=c;
            });
            auto result=executor.AdvanceWave(views,[](){},1,1,consume);
            if(!result.remotes.empty() || !executor.HasPendingRemoteCopy() || !executor.DeviceBusy())
                throw std::runtime_error("Remote wave was not deferred");
            bool resetRejected=false;
            try { executor.Reset(); } catch(const std::runtime_error &) { resetRejected=true; }
            if(!resetRejected) throw std::runtime_error("Reset lost in-flight remotes");
        }
        executor.AdvanceWave(views,[](){},1,1,consume);
        if(executor.DeviceBusy() || consumed!=seen.size() ||
           executor.Metrics().pipelinedRemoteCount!=consumed)
            throw std::runtime_error("Final remote drain lost packets");
        executor.Reset();
        std::cout << "Pipelined remote reuse/resize/final-drain passed: " << consumed << " packets\n";
    }
    Kokkos::finalize();
}
