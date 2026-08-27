#ifndef STORM_MONTE_CARLO_TRANSPORT_CORE_HPP
#define STORM_MONTE_CARLO_TRANSPORT_CORE_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

namespace STORM {

namespace monte_carlo_detail {

template<typename ParticleStore, typename = void>
struct HasLowerParticleStoreSize : std::false_type
{};

template<typename ParticleStore>
struct HasLowerParticleStoreSize<
    ParticleStore,
    std::void_t<decltype(std::declval<const ParticleStore &>().size())>>
    : std::true_type
{};

template<typename ParticleStore, typename = void>
struct HasUpperParticleStoreSize : std::false_type
{};

template<typename ParticleStore>
struct HasUpperParticleStoreSize<
    ParticleStore,
    std::void_t<decltype(std::declval<const ParticleStore &>().Size())>>
    : std::true_type
{};

template<typename ParticleStore>
struct ParticleStoreSize
{
    static std::size_t Get(const ParticleStore &store)
    {
        return Get(store, HasLowerParticleStoreSize<ParticleStore>{});
    }

private:
    static std::size_t Get(const ParticleStore &store, std::true_type)
    {
        return store.size();
    }

    static std::size_t Get(const ParticleStore &store, std::false_type)
    {
        static_assert(HasUpperParticleStoreSize<ParticleStore>::value,
                      "ParticleStore must provide size() or Size()");
        return store.Size();
    }
};

} // namespace monte_carlo_detail

// Shared HandleAll component. A processor owns the physics/event semantics for
// one particle and must return only after that particle has left local
// ownership. The store/executor own batch acquisition and consumption.
template<typename LocalTransportExecutor>
class MonteCarloTransportCore
{
public:
    explicit MonteCarloTransportCore(LocalTransportExecutor &executor) :
        executor(executor)
    {}

    template<typename ParticleStore, typename ProcessParticle>
    bool HandleAll(ParticleStore &store, ProcessParticle &processParticle,
                   size_t maximumParticles)
    {
        if(monte_carlo_detail::ParticleStoreSize<ParticleStore>::Get(store) == 0)
        {
            return true;
        }

        this->executor.Execute(store, maximumParticles, processParticle);
        return monte_carlo_detail::ParticleStoreSize<ParticleStore>::Get(store) == 0;
    }

private:
    LocalTransportExecutor &executor;
};

template<typename ParticleStore, typename LocalTransportExecutor, typename ProcessParticle>
bool HandleMonteCarloParticles(ParticleStore &store,
                               LocalTransportExecutor &executor,
                               ProcessParticle &processParticle,
                               size_t maximumParticles)
{
    MonteCarloTransportCore<LocalTransportExecutor> core(executor);
    return core.HandleAll(store, processParticle, maximumParticles);
}

} // namespace STORM

#endif // STORM_MONTE_CARLO_TRANSPORT_CORE_HPP
