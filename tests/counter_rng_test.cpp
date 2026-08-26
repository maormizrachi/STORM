#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#ifdef STORM_WITH_MPI
#include <mpi_utils/serialize/Serializer.hpp>
#endif

#include "particle/Particle.hpp"
#include "population/CombPopulationControl.hpp"
#include "utils/CounterRNG.hpp"

namespace {

void require(bool condition, const char *message)
{
    if(!condition)
        throw std::runtime_error(message);
}

struct TestPoint
{
    using coord_type = double;

    double x = 0.0;

    explicit TestPoint(double value = 0.0) : x(value) {}
};

TestPoint operator+(const TestPoint &left, const TestPoint &right)
{
    return TestPoint(left.x + right.x);
}

TestPoint operator-(const TestPoint &left, const TestPoint &right)
{
    return TestPoint(left.x - right.x);
}

TestPoint operator*(double factor, const TestPoint &point)
{
    return TestPoint(factor * point.x);
}

std::ostream &operator<<(std::ostream &stream, const TestPoint &point)
{
    return stream << point.x;
}

struct OneCellGrid
{
    std::size_t GetPointNo() const { return 1; }
    bool IsPointOutsideBox(const TestPoint &) const { return false; }
    TestPoint GetMeshPoint(std::size_t) const { return TestPoint(0.0); }
    bool IsPointInCell(const TestPoint &, std::size_t) const { return true; }
};

using TestParticle = STORM::Particle<TestPoint>;

TestParticle makeParticle(std::size_t id, double weight)
{
    TestParticle particle(id, TestPoint(0.0), TestPoint(1.0), 1.0);
    particle.cellIndex = 0;
    particle.weight = weight;
    particle.initialWeight = weight;
    particle.rngKey = 0x123456789abcdef0ULL + id;
    particle.rngCounter = 17 + id;
#ifdef STORM_WITH_MPI
    particle.rank = 3;
#endif
    return particle;
}

void requireNewIdentity(const TestParticle &particle)
{
    require(particle.id == std::numeric_limits<std::size_t>::max(),
            "new particle ID was not invalidated");
    require(particle.rngKey == std::numeric_limits<std::uint64_t>::max(),
            "new particle retained its parent's RNG key");
    require(particle.rngCounter == 0,
            "new particle retained its parent's RNG counter");
#ifdef STORM_WITH_MPI
    require(particle.rank == std::numeric_limits<STORM::rank_t>::max(),
            "new particle rank was not invalidated");
#endif
}

void testCounterRNG()
{
    constexpr std::uint64_t seed = 0x0123456789abcdefULL;
    const std::uint64_t key = STORM::CounterRNG::makeKey(seed, 2, 7);

    require(key == STORM::CounterRNG::makeKey(seed, 2, 7),
            "counter RNG key is not deterministic");
    require(key != STORM::CounterRNG::makeKey(seed, 7, 2),
            "creation rank and particle ID are interchangeable");

    for(std::uint64_t counter = 0; counter < 64; ++counter)
    {
        require(STORM::CounterRNG::next(key, counter) ==
                    STORM::CounterRNG::next(key, counter),
                "counter RNG draw is not deterministic");
        const double sample = STORM::CounterRNG::unitOpen(key, counter);
        require(sample > 0.0 && sample < 1.0,
                "counter RNG sample left the open unit interval");
    }

    require(STORM::counter_rng_detail::counterBitsToUnitOpen(0) == 0x1.0p-53,
            "zero bits do not map to the expected positive endpoint");
    require(STORM::counter_rng_detail::counterBitsToUnitOpen(
                std::numeric_limits<std::uint64_t>::max()) ==
                0x1.fffffffffffffp-1,
            "maximum bits do not map below one");

    std::uint64_t counter = 9;
    STORM::ParticleCounterEngine engine(key, counter);
    require(engine() == STORM::CounterRNG::next(key, 9),
            "particle RNG engine did not continue its stream");
    require(counter == 10, "particle RNG engine did not advance its counter");
}

void testCloneWithNewIdentity()
{
    const TestParticle source = makeParticle(5, 3.0);
    const TestParticle clone = STORM::cloneParticleWithNewIdentity(source);

    require(clone.id == std::numeric_limits<std::size_t>::max() &&
                clone.rngKey == std::numeric_limits<std::uint64_t>::max() &&
                clone.rngCounter == 0,
            "new-identity clone retained its parent's identity or RNG stream");
    require(clone.weight == source.weight && clone.cellIndex == source.cellIndex,
            "new-identity clone changed transport state");
#ifdef STORM_WITH_MPI
    require(clone.rank == source.rank,
            "new-identity clone changed routing ownership");
#endif
}

void testCombPopulationControl()
{
    const OneCellGrid grid;
    const TestParticle source = makeParticle(11, 8.0);

    STORM::CombPopulationControl<TestPoint, OneCellGrid> splitter(grid, 4, 1.0);
    const std::vector<TestParticle> split = splitter.activate({source});
    require(split.size() == 4, "comb split produced the wrong daughter count");
    for(const TestParticle &particle : split)
        requireNewIdentity(particle);

    std::set<std::uint64_t> daughterKeys;
    for(std::size_t id = 0; id < split.size(); ++id)
        daughterKeys.insert(STORM::CounterRNG::makeKey(1234, 0, id));
    require(daughterKeys.size() == split.size(),
            "new daughter IDs did not produce unique RNG keys");

    STORM::CombPopulationControl<TestPoint, OneCellGrid> keeper(grid, 1, 1.0);
    const std::vector<TestParticle> kept = keeper.activate({source});
    require(kept.size() == 1, "unsplit comb packet was not preserved");
    require(kept.front().id == source.id &&
                kept.front().rngKey == source.rngKey &&
                kept.front().rngCounter == source.rngCounter,
            "unsplit comb packet did not continue its RNG stream");

    std::vector<TestParticle> crowded;
    for(std::size_t id = 0; id < 4; ++id)
        crowded.push_back(makeParticle(id, 1.0));
    STORM::CombPopulationControl<TestPoint, OneCellGrid> resampler(grid, 2, 1.0);
    const std::vector<TestParticle> resampled = resampler.activate(crowded);
    require(resampled.size() == 2, "comb resampling produced the wrong count");
    for(const TestParticle &particle : resampled)
        requireNewIdentity(particle);
}

void testStratifiedCombPopulationControl()
{
    const OneCellGrid grid;
    const auto classifier = [](const TestParticle &) { return std::size_t{0}; };
    const TestParticle source = makeParticle(21, 8.0);

    STORM::StratifiedCombPopulationControl<TestPoint, OneCellGrid> splitter(
        grid, 1, classifier, 4, 1.0, 1);
    const std::vector<TestParticle> split = splitter.activate({source});
    require(split.size() == 4,
            "stratified comb split produced the wrong daughter count");
    for(const TestParticle &particle : split)
        requireNewIdentity(particle);

    std::vector<TestParticle> crowded;
    for(std::size_t id = 0; id < 4; ++id)
        crowded.push_back(makeParticle(id, 1.0));
    STORM::StratifiedCombPopulationControl<TestPoint, OneCellGrid> resampler(
        grid, 1, classifier, 2, 1.0, 1);
    const std::vector<TestParticle> resampled = resampler.activate(crowded);
    require(resampled.size() == 2,
            "stratified comb resampling produced the wrong count");
    for(const TestParticle &particle : resampled)
        requireNewIdentity(particle);
}

#ifdef STORM_WITH_MPI
void testSerializedRNGContinuation()
{
    TestParticle source = makeParticle(31, 1.0);
    source.rngKey = STORM::CounterRNG::makeKey(4567, source.rank, source.id);
    source.rngCounter = 23;

    Serializer serializer;
    const std::size_t dumped = source.dump(&serializer);
    TestParticle restored;
    const std::size_t loaded = restored.load(&serializer, 0);

    require(dumped == loaded && loaded == serializer.size(),
            "particle serialization byte counts disagree");
    require(restored.rank == source.rank && restored.id == source.id &&
                restored.rngKey == source.rngKey &&
                restored.rngCounter == source.rngCounter,
            "particle serialization did not preserve its RNG stream");
}
#endif

} // namespace

int main()
{
    try
    {
        testCounterRNG();
        testCloneWithNewIdentity();
        testCombPopulationControl();
        testStratifiedCombPopulationControl();
#ifdef STORM_WITH_MPI
        testSerializedRNGContinuation();
#endif
    }
    catch(const std::exception &error)
    {
        std::cerr << "storm_counter_rng_test: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
