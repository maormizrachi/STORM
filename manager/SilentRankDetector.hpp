#ifndef STORM_SILENT_RANK_DETECTOR_HPP
#define STORM_SILENT_RANK_DETECTOR_HPP

#ifdef STORM_WITH_MPI

#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace STORM
{

// Per-rank transport telemetry collected through a one-sided window on rank 0.
//
// Every rank puts a fixed-size slot (a summary record followed by per-peer
// arrays) into the window about once a second. The transfer is one-sided, so
// a rank stuck inside a physics call simply leaves a stale slot behind
// instead of blocking a collective. Rank 0 sums the records and reports
//  (a) ranks whose loop iteration did not move since the previous report,
//  (b) a particle-conservation check
//          start + created - finished - queued - (sent - received)
//      against the live count the AmountManager tree holds at the root, and
//  (c) the rank pairs where particles or messages sent by A to B were never
//      received by B, with whether A and B list each other as neighbours.
//
// The window is created and freed collectively; do both at points every
// rank is guaranteed to reach. Rank 0 holds size * (record + 4 arrays of
// size words + neighbour bitmap) bytes: about 75 MB for 1536 ranks.
class RankTelemetry
{
public:
    struct Record
    {
        std::uint64_t iteration = 0;
        std::uint64_t start = 0;        // particles this rank began the step with (initial + generated)
        std::uint64_t created = 0;      // particles created during transport (splits etc.)
        std::int64_t finished = 0;      // REMOVE/DONE/boundary-removed on this rank
        std::uint64_t sent = 0;         // particles handed to the send buffers
        std::uint64_t received = 0;     // particles detached from other ranks' arrival queues
        std::uint64_t queued = 0;       // particles waiting in arrival/deferred queues right now
        std::int64_t treePending = 0;   // AmountManager local count not yet flushed to the parent
        std::uint64_t activeSends = 0;  // 1 when the engine reports outstanding sends and empty queues
        std::int64_t rootRemaining = 0; // rank 0 only: AmountManager global count
        std::uint64_t messagesSent = 0; // particle buffers dispatched (MPI messages)
        std::uint64_t messagesReceived = 0;
    };
    static constexpr int kRecordWords = sizeof(Record) / sizeof(std::uint64_t);
    static_assert(sizeof(Record) % sizeof(std::uint64_t) == 0, "Record must be a whole number of 64-bit words");

    // Per-peer arrays, each `size` words: particles sent to peer, particles
    // received from peer, messages sent to peer, messages received from peer.
    struct PeerArrays
    {
        std::vector<std::uint64_t> particlesSentTo;
        std::vector<std::uint64_t> particlesReceivedFrom;
        std::vector<std::uint64_t> messagesSentTo;
        std::vector<std::uint64_t> messagesReceivedFrom;
        std::vector<std::uint64_t> neighbourBits; // bit r set when r is in this rank's neighbour list
    };

    // The per-peer arrays cost size*size*32 bytes on rank 0 (75 MB for 1536
    // ranks) and 32*size bytes per rank per publish. They are only needed to
    // chase lost messages, so they are off unless STORM_TELEMETRY_PAIRS is set.
    static bool PairsEnabled()
    {
        const char *value = std::getenv("STORM_TELEMETRY_PAIRS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }

    explicit RankTelemetry(MPI_Comm comm) : comm_(comm), pairs_(PairsEnabled())
    {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
        bitmapWords_ = pairs_ ? static_cast<size_t>(size_ + 63) / 64 : 0;
        slotWords_ = static_cast<size_t>(kRecordWords) + (pairs_ ? 4 * static_cast<size_t>(size_) : 0) + bitmapWords_;
        const MPI_Aint bytes = (rank_ == 0) ? static_cast<MPI_Aint>(size_) * slotWords_ * sizeof(std::uint64_t) : 0;
        MPI_Win_allocate(bytes, sizeof(std::uint64_t), MPI_INFO_NULL, comm_, &slots_, &window_);
        if(rank_ == 0)
        {
            std::memset(slots_, 0, static_cast<size_t>(bytes));
            previousIteration_.assign(static_cast<size_t>(size_), 0);
        }
        staging_.assign(slotWords_, 0);
        // Make the zeroed slots visible before anyone puts into them.
        MPI_Win_fence(0, window_);
    }

    ~RankTelemetry()
    {
        MPI_Win_free(&window_);
    }

    RankTelemetry(const RankTelemetry &) = delete;
    RankTelemetry &operator=(const RankTelemetry &) = delete;

    int Size() const { return size_; }

    // Publish this rank's slot. Arrays shorter than `size` are zero-padded.
    void Publish(const Record &record, const PeerArrays &peers)
    {
        std::memcpy(staging_.data(), &record, sizeof(Record));
        size_t offset = static_cast<size_t>(kRecordWords);
        if(!pairs_)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, window_);
            MPI_Put(staging_.data(), static_cast<int>(slotWords_), MPI_UINT64_T, 0,
                    static_cast<MPI_Aint>(rank_) * static_cast<MPI_Aint>(slotWords_),
                    static_cast<int>(slotWords_), MPI_UINT64_T, window_);
            MPI_Win_unlock(0, window_);
            return;
        }
        auto copyArray = [&](const std::vector<std::uint64_t> &src, size_t words)
        {
            const size_t n = src.size() < words ? src.size() : words;
            if(n) std::memcpy(staging_.data() + offset, src.data(), n * sizeof(std::uint64_t));
            if(n < words) std::memset(staging_.data() + offset + n, 0, (words - n) * sizeof(std::uint64_t));
            offset += words;
        };
        const size_t n = static_cast<size_t>(size_);
        copyArray(peers.particlesSentTo, n);
        copyArray(peers.particlesReceivedFrom, n);
        copyArray(peers.messagesSentTo, n);
        copyArray(peers.messagesReceivedFrom, n);
        copyArray(peers.neighbourBits, bitmapWords_);

        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, window_);
        MPI_Put(staging_.data(), static_cast<int>(slotWords_), MPI_UINT64_T, 0,
                static_cast<MPI_Aint>(rank_) * static_cast<MPI_Aint>(slotWords_),
                static_cast<int>(slotWords_), MPI_UINT64_T, window_);
        MPI_Win_unlock(0, window_);
    }

    // Rank 0 only. Writes the report (several lines) to `out`. Returns the
    // number of silent ranks.
    int Report(std::string &out, size_t maxListed = 12)
    {
        out.clear();
        if(rank_ != 0)
        {
            return 0;
        }
        const size_t n = static_cast<size_t>(size_);
        std::vector<std::uint64_t> snapshot(n * slotWords_);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, window_);
        std::memcpy(snapshot.data(), slots_, snapshot.size() * sizeof(std::uint64_t));
        MPI_Win_unlock(0, window_);

        auto recordOf = [&](size_t r) { Record rec; std::memcpy(&rec, snapshot.data() + r * slotWords_, sizeof(Record)); return rec; };
        auto arrayOf = [&](size_t r, int which) { return snapshot.data() + r * slotWords_ + kRecordWords + static_cast<size_t>(which) * n; };
        auto bitsOf = [&](size_t r) { return snapshot.data() + r * slotWords_ + kRecordWords + 4 * n; };
        auto isNeighbour = [&](size_t r, size_t peer) { return (bitsOf(r)[peer / 64] >> (peer % 64)) & 1ull; };

        Record sum;
        int silent = 0;
        int ranksWithQueue = 0, ranksWithTreePending = 0, ranksWithActiveSends = 0;
        std::string silentList, anomalyList;
        size_t anomaliesListed = 0;
        for(size_t i = 0; i < n; ++i)
        {
            const Record r = recordOf(i);
            sum.start += r.start;
            sum.created += r.created;
            sum.finished += r.finished;
            sum.sent += r.sent;
            sum.received += r.received;
            sum.queued += r.queued;
            sum.treePending += r.treePending;
            sum.activeSends += r.activeSends;
            sum.messagesSent += r.messagesSent;
            sum.messagesReceived += r.messagesReceived;
            if(r.queued) ++ranksWithQueue;
            if(r.treePending) ++ranksWithTreePending;
            if(r.activeSends) ++ranksWithActiveSends;
            if((r.queued || r.treePending || r.activeSends) && anomaliesListed < maxListed)
            {
                std::ostringstream a;
                a << (anomaliesListed ? " " : "") << i << "(q=" << r.queued << ",tree=" << r.treePending
                  << ",sends=" << r.activeSends << ",fin=" << r.finished << ",sent=" << r.sent
                  << ",recv=" << r.received << ")";
                anomalyList += a.str();
                ++anomaliesListed;
            }
            if(r.iteration == previousIteration_[i])
            {
                if(static_cast<size_t>(silent) < maxListed)
                {
                    silentList += (silent ? " " : "");
                    silentList += std::to_string(i);
                    silentList += (r.iteration == 0) ? "(never)" : "(iter=" + std::to_string(r.iteration) + ")";
                }
                ++silent;
            }
            previousIteration_[i] = r.iteration;
        }

        // Pairwise deficits: what A says it sent to B minus what B says it got from A.
        size_t particleDeficitPairs = 0, messageDeficitPairs = 0, deficitPairsNotMutual = 0;
        std::int64_t particleDeficitTotal = 0, messageDeficitTotal = 0;
        std::string pairList;
        size_t pairsListed = 0;
        for(size_t a = 0; pairs_ && a < n; ++a)
        {
            const std::uint64_t *pSent = arrayOf(a, 0);
            const std::uint64_t *mSent = arrayOf(a, 2);
            for(size_t b = 0; b < n; ++b)
            {
                if(!pSent[b] && !mSent[b]) continue;
                const std::int64_t pd = static_cast<std::int64_t>(pSent[b]) - static_cast<std::int64_t>(arrayOf(b, 1)[a]);
                const std::int64_t md = static_cast<std::int64_t>(mSent[b]) - static_cast<std::int64_t>(arrayOf(b, 3)[a]);
                if(pd == 0 && md == 0) continue;
                if(pd) { ++particleDeficitPairs; particleDeficitTotal += pd; }
                if(md) { ++messageDeficitPairs; messageDeficitTotal += md; }
                const bool mutual = isNeighbour(a, b) && isNeighbour(b, a);
                if(!mutual) ++deficitPairsNotMutual;
                if(pairsListed < maxListed)
                {
                    std::ostringstream p;
                    p << (pairsListed ? " " : "") << a << "->" << b << "(particles " << pSent[b] << "/" << arrayOf(b, 1)[a]
                      << ", messages " << mSent[b] << "/" << arrayOf(b, 3)[a]
                      << ", A_lists_B=" << isNeighbour(a, b) << " B_lists_A=" << isNeighbour(b, a) << ")";
                    pairList += p.str();
                    ++pairsListed;
                }
            }
        }

        const std::int64_t inFlight = static_cast<std::int64_t>(sum.sent) - static_cast<std::int64_t>(sum.received);
        const std::int64_t implied = static_cast<std::int64_t>(sum.start) + static_cast<std::int64_t>(sum.created)
                                     - sum.finished - static_cast<std::int64_t>(sum.queued) - inFlight;
        std::ostringstream s;
        s << "[Telemetry] start=" << sum.start << " created=" << sum.created << " finished=" << sum.finished
          << " queued=" << sum.queued << " sent=" << sum.sent << " received=" << sum.received
          << " in_flight=" << inFlight << " tree_pending=" << sum.treePending
          << " active_sends=" << sum.activeSends
          << " messages_sent=" << sum.messagesSent << " messages_received=" << sum.messagesReceived
          << " implied_remaining=" << implied << " root_remaining=" << recordOf(0).rootRemaining
          << " ranks_with_queue=" << ranksWithQueue << " ranks_with_tree_pending=" << ranksWithTreePending
          << " ranks_with_active_sends=" << ranksWithActiveSends;
        if(pairs_)
        {
            s << "\n[Telemetry] pair deficits: particle_pairs=" << particleDeficitPairs << " particles=" << particleDeficitTotal
              << " message_pairs=" << messageDeficitPairs << " messages=" << messageDeficitTotal
              << " pairs_not_mutual_neighbours=" << deficitPairsNotMutual;
        }
        if(!pairList.empty())
        {
            s << "\n[Telemetry] deficit pairs A->B (sent by A / received by B): " << pairList
              << ((particleDeficitPairs + messageDeficitPairs) > maxListed ? " ..." : "");
        }
        if(!anomalyList.empty())
        {
            s << "\n[Telemetry] ranks holding particles or counts: " << anomalyList
              << ((ranksWithQueue + ranksWithTreePending + ranksWithActiveSends) > static_cast<int>(maxListed) ? " ..." : "");
        }
        if(silent > 0)
        {
            s << "\n[Telemetry] silent_ranks=" << silent << " (no loop progress since last report): " << silentList
              << (static_cast<size_t>(silent) > maxListed ? " ..." : "");
        }
        out = s.str();
        return silent;
    }

private:
    MPI_Comm comm_;
    bool pairs_ = false;
    int rank_ = 0;
    int size_ = 1;
    size_t bitmapWords_ = 0;
    size_t slotWords_ = 0;
    MPI_Win window_ = MPI_WIN_NULL;
    std::uint64_t *slots_ = nullptr;
    std::vector<std::uint64_t> staging_;
    std::vector<std::uint64_t> previousIteration_;
};

} // namespace STORM

#endif // STORM_WITH_MPI

#endif // STORM_SILENT_RANK_DETECTOR_HPP
