#ifndef STORM_MONTE_CARLO_PARTICLE_QUEUE_HPP
#define STORM_MONTE_CARLO_PARTICLE_QUEUE_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace STORM {

// A single-array FIFO for particles that are consumed by the transport loop
// and appended by the same owner.  The queue deliberately has no MPI or GPU
// dependencies; RankHandler2 can expose the same logical operations while
// retaining its remote-memory implementation.
template<typename ParticleT>
class ParticleQueue
{
public:
    using value_type = ParticleT;

    ParticleQueue() = default;

    explicit ParticleQueue(size_t initialCapacity)
    {
        this->Reserve(initialCapacity);
    }

    ParticleQueue(const ParticleQueue &) = delete;
    ParticleQueue &operator=(const ParticleQueue &) = delete;

    ParticleQueue(ParticleQueue &&other) noexcept:
        storage_(std::move(other.storage_)),
        capacity_(other.capacity_),
        head_(other.head_),
        size_(other.size_)
    {
        other.capacity_ = 0;
        other.head_ = 0;
        other.size_ = 0;
    }

    ParticleQueue &operator=(ParticleQueue &&other) noexcept
    {
        if(this != &other)
        {
            this->storage_ = std::move(other.storage_);
            this->capacity_ = other.capacity_;
            this->head_ = other.head_;
            this->size_ = other.size_;
            other.capacity_ = 0;
            other.head_ = 0;
            other.size_ = 0;
        }
        return *this;
    }

    size_t Size(void) const
    {
        return this->size_;
    }

    bool Empty(void) const
    {
        return this->size_ == 0;
    }

    size_t Capacity(void) const
    {
        return this->capacity_;
    }

    void Clear(void)
    {
        this->head_ = 0;
        this->size_ = 0;
    }

    void Reserve(size_t desiredCapacity)
    {
        if(desiredCapacity <= this->capacity_)
        {
            return;
        }

        size_t newCapacity = std::max<size_t>(desiredCapacity, 1);
        if(this->capacity_ > 0)
        {
            if(this->capacity_ > std::numeric_limits<size_t>::max() / 2)
            {
                throw std::length_error("ParticleQueue::Reserve: capacity overflow");
            }
            newCapacity = std::max<size_t>(newCapacity, this->capacity_ * 2);
        }

        std::unique_ptr<ParticleT[]> newStorage =
            std::make_unique<ParticleT[]>(newCapacity);
        for(size_t i = 0; i < this->size_; i++)
        {
            newStorage[i] = std::move(this->At(i));
        }

        this->storage_ = std::move(newStorage);
        this->capacity_ = newCapacity;
        this->head_ = 0;
    }

    ParticleT &Front(void)
    {
        assert(not this->Empty());
        return this->storage_[this->head_];
    }

    const ParticleT &Front(void) const
    {
        assert(not this->Empty());
        return this->storage_[this->head_];
    }

    ParticleT &At(size_t logicalIndex)
    {
        assert(logicalIndex < this->size_);
        return this->storage_[this->PhysicalIndex(logicalIndex)];
    }

    const ParticleT &At(size_t logicalIndex) const
    {
        assert(logicalIndex < this->size_);
        return this->storage_[this->PhysicalIndex(logicalIndex)];
    }

    void Consume(void)
    {
        assert(not this->Empty());
        this->head_ = (this->head_ + 1) % this->capacity_;
        this->size_--;
        if(this->size_ == 0)
        {
            this->head_ = 0;
        }
    }

    void Append(const ParticleT &particle)
    {
        if(this->size_ == std::numeric_limits<size_t>::max())
        {
            throw std::overflow_error("ParticleQueue::Append: particle count overflow");
        }
        this->Reserve(this->size_ + 1);
        this->storage_[this->TailIndex()] = particle;
        this->size_++;
    }

    void Append(ParticleT &&particle)
    {
        if(this->size_ == std::numeric_limits<size_t>::max())
        {
            throw std::overflow_error("ParticleQueue::Append: particle count overflow");
        }
        this->Reserve(this->size_ + 1);
        this->storage_[this->TailIndex()] = std::move(particle);
        this->size_++;
    }

    template<typename Writer>
    void Append(size_t particlesNum, const Writer &writer)
    {
        if(particlesNum == 0)
        {
            return;
        }
        if(particlesNum > std::numeric_limits<size_t>::max() - this->size_)
        {
            throw std::overflow_error("ParticleQueue::Append: particle count overflow");
        }

        this->Reserve(this->size_ + particlesNum);
        for(size_t i = 0; i < particlesNum; i++)
        {
            writer(this->storage_[this->PhysicalIndex(this->size_ + i)], i);
        }
        this->size_ += particlesNum;
    }

    template<typename Func>
    void ForEachActive(const Func &func)
    {
        for(size_t i = 0; i < this->size_; i++)
        {
            func(this->At(i), i);
        }
    }

private:
    size_t PhysicalIndex(size_t logicalIndex) const
    {
        assert(this->capacity_ > 0);
        return (this->head_ + logicalIndex) % this->capacity_;
    }

    size_t TailIndex(void) const
    {
        return this->PhysicalIndex(this->size_);
    }

    std::unique_ptr<ParticleT[]> storage_;
    size_t capacity_ = 0;
    size_t head_ = 0;
    size_t size_ = 0;
};

} // namespace STORM

#endif // STORM_MONTE_CARLO_PARTICLE_QUEUE_HPP
