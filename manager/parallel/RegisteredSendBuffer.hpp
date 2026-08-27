#ifndef STORM_REGISTERED_SEND_BUFFER_HPP
#define STORM_REGISTERED_SEND_BUFFER_HPP

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <rma/RemoteMemoryAgent.hpp>

namespace STORM {

template<typename ParticleT, typename RankHandlerT>
class RegisteredSendBuffer
{
public:
    RegisteredSendBuffer() = default;
    RegisteredSendBuffer(const RegisteredSendBuffer &) = delete;
    RegisteredSendBuffer &operator=(const RegisteredSendBuffer &) = delete;

    RegisteredSendBuffer(RegisteredSendBuffer &&other) noexcept:
        storage(std::move(other.storage)),
        registrationHandler(other.registrationHandler),
        sourceRegistration(other.sourceRegistration)
    {
        other.registrationHandler = nullptr;
        other.sourceRegistration = {};
    }

    RegisteredSendBuffer &operator=(RegisteredSendBuffer &&other) noexcept
    {
        if(this != &other)
        {
            this->ReleaseRegistration();
            this->storage = std::move(other.storage);
            this->registrationHandler = other.registrationHandler;
            this->sourceRegistration = other.sourceRegistration;
            other.registrationHandler = nullptr;
            other.sourceRegistration = {};
        }
        return *this;
    }

    ~RegisteredSendBuffer()
    {
        this->ReleaseRegistration();
    }

    size_t size(void) const
    {
        return this->storage.size();
    }

    size_t capacity(void) const
    {
        return this->storage.capacity();
    }

    bool empty(void) const
    {
        return this->storage.empty();
    }

    ParticleT *data(void)
    {
        return this->storage.data();
    }

    const ParticleT *data(void) const
    {
        return this->storage.data();
    }

    void clear(void)
    {
        this->storage.clear();
    }

    void ReleaseStorage(void)
    {
        if(not this->storage.empty())
        {
            throw std::runtime_error(
                "RegisteredSendBuffer::ReleaseStorage called for a non-empty buffer");
        }
        this->ReleaseRegistration();
        std::vector<ParticleT>().swap(this->storage);
    }

    void push_back(const ParticleT &particle)
    {
        this->EnsureCapacity(this->storage.size() + 1);
        this->storage.push_back(particle);
    }

    void Append(const ParticleT *particles, size_t particlesNum)
    {
        if(particlesNum == 0)
        {
            return;
        }
        this->EnsureCapacity(this->storage.size() + particlesNum);
        this->storage.insert(this->storage.end(), particles, particles + particlesNum);
    }

    uint32_t SourceLkey(RankHandlerT *handler, bool useRegistration)
    {
        if(not useRegistration or handler == nullptr or this->storage.empty())
        {
            return 0;
        }
        if(this->sourceRegistration.handle != 0 and this->registrationHandler == handler)
        {
            return this->sourceRegistration.lkey;
        }
        this->ReleaseRegistration();
        this->sourceRegistration = handler->RegisterSendSource(this->storage.data(), this->storage.capacity());
        this->registrationHandler = (this->sourceRegistration.handle == 0)? nullptr : handler;
        return this->sourceRegistration.lkey;
    }

    void ReleaseRegistration(void)
    {
        if(this->sourceRegistration.handle != 0)
        {
            assert(this->registrationHandler != nullptr);
            this->registrationHandler->DeregisterSendSource(this->sourceRegistration.handle);
        }
        this->sourceRegistration = {};
        this->registrationHandler = nullptr;
    }

private:
    void EnsureCapacity(size_t desiredCapacity)
    {
        if(desiredCapacity <= this->storage.capacity())
        {
            return;
        }
        this->ReleaseRegistration();
        size_t newCapacity = std::max<size_t>(desiredCapacity, std::max<size_t>(1, this->storage.capacity() * 2));
        this->storage.reserve(newCapacity);
    }

    std::vector<ParticleT> storage;
    RankHandlerT *registrationHandler = nullptr;
    typename RemoteMemoryAgent<ParticleT>::SourceRegistration sourceRegistration{};
};

} // namespace STORM

#endif // STORM_REGISTERED_SEND_BUFFER_HPP
