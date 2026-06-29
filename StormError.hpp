#ifndef STORM_ERROR_HPP
#define STORM_ERROR_HPP

#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <utility>

namespace STORM {

class StormError : public std::runtime_error
{
public:
    explicit StormError(const std::string &msg)
        : std::runtime_error(msg), msg_(msg) {}

    template<typename T>
    void addEntry(const std::string &name, const T &value)
    {
        std::ostringstream oss;
        oss << value;
        entries_.emplace_back(name, oss.str());
        rebuildWhat();
    }

    const std::string &getErrorMessage() const { return msg_; }

    const std::vector<std::pair<std::string, std::string>> &getEntries() const { return entries_; }

private:
    void rebuildWhat()
    {
        std::string full = msg_;
        for(const auto &[name, val] : entries_)
        {
            full += "\n  " + name + ": " + val;
        }
        full_ = std::move(full);
    }

    const char *what() const noexcept override { return full_.empty() ? msg_.c_str() : full_.c_str(); }

    std::string msg_;
    std::string full_;
    std::vector<std::pair<std::string, std::string>> entries_;
};

inline void reportError(const StormError &eo, std::ostream &os = std::cout)
{
    os << eo.getErrorMessage() << std::endl;
    for(const auto &[name, val] : eo.getEntries())
    {
        os << "  " << name << ": " << val << std::endl;
    }
}

} // namespace STORM

using STORMError = STORM::StormError;

#endif // STORM_ERROR_HPP
