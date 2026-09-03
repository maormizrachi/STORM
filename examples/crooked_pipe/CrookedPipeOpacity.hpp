#ifndef STORM_CROOKED_PIPE_OPACITY_HPP
#define STORM_CROOKED_PIPE_OPACITY_HPP

#include <cstddef>
#include <vector>

#include "radiation/RadiationCell.hpp"
#include "radiation/RadiationOpacityModel.hpp"

namespace STORM {
namespace examples {

template<typename PointT, typename GridT>
class CrookedPipeOpacity : public RadiationOpacityModel<PointT, GridT, RadiationCell, 1>
{
public:
    CrookedPipeOpacity(const std::vector<int> &materialFlags, const std::vector<RadiationCell> &cells)
        : materialFlags_(materialFlags), cells_(&cells)
    {}

    double CalcPlanckOpacity(const RadiationCell &cell) override
    {
        std::size_t cellIndex = static_cast<std::size_t>(&cell - cells_->data());
        return materialFlags_[cellIndex] == 0 ? 2000.0 : 0.2;
    }

    double CalcScatteringOpacity(const RadiationCell &) override
    {
        return 0.0;
    }

private:
    const std::vector<int> &materialFlags_;
    const std::vector<RadiationCell> *cells_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_CROOKED_PIPE_OPACITY_HPP
