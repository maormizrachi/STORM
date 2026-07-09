#ifndef STORM_HOHLRAUM_OPACITY_HPP
#define STORM_HOHLRAUM_OPACITY_HPP

#include <cmath>
#include <cstddef>
#include <vector>
#include "radiation/RadiationOpacityModel.hpp"
#include "radiation/RadiationCell.hpp"
#include "PhysicalConstants.hpp"

namespace STORM {
namespace examples {

template<typename PointT, typename GridT>
class HohlraumOpacity : public RadiationOpacityModel<PointT, GridT, RadiationCell, 1>
{
public:
    HohlraumOpacity(const std::vector<int> &materialFlags, const std::vector<RadiationCell> &cells)
        : materialFlags_(materialFlags), cells_(&cells)
    {}

    double CalcPlanckOpacity(const RadiationCell &cell) override
    {
        std::size_t idx = cellIndex(cell);
        if(idx < materialFlags_.size() && materialFlags_[idx])
        {
            double T_keV = cell.temperature / constants::kev_kelvin;
            T_keV = std::max(T_keV, 1e-4);
            return 300.0 * std::pow(T_keV, -3.0);
        }
        return 1e-20;
    }

    double CalcScatteringOpacity(const RadiationCell &) override
    {
        return 0.0;
    }

private:
    std::size_t cellIndex(const RadiationCell &cell) const
    {
        return static_cast<std::size_t>(&cell - cells_->data());
    }

    const std::vector<int> &materialFlags_;
    const std::vector<RadiationCell> *cells_;
};

} // namespace examples
} // namespace STORM

#endif // STORM_HOHLRAUM_OPACITY_HPP
