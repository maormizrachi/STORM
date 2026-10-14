#ifndef STORM_SHAPIRO_SUTHERLAND_REFERENCE_HPP
#define STORM_SHAPIRO_SUTHERLAND_REFERENCE_HPP

// Bin-averaged reference for the plane observer: 20 equal bins in mu, each
// I/F = int I mu dmu / int mu dmu / F and Q/I = int Q mu dmu / int I mu dmu,
// as printed by planeparallel_reference.py.  Q/I uses Shapiro & Sutherland's
// sign (positive for E parallel to the surface).

#include <array>
#include <cstddef>

namespace shapiro_reference {

using Table = std::array<std::array<double, 2>, 20>;

inline constexpr Table k_scattering = {{
    {0.456685, 0.096820},
    {0.502481, 0.080760},
    {0.548694, 0.068115},
    {0.593356, 0.058193},
    {0.636915, 0.050107},
    {0.679691, 0.043339},
    {0.721891, 0.037562},
    {0.763652, 0.032557},
    {0.805069, 0.028169},
    {0.846211, 0.024284},
    {0.887128, 0.020817},
    {0.927859, 0.017699},
    {0.968434, 0.014880},
    {1.008877, 0.012316},
    {1.049206, 0.009974},
    {1.089437, 0.007825},
    {1.129583, 0.005846},
    {1.169653, 0.004016},
    {1.209658, 0.002320},
    {1.249605, 0.000742},
}};

inline constexpr Table k_lambda16 = {{
    {0.336926, 0.214374},
    {0.368633, 0.199684},
    {0.402353, 0.186676},
    {0.437007, 0.174899},
    {0.473205, 0.163689},
    {0.511507, 0.152690},
    {0.552423, 0.141711},
    {0.596427, 0.130666},
    {0.643980, 0.119528},
    {0.695527, 0.108315},
    {0.751507, 0.097071},
    {0.812351, 0.085855},
    {0.878485, 0.074731},
    {0.950327, 0.063767},
    {1.028286, 0.053025},
    {1.112769, 0.042559},
    {1.204170, 0.032418},
    {1.302880, 0.022637},
    {1.409279, 0.013245},
    {1.523739, 0.004262},
}};

inline constexpr Table k_lambdatau = {{
    {0.372210, 0.266889},
    {0.385780, 0.255561},
    {0.397317, 0.245964},
    {0.409032, 0.236130},
    {0.422932, 0.224757},
    {0.440574, 0.211261},
    {0.463326, 0.195540},
    {0.492470, 0.177867},
    {0.529245, 0.158791},
    {0.574865, 0.139018},
    {0.630523, 0.119285},
    {0.697392, 0.100250},
    {0.776628, 0.082422},
    {0.869364, 0.066138},
    {0.976712, 0.051569},
    {1.099760, 0.038749},
    {1.239569, 0.027618},
    {1.397175, 0.018054},
    {1.573587, 0.009902},
    {1.769788, 0.002997},
}};

//! 0 = scattering, 1 = lambda = 1/6, 2 = lambda = tau/(1+tau)
inline Table const &table(int which)
{
    return which == 0 ? k_scattering : (which == 1 ? k_lambda16 : k_lambdatau);
}

} // namespace shapiro_reference

#endif // STORM_SHAPIRO_SUTHERLAND_REFERENCE_HPP
