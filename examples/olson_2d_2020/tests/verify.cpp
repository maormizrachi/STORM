#include <iostream>
#include <iomanip>
#include "../OlsonPhysics.hpp"
int main()
{
    Olson::Spectrum s;
    Olson::EOS eos;
    std::cout << std::setprecision(17);
    for(int al = 0; al < 2; ++al)
    {
        for(double T : {.001, .005, .01, .019, .05, .1, .23, .5, 1.})
        {
            double meanEnergy = 0;
            const int n = 20000;
            for(int i = 0; i < n; ++i)
            {
                meanEnergy += s.Sample(T, al, (i + .5) / n) / n;
            }
            double e = Olson::Energy(T, al), heatCapacity = Olson::Cv(T, al);
            double back = eos.de2T(Olson::rho, e / Olson::rho, {double(al)}, {});
            std::cout << al << ' ' << T << ' ' << s.Mean(T, al) << ' ' << meanEnergy << ' ' << e << ' ' << heatCapacity << ' ' << back / units::kev_kelvin << ' ' << s.Sample(T, al, .1) << ' ' << s.Sample(T, al, .5) << ' ' << s.Sample(T, al, .9) << '\n';
        }
    }
}
