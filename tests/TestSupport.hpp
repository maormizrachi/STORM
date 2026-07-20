#ifndef STORM_TEST_SUPPORT_HPP
#define STORM_TEST_SUPPORT_HPP

#include <cstdlib>
#include <iostream>

#define STORM_TEST_CHECK(condition) \
    do { \
        if(!(condition)) { \
            std::cerr << "Test check failed at " << __FILE__ << ':' \
                      << __LINE__ << ": " << #condition << '\n'; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(false)

#endif // STORM_TEST_SUPPORT_HPP
