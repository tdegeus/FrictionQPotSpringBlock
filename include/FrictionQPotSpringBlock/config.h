/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotSpringBlock

*/

#ifndef FRICTIONQPOTSPRINGBLOCK_CONFIG_H
#define FRICTIONQPOTSPRINGBLOCK_CONFIG_H

#ifdef FRICTIONQPOTSPRINGBLOCK_ENABLE_ASSERT

    #define FRICTIONQPOTSPRINGBLOCK_ASSERT(expr) \
        FRICTIONQPOTSPRINGBLOCK_ASSERT_IMPL(expr, __FILE__, __LINE__)

    #define FRICTIONQPOTSPRINGBLOCK_ASSERT_IMPL(expr, file, line) \
        if (!(expr)) { \
            throw std::runtime_error( \
                std::string(file) + ':' + std::to_string(line) + \
                ": assertion failed (" #expr ") \n\t"); \
        }

#else

    #define FRICTIONQPOTSPRINGBLOCK_ASSERT(expr)

#endif

#define FRICTIONQPOTSPRINGBLOCK_REQUIRE(expr) \
    FRICTIONQPOTSPRINGBLOCK_REQUIRE_IMPL(expr, __FILE__, __LINE__)

#define FRICTIONQPOTSPRINGBLOCK_REQUIRE_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }

#define FRICTIONQPOTSPRINGBLOCK_VERSION_MAJOR 0
#define FRICTIONQPOTSPRINGBLOCK_VERSION_MINOR 0
#define FRICTIONQPOTSPRINGBLOCK_VERSION_PATCH 1

#define FRICTIONQPOTSPRINGBLOCK_VERSION_AT_LEAST(x, y, z) \
    (FRICTIONQPOTSPRINGBLOCK_VERSION_MAJOR > x || (FRICTIONQPOTSPRINGBLOCK_VERSION_MAJOR >= x && \
    (FRICTIONQPOTSPRINGBLOCK_VERSION_MINOR > y || (FRICTIONQPOTSPRINGBLOCK_VERSION_MINOR >= y && \
                                                   FRICTIONQPOTSPRINGBLOCK_VERSION_PATCH >= z))))

#define FRICTIONQPOTSPRINGBLOCK_VERSION(x, y, z) \
    (FRICTIONQPOTSPRINGBLOCK_VERSION_MAJOR == x && \
     FRICTIONQPOTSPRINGBLOCK_VERSION_MINOR == y && \
     FRICTIONQPOTSPRINGBLOCK_VERSION_PATCH == z)

#endif
