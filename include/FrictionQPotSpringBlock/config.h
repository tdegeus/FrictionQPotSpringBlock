/**
Defines used in the library.

\file config.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTSPRINGBLOCK_CONFIG_H
#define FRICTIONQPOTSPRINGBLOCK_CONFIG_H

/**
\cond
*/
#define Q(x) #x
#define QUOTE(x) Q(x)

#define FRICTIONQPOTSPRINGBLOCK_WARNING_IMPL(message, file, line) \
    std::cout << \
        std::string(file) + ':' + std::to_string(line) + \
        ": " message ") \n\t"; \

#define FRICTIONQPOTSPRINGBLOCK_ASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }
/**
\endcond
*/

/**
All assertions are implementation as::

    FRICTIONQPOTSPRINGBLOCK_ASSERT(...)

They can be enabled by::

    #define FRICTIONQPOTSPRINGBLOCK_ENABLE_ASSERT

(before including FrictionQPotSpringBlock).
The advantage is that:

-   File and line-number are displayed if the assertion fails.
-   FrictionQPotSpringBlock's assertions can be enabled/disabled independently from those of other libraries.

\throw std::runtime_error
*/
#ifdef FRICTIONQPOTSPRINGBLOCK_ENABLE_ASSERT
#define FRICTIONQPOTSPRINGBLOCK_ASSERT(expr) FRICTIONQPOTSPRINGBLOCK_ASSERT_IMPL(expr, __FILE__, __LINE__)
#else
#define FRICTIONQPOTSPRINGBLOCK_ASSERT(expr)
#endif

/**
Assertions that cannot be disable.

\throw std::runtime_error
*/
#define FRICTIONQPOTSPRINGBLOCK_REQUIRE(expr) FRICTIONQPOTSPRINGBLOCK_REQUIRE_IMPL(expr, __FILE__, __LINE__)

/**
All warnings are implemented as::

    FRICTIONQPOTSPRINGBLOCK_WARNING(...)

They can be disabled by::

    #define FRICTIONQPOTSPRINGBLOCK_DISABLE_WARNING
*/
#ifdef FRICTIONQPOTSPRINGBLOCK_DISABLE_WARNING
#define FRICTIONQPOTSPRINGBLOCK_WARNING(message)
#else
#define FRICTIONQPOTSPRINGBLOCK_WARNING(message) FRICTIONQPOTSPRINGBLOCK_WARNING_IMPL(message, __FILE__, __LINE__)
#endif

/**
All warnings specific to the Python API are implemented as::

    FRICTIONQPOTSPRINGBLOCK_WARNING_PYTHON(...)

They can be enabled by::

    #define FRICTIONQPOTSPRINGBLOCK_ENABLE_WARNING_PYTHON
*/
#ifdef FRICTIONQPOTSPRINGBLOCK_ENABLE_WARNING_PYTHON
#define FRICTIONQPOTSPRINGBLOCK_WARNING_PYTHON(message) FRICTIONQPOTSPRINGBLOCK_WARNING_IMPL(message, __FILE__, __LINE__)
#else
#define FRICTIONQPOTSPRINGBLOCK_WARNING_PYTHON(message)
#endif

#endif
