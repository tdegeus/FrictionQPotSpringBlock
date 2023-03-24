/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#ifndef FRICTIONQPOTSPRINGBLOCK_CONFIG_H
#define FRICTIONQPOTSPRINGBLOCK_CONFIG_H

/**
 * \cond
 */
#define Q(x) #x
#define QUOTE(x) Q(x)

#define FRICTIONQPOTSPRINGBLOCK_WARNING_IMPL(message, file, line) \
    std::cout << std::string(file) + ":" + std::to_string(line) + ": " message ") \n\t";

#define FRICTIONQPOTSPRINGBLOCK_ASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ":" + std::to_string(line) + ": assertion failed (" #expr ") \n\t" \
        ); \
    }

#ifdef FRICTIONQPOTSPRINGBLOCK_ENABLE_DEBUG
#define FRICTIONQPOTFEM_ENABLE_ASSERT
#endif
/**
 * \endcond
 */

/**
 * All assertions are implementation as:
 *
 *     FRICTIONQPOTSPRINGBLOCK_ASSERT(...)
 *
 * They can be enabled by:
 *
 *     #define FRICTIONQPOTSPRINGBLOCK_ENABLE_ASSERT
 *
 * (before including FrictionQPotSpringBlock).
 * The advantage is that:
 *
 * -   File and line-number are displayed if the assertion fails.
 * -   FrictionQPotSpringBlock's assertions can be enabled/disabled independently from those of
 * other libraries.
 *
 * \throw std::runtime_error
 */
#ifdef FRICTIONQPOTSPRINGBLOCK_ENABLE_ASSERT
#define FRICTIONQPOTSPRINGBLOCK_ASSERT(expr) \
    FRICTIONQPOTSPRINGBLOCK_ASSERT_IMPL(expr, __FILE__, __LINE__)
#else
#define FRICTIONQPOTSPRINGBLOCK_ASSERT(expr)
#endif

/**
 * Some costly assertions, that are there mostly for debugging, are implemented as:
 *
 *     FRICTIONQPOTSPRINGBLOCK_DEBUG(...)
 *
 * They can be enabled by:
 *
 *     #define FRICTIONQPOTSPRINGBLOCK_ENABLE_DEBUG
 *
 * (before including FrictionQPotSpringBlock).
 * See also #FRICTIONQPOTSPRINGBLOCK_ASSERT.
 *
 * \throw std::runtime_error
 */
#ifdef FRICTIONQPOTSPRINGBLOCK_ENABLE_DEBUG
#define FRICTIONQPOTSPRINGBLOCK_DEBUG(expr) \
    FRICTIONQPOTSPRINGBLOCK_ASSERT_IMPL(expr, __FILE__, __LINE__)
#else
#define FRICTIONQPOTSPRINGBLOCK_DEBUG(expr)
#endif

/**
 * Assertions that cannot be disable.
 *
 * \throw std::runtime_error
 */
#define FRICTIONQPOTSPRINGBLOCK_REQUIRE(expr) \
    FRICTIONQPOTSPRINGBLOCK_ASSERT_IMPL(expr, __FILE__, __LINE__)

/**
 * All warnings are implemented as::
 *
 *     FRICTIONQPOTSPRINGBLOCK_WARNING(...)
 *
 * They can be disabled by::
 *
 *     #define FRICTIONQPOTSPRINGBLOCK_DISABLE_WARNING
 */
#ifdef FRICTIONQPOTSPRINGBLOCK_DISABLE_WARNING
#define FRICTIONQPOTSPRINGBLOCK_WARNING(message)
#else
#define FRICTIONQPOTSPRINGBLOCK_WARNING(message) \
    FRICTIONQPOTSPRINGBLOCK_WARNING_IMPL(message, __FILE__, __LINE__)
#endif

/**
 * All warnings specific to the Python API are implemented as:
 *
 *     FRICTIONQPOTSPRINGBLOCK_WARNING_PYTHON(...)
 *
 * They can be enabled by:
 *
 *     #define FRICTIONQPOTSPRINGBLOCK_ENABLE_WARNING_PYTHON
 */
#ifdef FRICTIONQPOTSPRINGBLOCK_ENABLE_WARNING_PYTHON
#define FRICTIONQPOTSPRINGBLOCK_WARNING_PYTHON(message) \
    FRICTIONQPOTSPRINGBLOCK_WARNING_IMPL(message, __FILE__, __LINE__)
#else
#define FRICTIONQPOTSPRINGBLOCK_WARNING_PYTHON(message)
#endif

/**
 * Current version.
 *
 * Either:
 *
 * -   Configure using CMake at install time. Internally uses::
 *
 *         python -c "from setuptools_scm import get_version; print(get_version())"
 *
 * -   Define externally using::
 *
 *         -DFRICTIONQPOTSPRINGBLOCK_VERSION="`python -c "from setuptools_scm import get_version;
 * print(get_version())"`"
 *
 *     From the root of this project. This is what ``setup.py`` does.
 *
 * Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version using
 * ``setuptools_scm``. Tip: use the environment variable ``SETUPTOOLS_SCM_PRETEND_VERSION`` to
 * overwrite the automatic version.
 */
#ifndef FRICTIONQPOTSPRINGBLOCK_VERSION
#define FRICTIONQPOTSPRINGBLOCK_VERSION "@PROJECT_VERSION@"
#endif

/**
 * Tensor products / operations.
 */
namespace FrictionQPotSpringBlock {

/**
 * Container type.
 */
namespace array_type {

#ifdef FRICTIONQPOTSPRINGBLOCK_USE_XTENSOR_PYTHON

/**
 * Fixed (static) rank array.
 */
template <typename T, size_t N>
using tensor = xt::pytensor<T, N>;

/**
 * Arbitrary rank array.
 */
template <typename T>
using array = xt::pyarray<T>;

#else

/**
 * Fixed (static) rank array.
 */
template <typename T, size_t N>
using tensor = xt::xtensor<T, N>;

/**
 * Arbitrary rank array.
 */
template <typename T>
using array = xt::xarray<T>;

#endif

} // namespace array_type

/**
 * @brief Type using for size and shapes of arrays.
 */
using size_type = ptrdiff_t;

namespace detail {

/**
 * @brief Remove quotes from a string.
 * @param arg String.
 * @return String.
 */
inline std::string unquote(const std::string& arg)
{
    std::string ret = arg;
    ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
    return ret;
}

} // namespace detail

/**
 * Return version string, e.g. `"0.8.0"`
 * @return Version string.
 */
inline std::string version()
{
    return detail::unquote(std::string(QUOTE(FRICTIONQPOTSPRINGBLOCK_VERSION)));
}

} // namespace FrictionQPotSpringBlock

#endif
