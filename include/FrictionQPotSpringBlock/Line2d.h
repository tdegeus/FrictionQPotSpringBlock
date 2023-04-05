/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#ifndef FRICTIONQPOTSPRINGBLOCK_LINE2D_H
#define FRICTIONQPOTSPRINGBLOCK_LINE2D_H

#include "config.h"
#include "detail.h"

#include <prrng.h>

#include <GMatTensor/version.h>

namespace FrictionQPotSpringBlock {

namespace detail {

/**
 * @brief Get the default initial state.
 *
 * @param seed Global seed (`prod(shape)` consumed)
 * @param shape Shape
 * @return Initial state
 */
inline array_type::tensor<uint64_t, 2>
get_initstate_2d(uint64_t seed, const std::array<size_t, 2>& shape)
{
    array_type::tensor<uint64_t, 2> ret =
        seed + xt::arange<uint64_t>(shape[0] * shape[1]).reshape(shape);
    return ret;
};

} // namespace detail

/**
 * Line in 2d.
 */
namespace Line2d {

/**
 * @brief Return versions of this library and of all of its dependencies.
 * The output is a list of strings, e.g.::
 *
 *     "frictionqpotspringblock=0.1.0",
 *     "xtensor=0.20.1"
 *     ...
 *
 * @return List of strings.
 */
inline std::vector<std::string> version_dependencies()
{
    return GMatTensor::version_dependencies();
}

/**
 * @brief Return information on the compiler, platform, C++ standard, and the compilation data.
 * @return List of strings.
 */
inline std::vector<std::string> version_compiler()
{
    return GMatTensor::version_compiler();
}

/**
 * @brief Chunked storage of the cumulative sum of random numbers, used in all classes.
 */
using Generator =
    prrng::pcg32_tensor_cumsum<array_type::tensor<double, 3>, array_type::tensor<ptrdiff_t, 2>, 2>;

/**
 * @brief Identical to Line1d::System_Cuspy_Laplace() but with '2d' interactions.
 * @copybrief detail::Laplace2d
 */
class System_Cuspy_Laplace
    : public detail::System<2, detail::Cuspy<Generator>, Generator, detail::Laplace2d> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::Cuspy<Generator> m_pot; ///< @copybrief detail::System::m_potential
    detail::Laplace2d m_int; ///< @copybrief detail::System::m_interactions

public:
    /**
     * @copydoc Line1d::System_Cuspy_Laplace
     */
    System_Cuspy_Laplace(
        double m,
        double eta,
        double mu,
        double k_interactions,
        double k_frame,
        double dt,
        const std::array<size_t, 2>& shape,
        uint64_t seed,
        const std::string& distribution,
        const std::vector<double>& parameters,
        double offset = -100.0,
        size_t nchunk = 5000
    )
        : m_gen(
              std::array<size_t, 1>{nchunk},
              detail::get_initstate_2d(seed, shape),
              xt::eval(xt::zeros<uint64_t>(shape)),
              detail::string_to_distribution(distribution),
              parameters,
              prrng::alignment(/*buffer*/ 5, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::Cuspy<Generator>(mu, &m_gen);
        m_int = detail::Laplace2d(k_interactions, shape[0], shape[1]);
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen, &m_int);
    }
};

/**
 * @brief Identical to Line1d::System_Cuspy_QuarticGradient() but with '2d' interactions.
 * @copybrief detail::QuarticGradient2d
 */
class System_Cuspy_QuarticGradient
    : public detail::System<2, detail::Cuspy<Generator>, Generator, detail::QuarticGradient2d> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::Cuspy<Generator> m_pot; ///< @copybrief detail::System::m_potential
    detail::QuarticGradient2d m_int; ///< @copybrief detail::System::m_interactions

public:
    /**
     * @copydoc Line1d::System_Cuspy_QuarticGradient
     */
    System_Cuspy_QuarticGradient(
        double m,
        double eta,
        double mu,
        double k2,
        double k4,
        double k_frame,
        double dt,
        const std::array<size_t, 2>& shape,
        uint64_t seed,
        const std::string& distribution,
        const std::vector<double>& parameters,
        double offset = -100.0,
        size_t nchunk = 5000
    )
        : m_gen(
              std::array<size_t, 1>{nchunk},
              detail::get_initstate_2d(seed, shape),
              xt::eval(xt::zeros<uint64_t>(shape)),
              detail::string_to_distribution(distribution),
              parameters,
              prrng::alignment(/*buffer*/ 5, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::Cuspy<Generator>(mu, &m_gen);
        m_int = detail::QuarticGradient2d(k2, k4, shape[0], shape[1]);
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen, &m_int);
    }
};

} // namespace Line2d
} // namespace FrictionQPotSpringBlock

#endif
