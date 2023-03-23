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
    prrng::pcg32_tensor_cumsum<array_type::tensor<double, 3>, array_type::tensor<ptrdiff_t, 2>, 1>;

/**
 * @brief Identical to Line1d::System_Cuspy_Laplace() but with '2d' interactions.
 * @copybrief detail::Laplace2d
 */
class System_Cuspy_Laplace
    : public detail::System<2, detail::Cuspy<Generator>, Generator, detail::Laplace2d> {
protected:
    detail::Cuspy<Generator> m_pot; ///< copybrief detail::System::m_potential
    detail::Laplace2d m_int; ///< copybrief detail::System::m_interactions

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
        Generator* chunk)
    {
        size_type rows = chunk->data().shape(0);
        size_type cols = chunk->data().shape(1);
        m_pot = detail::Cuspy<Generator>(mu, chunk);
        m_int = detail::Laplace2d(k_interactions, rows, cols);
        std::array<size_type, 2> shape = {rows, cols};
        this->initSystem(m, eta, k_frame, mu, dt, shape, &m_pot, chunk, &m_int);
    }
};

} // namespace Line2d
} // namespace FrictionQPotSpringBlock

#endif
