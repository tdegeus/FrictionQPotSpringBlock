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
 * Return versions of this library and of all of its dependencies.
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
 * Return information on the compiler, the platform, the C++ standard, and the compilation data.
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
 * ## Introduction
 *
 * Identical to Line1d::System_Cuspy_Laplace() but with '2d' interactions.
 * @copybrief detail::Laplace2d
 */
class System_Cuspy_Laplace : public detail::SystemNd_FrameDamping<
                                 2,
                                 detail::Cuspy<Generator>,
                                 Generator,
                                 detail::Laplace2d,
                                 detail::Athermal> {
protected:
    detail::Cuspy<Generator> m_potential; ///< @copybrief Line1d::System_Cuspy_Laplace::m_potential
    detail::Laplace2d m_interactions; ///< @copybrief Line1d::System_Cuspy_Laplace::m_interactions
    Generator* m_chunk; ///< @copybrief Line1d::System_Cuspy_Laplace::m_chunk
    detail::Athermal m_fluctuations; ///< ??

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
        m_potential = detail::Cuspy<Generator>(mu, chunk);
        m_interactions = detail::Laplace2d(k_interactions, rows, cols);

        this->init_SystemNd_FrameDamping(
            m,
            eta,
            k_frame,
            mu,
            dt,
            std::array<size_type, 2>{rows, cols},
            &m_potential,
            m_chunk,
            &m_interactions,
            &m_fluctuations);

        this->refresh();
    }
};

} // namespace Line2d
} // namespace FrictionQPotSpringBlock

#endif
