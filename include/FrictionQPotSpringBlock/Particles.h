/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#ifndef FRICTIONQPOTSPRINGBLOCK_PARTICLES_H
#define FRICTIONQPOTSPRINGBLOCK_PARTICLES_H

#include "config.h"
#include "detail.h"

#include <prrng.h>

#include <GMatTensor/version.h>

namespace FrictionQPotSpringBlock {

/**
 * A (set of) particle(s).
 */
namespace Particles {

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
    prrng::pcg32_tensor_cumsum<array_type::tensor<double, 2>, array_type::tensor<ptrdiff_t, 1>, 1>;

/**
 * @brief Standard system with a cuspy potential energy landscape.
 *
 * ## Physics
 *
 * The system consists of particles of mass \f$ m \f$.
 * The principle degree of freedom is the slip ('position') \f$ u_i \f$ of each particle \f$ i \f$.
 * Each particle follows Newtonian dynamics \f$ m \ddot{u}_i = f_i \f$, with the residual force
 * \f$ f_i \f$ given by the sum of:
 *
 * -    @copydoc detail::Cuspy
 *
 * -    A drive though a spring attached to a load frame.
 *
 *      This corresponds to a parabolic potential energy for each particle,
 *      with the minimum at a set \f$ u_\text{frame} \f$:
 *      \f$ f_\text{frame}^{(i)} = k_\text{frame} (u_\text{frame} - u_i) \f$.
 *
 * -    Damping.
 *
 *      \f$ f_\text{damping}^{(i)} = - \eta v_i \f$, with the particle's velocity
 *      \f$ v_i \equiv \dot{u}_i \equiv \partial_t u_i \f$.
 *
 * ## Dynamics
 *
 * The dynamics follow
 * \f$ m a_i = f_i \f$, with the particle's acceleration
 * \f$ a_i \equiv \ddot{u}_i \equiv \partial_t^2 u_i \f$, and the residual force
 * \f$ f_i = f_\text{damping}^{(i)} + f_\text{pot}^{(i)} + f_\text{frame}^{(i)} \f$.
 * This differential equation is integrated using Velocity-Verlet in timeStep()
 * Note that this function updates the time #t (or in fact the increment number #inc),
 * and that this is the only function that does so.
 *
 * The implementation is is such that all forces are updated (if needed) every time the positions
 * and velocities are updated, either internally in timeStep(), or externally when calling
 * set_u() and set_v().
 * In absolute need refresh() can be called to force re-computation of forces,
 * but this should normally not be needed.
 */
class System_Cuspy : public detail::System<1, detail::Cuspy<Generator>, Generator> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::Cuspy<Generator> m_pot; ///< @copybrief detail::System::m_potential

public:
    /**
     * @param m @copybrief detail::System::m_m
     * @param eta @copybrief detail::System::m_eta
     * @param mu @copybrief detail::Cuspy::m_mu
     * @param k_frame @copybrief detail::System::m_k_frame
     * @param dt @copybrief detail::System::m_dt
     * @param shape @copybrief detail::System::shape
     * @param seed Global seed to use (`size` seeds will be consumed).
     * @param distribution Type of distribution (see prrng).
     * @param parameters Parameters of the distribution (see prrng).
     * @param offset Global offset to apply to the sequence of yield positions.
     * @param nchunk Number of random numbers to keep in memory.
     */
    System_Cuspy(
        double m,
        double eta,
        double mu,
        double k_frame,
        double dt,
        const std::array<size_t, 1>& shape,
        uint64_t seed,
        const std::string& distribution,
        const std::vector<double>& parameters,
        double offset = -100.0,
        size_t nchunk = 5000
    )
        : m_gen(
              std::array<size_t, 1>{nchunk},
              xt::eval(seed + xt::arange<uint64_t>(shape[0])),
              xt::eval(xt::zeros<uint64_t>(shape)),
              detail::string_to_distribution(distribution),
              parameters,
              prrng::alignment(/*buffer*/ 2, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::Cuspy<Generator>(mu, &m_gen);
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen);
    }
};

/**
 * @brief System in which the effect of temperature in mimicked by random forcing.
 * Otherwise the system is identical to Particles::System_Cuspy.
 *
 * ## Physics
 *
 * The random force drawn from a random distribution and is changed every `n` increments.
 * As such, it is defined by:
 *
 *  -   The mean and standard deviation of the random distribution.
 *  -   The first increment at which the random force is changed.
 *  -   The number of increments between two changes of the random force.
 *
 * The residual now reads
 * \f$ f_i = f_\text{damping}^{(i)} + f_\text{pot}^{(i)} +
 * f_\text{frame}^{(i)} + f_\text{random}^{(i)} \f$.
 *
 * ## Apply fixed force instead of fixed displacement
 *
 * To apply a fixed force (athermal or thermal) use a non-zero mean *and* set `k_frame = 0`.
 */
class System_Cuspy_RandomForcing : public detail::System<
                                       1,
                                       detail::Cuspy<Generator>,
                                       Generator,
                                       void,
                                       detail::RandomNormalForcing<1>,
                                       detail::None> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::Cuspy<Generator> m_pot; ///< @copybrief System_Cuspy::m_pot
    detail::RandomNormalForcing<1> m_ext; ///< Add extra random force to the residual.

public:
    /**
     * @copydoc System_Cuspy::System_Cuspy
     * @param mean @copybrief detail::RandomNormalForcing::m_mean
     * @param stddev @copybrief detail::RandomNormalForcing::m_stddev
     * @param seed_forcing Seed for the random number generator.
     * @param dinc_init Number of increments to wait to draw the first random force.
     * @param dinc @copybrief detail::RandomNormalForcing::m_dinc
     */
    System_Cuspy_RandomForcing(
        double m,
        double eta,
        double mu,
        double k_frame,
        double dt,
        double mean,
        double stddev,
        uint64_t seed_forcing,
        const array_type::tensor<ptrdiff_t, 1>& dinc_init,
        const array_type::tensor<ptrdiff_t, 1>& dinc,
        const std::array<size_t, 1>& shape,
        uint64_t seed,
        const std::string& distribution,
        const std::vector<double>& parameters,
        double offset = -100.0,
        size_t nchunk = 5000
    )
        : m_gen(
              std::array<size_t, 1>{nchunk},
              xt::eval(seed + xt::arange<uint64_t>(shape[0])),
              xt::eval(xt::zeros<uint64_t>(shape)),
              detail::string_to_distribution(distribution),
              parameters,
              prrng::alignment(/*buffer*/ 2, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::Cuspy<Generator>(mu, &m_gen);
        m_ext = detail::RandomNormalForcing<1>(
            m_gen.generators().shape(), mean, stddev, seed_forcing, dinc_init, dinc
        );
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen, nullptr, &m_ext);
    }

protected:
    /**
     * \cond
     */
    size_t quasistaticActivityFirst() const;
    size_t quasistaticActivityLast() const;
    double eventDrivenStep(double, bool, int);
    /**
     * \endcond
     */
};

/**
 * Same as System_Cuspy() but with a semi-smooth potential.
 * @copybrief detail::SemiSmooth
 */
class System_SemiSmooth : public detail::System<1, detail::SemiSmooth<Generator>, Generator> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::SemiSmooth<Generator> m_pot; ///< @copybrief System_Cuspy::m_pot

public:
    /**
     * @copydoc System_Cuspy::System_Cuspy
     * @param kappa @copydoc detail::SemiSmooth::m_kappa
     */
    System_SemiSmooth(
        double m,
        double eta,
        double mu,
        double kappa,
        double k_frame,
        double dt,
        const std::array<size_t, 1>& shape,
        uint64_t seed,
        const std::string& distribution,
        const std::vector<double>& parameters,
        double offset = -100.0,
        size_t nchunk = 5000
    )
        : m_gen(
              std::array<size_t, 1>{nchunk},
              xt::eval(seed + xt::arange<uint64_t>(shape[0])),
              xt::eval(xt::zeros<uint64_t>(shape)),
              detail::string_to_distribution(distribution),
              parameters,
              prrng::alignment(/*buffer*/ 2, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::SemiSmooth<Generator>(mu, kappa, &m_gen);
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen);
    }
};

/**
 * Same as System_Cuspy() but with a smooth potential.
 * @copybrief detail::Smooth
 */
class System_Smooth : public detail::System<1, detail::Smooth<Generator>, Generator> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::Smooth<Generator> m_pot; ///< @copybrief System_Cuspy::m_pot

public:
    /**
     * @copydoc System_Cuspy::System_Cuspy
     */
    System_Smooth(
        double m,
        double eta,
        double mu,
        double k_frame,
        double dt,
        const std::array<size_t, 1>& shape,
        uint64_t seed,
        const std::string& distribution,
        const std::vector<double>& parameters,
        double offset = -100.0,
        size_t nchunk = 5000
    )
        : m_gen(
              std::array<size_t, 1>{nchunk},
              xt::eval(seed + xt::arange<uint64_t>(shape[0])),
              xt::eval(xt::zeros<uint64_t>(shape)),
              detail::string_to_distribution(distribution),
              parameters,
              prrng::alignment(/*buffer*/ 2, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::Smooth<Generator>(mu, &m_gen);
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen);
    }
};

} // namespace Particles
} // namespace FrictionQPotSpringBlock

#endif
