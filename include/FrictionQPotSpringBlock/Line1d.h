/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#ifndef FRICTIONQPOTSPRINGBLOCK_LINE1D_H
#define FRICTIONQPOTSPRINGBLOCK_LINE1D_H

#include "config.h"
#include "detail.h"

#include <prrng.h>

#include <GMatTensor/version.h>

namespace FrictionQPotSpringBlock {

/**
 * Line in 1d.
 */
namespace Line1d {

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
 * @brief Standard system with a cuspy potential energy landscape and short range interactions.
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
 * -    @copydoc detail::Laplace1d
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
 * ## Parameters
 *
 * The following scaling argument holds to set \f$ k_\text{frame} \f$.
 * Because of the competition between disorder and elastic interactions, the interface
 * gets a roughness \f$ \delta u \sim \ell^\zeta \f$.
 * This corresponds to fluctuations of interactions
 * \f$ \delta f_\text{inter} \sim k_\text{inter} \partial^2 \delta u
 * \sim k_\text{inter} \ell^{\zeta - 2} \f$.
 * The fluctuations of the drive are
 * \f$ \delta f_\text{frame} \sim k_\text{frame} \delta u \sim k_\text{frame} \ell^\zeta \f$.
 * For the disorder to be relevant
 * \f$ \delta f_\text{inter} \sim \delta f_\text{frame} \f$.
 * Such that \f$ k_\text{frame} / k_\text{inter} \sim 1 / \ell^2 \f$.
 * With \f$ L = 1 / N \f$ the linear size of the system
 * \f$ k_\text{frame} / k_\text{inter} \sim 1 / L^2 \f$
 *
 * ## Dynamics
 *
 * The dynamics follow
 * \f$ m a_i = f_i \f$, with the particle's acceleration
 * \f$ a_i \equiv \ddot{u}_i \equiv \partial_t^2 u_i \f$, and the residual force
 * \f$ f_i = f_\text{damping}^{(i)} + f_\text{pot}^{(i)} +
 * f_\text{inter}^{(i)} + f_\text{frame}^{(i)} \f$.
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
class System_Cuspy_Laplace
    : public detail::System<1, detail::Cuspy<Generator>, Generator, detail::Laplace1d> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::Cuspy<Generator> m_pot; ///< @copybrief detail::System::m_potential
    detail::Laplace1d m_int; ///< @copybrief detail::System::m_interactions

public:
    /**
     * @param m @copybrief detail::System::m_m
     * @param eta @copybrief detail::System::m_eta
     * @param mu @copybrief detail::Cuspy::m_mu
     * @param k_interactions @copybrief detail::Laplace1d::m_k
     * @param k_frame @copybrief detail::System::m_k_frame
     * @param dt @copybrief detail::System::m_dt
     * @param shape @copybrief detail::System::shape
     * @param seed Global seed to use (`size` seeds will be consumed).
     * @param distribution Type of distribution (see prrng).
     * @param parameters Parameters of the distribution (see prrng).
     * @param offset Global offset to apply to the sequence of yield positions.
     * @param nchunk Number of random numbers to keep in memory.
     */
    System_Cuspy_Laplace(
        double m,
        double eta,
        double mu,
        double k_interactions,
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
              prrng::alignment(/*buffer*/ 5, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::Cuspy<Generator>(mu, &m_gen);
        m_int = detail::Laplace1d(k_interactions, shape[0]);
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen, &m_int);
    }
};

/**
 * @brief #System_Cuspy_Laplace assuming overdamped dynamics.
 * This system is meant only for minimisation during which the no-passing condition is used.
 * No dynamics can be simulated.
 */
class System_Cuspy_Laplace_Nopassing : public System_Cuspy_Laplace {
protected:
    double m_k_interactions; ///< @copybrief detail::Laplace1d::m_k

public:
    /**
     * @param mu @copybrief detail::Cuspy::m_mu
     * @param k_interactions @copybrief detail::Laplace1d::m_k
     * @param k_frame @copybrief detail::System::m_k_frame
     * @param shape @copybrief detail::System::shape
     * @param seed Global seed to use (`size` seeds will be consumed).
     * @param distribution Type of distribution (see prrng).
     * @param parameters Parameters of the distribution (see prrng).
     * @param offset Global offset to apply to the sequence of yield positions.
     * @param nchunk Number of random numbers to keep in memory.
     * @param eta @copybrief detail::System::m_eta
     * @param dt @copybrief detail::System::m_dt
     */
    System_Cuspy_Laplace_Nopassing(
        double mu,
        double k_interactions,
        double k_frame,
        const std::array<size_t, 1>& shape,
        uint64_t seed,
        const std::string& distribution,
        const std::vector<double>& parameters,
        double offset = -100.0,
        size_t nchunk = 5000,
        double eta = 0.0,
        double dt = 0.0
    )
        : System_Cuspy_Laplace(
              1.0,
              eta,
              mu,
              k_interactions,
              k_frame,
              dt,
              shape,
              seed,
              distribution,
              parameters,
              offset,
              nchunk
          ),
          m_k_interactions(k_interactions)
    {
    }

    /**
     * @warning This function is based on the no-passing condition and does not follow the dynamics.
     * The time is not updated.
     *
     * @copydoc detail::System::minimise
     */
    size_t minimise(
        double tol = 1e-5,
        size_t niter_tol = 10,
        size_t max_iter = 1e9,
        bool time_activity = false,
        bool max_iter_is_error = true
    ) override
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(!time_activity);
        (void)(time_activity);

        FRICTIONQPOTSPRINGBLOCK_ASSERT(tol < 1.0);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(max_iter + 1 < std::numeric_limits<long>::max());

        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);

        double uneigh;
        double u;
        double umin;
        ptrdiff_t i;
        ptrdiff_t j;
        size_t step;
        m_qs_inc_first = m_inc; // unused
        m_qs_inc_last = m_inc; // unused

        for (step = 1; step < max_iter + 1; ++step) {

            // "misuse" unused variable
            xt::noalias(m_v_n) = m_u;

            for (size_type p = 0; p < m_N; ++p) {

                if (p == 0) {
                    uneigh = m_v_n.back() + m_v_n(1);
                }
                else if (p == m_N - 1) {
                    uneigh = m_v_n(m_N - 2) + m_v_n.front();
                }
                else {
                    uneigh = m_v_n(p - 1) + m_v_n(p + 1);
                }

                i = m_gen.chunk_index_at_align()(p);
                auto* y = &m_gen.data()(p, 0);

                while (true) {
                    umin = 0.5 * (*(y + i) + *(y + i + 1));
                    u = (m_k_interactions * uneigh + m_k_frame * m_u_frame + m_mu * umin) /
                        (2 * m_k_interactions + m_k_frame + m_mu);
                    m_gen.align(p, u);
                    j = m_gen.chunk_index_at_align()(p);
                    if (j == i) {
                        break;
                    }
                    i = j;
                }
                m_u(p) = u;
                m_f_frame(p) = m_k_frame * (m_u_frame - u);
                m_f_potential(p) = m_mu * (umin - u);
            }

            this->computeForceInteractions();
            xt::noalias(m_f) = m_f_potential + m_f_interactions + m_f_frame;
            residuals.roll_insert(this->residual());

            if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
                this->quench(); // no dynamics are run: make sure that the user is not confused
                return 0;
            }
        }

        if (max_iter_is_error) {
            throw std::runtime_error("No convergence found");
        }

        return step;
    }

protected:
    /**
     * \cond
     */
    void timeStep();
    void timeSteps(size_t);
    size_t timeStepsUntilEvent(double, size_t, size_t);
    void flowSteps(size_t, double);
    /**
     * \endcond
     */
};

/**
 * @brief System in which the effect of temperature in mimicked by random forcing.
 * Otherwise the system is identical to Line1d::System_Cuspy_Laplace.
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
 * f_\text{inter}^{(i)} + f_\text{frame}^{(i)} + f_\text{random}^{(i)} \f$.
 *
 * ## Apply fixed force instead of fixed displacement
 *
 * To apply a fixed force (athermal or thermal) use a non-zero mean *and* set `k_frame = 0`.
 */
class System_Cuspy_Laplace_RandomForcing : public detail::System<
                                               1,
                                               detail::Cuspy<Generator>,
                                               Generator,
                                               detail::Laplace1d,
                                               detail::RandomNormalForcing<1>> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::Cuspy<Generator> m_pot; ///< @copybrief System_Cuspy_Laplace::m_pot
    detail::Laplace1d m_int; ///< @copybrief System_Cuspy_Laplace::m_int
    detail::RandomNormalForcing<1> m_ext; ///< Add extra random force to the residual.

public:
    /**
     * @copydoc System_Cuspy_Laplace::System_Cuspy_Laplace
     * @param mean @copybrief detail::RandomNormalForcing::m_mean
     * @param stddev @copybrief detail::RandomNormalForcing::m_stddev
     * @param seed_forcing Seed for the random number generator.
     * @param dinc_init Number of increments to wait to draw the first random force.
     * @param dinc @copybrief detail::RandomNormalForcing::m_dinc
     */
    System_Cuspy_Laplace_RandomForcing(
        double m,
        double eta,
        double mu,
        double k_interactions,
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
              prrng::alignment(/*buffer*/ 5, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::Cuspy<Generator>(mu, &m_gen);
        m_int = detail::Laplace1d(k_interactions, shape[0]);
        m_ext = detail::RandomNormalForcing<1>(
            m_gen.generators().shape(), mean, stddev, seed_forcing, dinc_init, dinc
        );
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen, &m_int, &m_ext);
    }

protected:
    /**
     * \cond
     */
    size_t minimise(double, size_t, size_t, bool, bool) override
    {
        return 0;
    };
    size_t quasistaticActivityFirst() const;
    size_t quasistaticActivityLast() const;
    double eventDrivenStep(double, bool, int);
    /**
     * \endcond
     */
};

/**
 * Same as System_Cuspy_Laplace() but with a semi-smooth potential.
 * @copybrief detail::SemiSmooth
 */
class System_SemiSmooth_Laplace
    : public detail::System<1, detail::SemiSmooth<Generator>, Generator, detail::Laplace1d> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::SemiSmooth<Generator> m_pot; ///< @copybrief System_Cuspy_Laplace::m_pot
    detail::Laplace1d m_int; ///< @copybrief System_Cuspy_Laplace::m_int

public:
    /**
     * @copydoc System_Cuspy_Laplace::System_Cuspy_Laplace
     * @param kappa @copydoc detail::SemiSmooth::m_kappa
     */
    System_SemiSmooth_Laplace(
        double m,
        double eta,
        double mu,
        double kappa,
        double k_interactions,
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
              prrng::alignment(/*buffer*/ 5, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::SemiSmooth<Generator>(mu, kappa, &m_gen);
        m_int = detail::Laplace1d(k_interactions, shape[0]);
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen, &m_int);
    }
};

/**
 * Same as System_Cuspy_Laplace() but with a smooth potential.
 * @copybrief detail::Smooth
 */
class System_Smooth_Laplace
    : public detail::System<1, detail::Smooth<Generator>, Generator, detail::Laplace1d> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::Smooth<Generator> m_pot; ///< @copybrief System_Cuspy_Laplace::m_pot
    detail::Laplace1d m_int; ///< @copybrief System_Cuspy_Laplace::m_int

public:
    /**
     * @copydoc System_Cuspy_Laplace::System_Cuspy_Laplace
     */
    System_Smooth_Laplace(
        double m,
        double eta,
        double mu,
        double k_interactions,
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
              prrng::alignment(/*buffer*/ 5, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::Smooth<Generator>(mu, &m_gen);
        m_int = detail::Laplace1d(k_interactions, shape[0]);
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen, &m_int);
    }
};

/**
 * Same as System_Cuspy_Laplace() but with a quartic interactions.
 * @copybrief detail::Quartic1d
 */
class System_Cuspy_Quartic
    : public detail::System<1, detail::Cuspy<Generator>, Generator, detail::Quartic1d> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::Cuspy<Generator> m_pot; ///< @copybrief System_Cuspy_Laplace::m_pot
    detail::Quartic1d m_int; ///< @copybrief System_Cuspy_Laplace::m_int

public:
    /**
     * @param m @copybrief detail::System::m_m
     * @param eta @copybrief detail::System::m_eta
     * @param mu @copybrief detail::Cuspy::m_mu
     * @param a1 @copybrief detail::Quartic1d::m_a1
     * @param a2 @copybrief detail::Quartic1d::m_a2
     * @param k_frame @copybrief detail::System::m_k_frame
     * @param dt @copybrief detail::System::m_dt
     * @param shape @copybrief detail::System::shape
     * @param seed Global seed to use (`size` seeds will be consumed).
     * @param distribution Type of distribution (see prrng).
     * @param parameters Parameters of the distribution (see prrng).
     * @param offset Global offset to apply to the sequence of yield positions.
     * @param nchunk Number of random numbers to keep in memory.
     */
    System_Cuspy_Quartic(
        double m,
        double eta,
        double mu,
        double a1,
        double a2,
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
              prrng::alignment(/*buffer*/ 5, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::Cuspy<Generator>(mu, &m_gen);
        m_int = detail::Quartic1d(a1, a2, shape[0]);
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen, &m_int);
    }
};

/**
 * Same as System_Cuspy_Laplace() but with a quartic interactions.
 * @copybrief detail::QuarticGradient1d
 */
class System_Cuspy_QuarticGradient : public detail::System<
                                         1,
                                         detail::Cuspy<Generator>,
                                         Generator,
                                         detail::QuarticGradient1d,
                                         detail::Athermal> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::Cuspy<Generator> m_pot; ///< @copybrief System_Cuspy_Laplace::m_pot
    detail::QuarticGradient1d m_int; ///< @copybrief System_Cuspy_Laplace::m_int

public:
    /**
     * @param m @copybrief detail::System::m_m
     * @param eta @copybrief detail::System::m_eta
     * @param mu @copybrief detail::Cuspy::m_mu
     * @param k2 @copybrief detail::QuarticGradient1d::m_k2
     * @param k4 @copybrief detail::QuarticGradient1d::m_k4
     * @param k_frame @copybrief detail::System::m_k_frame
     * @param dt @copybrief detail::System::m_dt
     * @param shape @copybrief detail::System::shape
     * @param seed Global seed to use (`size` seeds will be consumed).
     * @param distribution Type of distribution (see prrng).
     * @param parameters Parameters of the distribution (see prrng).
     * @param offset Global offset to apply to the sequence of yield positions.
     * @param nchunk Number of random numbers to keep in memory.
     */
    System_Cuspy_QuarticGradient(
        double m,
        double eta,
        double mu,
        double k2,
        double k4,
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
              prrng::alignment(/*buffer*/ 5, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::Cuspy<Generator>(mu, &m_gen);
        m_int = detail::QuarticGradient1d(k2, k4, shape[0]);
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen, &m_int);
    }
};

/**
 * Same as System_Cuspy_Laplace() but with a quartic interactions.
 * @copybrief detail::QuarticGradient1d
 */
class System_Cuspy_LongRange
    : public detail::System<1, detail::Cuspy<Generator>, Generator, detail::LongRange1d> {
protected:
    Generator m_gen; ///< @copybrief detail::System::m_chunk
    detail::Cuspy<Generator> m_pot; ///< @copybrief System_Cuspy_Laplace::m_pot
    detail::LongRange1d m_int; ///< @copybrief System_Cuspy_Laplace::m_int

public:
    /**
     * @param m @copybrief detail::System::m_m
     * @param eta @copybrief detail::System::m_eta
     * @param mu @copybrief detail::Cuspy::m_mu
     * @param k_interactions @copybrief detail::LongRange1d::m_k
     * @param alpha @copybrief detail::LongRange1d::m_alpha
     * @param k_frame @copybrief detail::System::m_k_frame
     * @param dt @copybrief detail::System::m_dt
     * @param shape @copybrief detail::System::shape
     * @param seed Global seed to use (`size` seeds will be consumed).
     * @param distribution Type of distribution (see prrng).
     * @param parameters Parameters of the distribution (see prrng).
     * @param offset Global offset to apply to the sequence of yield positions.
     * @param nchunk Number of random numbers to keep in memory.
     */
    System_Cuspy_LongRange(
        double m,
        double eta,
        double mu,
        double k_interactions,
        double alpha,
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
              prrng::alignment(/*buffer*/ 5, /*margin*/ 30, /*min_margin*/ 6, /*strict*/ false)
          )
    {
        m_gen += offset;
        m_pot = detail::Cuspy<Generator>(mu, &m_gen);
        m_int = detail::LongRange1d(k_interactions, alpha, shape[0]);
        this->initSystem(m, eta, k_frame, mu, dt, &m_pot, &m_gen, &m_int);
    }
};

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#endif
