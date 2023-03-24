/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#ifndef FRICTIONQPOTSPRINGBLOCK_DETAIL_H
#define FRICTIONQPOTSPRINGBLOCK_DETAIL_H

#include <string>

#include <xtensor/xadapt.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xtensor.hpp>

#include "config.h"

#include <GooseFEM/Iterate.h>
#include <GooseFEM/version.h>
#include <prrng.h>

#include <GMatTensor/version.h>

namespace FrictionQPotSpringBlock {

namespace detail {

/**
 * @brief A piece-wise quadratic local potential energy.
 * The resulting force is piece-wise linear:
 * \f$ f_\mathrm{pot}^{(i)} =  \mu (u_{\min}^{(i)} - u_i) \f$.
 * where \f$ \mu \f$ is the curvature of the quadratic potentials, and \f$ u_{\min}^{(i)} \f$
 * the position of the current local minimum.
 * The landscape is given by a cumulative sum of random numbers.
 */
template <class Generator>
class Cuspy {
protected:
    using stype = typename Generator::size_type; ///< Size type.
    stype m_N; ///< Number of particles.
    double m_mu; ///< Curvature of the potentials.
    Generator* m_chunk; ///< Pointer to chunk of yield 'positions' (automatically updated if needed)

public:
    Cuspy() = default;

    /**
     * @param mu @copydoc Cuspy::m_mu
     * @param chunk @copydoc Cuspy::m_chunk
     * @tparam Generator The type of the prrng::pcg32_tensor_cumsum object.
     */
    Cuspy(double mu, Generator* chunk) : m_mu(mu), m_chunk(chunk)
    {
        m_N = m_chunk->generators().size();
    }

    /**
     * @brief Update forces based on current slips.
     * @param u_array Slip per particle.
     * @param f_array Force per particle (output).
     */
    template <class T>
    void force(const T& u_array, T& f_array)
    {
        m_chunk->align(u_array);

        // this does not allocate data, but only creates a view
        xt::xtensor_pointer<const double, 2> yield = xt::adapt(
            m_chunk->data().data(),
            m_chunk->data().size(),
            xt::no_ownership(),
            std::array<stype, 2>{m_N, m_chunk->chunk_size()});

        // this does not allocate data, but only creates a view
        xt::xtensor_pointer<const ptrdiff_t, 1> i = xt::adapt(
            m_chunk->chunk_index_at_align().data(),
            m_chunk->chunk_index_at_align().size(),
            xt::no_ownership(),
            std::array<stype, 1>{m_N});

        for (stype p = 0; p < m_N; ++p) {
            const auto* l = &yield(p, i(p));
            f_array(p) = 0.5 * (*(l) + *(l + 1)) - u_array(p);
        }

        f_array *= m_mu;
    }

    /**
     * @copydoc System::maxUniformDisplacement
     * @param u_array Slip per particle.
     */
    template <class T>
    double maxUniformDisplacement(const T& u_array, int direction) const
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(direction == 1 || direction == -1);
        m_chunk->align(u_array);

        if (direction > 0) {
            return xt::amin(m_chunk->template right_of_align<T>() - u_array)();
        }

        return xt::amin(u_array - m_chunk->template left_of_align<T>())();
    }

    /**
     * @copydoc System::trigger
     * @param u_array Slip per particle.
     */
    template <class T>
    void trigger(T& u_array, size_t p, double eps, int direction) const
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(p < m_N);
        m_chunk->align(u_array);

        if (direction > 0) {
            u_array.flat(p) = m_chunk->template right_of_align<T>().flat(p) + 0.5 * eps;
        }
        else {
            u_array.flat(p) = m_chunk->template left_of_align<T>().flat(p) - 0.5 * eps;
        }
    }
};

/**
 * Athermal 'dummy' class
 */
class Athermal {
protected:
public:
    Athermal() = default;

    /**
     * @brief Do nothing.
     */
    template <class T, class S>
    void force(const T& /* u_array */, T& /* f_array */, S /* inc */)
    {
    }
};

/**
 * Each particle experiences a random force representing the effect of temperature.
 * The random force drawn from a random distribution and is changed every `n` increments.
 * As such, it is defined by:
 *
 *  -   The mean and standard deviation of the random distribution.
 *  -   The first increment at which the random force is changed.
 *  -   The number of increments between two changes of the random force.
 *
 * @tparam rank rank of the system
 */
template <size_t rank>
class RandomNormalForcing {
protected:
    std::array<size_type, rank> m_shape; ///< @copybrief System::shape
    size_type m_N; ///< Number of particles.
    array_type::tensor<double, rank> m_f_thermal; ///< Current applied 'random' forces.
    double m_mean; ///< Mean of the random distribution.
    double m_stddev; ///< Standard deviation of the random distribution.
    array_type::tensor<ptrdiff_t, rank> m_next; ///< Next increment at to draw.
    array_type::tensor<ptrdiff_t, rank> m_dinc; ///< \#increments between two draws.
    prrng::pcg32 m_rng; ///< Random number generator.

public:
    RandomNormalForcing() = default;

    /**
     * @param shape @copybrief System::shape
     * @param mean @copybrief RandomNormalForcing::m_mean
     * @param stddev @copybrief RandomNormalForcing::m_stddev
     * @param seed Seed for the random number generator.
     * @param dinc_init Number of increments to wait to draw the first random force.
     * @param dinc @copybrief RandomNormalForcing::m_dinc
     */
    template <class T>
    RandomNormalForcing(
        std::array<size_type, rank> shape,
        double mean,
        double stddev,
        uint64_t seed,
        const T& dinc_init,
        const T& dinc)
    {
        m_shape = shape;
        m_N = std::accumulate(m_shape.cbegin(), m_shape.cend(), 1, std::multiplies<size_type>{});
        m_f_thermal = xt::zeros<double>(m_shape);
        m_rng.seed(seed);
        m_mean = mean;
        m_stddev = stddev;
        m_next = dinc_init;
        m_dinc = dinc;
    }

    /**
     * @brief Update forces based on current slips.
     * @param u_array Slip per particle.
     * @param f_array Force per particle (output: added to current result).
     * @param inc Current increment.
     */
    template <class T, class S>
    void force(const T& u_array, T& f_array, S inc)
    {
        (void)(u_array);

        for (size_type p = 0; p < m_N; ++p) {
            if (inc >= m_next(p)) {
                m_f_thermal(p) = m_rng.normal(m_mean, m_stddev);
                m_next(p) += m_dinc(p);
            }
        }

        f_array += m_f_thermal;
    }

    /**
     * @brief State of the random number generator.
     * @return State.
     */
    uint64_t state() const
    {
        return m_rng.state();
    }

    /**
     * @brief Change the state of the random number generator.
     * @param state State.
     */
    void set_state(uint64_t state)
    {
        m_rng.restore(state);
    }

    /**
     * @brief Current random force.
     * @return Array.
     */
    const auto& f_thermal() const
    {
        return m_f_thermal;
    }

    /**
     * @brief Change the random force.
     * @param f_thermal New random force.
     */
    void set_f_thermal(const array_type::tensor<double, rank>& f_thermal)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(f_thermal, m_f_thermal.shape()));
        m_f_thermal = f_thermal;
    }

    /**
     * @brief Next increment at which the random force is changed.
     * @return Array
     */
    const auto& next()
    {
        return m_next;
    }

    /**
     * @brief Overwrite the next increment at which the random force is changed.
     * @param next Next increment.
     */
    void set_next(const array_type::tensor<ptrdiff_t, rank>& next)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(next, m_next.shape()));
        m_next = next;
    }
};

/**
 * @brief A potential energy landscape of each particle that is piecewise smooth.
 * This corresponds to a piecewise linear force.
 */
template <class Generator>
class SemiSmooth {
protected:
    using stype = typename Generator::size_type; ///< Size type.
    stype m_N; ///< @copydoc Cuspy::m_N
    double m_mu; ///< @copydoc Cuspy::m_mu
    double m_kappa; ///< Softening stiffness.
    Generator* m_chunk; ///< @copydoc Cuspy::m_chunk

public:
    SemiSmooth() = default;

    /**
     * @copydoc Cuspy::Cuspy(double, Generator*)
     * @param kappa @copydoc SemiSmooth::m_kappa
     */
    SemiSmooth(double mu, double kappa, Generator* chunk) : m_mu(mu), m_kappa(kappa), m_chunk(chunk)
    {
        m_N = m_chunk->generators().size();
    }

    /**
     * @copydoc Cuspy::force
     */
    template <class T>
    void force(const T& u_array, T& f_array)
    {
        m_chunk->align(u_array);

        // this does not allocate data, but only creates a view
        xt::xtensor_pointer<const double, 2> yield = xt::adapt(
            m_chunk->data().data(),
            m_chunk->data().size(),
            xt::no_ownership(),
            std::array<stype, 2>{m_N, m_chunk->chunk_size()});

        // this does not allocate data, but only creates a view
        xt::xtensor_pointer<const ptrdiff_t, 1> i = xt::adapt(
            m_chunk->chunk_index_at_align().data(),
            m_chunk->chunk_index_at_align().size(),
            xt::no_ownership(),
            std::array<stype, 1>{m_N});

        for (stype p = 0; p < m_N; ++p) {
            auto* y = &yield(p, i(p));
            double xi = 0.5 * (*(y) + *(y + 1));
            double u_u = (m_mu * xi + m_kappa * *(y + 1)) / (m_mu + m_kappa);
            double u_l = (m_mu * xi + m_kappa * *(y)) / (m_mu + m_kappa);
            double u = u_array(p);
            if (u < u_l) {
                f_array(p) = m_kappa * (u - *(y));
            }
            else if (u <= u_u) {
                f_array(p) = m_mu * (0.5 * (*(y) + *(y + 1)) - u);
            }
            else {
                f_array(p) = m_kappa * (u - *(y + 1));
            }
        }
    }

    /**
     * @copydoc Cuspy::maxUniformDisplacement
     */
    template <class T>
    double maxUniformDisplacement(const T& u_array, int direction) const
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(direction == 1 || direction == -1);
        m_chunk->align(u_array);

        bool positive = direction > 0;
        std::vector<double> du;
        du.reserve(m_N);

        // this does not allocate data, but only creates a view
        xt::xtensor_pointer<const double, 2> yield = xt::adapt(
            m_chunk->data().data(),
            m_chunk->data().size(),
            xt::no_ownership(),
            std::array<stype, 2>{m_N, m_chunk->chunk_size()});

        // this does not allocate data, but only creates a view
        xt::xtensor_pointer<const ptrdiff_t, 1> i = xt::adapt(
            m_chunk->chunk_index_at_align().data(),
            m_chunk->chunk_index_at_align().size(),
            xt::no_ownership(),
            std::array<stype, 1>{m_N});

        for (stype p = 0; p < m_N; ++p) {
            auto* y = &yield(p, i(p));
            double xi = 0.5 * (*(y) + *(y + 1));
            double u_u = (m_mu * xi + m_kappa * *(y + 1)) / (m_mu + m_kappa);
            double u_l = (m_mu * xi + m_kappa * *(y)) / (m_mu + m_kappa);
            double u = u_array(p);

            if (u < u_l) {
                return 0.0;
            }
            else if (u <= u_u) {
                if (positive) {
                    du.push_back(u_u - u);
                }
                else {
                    du.push_back(u - u_l);
                }
            }
            else {
                return 0.0;
            }
        }

        return *std::min_element(du.begin(), du.end());
    }

    /**
     * @copydoc Cuspy::trigger
     */
    template <class T>
    void trigger(T& u_array, size_t p, double eps, int direction) const
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(p < m_N);

        if (direction > 0) {
            u_array.flat(p) = m_chunk->template right_of_align<T>().flat(p) + 0.5 * eps;
        }
        else {
            u_array.flat(p) = m_chunk->template left_of_align<T>().flat(p) - 0.5 * eps;
        }
    }
};

/**
 * @brief A potential energy landscape of each particle that is smooth.
 */
template <class Generator>
class Smooth {
protected:
    using stype = typename Generator::size_type; ///< Size type.
    stype m_N; ///< @copydoc Cuspy::m_N
    double m_mu; ///< @copydoc Cuspy::m_mu
    Generator* m_chunk; ///< @copydoc Cuspy::m_chunk

public:
    Smooth() = default;

    /**
     * @param mu @copydoc Cuspy::m_mu
     * @param chunk @copydoc Cuspy::m_chunk
     */
    Smooth(double mu, Generator* chunk) : m_mu(mu), m_chunk(chunk)
    {
        m_N = m_chunk->generators().size();
    }

    /**
     * @copydoc Cuspy::force
     */
    template <class T>
    void force(const T& u_array, T& f_array)
    {
        m_chunk->align(u_array);

        // this does not allocate data, but only creates a view
        xt::xtensor_pointer<const double, 2> yield = xt::adapt(
            m_chunk->data().data(),
            m_chunk->data().size(),
            xt::no_ownership(),
            std::array<stype, 2>{m_N, m_chunk->chunk_size()});

        // this does not allocate data, but only creates a view
        xt::xtensor_pointer<const ptrdiff_t, 1> i = xt::adapt(
            m_chunk->chunk_index_at_align().data(),
            m_chunk->chunk_index_at_align().size(),
            xt::no_ownership(),
            std::array<stype, 1>{m_N});

        for (stype p = 0; p < m_N; ++p) {
            auto* y = &yield(p, i(p));
            double u = u_array(p);
            double umin = 0.5 * (*(y) + *(y + 1));
            double dy = 0.5 * (*(y + 1) - *(y));
            f_array(p) = -m_mu * dy / M_PI * std::sin(M_PI * (u - umin) / dy);
        }
    }

    /**
     * @copydoc Cuspy::maxUniformDisplacement
     */
    template <class T>
    double maxUniformDisplacement(const T& u_array, int direction) const
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(direction == 1 || direction == -1);
        (void)(u_array);
        (void)(direction);
        throw std::runtime_error("Operation not possible.");
        return 0.0;
    }

    /**
     * @copydoc Cuspy::trigger
     */
    template <class T>
    void trigger(T& u_array, size_t p, double eps, int direction) const
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(p < m_N);

        if (direction > 0) {
            u_array.flat(p) = m_chunk->template right_of_align<T>().flat(p) + 0.5 * eps;
        }
        else {
            u_array.flat(p) = m_chunk->template left_of_align<T>().flat(p) - 0.5 * eps;
        }
    }
};

/**
 * @brief Short range elastic interactions with other particles.
 * The interactions are the short-range Laplacian \f$ \Delta u_i \f$:
 * \f$ f_\mathrm{inter}^{(i)} = k (u_{i - 1} - 2 u_i + u_{i + 1}) \f$.
 */
class Laplace1d {
protected:
    size_type m_N; ///< Number of particles.
    double m_k; ///< Stiffness of the interactions.

public:
    Laplace1d() = default;

    /**
     * @param k @copydoc Laplace1d::m_k
     * @param N @copydoc Laplace1d::m_N
     */
    Laplace1d(double k, size_type N) : m_N(N), m_k(k)
    {
    }

    /**
     * @copydoc Cuspy::force
     */
    template <class T>
    void force(const T& u_array, T& f_array)
    {
        for (size_type p = 1; p < m_N - 1; ++p) {
            f_array(p) = u_array(p - 1) - 2 * u_array(p) + u_array(p + 1);
        }
        f_array.front() = u_array.back() - 2 * u_array.front() + u_array(1);
        f_array.back() = u_array(m_N - 2) - 2 * u_array.back() + u_array.front();

        f_array *= m_k;
    }
};

/**
 * @brief Short range interactions based on the Laplacian \f$ \Delta u_{i, j} \f$.
 *
 * Each particle has still one degree of freedom, but interacts with four neighbours in a '2d' grid.
 * The particle numbering is assumed to be as follows:
 *
 *      0  1  2  3  4
 *      5  6  7  8  9
 *     10 11 12 13 14
 *     15 16 17 18 19
 *     20 21 22 23 24
 *
 * whereby the interaction kernal is:
 *
 *        +1
 *     +1 -4 +1
 *        +1
 *
 * such that
 * \f$ f_{i,j} = k (u_{i - 1, j} + u_{i + 1, j} + u_{i, j - 1} + u_{i, j + 1} - 4 u_{i, j}) \f$.
 *
 * ## Indices
 *
 * From Python you can access the indices as follows:
 *
 *      # array with 'spatial' organisation of the flat indices of particles
 *      organisation = np.arange(rows * cols).reshape(rows, cols)
 *
 *      # flat particle indices of neighbours to all sides
 *      down = np.roll(organisation, -1, axis=0)
 *      up = np.roll(organisation, 1, axis=0)
 *      left = np.roll(organisation, 1, axis=1)
 *      right = np.roll(organisation, -1, axis=1)
 */
class Laplace2d {
protected:
    size_type m_rows; ///< Number of rows.
    size_type m_cols; ///< Number of columns.
    double m_k; ///< @copydoc Laplace1d::m_k

public:
    Laplace2d() = default;

    /**
     * @param k @copydoc Laplace1d::m_k
     * @param rows @copydoc Laplace2d::m_rows
     * @param cols @copydoc Laplace2d::m_cols
     */
    Laplace2d(double k, size_type rows, size_type cols) : m_rows(rows), m_cols(cols), m_k(k)
    {
    }

    /**
     * @copydoc Cuspy::force
     */
    template <class T>
    void force(const T& u_array, T& f_array)
    {
        for (size_type i = 0; i < m_rows; ++i) {
            for (size_type j = 0; j < m_cols; ++j) {
                f_array(i, j) = u_array.periodic(i - 1, j) + u_array.periodic(i + 1, j) +
                                u_array.periodic(i, j - 1) + u_array.periodic(i, j + 1) -
                                4 * u_array(i, j);
            }
        }

        f_array *= m_k;
    }
};

/**
 * @brief Short range interaction based on a quartic potential.
 * The idea is to have a potential
 *
 * \f$ E(r) = \frac{1}{2} k_2 (\varepsilon(r))^2 + \frac{1}{12} k_4 (\varepsilon(r))^4 \f$
 *
 * with
 *
 * \f$ \varepsilon(r) = \frac{\partial u}{\partial r} \f$
 *
 * such that
 *
 * \f$ \sigma(r) = \frac{\partial E}{\partial \varepsilon} =
 * k_2 \varepsilon(r) + \frac{k_4}{3} \varepsilon(r)^3 \f$
 *
 * such that
 *
 * \f$ f(r) = \frac{\partial \sigma}{\partial r} =
 * \frac{\partial \varepsilon}{\partial r}(k_2 + k_4 \varepsilon(r)^2) \f$
 *
 * which is discretised as
 *
 * \f$ f_\mathrm{inter}^{(i)} = \left( u_{i + 1} + u_{i - 1} - 2 u_i \right)
 * \left( k_2 + k_4 / 4 (u_{i + 1} - u_{i - 1})^2 \right)\f$
 *
 * where the gradient term is approximated by central difference
 * (which leads to \f$ 1/4 \f$ factor).
 *
 * Please note that \f$ k_4 \f$ does not have the same units as \f$ k_2 \f$.
 */
class QuarticGradient1d {
protected:
    size_type m_N; ///< Number of particles.
    double m_k2; ///< Stiffness of the interactions.
    double m_k4; ///< Stiffness of the interactions.

public:
    QuarticGradient1d() = default;

    /**
     * @param k2 @copydoc QuarticGradient1d::m_k2
     * @param k4 @copydoc QuarticGradient1d::m_k4
     * @param N @copydoc QuarticGradient1d::m_N
     */
    QuarticGradient1d(double k2, double k4, size_type N) : m_N(N), m_k2(k2), m_k4(k4)
    {
    }

    /**
     * @copydoc Cuspy::force
     */
    template <class T>
    void force(const T& u_array, T& f_array)
    {
        for (size_type p = 1; p < m_N - 1; ++p) {
            double du = 0.5 * (u_array(p + 1) - u_array(p - 1));
            f_array(p) =
                (u_array(p - 1) - 2 * u_array(p) + u_array(p + 1)) * (m_k2 + m_k4 * (du * du));
        }

        double duf = 0.5 * (u_array(1) - u_array.back());
        f_array.front() =
            (u_array.back() - 2 * u_array.front() + u_array(1)) * (m_k2 + m_k4 * (duf * duf));

        double dub = 0.5 * (u_array.front() - u_array(m_N - 2));
        f_array.back() =
            (u_array(m_N - 2) - 2 * u_array.back() + u_array.front()) * (m_k2 + m_k4 * (dub * dub));
    }
};

/**
 * @brief Short range interactions based on quartic interactions.
 */
class QuarticGradient2d {
protected:
    size_type m_rows; ///< @copydoc Laplace2d::m_rows
    size_type m_cols; ///< @copydoc Laplace2d::m_cols
    double m_k2; ///< @copydoc QuarticGradient1d::m_k2
    double m_k4; ///< @copydoc QuarticGradient1d::m_k4

public:
    QuarticGradient2d() = default;

    /**
     * @param k2 @copydoc QuarticGradient1d::m_k2
     * @param k4 @copydoc QuarticGradient1d::m_k4
     * @param rows @copydoc Laplace2d::m_rows
     * @param cols @copydoc Laplace2d::m_cols
     */
    QuarticGradient2d(double k2, double k4, size_type rows, size_type cols)
        : m_rows(rows), m_cols(cols), m_k2(k2), m_k4(k4)
    {
    }

    /**
     * @copydoc Cuspy::force
     */
    template <class T>
    void force(const T& u_array, T& f_array)
    {
        double mk4_3 = m_k4 / 3.0;
        double mk4_23 = 2.0 * mk4_3;

        for (size_type i = 0; i < m_rows; ++i) {
            for (size_type j = 0; j < m_cols; ++j) {
                double l = u_array.periodic(i + 1, j) + u_array.periodic(i - 1, j) +
                           u_array.periodic(i, j + 1) + u_array.periodic(i, j - 1) -
                           4 * u_array(i, j);
                double dudx = 0.5 * (u_array.periodic(i + 1, j) - u_array.periodic(i - 1, j));
                double dudy = 0.5 * (u_array.periodic(i, j + 1) - u_array.periodic(i, j - 1));
                double d2udxdy =
                    0.25 * (u_array.periodic(i + 1, j + 1) - u_array.periodic(i + 1, j - 1) -
                            u_array.periodic(i - 1, j + 1) + u_array.periodic(i - 1, j - 1));
                double d2udx2 =
                    u_array.periodic(i + 1, j) - 2 * u_array(i, j) + u_array.periodic(i - 1, j);
                double d2udy2 =
                    u_array.periodic(i, j + 1) - 2 * u_array(i, j) + u_array.periodic(i, j - 1);

                f_array(i, j) =
                    l * (m_k2 + mk4_3) + mk4_23 * (dudx * dudx * d2udx2 + dudy * dudy * d2udy2 +
                                                   2.0 * dudx * dudy * d2udxdy);
            }
        }
    }
};

/**
 * @brief Short range interaction based on a quartic potential.
 * See e.g. https://doi.org/10.1103/PhysRevLett.87.187002
 */
class Quartic1d {
protected:
    size_type m_N; ///< Number of particles.
    double m_a1; ///< Stiffness of the interactions.
    double m_a2; ///< Stiffness of the interactions.

public:
    Quartic1d() = default;

    /**
     * @param a1 @copydoc Quartic1d::m_a1
     * @param a2 @copydoc Quartic1d::m_a2
     * @param N @copydoc Quartic1d::m_N
     */
    Quartic1d(double a1, double a2, size_type N) : m_N(N), m_a1(a1), m_a2(a2)
    {
    }

    /**
     * @copydoc Cuspy::force
     */
    template <class T>
    void force(const T& u_array, T& f_array)
    {
        for (size_type p = 1; p < m_N - 1; ++p) {
            double dup = std::pow(u_array(p + 1) - u_array(p), 3.0);
            double dun = std::pow(u_array(p - 1) - u_array(p), 3.0);
            f_array(p) =
                m_a1 * (u_array(p - 1) - 2 * u_array(p) + u_array(p + 1)) + m_a2 * (dup + dun);
        }

        {
            double dup = std::pow(u_array(1) - u_array.front(), 3.0);
            double dun = std::pow(u_array.back() - u_array.front(), 3.0);
            f_array.front() =
                m_a1 * (u_array.back() - 2 * u_array.front() + u_array(1)) + m_a2 * (dup + dun);
        }

        {
            double dup = std::pow(u_array.front() - u_array.back(), 3.0);
            double dun = std::pow(u_array(m_N - 2) - u_array.back(), 3.0);
            f_array.back() = m_a1 * (u_array(m_N - 2) - 2 * u_array.back() + u_array.front()) +
                             m_a2 * (dup + dun);
        }
    }
};

/**
 * @brief Long range interactions.
 * The interactions decay as \f$ 1 / r^{d + \alpha} \f$.
 * See e.g. https://doi.org/10.1103/PhysRevLett.126.025702
 */
class LongRange1d {
protected:
    size_type m_N; ///< Number of particles.
    double m_k; ///< @copybrief detail::Laplace1d::m_k
    double m_alpha; ///< Range of interactions.
    ptrdiff_t m_n; ///< Alias of m_N.
    ptrdiff_t m_m; ///< Midpoint.
    array_type::tensor<double, 1> m_prefactor; ///< Prefactor for long-range interactions.

public:
    LongRange1d() = default;

    /**
     * @param k @copydoc LongRange1d::m_k
     * @param alpha @copydoc LongRange1d::m_alpha
     * @param N @copydoc LongRange1d::m_N
     */
    LongRange1d(double k, double alpha, size_type N) : m_N(N), m_k(k), m_alpha(alpha)
    {
        m_n = static_cast<ptrdiff_t>(m_N);
        m_m = (m_n - m_n % 2) / 2;

        m_prefactor = xt::empty<double>({m_N});

        for (ptrdiff_t d = 0; d < m_n; ++d) {
            if (d == 0) {
                m_prefactor(0) = 0.0;
            }
            else {
                m_prefactor(d) = m_k / std::pow(d, m_alpha + 1.0);
            }
        }
    }

    /**
     * @copydoc Cuspy::force
     */
    template <class T>
    void force(const T& u_array, T& f_array)
    {
        for (ptrdiff_t p = 0; p < m_n; ++p) {
            double f = 0.0;
            double u = u_array(p);
            for (ptrdiff_t i = 0; i < m_n; ++i) {
                if (i == p) {
                    continue;
                }
                ptrdiff_t d = std::abs(i - p);
                if (d > m_m) {
                    d = m_n - d;
                }
                f += (u_array(i) - u) * m_prefactor(d);
            }
            f_array(p) = f;
        }
    }
};

/**
 * @brief System in generic number of dimensions.
 *
 * ## Internal strategy
 *
 * Internally, everything is kept up-to-date by calling:
 *
 * -    updated_u() every time #m_u is updated
 * -    updated_v() every time #m_v is updated
 *
 * In addition refresh() can be called to update everything.
 *
 * The suggested usage for deriving classes is to:
 *
 * -    do the same
 * -    only use set_u() and set_v() to update #m_u and #m_v
 */
template <size_t rank, class Potential, class Generator, class Interactions, class External = void>
class System {
public:
    virtual ~System() = default;

    /**
     * @brief Number of particles.
     * @return Unsigned int
     */
    size_t N() const
    {
        return m_N;
    }

    /**
     * @brief Shape of the system.
     * @return Array of unsigned integers.
     */
    const auto& shape() const
    {
        return m_shape;
    }

    /**
     * @brief Class that generates and external force that is add to the residual force.
     * @return Reference.
     */
    const auto& external() const
    {
        return m_external;
    }

    /**
     * @brief Set time.
     *
     * Internally only an increment number is stored.
     * This function simply converts the time to an increment number.
     *
     * @param arg Float.
     */
    void set_t(double arg)
    {
        m_inc = static_cast<decltype(m_inc)>(std::round(arg / m_dt));
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(xt::allclose(this->t(), arg));
    }

    /**
     * @brief Set increment number.
     * @param arg Singed integer.
     */
    void set_inc(ptrdiff_t arg)
    {
        m_inc = arg;
        m_qs_inc_first = arg;
        m_qs_inc_last = arg;
    }

    /**
     * @brief Set position of the load frame.
     * @param arg Float
     */
    void set_u_frame(double arg)
    {
        m_u_frame = arg;
        this->computeForceFrame();
        this->computeForce();
    }

    /**
     * @brief Position of the load frame.
     * @return Float
     */
    double u_frame() const
    {
        return m_u_frame;
    }

    /**
     * @brief Set the slip ('position') of each particle.
     *
     * This updates the appropriate forces.
     *
     * @param arg The particles' slips.
     */
    void set_u(const array_type::tensor<double, rank>& arg)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, m_shape));
        xt::noalias(m_u) = arg;
        this->updated_u();
    }

    /**
     * @brief Set the velocity of each particle (the first time derivative of the slip).
     *
     * This updates the appropriate forces.
     *
     * @param arg The particles' velocities.
     */
    void set_v(const array_type::tensor<double, rank>& arg)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, m_shape));
        xt::noalias(m_v) = arg;
        this->updated_v();
    }

    /**
     * @brief Set the acceleration of each particle (the second time derivative of the slip).
     * @param arg The particles' accelerations.
     */
    void set_a(const array_type::tensor<double, rank>& arg)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, m_shape));
        xt::noalias(m_a) = arg;
    }

    /**
     * @brief Recompute all forces.
     */
    void refresh()
    {
        this->computeForcePotential();
        this->computeForceInteractions();
        this->computeForceFrame();
        this->computeForceDamping();
        this->computeForce();
    }

    /**
     * @brief Slip ('position') of each particle.
     * @return Array of floats.
     */
    const auto& u() const
    {
        return m_u;
    }

    /**
     * @brief Velocity of each particle.
     * @return Array of floats.
     */
    const auto& v() const
    {
        return m_v;
    }

    /**
     * @brief Acceleration of each particle.
     * @return Array of floats.
     */
    const auto& a() const
    {
        return m_a;
    }

    /**
     * @brief Resultant force acting on each particle.
     * @return Array of floats.
     */
    const auto& f() const
    {
        return m_f;
    }

    /**
     * @brief Force associated to potentials acting on each particle.
     * @return Array of floats.
     */
    const auto& f_potential() const
    {
        return m_f_potential;
    }

    /**
     * @brief Force associated to the load frame acting on each particle.
     * @return Array of floats.
     */
    const auto& f_frame() const
    {
        return m_f_frame;
    }

    /**
     * @brief Force associated to interactions between particles.
     * @return Array of floats.
     */
    const auto& f_interactions() const
    {
        return m_f_interactions;
    }

    /**
     * @brief Force associated to damping on each particle.
     * @return Array of floats.
     */
    const auto& f_damping() const
    {
        return m_f_damping;
    }

    /**
     * @brief The time.
     *
     * Note that the class uses the increment number internally.
     *
     * @return Float.
     */
    auto t() const
    {
        return static_cast<double>(m_inc) * m_dt;
    }

    /**
     * @brief The increment number.
     * @return Signed integer.
     */
    auto inc() const
    {
        return m_inc;
    }

    /**
     * @brief The instantaneous temperature.
     *
     * The following definition is used:
     * \f$ T \equiv m / 2 \sum\limits_{i = 1}^N v_i^2 / N \f$
     * such that temperature is defined in units of Boltzmann's constant.
     *
     * @return Float.
     */
    double temperature() const
    {
        return 0.5 * m_m * xt::norm_sq(m_v)() / static_cast<double>(m_N);
    }

    /**
     * @brief Residual.
     *
     * Tthe ratio between the norm of f() and f_frame().
     *
     * @return Float.
     */
    double residual() const
    {
        double r_fres = xt::norm_l2(m_f)();
        double r_fext = xt::norm_l2(m_f_frame)();
        if (r_fext != 0.0) {
            return r_fres / r_fext;
        }
        return r_fres;
    }

    /**
     * @brief Set velocities and accelerations to zero.
     *
     * Call this function after an energy minimisation (taken care of in minimise()).
     */
    void quench()
    {
        m_v.fill(0.0);
        m_a.fill(0.0);
        this->updated_v();
    }

    /**
     * @brief Effectuate one time step using the velocity Verlet algorithm.
     *
     * Updates the slip, velocity acceleration, and all forces of each particle.
     */
    void timeStep()
    {
        m_inc++;
        xt::noalias(m_v_n) = m_v;
        xt::noalias(m_a_n) = m_a;

        xt::noalias(m_u) = m_u + m_dt * m_v + 0.5 * std::pow(m_dt, 2.0) * m_a;
        this->updated_u();

        xt::noalias(m_v) = m_v_n + m_dt * m_a_n;
        this->updated_v();

        xt::noalias(m_a) = m_f * m_inv_m;

        xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
        this->updated_v();

        xt::noalias(m_a) = m_f * m_inv_m;

        xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
        this->updated_v();

        xt::noalias(m_a) = m_f * m_inv_m;

        if (xt::any(xt::isnan(m_u))) {
            throw std::runtime_error("NaN entries found");
        }
    }

    /**
     * @brief Make a number of time steps, see timeStep().
     *
     * @param n Number of steps to make.
     */
    void timeSteps(size_t n)
    {
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(n + 1 < std::numeric_limits<long>::max());
        for (size_t step = 0; step < n; ++step) {
            this->timeStep();
        }
    }

    /**
     * @brief Perform a series of time-steps until the next plastic event, or equilibrium.
     *
     * @param tol Relative force tolerance for equilibrium. See residual() for definition.
     * @param niter_tol Enforce the residual check for `niter_tol` consecutive increments.
     * @param max_iter Maximum number of iterations. Throws `std::runtime_error` otherwise.
     * @return
     *      -   Number of steps.
     *      -   `0` if there was no plastic activity and the residual was reached.
     */
    size_t timeStepsUntilEvent(double tol = 1e-5, size_t niter_tol = 10, size_t max_iter = 1e9)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(tol < 1.0);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(max_iter + 1 < std::numeric_limits<long>::max());

        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);
        auto i_n = m_chunk->index_at_align();
        size_t step;

        for (step = 1; step < max_iter + 1; ++step) {

            this->timeStep();

            if (xt::any(xt::not_equal(m_chunk->index_at_align(), i_n))) {
                return step;
            }

            residuals.roll_insert(this->residual());

            if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
                this->quench();
                return 0;
            }
        }

        return step;
    }

    /**
     * @brief Make a number of steps with the frame moving at a constant velocity.
     *
     * Make a number of steps with the following protocol.
     * 1.  Add a step \f$ v_\text{frame} \Delta t \f$ to the frame.
     * 2.  Make a timeStep().
     *
     * @param n
     *     Number of steps to make.
     *
     * @param v_frame
     *     Velocity of the frame.
     */
    void flowSteps(size_t n, double v_frame)
    {
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(n + 1 < std::numeric_limits<long>::max());

        for (size_t step = 0; step < n; ++step) {
            m_u_frame += v_frame * m_dt;
            this->timeStep();
        }
    }

    /**
     * @brief
     *      Minimise energy: run timeStep() until a mechanical equilibrium has been reached.
     *
     * @param tol
     *      Relative force tolerance for equilibrium. See residual() for definition.
     *
     * @param niter_tol
     *      Enforce the residual check for `niter_tol` consecutive increments.
     *
     * @param max_iter
     *      Maximum number of time-steps. Throws `std::runtime_error` otherwise.
     *
     * @param time_activity
     *      If `true` plastic activity is timed. After this function you can find:
     *      -   quasistaticActivityFirst() : Increment with the first plastic event.
     *      -   quasistaticActivityLast() : Increment with the last plastic event.
     *      Attention: if you are changing the chunk of yield positions during the minimisation you
     *      should copy quasistaticActivityFirst() after the first (relevant) call of minimise():
     *      each time you call minimise(), quasistaticActivityFirst() is reset.
     *
     * @param max_iter_is_error
     *      If `true` an error is thrown when the maximum number of time-steps is reached.
     *      If `false` the function simply returns `max_iter`.
     *
     * @return
     *      -   `0`: if stopped when the residual is reached (and number of steps `< max_iter`).
     *      -   `max_iter`: if no residual was reached, and `max_iter_is_error = false`.
     */
    virtual size_t minimise(
        double tol = 1e-5,
        size_t niter_tol = 10,
        size_t max_iter = 1e9,
        bool time_activity = false,
        bool max_iter_is_error = true)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(tol < 1.0);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(max_iter + 1 < std::numeric_limits<long>::max());

        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);

        array_type::tensor<ptrdiff_t, 1> i_n;
        long s = 0;
        long s_n = 0;
        bool init = true;
        size_t step;

        if (time_activity) {
            i_n = m_chunk->index_at_align();
        }

        for (step = 1; step < max_iter + 1; ++step) {

            this->timeStep();
            residuals.roll_insert(this->residual());

            if (time_activity) {
                s = xt::sum(xt::abs(m_chunk->index_at_align() - i_n))();
                if (s != s_n) {
                    if (init) {
                        init = false;
                        m_qs_inc_first = m_inc;
                    }
                    m_qs_inc_last = m_inc;
                }
                s_n = s;
            }

            if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
                this->quench();
                return 0;
            }
        }

        if (max_iter_is_error) {
            throw std::runtime_error("No convergence found");
        }

        return step;
    }

    /**
     * @brief Increment with the first plastic event.
     *
     * This value is only relevant if `time_activity = true` was used in the last call of
     * minimise().
     *
     * @return Increment.
     */
    size_t quasistaticActivityFirst() const
    {
        return m_qs_inc_first;
    }

    /**
     * @brief Increment with the last plastic event.
     *
     * This value is only relevant if `time_activity = true` was used in the last call of
     * minimise().
     *
     * @return Increment.
     */
    size_t quasistaticActivityLast() const
    {
        return m_qs_inc_last;
    }

    /**
     * @brief Find maximum particle displacement for which the system is linear and uniform.
     *
     * @param direction If `+1`: move right. If `-1` move left.
     * @return Float: particle displacement.
     */
    double maxUniformDisplacement(int direction)
    {
        return m_potential->maxUniformDisplacement(m_u, direction);
    };

    /**
     * @brief Make event driven step.
     *
     *  -   `kick = false`: Increment the position of the load-frame and that of the particles to a
     *      new mechanical equilibrium just before yielding (if `direction = 1`, the new position
     *      for particle `p` closest to yielding to the right is `x[p] = y[p] - eps / 2`).
     *      This assumes incrementing the load-frame infinitely slowly such that,
     *      because there is no yielding, the equilibrium configuration for a new position of the
     *      load frame is known.
     *
     *  -   `kick = true` : Advance the system uniformly
     *      (the particles and the frame are moved proportionally) such that the particle closest to
     *      yielding is brought just past yielding (if `direction = 1`, the new position for
     *      particle `p` closest to yielding right is `x[p] = y[p] + eps / 2`).
     *
     * @param eps
     *     Margin to keep to the position to the closest yield position.
     *
     * @param kick
     *     If `false`, the increment is elastic (no minimisation has to be applied after).
     *     If `true`, the increment leads to a state out of mechanical equilibrium.
     *
     * @param direction If `+1`: move right. If `-1` move left.
     *
     * @return Position increment of the frame.
     */
    double eventDrivenStep(double eps, bool kick, int direction = 1)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(direction == 1 || direction == -1);

        if (direction > 0 && !kick) {
            double du = this->maxUniformDisplacement(direction);
            if (du < 0.5 * eps) {
                return 0.0;
            }
            return this->advanceUniformly(du - 0.5 * eps, false);
        }

        if (direction > 0 && kick) {
            return this->advanceUniformly(eps, false);
        }

        // direction < 0

        if (!kick) {
            double du = this->maxUniformDisplacement(direction);
            if (du < 0.5 * eps) {
                return 0.0;
            }
            return this->advanceUniformly(0.5 * eps - du, false);
        }

        return this->advanceUniformly(-eps, false);
    }

    /**
     * @brief Trigger a specific particle.
     *
     * Advance to the yield position right plus a margin of `eps / 2`,
     * or to the left minus a margin `eps / 2`.
     *
     * @param p Particle index.
     * @param eps Margin.
     * @param direction If `+1`: move right. If `-1` move left.
     */
    void trigger(size_t p, double eps, int direction = 1)
    {
        return m_potential->trigger(m_u, p, eps, direction);
        this->updated_u();
    };

    /**
     * @brief Change the position of the particles and of the loading frame such that
     * the mean of f_frame() is equal to a target value, and mechanical equilibrium is maintained.
     *
     * @warning Assumes mechanical equilibrium. No assertions are made on this.
     */
    void advanceToFixedForce(double f_frame)
    {
        auto i_n = m_chunk->index_at_align();
        this->advanceUniformly((f_frame - xt::mean(m_f_frame)()) / m_mu, false);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::all(xt::equal(m_chunk->index_at_align(), i_n)));
    }

protected:
    /**
     * @brief Initialise the system.
     *
     * @param m @copybrief detail::System::m_m
     * @param eta @copybrief detail::System::m_eta
     * @param k_frame @copybrief detail::System::m_k_frame
     * @param mu @copybrief detail::System::m_mu
     * @param dt @copybrief detail::System::m_dt
     * @param shape @copybrief detail::System::m_shape
     * @param potential @copybrief detail::System::m_potential
     * @param chunk @copybrief detail::System::m_chunk
     * @param interactions @copybrief detail::System::m_interactions
     * @param external @copybrief detail::System::m_external
     */
    void initSystem(
        double m,
        double eta,
        double k_frame,
        double mu,
        double dt,
        std::array<size_type, rank> shape,
        Potential* potential,
        Generator* chunk,
        Interactions* interactions,
        External* external = nullptr)
    {
        m_shape = shape;
        m_N = static_cast<size_type>(m_chunk->generators().size());
        m_m = m;
        m_inv_m = 1.0 / m;
        m_eta = eta;
        m_mu = mu;
        m_k_frame = k_frame;
        m_u_frame = 0.0;
        m_dt = dt;
        m_f = xt::zeros<double>(m_shape);
        m_f_potential = xt::zeros<double>(m_shape);
        m_f_interactions = xt::zeros<double>(m_shape);
        m_f_frame = xt::zeros<double>(m_shape);
        m_f_damping = xt::zeros<double>(m_shape);
        m_u = xt::zeros<double>(m_shape);
        m_v = xt::zeros<double>(m_shape);
        m_a = xt::zeros<double>(m_shape);
        m_v_n = xt::zeros<double>(m_shape);
        m_a_n = xt::zeros<double>(m_shape);
        m_potential = potential;
        m_chunk = chunk;
        m_interactions = interactions;
        m_external = external;
        this->refresh();
    }

protected:
    /**
     * @brief Compute residual force.
     */
    void computeForce()
    {
        xt::noalias(m_f) = m_f_potential + m_f_interactions + m_f_damping + m_f_frame;
        if constexpr (!std::is_same<External, void>::value) {
            m_external->force(m_u, m_f, m_inc);
        }
    }

    /**
     * @brief Compute force due to the potential energy.
     */
    void computeForcePotential()
    {
        m_potential->force(m_u, m_f_potential);
    }

    /**
     * @brief Compute force due to interactions between particles.
     */
    void computeForceInteractions()
    {
        m_interactions->force(m_u, m_f_interactions);
    }

    /**
     * @brief Compute force due to the loading frame.
     */
    void computeForceFrame()
    {
        xt::noalias(m_f_frame) = m_k_frame * (m_u_frame - m_u);
    }

    /**
     * @brief Compute force due to damping.
     */
    void computeForceDamping()
    {
        xt::noalias(m_f_damping) = -m_eta * m_v;
    }

    /**
     * @brief Update forces that depend on slip.
     */
    void updated_u()
    {
        this->computeForcePotential();
        this->computeForceInteractions();
        this->computeForceFrame();
        this->computeForce();
    }

    /**
     * @brief Update forces that depend on velocity.
     */
    void updated_v()
    {
        this->computeForceDamping();
        this->computeForce();
    }

    /**
     * @brief Advance the system uniformly.
     *
     * -    All particle move by the same amount.
     * -    The frame moves proportionally such that equilibrium is maintained
     *      (the change in `f_potential` is compensated by the change in `f_frame`).
     *
     * Thereby:
     *
     * \f$ \Delta f^\mathrm{potential}_i = \mu \Delta u_i \f$
     *
     * \f$ \Delta f^\mathrm{frame} = k (\Delta u^\mathrm{frame} - \Delta u_i) \f$
     *
     * Setting \f$ \Delta f^\mathrm{potential}_i = \Delta f^\mathrm{frame} \f$ gives:
     *
     * -   Given \f$ \Delta u_i \f$ one finds:
     *     \f$ \Delta u^\mathrm{frame} = (k + \mu) / k \Delta u_i \f$.
     *
     * -   Given \f$ \Delta u^\mathrm{frame} \f$ one finds:
     *     \f$ \Delta u_i = k / (k + \mu) \Delta u^\mathrm{frame} \f$.
     *
     * @param du Displacement.
     *
     * @param input_is_frame
     *     If `true`: `du` \f$ = \Delta u^\mathrm{frame} \f$,
     *     if `false`: `du` \f$ = \Delta u_i \f$,
     *
     * @return `du` for the particles in `input_is_frame == true`, otherwise `du` of the frame.
     */
    double advanceUniformly(double du, bool input_is_frame = true)
    {
        double du_particles;
        double du_frame;

        if (input_is_frame) {
            du_frame = du;
            du_particles = du * m_k_frame / (m_k_frame + m_mu);
        }
        else {
            du_particles = du;
            du_frame = du * (m_k_frame + m_mu) / m_k_frame;
        }

        m_u += du_particles;
        m_u_frame += du_frame;
        this->updated_u();

        if (input_is_frame) {
            return du_particles;
        }

        return du_frame;
    }

protected:
    std::array<size_type, rank> m_shape; ///< @copybrief detail::System::shape
    size_type m_N; ///< @copybrief detail::System::N
    array_type::tensor<double, rank> m_f; ///< @copybrief detail::System::f
    array_type::tensor<double, rank> m_f_potential; ///< @copybrief detail::System::f_potential
    array_type::tensor<double, rank> m_f_interactions; ///< @copydoc detail::System::f_interactions
    array_type::tensor<double, rank> m_f_frame; ///< @copybrief detail::System::f_frame
    array_type::tensor<double, rank> m_f_damping; ///< @copybrief detail::System::f_damping
    array_type::tensor<double, rank> m_u; ///< @copybrief detail::System::u
    array_type::tensor<double, rank> m_v; ///< @copybrief detail::System::v
    array_type::tensor<double, rank> m_a; ///< @copybrief detail::System::a
    array_type::tensor<double, rank> m_v_n; ///< Temporary for integration.
    array_type::tensor<double, rank> m_a_n; ///< Temporary for integration.
    ptrdiff_t m_inc = 0; ///< @copybrief detail::System::inc
    ptrdiff_t m_qs_inc_first = 0; ///< @copybrief detail::System::quasistaticActivityFirst
    ptrdiff_t m_qs_inc_last = 0; ///< @copybrief detail::System::quasistaticActivityLast
    double m_dt; ///< Time step.
    double m_eta; ///< Damping constant (same for all particles).
    double m_m; ///< Mass (same for all particles).
    double m_inv_m; ///< `== 1 / m_m`
    double m_mu; ///< Curvature of the potentials.
    double m_k_frame; ///< Stiffness of the load fame (same for all particles).
    double m_u_frame = 0.0; ///< @copybrief detail::System::u_frame
    Potential* m_potential; ///< Class to get the forces from the local potential energy landscape.
    Generator* m_chunk; ///< @copybrief detail::Cuspy::m_chunk
    Interactions* m_interactions; ///< Class to get the forces from particle interaction.
    External* m_external; ///< Add an (time dependent) externally defined force to the residual.
};

} // namespace detail
} // namespace FrictionQPotSpringBlock

#endif
