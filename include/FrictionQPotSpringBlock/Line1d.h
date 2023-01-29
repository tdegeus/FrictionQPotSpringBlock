/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#ifndef FRICTIONQPOTSPRINGBLOCK_LINE1D_H
#define FRICTIONQPOTSPRINGBLOCK_LINE1D_H

#include <string>

#include <xtensor/xnorm.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xpad.hpp>

#include "config.h"

#include <GooseFEM/Iterate.h>
#include <GooseFEM/version.h>
#include <prrng.h>

#include <GMatTensor/version.h>

namespace FrictionQPotSpringBlock {

/**
 * Line in 1d.
 */
namespace Line1d {

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
    prrng::pcg32_tensor_cumsum<array_type::tensor<double, 2>, array_type::tensor<ptrdiff_t, 1>, 1>;

/**
 * ## Introduction
 *
 * The system in which particles experience a piece-wise quadratic local potential energy,
 * elastic interactions with neighbours, and a drive though a spring attached to a load frame.
 * The local landscape is characterised by yield positions.
 *
 * ## Physics & API
 *
 * The physics are as follows:
 * -   A particle has a mass \f$ m \f$.
 * -   Each particle \f$ i \f$ experiences damping equal to
 *     \f$ f_\text{damping}^{(i)} = - \eta v_i \f$, with the particle's velocity
 *     \f$ v_i \equiv \dot{x}_i \equiv \partial_t x_i \f$.
 * -   Each particle \f$ i \f$ has a potential energy such that
 *     \f$ f_\text{potential}^{(i)} =  \mu (x_{\min}^{(i)} - x_i) \f$.
 *     where \f$ \mu \f$ is the radius of curvature of the quadratic potentials, and
 *     \f$ x_{\min}^{(i)} \f$ the positions of the current local minimum.
 * -   Each particle \f$ i \f$ has interactions with its neighbours equal to
 *     \f$ f_\text{neighbours}^{(i)} =  k_\text{neighbours} (x_{i - 1} - 2 x_i + x_{i + 1}) \f$.
 * -   Each particle \f$ i \f$ is connected to the load frame given a force equal to
 *     \f$ f_\text{frame}^{(i)} =  k_\text{frame} (x_\text{frame} - x_i) \f$.
 *     Typically \f$ k_\text{frame} = \mathcal{O}(1 / N) \f$ with \f$ N \f$ the number of particles.
 *
 * The dynamics follow
 * \f$ m a_i = f_\text{residual}^{(i)} \f$, with the particle's acceleration
 * \f$ a_i \equiv \ddot{x}_i \equiv \partial_t^2 x_i \f$, and the residual force
 * \f$ f_\text{residual}^{(i)} = f_\text{damping}^{(i)} + f_\text{potential}^{(i)} +
 * f_\text{neighbours}^{(i)} + f_\text{frame}^{(i)} \f$.
 * This differential equation is integrated using Velocity-Verlet in timeStep()
 * Note that this function updates the time #t (or increment number #inc),
 * and that this is the only function that does so.
 *
 * The implementation is is such that all forces are updated (if needed) every time the positions
 * and velocities are updated, either internally in timeStep(), or externally when calling
 * set_x() and set_v().
 * In absolute need refresh() can be called to force re-computation of forces,
 * but this should normally not be needed.
 *
 * ## Yield positions
 *
 * For computational efficiency one can choose to store only a chunk of the sequence of yield
 * positions around the current position, which then has to updated when the particle is close to
 * running out-of-bounds, using set_y().
 *
 * ## Internal strategy
 *
 * Internally, updated_x() and updated_v() are called every time that the positions #m_x and the
 * velocities #m_v are updated.
 * Thereby, updated_x() updates position dependent forces,
 * and updated_v() updates velocity dependent forces.
 * Both finish by updating the residual force in computeForce().
 */
class System {

public:
    System() = default;

    virtual ~System() = default;

    /**
     * Constructor.
     *
     * @param m Particle mass (same for all particles).
     * @param eta Damping coefficient (same for all particles).
     * @param mu Elastic stiffness, i.e. the curvature of the potential (same for all particles).
     * @param k_neighbours Stiffness of the 'springs' connecting neighbours (same for all).
     * @param k_frame Stiffness of springs between particles and load frame (same for all).
     * @param dt Time step.
     * @param chunk Class in which chunks of the yield positions are stored (copy).
     */
    System(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        Generator* chunk)
    {
        this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, chunk);
    }

    /**
     * Number of particles.
     * @return unsigned int
     */
    size_t N() const
    {
        return m_N;
    }

    /**
     * @brief Organisation of the grid. To get e.g. the positions in the shape of the grid use:
     *
     *      idx = system.organisation
     *      x = system.x[idx]
     */
    virtual array_type::array<size_t> organisation() const
    {
        return xt::arange<size_t>(m_N);
    }

    /**
     * @brief Periodic index.
     *
     * @param index Index.
     * @return Index after applying periodicity.
     */
    ptrdiff_t periodic(ptrdiff_t index) const
    {
        ptrdiff_t n = static_cast<ptrdiff_t>(m_N);
        return (n + index % n) % n;
    }

    /**
     * Current global index of the potential energy landscape of each particle.
     * @return Array of shape [#N].
     */
    array_type::tensor<ptrdiff_t, 1> i() const
    {
        return m_chunk->index();
    }

    /**
     * Closest yield position right of the particle.
     * Convenience function: same as `system.y[np.arange(system.N), system.i + 1]`.
     * @return [#N].
     */
    array_type::tensor<double, 1> y_right() const
    {
        array_type::tensor<double, 1> ret = xt::empty<double>({m_N});

        for (size_t p = 0; p < m_N; ++p) {
            ret(p) = m_chunk->data()(p, m_chunk->chunk_index()(p) + 1);
        }

        return ret;
    }

    /**
     * Closest yield position left of the particle.
     * Convenience function: same as `system.y[np.arange(system.N), system.i]`.
     * @return [#N].
     */
    array_type::tensor<double, 1> y_left() const
    {
        array_type::tensor<double, 1> ret = xt::empty<double>({m_N});

        for (size_t p = 0; p < m_N; ++p) {
            ret(p) = m_chunk->data()(p, m_chunk->chunk_index()(p));
        }

        return ret;
    }

    /**
     * Set time.
     * @param arg double.
     */
    void set_t(double arg)
    {
        m_inc = static_cast<size_t>(std::round(arg / m_dt));
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(xt::allclose(this->t(), arg));
    }

    /**
     * Set increment.
     * @param arg size_t.
     */
    void set_inc(size_t arg)
    {
        m_inc = arg;
        m_qs_inc_first = arg;
        m_qs_inc_last = arg;
    }

    /**
     * Set position of the load frame.
     * @param arg double.
     */
    void set_x_frame(double arg)
    {
        m_x_frame = arg;
        this->computeForceFrame();
        this->computeForce();
    }

    /**
     * Position of the load frame.
     * @return double
     */
    double x_frame() const
    {
        return m_x_frame;
    }

    /**
     * Set the position of each particle.
     * This updates the appropriate forces.
     * As a rule of thumb this should be only way to update positions (even when deriving).
     *
     * @param arg The particles' positions [#N].
     */
    template <class T>
    void set_x(const T& arg)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, {m_N}));
        xt::noalias(m_x) = arg;
        this->updated_x();
    }

    /**
     * Set the velocity of each particle.
     * This updates the appropriate forces.
     * As a rule of thumb this should be only way to update positions (even when deriving).
     *
     * @param arg The particles' velocities [#N].
     */
    template <class T>
    void set_v(const T& arg)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, {m_N}));
        xt::noalias(m_v) = arg;
        this->updated_v();
    }

    /**
     * Set the acceleration of each particle.
     * @param arg The particles' accelerations [#N].
     */
    template <class T>
    void set_a(const T& arg)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, {m_N}));
        xt::noalias(m_a) = arg;
    }

    /**
     * Recompute all forces.
     */
    void refresh()
    {
        this->computeForcePotential();
        this->computeForceNeighbours();
        this->computeForceFrame();
        this->computeForceDamping();
        this->computeForce();
    }

    /**
     * Position of each particle.
     * @return [#N].
     */
    const array_type::tensor<double, 1>& x() const
    {
        return m_x;
    }

    /**
     * Velocity of each particle.
     * @return [#N].
     */
    const array_type::tensor<double, 1>& v() const
    {
        return m_v;
    }

    /**
     * Acceleration of each particle.
     * @return [#N].
     */
    const array_type::tensor<double, 1>& a() const
    {
        return m_a;
    }

    /**
     * Resultant force acting on each particle.
     * @return [#N].
     */
    const array_type::tensor<double, 1>& f() const
    {
        return m_f;
    }

    /**
     * Force associated to potentials acting on each particle.
     * @return [#N].
     */
    const array_type::tensor<double, 1>& f_potential() const
    {
        return m_f_potential;
    }

    /**
     * Force associated to the load frame acting on each particle.
     * @return [#N].
     */
    const array_type::tensor<double, 1>& f_frame() const
    {
        return m_f_frame;
    }

    /**
     * Force associated to neighbours acting on each particle.
     * @return [#N].
     */
    const array_type::tensor<double, 1>& f_neighbours() const
    {
        return m_f_neighbours;
    }

    /**
     * Force associated to damping on each particle.
     * @return [#N].
     */
    const array_type::tensor<double, 1>& f_damping() const
    {
        return m_f_damping;
    }

    /**
     * The time, see set_t().
     * @return double.
     */
    double t() const
    {
        return static_cast<double>(m_inc) * m_dt;
    }

    /**
     * The increment, see set_inc().
     * @return size_t.
     */
    size_t inc() const
    {
        return m_inc;
    }

    /**
     * The instantaneous temperature, defined as
     * \f$ T \equiv (m / N) \sum\limits_{i = 1}^N v_i^2 \f$
     * Note that by definition Boltzmann's constant is taken equal to 1.
     * @return double.
     */
    double temperature() const
    {
        return xt::norm_sq(m_v)() * m_m / m_N;
    }

    /**
     * Residual: the ratio between the norm of #f and #f_frame.
     * @return double.
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
     * Set #v and #a equal to zero.
     * Call this function after an energy minimisation (taken care of in minimise()).
     */
    void quench()
    {
        m_v.fill(0.0);
        m_a.fill(0.0);
        this->updated_v();
    }

    /**
     * Effectuates time step using the velocity Verlet algorithm.
     * Updates #x, #v, #a, and #f, and increment #inc.
     */
    void timeStep()
    {
        m_inc++;
        xt::noalias(m_v_n) = m_v;
        xt::noalias(m_a_n) = m_a;

        xt::noalias(m_x) = m_x + m_dt * m_v + 0.5 * m_dt * m_dt * m_a;
        this->updated_x();

        xt::noalias(m_v) = m_v_n + m_dt * m_a_n;
        this->updated_v();

        xt::noalias(m_a) = m_f * m_inv_m;

        xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
        this->updated_v();

        xt::noalias(m_a) = m_f * m_inv_m;

        xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
        this->updated_v();

        xt::noalias(m_a) = m_f * m_inv_m;

        if (xt::any(xt::isnan(m_x))) {
            throw std::runtime_error("NaN entries found");
        }
    }

    /**
     * Make a number of time steps, see timeStep().
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
     * Perform a series of time-steps until the next plastic event, or equilibrium.
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
        auto i_n = m_chunk->index();
        size_t step;

        for (step = 1; step < max_iter + 1; ++step) {

            this->timeStep();

            if (xt::any(xt::not_equal(m_chunk->index(), i_n))) {
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
            m_x_frame += v_frame * m_dt;
            this->timeStep();
        }
    }

    /**
     * Minimise energy: run timeStep() until a mechanical equilibrium has been reached.
     *
     * @param tol
     *     Relative force tolerance for equilibrium. See residual() for definition.
     *
     * @param niter_tol
     *     Enforce the residual check for `niter_tol` consecutive increments.
     *
     * @param max_iter
     *     Maximum number of time-steps. Throws `std::runtime_error` otherwise.
     *
     * @param time_activity
     *     If `true` plastic activity is timed. After this function you can find:
     *     -   quasistaticActivityFirst() : Increment with the first plastic event.
     *     -   quasistaticActivityLast() : Increment with the last plastic event.
     *     Attention: if you are changing the chunk of yield positions during the minimisation you
     *     should copy quasistaticActivityFirst() after the first (relevant) call of minimise():
     *     each time you call minimise(), quasistaticActivityFirst() is reset.
     *
     * @param max_iter_is_error
     *     If `true` an error is thrown when the maximum number of time-steps is reached.
     *     If `false` the function simply returns `max_iter`.
     *
     * @return
     *     -   `0`: if stopped when the residual is reached (and number of steps `< max_iter`).
     *     -   `max_iter`: if no residual was reached, and `max_iter_is_error = false`.
     */
    size_t minimise(
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
            i_n = this->i();
        }

        for (step = 1; step < max_iter + 1; ++step) {

            this->timeStep();
            residuals.roll_insert(this->residual());

            if (time_activity) {
                s = xt::sum(xt::abs(this->i() - i_n))();
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
     * Increment with the first plastic event.
     * This value is only relevant if `time_activity = true` was used in the last call of
     * minimise().
     * @return Increment.
     */
    size_t quasistaticActivityFirst() const
    {
        return m_qs_inc_first;
    }

    /**
     * Increment with the last plastic event.
     * This value is only relevant if `time_activity = true` was used in the last call of
     * minimise().
     * @return Increment.
     */
    size_t quasistaticActivityLast() const
    {
        return m_qs_inc_last;
    }

    /**
     * Same as minimise() but assuming overdamped dynamics and using the no passing rule.
     *
     * @warning
     *     The increment is not updated as time is not physical. The mass and viscosity are ignored.
     */
    size_t minimise_nopassing(double tol = 1e-5, size_t niter_tol = 10, size_t max_iter = 1e9)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(tol < 1.0);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(max_iter + 1 < std::numeric_limits<long>::max());

        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);

        double xneigh;
        double x;
        double xmin;
        ptrdiff_t i;
        ptrdiff_t j;

        for (size_t step = 1; step < max_iter + 1; ++step) {

            // "misuse" unused variable
            xt::noalias(m_v_n) = m_x;

            for (size_t p = 0; p < m_N; ++p) {

                if (p == 0) {
                    xneigh = m_v_n.back() + m_v_n(1);
                }
                else if (p == m_N - 1) {
                    xneigh = m_v_n(m_N - 2) + m_v_n.front();
                }
                else {
                    xneigh = m_v_n(p - 1) + m_v_n(p + 1);
                }

                i = m_chunk->chunk_index()(p);
                auto* y = &m_chunk->data()(p, 0);

                while (true) {
                    xmin = 0.5 * (*(y + i) + *(y + i + 1));
                    x = (m_k_neighbours * xneigh + m_k_frame * m_x_frame + m_mu * xmin) /
                        (2 * m_k_neighbours + m_k_frame + m_mu);
                    m_chunk->align(p, x);
                    j = m_chunk->chunk_index()(p);
                    if (j == i) {
                        break;
                    }
                    i = j;
                }
                m_x(p) = x;
            }

            this->updated_x();
            residuals.roll_insert(this->residual());

            if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
                this->quench(); // no dynamics are run: make sure that the user is not confused
                return 0;
            }
        }

        throw std::runtime_error("No convergence found");
    }

    /**
     * @brief Find maximum particle displacement during for which the system is linear and uniform.
     * @param direction If `+1`: move right. If `-1` move left.
     * @return Particle displacement.
     */
    virtual double maxUniformDisplacement(int direction = 1)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(direction == 1 || direction == -1);

        if (direction > 0) {
            return xt::amin(this->y_right() - m_x)();
        }

        return xt::amin(m_x - this->y_left())();
    }

    /**
     * Make event driven step.
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
            double dx = this->maxUniformDisplacement(direction);
            if (dx < 0.5 * eps) {
                return 0.0;
            }
            return this->advanceUniformly(dx - 0.5 * eps, false);
        }

        if (direction > 0 && kick) {
            return this->advanceUniformly(eps, false);
        }

        // direction < 0

        if (!kick) {
            double dx = this->maxUniformDisplacement(direction);
            if (dx < 0.5 * eps) {
                return 0.0;
            }
            return this->advanceUniformly(0.5 * eps - dx, false);
        }

        return this->advanceUniformly(-eps, false);
    }

    /**
     * Trigger a specific particle:
     * advance to the yield position right plus a margin of `eps / 2`,
     * or to the left minus a margin `eps / 2`.
     *
     * @param p Particle index.
     * @param eps Margin.
     * @param direction If `+1`: move right. If `-1` move left.
     */
    void trigger(size_t p, double eps, int direction = 1)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(p < m_N);
        if (direction > 0) {
            m_x(p) = m_chunk->data()(p, m_chunk->chunk_index()(p) + 1) + 0.5 * eps;
        }
        else {
            m_x(p) = m_chunk->data()(p, m_chunk->chunk_index()(p)) - 0.5 * eps;
        }
        this->updated_x();
    }

    /**
     * Change the position of the particles and of the loading frame such that
     * the mean of f_frame() is equal to a target value, and mechanical equilibrium is maintained.
     *
     * @warning Assumes mechanical equilibrium. No assertions are made on this.
     */
    void advanceToFixedForce(double f_frame)
    {
        auto i_n = this->i();
        this->advanceUniformly((f_frame - xt::mean(m_f_frame)()) / m_mu, false);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::all(xt::equal(this->i(), i_n)));
    }

protected:
    /**
     * @copydoc System(double, double, double, double, double, double, Generator*)
     */
    void initSystem(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        Generator* chunk)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(chunk->data().dimension() == 2);

        m_N = chunk->data().shape(0);
        m_m = m;
        m_inv_m = 1.0 / m;
        m_eta = eta;
        m_mu = mu;
        m_k_neighbours = k_neighbours;
        m_k_frame = k_frame;
        m_dt = dt;
        m_f = xt::zeros<double>({m_N});
        m_f_potential = xt::zeros<double>({m_N});
        m_f_neighbours = xt::zeros<double>({m_N});
        m_f_frame = xt::zeros<double>({m_N});
        m_f_damping = xt::zeros<double>({m_N});
        m_x = xt::zeros<double>({m_N});
        m_v = xt::zeros<double>({m_N});
        m_a = xt::zeros<double>({m_N});
        m_v_n = xt::zeros<double>({m_N});
        m_a_n = xt::zeros<double>({m_N});
        m_chunk = chunk;

        this->updated_x();
        this->updated_v();
    }

    /**
     * Compute #f based on the current #x and #v.
     */
    virtual void computeForce()
    {
        xt::noalias(m_f) = m_f_potential + m_f_neighbours + m_f_damping + m_f_frame;
    }

    /**
     * Compute #f_potential based on the current #x.
     */
    virtual void computeForcePotential()
    {
        m_chunk->align(m_x);

        for (size_t p = 0; p < m_N; ++p) {
            auto* l = &m_chunk->data()(p, m_chunk->chunk_index()(p));
            m_f_potential(p) = 0.5 * (*(l) + *(l + 1)) - m_x(p);
        }

        m_f_potential *= m_mu;
    }

    /**
     * Compute #f_neighbours based on the current #x.
     */
    virtual void computeForceNeighbours()
    {
        for (size_t p = 1; p < m_N - 1; ++p) {
            m_f_neighbours(p) = m_x(p - 1) - 2 * m_x(p) + m_x(p + 1);
        }
        m_f_neighbours.front() = m_x.back() - 2 * m_x.front() + m_x(1);
        m_f_neighbours.back() = m_x(m_N - 2) - 2 * m_x.back() + m_x.front();

        m_f_neighbours *= m_k_neighbours;
    }

    /**
     * Compute #f_frame based on the current #x.
     */
    void computeForceFrame()
    {
        xt::noalias(m_f_frame) = m_k_frame * (m_x_frame - m_x);
    }

    /**
     * Compute #f_damping based on the current #v.
     */
    void computeForceDamping()
    {
        xt::noalias(m_f_damping) = -m_eta * m_v;
    }

    /**
     * Update forces that depend on #m_x.
     */
    void updated_x()
    {
        this->computeForcePotential();
        this->computeForceNeighbours();
        this->computeForceFrame();
        this->computeForce();
    }

    /**
     * Update forces that depend on #m_v.
     */
    void updated_v()
    {
        this->computeForceDamping();
        this->computeForce();
    }

    /**
     * Advance the system uniformly: the particles and the frame are moved proportionally,
     * while all particles are moved in the same way.
     * This maintains equilibrium as long as no particle yields.
     * In particular:
     *
     * \f$ \Delta f^\mathrm{potential}_i = \mu \Delta x_i \f$
     *
     * \f$ \Delta f^\mathrm{frame} = k (\Delta x^\mathrm{frame} - \Delta x_i) \f$
     *
     * This gives two cases:
     *
     * -   Given \f$ \Delta x_i \f$ one finds:
     *     \f$ \Delta x^\mathrm{frame} = (k + \mu) / k \Delta x_i \f$.
     *
     * -   Given \f$ \Delta x^\mathrm{frame} \f$ one finds:
     *     \f$ \Delta x_i = k / (k + \mu) \Delta x^\mathrm{frame} \f$.
     *
     * @param dx Displacement.
     *
     * @param input_is_frame
     *     If `true`: `dx` \f$ = \Delta x^\mathrm{frame} \f$,
     *     if `false`: `dx` \f$ = \Delta x_i \f$,
     *
     * @return `dx` for the particles in `input_is_frame == true`, otherwise `dx` of the frame.
     */
    double advanceUniformly(double dx, bool input_is_frame = true)
    {
        double dx_particles;
        double dx_frame;

        if (input_is_frame) {
            dx_frame = dx;
            dx_particles = dx * m_k_frame / (m_k_frame + m_mu);
        }
        else {
            dx_particles = dx;
            dx_frame = dx * (m_k_frame + m_mu) / m_k_frame;
        }

        m_x += dx_particles;
        m_x_frame += dx_frame;
        this->updated_x();

        if (input_is_frame) {
            return dx_particles;
        }

        return dx_frame;
    }

protected:
    array_type::tensor<double, 1> m_f; ///< See #f.
    array_type::tensor<double, 1> m_f_potential; ///< See #f_potential.
    array_type::tensor<double, 1> m_f_neighbours; ///< See #f_neighbours.
    array_type::tensor<double, 1> m_f_frame; ///< See #f_frame.
    array_type::tensor<double, 1> m_f_damping; ///< See #f_damping.
    array_type::tensor<double, 1> m_x; ///< See #x.
    array_type::tensor<double, 1> m_v; ///< See #v.
    array_type::tensor<double, 1> m_a; ///< See #a.
    array_type::tensor<double, 1> m_v_n; ///< #v at last time-step.
    array_type::tensor<double, 1> m_a_n; ///< #a at last time-step.
    Generator* m_chunk; ///< Potential energy landscape.
    size_t m_N; ///< See #N.
    size_t m_inc = 0; ///< Increment number (`time == m_inc * m_dt`).
    size_t m_qs_inc_first = 0; ///< First increment with plastic activity during minimisation.
    size_t m_qs_inc_last = 0; ///< Last increment with plastic activity during minimisation.
    double m_dt; ///< Time step.
    double m_eta; ///< Damping constant (same for all particles).
    double m_m; ///< Mass (same for all particles).
    double m_inv_m; ///< 1 / #m_m
    double m_mu; ///< Stiffness: curvature of the potential energy (same for all particles).
    double m_k_neighbours; ///< Stiffness of interactions (same for all particles).
    double m_k_frame; ///< Stiffness of the load fame (same for all particles).
    double m_x_frame = 0.0; ///< See #set_x_frame.
};

/**
 * ## Introduction
 *
 * System in which the effect of temperature in mimicked by random forcing.
 * The random force drawn from a random distribution and is changed every `n` increments.
 * As such, it is defined by:
 *
 *  -   The mean and standard deviation of the random distribution.
 *  -   The first increment at which the random force is changed.
 *  -   The number of increments between two changes of the random force.
 *
 * ## Physics & API
 *
 * The physics and API are the same as in System(), with the difference that the residual force now
 * reads \f$ f_\text{residual}^{(i)} = f_\text{damping}^{(i)} + f_\text{potential}^{(i)} +
 * f_\text{neighbours}^{(i)} + f_\text{frame}^{(i)} + f_\text{random}^{(i)} \f$.
 *
 * ## Apply fixed force instead of fixed displacement
 *
 * To apply a fixed force (athermal or thermal) use a non-zero mean *and* set `k_frame = 0`.
 *
 * ## Internal strategy
 *
 * To avoid code duplication this class derives from System().
 * To ensure the correct physics computeForce() is overridden to add \f$ f_\text{random}^{(i)} \f$.
 */
class SystemThermalRandomForceNormal : public System {
protected:
    array_type::tensor<double, 1> m_f_thermal; ///< Current applied 'random' forces.
    double m_mean; ///< Mean of the random distribution.
    double m_stddev; ///< Standard deviation of the random distribution.
    array_type::tensor<size_t, 1> m_next; ///< Next increment at which the random force is changed.
    array_type::tensor<size_t, 1> m_dinc; ///< \#increment between two changes of the random force.
    prrng::pcg32 m_rng; ///< Random number generator.

public:
    SystemThermalRandomForceNormal() = default;

    /**
     * @copydoc System(double, double, double, double, double, double, Generator*)
     * @param mean Mean of the random distribution from which the random force is drawn.
     * @param stddev Standard deviation of that distribution.
     * @param seed Seed for the random number generator.
     * @param dinc_init Number of increments to wait to draw the first random force.
     * @param dinc Number of increments between two changes of the random force.
     */
    SystemThermalRandomForceNormal(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        Generator* chunk,
        double mean,
        double stddev,
        uint64_t seed,
        const array_type::tensor<size_t, 1>& dinc_init,
        const array_type::tensor<size_t, 1>& dinc)
    {
        this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, chunk);
        m_rng.seed(seed);
        m_f_thermal = m_rng.normal<decltype(m_f_thermal)>({m_N}, mean, stddev);
        m_mean = mean;
        m_stddev = stddev;
        m_next = dinc_init;
        m_dinc = dinc;
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
    const array_type::tensor<double, 1>& f_thermal() const
    {
        return m_f_thermal;
    }

    /**
     * @brief Change the random force.
     * @param f_thermal New random force.
     */
    void set_f_thermal(const array_type::tensor<double, 1>& f_thermal)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(f_thermal, m_f_thermal.shape()));
        m_f_thermal = f_thermal;
    }

    /**
     * @brief Aext increment at which the random force is changed.
     * @return Array
     */
    const array_type::tensor<size_t, 1>& next()
    {
        return m_next;
    }

    /**
     * @brief Set the next increment at which the random force is changed.
     * @param next Next increment.
     */
    void set_next(const array_type::tensor<size_t, 1>& next)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(next, m_next.shape()));
        m_next = next;
    }

protected:
    /**
     * Update 'thermal' force
     */
    void updateThermalForce()
    {
        for (size_t p = 0; p < m_N; ++p) {
            if (m_inc >= m_next(p)) {
                m_f_thermal(p) = m_rng.normal(m_mean, m_stddev);
                m_next(p) += m_dinc(p);
            }
        }
    }

    void computeForce() override
    {
        this->updateThermalForce();
        xt::noalias(m_f) = m_f_potential + m_f_neighbours + m_f_damping + m_f_frame + m_f_thermal;
    }
};

/**
 * ## Introduction
 *
 * System in which the effect of temperature in mimicked by random forcing.
 * The random forces can be set:
 * -   Instantaneously, using setRandomForce().
 * -   As a sequence depending on the current increment, using setRandomForceSequence().
 *
 * ## Physics & API
 *
 * The physics and API are the same as in System(), with the difference that the residual force now
 * reads \f$ f_\text{residual}^{(i)} = f_\text{damping}^{(i)} + f_\text{potential}^{(i)} +
 * f_\text{neighbours}^{(i)} + f_\text{frame}^{(i)} + f_\text{random}^{(i)} \f$.
 * Thereby \f$ f_\text{random}^{(i)} \f$ is specified instantaneously setRandomForce() or for a
 * time-history in setRandomForceSequence().
 *
 * ## Apply fixed force instead of fixed displacement
 *
 * To apply a fixed force (athermal or thermal) use setRandomForce() or setRandomForceSequence()
 * with a non-zero mean *and* set `k_frame = 0`.
 *
 * ## Internal strategy
 *
 * To avoid code duplication this class derives from System().
 * To ensure the correct physics computeForce() is overridden to add \f$ f_\text{random}^{(i)} \f$.
 */
class SystemThermalRandomForcing : public System {
public:
    SystemThermalRandomForcing() = default;

    /**
     * @copydoc System(double, double, double, double, double, double, Generator*)
     */
    SystemThermalRandomForcing(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        Generator* chunk)
    {
        this->initSystemThermalRandomForcing(m, eta, mu, k_neighbours, k_frame, dt, chunk);
    }

    /**
     * Set random force.
     * This force will be applied until it is overwritten, or a random force sequence is set in
     * setRandomForceSequence().
     * If a sequence was set before, it will be discarded and replaced by this force.
     *
     * @param f Force per particle.
     */
    template <class T>
    void setRandomForce(const T& f)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(f.dimension() == 1);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(f.size() == m_N);

        m_seq = false;
        m_f_thermal = f;
    }

    /**
     * Set sequence of random forces.
     * This sequence specifies for each particle which force should be applied at which increment.
     *
     * @param f Sequence of forces per particle [#N, `n`].
     *
     * @param start_inc
     *     Start and end increment of each item [#N, `n + 1`].
     *     This implies that on a particle `p`, the force `f(p, i)` will be applied starting from
     *     increment `start_inc(p, i)` until (but not including) `start_inc(p, i + 1)`.
     *     The sequence is thus bounded and should be updated in time.
     */
    template <class T, class U>
    void setRandomForceSequence(const T& f, const U& start_inc)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(f, {m_N, start_inc.shape(1) - 1}));
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(start_inc, {m_N, f.shape(1) + 1}));
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::all(xt::view(start_inc, xt::all(), 0) <= m_inc));
        FRICTIONQPOTSPRINGBLOCK_ASSERT(
            xt::all(xt::view(start_inc, xt::all(), start_inc.shape(1) - 1) > m_inc));
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::all(xt::equal(start_inc, xt::sort(start_inc, 1))));

        m_seq = true;
        m_seq_f = f;
        m_seq_s = start_inc;
        m_seq_i = xt::argmax(m_seq_s >= m_inc, 1);

        for (size_t p = 0; p < m_N; ++p) {
            if (m_inc >= m_seq_s(p, m_seq_i(p))) {
                m_f_thermal(p) = m_seq_f(p, m_seq_i(p));
            }
        }
    }

protected:
    /**
     * @copydoc SystemThermalRandomForcing(
     *      double, double, double, double, double, double, Generator*)
     */
    void initSystemThermalRandomForcing(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        Generator* chunk)
    {
        m_seq = false;
        this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, chunk);
        m_f_thermal = xt::zeros<double>({m_N});
    }

    /**
     * Update 'thermal' force for sequence (if needed).
     */
    void updateThermalForce()
    {
        if (m_seq) {
            for (size_t p = 0; p < m_N; ++p) {
                if (m_inc >= m_seq_s(p, m_seq_i(p) + 1)) {
                    m_seq_i(p)++;
                    FRICTIONQPOTSPRINGBLOCK_ASSERT(m_seq_i(p) < m_seq_f.shape(1));
                    FRICTIONQPOTSPRINGBLOCK_ASSERT(m_inc < m_seq_s(p, m_seq_i(p) + 1));
                }
                if (m_inc >= m_seq_s(p, m_seq_i(p))) {
                    m_f_thermal(p) = m_seq_f(p, m_seq_i(p));
                }
            }
        }
    }

    void computeForce() override
    {
        this->updateThermalForce();
        xt::noalias(m_f) = m_f_potential + m_f_neighbours + m_f_damping + m_f_frame + m_f_thermal;
    }

    bool m_seq = false; ///< Use sequence to set random forces, set in setRandomForceSequence().
    array_type::tensor<double, 2> m_seq_f; ///< Sequence of random forces.
    array_type::tensor<size_t, 2> m_seq_s; ///< Start/end increment of each item in the sequence.
    array_type::tensor<size_t, 1> m_seq_i; ///< Current column in #m_seq_f for each particle.
    array_type::tensor<double, 1> m_f_thermal; ///< Current applied 'random' forces.
};

/**
 * ## Introduction
 *
 * Identical to System() but with piece-wise continuous forces.
 *
 * ## Internal strategy
 *
 * To avoid code duplication this class derives from System().
 * To ensure the correct physics computeForcePotential() is overridden.
 */
class SystemSemiSmooth : public System {
public:
    SystemSemiSmooth() = default;

    /**
     * @copydoc System(double, double, double, double, double, double, Generator*)
     * @param kappa Softening stiffness.
     */
    SystemSemiSmooth(
        double m,
        double eta,
        double mu,
        double kappa,
        double k_neighbours,
        double k_frame,
        double dt,
        Generator* chunk)
    {
        m_kappa = kappa;
        this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, chunk);
    }

    double maxUniformDisplacement(int direction = 1) override
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(direction == 1 || direction == -1);

        bool positive = direction > 0;
        std::vector<double> dx;
        dx.reserve(m_N);

        for (size_t p = 0; p < m_N; ++p) {

            auto* y = &m_chunk->data()(p, m_chunk->chunk_index()(p));
            double xi = 0.5 * (*(y) + *(y + 1));
            double u = (m_mu * xi + m_kappa * *(y + 1)) / (m_mu + m_kappa);
            double l = (m_mu * xi + m_kappa * *(y)) / (m_mu + m_kappa);
            double x = m_x(p);

            if (x < l) {
                return 0.0;
            }
            else if (x <= u) {
                if (positive) {
                    dx.push_back(u - x);
                }
                else {
                    dx.push_back(x - l);
                }
            }
            else {
                return 0.0;
            }
        }

        return *std::min_element(dx.begin(), dx.end());
    }

protected:
    void computeForcePotential() override
    {
        m_chunk->align(m_x);

        for (size_t p = 0; p < m_N; ++p) {

            auto* y = &m_chunk->data()(p, m_chunk->chunk_index()(p));
            double xi = 0.5 * (*(y) + *(y + 1));
            double u = (m_mu * xi + m_kappa * *(y + 1)) / (m_mu + m_kappa);
            double l = (m_mu * xi + m_kappa * *(y)) / (m_mu + m_kappa);
            double x = m_x(p);
            if (x < l) {
                m_f_potential(p) = m_kappa * (x - *(y));
            }
            else if (x <= u) {
                m_f_potential(p) = m_mu * (0.5 * (*(y) + *(y + 1)) - x);
            }
            else {
                m_f_potential(p) = m_kappa * (x - *(y + 1));
            }
        }
    }

protected:
    double m_kappa; ///< Softening stiffness.
};

/**
 * ## Introduction
 *
 * Identical to System() but with continuous and smooth forces.
 *
 * ## Internal strategy
 *
 * To avoid code duplication this class derives from System().
 * To ensure the correct physics computeForcePotential() is overridden.
 */
class SystemSmooth : public System {
public:
    SystemSmooth() = default;

    /**
     * @copydoc System(double, double, double, double, double, double, Generator*)
     */
    SystemSmooth(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        Generator* chunk)
    {
        this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, chunk);
    }

    double maxUniformDisplacement(int direction = 1) override
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(direction == 1 || direction == -1);
        (void)(direction);
        return 0.0;
    }

protected:
    void computeForcePotential() override
    {
        m_chunk->align(m_x);

        for (size_t p = 0; p < m_N; ++p) {
            auto* y = &m_chunk->data()(p, m_chunk->chunk_index()(p));
            double x = m_x(p);
            double xmin = 0.5 * (*(y) + *(y + 1));
            double dy = 0.5 * (*(y + 1) - *(y));
            m_f_potential(p) = -m_mu * dy / M_PI * std::sin(M_PI * (x - xmin) / dy);
        }
    }
};

/**
 * ## Introduction
 *
 * Identical to System() but with interactions with a quartic term.
 * See e.g. https://doi.org/10.1103/PhysRevLett.87.187002
 *
 * The idea is to have a potential
 *
 * \f$ E(r) = \frac{1}{2} k_2 (\varepsilon(r))^2 + \frac{1}{12} k_4 (\varepsilon(r))^4 \f$
 *
 * with
 *
 * \f$ \varepsilon(r) = \frac{\partial x}{\partial r} \f$
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
 * \f$ f_i = \left( x_{i + 1} + x_{i - 1} - 2 x_i \right)
 * \left( k_2 + k_4 / 4 (x_{i + 1} - x_{i - 1})^2 \right)\f$
 *
 * where the gradient term is approximated by central difference
 * (which leads to \f$ 1/4 \f$ factor).
 *
 * Please note that \f$ k_4 \f$ does not have the same units as \f$ k_2 \f$.
 *
 * ## Internal strategy
 *
 * To avoid code duplication this class derives from System().
 * To ensure the correct physics computeForceNeighbours() is overridden.
 * All methods and functionality of System() are available.
 */
class SystemQuartic : public System {
protected:
    double m_k2; ///< Interaction: proportional to the Laplacian.
    double m_k4; ///< Interaction: proportional to the Laplacian times the square of the gradient.

public:
    SystemQuartic() = default;

    /**
     * @param m Particle mass (same for all particles).
     * @param eta Damping coefficient (same for all particles).
     * @param mu Elastic stiffness, i.e. the curvature of the potential (same for all particles).
     * @param k_frame Stiffness of springs between particles and load frame (same for all).
     * @param dt Time step.
     * @param chunk Class in which chunks of the yield positions are stored (copy).
     * @param k2 Interaction stiffness (same for all), see class description.
     * @param k4 Interaction strength quartic term s (same for all), see class description.
     */
    SystemQuartic(
        double m,
        double eta,
        double mu,
        double k2,
        double k4,
        double k_frame,
        double dt,
        Generator* chunk)
    {
        this->initSystem(m, eta, mu, 0.0, k_frame, dt, chunk);
        m_k2 = k2;
        m_k4 = k4;
    }

protected:
    void computeForceNeighbours() override
    {
        for (size_t p = 1; p < m_N - 1; ++p) {
            double dx = 0.5 * (m_x(p + 1) - m_x(p - 1));
            m_f_neighbours(p) = (m_x(p - 1) - 2 * m_x(p) + m_x(p + 1)) * (m_k2 + m_k4 * (dx * dx));
        }

        double dxf = 0.5 * (m_x(1) - m_x.back());
        m_f_neighbours.front() =
            (m_x.back() - 2 * m_x.front() + m_x(1)) * (m_k2 + m_k4 * (dxf * dxf));

        double dxb = 0.5 * (m_x.front() - m_x(m_N - 2));
        m_f_neighbours.back() =
            (m_x(m_N - 2) - 2 * m_x.back() + m_x.front()) * (m_k2 + m_k4 * (dxb * dxb));
    }
};

/**
 * ## Introduction
 *
 * Identical to System() but with interactions with a quartic term.
 * See e.g. https://doi.org/10.1103/PhysRevLett.87.187002
 *
 */
class SystemQuarticFD : public System {
protected:
    double m_a1; ///< Interaction: proportional to the Laplacian.
    double m_a2; ///< Interaction: proportional to the fourth order term

public:
    SystemQuarticFD() = default;

    /**
     * @param m Particle mass (same for all particles).
     * @param eta Damping coefficient (same for all particles).
     * @param mu Elastic stiffness, i.e. the curvature of the potential (same for all particles).
     * @param k_frame Stiffness of springs between particles and load frame (same for all).
     * @param dt Time step.
     * @param chunk Class in which chunks of the yield positions are stored (copy).
     * @param a1 Interaction stiffness (same for all), see class description.
     * @param a2 Interaction strength quartic term s (same for all), see class description.
     */
    SystemQuarticFD(
        double m,
        double eta,
        double mu,
        double a1,
        double a2,
        double k_frame,
        double dt,
        Generator* chunk)
    {
        this->initSystem(m, eta, mu, 0.0, k_frame, dt, chunk);
        m_a1 = a1;
        m_a2 = a2;
    }

protected:
    void computeForceNeighbours() override
    {
        for (size_t p = 1; p < m_N - 1; ++p) {
            double dxp = std::pow(m_x(p + 1) - m_x(p), 3.0);
            double dxn = std::pow(m_x(p - 1) - m_x(p), 3.0);
            m_f_neighbours(p) = m_a1 * (m_x(p - 1) - 2 * m_x(p) + m_x(p + 1)) + m_a2 * (dxp + dxn);
        }

        {
            double dxp = std::pow(m_x(1) - m_x.front(), 3.0);
            double dxn = std::pow(m_x.back() - m_x.front(), 3.0);
            m_f_neighbours.front() =
                m_a1 * (m_x.back() - 2 * m_x.front() + m_x(1)) + m_a2 * (dxp + dxn);
        }

        {
            double dxp = std::pow(m_x.front() - m_x.back(), 3.0);
            double dxn = std::pow(m_x(m_N - 2) - m_x.back(), 3.0);
            m_f_neighbours.back() =
                m_a1 * (m_x(m_N - 2) - 2 * m_x.back() + m_x.front()) + m_a2 * (dxp + dxn);
        }
    }
};

/**
 * ## Introduction
 *
 * Identical to System() but with long-range interactions.
 * Here, the interactions decay as \f$ 1 / r^{d + \alpha} \f$.
 * See e.g. https://doi.org/10.1103/PhysRevLett.126.025702
 *
 * ## Internal strategy
 *
 * To avoid code duplication this class derives from System().
 * To ensure the correct physics computeForceNeighbours() is overridden.
 */
class SystemLongRange : public System {
protected:
    double m_alpha; ///< Range of interactions.
    array_type::tensor<double, 1> m_prefactor; ///< Prefactor for long-range interactions.
    ptrdiff_t m_n; ///< Alias of m_N.
    ptrdiff_t m_m; ///< Midpoint.

public:
    SystemLongRange() = default;

    /**
     * @copydoc System(double, double, double, double, double, double, Generator*)
     * @param alpha Range of interactions.
     */
    SystemLongRange(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double alpha,
        double k_frame,
        double dt,
        Generator* chunk)
    {
        m_n = static_cast<ptrdiff_t>(chunk->data().shape(0));
        m_m = (m_n - m_n % 2) / 2;
        this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, chunk);
        m_alpha = alpha;

        m_prefactor = xt::empty<double>({m_N});

        for (ptrdiff_t d = 0; d < m_n; ++d) {
            if (d == 0) {
                m_prefactor(0) = 0.0;
            }
            else {
                m_prefactor(d) = m_k_neighbours / std::pow(d, m_alpha + 1.0);
            }
        }
    }

    /**
     * @brief Distance from a particle `p` to all the other particles.
     *
     * @param p Particle index.
     * @return Distance to `p` for each particle.
     */
    array_type::tensor<ptrdiff_t, 1> distance(ptrdiff_t p) const
    {
        array_type::tensor<ptrdiff_t, 1> ret = xt::empty<ptrdiff_t>({m_N});

        for (ptrdiff_t i = 0; i < m_n; ++i) {
            ptrdiff_t d = std::abs(i - p);
            if (d > m_m) {
                d = m_n - d;
            }
            ret(i) = d;
        }

        return ret;
    }

protected:
    void computeForceNeighbours() override
    {
        for (ptrdiff_t p = 0; p < m_n; ++p) {
            double f = 0.0;
            double x = m_x(p);
            for (ptrdiff_t i = 0; i < m_n; ++i) {
                if (i == p) {
                    continue;
                }
                ptrdiff_t d = std::abs(i - p);
                if (d > m_m) {
                    d = m_n - d;
                }
                f += (m_x(i) - x) * m_prefactor(d);
            }
            m_f_neighbours(p) = f;
        }
    }
};

/**
 * ## Introduction
 *
 * Identical to System() but with '2d' interactions.
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
 * ## Internal strategy
 *
 * To avoid code duplication this class derives from System().
 * To ensure the correct physics computeForceNeighbours() is overridden.
 */
class System2d : public System {
protected:
    size_t m_m; ///< Number of 'rows'
    size_t m_n; ///< Number of 'columns'

public:
    System2d() = default;

    /**
     * @copydoc System(double, double, double, double, double, double, Generator*)
     * @param width Number of 'columns'
     */
    System2d(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        Generator* chunk,
        size_t width)
    {
        size_t N = chunk->data().shape(0);
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(N % width == 0);

        m_n = width;
        m_m = N / m_n;

        FRICTIONQPOTSPRINGBLOCK_REQUIRE(m_n >= 2);
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(m_m >= 2);

        this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, chunk);
    }

    array_type::array<size_t> organisation() const override
    {
        array_type::array<size_t> ret = xt::empty<size_t>({m_m, m_n});

        for (size_t i = 0; i < m_m; ++i) {
            for (size_t j = 0; j < m_n; ++j) {
                ret(i, j) = i * m_m + j;
            }
        }

        return ret;
    }

    /**
     * @brief Get the 'upper' neighbour of each particle.
     * For each particle `(i, j)` this function returns `(i - 1, j)`.
     *
     * @return 1d array.
     */
    array_type::tensor<size_t, 1> up()
    {
        array_type::tensor<size_t, 1> ret = xt::empty<size_t>({m_N});

        for (size_t i = 1; i < m_m - 1; ++i) {
            for (size_t j = 1; j < m_n - 1; ++j) {
                ret(i * m_m + j) = (i - 1) * m_m + j;
            }
        }

        for (size_t j = 1; j < m_n - 1; ++j) {
            ret(j) = (m_m - 1) * m_m + j;
            ret((m_m - 1) * m_m + j) = (m_m - 2) * m_m + j;
        }

        for (size_t i = 1; i < m_m - 1; ++i) {
            ret(i * m_m) = (i - 1) * m_m;
            ret(i * m_m + m_n - 1) = (i - 1) * m_m + m_n - 1;
        }

        ret(0) = (m_m - 1) * m_m;
        ret(m_n - 1) = (m_m - 1) * m_m + m_n - 1;
        ret((m_m - 1) * m_m) = (m_m - 2) * m_m;
        ret((m_m - 1) * m_m + m_n - 1) = (m_m - 2) * m_m + m_n - 1;

        return ret;
    }

    /**
     * @brief Get the 'bottom' neighbour of each particle.
     * For each particle `(i, j)` this function returns `(i + 1, j)` (accounting for periodicity).
     *
     * @return 1d array.
     */
    array_type::tensor<size_t, 1> down()
    {
        array_type::tensor<size_t, 1> ret = xt::empty<size_t>({m_N});

        for (size_t i = 1; i < m_m - 1; ++i) {
            for (size_t j = 1; j < m_n - 1; ++j) {
                ret(i * m_m + j) = (i + 1) * m_m + j;
            }
        }

        for (size_t j = 1; j < m_n - 1; ++j) {
            ret(j) = m_m + j;
            ret((m_m - 1) * m_m + j) = j;
        }

        for (size_t i = 1; i < m_m - 1; ++i) {
            ret(i * m_m) = (i + 1) * m_m;
            ret(i * m_m + m_n - 1) = (i + 1) * m_m + m_n - 1;
        }

        ret(0) = m_m;
        ret(m_n - 1) = m_m + m_n - 1;
        ret((m_m - 1) * m_m) = 0;
        ret((m_m - 1) * m_m + m_n - 1) = m_n - 1;

        return ret;
    }

    /**
     * @brief Get the 'left' neighbour of each particle.
     * For each particle `(i, j)` this function returns `(i, j - 1)` (accounting for periodicity).
     *
     * @return 1d array.
     */
    array_type::tensor<size_t, 1> left()
    {
        array_type::tensor<size_t, 1> ret = xt::empty<size_t>({m_N});

        for (size_t i = 1; i < m_m - 1; ++i) {
            for (size_t j = 1; j < m_n - 1; ++j) {
                ret(i * m_m + j) = i * m_m + j - 1;
            }
        }

        for (size_t j = 1; j < m_n - 1; ++j) {
            ret(j) = j - 1;
            ret((m_m - 1) * m_m + j) = (m_m - 1) * m_m + j - 1;
        }

        for (size_t i = 1; i < m_m - 1; ++i) {
            ret(i * m_m) = i * m_m + m_n - 1;
            ret(i * m_m + m_n - 1) = i * m_m + m_n - 2;
        }

        ret(0) = m_n - 1;
        ret(m_n - 1) = m_n - 2;
        ret((m_m - 1) * m_m) = (m_m - 1) * m_m + m_n - 1;
        ret((m_m - 1) * m_m + m_n - 1) = (m_m - 1) * m_m + m_n - 2;

        return ret;
    }

    /**
     * @brief Get the 'right' neighbour of each particle.
     * For each particle `(i, j)` this function returns `(i, j + 1)` (accounting for periodicity).
     *
     * @return 1d array.
     */
    array_type::tensor<size_t, 1> right()
    {
        array_type::tensor<size_t, 1> ret = xt::empty<size_t>({m_N});

        for (size_t i = 1; i < m_m - 1; ++i) {
            for (size_t j = 1; j < m_n - 1; ++j) {
                ret(i * m_m + j) = i * m_m + j + 1;
            }
        }

        for (size_t j = 1; j < m_n - 1; ++j) {
            ret(j) = j + 1;
            ret((m_m - 1) * m_m + j) = (m_m - 1) * m_m + j + 1;
        }

        for (size_t i = 1; i < m_m - 1; ++i) {
            ret(i * m_m) = i * m_m + 1;
            ret(i * m_m + m_n - 1) = i * m_m;
        }

        ret(0) = 1;
        ret(m_n - 1) = 0;
        ret((m_m - 1) * m_m) = (m_m - 1) * m_m + 1;
        ret((m_m - 1) * m_m + m_n - 1) = (m_m - 1) * m_m;

        return ret;
    }

protected:
    void computeForceNeighbours() override
    {
        // interior
        for (size_t i = 1; i < m_m - 1; ++i) {
            for (size_t j = 1; j < m_n - 1; ++j) {
                m_f_neighbours(i * m_m + j) = // (i, j)
                    m_x((i - 1) * m_m + j) + // (i - 1, j)
                    m_x((i + 1) * m_m + j) + // (i + 1, j)
                    m_x(i * m_m + j - 1) + // (i, j - 1)
                    m_x(i * m_m + j + 1) - // (i, j + 1)
                    4 * m_x(i * m_m + j); // (i, j)
            }
        }

        for (size_t j = 1; j < m_n - 1; ++j) {
            // top
            m_f_neighbours(j) = // (i, j)
                m_x((m_m - 1) * m_m + j) + // (i - 1, j)  ->  (m - 1, j)
                m_x(m_m + j) + // (i + 1, j)
                m_x(j - 1) + // (i, j - 1)
                m_x(j + 1) - // (i, j + 1)
                4 * m_x(j); // (i, j)

            // bottom
            m_f_neighbours((m_m - 1) * m_m + j) = // (i, j)
                m_x((m_m - 2) * m_m + j) + // (i - 1, j)
                m_x(j) + // (i + 1, j)  ->  (0, j)
                m_x((m_m - 1) * m_m + j - 1) + // (i, j - 1)
                m_x((m_m - 1) * m_m + j + 1) - // (i, j + 1)
                4 * m_x((m_m - 1) * m_m + j); // (i, j)
        }

        for (size_t i = 1; i < m_m - 1; ++i) {
            // (i, j - 1)-edge
            m_f_neighbours(i * m_m) = // (i, j)
                m_x((i - 1) * m_m) + // (i - 1, j)
                m_x((i + 1) * m_m) + // (i + 1, j)
                m_x(i * m_m + m_n - 1) + // (i, j - 1)  ->  (i, n - 1)
                m_x(i * m_m + 1) - // (i, j + 1)
                4 * m_x(i * m_m); // (i, j)

            // (i, j + 1)-edge
            m_f_neighbours(i * m_m + m_n - 1) = // (i, j)
                m_x((i - 1) * m_m + m_n - 1) + // (i - 1, j)
                m_x((i + 1) * m_m + m_n - 1) + // (i + 1, j)
                m_x(i * m_m + m_n - 2) + // (i, j - 1)
                m_x(i * m_m) - // (i, j + 1)  ->  (i, 0)
                4 * m_x(i * m_m + m_n - 1); // (i, j)
        }

        // corner: top-left
        m_f_neighbours(0) = // (i, j)  ->  (0, 0)
            m_x((m_m - 1) * m_m) + // (i - 1, j)  ->  (m - 1, 0)
            m_x(m_m) + // (i + 1, j)  ->  (1, 0)
            m_x(m_n - 1) + // (i, j - 1)  ->  (0, n - 1)
            m_x(1) - // (i, j + 1)  ->  (0, 1)
            4 * m_x(0); // (i, j)  ->  (0, 0)

        // corner: top-right
        m_f_neighbours(m_n - 1) = // (i, j)  ->  (0, n - 1)
            m_x((m_m - 1) * m_m + m_n - 1) + // (i - 1, j)  ->  (m - 1, n - 1)
            m_x(m_m + m_n - 1) + // (i + 1, j)  ->  (1, n - 1)
            m_x(m_n - 2) + // (i, j - 1)  ->  (0, n - 2)
            m_x(0) - // (i, j + 1)  ->  (0, 0)
            4 * m_x(m_n - 1); // (i, j)  ->  (0, n - 1)

        // corner: bottom-left
        m_f_neighbours((m_m - 1) * m_m) = // (i, j)  ->  (m - 1, 0)
            m_x((m_m - 2) * m_m) + // (i - 1, j)  ->  (m - 2, 0)
            m_x(0) + // (i + 1, j)  ->  (0, 0)
            m_x((m_m - 1) * m_m + m_n - 1) + // (i, j - 1)  ->  (m - 1, n - 1)
            m_x((m_m - 1) * m_m + 1) - // (i, j + 1)  ->  (m - 1, 1)
            4 * m_x((m_m - 1) * m_m); // (i, j)  ->  (m - 1, 0)

        // corner: bottom-right
        m_f_neighbours((m_m - 1) * m_m + m_n - 1) = // (i, j)  ->  (m - 1, n - 1)
            m_x((m_m - 2) * m_m + m_n - 1) + // (i - 1, j)  ->  (m - 2, n - 1)
            m_x(m_n - 1) + // (i + 1, j)  ->  (0, n - 1)
            m_x((m_m - 1) * m_m + m_n - 2) + // (i, j - 1)  ->  (m - 1, n - 2)
            m_x((m_m - 1) * m_m) - // (i, j + 1)  ->  (m - 1, 0)
            4 * m_x((m_m - 1) * m_m + m_n - 1); // (i, j)  ->  (m - 1, n - 1)

        m_f_neighbours *= m_k_neighbours;
    }
};

/**
 * ## Introduction
 *
 * A 2d system with a quartic interaction.
 */
class System2dQuartic : public System2d {
protected:
    double m_k2; ///< Interaction: proportional to the Laplacian.
    double m_k4; ///< Interaction: proportional to the Laplacian times the square of the gradient.
    array_type::tensor<size_t, 2> m_indices; ///< Indices.

public:
    System2dQuartic() = default;

    /**
     * @param m Particle mass (same for all particles).
     * @param eta Damping coefficient (same for all particles).
     * @param mu Elastic stiffness, i.e. the curvature of the potential (same for all particles).
     * @param k_frame Stiffness of springs between particles and load frame (same for all).
     * @param dt Time step.
     * @param chunk Class in which chunks of the yield positions are stored (copy).
     * @param k2 Interaction stiffness (same for all), see class description.
     * @param k4 Interaction strength quartic term s (same for all), see class description.
     * @param width Number of 'columns'
     */
    System2dQuartic(
        double m,
        double eta,
        double mu,
        double k2,
        double k4,
        double k_frame,
        double dt,
        Generator* chunk,
        size_t width)
    {
        size_t N = chunk->data().shape(0);
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(N % width == 0);

        m_n = width;
        m_m = N / m_n;

        FRICTIONQPOTSPRINGBLOCK_REQUIRE(m_n >= 2);
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(m_m >= 2);

        this->initSystem(m, eta, mu, 0.0, k_frame, dt, chunk);
        m_k2 = k2;
        m_k4 = k4;

        xt::xtensor<size_t, 2> indices = this->organisation();
        m_indices = xt::pad(indices, 1, xt::pad_mode::periodic);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(m_indices.shape(0) == m_m + 2);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(m_indices.shape(1) == m_n + 2);
    }

protected:
    void computeForceNeighbours() override
    {
        double mk4_3 = m_k4 / 3.0;
        double mk4_23 = 2.0 * mk4_3;

        for (size_t m = 0; m < m_m; ++m) {
            for (size_t n = 0; n < m_n; ++n) {

                size_t i_j = m_indices(m + 1, n + 1);
                size_t im_j = m_indices(m, n + 1);
                size_t ip_j = m_indices(m + 2, n + 1);
                size_t i_jm = m_indices(m + 1, n);
                size_t i_jp = m_indices(m + 1, n + 2);
                size_t im_jm = m_indices(m, n);
                size_t ip_jm = m_indices(m + 2, n);
                size_t im_jp = m_indices(m, n + 2);
                size_t ip_jp = m_indices(m + 2, n + 2);

                double l = m_x(ip_j) + m_x(im_j) + m_x(i_jp) + m_x(i_jm) - 4 * m_x(i_j);
                double dudx = 0.5 * (m_x(ip_j) - m_x(im_j));
                double dudy = 0.5 * (m_x(i_jp) - m_x(i_jm));
                double d2udxdy = 0.25 * (m_x(ip_jp) - m_x(ip_jm) - m_x(im_jp) + m_x(im_jm));
                double d2udx2 = m_x(ip_j) - 2 * m_x(i_j) + m_x(im_j);
                double d2udy2 = m_x(i_jp) - 2 * m_x(i_j) + m_x(i_jm);

                m_f_neighbours(i_j) =
                    l * (m_k2 + mk4_3) + mk4_23 * (dudx * dudx * d2udx2 + dudy * dudy * d2udy2 +
                                                   2.0 * dudx * dudy * d2udxdy);
            }
        }
    }
};

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#endif
