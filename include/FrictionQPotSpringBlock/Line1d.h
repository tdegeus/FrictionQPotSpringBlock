/**
Line in 1d.

\file Line1d.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_H
#define FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_H

#include <string>

#include <xtensor/xnorm.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xtensor.hpp>

#include "config.h"

#include <GooseFEM/Iterate.h>
#include <GooseFEM/version.h>
#include <QPot.h>

#include <GMatTensor/version.h>

namespace FrictionQPotSpringBlock {

/**
Line in 1d.
*/
namespace Line1d {

/**
Return versions of this library and of all of its dependencies.
The output is a list of strings, e.g.::

    "frictionqpotspringblock=0.1.0",
    "xtensor=0.20.1"
    ...

\return List of strings.
*/
inline std::vector<std::string> version_dependencies()
{
    return GMatTensor::version_dependencies();
}

/**
Return information on the compiler, the platform, the C++ standard, and the compilation data.
\return List of strings.
*/
inline std::vector<std::string> version_compiler()
{
    return GMatTensor::version_compiler();
}

/**
## Introduction

The system in which particles experience a piece-wise quadratic local potential energy,
elastic interactions with neighbours, and a drive though a spring attached to a load frame.
The local landscape is characterised by yield positions.

## Physics & API

The physics are as follows:
-   A particle has a mass \f$ m \f$.
-   Each particle \f$ i \f$ experiences damping equal to
    \f$ f_\text{damping}^{(i)} = - \eta v_i \f$, with the particle's velocity
    \f$ v_i \equiv \dot{x}_i \equiv \partial_t x_i \f$.
-   Each particle \f$ i \f$ has a potential energy such that
    \f$ f_\text{potential}^{(i)} =  \mu (x_{\min}^{(i)} - x_i) \f$.
    where \f$ \mu \f$ is the radius of curvature of the quadratic potentials, and
    \f$ x_{\min}^{(i)} \f$ the positions of the current local minimum.
-   Each particle \f$ i \f$ has interactions with its neighbours equal to
    \f$ f_\text{neighbours}^{(i)} =  k_\text{neighbours} (x_{i - 1} - 2 x_i + x_{i + 1}) \f$.
-   Each particle \f$ i \f$ is connected to the load frame given a force equal to
    \f$ f_\text{frame}^{(i)} =  k_\text{frame} (x_\text{frame} - x_i) \f$.
    Typically \f$ k_\text{frame} = \mathcal{O}(1 / N) \f$ with \f$ N \f$ the number of particles.

The dynamics follow
\f$ m a_i = f_\text{residual}^{(i)} \f$, with the particle's acceleration
\f$ a_i \equiv \ddot{x}_i \equiv \partial_t^2 x_i \f$, and the residual force
\f$ f_\text{residual}^{(i)} = f_\text{damping}^{(i)} + f_\text{potential}^{(i)} +
f_\text{neighbours}^{(i)} + f_\text{frame}^{(i)} \f$.
This differential equation is integrated using Velocity-Verlet in timeStep()
Note that this function updates the time #t (or increment number #inc),
and that this is the only function that does so.

The implementation is is such that all forces are updated (if needed) every time the positions
and velocities are updated, either internally in timeStep(), or externally when calling
set_x() and set_v().
In absolute need refresh() can be called to force re-computation of forces,
but this should normally not be needed.

## Yield positions

For computational efficiency one can choose to store only a chunk of the sequence of yield positions
around the current position, which then has to updated when the particle is close to running
out-of-bounds, using set_y().

## Internal strategy

Internally, updated_x() and updated_v() are called every time that the positions #m_x and the
velocities #m_v are updated.
Thereby, updated_x() updates position dependent forces,
and updated_v() updates velocity dependent forces.
Both finish by updating the residual force in computeForce().
*/
class System {

public:
    System() = default;

    /**
    Constructor.

    \param m Particle mass (same for all particles).
    \param eta Damping coefficient (same for all particles).
    \param mu Elastic stiffness, i.e. the curvature of the potential (same for all particles).
    \param k_neighbours Stiffness of the 'springs' connecting neighbours (same for all particles).
    \param k_frame Stiffness of springs between particles and load frame (same for all particles).
    \param dt Time step.
    \param x_yield Yield positions [#N, n_yield].
    */
    template <class T>
    System(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        const T& x_yield)
    {
        this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, x_yield);
    }

    /**
    Number of particles.
    \return unsigned int
    */
    size_t N() const
    {
        return m_N;
    }

    /**
    Return yield positions.
    \return Array of shape [#N, n_yield].
    */
    const array_type::tensor<double, 2>& y()
    {
        return m_y;
    }

    /**
    Current index (in the current chunk) of the potential energy landscape of each particle.
    \return Array of shape [#N].
    */
    const array_type::tensor<long, 1>& i() const
    {
        return m_i;
    }

    /**
    Closest yield position right of the particle.
    Convenience function: same as `system.y[np.arange(system.N), system.i + 1]`.
    \return [#N].
    */
    array_type::tensor<double, 1> y_right() const
    {
        array_type::tensor<double, 1> ret = xt::empty<double>({m_N});

        for (size_t p = 0; p < m_N; ++p) {
            ret(p) = m_y(p, m_i(p) + 1);
        }

        return ret;
    }

    /**
    Closest yield position left of the particle.
    Convenience function: same as `system.y[np.arange(system.N), system.i]`.
    \return [#N].
    */
    array_type::tensor<double, 1> y_left() const
    {
        array_type::tensor<double, 1> ret = xt::empty<double>({m_N});

        for (size_t p = 0; p < m_N; ++p) {
            ret(p) = m_y(p, m_i(p));
        }

        return ret;
    }

    /**
    Set time.
    \param arg double.
    */
    void set_t(double arg)
    {
        m_inc = static_cast<size_t>(std::round(arg / m_dt));
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(xt::allclose(this->t(), arg));
    }

    /**
    Set increment.
    \param arg size_t.
    */
    void set_inc(size_t arg)
    {
        m_inc = arg;
    }

    /**
    Set position of the load frame.
    \param arg double.
    */
    void set_x_frame(double arg)
    {
        m_x_frame = arg;
        this->computeForceFrame();
        this->computeForce();
    }

    /**
    Position of the load frame.
    \return double
    */
    double x_frame() const
    {
        return m_x_frame;
    }

    /**
    Overwrite the chunk of yield positions changes #i.
    \param arg Array [#N, n_yield].
    */
    template <class T>
    void set_y(const T& arg)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(arg.dimension() == 2);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(arg.shape(0) == m_N);
        xt::noalias(m_y) = arg;
        this->updated_x();
    }

    /**
    Set the position of each particle.
    This updates the appropriate forces.
    As a rule of thumb this should be only way to update positions (even when deriving).

    \param arg The particles' positions [#N].
    */
    template <class T>
    void set_x(const T& arg)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, {m_N}));
        xt::noalias(m_x) = arg;
        this->updated_x();
    }

    /**
    Set the velocity of each particle.
    This updates the appropriate forces.
    As a rule of thumb this should be only way to update positions (even when deriving).

    \param arg The particles' velocities [#N].
    */
    template <class T>
    void set_v(const T& arg)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, {m_N}));
        xt::noalias(m_v) = arg;
        this->updated_v();
    }

    /**
    Set the acceleration of each particle.
    \param arg The particles' accelerations [#N].
    */
    template <class T>
    void set_a(const T& arg)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::has_shape(arg, {m_N}));
        xt::noalias(m_a) = arg;
    }

    /**
    Recompute all forces.
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
    Position of each particle.
    \return [#N].
    */
    const array_type::tensor<double, 1>& x() const
    {
        return m_x;
    }

    /**
    Velocity of each particle.
    \return [#N].
    */
    const array_type::tensor<double, 1>& v() const
    {
        return m_v;
    }

    /**
    Acceleration of each particle.
    \return [#N].
    */
    const array_type::tensor<double, 1>& a() const
    {
        return m_a;
    }

    /**
    Resultant force acting on each particle.
    \return [#N].
    */
    const array_type::tensor<double, 1>& f() const
    {
        return m_f;
    }

    /**
    Force associated to potentials acting on each particle.
    \return [#N].
    */
    const array_type::tensor<double, 1>& f_potential() const
    {
        return m_f_potential;
    }

    /**
    Force associated to the load frame acting on each particle.
    \return [#N].
    */
    const array_type::tensor<double, 1>& f_frame() const
    {
        return m_f_frame;
    }

    /**
    Force associated to neighbours acting on each particle.
    \return [#N].
    */
    const array_type::tensor<double, 1>& f_neighbours() const
    {
        return m_f_neighbours;
    }

    /**
    Force associated to damping on each particle.
    \return [#N].
    */
    const array_type::tensor<double, 1>& f_damping() const
    {
        return m_f_damping;
    }

    /**
    The time, see set_t().
    \return double.
    */
    double t() const
    {
        return static_cast<double>(m_inc) * m_dt;
    }

    /**
    The increment, see set_inc().
    \return size_t.
    */
    size_t inc() const
    {
        return m_inc;
    }

    /**
    The instantaneous temperature, defined as
    \f$ T \equiv (m / N) \sum\limits_{i = 1}^N v_i^2 \f$
    Note that by definition Boltzmann's constant is taken equal to 1.
    \return double.
    */
    double temperature() const
    {
        return xt::norm_sq(m_v)() * m_m / m_N;
    }

    /**
    Residual: the ratio between the norm of #f and #f_frame.
    \return double.
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
    Return the last list of residuals (use e.g. `system.risiduals.data`).
    */
    const GooseFEM::Iterate::StopList& residuals() const
    {
        return m_residuals;
    }

    /**
    Set #v and #a equal to zero.
    Call this function after an energy minimisation (taken care of in minimise()).
    */
    void quench()
    {
        m_v.fill(0.0);
        m_a.fill(0.0);
        this->updated_v();
    }

    /**
    Effectuates time step using the velocity Verlet algorithm.
    Updates #x, #v, #a, and #f, and increment #inc.
    */
    void timeStep()
    {
        m_inc++;
        xt::noalias(m_v_n) = m_v;
        xt::noalias(m_a_n) = m_a;

        xt::noalias(m_x) = m_x + m_dt * m_v + 0.5 * std::pow(m_dt, 2.0) * m_a;
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
    Make a number of time steps, see timeStep().

    \param n Number of steps to make.

    \param nmargin
        Number of potentials to leave as margin.
        -   `nmargin > 0`: stop if the yield-index of any particle is `nmargin` from the start/end
            and return a negative number.
        -   `nmargin == 0`: no bounds-check is performed and the first (last) potential
            is assumed infinitely elastic to the right (left).

    \return
        -   Number of steps: `== n`.
        -   Negative number: if stopped because of a yield-index margin.
    */
    long timeSteps(size_t n, size_t nmargin = 1)
    {
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(n + 1 < std::numeric_limits<long>::max());

        size_t nyield = m_y.shape(1);
        size_t nmax = nyield - nmargin;
        long step;

        for (step = 1; step < static_cast<long>(n + 1); ++step) {

            this->timeStep();

            if (nmargin > 0) {
                if (xt::any(m_i < nmargin) || xt::any(m_i > nmax)) {
                    return -step;
                }
            }
        }

        return step;
    }

    /**
    Perform a series of time-steps until the next plastic event, or equilibrium.

    \param nmargin Number of potentials to have as margin **initially**.
    \param tol Relative force tolerance for equilibrium. See residual() for definition.
    \param niter_tol Enforce the residual check for `niter_tol` consecutive increments.
    \param max_iter Maximum number of iterations. Throws `std::runtime_error` otherwise.

    \return
        -   Number of steps.
        -   `0` if there was no plastic activity and the residual was reached.
    */
    size_t timeStepsUntilEvent(
        size_t nmargin = 1,
        double tol = 1e-5,
        size_t niter_tol = 10,
        size_t max_iter = 1e9)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(tol < 1.0);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(max_iter + 1 < std::numeric_limits<long>::max());

        double tol2 = tol * tol;
        m_residuals.reset(niter_tol);
        size_t nyield = m_y.shape(1);
        size_t nmax = nyield - nmargin;
        auto i_n = m_i;
        size_t step;

        FRICTIONQPOTSPRINGBLOCK_ASSERT(nmargin < nyield);
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(xt::all(m_i >= nmargin) && xt::all(m_i <= nmax));

        for (step = 1; step < max_iter + 1; ++step) {

            this->timeStep();

            if (xt::any(xt::not_equal(m_i, i_n))) {
                return step;
            }

            m_residuals.roll_insert(this->residual());

            if ((m_residuals.descending() && m_residuals.all_less(tol)) ||
                m_residuals.all_less(tol2)) {
                this->quench();
                return 0;
            }
        }

        return step;
    }

    /**
    Make a number of steps with the following protocol.
    1.  Add a step \f$ v_\text{frame} \Delta t \f$ to the frame.
    2.  Make a timeStep().

    \param n
        Number of steps to make.

    \param v_frame
        Velocity of the frame.

    \param nmargin
        Number of potentials to leave as margin.
        -   `nmargin > 0`: stop if the yield-index of any particle is `nmargin` from the start/end
            and return a negative number.
        -   `nmargin == 0`: no bounds-check is performed and the first (last) potential
            is assumed infinitely elastic to the right (left).

    \return
        Number of time-steps made (negative if failure).
    */
    long flowSteps(size_t n, double v_frame, size_t nmargin = 1)
    {
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(n + 1 < std::numeric_limits<long>::max());

        size_t nyield = m_y.shape(1);
        size_t nmax = nyield - nmargin;
        long step;

        FRICTIONQPOTSPRINGBLOCK_ASSERT(nmargin < nyield);
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(xt::all(m_i >= nmargin) && xt::all(m_i <= nmax));

        for (step = 1; step < static_cast<long>(n + 1); ++step) {

            m_x_frame += v_frame * m_dt;
            this->timeStep();

            if (nmargin > 0) {
                if (xt::any(m_i < nmargin) || xt::any(m_i > nmax)) {
                    return -step;
                }
            }
        }

        return step;
    }

    /**
    Minimise energy: run timeStep() until a mechanical equilibrium has been reached.

    \param nmargin
        Number of potentials to leave as margin.
        -   `nmargin > 0`: stop if the yield-index of any particle is `nmargin` from the start/end
            and return a negative number.
        -   `nmargin == 0`: no bounds-check is performed and the first (last) potential
            is assumed infinitely elastic to the right (left).

    \param tol
        Relative force tolerance for equilibrium. See residual() for definition.

    \param niter_tol
        Enforce the residual check for `niter_tol` consecutive increments.

    \param max_iter
        Maximum number of time-steps. Throws `std::runtime_error` otherwise.

    \param time_activity
        If `true` plastic activity is timed. After this function you can find:
        -   quasistaticActivityFirst() : Increment with the first plastic event.
        -   quasistaticActivityLast() : Increment with the last plastic event.
        Attention: if you are changing the chunk of yield positions during the minimisation you
        should copy quasistaticActivityFirst() after the first (relevant) call of minimise():
        each time you call minimise(), quasistaticActivityFirst() is reset.

    \param max_iter_is_error
        If `true` an error is thrown when the maximum number of time-steps is reached.
        If `false` the function simply returns `max_iter`.

    \return
        -   `0`: if stopped when the residual is reached (and number of steps `< max_iter`).
        -   `max_iter`: if no residual was reached, and `max_iter_is_error = false`.
        -   Negative number: if stopped because of a yield-index margin.
    */
    long minimise(
        size_t nmargin = 1,
        double tol = 1e-5,
        size_t niter_tol = 10,
        size_t max_iter = 1e9,
        bool time_activity = false,
        bool max_iter_is_error = true)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(tol < 1.0);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(max_iter + 1 < std::numeric_limits<long>::max());

        double tol2 = tol * tol;
        m_residuals.reset(niter_tol);

        size_t nyield = m_y.shape(1);
        size_t nmax = nyield - nmargin;
        auto i_n = m_i;
        long s = 0;
        long s_n = 0;
        bool init = true;
        long step;

        FRICTIONQPOTSPRINGBLOCK_ASSERT(nmargin < nyield);
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(xt::all(m_i >= nmargin) && xt::all(m_i <= nmax));

        for (step = 1; step < static_cast<long>(max_iter + 1); ++step) {

            this->timeStep();
            m_residuals.roll_insert(this->residual());

            if (nmargin > 0) {
                if (xt::any(m_i < nmargin) || xt::any(m_i > nmax)) {
                    return -step;
                }
            }

            if (time_activity) {
                s = xt::sum(xt::abs(m_i - i_n))();
                if (s != s_n) {
                    if (init) {
                        init = false;
                        m_qs_inc_first = m_inc;
                    }
                    m_qs_inc_last = m_inc;
                }
                s_n = s;
            }

            if ((m_residuals.descending() && m_residuals.all_less(tol)) ||
                m_residuals.all_less(tol2)) {
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
    Increment with the first plastic event.
    This value is only relevant if `time_activity = true` was used in the last call of minimise().
    \return Increment.
    */
    size_t quasistaticActivityFirst() const
    {
        return m_qs_inc_first;
    }

    /**
    Increment with the last plastic event.
    This value is only relevant if `time_activity = true` was used in the last call of minimise().
    \return Increment.
    */
    size_t quasistaticActivityLast() const
    {
        return m_qs_inc_last;
    }

    /**
    Same as minimise() but assuming overdamped dynamics and using the no passing rule.

    \warning
        The increment is not updated as time is not physical. The mass and viscosity are ignored.
    */
    long minimise_nopassing(
        size_t nmargin = 1,
        double tol = 1e-5,
        size_t niter_tol = 10,
        size_t max_iter = 1e9)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(tol < 1.0);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(max_iter + 1 < std::numeric_limits<long>::max());

        double tol2 = tol * tol;
        m_residuals.reset(niter_tol);

        double xneigh;
        double x;
        double xmin;
        long i;
        long j;
        size_t nyield = m_y.shape(1);
        size_t nmax = nyield - nmargin;

        FRICTIONQPOTSPRINGBLOCK_ASSERT(nmargin < nyield);
        FRICTIONQPOTSPRINGBLOCK_REQUIRE(xt::all(m_i >= nmargin) && xt::all(m_i <= nmax));

        for (long step = 1; step < static_cast<long>(max_iter + 1); ++step) {

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

                i = m_i(p);
                auto* y = &m_y(p, 0);

                while (true) {
                    xmin = 0.5 * (*(y + i) + *(y + i + 1));
                    x = (m_k_neighbours * xneigh + m_k_frame * m_x_frame + m_mu * xmin) /
                        (2 * m_k_neighbours + m_k_frame + m_mu);
                    j = QPot::iterator::lower_bound(y, y + nyield, x, i);
                    if ((j < nmargin) || (j > nmax)) {
                        xt::noalias(m_x) = m_v_n;
                        return -step;
                    }
                    if (j == i) {
                        break;
                    }
                    i = j;
                }
                m_i(p) = j;
                m_x(p) = x;
            }

            this->updated_x();
            m_residuals.roll_insert(this->residual());

            if ((m_residuals.descending() && m_residuals.all_less(tol)) ||
                m_residuals.all_less(tol2)) {
                this->quench(); // no dynamics are run: make sure that the user is not confused
                return 0;
            }
        }

        throw std::runtime_error("No convergence found");
    }

    /**
    Make event driven step.

    -   `kick = false`: Increment the position of the load-frame and that of the particles to a
        new mechanical equilibrium just before yielding
        (if `direction = 1`, the new position for the particle `p` closest to yielding right is
        `x[p] = y[p] - eps / 2`).
        This assumes incrementing the load-frame infinitely slowly such that,
        in the absence of yielding, the equilibrium configuration for a new position of the load
        frame is knows.
    -   `kick = true` : Advance the system uniformly
        (the particles and the frame are moved proportionally depending on the relative stiffness)
        such that the particle closest to yielding if brought just past yielding
        (if `direction = 1`, the new position for the particle `p` closest to yielding right is
        `x[p] = y[p] + eps / 2`).

    \param eps
        Margin to keep to the position to the closest yield position.

    \param kick
        If `false`, the increment is elastic (no minimisation has to be applied after).
        If `true`, the increment leads to a state out of mechanical equilibrium.

    \param direction If `+1`: move right. If `-1` move left.
    \return Position increment of the frame.
    */
    double eventDrivenStep(double eps, bool kick, int direction = 1)
    {
        if (direction > 0 && !kick) {
            double dx = xt::amin(this->y_right() - m_x)();
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
            double dx = xt::amin(m_x - this->y_left())();
            if (dx < 0.5 * eps) {
                return 0.0;
            }
            return this->advanceUniformly(0.5 * eps - dx, false);
        }

        return this->advanceUniformly(-eps, false);
    }

    /**
    Trigger a specific particle:
    advance to the yield position right plus a margin of `eps / 2`,
    or to the left minus a margin `eps / 2`.

    \param p Particle index.
    \param eps Margin.
    \param direction If `+1`: move right. If `-1` move left.
    */
    void trigger(size_t p, double eps, int direction = 1)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(p < m_N);
        if (direction > 0) {
            m_x(p) = m_y(p, m_i(p) + 1) + 0.5 * eps;
        }
        else {
            m_x(p) = m_y(p, m_i(p)) - 0.5 * eps;
        }
        this->updated_x();
    }

    /**
    Change the position of the particles and of the loading frame such that
    the mean of f_frame() is equal to a target value, and mechanical equilibrium is maintained.

    \warning Assumes mechanical equilibrium. No assertions are made on this.
    */
    void advanceToFixedForce(double f_frame)
    {
        auto i_n = this->i();
        this->advanceUniformly((f_frame - xt::mean(m_f_frame)()) / m_mu, false);
        FRICTIONQPOTSPRINGBLOCK_ASSERT(xt::all(xt::equal(this->i(), i_n)));
    }

protected:
    /**
    \copydoc System(double, double, double, double, double, double, const T&)
    */
    template <class T>
    void initSystem(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        const T& x_yield)
    {
        FRICTIONQPOTSPRINGBLOCK_ASSERT(x_yield.dimension() == 2);

        m_N = x_yield.shape(0);
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
        m_i = xt::zeros<long>({m_N}); // consistent with `lower_bound`
        m_y = x_yield;

        this->updated_x();
        this->updated_v();
    }

    /**
    Compute #f based on the current #x and #v.
    */
    virtual void computeForce()
    {
        xt::noalias(m_f) = m_f_potential + m_f_neighbours + m_f_damping + m_f_frame;
    }

    /**
    Compute #f_potential based on the current #x.
    */
    void computeForcePotential()
    {
        QPot::inplace::lower_bound(m_y, m_x, m_i);

        for (size_t p = 0; p < m_N; ++p) {
            auto* l = &m_y(p, m_i(p));
            m_f_potential(p) = m_mu * (0.5 * (*(l) + *(l + 1)) - m_x(p));
        }
    }

    /**
    Compute #f_neighbours based on the current #x.
    */
    void computeForceNeighbours()
    {
        for (size_t p = 1; p < m_N - 1; ++p) {
            m_f_neighbours(p) = m_k_neighbours * (m_x(p - 1) - 2 * m_x(p) + m_x(p + 1));
        }
        m_f_neighbours.front() = m_k_neighbours * (m_x.back() - 2 * m_x.front() + m_x(1));
        m_f_neighbours.back() = m_k_neighbours * (m_x(m_N - 2) - 2 * m_x.back() + m_x.front());
    }

    /**
    Compute #f_frame based on the current #x.
    */
    void computeForceFrame()
    {
        xt::noalias(m_f_frame) = m_k_frame * (m_x_frame - m_x);
    }

    /**
    Compute #f_damping based on the current #v.
    */
    void computeForceDamping()
    {
        xt::noalias(m_f_damping) = -m_eta * m_v;
    }

    /**
    Update forces that depend on #m_x.
    */
    void updated_x()
    {
        this->computeForcePotential();
        this->computeForceNeighbours();
        this->computeForceFrame();
        this->computeForce();
    }

    /**
    Update forces that depend on #m_v.
    */
    void updated_v()
    {
        this->computeForceDamping();
        this->computeForce();
    }

    /**
    Advance the system uniformly: the particles and the frame are moved proportionally,
    while all particles are moved in the same way.
    This maintains equilibrium as long as no particle yields.
    In particular:

    \f$ \Delta f^\mathrm{potential}_i = \mu \Delta x_i \f$

    \f$ \Delta f^\mathrm{frame} = k (\Delta x^\mathrm{frame} - \Delta x_i) \f$

    This gives two cases:

    -   Given \f$ \Delta x_i \f$ one finds:
        \f$ \Delta x^\mathrm{frame} = (k + \mu) / k \Delta x_i \f$.

    -   Given \f$ \Delta x^\mathrm{frame} \f$ one finds:
        \f$ \Delta x_i = k / (k + \mu) \Delta x^\mathrm{frame} \f$.

    \param dx Displacement.

    \param input_is_frame
        If `true`: `dx` \f$ = \Delta x^\mathrm{frame} \f$,
        if `false`: `dx` \f$ = \Delta x_i \f$,

    \return `dx` for the particles in `input_is_frame == true`, otherwise `dx` of the frame.
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
    array_type::tensor<double, 2> m_y; ///< Potential energy landscape.
    array_type::tensor<long, 1> m_i; ///< Current index in the potential energy landscape.
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
    GooseFEM::Iterate::StopList m_residuals; ///< List of residuals of the last minimisation.
};

/**
## Introduction

System in which the effect of temperature in mimicked by random forcing.
The random forces can be set:
-   Instantaneously, using setRandomForce().
-   As a sequence depending on the current increment, using setRandomForceSequence().

## Physics & API

The physics and API are the same as in System(), with the difference that the residual force now
reads \f$ f_\text{residual}^{(i)} = f_\text{damping}^{(i)} + f_\text{potential}^{(i)} +
f_\text{neighbours}^{(i)} + f_\text{frame}^{(i)} + f_\text{random}^{(i)} \f$.
Thereby \f$ f_\text{random}^{(i)} \f$ is specified instantaneously setRandomForce() or for a
time-history in setRandomForceSequence().

## Apply fixed force instead of fixed displacement

To apply a fixed force (athermal or thermal) use setRandomForce() or setRandomForceSequence()
with a non-zero mean *and* set `k_frame = 0`.

## Internal strategy

To avoid code duplication this class derives from System().
To ensure the correct physics computeForce() is overridden to add \f$ f_\text{random}^{(i)} \f$.
*/
class SystemThermalRandomForcing : public System {
public:
    SystemThermalRandomForcing() = default;

    /**
    \copydoc System(double, double, double, double, double, double, const T&)
    */
    template <class T>
    SystemThermalRandomForcing(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        const T& x_yield)
    {
        this->initSystemThermalRandomForcing(m, eta, mu, k_neighbours, k_frame, dt, x_yield);
    }

    /**
    Set random force.
    This force will be applied until it is overwritten, or a random force sequence is set in
    setRandomForceSequence().
    If a sequence was set before, it will be discarded and replaced by this force.

    \param f Force per particle.
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
    Set sequence of random forces.
    This sequence specifies for each particle which force should be applied at which increment.

    \param f Sequence of forces per particle [#N, `n`].

    \param start_inc
        Start and end increment of each item [#N, `n + 1`].
        This implies that on a particle `p`, the force `f(p, i)` will be applied starting from
        increment `start_inc(p, i)` until (but not including) `start_inc(p, i + 1)`.
        The sequence is thus bounded and should be updated in time.
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
    \copydoc SystemThermalRandomForcing(double, double, double, double, double, double, const T&)
    */
    template <class T>
    void initSystemThermalRandomForcing(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        const T& x_yield)
    {
        m_seq = false;
        m_f_thermal = xt::zeros<double>({x_yield.shape(0)});
        this->initSystem(m, eta, mu, k_neighbours, k_frame, dt, x_yield);
    }

    /**
    Update 'thermal' force for sequence (if needed).
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

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#endif
