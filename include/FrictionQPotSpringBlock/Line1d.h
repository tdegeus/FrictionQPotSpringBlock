/**
Line in 1d.

\file Line1d.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_H
#define FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_H

#include <xtensor/xtensor.hpp>

#include "config.h"
#include "version.h"

#include <GooseFEM/Iterate.h>
#include <GooseFEM/version.h>
#include <QPot.h>
#include <string>
#include <xtensor/xnorm.hpp>
#include <xtensor/xshape.hpp>

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
inline std::vector<std::string> version_dependencies();

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
    \f$ f_\text{frame}^{(i)} =  k_\text{frame} (x_\text{frame} - x_i \f$.
    Typically \f$ k_\text{frame} = \mathcal{O}(1 / N) \f$ with \f$ N \f$ the number of particles.

The dynamics follow
\f$ m a_i = f_\text{residual}^{(i)} \f$, with the particle's acceleration
\f$ a_i \equiv \ddot{x}_i \equiv \partial_t^2 x_i \f$, and the residual force
\f$ f_\text{residual}^{(i)} = f_\text{damping}^{(i)} + f_\text{potential}^{(i)} +
f_\text{neighbours}^{(i)} + f_\text{frame}^{(i)} \f$.
This differential equation is integrated using Velocity-Verlet in timeStep()
Note that this function updates the time t() (or increment number inc()),
and that this is the only function that does so.

The implementation is is such that all forces are updated (if needed) every time the positions
and velocities are updated, either internally in timeStep(), or externally when calling
set_x() and set_v().

## Yield positions

For computational efficiency one can choose to store only a chunk of the sequence of yield positions
around the current position, which is updated when the particle is close to running out-of-bounds,
see shift_y() and shift_dy(), or set_y().

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
        const T& x_yield);

    /**
    Number of particles.

    \return unsigned int
    */
    size_t N() const;

    /**
    Return yield positions.
    \return Array of shape [#N, n].
    */
    const array_type::tensor<double, 2>& y();

    /**
    Current index in the current chunk of the potential energy landscape (for each particle).
    \return Array of shape [#N].
    */
    const array_type::tensor<long, 1>& i() const;

    /**
    Distance to yield to the right (for each particle).
    Convenience function: same as `y[arange(N), i + 1] - x`.

    \return [#N].
    */
    array_type::tensor<double, 1> yieldDistanceRight() const;

    /**
    Distance to yield to the left for each particle.
    Convenience function: same as `x - y[arange(N), i + 1]`.

    \return [#N].
    */
    array_type::tensor<double, 1> yieldDistanceLeft() const;

    /**
    Set time.
    \param arg double.
    */
    void set_t(double arg);

    /**
    Set increment.
    \param arg size_t.
    */
    void set_inc(size_t arg);

    /**
    Set position of the load frame.

    \param arg double.
    */
    void set_x_frame(double arg);

    /**
    Position of the load frame.

    \return double
    */
    double x_frame() const;

    /**
    Overwrite the yield positions.

    \param arg Array [#N, n].
    */
    template <class T>
    void set_y(const T& arg);

    /**
    Set the position of each particle.
    This updates the appropriate forces.
    As a rule of thumb this should be only way to update positions (even when deriving).

    \param arg The particles' positions [#N].
    */
    template <class T>
    void set_x(const T& arg);

    /**
    Set the velocity of each particle.
    This updates the appropriate forces.
    As a rule of thumb this should be only way to update positions (even when deriving).

    \param arg The particles' velocities [#N].
    */
    template <class T>
    void set_v(const T& arg);

    /**
    Set the acceleration of each particle.

    \param arg The particles' accelerations [#N].
    */
    template <class T>
    void set_a(const T& arg);

    /**
    Position of each particle.

    \return [#N].
    */
    const array_type::tensor<double, 1>& x() const;

    /**
    Velocity of each particle.

    \return [#N].
    */
    const array_type::tensor<double, 1>& v() const;

    /**
    Acceleration of each particle.

    \return [#N].
    */
    const array_type::tensor<double, 1>& a() const;

    /**
    Resultant force acting on each particle.

    \return [#N].
    */
    const array_type::tensor<double, 1>& f() const;

    /**
    Force associated to potentials acting on each particle.

    \return [#N].
    */
    const array_type::tensor<double, 1>& f_potential() const;

    /**
    Force associated to the load frame acting on each particle.

    \return [#N].
    */
    const array_type::tensor<double, 1>& f_frame() const;

    /**
    Force associated to neighbours acting on each particle.

    \return [#N].
    */
    const array_type::tensor<double, 1>& f_neighbours() const;

    /**
    Force associated to damping on each particle.

    \return [#N].
    */
    const array_type::tensor<double, 1>& f_damping() const;

    /**
    The time, see set_t().
    \return double.
    */
    double t() const;

    /**
    The increment, see set_inc().
    \return size_t.
    */
    size_t inc() const;

    /**
    The instantaneous temperature, defined as
    \f$ T \equiv m / N \sum\limits_{i = 1}^N v_i^2 \f$
    Note that by definition Boltzmann's constant is taken equal to 1.
    \return double.
    */
    double temperature() const;

    /**
    Residual: the ratio between the norm of #f and #f_frame.

    \return double.
    */
    double residual() const;

    /**
    Set #v and #a equal to zero.
    Call this function after an energy minimisation (taken care of in minimise()).
    */
    void quench();

    /**
    Effectuates time step using the velocity Verlet algorithm.
    Updates #x, #v, #a, and #f, and increment #inc.
    */
    void timeStep();

    /**
    Make a number of time steps, see timeStep().
    \param n Number of steps to make.
    */
    void timeSteps(size_t n);

    /**
    Perform a series of time-steps until the next plastic event, or equilibrium.

    \param tol
        Relative force tolerance for equilibrium. See residual() for definition.

    \param niter_tol
        Enforce the residual check for ``niter_tol`` consecutive increments.

    \param max_iter
        Maximum number of iterations. Throws ``std::runtime_error`` otherwise.

    \return
        The number of iterations.
        `0` is returned if there was no plastic activity and the residual was reached.
    */
    size_t timeStepsUntilEvent(double tol = 1e-5, size_t niter_tol = 10, size_t max_iter = 1e9);

    /**
    Make a number of steps with the following protocol.
    (1) Add a step \f$ v_\text{frame} \Delta t \f$ to the frame.
    (2) Make a timeStep().

    \param n Number of steps to make.
    \param v_frame Velocity of the frame.
    */
    void flowSteps(size_t n, double v_frame);

    /**
    \copydoc flowSteps(size_t, double)

    This function stops if the yield-index of any particle is close the end.
    In that case the function returns zero (in all other cases it returns a positive number).

    \param nmargin
        Number of potentials to leave as margin.
    */
    size_t flowSteps_boundcheck(size_t n, double v_frame, size_t nmargin = 5);

    /**
    Minimise energy: run timeStep() until a mechanical equilibrium has been reached.

    A concern is running out-of-bounds in terms of yield positions.
    Consider setting `nmargin` to your need:

    -   If `nmargin > 0` this function checks, after every time-step, that the yield-positions
        are not within `nmargin` from running out-of-bounds (for every particle).
        If that is the case the function stops and
        returns the number of iterations as a negative number.
        It is relatively safe to update the chunk of yield-positions and continue the minimisation
        by recalling this function.
        If unlucky, this can slightly delay finding equilibrium.

    -   If `nmargin == 0` there is no protection from running out-of-bounds.

    \param nmargin
        Number of potentials to leave as margin (see above).

    \param tol
        Relative force tolerance for equilibrium. See residual() for definition.

    \param niter_tol
        Enforce the residual check for ``niter_tol`` consecutive increments.

    \param max_iter
        Maximum number of iterations. Throws ``std::runtime_error`` otherwise.

    \return
        The number of iterations.
        **If a negative number if returned, equilibrium was not reached.**
    */
    long
    minimise(size_t nmargin = 0, double tol = 1e-5, size_t niter_tol = 10, size_t max_iter = 1e9);

    /**
    Same as minimise() but with the difference that plastic activity is timed.
    After this function you can find:

    -   quasistaticActivityFirst() : Increment with the first plastic event.
    -   quasistaticActivityLast() : Increment with the last plastic event.

    \warning
        The number of iterations.
        **If a negative number if returned, equilibrium was not reached.**
    */
    long minimise_timeactivity(
        size_t nmargin = 0,
        double tol = 1e-5,
        size_t niter_tol = 10,
        size_t max_iter = 1e9);

    /**
    Increment with the first plastic event.
    This is the output of the last call of minimise_timeactivity().
    */
    size_t quasistaticActivityFirst() const;

    /**
    Increment with the last plastic event.
    This is the output of the last call of minimise_timeactivity().
    */
    size_t quasistaticActivityLast() const;

    /**
    Same as minimise() but assuming  overdamped dynamics and using the no passing rule.

    \warning
        The increment is not updated as time is not physical. The mass and viscosity are ignored.
    */
    long minimise_nopassing(
        size_t nmargin = 0,
        double tol = 1e-5,
        size_t niter_tol = 10,
        size_t max_iter = 1e9);

    /**
    Make event driven step.
    *   `kick = false`: Increment the position of the load-frame and that of the particles to a
        new mechanical equilibrium just before yielding
        (if `direction = 1`, the new position for the particle `p` closest to yielding right is
        `x[p] = y[p] - eps / 2`).
        This assumes incrementing the load-frame infinitely slowly such that,
        in the absence of yielding, the equilibrium configuration for a new position of the load
        frame is knows.
    *   `kick = true` : Advance the system uniformly
        (the particles and the frame are moved proportionally depending on the relative stiffness)
        such that the particle closest to yielding if brought just past yielding
        (if `direction = 1`, the new position for the particle `p` closest to yielding right is
        `x[p] = y[p] + eps / 2`).

    \param eps
        Margin to keep to the position to the closest yield position.

    \param kick
        If ``false``, the increment is elastic (no minimisation has to be applied after).
        If ``true``, the increment leads to a state out of mechanical equilibrium.

    \param direction If ``+1``: move right. If ``-1`` move left.
    \return Position increment of the frame.
    */
    double eventDrivenStep(double eps, bool kick, int direction = 1);

    /**
    Trigger a specific particle:
    advance to the yield position right plus a margin of `eps / 2`,
    or to the left minus a margin `eps / 2`.

    \param p Particle index.
    \param eps Margin.
    \param direction If ``+1``: move right. If ``-1`` move left.
    */
    void trigger(size_t p, double eps, int direction = 1);

    /**
    Trigger the closest point to yielding right or left, see trigger().

    \param eps Margin.
    \param direction If ``+1``: move right. If ``-1`` move left.
    \return The index of the triggered particle.
    */
    size_t triggerWeakest(double eps, int direction = 1);

    /**
    Change the position of the particles and of the loading frame such that
    the mean of f_frame() is equal to a target value, and mechanical equilibrium is maintained.

    \warning Call from a state of mechanical equilibrium. No assertions on this are made.
    */
    void advanceToFixedForce(double f_frame);

private:
    long _minimise_nocheck(double tol, size_t niter_tol, long max_iter);
    long _minimise_check(size_t nmargin, double tol, size_t niter_tol, long max_iter);
    long _minimise_timeactivity_nocheck(double tol, size_t niter_tol, long max_iter);
    long _minimise_timeactivity_check(size_t nmargin, double tol, size_t niter_tol, long max_iter);
    long _minimise_nopassing_nocheck(double tol, size_t niter_tol, long max_iter);
    long _minimise_nopassing_check(size_t nmargin, double tol, size_t niter_tol, long max_iter);

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
        const T& x_yield);

    /**
    Compute #f based on the current #x and #v.
    */
    virtual void computeForce();

    /**
    Compute #f_potential based on the current #x.
    */
    void computeForcePotential();

    /**
    Compute #f_neighbours based on the current #x.
    */
    void computeForceNeighbours();

    /**
    Compute #f_frame based on the current #x.
    */
    void computeForceFrame();

    /**
    Compute #f_damping based on the current #v.
    */
    void computeForceDamping();

    /**
    Update forces that depend on #m_x.
    */
    void updated_x();

    /**
    Update forces that depend on #m_v.
    */
    void updated_v();

    /**
    Advance the system uniform: the particles and the frame are moved proportionally,
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
    double advanceUniformly(double dx, bool input_is_frame = true);

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
        const T& x_yield);

    /**
    Set random force.
    This force will be applied until it is overwritten, or a random force sequence is set in
    setRandomForceSequence().
    If a sequence was set before, it will be discarded and replaced by this force.

    \param f Force per particle.
    */
    template <class T>
    void setRandomForce(const T& f);

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
    void setRandomForceSequence(const T& f, const U& start_inc);

protected:
    // clang-format off
    /**
    \copydoc SystemThermalRandomForcing(double, double, double, double, double, double, const T&)
    */
    // clang-format on
    template <class T>
    void initSystemThermalRandomForcing(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        const T& x_yield);

    /**
    Update 'thermal' force for sequence (if needed).
    */
    void updateThermalForce();

    void computeForce() override;

    bool m_seq = false; ///< Use sequence to set random forces, set in setRandomForceSequence().
    array_type::tensor<double, 2> m_seq_f; ///< Sequence of random forces.
    array_type::tensor<size_t, 2> m_seq_s; ///< Start/end increment of each item in the sequence.
    array_type::tensor<size_t, 1> m_seq_i; ///< Current column in #m_seq_f for each particle.
    array_type::tensor<double, 1> m_f_thermal; ///< Current applied 'random' forces.
};

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#include "Line1d.hpp"

#endif
