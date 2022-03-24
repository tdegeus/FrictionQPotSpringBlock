/**
Line in 1d.

\file Line1d.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_H
#define FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_H

#include "config.h"
#include "version.h"

#include <GooseFEM/Iterate.h>
#include <GooseFEM/version.h>
#include <QPot/Chunked.hpp>
#include <string>
#include <xtensor/xnorm.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xtensor.hpp>

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
The system in which particles experience a piece-wise quadratic local potential energy,
elastic interactions with neighbours, and a drive though a spring attached to a load frame.
The local landscape is characterised by yield positions.
This class stores a chunk of the sequence of yield positions around the current position.
When the particle is close to running out-of-bounds,
you have the option to change the chunk of yield positions.

The physics are as follows:
*   A particle has a mass \f$ m \f$.
*   Each particle \f$ i \f$ experiences damping equal to
    \f$ f_\text{damping}^{(i)} = - \eta v_i \f$.
*   Each particle \f$ i \f$ has a potential energy such that
    \f$ f_\text{potential}^{(i)} =  \mu (x_{\min}^{(i)} - x_i) \f$.
    where \f$ \mu \f$ is the radius of curvature of the quadratic potentials, and
    \f$ x_{\min}^{(i)} \f$ the positions of the current local minimum.
*   Each particle \f$ i \f$ has interactions with its neighbours equal to
    \f$ f_\text{neighbours}^{(i)} =  k_\text{neighbours} (x_{i - 1} - 2 x_i + x_{i + 1}) \f$.
*   Each particle \f$ i \f$ is connected to the load frame given a force equal to
    \f$ f_\text{frame}^{(i)} =  k_\text{frame} (x_\text{frame} - x_i \f$.
    Typically \f$ k_\text{frame} = \mathcal{O}(1 / N) \f$ with \f$ N \f$ the number of particles.
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
    \param x_yield Initial yield positions [#N, n_yield].
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
    \copydoc System(double, double, double, double, double, double, const T&)
    \param istart Starting index corresponding to x_yield[:, 0], [#N].
    */
    template <class T, class I>
    System(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        const T& x_yield,
        const I& istart);

    /**
    Number of particles.

    \return unsigned int
    */
    size_t N() const;

    /**
    Return yield positions.
    \return Array.
    */
    xt::xtensor<double, 2> y();

    /**
    Return reference to the underlying QPot::Chunked storage.

    \param p Particle number.
    \return Reference.
    */
    QPot::Chunked& refChunked(size_t p);

    /**
    \copydoc QPot::Chunked::set_y(long, const T&)
    */
    template <class I, class T>
    void set_y(const I& istart, const T& y);

    /**
    \copydoc QPot::Chunked::shift_y(long, const T&, size_t)
    */
    template <class I, class T>
    void shift_y(const I& istart, const T& y, size_t nbuffer = 0);

    /**
    \copydoc QPot::Chunked::shift_dy(long, const T&, size_t)
    */
    template <class I, class T>
    void shift_dy(const I& istart, const T& dy, size_t nbuffer = 0);

    /**
    \copydoc QPot::Chunked::set_y(long, const T&)
    \param p Particle number.
    */
    template <class T>
    void set_y(size_t p, long istart, const T& y);

    /**
    \copydoc QPot::Chunked::shift_y(long, const T&, size_t)
    \param p Particle number.
    */
    template <class T>
    void shift_y(size_t p, long istart, const T& y, size_t nbuffer = 0);

    /**
    \copydoc QPot::Chunked::shift_dy(long, const T&, size_t)
    \param p Particle number.
    */
    template <class T>
    void shift_dy(size_t p, long istart, const T& dy, size_t nbuffer = 0);

    /**
    \copydoc QPot::Chunked::ymin()
    */
    xt::xtensor<double, 1> ymin() const;

    /**
    \copydoc QPot::Chunked::ymin_chunk()
    */
    xt::xtensor<double, 1> ymin_chunk() const;

    /**
    \copydoc QPot::Chunked::yleft()
    */
    xt::xtensor<double, 1> yleft() const;

    /**
    \copydoc QPot::Chunked::yright()
    */
    xt::xtensor<double, 1> yright() const;

    /**
    \copydoc QPot::Chunked::i_chunk()
    */
    xt::xtensor<size_t, 1> i_chunk() const;

    /**
    \copydoc QPot::Chunked::istart()
    */
    xt::xtensor<long, 1> istart() const;

    /**
    \copydoc QPot::Chunked::istop()
    */
    xt::xtensor<long, 1> istop() const;

    /**
    \copydoc QPot::Chunked::inbounds_left()
    */
    xt::xtensor<bool, 1> inbounds_left(size_t n = 0) const;

    /**
    \copydoc QPot::Chunked::inbounds_right()
    */
    xt::xtensor<bool, 1> inbounds_right(size_t n = 0) const;

    /**
    \copydoc QPot::Chunked::inbounds_left()
    */
    bool all_inbounds_left(size_t n = 0) const;

    /**
    \copydoc QPot::Chunked::inbounds_right()
    */
    bool all_inbounds_right(size_t n = 0) const;

    /**
    \copydoc QPot::Chunked::inbounds()
    */
    bool all_inbounds(size_t n = 0) const;

    /**
    Check if any yield position chunk needs to be updated based on the current x().

    \return true if redraw is needed for one of more particle.
    */
    bool any_redraw() const;

    /**
    Check if any yield position chunk needs to be updated if the position would be updated
    to a given value.

    \param x Trial particle positions (internally the position is not updated).
    \return true if redraw is needed for one of more particle.
    */
    template <class T>
    bool any_redraw(const T& x) const;

    /**
    Current index in the global potential energy landscape (for each particle).

    \return [#N].
    */
    xt::xtensor<long, 1> i() const;

    /**
    Distance to yield to the right (for each particle).

    \return [#N].
    */
    xt::xtensor<double, 1> yieldDistanceRight() const;

    /**
    Distance to yield to the left (for each particle).

    \return [#N].
    */
    xt::xtensor<double, 1> yieldDistanceLeft() const;

    /**
     * Set time.
     * \param arg double.
     */
    void set_t(double arg);

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
    xt::xtensor<double, 1> x() const;

    /**
    Velocity of each particle.

    \return [#N].
    */
    xt::xtensor<double, 1> v() const;

    /**
    Acceleration of each particle.

    \return [#N].
    */
    xt::xtensor<double, 1> a() const;

    /**
    Resultant force acting on each particle.

    \return [#N].
    */
    xt::xtensor<double, 1> f() const;

    /**
    Force associated to potentials acting on each particle.

    \return [#N].
    */
    xt::xtensor<double, 1> f_potential() const;

    /**
    Force associated to the load frame acting on each particle.

    \return [#N].
    */
    xt::xtensor<double, 1> f_frame() const;

    /**
    Force associated to neighbours acting on each particle.

    \return [#N].
    */
    xt::xtensor<double, 1> f_neighbours() const;

    /**
    Force associated to damping on each particle.

    \return [#N].
    */
    xt::xtensor<double, 1> f_damping() const;

    /**
     * The time, see set_t().
     * \return double.
     */
    double t() const;

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
    Updates #x, #v, #a, and #f.
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
    size_t
    timeStepsUntilEvent(double tol = 1e-5, size_t niter_tol = 10, size_t max_iter = 10000000);

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

    \param tol
        Relative force tolerance for equilibrium. See residual() for definition.

    \param niter_tol
        Enforce the residual check for ``niter_tol`` consecutive increments.

    \param max_iter
        Maximum number of iterations. Throws ``std::runtime_error`` otherwise.

    \return The number of iterations.
    */
    size_t minimise(double tol = 1e-5, size_t niter_tol = 10, size_t max_iter = 10000000);

    /**
    \copydoc System::minimise(double, size_t, size_t)

    This function stops if the yield-index of any particle is close the end.
    In that case the function returns zero (in all other cases it returns a positive number),
    and reverts the state of the system to that before calling this function.

    \param nmargin
        Number of potentials to leave as margin.
    */
    size_t minimise_boundcheck(
        size_t nmargin = 5,
        double tol = 1e-5,
        size_t niter_tol = 20,
        size_t max_iter = 10000000);

    /**
    Minimise energy: run timeStep() until a mechanical equilibrium has been reached.
    Compared to minimise() this function measures the duration of an avalanche.

    \param tol
        Relative force tolerance for equilibrium. See residual() for definition.

    \param niter_tol
        Enforce the residual check for ``niter_tol`` consecutive increments.

    \param max_iter
        Maximum number of iterations. Throws ``std::runtime_error`` otherwise.

    \return The number of iterations elapsed during an avalanche (in units of dt()).
    */
    size_t
    minimise_timeactivity(double tol = 1e-5, size_t niter_tol = 10, size_t max_iter = 10000000);

    /**
    \copydoc System::minimise_timeactivity(double, size_t, size_t)

    This function stops if the yield-index of any particle is close the end.
    In that case the function returns zero (in all other cases it returns a positive number),
    and reverts the state of the system to that before calling this function.

    \param nmargin
        Number of potentials to leave as margin.
    */
    size_t minimise_timeactivity_boundcheck(
        size_t nmargin = 5,
        double tol = 1e-5,
        size_t niter_tol = 10,
        size_t max_iter = 10000000);

    /**
    Advance the system elastically: the particles and the frame are moved proportionally,
    such that equilibrium is maintained.
    Note that the displacement is uniform: all particles are moved in the same way.
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
    double advanceElastic(double dx, bool input_is_frame = true);

    /**
    Make event driving step.

    \param eps
        Margin in position to the closest yield position.

    \param kick
        If ``false``, increment positions to ``eps / 2`` of yielding again.
        If ``true``, increment positions by ``eps``.

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

protected:
    /**
    \copydoc System(double, double, double, double, double, double, const T&, const I&)
    */
    template <class T, class I>
    void init(
        double m,
        double eta,
        double mu,
        double k_neighbours,
        double k_frame,
        double dt,
        const T& x_yield,
        const I& istart);

    /**
    Compute #f based on the current #x and #v.
    */
    void computeForce();

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

protected:
    xt::xtensor<double, 1> m_f; ///< See #f.
    xt::xtensor<double, 1> m_f_potential; ///< See #f_potential.
    xt::xtensor<double, 1> m_f_neighbours; ///< See #f_neighbours.
    xt::xtensor<double, 1> m_f_frame; ///< See #f_frame.
    xt::xtensor<double, 1> m_f_damping; ///< See #f_damping.
    xt::xtensor<double, 1> m_x; ///< See #x.
    xt::xtensor<double, 1> m_v; ///< See #v.
    xt::xtensor<double, 1> m_a; ///< See #a.
    xt::xtensor<double, 1> m_v_n; ///< #v at last time-step.
    xt::xtensor<double, 1> m_a_n; ///< #a at last time-step.
    xt::xtensor<double, 1> m_x_t; ///< #v at some point in history (used in #minimise_boundcheck).
    xt::xtensor<double, 1> m_v_t; ///< #v at some point in history (used in #minimise_boundcheck).
    xt::xtensor<double, 1> m_a_t; ///< #a at some point in history (used in #minimise_boundcheck).
    std::vector<QPot::Chunked> m_y; ///< Potential energy landscape.
    size_t m_N; ///< See #N.
    double m_t = 0.0; ///< See #set_t.
    double m_t_t; ///< #t at some point in history (used in #minimise_boundcheck)
    double m_dt; ///< Time step.
    double m_eta; ///< Damping constant (same for all particles).
    double m_m; ///< Mass (same for all particles).
    double m_mu; ///< Stiffness: curvature of the potential energy (same for all particles).
    double m_k_neighbours; ///< Stiffness of interactions (same for all particles).
    double m_k_frame; ///< Stiffness of the load fame (same for all particles).
    double m_x_frame = 0.0; ///< See #set_x_frame.
};

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#include "Line1d.hpp"

#endif
