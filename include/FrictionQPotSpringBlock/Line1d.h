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
This class stores a sequence of yield positions around the current position.
When the particle is close to running out-of-bounds the system redraws a sequence of yield positions
by drawing yield distances from a distribution (specified as function in the constructor).
*/
class System {

public:
    System() = default;

    /**
    Constructor.

    \param N Number of particles.
    \param y Initial yield positions [#N, n_yield].
    */
    template <class T>
    System(size_t N, const T& y);

    /**
    Constructor.

    \param N Number of particles.
    \param y Initial yield positions [#N, n_yield].
    \param istart Starting index corresponding to y[:, 0], [#N].
    */
    template <class T, class I>
    System(size_t N, const T& y, const I& istart);

    /**
    Number of particles.

    \return unsigned int
    */
    size_t N() const;

    /**
    Return reference to the underlying QPot::Chunked storage

    \param p Particle number.
    \return Reference.
    */
    QPot::Chunked& y(size_t p);

    /**
    \copydoc QPot::Chunked::set_y(long, const T&)
    */
    template <class I, class T>
    void set_y(const I& istart, const T& y);

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
    \copydoc QPot::Chunked::boundcheck_left()
    */
    xt::xtensor<bool, 1> boundcheck_left(size_t n = 0) const;

    /**
    \copydoc QPot::Chunked::boundcheck_right()
    */
    xt::xtensor<bool, 1> boundcheck_right(size_t n = 0) const;

    /**
    Check if any yield position chunk needs to be updated based on the current x().

    \return true if redraw is needed for one of more particle.
    */
    bool any_redraw() const;

    /**
    Check if any yield position chunk needs to be updated if the position would be updated to
    a given value.

    \param x Trial particle positions (internally the position is not updated).
    \return true if redraw is needed for one of more particle.
    */
    template <class T>
    bool any_redraw(const T& x) const;

    /**
    Check if any particle if within `n` potentials for the left- or the right-most yield
    positions of the current chunk.

    \param n Size of boundary region.
    \return true if one of more particle is in the left or right boundary region.
    */
    bool any_shift(size_t n) const;

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
    Set time step.

    \param arg double.
    */
    void set_dt(double arg);

    /**
    Set damping coefficient (same for all particles).
    The damping force is ``- eta * v``.

    \param arg double.
    */
    void set_eta(double arg);

    /**
    Set particle mass (same for all particles).

    \param arg double.
    */
    void set_m(double arg);

    /**
    Set elastic stiffness (same for all particles).

    \param arg double.
    */
    void set_mu(double arg);

    /**
    Set stiffness of the springs connecting neighbours (same for all particles).

    \param arg double.
    */
    void set_k_neighbours(double arg);

    /**
    Set stiffness of the springs connecting a particle to the load frame (same for all particles).
    To be set ~1 / N.

    \param arg double.
    */
    void set_k_frame(double arg);

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
    The time step, see set_dt().

    \return double.
    */
    double dt() const;

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

    \param dx_of_frame
        If `true`: `dx` \f$ = \Delta x^\mathrm{frame} \f$,
        if `false`: `dx` \f$ = \Delta x_i \f$,
    */
    void advanceElastic(double dx, bool dx_of_frame = true);

    /**
    Event driven advance right to closest yielding point, leaving ``eps / 2`` as margin.

    \param eps Margin.
    */
    void advanceEventRightElastic(double eps);

    /**
    Event driven: advance right by ``eps``.

    \param eps Step size.
    */
    void advanceEventRightKick(double eps);

    /**
    \cond
    */
    [[deprecated]] void advanceRightElastic(double eps);

    [[deprecated]] void advanceRightKick(double eps);
    /**
    \endcond
    */

    /**
    Trigger a specific particle:
    advance to the yield positions right plus a margin of `eps / 2`.

    \param p Particle index.
    \param eps Margin.
    */
    void triggerRight(size_t p, double eps);

    /**
    Trigger the closest point to yielding right:
    advance to the yield positions right plus a margin of `eps / 2`.

    \param eps Margin.
    */
    void triggerWeakestRight(double eps);

protected:
    /**
    Initialise the system.

    \param N Number of particles.
    \param y Initial yield positions.
    \param istart Starting index corresponding to y[:, 0]
    */
    template <class T, class I>
    void init(size_t N, const T& y, const I& istart);

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
    std::vector<QPot::Chunked> m_y; ///< Potential energy landscape.
    size_t m_N; ///< See #N.
    double m_t = 0.0; ///< See #set_t.
    double m_dt = 0.1; ///< See #set_dt.
    double m_eta = 2.0 * std::sqrt(3.0) / 10.0; ///< See #set_eta.
    double m_m = 1.0; ///< See #set_m.
    double m_mu = 1.0; ///< See #set_mu.
    double m_k_neighbours = 1.0; ///< See #set_k_neighbours.
    double m_k_frame = 1.0; ///< See #set_k_frame.
    double m_x_frame; ///< See #set_x_frame.
};

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#include "Line1d.hpp"

#endif
