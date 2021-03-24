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

#include <QPot/Redraw.hpp>
#include <GooseFEM/version.h>
#include <GooseFEM/Iterate.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xshape.hpp>
#include <string>

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
    \param func Function to draw yield distances.
    */
    template <class F>
    System(size_t N, F func);

    /**
    Constructor.

    \param N
        Number of particles.

    \param func
        Function to draw yield distances.

    \param ntotal
        Number of yield-positions to keep in memory.

    \param nbuffer
        Number of yield-positions to buffer when shifting left/right.

    \param noffset
        Number of yield-positions from the end to consider for redrawing
        (allows grouping of redraws for several particles).
    */
    template <class F>
    System(size_t N, F func, size_t ntotal, size_t nbuffer, size_t noffset);

    /**
    Number of particles.

    \return unsigned int
    */
    size_t N() const;

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

    \param arg [#N].
    \return ``true`` is there was a redraw.
    */
    bool set_x(const xt::xtensor<double, 1>& arg);

    /**
    Set the velocity of each particle.
    This updates the appropriate forces.
    As a rule of thumb this should be only way to update positions (even when deriving).

    \param arg [#N].
    */
    void set_v(const xt::xtensor<double, 1>& arg);

    /**
    Set the acceleration of each particle.

    \param arg [#N].
    */
    void set_a(const xt::xtensor<double, 1>& arg);

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

    \return ``true`` is there was a redraw.
    */
    bool timeStep();

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
    size_t minimise(double tol = 1e-5, size_t niter_tol = 10, size_t max_iter = 1000000);

    /**
    Event driven advance right to closest yielding point, leaving ``delta_x / 2`` as margin.

    \param delta_x Margin.
    \return ``true`` is there was a redraw.
    */
    bool advanceRightElastic(double delta_x);

    /**
    Event driven: advance right by ``delta_x``.

    \param delta_x Step size.
    \return ``true`` is there was a redraw.
    */
    bool advanceRightKick(double delta_x);

    /**
    Current yield position to the left (for each particle).

    \return [#N].
    */
    xt::xtensor<double, 1> yieldLeft() const;

    /**
    Current yield position to the right (for each particle).

    \return [#N].
    */
    xt::xtensor<double, 1> yieldRight() const;

    /**
    Current index in the local potential energy landscape (for each particle).

    \return [#N].
    */
    xt::xtensor<int, 1> yieldIndex() const;

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

    */
    xt::xtensor<int, 1> yield_currentRedraw() const;

    void yield_redraw(const xt::xtensor<int, 1>& index);

    void yield_redrawRight(const xt::xtensor<size_t, 1>& index);

    void yield_redrawLeft(const xt::xtensor<size_t, 1>& index);

    QPot::RedrawList& QPot();

protected:

    /**
    Allocate the system.

    \param N Number of particles.
    */
    void allocateSystem(size_t N);

    /**
    Initialise potential energy landscape.
    Call after System::allocateSystem.

    \param func Function to draw yield distances.
    */
    template <class F>
    void initYield(F func);

    /**
    Initialise potential energy landscape.
    Call after System::allocateSystem.

    \param func
        Function to draw yield distances.

    \param ntotal
        Number of yield-positions to keep in memory.

    \param nbuffer
        Number of yield-positions to buffer when shifting left/right.

    \param noffset
        Number of yield-positions from the end to consider for redrawing
        (allows grouping of redraws for several particles).
    */
    template <class F>
    void initYield(F func, size_t ntotal, size_t nbuffer, size_t noffset);

    /**
    Initialise the system.
    Call after System::initYield.

    \param N Number of particles.
    \param func Function to draw yield distances.
    */
    void initSystem();

    /**
    Compute #f based on the current #x and #v.
    */
    void computeForce();

    /**
    Compute #f_potential based on the current #x.

    \return ``true`` is there was a redraw.
    */
    bool computeForcePotential();

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
    QPot::RedrawList m_y; ///< Potential energy landscape.
    xt::xtensor<double, 1> m_y_l; ///< Current yielding position left.
    xt::xtensor<double, 1> m_y_r; ///< Current yielding position right.
    size_t m_N; ///< See #N.
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
