/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotSpringBlock

*/

#ifndef FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_H
#define FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_H

#include "config.h"

#include <QPot/Redraw.hpp>
#include <GooseFEM/Iterate.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xshape.hpp>
#include <fmt/core.h>

namespace FrictionQPotSpringBlock {
namespace Line1d {

// -------------------------------------
// Return versions of returned libraries
// -------------------------------------

inline std::vector<std::string> versionInfo();

// ------
// System
// ------

class System {

public:

    System() = default;

    // Constructor.
    template <class F>
    System(size_t N, F function_to_draw_distances);

    // Change parameters.
    void set_dt(double dt);
    void set_eta(double eta);
    void set_m(double m);
    void set_mu(double mu);
    void set_k_neighbours(double k_neighbours);
    void set_k_frame(double k_frame);
    void set_x_frame(double x_frame);

    // Return state variables.
    double get_x_frame() const; // position of loading frame
    xt::xtensor<double,1> get_x() const; // positions
    xt::xtensor<double,1> get_v() const; // velocities
    xt::xtensor<double,1> get_f() const; // total force
    xt::xtensor<double,1> get_f_potential() const; // force associated to potentials
    xt::xtensor<double,1> get_f_frame() const; // force associated to the loading frame
    xt::xtensor<double,1> get_f_neighbours() const; // force associated to neighbours

    // Effectuate time step. Updates x, v, a, f.
    void timeStep();

    // Minimise energy: effectuate time steps until a mechanical equilibrium is reached.
    // Returns the number of iterations.
    size_t minimise();

    // Event driven: advance right to closest yielding point, leaving "delta_x / 2" as margin.
    void advanceRightElastic(double delta_x);

    // Event driven: advance right by "delta_x".
    void advanceRightKick(double delta_x);

    // Obtain information on the potential energy landscape.
    xt::xtensor<int,1> getYieldIndex() const;
    xt::xtensor<double,1> getYieldDistanceRight() const;
    xt::xtensor<double,1> getYieldDistanceLeft() const;

protected:

    // Compute resultant force.
    void computeForce();

    // Compute individual force components.
    void computeForcePotential();
    void computeForceNeighbours();
    void computeForceFrame();
    void computeForceDamping();

protected:

    // State variables.
    xt::xtensor<double,1> m_f; // resultant force
    xt::xtensor<double,1> m_f_potential; // force related to local potentials
    xt::xtensor<double,1> m_f_neighbours; // force related to interaction with neighbours
    xt::xtensor<double,1> m_f_frame; // force related to loading frame
    xt::xtensor<double,1> m_f_damping; // force related to damping
    xt::xtensor<double,1> m_x; // position
    xt::xtensor<double,1> m_v; // velocity
    xt::xtensor<double,1> m_v_n; // velocity (last time-step)
    xt::xtensor<double,1> m_a; // acceleration
    xt::xtensor<double,1> m_a_n; // acceleration (last time-step)
    QPot::RedrawList m_y; // potential energy landscape
    xt::xtensor<double,1> m_y_l; // current yielding position, left
    xt::xtensor<double,1> m_y_r; // current yielding position, right

    // Basic parameters.
    size_t m_N; // number of points
    double m_dt = 0.1; // time-step
    double m_eta = 2.0 * std::sqrt(3.0) / 10.0; // damping coefficient (uniform)
    double m_m = 1.0; // mass (uniform)
    double m_mu = 1.0; // elastic stiffness
    double m_k_neighbours = 1.0; // coupling to neighbours
    double m_k_frame = 1.0; // coupling the loading frame (to be set ~1/N)

    // Loading frame.
    double m_x_frame; // position of the loading frame (to be set)

};

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#include "Line1d.hpp"

#endif
