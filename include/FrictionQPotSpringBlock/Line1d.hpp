/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotSpringBlock

*/

#ifndef FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_HPP
#define FRICTIONQPOTSPRINGBLOCK_UNIFORMSINGLELAYER2D_HPP

#include "Line1d.h"

namespace FrictionQPotSpringBlock {
namespace Line1d {

inline std::vector<std::string> version_dependencies()
{
    std::vector<std::string> ret;

    ret.push_back("frictionqpotspringblock=" + version());

    ret.push_back("qpot=" + QPot::version());

    ret.push_back("goosefem=" + GooseFEM::version());

    ret.push_back("xtensor=" +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_PATCH))));

    return ret;
}

template <class F>
inline System::System(size_t N, F func)
{
    this->allocateSystem(N);
    this->initYield(func);
    this->initSystem();
}

template <class F>
inline System::System(size_t N, F func, size_t ntotal, size_t nbuffer, size_t noffset)
{
    this->allocateSystem(N);
    this->initYield(func, ntotal, nbuffer, noffset);
    this->initSystem();
}

inline void System::allocateSystem(size_t N)
{
    m_N = N;
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
}

template <class F>
inline void System::initYield(F func)
{
    m_y = QPot::RedrawList(m_x, func);
    m_y_l = m_y.currentYieldLeft();
    m_y_r = m_y.currentYieldRight();
}

template <class F>
inline void System::initYield(F func, size_t ntotal, size_t nbuffer, size_t noffset)
{
    m_y = QPot::RedrawList(m_x, func, ntotal, nbuffer, noffset);
    m_y_l = m_y.currentYieldLeft();
    m_y_r = m_y.currentYieldRight();
}

inline void System::initSystem()
{
    this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();
    this->computeForceDamping();
    this->computeForce();
}

inline size_t System::N() const
{
    return m_N;
}

inline void System::set_dt(double arg)
{
    m_dt = arg;
}

inline void System::set_eta(double arg)
{
    m_eta = arg;
}

inline void System::set_m(double arg)
{
    m_m = arg;
}

inline void System::set_mu(double arg)
{
    m_mu = arg;
}

inline void System::set_k_neighbours(double arg)
{
    m_k_neighbours = arg;
}

inline void System::set_k_frame(double arg)
{
    m_k_frame = arg;
}

inline void System::set_x_frame(double arg)
{
    m_x_frame = arg;
    this->computeForceFrame();
    this->computeForce();
}

inline bool System::set_x(const xt::xtensor<double, 1>& arg)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(arg.size() == m_N);
    xt::noalias(m_x) = arg;
    bool redraw = this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();
    this->computeForce();
    return redraw;
}

inline void System::set_v(const xt::xtensor<double, 1>& arg)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(arg.size() == m_N);
    xt::noalias(m_v) = arg;
    this->computeForceDamping();
    this->computeForce();
}

inline void System::set_a(const xt::xtensor<double, 1>& arg)
{
    FRICTIONQPOTSPRINGBLOCK_ASSERT(arg.size() == m_N);
    xt::noalias(m_a) = arg;
}

inline bool System::computeForcePotential()
{
    bool redraw = m_y.setPosition(m_x);
    xt::noalias(m_y_l) = m_y.currentYieldLeft();
    xt::noalias(m_y_r) = m_y.currentYieldRight();
    xt::noalias(m_f_potential) = m_mu * (0.5 * (m_y_r + m_y_l) - m_x);
    return redraw;
}

inline void System::computeForceNeighbours()
{
    for (size_t p = 1; p < m_N - 1; ++p) {
        m_f_neighbours(p) = m_k_neighbours * (m_x(p - 1) - 2 * m_x(p) + m_x(p + 1));
    }
    m_f_neighbours(0) = m_k_neighbours * (m_x(m_N - 1) - 2 * m_x(0) + m_x(1));
    m_f_neighbours(m_N - 1) = m_k_neighbours * (m_x(m_N - 2) - 2 * m_x(m_N - 1) + m_x(0));
}

inline void System::computeForceFrame()
{
    xt::noalias(m_f_frame) = m_k_frame * (m_x_frame - m_x);
}

inline void System::computeForceDamping()
{
    xt::noalias(m_f_damping) = - m_eta * m_v;
}

inline void System::computeForce()
{
    xt::noalias(m_f) = m_f_potential + m_f_neighbours + m_f_damping + m_f_frame;
}

inline bool System::timeStep()
{
    xt::noalias(m_v_n) = m_v;
    xt::noalias(m_a_n) = m_a;

    // positions
    xt::noalias(m_x) = m_x + m_dt * m_v + 0.5 * std::pow(m_dt, 2.0) * m_a;
    bool redraw = this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();

    // velocities
    xt::noalias(m_v) = m_v_n + m_dt * m_a_n;
    this->computeForceDamping();

    // accelerations
    this->computeForce();
    xt::noalias(m_a) = m_f / m_m;

    // velocities
    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->computeForceDamping();

    // accelerations
    this->computeForce();
    xt::noalias(m_a) = m_f / m_m;

    // velocities
    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->computeForceDamping();

    // accelerations
    this->computeForce();
    xt::noalias(m_a) = m_f / m_m;

    if (xt::any(xt::isnan(m_x))) {
        throw std::runtime_error("NaN entries found");
    }

    return redraw;
}

inline double System::residual() const
{
    double Fext = xt::norm_l2(m_f_frame)();
    double Fnorm = (Fext < std::numeric_limits<double>::epsilon()) ? 1 : Fext;
    return xt::norm_l2(m_f)() / Fnorm;
}

inline void System::quench()
{
    m_v.fill(0.0);
    m_a.fill(0.0);
}

inline size_t System::minimise(double tol, size_t niter_tol, size_t max_iter)
{
    GooseFEM::Iterate::StopList stop(niter_tol);

    for (size_t iiter = 0; iiter < max_iter; ++iiter) {

        this->timeStep();

        if (stop.stop(this->residual(), tol)) {
            this->quench();
            return iiter;
        }
    }

    throw std::runtime_error("No convergence found");
}

inline bool System::advanceRightElastic(double eps)
{
    double dx = xt::amin(this->yieldDistanceRight())();
    if (dx < eps / 2.0) {
        return false;
    }
    double deltax = dx - eps / 2.0;
    m_x += deltax;
    m_x_frame += (deltax * m_mu / m_k_frame);
    bool redraw = this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();
    this->computeForce();
    return redraw;
}

inline bool System::advanceRightKick(double eps)
{
    double deltax = eps;
    m_x += deltax;
    m_x_frame += (deltax * m_mu / m_k_frame);
    bool redraw = this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();
    this->computeForce();
    return redraw;
}

inline double System::x_frame() const
{
    return m_x_frame;
}

inline xt::xtensor<double, 1> System::x() const
{
    return m_x;
}

inline xt::xtensor<double, 1> System::v() const
{
    return m_v;
}

inline xt::xtensor<double, 1> System::a() const
{
    return m_a;
}

inline xt::xtensor<double, 1> System::f() const
{
    return m_f;
}

inline xt::xtensor<double, 1> System::f_potential() const
{
    return m_f_potential;
}

inline xt::xtensor<double, 1> System::f_frame() const
{
    return m_f_frame;
}

inline xt::xtensor<double, 1> System::f_neighbours() const
{
    return m_f_neighbours;
}

inline xt::xtensor<double, 1> System::f_damping() const
{
    return m_f_damping;
}

inline xt::xtensor<double, 1> System::yieldLeft() const
{
    return m_y.currentYieldLeft();
}

inline xt::xtensor<double, 1> System::yieldRight() const
{
    return m_y.currentYieldRight();
}

inline xt::xtensor<int, 1> System::yieldIndex() const
{
    return m_y.currentIndex();
}

inline xt::xtensor<double, 1> System::yieldDistanceRight() const
{
    return m_y.currentYieldRight() - m_x;
}

inline xt::xtensor<double, 1> System::yieldDistanceLeft() const
{
    return m_x - m_y.currentYieldLeft();
}

inline QPot::RedrawList& System::getRedrawList()
{
    return m_y;
}

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#endif
