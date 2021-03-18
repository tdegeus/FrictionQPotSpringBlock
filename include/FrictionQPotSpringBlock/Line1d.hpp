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

    ret.push_back("qpot=" +
        detail::unquote(std::string(QUOTE(QPOT_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(QPOT_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(QPOT_VERSION_PATCH))));

    ret.push_back("goosefem=" + GooseFEM::version());

    ret.push_back("xtensor=" +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_PATCH))));

    return ret;
}

template <class F>
System::System(size_t N, F func) : m_N(N)
{
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

    m_y = QPot::RedrawList(m_x, func);
    m_y_l = xt::zeros<double>({m_N});
    m_y_r = xt::zeros<double>({m_N});
}

size_t System::N() const
{
    return m_N;
}

void System::set_dt(double arg)
{
    m_dt = arg;
}

void System::set_eta(double arg)
{
    m_eta = arg;
}

void System::set_m(double arg)
{
    m_m = arg;
}

void System::set_mu(double arg)
{
    m_mu = arg;
}

void System::set_k_neighbours(double arg)
{
    m_k_neighbours = arg;
}

void System::set_k_frame(double arg)
{
    m_k_frame = arg;
}

void System::set_x_frame(double arg)
{
    m_x_frame = arg;
    this->computeForce();
}

void System::computeForcePotential()
{
    m_y.setPosition(m_x);
    xt::noalias(m_y_l) = m_y.currentYieldLeft();
    xt::noalias(m_y_r) = m_y.currentYieldRight();
    xt::noalias(m_f_potential) = m_mu * (0.5 * (m_y_r + m_y_l) - m_x);
}

void System::computeForceNeighbours()
{
    for (size_t p = 1; p < m_N - 1; ++p) {
        m_f_neighbours(p) = m_k_neighbours * (m_x(p - 1) - 2 * m_x(p) + m_x(p + 1));
    }
    m_f_neighbours(0) = m_k_neighbours * (m_x(m_N - 1) - 2 * m_x(0) + m_x(1));
    m_f_neighbours(m_N - 1) = m_k_neighbours * (m_x(m_N - 2) - 2 * m_x(m_N - 1) + m_x(0));
}

void System::computeForceFrame()
{
    xt::noalias(m_f_frame) = m_k_frame * (m_x_frame - m_x);
}

void System::computeForceDamping()
{
    xt::noalias(m_f_damping) = - m_eta * m_v;
}

void System::computeForce()
{
    this->computeForcePotential();
    this->computeForceNeighbours();
    this->computeForceFrame();
    this->computeForceDamping();
    xt::noalias(m_f) = m_f_potential + m_f_neighbours + m_f_damping + m_f_frame;
}

void System::timeStep()
{
    xt::noalias(m_v_n) = m_v;
    xt::noalias(m_a_n) = m_a;

    xt::noalias(m_x) = m_x + m_dt * m_v + 0.5 * std::pow(m_dt, 2.0) * m_a;

    xt::noalias(m_v) = m_v_n + m_dt * m_a_n;

    this->computeForce();
    xt::noalias(m_a) = m_f / m_m;

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);

    this->computeForce();
    xt::noalias(m_a) = m_f / m_m;

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);

    this->computeForce();
    xt::noalias(m_a) = m_f / m_m;

    if (xt::any(xt::isnan(m_x))) {
        throw std::runtime_error("NaN entries found");
    }
}

double System::residual() const
{
    double Fext = xt::norm_l2(m_f_frame)();
    double Fnorm = (Fext < std::numeric_limits<double>::epsilon()) ? 1 : Fext;
    return xt::norm_l2(m_f)() / Fnorm;
}

void System::quench()
{
    m_v.fill(0.0);
    m_a.fill(0.0);
}

size_t System::minimise(double tol, size_t niter_tol, size_t max_iter)
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

void System::advanceRightElastic(double eps)
{
    double dx = xt::amin(this->yieldDistanceRight())();
    if (dx < eps / 2.0) {
        return;
    }
    double deltax = dx - eps / 2.0;
    m_x += deltax;
    m_x_frame += (deltax * m_mu / m_k_frame);
    this->computeForce();
}

void System::advanceRightKick(double eps)
{
    double deltax = eps;
    m_x += deltax;
    m_x_frame += (deltax * m_mu / m_k_frame);
    this->computeForce();
}

double System::x_frame() const
{
    return m_x_frame;
}

xt::xtensor<double, 1> System::x() const
{
    return m_x;
}

xt::xtensor<double, 1> System::v() const
{
    return m_v;
}

xt::xtensor<double, 1> System::a() const
{
    return m_a;
}

xt::xtensor<double, 1> System::f() const
{
    return m_f;
}

xt::xtensor<double, 1> System::f_potential() const
{
    return m_f_potential;
}

xt::xtensor<double, 1> System::f_frame() const
{
    return m_f_frame;
}

xt::xtensor<double, 1> System::f_neighbours() const
{
    return m_f_neighbours;
}

xt::xtensor<double, 1> System::f_damping() const
{
    return m_f_damping;
}

xt::xtensor<double, 1> System::yieldLeft() const
{
    return m_y.currentYieldLeft();
}

xt::xtensor<double, 1> System::yieldRight() const
{
    return m_y.currentYieldRight();
}

xt::xtensor<int, 1> System::yieldIndex() const
{
    return m_y.currentIndex();
}

xt::xtensor<double, 1> System::yieldDistanceRight() const
{
    return m_y.currentYieldRight() - m_x;
}

xt::xtensor<double, 1> System::yieldDistanceLeft() const
{
    return m_x - m_y.currentYieldLeft();
}

} // namespace Line1d
} // namespace FrictionQPotSpringBlock

#endif
