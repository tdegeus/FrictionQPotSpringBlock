#include <FrictionQPotSpringBlock/Line1d.h>
#include <cpppath.h>
#include <fmt/core.h>
#include <highfive/H5Easy.hpp>
#include <prrng.h>

template <class S, class T>
void check(const S& source, const T& file, const std::string& path)
{
    auto ret = H5Easy::load<S>(file, path);

    if (!xt::allclose(ret, source)) {
        throw std::runtime_error("Failed to match historic data");
    }
}

int main()
{
    size_t N = 1000;

    xt::xtensor<size_t, 1> initstate = xt::arange<size_t>(N);
    xt::xtensor<size_t, 1> initseq = xt::zeros<size_t>({N});
    auto generators = prrng::auto_pcg32(initstate, initseq);

    xt::xtensor<double, 2> y = 2.0 * generators.random({20000});
    y = xt::cumsum(y, 1);
    y -= 50.0;

    double xdelta = 1e-3;

    FrictionQPotSpringBlock::Line1d::System sys(
        1.0, 2.0 * std::sqrt(3.0) / 10.0, 1.0, 1.0, 1.0 / double(N), 0.1, y);

    size_t ninc = 1000;
    xt::xtensor<double, 1> ret_x_frame = xt::empty<double>({ninc});
    xt::xtensor<double, 1> ret_f_frame = xt::empty<double>({ninc});
    xt::xtensor<long, 1> ret_S = xt::empty<long>({ninc});

    for (size_t inc = 0; inc < ninc; ++inc) {

        // Extract output data.
        auto i_n = sys.i();

        // Apply event-driven protocol.
        if (inc == 0) {
            sys.set_x_frame(0.0); // initial quench
        }
        else {
            sys.eventDrivenStep(xdelta, inc % 2 == 0); // normal event driven step
        }

        // Minimise energy.
        if (inc % 2 == 0) {
            size_t niter = sys.minimise();
            fmt::print("inc = {0:4d}, niter = {1:d}\n", inc, niter);
        }

        // Extract output data.
        ret_x_frame(inc) = sys.x_frame();
        ret_f_frame(inc) = xt::mean(sys.f_frame())();
        ret_S(inc) = xt::sum(sys.i() - i_n)();
    }

    auto path = cpppath::join({cpppath::dirname(__FILE__), "QuasiStatic.h5"});
    H5Easy::File file(path);
    check(ret_x_frame, file, "x_frame");
    check(ret_f_frame, file, "f_frame");
    check(ret_S, file, "S");

    return 0;
}
