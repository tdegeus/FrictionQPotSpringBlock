#include <FrictionQPotSpringBlock/Line1d.h>
#include <fmt/core.h>
#include <fstream>
#include <prrng.h>
#include <xtensor/xcsv.hpp>

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
    xt::xtensor<double, 2> ret = xt::empty<double>({size_t(2), ninc});

    for (size_t inc = 0; inc < ninc; ++inc) {

        // Apply event-driven protocol.
        if (inc == 0) {
            sys.set_x_frame(0.0); // initial quench
        }
        else if (inc % 2 != 0) {
            sys.eventDrivenStep(xdelta, false); // elastically advance -> mechanical equilibrium
        }
        else {
            sys.eventDrivenStep(xdelta, true); // apply kick
        }

        // Minimise energy.
        if (inc % 2 == 0) {
            size_t niter = sys.minimise();
            fmt::print("inc = {0:4d}, niter = {1:d}\n", inc, niter);
        }

        // Extract output data.
        ret(0, inc) = sys.x_frame();
        ret(1, inc) = xt::mean(sys.f_frame())();
    }

    std::ofstream outfile("Load.txt");
    xt::dump_csv(outfile, ret);
    outfile.close();

    return 0;
}
