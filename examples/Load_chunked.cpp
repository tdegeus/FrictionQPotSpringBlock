
#include <FrictionQPotSpringBlock/Line1d.h>
#include <prrng.h>
#include <fmt/core.h>
#include <xtensor/xcsv.hpp>
#include <fstream>

int main()
{
    size_t N = 1000;

    xt::xtensor<size_t, 1> initstate = xt::arange<size_t>(N);
    xt::xtensor<size_t, 1> initseq = xt::zeros<size_t>({N});
    auto generators = prrng::auto_pcg32(initstate, initseq);
    xt::xtensor<long, 1> istart = xt::zeros<long>({N});

    size_t nchunk = 2000;
    size_t nbuffer = 300;
    size_t ncheck = 200;
    size_t nmax = 200;

    xt::xtensor<double, 2> y = 2.0 * generators.random({nchunk});
    y = xt::cumsum(y, 1);
    y -= 50.0;

    double xdelta = 1e-3;

    FrictionQPotSpringBlock::Line1d::System sys(N, y);

    sys.set_dt(0.1);
    sys.set_eta(2.0 * std::sqrt(3.0) / 10.0);
    sys.set_m(1.0);
    sys.set_mu(1.0);
    sys.set_k_neighbours(1.0);
    sys.set_k_frame(1.0 / double(N));

    size_t ninc = 1000;
    xt::xtensor<double, 2> ret = xt::empty<double>({size_t(2), ninc});

    for (size_t inc = 0; inc < ninc; ++inc) {

        // Apply event-driven protocol.
        if (inc == 0) {
            sys.set_x_frame(0.0); // initial quench
        }
        else if (inc % 2 != 0) {
            sys.advanceRightElastic(xdelta); // elastically advance -> mechanical equilibrium
        }
        else {
            sys.advanceRightKick(xdelta); // apply kick
        }

        // Minimise energy.
        if (inc % 2 == 0) {
            size_t niter = sys.minimise();
            fmt::print("inc = {0:4d}, niter = {1:d}\n", inc, niter);
        }

        // change the chunk of yield positions held in memory
        // reminder: a sequence of yield positions is taken from the random generator:
        // each particle has one generator, with its own seed, and one sequence
        // the assumptions are made that:
        // - no shift is needed during minimisation
        // - one shift suffices here
        if (sys.any_shift(ncheck)) {
            auto r = sys.boundcheck_right(nmax);
            for (size_t p = 0; p < N; ++p) {
                if (!r(p)) {
                    QPot::Chunked& yp = sys.y(p);

                    istart(p) += nchunk;

                    yp.shift_dy(
                        istart(p),
                        xt::eval(2.0 * generators[p].random({nchunk})),
                        nbuffer);
                }
            }
        }

        // Extract output data.
        ret(0, inc) = sys.x_frame();
        ret(1, inc) = xt::mean(sys.f_frame())();
    }

    std::ofstream outfile("Load_chunked.txt");
    xt::dump_csv(outfile, ret);
    outfile.close();

    return 0;
}
