#define CATCH_CONFIG_MAIN // tells Catch to provide a main() - only do this in one cpp file
#include <FrictionQPotSpringBlock/Line1d.h>
#include <catch2/catch.hpp>
#include <prrng.h>
#include <xtensor/xrandom.hpp>

TEST_CASE("FrictionQPotSpringBlock::Line1d", "Line1d.h")
{

    SECTION("System::eventDrivenStep")
    {
        size_t N = 3;
        xt::xtensor<long, 1> istart = xt::zeros<long>({N});
        xt::xtensor<double, 2> y = xt::ones<double>({N, size_t(100)});
        y = xt::cumsum(y, 1);
        y -= 48.5;

        FrictionQPotSpringBlock::Line1d::System sys(1.0, 1.0, 1.0, 1.0, 0.1, 1.0, y, istart);
        REQUIRE(sys.residual() < 1e-5);

        auto i_n = sys.i();
        sys.eventDrivenStep(0.2, false);
        REQUIRE(sys.residual() < 1e-5);
        REQUIRE(xt::allclose(sys.x(), (0.5 - 0.1) * xt::ones<double>({N})));
        REQUIRE(xt::all(xt::equal(sys.i(), i_n)));
        REQUIRE(sys.x_frame() == Approx((0.5 - 0.1) * (1.0 + 0.1) / 0.1));

        i_n = sys.i();
        sys.eventDrivenStep(0.2, true);
        REQUIRE(xt::allclose(sys.x(), (0.5 + 0.1) * xt::ones<double>({N})));
        REQUIRE(!xt::all(xt::equal(sys.i(), i_n)));
        REQUIRE(sys.x_frame() == Approx((0.5 + 0.1) * (1.0 + 0.1) / 0.1));

        i_n = sys.i();
        sys.eventDrivenStep(0.2, false);
        REQUIRE(xt::allclose(sys.x(), (1.5 - 0.1) * xt::ones<double>({N})));
        REQUIRE(xt::all(xt::equal(sys.i(), i_n)));
        REQUIRE(sys.x_frame() == Approx((1.5 - 0.1) * (1.0 + 0.1) / 0.1));

        i_n = sys.i();
        sys.eventDrivenStep(0.2, true);
        REQUIRE(xt::allclose(sys.x(), (1.5 + 0.1) * xt::ones<double>({N})));
        REQUIRE(!xt::all(xt::equal(sys.i(), i_n)));
        REQUIRE(sys.x_frame() == Approx((1.5 + 0.1) * (1.0 + 0.1) / 0.1));
    }

    SECTION("System::trigger")
    {
        size_t N = 3;
        xt::xtensor<long, 1> istart = xt::zeros<long>({N});
        xt::xtensor<double, 2> y = xt::ones<double>({N, size_t(100)});
        y = xt::cumsum(y, 1);
        y -= 48.5;

        FrictionQPotSpringBlock::Line1d::System sys(1.0, 1.0, 1.0, 1.0, 0.1, 1.0, y, istart);

        sys.trigger(0, 0.2);

        xt::xtensor<double, 1> x = xt::zeros<double>({N});
        x(0) = 0.5 + 0.1;
        REQUIRE(xt::allclose(sys.x(), x));
    }

    SECTION("System::triggerWeakest")
    {
        size_t N = 3;
        xt::xtensor<long, 1> istart = xt::zeros<long>({N});
        xt::xtensor<double, 2> y = xt::ones<double>({N, size_t(100)});
        y = xt::cumsum(y, 1);
        y -= 48.5;

        FrictionQPotSpringBlock::Line1d::System sys(1.0, 1.0, 1.0, 1.0, 0.1, 1.0, y, istart);

        xt::xtensor<double, 1> x = xt::zeros<double>({N});
        x(0) = 0.5 - 0.1;
        sys.set_x(x);

        sys.triggerWeakest(0.2);

        x(0) = 0.5 + 0.1;
        REQUIRE(xt::allclose(sys.x(), x));
    }

    SECTION("Chunked sequence, only move chunk right, only do so when needed")
    {
        size_t N = 3;
        auto seed = time(NULL);
        xt::xtensor<size_t, 1> initstate = seed + xt::arange<size_t>(N);

        size_t nchunk = 100; // size of chunk of yield positions kept in memory
        size_t nbuffer = 40; // buffer when shifting chunks of yield positions
        size_t ncheck = 15; // boundary region to check of chunk-shifting is needed
        size_t nmax = 20; // maximal boundary region for which chunk-shifting is applied
        double init_offset = 50.0; // initial negative position shift

        // allocate generators
        auto generators = prrng::auto_pcg32(initstate);
        auto regenerators = prrng::auto_pcg32(initstate);

        // store the current state of the generators
        auto state = generators.state();

        // draw initial chunk from the generators and convert to yield positions
        xt::xtensor<double, 2> y = 2.0 * generators.random({nchunk});
        y = xt::cumsum(y, 1);
        y -= init_offset;

        // state information of the first yield position of each particle (for restoring later)
        xt::xtensor<long, 1> istart = xt::zeros<long>({N});
        xt::xtensor<double, 1> ymin = xt::view(y, xt::all(), 0);

        // keep history to the state to facilitate restoring the right chunk for a given position
        // (the buffer can make that the current position is in the last chunk)
        auto state_n = state;
        auto istart_n = istart;
        auto ymin_n = ymin;

        // initialise system
        FrictionQPotSpringBlock::Line1d::System sys(1.0, 1.0, 1.0, 1.0, 0.1, 1.0, y, istart);
        FrictionQPotSpringBlock::Line1d::System resys(1.0, 1.0, 1.0, 1.0, 0.1, 1.0, y, istart);

        size_t n = 50;
        xt::xtensor<double, 1> x = 10.0 * xt::ones<double>({N});
        x(0) = 5.0;
        x(1) = 7.0;

        for (size_t i = 0; i < n; ++i) {

            auto xi = xt::eval((double)(i)*x);
            REQUIRE(!sys.any_redraw(xi));
            sys.set_x(xi);

            if (!sys.all_inbounds(ncheck)) {
                auto l = sys.inbounds_left(nmax);
                auto r = sys.inbounds_right(nmax);

                REQUIRE(xt::all(l));

                for (size_t p = 0; p < N; ++p) {
                    if (!r(p)) {
                        QPot::Chunked& yp = sys.refChunked(p);

                        state_n(p) = state(p);
                        istart_n(p) = istart(p);
                        ymin_n(p) = ymin(p);

                        state(p) = generators[p].state();
                        istart(p) += nchunk;

                        yp.shift_dy(
                            istart(p), xt::eval(2.0 * generators[p].random({nchunk})), nbuffer);

                        ymin(p) = yp.ymin_chunk();
                    }
                }

                REQUIRE(!sys.any_redraw());
            }

            regenerators.restore(state_n);
            xt::xtensor<double, 2> ry = 2.0 * regenerators.random({2 * nchunk});
            xt::view(ry, xt::all(), 0) = ymin_n;
            ry = xt::cumsum(ry, 1);
            resys.set_y(istart_n, ry);
            resys.set_x(sys.x());

            REQUIRE(xt::allclose(sys.i(), resys.i()));
            REQUIRE(xt::allclose(sys.yieldDistanceLeft(), resys.yieldDistanceLeft()));
            REQUIRE(xt::allclose(sys.yieldDistanceRight(), resys.yieldDistanceRight()));
            REQUIRE(xt::allclose(sys.yleft(), resys.yleft()));
            REQUIRE(xt::allclose(sys.yright(), resys.yright()));
        }
    }

    SECTION("Chunked sequence, shift optimally left")
    {
        size_t N = 3;
        auto seed = time(NULL);
        xt::xtensor<long, 1> istart = xt::zeros<long>({N});
        xt::xtensor<size_t, 1> initstate = seed + xt::arange<size_t>(N);

        size_t nchunk = 100; // size of chunk of yield positions kept in memory
        size_t nbuffer = 10; // buffer to keep left
        double init_offset = 5.0; // initial negative position shift

        // allocate generators
        auto generators = prrng::auto_pcg32(initstate);
        auto regenerators = prrng::auto_pcg32(initstate);
        auto state = generators.state();
        xt::xtensor<long, 1> istate = xt::zeros<long>({N});

        // draw initial chunk from the generators and convert to yield positions
        xt::xtensor<double, 2> y = 2.0 * generators.random({nchunk});
        y = xt::cumsum(y, 1);
        y -= init_offset;

        // initialise system
        FrictionQPotSpringBlock::Line1d::System sys(1.0, 1.0, 1.0, 1.0, 0.1, 1.0, y, istart);
        FrictionQPotSpringBlock::Line1d::System resys(1.0, 1.0, 1.0, 1.0, 0.1, 1.0, y, istart);

        // sequence of particle positions
        size_t n = 50;
        xt::xtensor<double, 1> x = 10.0 * xt::ones<double>({N});
        x(0) = 5.0;
        x(1) = 7.0;

        for (size_t i = 0; i < n; ++i) {

            auto xi = xt::eval((double)(i)*x);
            sys.set_x(xi);
            auto i0 = sys.i();

            if (xt::any(sys.i_chunk() > nbuffer)) {
                for (size_t p = 0; p < N; ++p) {
                    QPot::Chunked& yp = sys.refChunked(p);
                    auto nb = yp.size() - yp.i_chunk() + nbuffer;
                    if (nb >= nchunk) {
                        continue;
                    }
                    state(p) = generators[p].state();
                    istate(p) = yp.istop();
                    yp.shift_dy(
                        yp.istop(), xt::eval(2.0 * generators[p].random({nchunk - nb})), nb);
                }
            }

            REQUIRE(xt::all(xt::equal(sys.i(), i0)));
            REQUIRE(xt::all(sys.i() - sys.istart() <= nbuffer));
            REQUIRE(xt::all(xt::equal(sys.i_chunk(), sys.i() - sys.istart())));

            // restore state: start with the latests draw that is quite close and reverse back
            // in the sequence until the start of the current chunk held in memory
            regenerators.restore(state);
            regenerators.advance(xt::eval(sys.istart() - istate));

            // generate the yield distances, convert to yield positions using the first yield
            // position of the current chunk as memory (was also the state from which the random
            // numbers were generated)
            xt::xtensor<double, 2> ry = 2.0 * regenerators.random({nchunk});
            xt::view(ry, xt::all(), 0) = sys.ymin();
            ry = xt::cumsum(ry, 1);

            resys.set_y(sys.istart(), ry);
            resys.set_x(sys.x());

            REQUIRE(xt::allclose(sys.i(), resys.i()));
            REQUIRE(xt::allclose(sys.yieldDistanceLeft(), resys.yieldDistanceLeft()));
            REQUIRE(xt::allclose(sys.yieldDistanceRight(), resys.yieldDistanceRight()));
            REQUIRE(xt::allclose(sys.yleft(), resys.yleft()));
            REQUIRE(xt::allclose(sys.yright(), resys.yright()));
        }
    }
}
