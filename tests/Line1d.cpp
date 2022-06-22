#include <FrictionQPotSpringBlock/Line1d.h>
#include <catch2/catch_all.hpp>
#include <prrng.h>
#include <xtensor/xrandom.hpp>

TEST_CASE("FrictionQPotSpringBlock::Line1d", "Line1d.h")
{

    SECTION("System::eventDrivenStep")
    {
        size_t N = 3;
        xt::xtensor<double, 2> y = xt::ones<double>({N, size_t(100)});
        y = xt::cumsum(y, 1);
        y -= 48.5;

        FrictionQPotSpringBlock::Line1d::System sys(1.0, 1.0, 1.0, 1.0, 0.1, 1.0, y);
        REQUIRE(sys.residual() < 1e-5);

        auto i_n = sys.i();
        sys.eventDrivenStep(0.2, false);
        REQUIRE(sys.residual() < 1e-5);
        REQUIRE(xt::allclose(sys.x(), (0.5 - 0.1) * xt::ones<double>({N})));
        REQUIRE(xt::all(xt::equal(sys.i(), i_n)));
        REQUIRE(sys.x_frame() == Catch::Approx((0.5 - 0.1) * (1.0 + 0.1) / 0.1));

        i_n = sys.i();
        sys.eventDrivenStep(0.2, true);
        REQUIRE(xt::allclose(sys.x(), (0.5 + 0.1) * xt::ones<double>({N})));
        REQUIRE(!xt::all(xt::equal(sys.i(), i_n)));
        REQUIRE(sys.x_frame() == Catch::Approx((0.5 + 0.1) * (1.0 + 0.1) / 0.1));

        i_n = sys.i();
        sys.eventDrivenStep(0.2, false);
        REQUIRE(xt::allclose(sys.x(), (1.5 - 0.1) * xt::ones<double>({N})));
        REQUIRE(xt::all(xt::equal(sys.i(), i_n)));
        REQUIRE(sys.x_frame() == Catch::Approx((1.5 - 0.1) * (1.0 + 0.1) / 0.1));

        i_n = sys.i();
        sys.eventDrivenStep(0.2, true);
        REQUIRE(xt::allclose(sys.x(), (1.5 + 0.1) * xt::ones<double>({N})));
        REQUIRE(!xt::all(xt::equal(sys.i(), i_n)));
        REQUIRE(sys.x_frame() == Catch::Approx((1.5 + 0.1) * (1.0 + 0.1) / 0.1));
    }

    SECTION("System::trigger")
    {
        size_t N = 3;
        xt::xtensor<double, 2> y = xt::ones<double>({N, size_t(100)});
        y = xt::cumsum(y, 1);
        y -= 48.5;

        FrictionQPotSpringBlock::Line1d::System sys(1.0, 1.0, 1.0, 1.0, 0.1, 1.0, y);

        sys.trigger(0, 0.2);

        xt::xtensor<double, 1> x = xt::zeros<double>({N});
        x(0) = 0.5 + 0.1;
        REQUIRE(xt::allclose(sys.x(), x));
    }

    SECTION("System::triggerWeakest")
    {
        size_t N = 3;
        xt::xtensor<double, 2> y = xt::ones<double>({N, size_t(100)});
        y = xt::cumsum(y, 1);
        y -= 48.5;

        FrictionQPotSpringBlock::Line1d::System sys(1.0, 1.0, 1.0, 1.0, 0.1, 1.0, y);

        xt::xtensor<double, 1> x = xt::zeros<double>({N});
        x(0) = 0.5 - 0.1;
        sys.set_x(x);

        sys.triggerWeakest(0.2);

        x(0) = 0.5 + 0.1;
        REQUIRE(xt::allclose(sys.x(), x));
    }
}
