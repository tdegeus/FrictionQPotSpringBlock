
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotSpringBlock/Line1d.h>
#include <QPot/random.hpp>

TEST_CASE("FrictionQPotSpringBlock::Line1d", "Line1d.h")
{

SECTION("System::advanceRightElastic")
{
    size_t N = 3;
    auto uniform = QPot::random::UniformList();

    FrictionQPotSpringBlock::Line1d::System sys(N, uniform);

    sys.advanceRightElastic(0.2);

    xt::xtensor<double, 1> x = (0.5 - 0.1) * xt::ones<double>({N});

    REQUIRE(xt::allclose(x, sys.x()));
}

SECTION("Reconstruct sequence")
{
    size_t N = 3;
    auto seed = time(NULL);
    auto random = QPot::random::RandList();

    QPot::random::seed(seed);
    FrictionQPotSpringBlock::Line1d::System sys(N, random);

    size_t n = 20;
    xt::xtensor<double, 1> x = 100.0 * xt::ones<double>({N});
    std::vector<xt::xtensor<int, 1>> redraw;

    for (size_t i = 0; i < n; ++i) {
        bool r = sys.set_x(xt::eval((double)(i) * x));
        if (r) {
            redraw.push_back(sys.QPot().currentRedraw());
        }
    }

    QPot::random::seed(seed);
    FrictionQPotSpringBlock::Line1d::System other(N, random);

    for (auto& i : redraw) {
        other.QPot().redraw(i);
    }

    other.set_x(sys.x());

    REQUIRE(xt::allclose(sys.yieldLeft(), other.yieldLeft()));
}

}
