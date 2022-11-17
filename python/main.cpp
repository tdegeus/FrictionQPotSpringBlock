/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/xtensor_python_config.hpp> // todo: remove for xtensor-python >0.26.1

#define FRICTIONQPOTSPRINGBLOCK_USE_XTENSOR_PYTHON
#include <FrictionQPotSpringBlock/Line1d.h>
#include <prrng.h>

namespace py = pybind11;

/**
 * Overrides the `__name__` of a module.
 * Classes defined by pybind11 use the `__name__` of the module as of the time they are defined,
 * which affects the `__repr__` of the class type objects.
 */
class ScopedModuleNameOverride {
public:
    explicit ScopedModuleNameOverride(py::module m, std::string name) : module_(std::move(m))
    {
        original_name_ = module_.attr("__name__");
        module_.attr("__name__") = name;
    }
    ~ScopedModuleNameOverride()
    {
        module_.attr("__name__") = original_name_;
    }

private:
    py::module module_;
    py::object original_name_;
};

template <class C, class S, class G>
void mysystem(C& cls)
{
    cls.def_property_readonly("N", &S::N, "Number of particles");
    cls.def("i", &S::i, "Index: y[:, i] < x <= y[:, i + 1]");

    cls.def(
        "y_right",
        &S::y_right,
        "Convenience function: same as `system.chunked.chunk[arange(system.N), system.i + 1]`.");

    cls.def(
        "y_left",
        &S::y_left,
        "Convenience function: same as `system.chunked.chunk[arange(system.N), system.i]`.");

    cls.def_property(
        "x",
        &S::x,
        &S::template set_x<xt::pytensor<double, 1>>,
        "Particle positions (updating updates all relevant variables)");

    cls.def_property(
        "v",
        &S::v,
        &S::template set_v<xt::pytensor<double, 1>>,
        "Particle velocities (updating updates all relevant variables)");

    cls.def_property(
        "a",
        &S::a,
        &S::template set_a<xt::pytensor<double, 1>>,
        "Particle accelerations (updating updates all relevant variables)");

    cls.def_property("inc", &S::inc, &S::set_inc, "Increment");
    cls.def_property("t", &S::t, &S::set_t, "Time");
    cls.def_property("x_frame", &S::x_frame, &S::set_x_frame, "Frame pos.");
    cls.def_property_readonly("f", &S::f, "Residual forces");
    cls.def_property_readonly("f_potential", &S::f_potential, "Elastic forces");
    cls.def_property_readonly("f_frame", &S::f_frame, "Frame forces");
    cls.def_property_readonly("f_neighbours", &S::f_neighbours, "Interaction forces");
    cls.def_property_readonly("f_damping", &S::f_damping, "Particle damping forces");
    cls.def("temperature", &S::temperature, "Temperature");
    cls.def("residual", &S::residual, "Residual");

    cls.def("refresh", &S::refresh, "refresh");
    cls.def("quench", &S::quench, "quench");
    cls.def("timeStep", &S::timeStep, "timeStep");

    cls.def("timeSteps", &S::timeSteps, "timeSteps", py::arg("n"));

    cls.def(
        "timeStepsUntilEvent",
        &S::timeStepsUntilEvent,
        "timeStepsUntilEvent",
        py::arg("tol") = 1e-5,
        py::arg("niter_tol") = 10,
        py::arg("max_iter") = size_t(1e9));

    cls.def("flowSteps", &S::flowSteps, "flowSteps", py::arg("n"), py::arg("v_frame"));

    cls.def(
        "minimise",
        &S::minimise,
        "minimise",
        py::arg("tol") = 1e-5,
        py::arg("niter_tol") = 10,
        py::arg("max_iter") = size_t(1e9),
        py::arg("time_activity") = false,
        py::arg("max_iter_is_error") = true);

    cls.def("quasistaticActivityFirst", &S::quasistaticActivityFirst, "quasistaticActivityFirst");

    cls.def("quasistaticActivityLast", &S::quasistaticActivityLast, "quasistaticActivityLast");

    cls.def(
        "minimise_nopassing",
        &S::minimise_nopassing,
        "minimise_nopassing",
        py::arg("tol") = 1e-5,
        py::arg("niter_tol") = 10,
        py::arg("max_iter") = size_t(1e9));

    cls.def(
        "eventDrivenStep",
        &S::eventDrivenStep,
        "eventDrivenStep",
        py::arg("eps"),
        py::arg("kick"),
        py::arg("direction") = 1);

    cls.def(
        "trigger", &S::trigger, "trigger", py::arg("p"), py::arg("eps"), py::arg("direction") = 1);

    cls.def(
        "advanceToFixedForce", &S::advanceToFixedForce, "advanceToFixedForce", py::arg("f_frame"));
};

PYBIND11_MODULE(_FrictionQPotSpringBlock, m)
{
    // Ensure members to display as `FrictionQPotSpringBlock.X`
    // (not `FrictionQPotSpringBlock._FrictionQPotSpringBlock.X`)
    ScopedModuleNameOverride name_override(m, "FrictionQPotSpringBlock");

    xt::import_numpy();

    m.doc() = "Spring-block friction model with local disordered potential energy landscape";

    namespace M = FrictionQPotSpringBlock;

    m.def("version", &M::version, "Return version string.");

    // ------------------------------
    // FrictionQPotSpringBlock.Line1d
    // ------------------------------

    {

        py::module sm = m.def_submodule("Line1d", "Line1d");

        namespace SM = FrictionQPotSpringBlock::Line1d;

        sm.def(
            "version_dependencies",
            &SM::version_dependencies,
            "Return version information of library and its dependencies.");

        sm.def("version_compiler", &SM::version_compiler, "Return compiler information.");

        py::class_<SM::YieldSequence, SM::Generator>(sm, "YieldSequence")

            .def(
                py::init<const xt::pytensor<double, 2>&, const std::vector<size_t>&>(),
                "Sequence of yield positions",
                py::arg("data"),
                py::arg("align") = prrng::alignment())

            .def("__repr__", [](const SM::YieldSequence&) {
                return "<FrictionQPotSpringBlock.Line1d.YieldSequence>";
            });

        {
            using S = SM::System;

            py::class_<S> cls(sm, "System");

            cls.def(
                py::init<double, double, double, double, double, double, SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_neighbours"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunked"));

            mysystem<py::class_<S>, S, SM::Generator>(cls);

            cls.def("__repr__", [](const S&) { return "<FrictionQPotSpringBlock.Line1d.System>"; });
        }

        {
            using S = SM::SystemThermalRandomForcing;

            py::class_<S> cls(sm, "SystemThermalRandomForcing");

            cls.def(
                py::init<double, double, double, double, double, double, SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_neighbours"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunked"));

            mysystem<py::class_<S>, S, SM::Generator>(cls);

            cls.def(
                "setRandomForce",
                &SM::SystemThermalRandomForcing::setRandomForce<xt::pytensor<double, 1>>,
                "setRandomForce",
                py::arg("f"));

            cls.def(
                "setRandomForceSequence",
                &SM::SystemThermalRandomForcing::
                    setRandomForceSequence<xt::pytensor<double, 2>, xt::pytensor<size_t, 2>>,
                "setRandomForceSequence",
                py::arg("f"),
                py::arg("start_inc"));

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line1d.SystemThermalRandomForcing>";
            });
        }

        {
            using S = SM::SystemSemiSmooth;

            py::class_<S> cls(sm, "SystemSemiSmooth");

            cls.def(
                py::init<
                    double,
                    double,
                    double,
                    double,
                    double,
                    double,
                    double,
                    SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("kappa"),
                py::arg("k_neighbours"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunked"));

            mysystem<py::class_<S>, S, SM::Generator>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line1d.SystemSemiSmooth>";
            });
        }

        {
            using S = SM::SystemSmooth;

            py::class_<S> cls(sm, "SystemSmooth");

            cls.def(
                py::init<double, double, double, double, double, double, SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_neighbours"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunked"));

            mysystem<py::class_<S>, S, SM::Generator>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line1d.SystemSmooth>";
            });
        }
    }
}
