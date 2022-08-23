/**
Python API

\file main.cpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/xtensor_python_config.hpp> // todo: remove for xtensor-python >0.26.1

#define FRICTIONQPOTSPRINGBLOCK_USE_XTENSOR_PYTHON
#include <FrictionQPotSpringBlock/Line1d.h>

namespace py = pybind11;

/**
Overrides the `__name__` of a module.
Classes defined by pybind11 use the `__name__` of the module as of the time they are defined,
which affects the `__repr__` of the class type objects.
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

        py::class_<SM::System>(sm, "System")

            .def(
                py::init<
                    double,
                    double,
                    double,
                    double,
                    double,
                    double,
                    const xt::pytensor<double, 2>&>(),
                "System",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_neighbours"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("x_yield"))

            .def_property_readonly("N", &SM::System::N, "Number of particles")
            .def_property_readonly("i", &SM::System::i, "Index: y[:, i] < x <= y[:, i + 1]")

            .def(
                "y_right",
                &SM::System::y_right,
                "Convenience function: same as `system.y[arange(system.N), system.i + 1]`.")

            .def(
                "y_left",
                &SM::System::y_left,
                "Convenience function: same as `system.y[arange(system.N), system.i]`.")

            .def_property(
                "y",
                &SM::System::y,
                &SM::System::template set_y<xt::pytensor<double, 2>>,
                "Yield positions (updating updates all relevant variables)")

            .def_property(
                "x",
                &SM::System::x,
                &SM::System::template set_x<xt::pytensor<double, 1>>,
                "Particle positions (updating updates all relevant variables)")

            .def_property(
                "v",
                &SM::System::v,
                &SM::System::template set_v<xt::pytensor<double, 1>>,
                "Particle velocities (updating updates all relevant variables)")

            .def_property(
                "a",
                &SM::System::a,
                &SM::System::template set_a<xt::pytensor<double, 1>>,
                "Particle accelerations (updating updates all relevant variables)")

            .def_property("inc", &SM::System::inc, &SM::System::set_inc, "Increment")
            .def_property("t", &SM::System::t, &SM::System::set_t, "Time")
            .def_property("x_frame", &SM::System::x_frame, &SM::System::set_x_frame, "Frame pos.")
            .def_property_readonly("f", &SM::System::f, "Residual forces")
            .def_property_readonly("f_potential", &SM::System::f_potential, "Elastic forces")
            .def_property_readonly("f_frame", &SM::System::f_frame, "Frame forces")
            .def_property_readonly("f_neighbours", &SM::System::f_neighbours, "Interaction forces")
            .def_property_readonly("f_damping", &SM::System::f_damping, "Particle damping forces")
            .def("temperature", &SM::System::temperature, "Temperature")
            .def("residual", &SM::System::residual, "Residual")

            .def("refresh", &SM::System::refresh, "refresh")
            .def("quench", &SM::System::quench, "quench")
            .def("timeStep", &SM::System::timeStep, "timeStep")

            .def(
                "timeSteps",
                &SM::System::timeSteps,
                "timeSteps",
                py::arg("n"),
                py::arg("nmargin") = 1)

            .def(
                "timeStepsUntilEvent",
                &SM::System::timeStepsUntilEvent,
                "timeStepsUntilEvent",
                py::arg("nmargin") = 1,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 10,
                py::arg("max_iter") = size_t(1e9))

            .def(
                "flowSteps",
                &SM::System::flowSteps,
                "flowSteps",
                py::arg("n"),
                py::arg("v_frame"),
                py::arg("nmargin") = 1)

            .def(
                "minimise",
                &SM::System::minimise,
                "minimise",
                py::arg("nmargin") = 1,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 10,
                py::arg("max_iter") = size_t(1e9),
                py::arg("time_activity") = false,
                py::arg("max_iter_is_error") = true)

            .def(
                "quasistaticActivityFirst",
                &SM::System::quasistaticActivityFirst,
                "quasistaticActivityFirst")

            .def(
                "quasistaticActivityLast",
                &SM::System::quasistaticActivityLast,
                "quasistaticActivityLast")

            .def(
                "minimise_nopassing",
                &SM::System::minimise_nopassing,
                "minimise_nopassing",
                py::arg("nmargin") = 1,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 10,
                py::arg("max_iter") = size_t(1e9))

            .def(
                "eventDrivenStep",
                &SM::System::eventDrivenStep,
                "eventDrivenStep",
                py::arg("eps"),
                py::arg("kick"),
                py::arg("direction") = 1)

            .def(
                "trigger",
                &SM::System::trigger,
                "trigger",
                py::arg("p"),
                py::arg("eps"),
                py::arg("direction") = 1)

            .def(
                "advanceToFixedForce",
                &SM::System::advanceToFixedForce,
                "advanceToFixedForce",
                py::arg("f_frame"))

            .def("__repr__", [](const SM::System&) {
                return "<FrictionQPotSpringBlock.Line1d.System>";
            });

        {
            py::class_<SM::SystemThermalRandomForcing, SM::System> cls(
                sm, "SystemThermalRandomForcing");

            cls.def(
                py::init<
                    double,
                    double,
                    double,
                    double,
                    double,
                    double,
                    const xt::pytensor<double, 2>&>(),
                "SystemThermalRandomForcing",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_neighbours"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("x_yield"));

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

            cls.def("__repr__", [](const SM::SystemThermalRandomForcing&) {
                return "<FrictionQPotSpringBlock.Line1d.SystemThermalRandomForcing>";
            });
        }
    }
}
