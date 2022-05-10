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

#define FRICTIONQPOTSPRINGBLOCK_NOALIAS
#include <FrictionQPotSpringBlock/Line1d.h>
#include <FrictionQPotSpringBlock/version.h>

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

        py::class_<SM::System>(sm, "System")

            .def(
                py::init<
                    double,
                    double,
                    double,
                    double,
                    double,
                    double,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<long, 1>&>(),
                "System",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_neighbours"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("x_yield"),
                py::arg("istart"))

            .def("N", &SM::System::N, "N")
            .def("y", &SM::System::y, "y")

            .def(
                "refChunked",
                &SM::System::refChunked,
                "refChunked",
                py::return_value_policy::reference_internal)

            .def(
                "set_y",
                py::overload_cast<const xt::pytensor<long, 1>&, const xt::pytensor<double, 2>&>(
                    &SM::System::set_y<xt::pytensor<long, 1>, xt::pytensor<double, 2>>),
                "set_y",
                py::arg("istart"),
                py::arg("y"))

            .def(
                "shift_y",
                py::overload_cast<
                    const xt::pytensor<long, 1>&,
                    const xt::pytensor<double, 2>&,
                    size_t>(&SM::System::shift_y<xt::pytensor<long, 1>, xt::pytensor<double, 2>>),
                "shift_y",
                py::arg("istart"),
                py::arg("y"),
                py::arg("nbuffer") = 0)

            .def(
                "shift_dy",
                py::overload_cast<
                    const xt::pytensor<long, 1>&,
                    const xt::pytensor<double, 2>&,
                    size_t>(&SM::System::shift_dy<xt::pytensor<long, 1>, xt::pytensor<double, 2>>),
                "shift_dy",
                py::arg("istart"),
                py::arg("dy"),
                py::arg("nbuffer") = 0)

            .def(
                "set_y",
                py::overload_cast<size_t, long, const xt::pytensor<double, 1>&>(
                    &SM::System::set_y<xt::pytensor<double, 1>>),
                "set_y",
                py::arg("p"),
                py::arg("istart"),
                py::arg("y"))

            .def(
                "shift_y",
                py::overload_cast<size_t, long, const xt::pytensor<double, 1>&, size_t>(
                    &SM::System::shift_y<xt::pytensor<double, 1>>),
                "shift_y",
                py::arg("p"),
                py::arg("istart"),
                py::arg("y"),
                py::arg("nbuffer") = 0)

            .def(
                "shift_dy",
                py::overload_cast<size_t, long, const xt::pytensor<double, 1>&, size_t>(
                    &SM::System::shift_dy<xt::pytensor<double, 1>>),
                "shift_dy",
                py::arg("p"),
                py::arg("istart"),
                py::arg("dy"),
                py::arg("nbuffer") = 0)

            .def("ymin", &SM::System::ymin, "ymin")
            .def("ymax", &SM::System::ymax, "ymax")
            .def("ymin_chunk", &SM::System::ymin_chunk, "ymin_chunk")
            .def("yleft", &SM::System::yleft, "yleft")
            .def("yright", &SM::System::yright, "yright")
            .def("i_chunk", &SM::System::i_chunk, "i_chunk")
            .def("istart", &SM::System::istart, "istart")
            .def("istop", &SM::System::istop, "istop")

            .def(
                "inbounds_left",
                &SM::System::inbounds_left,
                "inbounds_left",
                py::arg("nmargin") = 0)

            .def(
                "inbounds_right",
                &SM::System::inbounds_right,
                "inbounds_right",
                py::arg("nmargin") = 0)

            .def(
                "all_inbounds_left",
                &SM::System::all_inbounds_left,
                "all_inbounds_left",
                py::arg("nmargin") = 0)

            .def(
                "all_inbounds_right",
                &SM::System::all_inbounds_right,
                "all_inbounds_right",
                py::arg("nmargin") = 0)

            .def("all_inbounds", &SM::System::all_inbounds, "all_inbounds", py::arg("nmargin") = 0)

            .def(
                "any_redraw",
                static_cast<bool (SM::System::*)(const xt::pytensor<double, 1>&) const>(
                    &SM::System::any_redraw),
                "any_redraw",
                py::arg("x"))

            .def(
                "any_redraw",
                static_cast<bool (SM::System::*)() const>(&SM::System::any_redraw),
                "any_redraw")

            .def("i", &SM::System::i, "i")
            .def("yieldDistanceRight", &SM::System::yieldDistanceRight, "yieldDistanceRight")
            .def("yieldDistanceLeft", &SM::System::yieldDistanceLeft, "yieldDistanceLeft")
            .def("set_t", &SM::System::set_t, "set_t", py::arg("arg"))
            .def("set_inc", &SM::System::set_inc, "set_inc", py::arg("arg"))
            .def("set_x_frame", &SM::System::set_x_frame, "set_x_frame", py::arg("arg"))
            .def("x_frame", &SM::System::x_frame, "x_frame")
            .def("set_x", &SM::System::set_x<xt::pytensor<double, 1>>, "x")
            .def("set_v", &SM::System::set_v<xt::pytensor<double, 1>>, "v")
            .def("set_a", &SM::System::set_a<xt::pytensor<double, 1>>, "a")
            .def("x", &SM::System::x, "x")
            .def("v", &SM::System::v, "v")
            .def("a", &SM::System::a, "a")
            .def("f", &SM::System::f, "f")
            .def("f_potential", &SM::System::f_potential, "f_potential")
            .def("f_frame", &SM::System::f_frame, "f_frame")
            .def("f_neighbours", &SM::System::f_neighbours, "f_neighbours")
            .def("f_damping", &SM::System::f_damping, "f_damping")
            .def("t", &SM::System::t, "t")
            .def("inc", &SM::System::inc, "inc")
            .def("temperature", &SM::System::temperature, "temperature")
            .def("residual", &SM::System::residual, "residual")
            .def("quench", &SM::System::quench, "quench")

            .def("timeStep", &SM::System::timeStep, "timeStep")
            .def("timeSteps", &SM::System::timeSteps, "timeSteps", py::arg("n"))

            .def(
                "timeStepsUntilEvent",
                &SM::System::timeStepsUntilEvent,
                "timeStepsUntilEvent",
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 10,
                py::arg("max_iter") = size_t(1e9))

            .def("flowSteps", &SM::System::flowSteps, "flowSteps", py::arg("n"), py::arg("v_frame"))

            .def(
                "flowSteps_boundcheck",
                &SM::System::flowSteps_boundcheck,
                "flowSteps_boundcheck",
                py::arg("n"),
                py::arg("v_frame"),
                py::arg("nmargin") = 5)

            .def(
                "minimise",
                &SM::System::minimise,
                "minimise",
                py::arg("nmargin") = 0,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 10,
                py::arg("max_iter") = size_t(1e9))

            .def(
                "minimise_timeactivity",
                &SM::System::minimise_timeactivity,
                "minimise_timeactivity",
                py::arg("nmargin") = 0,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 10,
                py::arg("max_iter") = size_t(1e9))

            .def(
                "minimise_nopassing",
                &SM::System::minimise_nopassing,
                "minimise_nopassing",
                py::arg("nmargin") = 0,
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
                "triggerWeakest",
                &SM::System::triggerWeakest,
                "triggerWeakest",
                py::arg("eps"),
                py::arg("direction") = 1)

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
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<long, 1>&>(),
                "SystemThermalRandomForcing",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_neighbours"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("x_yield"),
                py::arg("istart"));

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

        {
            py::class_<SM::ForceDrivenSystemThermalRandomForcing, SM::SystemThermalRandomForcing>
                cls(sm, "ForceDrivenSystemThermalRandomForcing");

            cls.def(
                py::init<
                    double,
                    double,
                    double,
                    double,
                    double,
                    double,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<long, 1>&>(),
                "ForceDrivenSystemThermalRandomForcing",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_neighbours"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("x_yield"),
                py::arg("istart"));

            cls.def(
                "setRemoteForce",
                &SM::ForceDrivenSystemThermalRandomForcing::setRemoteForce,
                "setRemoteForce",
                py::arg("f"));

            cls.def("__repr__", [](const SM::ForceDrivenSystemThermalRandomForcing&) {
                return "<FrictionQPotSpringBlock.Line1d.ForceDrivenSystemThermalRandomForcing>";
            });
        }
    }
}
