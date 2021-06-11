/**
Python API

\file main.cpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

// #define QPOT_ENABLE_ASSERT
// #define GOOSEFEM_ENABLE_ASSERT
// #define FRICTIONQPOTSPRINGBLOCK_ENABLE_ASSERT
#define FRICTIONQPOTSPRINGBLOCK_ENABLE_WARNING_PYTHON

#include <FrictionQPotSpringBlock/version.h>
#include <FrictionQPotSpringBlock/Line1d.h>

namespace py = pybind11;

PYBIND11_MODULE(FrictionQPotSpringBlock, m)
{
    xt::import_numpy();

    m.doc() = "Spring-block friction model with local disordered potential energy landscape";

    namespace M = FrictionQPotSpringBlock;

    m.def("version",
          &M::version,
          "Return version string.");

    // ------------------------------
    // FrictionQPotSpringBlock.Line1d
    // ------------------------------

    {

    py::module sm = m.def_submodule("Line1d", "Line1d");

    namespace SM = FrictionQPotSpringBlock::Line1d;

    sm.def("version_dependencies",
           &SM::version_dependencies,
           "Return version information of library and its dependencies.");

    py::class_<SM::System>(sm, "System")

        .def(py::init<size_t, const xt::pytensor<double, 2>&>(),
             "System",
             py::arg("N"),
             py::arg("y"))

        .def(py::init<size_t, const xt::pytensor<double, 2>&, const xt::pytensor<long, 1>&>(),
             "System",
             py::arg("N"),
             py::arg("y"),
             py::arg("istart"))

        .def("N", &SM::System::N, "N")

        .def("set_y",
             py::overload_cast<const xt::pytensor<long, 1>&, const xt::pytensor<double, 2>&>(
                &SM::System::set_y<xt::pytensor<long, 1>, xt::pytensor<double, 2>>),
             "Reset the chunk of all particles.",
             py::arg("istart"),
             py::arg("y"))

        .def("set_y",
             py::overload_cast<size_t, long, const std::vector<double>&>(
                &SM::System::set_y<std::vector<double>>),
             "Reset the chunk of a particles.",
             py::arg("p"),
             py::arg("istart"),
             py::arg("y"))

        .def("shift_y",
             &SM::System::shift_y<std::vector<double>>,
             "shift_y",
             py::arg("p"),
             py::arg("istart"),
             py::arg("y"),
             py::arg("nbuffer") = 0)

        .def("shift_dy",
             &SM::System::shift_dy<std::vector<double>>,
             "shift_dy",
             py::arg("p"),
             py::arg("istart"),
             py::arg("dy"),
             py::arg("nbuffer") = 0)

        .def("ymin_chunk", &SM::System::ymin_chunk, "ymin_chunk")
        .def("yleft", &SM::System::yleft, "yleft")
        .def("yright", &SM::System::yright, "yright")
        .def("istart", &SM::System::istart, "istart")
        .def("boundcheck_left", &SM::System::boundcheck_left, "boundcheck_left", py::arg("n") = 0)
        .def("boundcheck_right", &SM::System::boundcheck_right, "boundcheck_right", py::arg("n") = 0)

        .def("any_redraw",
             static_cast<bool (SM::System::*)(const xt::pytensor<double, 1>&) const>(&SM::System::any_redraw),
             "any_redraw",
             py::arg("x"))

        .def("any_redraw",
             static_cast<bool (SM::System::*)() const>(&SM::System::any_redraw),
             "any_redraw")

        .def("any_shift", &SM::System::any_shift, "any_shift", py::arg("n"))
        .def("i", &SM::System::i, "i")
        .def("yieldDistanceRight", &SM::System::yieldDistanceRight, "yieldDistanceRight")
        .def("yieldDistanceLeft", &SM::System::yieldDistanceLeft, "yieldDistanceLeft")
        .def("set_dt", &SM::System::set_dt, "set_dt", py::arg("arg"))
        .def("set_eta", &SM::System::set_eta, "set_eta", py::arg("arg"))
        .def("set_m", &SM::System::set_m, "set_m", py::arg("arg"))
        .def("set_mu", &SM::System::set_mu, "set_mu", py::arg("arg"))
        .def("set_k_neighbours", &SM::System::set_k_neighbours, "set_k_neighbours", py::arg("arg"))
        .def("set_k_frame", &SM::System::set_k_frame, "set_k_frame", py::arg("arg"))
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
        .def("dt", &SM::System::dt, "dt")
        .def("residual", &SM::System::residual, "residual")
        .def("quench", &SM::System::quench, "quench")
        .def("timeStep", &SM::System::timeStep, "timeStep")

        .def("timeStepsUntilEvent",
             &SM::System::timeStepsUntilEvent,
             "timeStepsUntilEvent",
             py::arg("tol") = 1e-5,
             py::arg("niter_tol") = 20,
             py::arg("max_iter") = 10000000)

        .def("minimise",
             &SM::System::minimise,
             "minimise",
             py::arg("tol") = 1e-5,
             py::arg("niter_tol") = 20,
             py::arg("max_iter") = 10000000)

        .def("minimise_timeactivity",
             &SM::System::minimise_timeactivity,
             "minimise_timeactivity",
             py::arg("tol") = 1e-5,
             py::arg("niter_tol") = 20,
             py::arg("max_iter") = 10000000)

        .def("advanceElastic", &SM::System::advanceElastic, "advanceElastic", py::arg("dx"), py::arg("dx_of_frame") = true)
        .def("advanceEventRightElastic", &SM::System::advanceEventRightElastic, "advanceEventRightElastic", py::arg("eps"))
        .def("advanceEventRightKick", &SM::System::advanceEventRightKick, "advanceEventRightKick", py::arg("eps"))
        .def("triggerRight", &SM::System::triggerRight, "triggerRight", py::arg("p"), py::arg("eps"))
        .def("triggerWeakestRight", &SM::System::triggerWeakestRight, "triggerWeakestRight", py::arg("eps"))

        // deprecated
        .def("advanceRightElastic", &SM::System::advanceRightElastic, "advanceRightElastic", py::arg("arg"))
        .def("advanceRightKick", &SM::System::advanceRightKick, "advanceRightKick", py::arg("arg"))

        .def("__repr__", [](const SM::System&) {
            return "<FrictionQPotSpringBlock.Line1d.System>";
        });

      }
}
