/**
Python API

\file main.cpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#include <pybind11/pybind11.h>
#include <pyxtensor/pyxtensor.hpp>

// #define QPOT_ENABLE_ASSERT
// #define GOOSEFEM_ENABLE_ASSERT
// #define FRICTIONQPOTSPRINGBLOCK_ENABLE_ASSERT

#include <FrictionQPotSpringBlock/version.h>
#include <FrictionQPotSpringBlock/Line1d.h>

namespace py = pybind11;

PYBIND11_MODULE(FrictionQPotSpringBlock, m)
{

    m.doc() = "Spring-block friction model with local disordered potential energy landscape";

    namespace M = FrictionQPotSpringBlock;

    m.def("version",
          &M::version,
          "Return version string.");

    // ------------------------------
    // FrictionQPotSpringBlock.Line1d
    // ------------------------------

    py::module sm = m.def_submodule("Line1d", "Line1d");

    namespace SM = FrictionQPotSpringBlock::Line1d;

    sm.def("version_dependencies",
           &SM::version_dependencies,
           "Return version information of library and its dependencies.");

    py::class_<SM::System>(sm, "System")

        .def(py::init<
                size_t,
                std::function<xt::xtensor<double, 2>(std::vector<size_t>)>>(),
             "System",
             py::arg("N"),
             py::arg("function_to_draw_distances"))

        .def("N", &SM::System::N, "N")

        .def("set_dt", &SM::System::set_dt, "set_dt", py::arg("arg"))
        .def("set_eta", &SM::System::set_eta, "set_eta", py::arg("arg"))
        .def("set_m", &SM::System::set_m, "set_m", py::arg("arg"))
        .def("set_mu", &SM::System::set_mu, "set_mu", py::arg("arg"))
        .def("set_k_neighbours", &SM::System::set_k_neighbours, "set_k_neighbours", py::arg("arg"))
        .def("set_k_frame", &SM::System::set_k_frame, "set_k_frame", py::arg("arg"))
        .def("set_x_frame", &SM::System::set_x_frame, "set_x_frame", py::arg("arg"))
        .def("get_x_frame", &SM::System::get_x_frame, "get_x_frame")
        .def("get_x", &SM::System::get_x, "get_x")
        .def("get_v", &SM::System::get_v, "get_v")
        .def("get_f", &SM::System::get_f, "get_f")
        .def("get_f_potential", &SM::System::get_f_potential, "get_f_potential")
        .def("get_f_frame", &SM::System::get_f_frame, "get_f_frame")
        .def("get_f_neighbours", &SM::System::get_f_neighbours, "get_f_neighbours")
        .def("timeStep", &SM::System::timeStep, "timeStep")
        .def("minimise", &SM::System::minimise, "minimise")
        // .def("minimise",
        //      &SM::System::minimise,
        //      "minimise",
        //      py::arg("tol") = 1e-5,
        //      py::arg("niter_tol") = 20,
        //      py::arg("max_iter") = 1000000)
        .def("advanceRightElastic", &SM::System::advanceRightElastic, "advanceRightElastic", py::arg("arg"))
        .def("advanceRightKick", &SM::System::advanceRightKick, "advanceRightKick", py::arg("arg"))
        .def("currentYieldLeft", &SM::System::currentYieldLeft, "currentYieldLeft")
        .def("currentYieldRight", &SM::System::currentYieldRight, "currentYieldRight")
        .def("getYieldIndex", &SM::System::getYieldIndex, "getYieldIndex")
        .def("getYieldDistanceRight", &SM::System::getYieldDistanceRight, "getYieldDistanceRight")
        .def("getYieldDistanceLeft", &SM::System::getYieldDistanceLeft, "getYieldDistanceLeft")

        .def("__repr__", [](const SM::System&) {
            return "<FrictionQPotSpringBlock.Line1d.System>";
        });

}
