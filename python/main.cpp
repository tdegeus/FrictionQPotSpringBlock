/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/xtensor_python_config.hpp> // todo: remove for xtensor-python >0.26.1

#define FRICTIONQPOTSPRINGBLOCK_USE_XTENSOR_PYTHON
#include <FrictionQPotSpringBlock/Line1d.h>
#include <FrictionQPotSpringBlock/Line2d.h>
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

template <class Binder, class System>
void mySystemNd(Binder& cls)
{
    cls.def_property_readonly("size", &System::size, "Number of particles");
    cls.def_property_readonly("shape", &System::shape, "Shape of the system");
    cls.def_property("u", &System::u, &System::set_u, "Particle slip ('positions').");
    cls.def_property("v", &System::v, &System::set_v, "Particle velocities.");
    cls.def_property("a", &System::a, &System::set_a, "Particle accelerations.");
    cls.def_property("inc", &System::inc, &System::set_inc, "Increment");
    cls.def_property("t", &System::t, &System::set_t, "Time");
    cls.def_property("u_frame", &System::u_frame, &System::set_u_frame, "Frame position");
    cls.def_property_readonly("f", &System::f, "Residual forces");
    cls.def_property_readonly("f_potential", &System::f_potential, "Elastic forces");
    cls.def_property_readonly("f_frame", &System::f_frame, "Frame forces");
    cls.def_property_readonly("f_interactions", &System::f_interactions, "Interaction forces");
    cls.def_property_readonly("f_damping", &System::f_damping, "Particle damping forces");
    cls.def("temperature", &System::temperature, "Temperature");
    cls.def("residual", &System::residual, "Residual");
    cls.def("refresh", &System::refresh, "refresh");
    cls.def("quench", &System::quench, "quench");

    cls.def(
        "maxUniformDisplacement",
        &System::maxUniformDisplacement,
        "maxUniformDisplacement",
        py::arg("direction") = 1
    );

    cls.def(
        "trigger",
        &System::trigger,
        "trigger",
        py::arg("p"),
        py::arg("eps"),
        py::arg("direction") = 1
    );

    cls.def(
        "advanceToFixedForce",
        &System::advanceToFixedForce,
        "advanceToFixedForce",
        py::arg("f_frame")
    );
}

template <class Binder, class System>
void mySystemNdAthermal(Binder& cls)
{
    cls.def(
        "minimise",
        &System::minimise,
        "minimise",
        py::arg("tol") = 1e-5,
        py::arg("niter_tol") = 10,
        py::arg("max_iter") = size_t(1e9),
        py::arg("time_activity") = false,
        py::arg("max_iter_is_error") = true
    );

    cls.def(
        "eventDrivenStep",
        &System::eventDrivenStep,
        "eventDrivenStep",
        py::arg("eps"),
        py::arg("kick"),
        py::arg("direction") = 1
    );

    cls.def(
        "quasistaticActivityFirst", &System::quasistaticActivityFirst, "quasistaticActivityFirst"
    );

    cls.def("quasistaticActivityLast", &System::quasistaticActivityLast, "quasistaticActivityLast");
}

template <class Binder, class System>
void mySystemNdExternal(Binder& cls)
{
    cls.def_property_readonly("external", &System::external, "Class adding external force");
}

template <class Binder, class System>
void mySystemNdDynamics(Binder& cls)
{
    cls.def("timeStep", &System::timeStep, "timeStep");
    cls.def("timeSteps", &System::timeSteps, "timeSteps", py::arg("n"));

    cls.def(
        "timeStepsUntilEvent",
        &System::timeStepsUntilEvent,
        "timeStepsUntilEvent",
        py::arg("tol") = 1e-5,
        py::arg("niter_tol") = 10,
        py::arg("max_iter") = size_t(1e9)
    );

    cls.def("flowSteps", &System::flowSteps, "flowSteps", py::arg("n"), py::arg("v_frame"));
}

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
    // FrictionQPotSpringBlock.detail
    // ------------------------------

    {
        py::module sm = m.def_submodule("detail", "detail");
        namespace SM = FrictionQPotSpringBlock::detail;

        {
            using S = SM::RandomNormalForcing<1>;

            py::class_<S> cls(sm, "RandomNormalForcing_1");

            cls.def_property("state", &S::state, &S::set_state, "State of RNG");
            cls.def_property("f_thermal", &S::f_thermal, &S::set_f_thermal, "Random force");
            cls.def_property("next", &S::next, &S::set_next, "Next draw increment");

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.detail.RandomNormalForcing_1>";
            });
        } // namespace FrictionQPotSpringBlock::detail;

        {
            using S = SM::Athermal;

            py::class_<S> cls(sm, "Athermal");

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.detail.Athermal>";
            });
        }
    }

    // ------------------------------
    // FrictionQPotSpringBlock.Line1d
    // ------------------------------

    {

        py::module sm = m.def_submodule("Line1d", "Line1d");
        namespace SM = FrictionQPotSpringBlock::Line1d;

        sm.def(
            "version_dependencies",
            &SM::version_dependencies,
            "Return version information of library and its dependencies."
        );

        sm.def("version_compiler", &SM::version_compiler, "Return compiler information.");

        {
            using S = SM::System_Cuspy_Laplace;

            py::class_<S> cls(sm, "System_Cuspy_Laplace");

            cls.def(
                py::init<double, double, double, double, double, double, SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_interactions"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunk")
            );

            mySystemNd<py::class_<S>, S>(cls);
            mySystemNdAthermal<py::class_<S>, S>(cls);
            mySystemNdDynamics<py::class_<S>, S>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace>";
            });
        }

        {
            using S = SM::System_Cuspy_Laplace_Nopassing;

            py::class_<S> cls(sm, "System_Cuspy_Laplace_Nopassing");

            cls.def(
                py::init<double, double, double, SM::Generator*, double, double>(),
                "Constructor.",
                py::arg("mu"),
                py::arg("k_interactions"),
                py::arg("k_frame"),
                py::arg("chunk"),
                py::arg("eta") = 0.0,
                py::arg("dt") = 0.0
            );

            mySystemNd<py::class_<S>, S>(cls);
            mySystemNdAthermal<py::class_<S>, S>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace_Nopassing>";
            });
        }

        {
            using S = SM::System_Cuspy_Laplace_RandomForcing;

            py::class_<S> cls(sm, "System_Cuspy_Laplace_RandomForcing");

            cls.def(
                py::init<
                    double,
                    double,
                    double,
                    double,
                    double,
                    double,
                    SM::Generator*,
                    double,
                    double,
                    uint64_t,
                    const xt::pytensor<ptrdiff_t, 1>&,
                    const xt::pytensor<ptrdiff_t, 1>&>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_interactions"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunk"),
                py::arg("mean"),
                py::arg("stddev"),
                py::arg("seed"),
                py::arg("dinc_init"),
                py::arg("dinc")
            );

            mySystemNd<py::class_<S>, S>(cls);
            mySystemNdDynamics<py::class_<S>, S>(cls);
            mySystemNdExternal<py::class_<S>, S>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace_RandomForcing>";
            });
        }

        {
            using S = SM::System_SemiSmooth_Laplace;

            py::class_<S> cls(sm, "System_SemiSmooth_Laplace");

            cls.def(
                py::init<double, double, double, double, double, double, double, SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("kappa"),
                py::arg("k_interactions"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunk")
            );

            mySystemNd<py::class_<S>, S>(cls);
            mySystemNdAthermal<py::class_<S>, S>(cls);
            mySystemNdDynamics<py::class_<S>, S>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line1d.System_SemiSmooth_Laplace>";
            });
        }

        {
            using S = SM::System_Smooth_Laplace;

            py::class_<S> cls(sm, "System_Smooth_Laplace");

            cls.def(
                py::init<double, double, double, double, double, double, SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_interactions"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunk")
            );

            mySystemNd<py::class_<S>, S>(cls);
            mySystemNdAthermal<py::class_<S>, S>(cls);
            mySystemNdDynamics<py::class_<S>, S>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line1d.System_Smooth_Laplace>";
            });
        }

        {
            using S = SM::System_Cuspy_Quartic;

            py::class_<S> cls(sm, "System_Cuspy_Quartic");

            cls.def(
                py::init<double, double, double, double, double, double, double, SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("a1"),
                py::arg("a2"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunk")
            );

            mySystemNd<py::class_<S>, S>(cls);
            mySystemNdAthermal<py::class_<S>, S>(cls);
            mySystemNdDynamics<py::class_<S>, S>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line1d.System_Cuspy_Quartic>";
            });
        }

        {
            using S = SM::System_Cuspy_QuarticGradient;

            py::class_<S> cls(sm, "System_Cuspy_QuarticGradient");

            cls.def(
                py::init<double, double, double, double, double, double, double, SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k2"),
                py::arg("k4"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunk")
            );

            mySystemNd<py::class_<S>, S>(cls);
            mySystemNdAthermal<py::class_<S>, S>(cls);
            mySystemNdDynamics<py::class_<S>, S>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line1d.System_Cuspy_QuarticGradient>";
            });
        }

        {
            using S = SM::System_Cuspy_LongRange;

            py::class_<S> cls(sm, "System_Cuspy_LongRange");

            cls.def(
                py::init<double, double, double, double, double, double, double, SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_interactions"),
                py::arg("alpha"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunk")
            );

            mySystemNd<py::class_<S>, S>(cls);
            mySystemNdAthermal<py::class_<S>, S>(cls);
            mySystemNdDynamics<py::class_<S>, S>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line1d.System_Cuspy_LongRange>";
            });
        }
    }

    // ------------------------------
    // FrictionQPotSpringBlock.Line2d
    // ------------------------------

    {
        py::module sm = m.def_submodule("Line2d", "Line2d");
        namespace SM = FrictionQPotSpringBlock::Line2d;

        sm.def(
            "version_dependencies",
            &SM::version_dependencies,
            "Return version information of library and its dependencies."
        );

        sm.def("version_compiler", &SM::version_compiler, "Return compiler information.");

        {
            using S = SM::System_Cuspy_Laplace;

            py::class_<S> cls(sm, "System_Cuspy_Laplace");

            cls.def(
                py::init<double, double, double, double, double, double, SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k_interactions"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunk")
            );

            mySystemNd<py::class_<S>, S>(cls);
            mySystemNdAthermal<py::class_<S>, S>(cls);
            mySystemNdDynamics<py::class_<S>, S>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line2d.System_Cuspy_Laplace>";
            });
        }

        {
            using S = SM::System_Cuspy_QuarticGradient;

            py::class_<S> cls(sm, "System_Cuspy_QuarticGradient");

            cls.def(
                py::init<double, double, double, double, double, double, double, SM::Generator*>(),
                "Constructor.",
                py::arg("m"),
                py::arg("eta"),
                py::arg("mu"),
                py::arg("k2"),
                py::arg("k4"),
                py::arg("k_frame"),
                py::arg("dt"),
                py::arg("chunk")
            );

            mySystemNd<py::class_<S>, S>(cls);
            mySystemNdAthermal<py::class_<S>, S>(cls);
            mySystemNdDynamics<py::class_<S>, S>(cls);

            cls.def("__repr__", [](const S&) {
                return "<FrictionQPotSpringBlock.Line2d.System_Cuspy_QuarticGradient>";
            });
        }
    }
}
