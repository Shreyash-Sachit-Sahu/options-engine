/**
 * @file bindings.cpp
 * @brief pybind11 bindings for the C++ pricing engine.
 *
 * Exposes the pricer module to Python with full docstrings.
 * Build with CMake: pybind11_add_module(pricer ...)
 *
 * Usage from Python:
 *   import pricer
 *   price = pricer.bs_call(100, 100, 1.0, 0.05, 0.2)
 *   g = pricer.greeks(100, 100, 1.0, 0.05, 0.2)
 *   print(g.delta, g.gamma, g.vega, g.theta, g.rho)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pricer/bs_pricer.hpp"
#include "pricer/mc_pricer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pricer, m) {
    m.doc() = "High-performance options pricing engine (C++ core)";

    // ─── Greeks struct ───────────────────────────────────────────────────────
    py::class_<pricer::Greeks>(m, "Greeks",
        "Container for option Greeks (delta, gamma, vega, theta, rho)")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double>(),
             py::arg("delta"), py::arg("gamma"), py::arg("vega"),
             py::arg("theta"), py::arg("rho"))
        .def_readwrite("delta", &pricer::Greeks::delta, "Option delta (dC/dS)")
        .def_readwrite("gamma", &pricer::Greeks::gamma, "Option gamma (d²C/dS²)")
        .def_readwrite("vega",  &pricer::Greeks::vega,  "Option vega (dC/dσ)")
        .def_readwrite("theta", &pricer::Greeks::theta, "Option theta (dC/dT)")
        .def_readwrite("rho",   &pricer::Greeks::rho,   "Option rho (dC/dr)")
        .def("__repr__", [](const pricer::Greeks& g) {
            return "Greeks(delta=" + std::to_string(g.delta)
                 + ", gamma=" + std::to_string(g.gamma)
                 + ", vega=" + std::to_string(g.vega)
                 + ", theta=" + std::to_string(g.theta)
                 + ", rho=" + std::to_string(g.rho) + ")";
        });

    // ─── MCResult struct ─────────────────────────────────────────────────────
    py::class_<pricer::MCResult>(m, "MCResult",
        "Monte Carlo pricing result with price, standard error, and path count")
        .def(py::init<>())
        .def_readwrite("price",     &pricer::MCResult::price,     "Option price")
        .def_readwrite("std_error", &pricer::MCResult::std_error, "Standard error")
        .def_readwrite("n_paths",   &pricer::MCResult::n_paths,   "Number of paths")
        .def("__repr__", [](const pricer::MCResult& r) {
            return "MCResult(price=" + std::to_string(r.price)
                 + ", std_error=" + std::to_string(r.std_error)
                 + ", n_paths=" + std::to_string(r.n_paths) + ")";
        });

    // ─── Black-Scholes Pricing ───────────────────────────────────────────────
    m.def("bs_call", &pricer::bs_call,
          "Black-Scholes European call price",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"));

    m.def("bs_put", &pricer::bs_put,
          "Black-Scholes European put price",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"));

    // ─── Greeks ──────────────────────────────────────────────────────────────
    m.def("greeks_call", &pricer::greeks_call,
          "Compute all Greeks for a European call",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"));

    m.def("greeks_put", &pricer::greeks_put,
          "Compute all Greeks for a European put",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"));

    m.def("greeks", &pricer::greeks,
          "Compute all Greeks for a European option",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
          py::arg("flag") = "call");

    // ─── Implied Volatility ──────────────────────────────────────────────────
    m.def("implied_vol", &pricer::implied_vol,
          "Compute implied volatility via Brent's method",
          py::arg("market_price"), py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("flag") = "call",
          py::arg("tol") = 1e-12, py::arg("max_iter") = 200);

    // ─── Monte Carlo ─────────────────────────────────────────────────────────
    m.def("mc_price", &pricer::mc_price,
          "Monte Carlo European option price with antithetic variates",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
          py::arg("n_paths") = 50000, py::arg("seed") = 42,
          py::arg("flag") = "call");

    m.def("mc_price_multistep", &pricer::mc_price_multistep,
          "Monte Carlo with multi-step path simulation (for exotics)",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
          py::arg("n_paths") = 50000, py::arg("n_steps") = 252,
          py::arg("seed") = 42, py::arg("flag") = "call");

    // ─── Utility functions ───────────────────────────────────────────────────
    m.def("norm_cdf", &pricer::norm_cdf, "Standard normal CDF", py::arg("x"));
    m.def("norm_pdf", &pricer::norm_pdf, "Standard normal PDF", py::arg("x"));
}
