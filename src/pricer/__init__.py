"""
Pricer package initialization.
Auto-detects C++ compiled module; falls back to pure-Python implementation.
"""

try:
    # Try importing the C++ pybind11 module
    from pricer import (  # type: ignore
        bs_call, bs_put, greeks, greeks_call, greeks_put,
        implied_vol, mc_price, mc_price_multistep,
        norm_cdf, norm_pdf, Greeks, MCResult
    )
    BACKEND = "C++"
except ImportError:
    # Fallback to pure-Python implementation
    from .pricer_py import (
        bs_call, bs_put, greeks, greeks_call, greeks_put,
        implied_vol, mc_price, mc_price_multistep,
        norm_cdf, norm_pdf, Greeks, MCResult
    )
    BACKEND = "Python"

__all__ = [
    "bs_call", "bs_put", "greeks", "greeks_call", "greeks_put",
    "implied_vol", "mc_price", "mc_price_multistep",
    "norm_cdf", "norm_pdf", "Greeks", "MCResult", "BACKEND"
]
