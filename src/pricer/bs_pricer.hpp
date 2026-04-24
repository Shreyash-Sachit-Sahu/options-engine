#pragma once
/**
 * @file bs_pricer.hpp
 * @brief Black-Scholes Analytical Pricing Engine
 *
 * Implements closed-form BSM pricing, Greeks computation, and
 * implied volatility via Brent's method. All calculations use
 * double precision. No external dependencies beyond <cmath>.
 *
 * Reference: Hull, J.C. "Options, Futures, and Other Derivatives", 11th Ed.
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <limits>

namespace pricer {

// ─── Greeks Result Struct ────────────────────────────────────────────────────
struct Greeks {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;

    Greeks() : delta(0), gamma(0), vega(0), theta(0), rho(0) {}
    Greeks(double d, double g, double v, double t, double r)
        : delta(d), gamma(g), vega(v), theta(t), rho(r) {}
};

// ─── Normal Distribution Functions ───────────────────────────────────────────

/**
 * @brief Standard normal CDF using the complementary error function.
 *        Accuracy: ~15 significant digits (full double precision).
 *        Reference: Abramowitz & Stegun, Eq. 7.1.13
 */
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);  // M_SQRT1_2 = 1/sqrt(2)
}

/**
 * @brief Standard normal PDF.
 */
inline double norm_pdf(double x) {
    static const double INV_SQRT_2PI = 0.3989422804014327;
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

// ─── d1 / d2 helpers ────────────────────────────────────────────────────────

inline double compute_d1(double S, double K, double T, double r, double sigma) {
    return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
}

inline double compute_d2(double d1, double sigma, double T) {
    return d1 - sigma * std::sqrt(T);
}

// ─── Black-Scholes Pricing ───────────────────────────────────────────────────

/**
 * @brief Black-Scholes European call price.
 * @param S  Spot price
 * @param K  Strike price
 * @param T  Time to maturity (years)
 * @param r  Risk-free rate (annualized, continuous compounding)
 * @param sigma  Volatility (annualized)
 * @return Call option price
 */
inline double bs_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(S - K, 0.0);
    if (sigma <= 0.0) return std::max(S - K * std::exp(-r * T), 0.0);

    double d1 = compute_d1(S, K, T, r, sigma);
    double d2 = compute_d2(d1, sigma, T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

/**
 * @brief Black-Scholes European put price.
 */
inline double bs_put(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(K - S, 0.0);
    if (sigma <= 0.0) return std::max(K * std::exp(-r * T) - S, 0.0);

    double d1 = compute_d1(S, K, T, r, sigma);
    double d2 = compute_d2(d1, sigma, T);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

// ─── Greeks ──────────────────────────────────────────────────────────────────

/**
 * @brief Compute all Greeks for a European call option.
 *
 * Delta: dC/dS = N(d1)
 * Gamma: d²C/dS² = N'(d1) / (S * sigma * sqrt(T))
 * Vega:  dC/dsigma = S * N'(d1) * sqrt(T)  (per 1.0 vol, not per 1%)
 * Theta: dC/dT (per year)
 * Rho:   dC/dr = K * T * exp(-rT) * N(d2)  (per 1.0 rate)
 */
inline Greeks greeks_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) {
        Greeks g;
        g.delta = (S > K) ? 1.0 : 0.0;
        return g;
    }

    double sqrtT = std::sqrt(T);
    double d1 = compute_d1(S, K, T, r, sigma);
    double d2 = compute_d2(d1, sigma, T);
    double nd1 = norm_pdf(d1);
    double Nd1 = norm_cdf(d1);
    double Nd2 = norm_cdf(d2);
    double Ke_rT = K * std::exp(-r * T);

    Greeks g;
    g.delta = Nd1;
    g.gamma = nd1 / (S * sigma * sqrtT);
    g.vega  = S * nd1 * sqrtT;
    g.theta = -(S * nd1 * sigma) / (2.0 * sqrtT) - r * Ke_rT * Nd2;
    g.rho   = Ke_rT * T * Nd2;
    return g;
}

/**
 * @brief Compute all Greeks for a European put option.
 */
inline Greeks greeks_put(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) {
        Greeks g;
        g.delta = (S < K) ? -1.0 : 0.0;
        return g;
    }

    double sqrtT = std::sqrt(T);
    double d1 = compute_d1(S, K, T, r, sigma);
    double d2 = compute_d2(d1, sigma, T);
    double nd1 = norm_pdf(d1);
    double Nd1 = norm_cdf(d1);
    double Nd2 = norm_cdf(d2);
    double Ke_rT = K * std::exp(-r * T);

    Greeks g;
    g.delta = Nd1 - 1.0;
    g.gamma = nd1 / (S * sigma * sqrtT);
    g.vega  = S * nd1 * sqrtT;
    g.theta = -(S * nd1 * sigma) / (2.0 * sqrtT) + r * Ke_rT * (1.0 - Nd2);
    g.rho   = -Ke_rT * T * (1.0 - Nd2);
    return g;
}

/**
 * @brief Convenience: compute Greeks for 'call' or 'put'.
 */
inline Greeks greeks(double S, double K, double T, double r, double sigma,
                     const std::string& flag = "call") {
    if (flag == "call" || flag == "c") return greeks_call(S, K, T, r, sigma);
    if (flag == "put"  || flag == "p") return greeks_put(S, K, T, r, sigma);
    throw std::invalid_argument("flag must be 'call' or 'put', got: " + flag);
}

// ─── Implied Volatility (Brent's Method) ─────────────────────────────────────

/**
 * @brief Compute implied volatility using Brent's method.
 *
 * Brent's method combines bisection, secant, and inverse quadratic interpolation
 * for superlinear convergence with guaranteed robustness.
 *
 * Reference: Brent, R.P. "Algorithms for Minimization Without Derivatives", 1973.
 *
 * @param market_price  Observed market price of the option
 * @param S   Spot price
 * @param K   Strike price
 * @param T   Time to maturity
 * @param r   Risk-free rate
 * @param flag  "call" or "put"
 * @param tol   Convergence tolerance (default: 1e-12)
 * @param max_iter  Maximum iterations (default: 200)
 * @return Implied volatility
 */
inline double implied_vol(double market_price, double S, double K, double T,
                          double r, const std::string& flag = "call",
                          double tol = 1e-12, int max_iter = 200) {
    // Select pricing function
    auto price_fn = (flag == "call" || flag == "c") ? bs_call : bs_put;

    // Bracket [a, b] for sigma
    double a = 1e-6;
    double b = 10.0;

    double fa = price_fn(S, K, T, r, a) - market_price;
    double fb = price_fn(S, K, T, r, b) - market_price;

    if (fa * fb > 0.0) {
        throw std::runtime_error("Cannot bracket implied vol: market price out of range");
    }

    // Ensure f(a) and f(b) are oriented correctly
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a, fc = fa;
    bool mflag = true;
    double s = 0.0, fs = 0.0;
    double d = 0.0;

    for (int i = 0; i < max_iter; ++i) {
        if (std::abs(fb) < tol) return b;
        if (std::abs(a - b) < tol) return b;

        if (std::abs(fa - fc) > tol && std::abs(fb - fc) > tol) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Conditions for bisection
        double cond_min = (3.0 * a + b) / 4.0;
        double cond_max = b;
        if (cond_min > cond_max) std::swap(cond_min, cond_max);

        bool cond1 = !(s >= cond_min && s <= cond_max);
        bool cond2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2.0;
        bool cond3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2.0;
        bool cond4 = mflag && std::abs(b - c) < tol;
        bool cond5 = !mflag && std::abs(c - d) < tol;

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        fs = price_fn(S, K, T, r, s) - market_price;
        d = c;
        c = b;
        fc = fb;

        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }

    return b;  // Best estimate after max_iter
}

}  // namespace pricer
