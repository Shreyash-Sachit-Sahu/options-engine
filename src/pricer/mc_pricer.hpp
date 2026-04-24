#pragma once
/**
 * @file mc_pricer.hpp
 * @brief Monte Carlo Options Pricing Engine with Antithetic Variates
 *
 * Implements Monte Carlo simulation for European options pricing using
 * antithetic variates for variance reduction (halves std error for same
 * path count). Uses Mersenne Twister 19937 PRNG from <random>.
 *
 * Reference: Glasserman, P. "Monte Carlo Methods in Financial Engineering", 2003.
 */

#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <string>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

namespace pricer {

// ─── MC Result Struct ────────────────────────────────────────────────────────
struct MCResult {
    double price;
    double std_error;
    int    n_paths;

    MCResult() : price(0), std_error(0), n_paths(0) {}
    MCResult(double p, double se, int n) : price(p), std_error(se), n_paths(n) {}
};

// ─── Monte Carlo Pricer ──────────────────────────────────────────────────────

/**
 * @brief Monte Carlo European option price with antithetic variates.
 *
 * For each path i, we generate Z_i ~ N(0,1) and simulate two terminal prices:
 *   S_T(+) = S * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z_i)
 *   S_T(-) = S * exp((r - 0.5*sigma^2)*T - sigma*sqrt(T)*Z_i)
 *
 * The antithetic estimate averages the payoff of both paths, which is
 * negatively correlated and reduces variance by ~50%.
 *
 * @param S       Spot price
 * @param K       Strike price
 * @param T       Time to maturity (years)
 * @param r       Risk-free rate
 * @param sigma   Volatility
 * @param n_paths Number of simulation paths (default: 50000)
 * @param seed    RNG seed (default: 42)
 * @param flag    "call" or "put" (default: "call")
 * @return MCResult with price, standard error, and path count
 */
inline MCResult mc_price(double S, double K, double T, double r, double sigma,
                         int n_paths = 50000, unsigned int seed = 42,
                         const std::string& flag = "call") {
    if (T <= 0.0) {
        double payoff = 0.0;
        if (flag == "call" || flag == "c") payoff = std::max(S - K, 0.0);
        else payoff = std::max(K - S, 0.0);
        return MCResult(payoff, 0.0, 0);
    }

    bool is_call = (flag == "call" || flag == "c");
    if (!is_call && flag != "put" && flag != "p") {
        throw std::invalid_argument("flag must be 'call' or 'put', got: " + flag);
    }

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol_sqrt_T = sigma * std::sqrt(T);
    double discount = std::exp(-r * T);

    double sum_payoff = 0.0;
    double sum_payoff_sq = 0.0;

    for (int i = 0; i < n_paths; ++i) {
        double Z = normal(rng);

        // Path +Z
        double ST_plus  = S * std::exp(drift + vol_sqrt_T * Z);
        // Path -Z (antithetic)
        double ST_minus = S * std::exp(drift - vol_sqrt_T * Z);

        double payoff_plus, payoff_minus;
        if (is_call) {
            payoff_plus  = std::max(ST_plus  - K, 0.0);
            payoff_minus = std::max(ST_minus - K, 0.0);
        } else {
            payoff_plus  = std::max(K - ST_plus,  0.0);
            payoff_minus = std::max(K - ST_minus, 0.0);
        }

        // Antithetic average
        double payoff_avg = 0.5 * (payoff_plus + payoff_minus);
        sum_payoff    += payoff_avg;
        sum_payoff_sq += payoff_avg * payoff_avg;
    }

    double mean_payoff = sum_payoff / n_paths;
    double variance = (sum_payoff_sq / n_paths) - (mean_payoff * mean_payoff);
    double std_error = std::sqrt(std::max(variance, 0.0) / n_paths);

    double price = discount * mean_payoff;
    double se    = discount * std_error;

    return MCResult(price, se, n_paths);
}

/**
 * @brief Monte Carlo pricer with multi-step path simulation.
 *        Useful for path-dependent exotic options (Asian, Barrier, etc.)
 *
 * @param S       Spot price
 * @param K       Strike price
 * @param T       Time to maturity (years)
 * @param r       Risk-free rate
 * @param sigma   Volatility
 * @param n_paths Number of simulation paths
 * @param n_steps Number of time steps per path
 * @param seed    RNG seed
 * @param flag    "call" or "put"
 * @return MCResult
 */
inline MCResult mc_price_multistep(double S, double K, double T, double r,
                                    double sigma, int n_paths = 50000,
                                    int n_steps = 252, unsigned int seed = 42,
                                    const std::string& flag = "call") {
    if (T <= 0.0) {
        double payoff = (flag == "call" || flag == "c") ?
                         std::max(S - K, 0.0) : std::max(K - S, 0.0);
        return MCResult(payoff, 0.0, 0);
    }

    bool is_call = (flag == "call" || flag == "c");
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double dt = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol_sqrt_dt = sigma * std::sqrt(dt);
    double discount = std::exp(-r * T);

    double sum_payoff = 0.0;
    double sum_payoff_sq = 0.0;

    for (int i = 0; i < n_paths; ++i) {
        double S_plus  = S;
        double S_minus = S;

        for (int j = 0; j < n_steps; ++j) {
            double Z = normal(rng);
            S_plus  *= std::exp(drift + vol_sqrt_dt * Z);
            S_minus *= std::exp(drift - vol_sqrt_dt * Z);
        }

        double payoff_plus, payoff_minus;
        if (is_call) {
            payoff_plus  = std::max(S_plus  - K, 0.0);
            payoff_minus = std::max(S_minus - K, 0.0);
        } else {
            payoff_plus  = std::max(K - S_plus,  0.0);
            payoff_minus = std::max(K - S_minus, 0.0);
        }

        double payoff_avg = 0.5 * (payoff_plus + payoff_minus);
        sum_payoff    += payoff_avg;
        sum_payoff_sq += payoff_avg * payoff_avg;
    }

    double mean_payoff = sum_payoff / n_paths;
    double variance = (sum_payoff_sq / n_paths) - (mean_payoff * mean_payoff);
    double std_error = std::sqrt(std::max(variance, 0.0) / n_paths);

    return MCResult(discount * mean_payoff, discount * std_error, n_paths);
}

}  // namespace pricer
