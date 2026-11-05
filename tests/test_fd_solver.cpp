#include <cmath>

#include "check.hpp"
#include "fpinn/black_scholes.hpp"
#include "fpinn/fd_solver.hpp"

using namespace fpinn;

int main() {
    FractionalBS p;  // K=100, r=0.05, sigma=0.2, T=1, L=1.5
    FDConfig cfg;
    cfg.nx = 800;
    cfg.nt = 400;

    // alpha = 1: must reproduce the closed-form Black-Scholes prices and Greeks.
    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        p.alpha = 1.0;
        p.type = type;
        FDSolver fd(p, cfg);
        fd.solve();
        double max_err = 0.0, max_derr = 0.0, max_gerr = 0.0;
        for (double S = 60.0; S <= 160.0; S += 2.5) {
            const BSResult bs = black_scholes(type, S, p.K, p.r, p.sigma, p.T);
            max_err = std::max(max_err, std::abs(fd.price(S, p.T) - bs.price));
            max_derr = std::max(max_derr, std::abs(fd.delta(S, p.T) - bs.delta));
            max_gerr = std::max(max_gerr, std::abs(fd.gamma(S, p.T) - bs.gamma));
        }
        CHECK(max_err < 5e-3);   // price error below half a cent on a 100 strike
        CHECK(max_derr < 1e-3);
        CHECK(max_gerr < 1e-4);
        // Mid-way in time as well (linear interpolation between levels).
        const BSResult mid = black_scholes(type, 100.0, p.K, p.r, p.sigma, 0.37);
        CHECK_NEAR(fd.price(100.0, 0.37), mid.price, 1e-2);
    }

    // alpha < 1: fractional put-call parity  C - P = S - K E_alpha(-r T^alpha).
    p.alpha = 0.6;
    p.type = OptionType::Call;
    FDSolver call(p, cfg);
    call.solve();
    p.type = OptionType::Put;
    FDSolver put(p, cfg);
    put.solve();
    for (double S : {70.0, 100.0, 130.0}) {
        CHECK_NEAR(call.price(S, p.T) - put.price(S, p.T), S - p.K * p.discount(p.T), 2e-3);
    }

    // alpha = 1/2: compare with the exact subordination formula (independent check).
    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        FractionalBS h = p;
        h.alpha = 0.5;
        h.type = type;
        FDSolver fd(h, cfg);
        fd.solve();
        double max_err = 0.0;
        for (double S = 60.0; S <= 160.0; S += 5.0) {
            max_err = std::max(max_err, std::abs(fd.price(S, h.T) - fractional_bs_alpha_half(type, S, h.K, h.r,
                                                                                              h.sigma, h.T)));
        }
        CHECK(max_err < 2e-3);
        CHECK_NEAR(fd.price(100.0, 0.3), fractional_bs_alpha_half(type, 100.0, h.K, h.r, h.sigma, 0.3), 3e-3);
    }

    // alpha < 1: grid convergence (differences shrink under refinement).
    p.type = OptionType::Call;
    auto atm = [&](int nx, int nt) {
        FDSolver s(p, FDConfig{nx, nt, 2.0});
        s.solve();
        return s.price(100.0, p.T);
    };
    const double v1 = atm(200, 100), v2 = atm(400, 200), v3 = atm(800, 400);
    CHECK(std::abs(v3 - v2) < 0.6 * std::abs(v2 - v1));
    CHECK(std::abs(v3 - v2) < 2e-2);

    // Truncation check: a wider domain gives the same price.
    FractionalBS wide = p;
    wide.L = 2.5;
    FDSolver w(wide, FDConfig{1333, 400, 2.0});
    w.solve();
    CHECK_NEAR(w.price(100.0, p.T), call.price(100.0, p.T), 5e-3);

    return test_summary("test_fd_solver");
}
