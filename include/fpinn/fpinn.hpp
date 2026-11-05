#pragma once

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <random>
#include <string>
#include <vector>

#include "fpinn/black_scholes.hpp"
#include "fpinn/caputo.hpp"
#include "fpinn/mlp.hpp"
#include "fpinn/problem.hpp"

namespace fpinn {

/// Known function added to the trial solution so the network only has to learn a
/// smooth correction (see FPINNSolver).
enum class Baseline {
    None,            ///< plain fPINN
    SubordinatedBS,  ///< w = sum_q c_q * BS(x, tau^alpha * y_q): Gauss-quadrature average of
                     ///< classical prices over the model's random clock (1 node = mean time)
};
Baseline parse_baseline(const std::string& s);
const char* to_string(Baseline b);

struct TrainConfig {
    std::vector<int> hidden = {32, 32, 32};  ///< hidden layer widths
    int nx = 64;            ///< collocation points in x per Adam step (re-sampled every step)
    int nt = 40;            ///< number of time steps of the L1 scheme
    double grading = 1.5;   ///< time-grid grading (1 = uniform)
    int adam_steps = 6000;  ///< Adam iterations
    double lr = 3e-3;       ///< initial learning rate
    double lr_final = 3e-5; ///< final learning rate (geometric decay)
    int lbfgs_steps = 2000; ///< L-BFGS iterations after Adam (0 = off)
    int lbfgs_nx = 160;     ///< fixed collocation points in x used by L-BFGS
    std::uint64_t seed = 1234;
    int log_every = 500;
    /// Divide the residual at level n by the L1 diagonal weight W(n,n) (~ dtau^-alpha).
    /// Without it the first, tiny time steps (where the payoff kink is still sharp)
    /// dominate the loss.
    bool scale_residuals = true;
    /// Time input of the network is 2 (tau/T)^time_power - 1.  0.5 matches the
    /// sqrt(tau) spreading of diffusion; 1 is a plain linear rescaling.
    double time_power = 0.5;
    Baseline baseline = Baseline::None;
    int baseline_nodes = 3;  ///< quadrature nodes of the baseline (1 = BS at the mean random time)
    /// Add the similarity variable (x/L) / ((tau/T)^(alpha/2) + similarity_floor)
    /// as a third network input.
    bool similarity_feature = true;
    double similarity_floor = 0.05;
    /// With a baseline, its Caputo derivative is computed on a fine sub-grid with this
    /// many sub-steps per coarse step (the first step gets 4x as many, graded).
    int baseline_substeps = 16;
};

/// Fractional physics-informed neural network (fPINN, Pang, Lu & Karniadakis 2019)
/// for the time-fractional Black-Scholes equation.
///
/// Ingredients:
///  * trial solution with hard-wired boundary conditions
///        u(x,tau) = w(x,tau) + A(x,tau) + B(x) * N(x/L, 2 (tau/T)^p - 1),
///    w = optional known baseline (absorbs the payoff kink), A = linear
///    interpolation of the boundary values of u - w, B = 1 - (x/L)^2;
///  * integer-order derivatives u_x, u_xx by automatic differentiation;
///  * the non-local Caputo derivative by the L1 scheme on the time grid
///    tau_0..tau_M, with u(x, tau_0) = payoff(x) inserted exactly (this is how
///    the initial condition is imposed).  With a baseline, u = w + z: the L1
///    scheme on the coarse grid is applied only to the smooth part z (z_0 = 0),
///    while D^alpha w is computed on a fine sub-grid, so the payoff kink never
///    enters the coarse memory sum;
///  * loss = mean squared PDE residual over (x_i, tau_n), n = 1..M, each level
///    optionally scaled by 1 / W(n,n) (see TrainConfig::scale_residuals).
class FPINNSolver {
public:
    FPINNSolver(const FractionalBS& problem, const TrainConfig& cfg);

    /// Mean squared residual at the collocation abscissae xs (all time levels).
    /// If grad != nullptr it receives d(loss)/d(params).
    double loss_and_grad(const std::vector<double>& xs, std::vector<double>* grad) const;

    /// Optional monitor called at every log step; its return value is appended to the log line.
    using Monitor = std::function<std::string()>;
    void train(std::ostream* log = nullptr, const Monitor& monitor = {});

    /// Solution in dimensionless variables: (v, v_x, v_xx) at (x, tau).
    Jet solution(double x, double tau) const;

    double price(double S, double tau) const;  ///< V(S, tau)
    double delta(double S, double tau) const;  ///< dV/dS (from autodiff)
    double gamma(double S, double tau) const;  ///< d2V/dS2 (from autodiff)

    /// Price given by the baseline alone (no network); 0 if there is no baseline.
    /// Useful to see how much the network actually contributes.
    double baseline_price(double S, double tau) const;

    const FractionalBS& problem() const { return p_; }
    const TrainConfig& config() const { return cfg_; }
    const L1Operator& l1() const { return op_; }
    MLP& net() { return net_; }
    const MLP& net() const { return net_; }

    /// Stratified random sample of n abscissae in (-L, L): one per equal sub-interval.
    std::vector<double> sample_x(std::mt19937_64& rng, int n) const;

private:
    Jet trial(double x, double tau, double bc_l, double bc_r, MLP::Workspace& ws) const;
    void inputs(double x, double tau, double* in, double* dir) const;
    /// Baseline w and its x-derivatives; tau_alpha = tau^alpha.
    BSJet baseline(double x, double tau_alpha) const;

    FractionalBS p_;
    TrainConfig cfg_;
    L1Operator op_;
    MLP net_;
    std::vector<double> bc_l_, bc_r_;  // boundary values on the time grid
    std::vector<double> res_w2_;       // squared residual weight per time level
    // Baseline only: fine time grid, its baseline times, index of each coarse level
    // in it, and the fine L1 rows belonging to the coarse levels.
    std::vector<double> clock_y_, clock_w_;  // random-clock quadrature nodes / weights
    std::vector<double> fine_tau_alpha_;
    std::vector<int> fine_idx_;
    std::vector<std::vector<double>> fine_rows_;
};

}  // namespace fpinn
