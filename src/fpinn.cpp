#include "fpinn/fpinn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>

#include "fpinn/black_scholes.hpp"
#include "fpinn/optim.hpp"
#include "fpinn/special_functions.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fpinn {

namespace {
std::vector<int> layer_sizes(int inputs, const std::vector<int>& hidden) {
    std::vector<int> s{inputs};
    s.insert(s.end(), hidden.begin(), hidden.end());
    s.push_back(1);
    return s;
}
}  // namespace

Baseline parse_baseline(const std::string& s) {
    if (s == "none") return Baseline::None;
    if (s == "bs" || s == "subordinated-bs") return Baseline::SubordinatedBS;
    throw std::invalid_argument("unknown baseline '" + s + "' (use none or bs)");
}

const char* to_string(Baseline b) { return b == Baseline::None ? "none" : "subordinated-bs"; }

BSJet FPINNSolver::baseline(double x, double tau_alpha) const {
    BSJet w{0.0, 0.0, 0.0};
    for (std::size_t q = 0; q < clock_y_.size(); ++q) {
        const BSJet b = black_scholes_x(p_.type, x, p_.r, p_.sigma, tau_alpha * clock_y_[q]);
        w.v += clock_w_[q] * b.v;
        w.vx += clock_w_[q] * b.vx;
        w.vxx += clock_w_[q] * b.vxx;
    }
    return w;
}

// Network inputs and their x-derivative (all inputs are affine in x for fixed tau):
//   in[0] = x / L                                  (space)
//   in[1] = 2 (tau/T)^p - 1                        (time)
//   in[2] = (x / L) / ((tau/T)^(alpha/2) + delta)  (similarity variable, optional)
// The similarity variable stretches the region around the strike at small tau,
// where the solution is a smoothed version of the payoff kink of width ~ tau^(alpha/2).
void FPINNSolver::inputs(double x, double tau, double* in, double* dir) const {
    const double L = p_.L;
    in[0] = x / L;
    dir[0] = 1.0 / L;
    in[1] = 2.0 * std::pow(tau / p_.T, cfg_.time_power) - 1.0;
    dir[1] = 0.0;
    if (cfg_.similarity_feature) {
        const double c = 1.0 / (std::pow(tau / p_.T, 0.5 * p_.alpha) + cfg_.similarity_floor);
        in[2] = c * x / L;
        dir[2] = c / L;
    }
}

FPINNSolver::FPINNSolver(const FractionalBS& problem, const TrainConfig& cfg)
    : p_(problem),
      cfg_(cfg),
      op_(problem.alpha, make_time_grid(problem.T, cfg.nt, cfg.grading)),
      net_(layer_sizes(cfg.similarity_feature ? 3 : 2, cfg.hidden)) {
    p_.validate();
    if (cfg_.nx < 1) throw std::invalid_argument("TrainConfig: nx must be >= 1");
    if (cfg_.lbfgs_steps > 0 && cfg_.lbfgs_nx < 1) throw std::invalid_argument("TrainConfig: lbfgs_nx must be >= 1");
    if (cfg_.log_every < 1) throw std::invalid_argument("TrainConfig: log_every must be >= 1");
    if (!(cfg_.time_power > 0.0)) throw std::invalid_argument("TrainConfig: time_power must be > 0");
    net_.init_xavier(cfg_.seed);
    const std::vector<double>& tau = op_.grid();
    res_w2_.assign(tau.size(), 1.0);
    if (cfg_.scale_residuals) {
        for (int n = 1; n <= op_.steps(); ++n) {
            const double s = 1.0 / op_.weight(n, n);
            res_w2_[n] = s * s;
        }
    }
    bc_l_.resize(tau.size());
    bc_r_.resize(tau.size());
    for (std::size_t j = 0; j < tau.size(); ++j) {
        bc_l_[j] = p_.bc_left(tau[j]);
        bc_r_[j] = p_.bc_right(tau[j]);
    }

    if (cfg_.baseline != Baseline::None) {
        random_clock_gauss_rule(p_.alpha, cfg_.baseline_nodes, clock_y_, clock_w_);
        // Fine grid containing every coarse level: the first coarse step is split
        // into 4*sub graded pieces (the baseline behaves like tau^(alpha/2) there),
        // every other coarse step into sub equal pieces.
        const int sub = std::max(1, cfg_.baseline_substeps);
        const int M = op_.steps();
        std::vector<double> ft{0.0};
        fine_idx_.assign(M + 1, 0);
        const int k1 = 4 * sub;
        for (int k = 1; k <= k1; ++k) ft.push_back(tau[1] * std::pow(static_cast<double>(k) / k1, 3.0));
        ft.back() = tau[1];
        fine_idx_[1] = static_cast<int>(ft.size()) - 1;
        for (int j = 2; j <= M; ++j) {
            for (int k = 1; k <= sub; ++k) ft.push_back(tau[j - 1] + (tau[j] - tau[j - 1]) * k / sub);
            ft.back() = tau[j];
            fine_idx_[j] = static_cast<int>(ft.size()) - 1;
        }
        const L1Operator fine(p_.alpha, ft);
        fine_rows_.assign(M + 1, {});
        for (int n = 1; n <= M; ++n) fine_rows_[n] = fine.row(fine_idx_[n]);
        fine_tau_alpha_.resize(ft.size());
        for (std::size_t k = 0; k < ft.size(); ++k) fine_tau_alpha_[k] = std::pow(ft[k], p_.alpha);
    }
}

// u = w + A + B N with
//   w(x,tau) = baseline (or 0)
//   A(x,tau) = ((L - x) cl + (x + L) cr) / (2L),  cl/cr = boundary values of u - w  (A_xx = 0)
//   B(x)     = 1 - (x/L)^2                        (B' = -2x/L^2, B'' = -2/L^2)
Jet FPINNSolver::trial(double x, double tau, double bc_l, double bc_r, MLP::Workspace& ws) const {
    const double L = p_.L;
    double in[3], dir[3];
    inputs(x, tau, in, dir);
    const Jet n = net_.forward(in, dir, ws);
    BSJet w{0.0, 0.0, 0.0};
    if (cfg_.baseline != Baseline::None) {
        const double ta = std::pow(tau, p_.alpha);
        w = baseline(x, ta);
        bc_l -= baseline(-L, ta).v;
        bc_r -= baseline(L, ta).v;
    }
    const double A = ((L - x) * bc_l + (x + L) * bc_r) / (2.0 * L);
    const double Ax = (bc_r - bc_l) / (2.0 * L);
    const double B = 1.0 - (x / L) * (x / L);
    const double Bx = -2.0 * x / (L * L);
    const double Bxx = -2.0 / (L * L);
    return {w.v + A + B * n.v, w.vx + Ax + Bx * n.v + B * n.d, w.vxx + Bxx * n.v + 2.0 * Bx * n.d + B * n.dd};
}

std::vector<double> FPINNSolver::sample_x(std::mt19937_64& rng, int n) const {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> xs(n);
    const double h = 2.0 * p_.L / n;
    for (int i = 0; i < n; ++i) {
        double u = U(rng);
        u = std::clamp(u, 1e-6, 1.0 - 1e-6);  // keep strictly inside the domain
        xs[i] = -p_.L + (i + u) * h;
    }
    return xs;
}

double FPINNSolver::loss_and_grad(const std::vector<double>& xs, std::vector<double>* grad) const {
    const int N = static_cast<int>(xs.size());
    const int M = op_.steps();
    const int S = M + 1;  // stride per x point
    const std::vector<double>& tau = op_.grid();
    const double a = p_.diffusion(), b = p_.drift(), r = p_.r;

    const bool use_baseline = cfg_.baseline != Baseline::None;
    std::vector<double> R(static_cast<std::size_t>(N) * S, 0.0);
    // Per-point partial sums, added up afterwards in a fixed order: floating-point
    // addition is not associative, so an OpenMP reduction would make training
    // depend on thread timing and thread count.  This way it is bitwise reproducible.
    std::vector<double> loss_i(N, 0.0);

    // ---- Pass 1: evaluate the trial solution and the residuals. ----
#pragma omp parallel
    {
        MLP::Workspace ws = net_.make_workspace();
        std::vector<double> u(S), ux(S), uxx(S), z(S), wf(fine_tau_alpha_.size());
#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            double* res = R.data() + static_cast<std::size_t>(i) * S;
            if (use_baseline) {
                for (std::size_t k = 0; k < wf.size(); ++k) {
                    wf[k] = baseline(xs[i], fine_tau_alpha_[k]).v;
                }
            }
            // z = part of u that goes through the coarse L1 history.
            // Without baseline z = u and z_0 = payoff (initial condition imposed exactly).
            // With baseline z = u - w and z_0 = 0 (w(x,0) is the payoff).
            u[0] = p_.payoff(xs[i]);
            z[0] = use_baseline ? 0.0 : u[0];
            for (int j = 1; j <= M; ++j) {
                const Jet t = trial(xs[i], tau[j], bc_l_[j], bc_r_[j], ws);
                u[j] = t.v;
                ux[j] = t.d;
                uxx[j] = t.dd;
                z[j] = use_baseline ? u[j] - wf[fine_idx_[j]] : u[j];
            }
            for (int n = 1; n <= M; ++n) {
                double dw = 0.0;  // D^alpha w on the fine grid
                if (use_baseline) {
                    const std::vector<double>& row = fine_rows_[n];
                    for (std::size_t k = 0; k < row.size(); ++k) dw += row[k] * wf[k];
                }
                res[n] = dw + op_.apply(n, z.data()) - (a * uxx[n] + b * ux[n] - r * u[n]);
                loss_i[i] += res_w2_[n] * res[n] * res[n];
            }
        }
    }
    double loss = 0.0;
    for (double l : loss_i) loss += l;
    const double scale = 1.0 / (static_cast<double>(N) * M);
    loss *= scale;
    if (grad == nullptr) return loss;

    // ---- Pass 2: back-propagate. ----
    // R_n = sum_j W(n,j) u_j - (a u_xx,n + b u_x,n - r u_n),  loss = scale * sum_n w_n^2 R_n^2.
    // Store rho_n = w_n^2 R_n so that d(loss)/dR_n = 2 scale rho_n.
    for (int i = 0; i < N; ++i) {
        double* res = R.data() + static_cast<std::size_t>(i) * S;
        for (int n = 1; n <= M; ++n) res[n] *= res_w2_[n];
    }
    // One gradient slot per collocation point (N * num_params doubles, a few MB),
    // summed in index order below for the same reproducibility reason as loss_i.
    const std::size_t P = net_.num_params();
    std::vector<double> G(static_cast<std::size_t>(N) * P, 0.0);
#pragma omp parallel
    {
        MLP::Workspace ws = net_.make_workspace();
        const double L = p_.L;
#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            const double* res = R.data() + static_cast<std::size_t>(i) * S;
            double* g_i = G.data() + static_cast<std::size_t>(i) * P;
            const double x = xs[i];
            const double B = 1.0 - (x / L) * (x / L);
            const double Bx = -2.0 * x / (L * L);
            const double Bxx = -2.0 / (L * L);
            for (int j = 1; j <= M; ++j) {
                // Adjoints of u_j, u_x,j, u_xx,j.
                double hist = 0.0;
                for (int n = j; n <= M; ++n) hist += res[n] * op_.weight(n, j);
                const double gu = 2.0 * scale * (hist + r * res[j]);
                const double gux = -2.0 * scale * b * res[j];
                const double guxx = -2.0 * scale * a * res[j];
                // Chain rule through u = A + B N.
                const Jet seed{B * gu + Bx * gux + Bxx * guxx, B * gux + 2.0 * Bx * guxx, B * guxx};
                double in[3], dir[3];
                inputs(x, tau[j], in, dir);
                net_.forward(in, dir, ws);
                net_.backward(seed, ws, g_i);
            }
        }
    }
    grad->assign(P, 0.0);
    double* gout = grad->data();
    const long long Pll = static_cast<long long>(P);
#pragma omp parallel for schedule(static)
    for (long long k = 0; k < Pll; ++k) {
        double s = 0.0;
        for (int i = 0; i < N; ++i) s += G[static_cast<std::size_t>(i) * P + k];
        gout[k] = s;
    }
    return loss;
}

void FPINNSolver::train(std::ostream* log, const Monitor& monitor) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(clock::now() - t0).count(); };

    std::mt19937_64 rng(cfg_.seed + 1);
    std::vector<double>& params = net_.params();
    std::vector<double> grad(params.size());

    if (log) {
        *log << "Training fPINN: " << net_.num_params() << " parameters, " << cfg_.nx << " x-points x "
             << cfg_.nt << " time levels per step\n";
    }

    // Stage 1: Adam with re-sampled collocation points and geometric learning-rate decay.
    Adam adam(params.size());
    for (int step = 0; step < cfg_.adam_steps; ++step) {
        const double frac = cfg_.adam_steps > 1 ? static_cast<double>(step) / (cfg_.adam_steps - 1) : 0.0;
        const double lr = cfg_.lr * std::pow(cfg_.lr_final / cfg_.lr, frac);
        const std::vector<double> xs = sample_x(rng, cfg_.nx);
        const double loss = loss_and_grad(xs, &grad);
        if (!std::isfinite(loss)) throw std::runtime_error("training diverged (loss is not finite)");
        adam.step(params, grad, lr);
        if (log && (step % cfg_.log_every == 0 || step == cfg_.adam_steps - 1)) {
            *log << "  adam  " << std::setw(6) << step << "  loss " << std::scientific << std::setprecision(3)
                 << loss << "  lr " << lr << std::defaultfloat << std::fixed << std::setprecision(1) << "  t "
                 << elapsed() << "s" << std::defaultfloat;
            if (monitor) *log << "  " << monitor();
            *log << '\n' << std::flush;
        }
    }

    // Stage 2: L-BFGS on a fixed, denser set of collocation points (full-batch, deterministic).
    if (cfg_.lbfgs_steps > 0) {
        const std::vector<double> xs = sample_x(rng, cfg_.lbfgs_nx);
        Objective f = [&](const std::vector<double>& w, std::vector<double>& g) {
            params = w;
            return loss_and_grad(xs, &g);
        };
        std::vector<double> w = params;
        LBFGSConfig lc;
        lc.max_iter = cfg_.lbfgs_steps;
        lbfgs_minimize(f, w, lc, [&](int it, double loss) {
            if (log && (it % std::max(1, cfg_.log_every / 5) == 0 || it == cfg_.lbfgs_steps)) {
                params = w;
                *log << "  lbfgs " << std::setw(6) << it << "  loss " << std::scientific << std::setprecision(3)
                     << loss << std::defaultfloat << std::fixed << std::setprecision(1) << "  t " << elapsed()
                     << "s" << std::defaultfloat;
                if (monitor) *log << "  " << monitor();
                *log << '\n' << std::flush;
            }
        });
        params = w;
    }
    if (log) *log << "Training finished in " << std::fixed << std::setprecision(1) << elapsed() << " s\n"
                  << std::defaultfloat;
}

Jet FPINNSolver::solution(double x, double tau) const {
    if (tau <= 0.0) {  // the payoff itself (derivatives in the a.e. sense)
        const bool call = p_.type == OptionType::Call;
        const double ex = std::exp(x);
        const bool itm = call ? x > 0.0 : x < 0.0;
        if (!itm) return {0.0, 0.0, 0.0};
        return call ? Jet{ex - 1.0, ex, ex} : Jet{1.0 - ex, -ex, -ex};
    }
    MLP::Workspace ws = net_.make_workspace();
    return trial(x, tau, p_.bc_left(tau), p_.bc_right(tau), ws);
}

double FPINNSolver::baseline_price(double S, double tau) const {
    if (cfg_.baseline == Baseline::None) return 0.0;
    return p_.K * baseline(p_.x_of(S), std::pow(std::max(tau, 0.0), p_.alpha)).v;
}

double FPINNSolver::price(double S, double tau) const { return p_.K * solution(p_.x_of(S), tau).v; }

double FPINNSolver::delta(double S, double tau) const {
    const double x = p_.x_of(S);
    return solution(x, tau).d * std::exp(-x);
}

double FPINNSolver::gamma(double S, double tau) const {
    const double x = p_.x_of(S);
    const Jet j = solution(x, tau);
    return (j.dd - j.d) * std::exp(-2.0 * x) / p_.K;
}

}  // namespace fpinn
