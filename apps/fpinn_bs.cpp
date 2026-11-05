// fpinn_bs: train a fractional physics-informed neural network for the
// time-fractional Black-Scholes equation and compare it with a finite-difference
// reference (and with the closed-form Black-Scholes price when alpha = 1).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "cli.hpp"
#include "fpinn/black_scholes.hpp"
#include "fpinn/fd_solver.hpp"
#include "fpinn/fpinn.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace fpinn;

namespace {

const char* kUsage = R"(Usage: fpinn_bs [options]

Model (time-fractional Black-Scholes, European option)
  --type call|put       option type                         [call]
  --alpha A             fractional order in (0, 1]          [0.8]
  --K K                 strike                              [100]
  --r R                 risk-free rate                      [0.05]
  --sigma S             volatility                          [0.2]
  --T T                 maturity in years                   [1]
  --L L                 domain half-width in ln(S/K)        [1.5]

fPINN
  --hidden 32,32,32     hidden layer widths                 [32,32,32]
  --nt N                L1 time steps                       [40]
  --grading G           time-grid grading (1 = uniform)     [1.5]
  --nx N                x collocation points per Adam step  [64]
  --adam-steps N        Adam iterations                     [6000]
  --lr LR               initial learning rate               [3e-3]
  --lr-final LR         final learning rate                 [3e-5]
  --lbfgs-steps N       L-BFGS iterations after Adam        [1000]
  --lbfgs-nx N          x collocation points for L-BFGS     [160]
  --scale-residuals 0|1 divide level-n residual by W(n,n)   [1]
  --time-power P        network time input (tau/T)^P        [0.5]
  --baseline bs|none    subordinated Black-Scholes baseline  [bs]
  --baseline-nodes Q    random-clock quadrature nodes        [3]
  --similarity 0|1      similarity-variable network input   [1]
  --similarity-floor D  floor delta in the similarity input [0.05]
  --seed N              random seed                         [1234]
  --log-every N         log interval                        [500]

Reference / output
  --fd-nx N             finite-difference space intervals   [1200]
  --fd-nt N             finite-difference time steps        [1000]
  --fd-grading G        finite-difference time grading      [2]
  --out DIR             output directory (CSV + model)      [results]
  --threads N           OpenMP threads (0 = default)        [0]
)";

struct Errors {
    double max_abs = 0.0, rel_l2 = 0.0;
};

template <class F, class G>
Errors compare(F approx, G exact, double lo, double hi, double tau, int n = 201) {
    Errors e;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        const double S = lo + (hi - lo) * i / (n - 1);
        const double a = approx(S, tau), b = exact(S, tau);
        e.max_abs = std::max(e.max_abs, std::abs(a - b));
        num += (a - b) * (a - b);
        den += b * b;
    }
    e.rel_l2 = std::sqrt(num / den);
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args(argc, argv);
        if (args.help()) {
            std::cout << kUsage;
            return 0;
        }
        FractionalBS p;
        p.type = parse_option_type(args.get("type", "call"));
        p.alpha = args.get("alpha", 0.8);
        p.K = args.get("K", 100.0);
        p.r = args.get("r", 0.05);
        p.sigma = args.get("sigma", 0.2);
        p.T = args.get("T", 1.0);
        p.L = args.get("L", 1.5);
        p.validate();

        TrainConfig cfg;
        cfg.hidden = args.get_ints("hidden", cfg.hidden);
        cfg.nt = args.get("nt", cfg.nt);
        cfg.grading = args.get("grading", cfg.grading);
        cfg.nx = args.get("nx", cfg.nx);
        cfg.adam_steps = args.get("adam-steps", cfg.adam_steps);
        cfg.lr = args.get("lr", cfg.lr);
        cfg.lr_final = args.get("lr-final", cfg.lr_final);
        cfg.lbfgs_steps = args.get("lbfgs-steps", cfg.lbfgs_steps);
        cfg.lbfgs_nx = args.get("lbfgs-nx", cfg.lbfgs_nx);
        cfg.scale_residuals = args.get("scale-residuals", 1) != 0;
        cfg.time_power = args.get("time-power", cfg.time_power);
        cfg.baseline = parse_baseline(args.get("baseline", "bs"));
        cfg.baseline_nodes = args.get("baseline-nodes", cfg.baseline_nodes);
        cfg.similarity_feature = args.get("similarity", 1) != 0;
        cfg.similarity_floor = args.get("similarity-floor", cfg.similarity_floor);
        cfg.seed = static_cast<std::uint64_t>(args.get("seed", 1234));
        cfg.log_every = std::max(1, args.get("log-every", cfg.log_every));

        FDConfig fdc;
        fdc.nx = args.get("fd-nx", fdc.nx);
        fdc.nt = args.get("fd-nt", fdc.nt);
        fdc.grading = args.get("fd-grading", fdc.grading);
        const std::string out = args.get("out", "results");
        const int threads = args.get("threads", 0);
        args.finish();
#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
        const int nthreads = omp_get_max_threads();
#else
        (void)threads;
        const int nthreads = 1;
#endif

        std::printf("Time-fractional Black-Scholes, European %s\n", to_string(p.type));
        std::printf("  K=%g r=%g sigma=%g T=%g alpha=%g  domain S in [%.2f, %.2f]  threads=%d\n", p.K, p.r,
                    p.sigma, p.T, p.alpha, p.S_of(-p.L), p.S_of(p.L), nthreads);
        std::printf("  fractional discount E_alpha(-r T^alpha) = %.6f  (classical exp(-rT) = %.6f)\n\n",
                    p.discount(p.T), std::exp(-p.r * p.T));

        // 1. Reference solution.
        FDSolver fd(p, fdc);
        fd.solve();
        const bool classical = (p.alpha == 1.0);
        const double lo = 0.5 * p.K, hi = 1.5 * p.K;
        auto bs_price = [&](double S, double tau) {
            return black_scholes(p.type, S, p.K, p.r, p.sigma, tau).price;
        };
        auto fd_price = [&](double S, double tau) { return fd.price(S, tau); };
        if (classical) {
            const Errors e = compare(fd_price, bs_price, lo, hi, p.T);
            std::printf("FD reference vs closed form (alpha=1): max|err| = %.2e\n\n", e.max_abs);
        }

        // 2. Train the fPINN.
        FPINNSolver pinn(p, cfg);
        auto pinn_price = [&](double S, double tau) { return pinn.price(S, tau); };
        auto monitor = [&] {
            const Errors e = compare(pinn_price, fd_price, lo, hi, p.T, 51);
            std::ostringstream ss;
            ss.precision(3);
            ss << "max|V-V_fd|(tau=T) " << std::scientific << e.max_abs;
            return ss.str();
        };
        pinn.train(&std::cout, monitor);

        // 3. Report.
        std::printf("\nPrices at tau = T (S in [%.0f, %.0f]):\n", lo, hi);
        std::printf("%8s %12s %12s %10s %10s %11s %11s%s\n", "S", "V_fPINN", "V_FD", "|diff|", "delta", "gamma",
                    "gamma_FD", classical ? "   V_BS(exact)" : "");
        for (double S : {60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0}) {
            const double s = S / 100.0 * p.K;
            const double v = pinn.price(s, p.T), vf = fd.price(s, p.T);
            std::printf("%8.2f %12.5f %12.5f %10.2e %10.5f %11.6f %11.6f", s, v, vf, std::abs(v - vf),
                        pinn.delta(s, p.T), pinn.gamma(s, p.T), fd.gamma(s, p.T));
            if (classical) std::printf("   %12.5f", bs_price(s, p.T));
            std::printf("\n");
        }

        std::printf("\nError summary (S in [%.0f, %.0f]):\n", lo, hi);
        for (double frac : {0.25, 0.5, 0.75, 1.0}) {
            const double tau = frac * p.T;
            const Errors ef = compare(pinn_price, fd_price, lo, hi, tau);
            std::printf("  tau=%.2f  fPINN vs FD: max|err| %.3e  rel-L2 %.3e", tau, ef.max_abs, ef.rel_l2);
            if (classical) {
                const Errors eb = compare(pinn_price, bs_price, lo, hi, tau);
                std::printf("   | fPINN vs exact BS: max|err| %.3e  rel-L2 %.3e", eb.max_abs, eb.rel_l2);
            }
            std::printf("\n");
        }
        if (cfg.baseline != Baseline::None) {
            auto base = [&](double S, double tau) { return pinn.baseline_price(S, tau); };
            const Errors e = compare(base, fd_price, lo, hi, p.T);
            std::printf("  (baseline formula alone, no network, tau=T: max|err| %.3e  rel-L2 %.3e)\n", e.max_abs,
                        e.rel_l2);
        }

        // 4. Write results.
        std::filesystem::create_directories(out);
        {
            std::ofstream f(out + "/prices_at_T.csv");
            f << "S,V_fpinn,V_fd,delta_fpinn,delta_fd,gamma_fpinn,gamma_fd" << (classical ? ",V_bs" : "") << "\n";
            for (int i = 0; i <= 200; ++i) {
                const double S = lo + (hi - lo) * i / 200.0;
                f << S << ',' << pinn.price(S, p.T) << ',' << fd.price(S, p.T) << ',' << pinn.delta(S, p.T) << ','
                  << fd.delta(S, p.T) << ',' << pinn.gamma(S, p.T) << ',' << fd.gamma(S, p.T);
                if (classical) f << ',' << bs_price(S, p.T);
                f << '\n';
            }
        }
        {
            std::ofstream f(out + "/surface.csv");
            f << "tau,S,V_fpinn,V_fd\n";
            for (int it = 0; it <= 20; ++it) {
                const double tau = p.T * it / 20.0;
                for (int i = 0; i <= 50; ++i) {
                    const double S = lo + (hi - lo) * i / 50.0;
                    f << tau << ',' << S << ',' << pinn.price(S, tau) << ',' << fd.price(S, tau) << '\n';
                }
            }
        }
        {
            std::ofstream f(out + "/fpinn.model");
            pinn.net().save(f);
        }
        std::printf("\nWrote %s/prices_at_T.csv, %s/surface.csv, %s/fpinn.model\n", out.c_str(), out.c_str(),
                    out.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n(run with --help for usage)\n", e.what());
        return 1;
    }
}
