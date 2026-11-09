#include <cmath>
#include <random>
#include <sstream>
#include <vector>

#include "check.hpp"
#include "fpinn/fpinn.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace fpinn;

int main() {
    for (double alpha : {1.0, 0.6}) {
        for (OptionType type : {OptionType::Call, OptionType::Put}) {
          for (Baseline base : {Baseline::None, Baseline::SubordinatedBS}) {
            FractionalBS p;
            p.alpha = alpha;
            p.type = type;
            TrainConfig cfg;
            cfg.hidden = {6, 5};
            cfg.nt = 7;
            cfg.baseline = base;
            cfg.baseline_substeps = 4;
            cfg.similarity_feature = (type == OptionType::Call);  // cover both input layouts
            FPINNSolver s(p, cfg);
            // Perturb biases away from zero.
            std::mt19937_64 rng(11);
            std::normal_distribution<double> nd(0.0, 0.3);
            for (double& w : s.net().params()) w += nd(rng);

            // Full loss gradient against central finite differences.
            const std::vector<double> xs = {-1.2, -0.3, 0.05, 0.6, 1.4};
            std::vector<double> grad;
            s.loss_and_grad(xs, &grad);
            double max_err = 0.0, max_g = 0.0;
            for (std::size_t k = 0; k < grad.size(); ++k) {
                double& w = s.net().params()[k];
                const double w0 = w, eps = 1e-6;
                w = w0 + eps;
                const double lp = s.loss_and_grad(xs, nullptr);
                w = w0 - eps;
                const double lm = s.loss_and_grad(xs, nullptr);
                w = w0;
                max_err = std::max(max_err, std::abs((lp - lm) / (2 * eps) - grad[k]));
                max_g = std::max(max_g, std::abs(grad[k]));
            }
            CHECK(max_g > 1e-4);
            CHECK_NEAR(max_err / max_g, 0.0, 1e-6);

            // Hard boundary conditions hold exactly; tau = 0 returns the payoff.
            for (double tau : {0.01, 0.5, 1.0}) {
                CHECK_NEAR(s.solution(-p.L, tau).v, p.bc_left(tau), 1e-13);
                CHECK_NEAR(s.solution(p.L, tau).v, p.bc_right(tau), 1e-13);
            }
            CHECK_NEAR(s.solution(0.3, 0.0).v, p.payoff(0.3), 0.0);
          }
        }
    }

    // Reproducibility: loss and gradient are bitwise independent of the thread count,
    // and two identical training runs give identical weights.
    {
        FractionalBS p;
        p.alpha = 0.7;
        TrainConfig cfg;
        cfg.hidden = {16, 16};
        cfg.nt = 10;
        cfg.nx = 37;
        cfg.adam_steps = 60;
        cfg.lbfgs_steps = 10;
        cfg.lbfgs_nx = 41;
        FPINNSolver s(p, cfg);
        std::mt19937_64 rng(9);
        const std::vector<double> xs = s.sample_x(rng, 53);
        std::vector<double> g_many, g_one;
        const double l_many = s.loss_and_grad(xs, &g_many);
#ifdef _OPENMP
        const int nt = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
        const double l_one = s.loss_and_grad(xs, &g_one);
#ifdef _OPENMP
        omp_set_num_threads(nt);
#endif
        CHECK(l_many == l_one);
        CHECK(g_many == g_one);

        FPINNSolver a(p, cfg), b(p, cfg);
        a.train(nullptr);
        b.train(nullptr);
        CHECK(a.net().params() == b.net().params());
    }

    // Smoke test: a short training run must reduce the loss substantially.
    {
        FractionalBS p;
        p.alpha = 0.7;
        TrainConfig cfg;
        cfg.hidden = {16, 16};
        cfg.nt = 10;
        cfg.nx = 32;
        cfg.adam_steps = 400;
        cfg.lbfgs_steps = 50;
        cfg.lbfgs_nx = 48;
        FPINNSolver s(p, cfg);
        std::mt19937_64 rng(5);
        const std::vector<double> xs = s.sample_x(rng, 64);
        const double before = s.loss_and_grad(xs, nullptr);
        s.train(nullptr);
        const double after = s.loss_and_grad(xs, nullptr);
        std::printf("smoke training: loss %.3e -> %.3e\n", before, after);
        CHECK(after < 0.05 * before);
    }

    return test_summary("test_fpinn");
}
