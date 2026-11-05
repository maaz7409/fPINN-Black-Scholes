// fd_bs: finite-difference solver for the time-fractional Black-Scholes equation.
// Prints how the option price changes with the fractional order alpha.

#include <cmath>
#include <cstdio>
#include <iostream>

#include "cli.hpp"
#include "fpinn/black_scholes.hpp"
#include "fpinn/fd_solver.hpp"

using namespace fpinn;

int main(int argc, char** argv) {
    try {
        Args args(argc, argv);
        if (args.help()) {
            std::cout << "Usage: fd_bs [--type call|put] [--alphas 0.5,0.7,0.9,1] [--K 100] [--r 0.05]\n"
                         "             [--sigma 0.2] [--T 1] [--L 1.5] [--nx 1200] [--nt 1000] [--grading 2]\n";
            return 0;
        }
        FractionalBS p;
        p.type = parse_option_type(args.get("type", "call"));
        p.K = args.get("K", 100.0);
        p.r = args.get("r", 0.05);
        p.sigma = args.get("sigma", 0.2);
        p.T = args.get("T", 1.0);
        p.L = args.get("L", 1.5);
        const std::vector<double> alphas = args.get_doubles("alphas", {0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
        FDConfig cfg;
        cfg.nx = args.get("nx", cfg.nx);
        cfg.nt = args.get("nt", cfg.nt);
        cfg.grading = args.get("grading", cfg.grading);
        args.finish();

        const double spots[] = {0.8 * p.K, 0.9 * p.K, p.K, 1.1 * p.K, 1.2 * p.K};
        std::printf("European %s, K=%g r=%g sigma=%g T=%g   (FD grid %d x %d)\n\n", to_string(p.type), p.K, p.r,
                    p.sigma, p.T, cfg.nx, cfg.nt);
        std::printf("%6s %12s", "alpha", "E_a(-rT^a)");
        for (double S : spots) std::printf("   V(S=%6.1f)", S);
        std::printf("\n");
        for (double a : alphas) {
            p.alpha = a;
            p.validate();
            FDSolver fd(p, cfg);
            fd.solve();
            std::printf("%6.2f %12.6f", a, p.discount(p.T));
            for (double S : spots) std::printf("   %11.5f", fd.price(S, p.T));
            std::printf("\n");
        }
        std::printf("%6s %12.6f", "BS", std::exp(-p.r * p.T));
        for (double S : spots) std::printf("   %11.5f", black_scholes(p.type, S, p.K, p.r, p.sigma, p.T).price);
        std::printf("   (closed form, alpha = 1)\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
