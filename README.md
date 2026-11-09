# fPINN-Black-Scholes

A **fractional physics-informed neural network (fPINN)** for pricing European options under the
**time-fractional Black–Scholes model**, written from scratch in C++17. There is no ML framework and no GPU required.

```
D_τ^α V = ½ σ² S² V_SS + r S V_S − r V ,    V(S, 0) = payoff(S),    0 < α ≤ 1
```

`D_τ^α` is the Caputo derivative in time-to-maturity `τ`. It gives the price a memory of its past; `α = 1` is classical Black–Scholes.

## How it works

* **Network:** a tanh MLP `N(x, τ)` in log-moneyness `x = ln(S/K)`.
  * Hand-written forward-mode (Taylor) differentiation gives exact `N_x`, `N_xx`.
  * A matching reverse pass gives the exact gradient of the loss with respect to every weight.
* **fPINN idea** (Pang, Lu & Karniadakis 2019):
  * integer-order derivatives come from automatic differentiation;
  * the non-local Caputo derivative is discretised with the **L1 scheme** on a graded time grid.
* **Constraints:**
  * boundary conditions are built into the trial solution `v = A + B·N`;
  * the initial condition (payoff) is inserted exactly into the L1 memory sum.
* **Loss:** mean squared PDE residual, each time level scaled by its L1 diagonal weight.
* **Optimisers:** Adam with re-sampled collocation points, followed by L-BFGS. OpenMP-parallel and reproducible.
* **Outputs:** price, Delta and Gamma from the network's exact derivatives.
* **Optional `--baseline bs`:** a Gauss-quadrature approximation of the model's "random clock" (subordination) representation. The network then learns only a correction.
  The program prints the baseline-alone error too: with 3 nodes the formula alone is already about as accurate as the network.

## Validation

* A finite-difference reference solver (L1 in time, central differences in space).
* It is checked against the closed-form Black–Scholes price (α = 1, error ≈ 0.002) and an exact subordination formula for α = ½ (error ≈ 0.0003).
* Fractional put–call parity holds.
* The unit tests check every hand-written gradient against finite differences.

Default run (call, K = 100, r = 5 %, σ = 20 %, T = 1, 32-32-32 network, 40 time levels; about 75 s on 4 threads, 40 s on 8).
Errors are maximum absolute price errors for S ∈ [50, 150] at τ = T:

| α | fPINN vs reference | rel. L2 |
|---|---|---|
| 0.5 | 0.047 | 0.10 % |
| 0.8 | 0.069 | 0.15 % |
| 1.0 (vs exact BS) | 0.21 | 0.44 % |

With `--baseline bs` (α = 0.8) the error drops to 0.020, but the baseline formula alone already reaches 0.019.

Training is deterministic: the loss and gradient are summed in a fixed order, so a given seed gives
bit-identical weights for any thread count (on the same compiler and maths library).

## Build & run

Requires a C++17 compiler and CMake ≥ 3.16; OpenMP is optional.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
(cd build && ctest --output-on-failure)

./build/fpinn_bs --alpha 0.8                # train, compare with FD, write results/*.csv
./build/fpinn_bs --alpha 1 --type put       # classical put vs the exact formula
./build/fd_bs                               # price vs alpha (finite differences)
./build/fpinn_bs --help                     # all options
```

`fpinn_bs` writes:

* `results/prices_at_T.csv`: price, Delta and Gamma from the fPINN and from FD;
* `results/surface.csv`: the price surface over (τ, S);
* `results/fpinn.model`: the trained weights.

## Layout

| path | contents |
|---|---|
| `include/fpinn/problem.hpp` | model parameters, payoff, exact far-field boundary values (`E_α(−rτ^α)` discounting) |
| `include/fpinn/caputo.hpp` | graded time grid, L1 operator |
| `include/fpinn/mlp.hpp` | MLP with Taylor-mode forward and reverse pass |
| `include/fpinn/fpinn.hpp` | `TrainConfig`, `FPINNSolver` (loss, gradient, training, pricing) |
| `include/fpinn/optim.hpp` | Adam, L-BFGS |
| `include/fpinn/fd_solver.hpp` | finite-difference reference solver |
| `include/fpinn/black_scholes.hpp` | closed-form BS, exact α = ½ price |
| `include/fpinn/special_functions.hpp` | normal CDF, Mittag-Leffler function, random-clock Gauss rule |
| `apps/` | `fpinn_bs`, `fd_bs` |
| `tests/` | unit tests (self-contained, run with `ctest`) |

## Why no CUDA?

The network has about 2 300 weights and a training step evaluates about 2 500 points: far too little work to benefit from a GPU.
The CPU version trains in about a minute.

The per-point forward/backward pass is a self-contained function on small arrays. That makes it easy to move into a CUDA kernel if the problem is scaled up, for example to multi-asset models.

## References

* W. Wyss, *The fractional Black–Scholes equation*, FCAA 3 (2000).
* G. Pang, L. Lu, G. E. Karniadakis, *fPINNs: Fractional physics-informed neural networks*, SIAM J. Sci. Comput. 41 (2019).
* M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-informed neural networks*, J. Comput. Phys. 378 (2019).
* M. Stynes, E. O'Riordan, J. L. Gracia, *Error analysis of a finite difference method on graded meshes for a time-fractional diffusion equation*, SIAM J. Numer. Anal. 55 (2017).
