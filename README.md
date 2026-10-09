# NeST-BO: Fast Local Bayesian Optimization via Newton-Step Targeting of Gradient and Hessian Information

This repository contains the code for **NeST-BO** and **NeST-BO-sub**, the algorithms proposed in the paper
[_NeST-BO: Fast Local Bayesian Optimization via Newton-Step Targeting of Gradient and Hessian Information_](https://arxiv.org/abs/2510.05516) (AISTATS 2026).

## Method overview

NeST-BO is a local Bayesian optimization method that takes Newton steps on a Gaussian process (GP) surrogate. Each iteration:

1. Fits a GP with a squared-exponential (ARD) kernel to all data collected so far.
2. **Inner loop.** It greedily selects `M` points inside a box of half-width `delta` around the current iterate `x_t`. Each point maximizes the NeST acquisition function, which is the reduction in the summed posterior variance of the gradient and the Hessian at `x_t`.
3. Evaluates the objective at these points, then refits the GP.
4. **Outer step.** It computes the GP's predicted gradient and Hessian at `x_t`:
   - If the Hessian is positive definite, it takes a Newton step with Armijo backtracking on the GP mean.
   - Otherwise, it takes a normalized gradient step, also with Armijo backtracking.

**NeST-BO-sub** runs the same procedure in a low-dimensional random subspace. It uses BAxUS-style nested embeddings, and the subspace dimension grows after 10 consecutive iterations without improvement. Use it for high-dimensional problems.

All problems are formulated as **minimization**.

See [How the code works](#how-the-code-works) for a step-by-step description of the implementation.

## Installation

The code was developed with Python 3.11.

```sh
git clone https://github.com/PaulsonLab/NeST-BO.git
cd NeST-BO
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch, BoTorch, GPyTorch, Hydra, Gymnasium, and LassoBench (from GitHub). That is enough for the synthetic and Leukemia benchmarks. Some benchmarks need extra packages:

| Benchmarks | Extra dependency | Install |
| --- | --- | --- |
| `lunar` | Box2D | `pip install "gymnasium[box2d]==1.1.0"` (requires `swig`) |
| `pusher` | Box2D and pygame | `pip install "gymnasium[box2d]==1.1.0" pygame` |
| `swimmer`, `ant` | MuJoCo | `pip install "gymnasium[mujoco]==1.1.0"` |

## Running experiments

All experiments run through `main.py`. You must choose a benchmark. The `algorithm` setting selects `nestbo` (NeST-BO, the default) or `nestbo_sub` (NeST-BO-sub):

```sh
python main.py benchmark=<benchmark_name> algorithm=<nestbo|nestbo_sub> seed=<seed>
```

For example:

```sh
python main.py benchmark=ackley seed=0                             # NeST-BO
python main.py benchmark=ackley_dummy algorithm=nestbo_sub seed=0  # NeST-BO-sub
```

Running `python main.py` without `benchmark=` prints the list of available benchmarks.

### Overriding settings

The project uses [Hydra](https://hydra.cc/). Any value in [configs/default.yaml](configs/default.yaml) or in the benchmark config can be overridden on the command line without editing files:

```sh
# Larger budget, different dimension and inner-loop batch size
python main.py benchmark=ackley seed=1 benchmark.n_tot=1000 benchmark.dim=50 benchmark.M=50

# Scale the fallback gradient step by the GP lengthscales
python main.py benchmark=ackley gd_lengthscale_scaling=true
```

To run several seeds or both algorithms in one call, use Hydra's multirun mode:

```sh
python main.py -m benchmark=rover algorithm=nestbo,nestbo_sub seed=0,1,2,3,4
```

### Output

Each run creates a folder `outputs/<timestamp>/` with the Hydra log and the resolved config. The log ends with the best objective value found (`min obj value: ...`). A progress bar shows the best value during the run.

Both loops return the full data from `exec_alg()`: the inputs `X` (scaled to the unit box, or to the subspace for NeST-BO-sub), the objective values `Y`, and the best-so-far history. The history has `n_tot` entries: the best value of the initial design, followed by one entry per evaluation after it. `main.py` does not save them to disk. Add your own `torch.save(...)` in `main.py` if you need them.

## Benchmarks

Each benchmark is defined by a file in [configs/benchmark/](configs/benchmark/). Inputs are optimized in `[0, 1]^dim` (or `[-1, 1]` in the NeST-BO-sub subspace) and rescaled to `[lb, ub]` before each evaluation.

| Benchmark | Description | `dim` | Budget after initial design (`n_tot`) | Algorithm |
| --- | --- | --- | --- | --- |
| `ackley` | Ackley function | 20 | 800 | `nestbo` |
| `griewank` | Griewank function | 20 | 500 | `nestbo` |
| `sphere` | Sphere function | 20 | 500 | `nestbo` |
| `ackley_dummy` | Ackley, 30 active dimensions out of 1000 | 1000 | 200 | `nestbo_sub` |
| `griewank_dummy` | Griewank, 30 active dimensions out of 1000 | 1000 | 200 | `nestbo_sub` |
| `rosenbrock_dummy` | Rosenbrock, 30 active dimensions out of 1000 | 1000 | 200 | `nestbo_sub` |
| `rover` | Rover trajectory planning | 60 | 800 | both |
| `pusher` | Robot pushing, two robots (Box2D) | 14 | 300 | both |
| `lunar` | Lunar Lander-v3 controller (Gymnasium) | 12 | 300 | both |
| `swimmer` | Swimmer-v5 linear policy (MuJoCo) | 16 | 300 | both |
| `ant` | Ant-v4 linear policy (MuJoCo) | 888 | 300 | both |
| `leukemia` | Weighted Lasso hyperparameter tuning on the Leukemia dataset (LassoBench) | 7129 | 300 | both |

The algorithm column follows from the config contents. NeST-BO needs `M`, and NeST-BO-sub needs `target_dim_init`. A benchmark with only one of these fields runs only with the matching algorithm. To use the other algorithm, add the missing field with Hydra's `+` prefix, e.g. `python main.py benchmark=ackley_dummy +benchmark.M=20`.

The RL, rover, and pusher rewards are negated, so lower values are better.

### Configuration parameters

| Key | Meaning |
| --- | --- |
| `n_tot` | Evaluation budget after the initial design. The run makes `n_tot − 1` evaluations after the initial design, because the initial design counts as the first entry of the best-so-far history |
| `N_init` | Number of initial Sobol points. The starting point is evaluated in addition to these, except with `start_point=best_sobol`. The initial design does not count toward `n_tot` |
| `dim` | Input dimension of the objective |
| `lb`, `ub` | Lower and upper bounds of the search space (scalar or per dimension) |
| `delta` | Half-width of the local box for the inner-loop acquisition search (in unit-box coordinates) |
| `M` | Number of inner-loop points per iteration (NeST-BO only; NeST-BO-sub uses the current subspace dimension) |
| `target_dim_init` | Minimum initial subspace dimension (NeST-BO-sub only) |
| `dim_true` | Number of active dimensions in the `*_dummy` problems |
| `start_point` | How the starting point is chosen: `random`, `center`, `init`, or `best_sobol` (see [Starting point](#starting-point)) |
| `init_point` | The starting point in the original bounds `[lb, ub]`, used when `start_point=init` |
| `fn` | Objective to instantiate (Hydra `_target_`) |

Global settings in [configs/default.yaml](configs/default.yaml):

| Key | Default | Meaning |
| --- | --- | --- |
| `algorithm` | `nestbo` | `nestbo` (NeST-BO) or `nestbo_sub` (NeST-BO-sub) |
| `seed` | `0` | Random seed for the initial design and starting point |
| `device` | `"cpu"` | Torch device |
| `gd_lengthscale_scaling` | `false` | Multiply the normalized gradient fallback step by the GP lengthscales |

### Starting point

Each benchmark config sets `start_point`, and you can override it on the command line:

| `start_point` | Starting point |
| --- | --- |
| `random` | A uniform random point in the search space, determined by `seed` |
| `center` | The center of `[lb, ub]` |
| `init` | The point given in `init_point`, in the original bounds `[lb, ub]` |
| `best_sobol` | The best of the `N_init` initial Sobol points; no extra evaluation is spent |

In total, a run makes `N_init + n_tot` evaluations, or `N_init + n_tot − 1` with `best_sobol`.

```sh
python main.py benchmark=sphere benchmark.start_point=best_sobol
python main.py benchmark=ackley benchmark.dim=3 benchmark.M=3 benchmark.start_point=init "benchmark.init_point=[1.0,-2.0,0.5]"
```

For NeST-BO-sub, all options work in the subspace. `random` and `best_sobol` draw their points there, and `center` maps exactly to the center of the domain. A point given with `init` usually cannot be represented exactly in the initial subspace. NeST-BO-sub then starts from its closest point in the subspace (least-squares projection) and logs a warning with the size of the deviation.

### Adding a benchmark

1. Write a callable that takes an `(n, dim)` tensor in the original bounds and returns `n` objective values to be minimized. See [src/benchmark/Sphere.py](src/benchmark/Sphere.py).
2. Add `configs/benchmark/<name>.yaml` with the keys above, setting `fn._target_` to your callable.
3. Run `python main.py benchmark=<name>`.

## How the code works

This section follows one run of [src/optimization_loop_NeSTBO.py](src/optimization_loop_NeSTBO.py) in order. NeST-BO-sub ([src/optimization_loop_NeSTBO_sub.py](src/optimization_loop_NeSTBO_sub.py)) runs the same loop in a subspace. Its differences are listed [at the end](#nest-bo-sub-differences).

### Coordinates

The optimizer works in the unit box `[0,1]^dim`. A point `u` is mapped to `x = lb + (ub − lb)·u` only when the objective is evaluated. The local box `delta`, the GP lengthscales and all steps are therefore in unit-box coordinates.

### Initial design

Setup evaluates `N_init` scrambled Sobol points plus the starting point (see [Starting point](#starting-point)). These points form the initial training data.

### Each iteration

Each iteration runs four steps: build the GP, select `M` points with the acquisition function, refit the GP, and take one step. It costs `M + 1` evaluations. The run stops when the budget `n_tot` is used, which can happen in the middle of an iteration.

#### 1. Build the GP

A new `DerivativeExactGPSEModel` ([src/model.py](src/model.py)) is built from all data collected so far:

- **Prior:** zero mean and an ARD squared-exponential kernel with an output scale (`ScaleKernel(RBFKernel)`). The lengthscales start at 0.6931.
  - NeST-BO constrains the lengthscales to `[0.005, 10]`.
  - NeST-BO-sub uses GPyTorch's default constraint.
- **Likelihood:** Gaussian, with GPyTorch's default noise floor of `1e-4`.
- **Targets:** standardized to zero mean and unit variance whenever there is more than one point.
- **Hyperparameters:** fitted by maximizing the exact marginal log-likelihood with BoTorch's `fit_gpytorch_mll`. If fitting fails, the loop prints `cant fit GP` and continues with the current hyperparameters.

The model computes the GP's posterior mean gradient and Hessian at the iterate `x_t` analytically, from the derivatives of the SE kernel:

- `posterior_derivative(x)` returns `∂k(x, X) K⁻¹ y`.
- `posterior_hessian(x)` returns `∂²k(x, X) K⁻¹ y`.

Here `K = k(X, X) + σ²I`.

#### 2. Select inner-loop points (acquisition)

`NewtonInformation` in [src/Acquisition_NeSTBO.py](src/Acquisition_NeSTBO.py) scores a candidate `z` by how much one observation at `z` would reduce the GP's uncertainty about the gradient and Hessian at `x_t`. This is the paper's acquisition with `ŝ = 1`. For the lookahead data `X' = X ∪ {z}` with `K' = k(X', X') + σ²I`, the value is

```
α(z) = Σ_i  ∂_i k(x_t, X') K'⁻¹ ∂_i k(X', x_t)              (gradient term)
     + Σ_ij ∂²_ij k(x_t, X') K'⁻¹ ∂²_ij k(X', x_t)          (Hessian term)
```

How it is computed:
- **Equivalent to the paper's form:** this is the reduction in `tr Σ_g + tr Σ_H`, the total posterior variance of the gradient and the Hessian entries. The prior variances do not depend on `z`, so they are omitted.
- **Kernel derivatives:** computed analytically.
- **Inverse:** `K'⁻¹` comes from a Cholesky factorization.
- **Gradient with respect to `z`:** computed by autograd for the optimizer.
- **Independent of the objective values:** the acquisition depends only on input locations, so the points can be chosen before they are evaluated.

`optimize_acqf_custom_bo` maximizes `α` with BoTorch's `optimize_acqf`:
- **Bounds:** the local box `[x_t − delta, x_t + delta]`, clipped to the domain.
- **Settings:** `q = 1`, 20 raw samples to pick 5 restarts, and L-BFGS-B with at most 300 iterations.

The `M` points are selected **greedily**. After each point is chosen, it is evaluated and added to the GP's training data without refitting the hyperparameters. The next point is then selected against the updated data.

#### 3. Refit the GP

After the inner loop, the hyperparameters are refitted on all data. The loop then computes the posterior mean gradient `g` and Hessian `H` at `x_t`.

#### 4. Move: Newton step or gradient step

| Condition | Direction | Initial step `s` | Shrink factor `β` |
| --- | --- | --- | --- |
| All eigenvalues of `H` are > 0 | Newton: `d = −H⁻¹ g` | 1.0 | 0.8 (0.5 in NeST-BO-sub) |
| Otherwise | Normalized gradient: `d = −g / ‖g‖`, multiplied by the GP lengthscales if `gd_lengthscale_scaling=true` | 0.5 | 0.5 |

**Line search.** Both steps use Armijo backtracking on the GP posterior mean `μ`, with `σ = 0.1`. A step size `s` is accepted when

```
μ(x_t) − μ(x_t + s·d) ≥ −σ · s · dᵀg
```

If the condition fails, the step size shrinks as `s ← β·s`. The search stops once the condition holds or `s ≤ 0.05`. The step `x_t + s·d` is then taken even if the condition was never met.

The new iterate is clipped to the domain and evaluated. It becomes `x_{t+1}`, and its evaluation is added to the data.

### NeST-BO-sub differences

- **Subspace.** The search runs in a subspace `v ∈ [-1,1]^d` with a sparse embedding matrix `S` (`d × dim`) in the style of BAxUS:
  - Each input dimension is assigned to one subspace coordinate, with a random sign.
  - A point maps to the unit box as `u = (v·S + 1)/2`.
- **Initial subspace dimension.** It is set by the BAxUS rule, but is at least `target_dim_init`.
- **Inner-loop size.** `M` equals the current subspace dimension `d`.
- **Growing the subspace.** After each iteration, the best value from that iteration's evaluations is compared with the best value so far.
  - After 10 iterations in a row without improvement, each subspace coordinate is split into up to 3 coordinates.
  - The existing data and the current iterate are lifted into the larger subspace, so no evaluations are lost.
  - The next GP is built in the new dimension.

## Repository structure

```
main.py                              # Hydra entry point (algorithm=nestbo or nestbo_sub)
configs/
  default.yaml                       # global settings
  benchmark/*.yaml                   # one config per benchmark
src/
  Acquisition_NeSTBO.py              # NeST acquisition function and its optimizer
  model.py                           # GP model with posterior gradient and Hessian
  optimization_loop_NeSTBO.py        # NeST-BO loop
  optimization_loop_NeSTBO_sub.py    # NeST-BO-sub loop (BAxUS-style embeddings)
  benchmark/                         # benchmark objectives
```

## Citation

If you use this code in your research, please cite:

```
@article{tang2025nest,
  title={NeST-BO: Fast Local Bayesian Optimization via Newton-Step Targeting of Gradient and Hessian Information},
  author={Tang, Wei-Ting and Kudva, Akshay and Paulson, Joel A},
  journal={arXiv preprint arXiv:2510.05516},
  year={2025}
}
```

## License

See [LICENSE](LICENSE).
