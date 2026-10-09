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

Both loops return the full data from `exec_alg()`: the inputs `X` (scaled to the unit box, or to the subspace for NeST-BO-sub), the objective values `Y`, and the best-so-far history. `main.py` does not save them to disk. Add your own `torch.save(...)` in `main.py` if you need them.

## Benchmarks

Each benchmark is defined by a file in [configs/benchmark/](configs/benchmark/). Inputs are optimized in `[0, 1]^dim` (or `[-1, 1]` in the NeST-BO-sub subspace) and rescaled to `[lb, ub]` before each evaluation.

| Benchmark | Description | `dim` | Budget (`n_tot`) | Algorithm |
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
| `n_tot` | Total number of function evaluations, including the initial points |
| `N_init` | Number of initial Sobol points. The starting point is evaluated in addition to these, except with `start_point=best_sobol` |
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

```sh
python main.py benchmark=sphere benchmark.start_point=best_sobol
python main.py benchmark=sphere benchmark.dim=3 benchmark.M=3 benchmark.start_point=init "benchmark.init_point=[100.0,-50.0,0.0]"
```

For NeST-BO-sub, all options work in the subspace. `random` and `best_sobol` draw their points there, and `center` maps exactly to the center of the domain. A point given with `init` usually cannot be represented exactly in the initial subspace. NeST-BO-sub then starts from its closest point in the subspace (least-squares projection) and logs a warning with the size of the deviation.

### Adding a benchmark

1. Write a callable that takes an `(n, dim)` tensor in the original bounds and returns `n` objective values to be minimized. See [src/benchmark/Sphere.py](src/benchmark/Sphere.py).
2. Add `configs/benchmark/<name>.yaml` with the keys above, setting `fn._target_` to your callable.
3. Run `python main.py benchmark=<name>`.

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
