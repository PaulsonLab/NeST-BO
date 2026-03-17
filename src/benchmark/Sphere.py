#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:10:40 2026

@author: jontwt
"""
from botorch.test_functions.synthetic import SyntheticTestFunction
import torch
from torch import Tensor

class Sphere(SyntheticTestFunction):
    r"""Rosenbrock synthetic test function.

    d-dimensional function (usually evaluated on `[-5, 10]^d`):

        f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)

    f has one minimizer for its global minimum at `z_1 = (1, 1, ..., 1)` with
    `f(z_i) = 0.0`.
    """

    _optimal_value = 0.0

    def __init__(
        self,
        dim=None,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
            dtype: The dtype that is used for the bounds of the function.
        """
        self.dim = dim
        self.continuous_inds = list(range(dim))
        if bounds is None:
            bounds = [(-dim**2, dim**2) for _ in range(self.dim)]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, dtype=dtype)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # return torch.sum(
        #     100.0 * (X[..., 1:] - X[..., :-1].pow(2)).pow(2) + (X[..., :-1] - 1).pow(2),
        #     dim=-1,
        # )
        return torch.sum(X**2, dim=1) 