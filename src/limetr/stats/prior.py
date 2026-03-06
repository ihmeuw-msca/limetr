"""
Prior Module
"""

from collections.abc import Iterable
from numbers import Number
from typing import Any

import numpy as np
from numpy.typing import NDArray

from limetr.utils import broadcast, get_maxlen


class Prior:
    """
    Generic prior class, need to be inherited.

    Parameters
    ----------
    info
        Information of the prior. Length of the list is number of
        information components. Each component can be either scalar or a
        vector. If there are more than one vector with size more than one,
        the size of the vectors need to match. By default ``None``.
    size
        Size of the prior, by default 0.

    """

    def __init__(self, info: list[Any] | None = None, size: int = 0):
        size = max(int(size), get_maxlen(info))
        info = broadcast(info, size)
        self.info = info
        self.size = size

    @property
    def is_empty(self) -> bool:
        """
        Returns
        -------
        bool
            If prior has zero size.

        """
        return self.size == 0

    # pylint:disable=unused-argument
    # pylint:disable=no-self-use
    def objective(self, var: NDArray) -> float:
        """
        Objective function for optimization interface.

        Parameters
        ----------
        var
            Variable that prior is acting on.

        Returns
        -------
        float
            Objective value regarding the log likelihood of the prior.

        """
        return 0.0

    def gradient(self, var: NDArray) -> NDArray:
        """
        Gradient function for optimization interface.

        Parameters
        ----------
        var
            Variable that prior is acting on.

        Returns
        -------
        NDArray
            Gradient value regarding the log likelihood of the prior.

        """
        return np.zeros(len(var))

    def hessian(self, var: NDArray) -> NDArray:
        """
        Hessian function for optimization interface.

        Parameters
        ----------
        var
            Variable that prior is acting on.

        Returns
        -------
        NDArray
            Hessian value regarding the log likelihood of the prior.

        """
        return np.zeros((len(var), len(var)))

    def __repr__(self) -> str:
        return f"Prior(size={self.size})"


class GaussianPrior(Prior):
    """
    Gaussian Prior

    Parameters
    ----------
    mean
        Mean of the Gaussian prior, by default 0.
    sd
        Standard deviation of the Gaussian prior, by default inf.
    size
        Size of the prior, by default 0.

    Raises
    ------
    ValueError
        If any standard deviations are less or equal to zero.

    """

    def __init__(
        self,
        mean: Number | Iterable = 0.0,
        sd: Number | Iterable = np.inf,
        size: int = 0,
    ):
        super().__init__([mean, sd], size=size)
        if not all(self.info[1] > 0):
            raise ValueError("Standard deviations have to be positive numbers.")
        self.mean = self.info[0]
        self.sd = self.info[1]

    def objective(self, var: NDArray) -> float:
        return 0.5 * np.sum((var - self.mean) ** 2 / self.sd**2)

    def gradient(self, var: NDArray) -> NDArray:
        return (var - self.mean) / self.sd**2

    # pylint: disable=unused-argument
    def hessian(self, var: NDArray) -> NDArray:
        return np.diag(1 / self.sd**2)

    def __repr__(self) -> str:
        return f"GaussianPrior(mean={self.mean}, sd={self.sd})"


class UniformPrior(Prior):
    """
    Uniform Prior

    Parameters
    ----------
    lb
        Lower bounds of the prior, by default -inf.
    ub
        Upper bounds of the prior, by default inf.
    size
        Size of the prior, by default 0.

    Raises
    ------
    ValueError
        If any lower bounds are greater than upper bounds.

    """

    def __init__(
        self,
        lb: Number | Iterable = -np.inf,
        ub: Number | Iterable = np.inf,
        size: int = 0,
    ):
        super().__init__([lb, ub], size=size)
        if any(self.info[0] > self.info[1]):
            raise ValueError(
                "Lower bounds must be less or equal than upper bounds."
            )
        self.lb = self.info[0]
        self.ub = self.info[1]

    def __repr__(self) -> str:
        return f"UniformPrior(lb={self.lb}, ub={self.ub})"


class LinearPrior(Prior):
    """
    Linear Prior

    Parameters
    ----------
    mat
        Linear mapping (matrix).
    info
        Information array of the prior.

    """

    def __init__(self, mat: Iterable, info: list[Any]):
        mat = np.asarray(mat)
        if mat.ndim != 2:
            raise ValueError("`mat` has to be a matrix.")
        Prior.__init__(self, info, mat.shape[0])
        self.mat = mat

    def __repr__(self) -> str:
        return f"LinearPrior(shape={self.mat.shape})"


class LinearGaussianPrior(LinearPrior, GaussianPrior):
    """
    Linear Gaussian Prior

    Parameters
    ----------
    mat
        Linear mapping (matrix).
    mean
        Mean of the prior, by default 0.0.
    sd
        Standard deviation of the prior, by default inf.

    """

    def __init__(
        self,
        mat: Iterable,
        mean: Number | Iterable = 0.0,
        sd: Number | Iterable = np.inf,
    ):
        LinearPrior.__init__(self, mat, [mean, sd])
        GaussianPrior.__init__(self, self.info[0], self.info[1], size=self.size)

    def objective(self, var: NDArray) -> float:
        trans_var = self.mat.dot(var)
        return super().objective(trans_var)

    def gradient(self, var: NDArray) -> NDArray:
        trans_var = self.mat.dot(var)
        return self.mat.T.dot(super().gradient(trans_var))

    def hessian(self, var: NDArray) -> NDArray:
        trans_var = self.mat.dot(var)
        return self.mat.T.dot(super().hessian(trans_var).dot(self.mat))

    def __repr__(self) -> str:
        return f"LinearGaussianPrior(mean={self.mean}, sd={self.sd}, shape={self.mat.shape})"


class LinearUniformPrior(LinearPrior, UniformPrior):
    """
    Linear Uniform Prior

    Parameters
    ----------
    mat
        Linear mapping (matrix).
    lb
        Lower bounds of the prior, by default -inf.
    ub
        Upper bounds of the prior, by default inf.

    """

    def __init__(
        self,
        mat: Iterable,
        lb: Number | Iterable = -np.inf,
        ub: Number | Iterable = np.inf,
    ):
        LinearPrior.__init__(self, mat, [lb, ub])
        UniformPrior.__init__(self, self.info[0], self.info[1], size=self.size)

    def __repr__(self) -> str:
        return f"LinearUniformPrior(lb={self.lb}, ub={self.ub}, shape={self.mat.shape})"
