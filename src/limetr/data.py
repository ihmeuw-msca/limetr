"""
data
~~~~

Data module.
"""

import operator
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from limetr.utils import default_vec_factory


class Data:
    """
    Data containers observations and group information.

    Parameters
    ----------
    obs
        Observations. Assumed to be sorted by the group id.
    obs_se
        Standard deviations of observation. Default is one.
    group_sizes
        Number of observations for each group. Default is ``None``.
    weight
        Weights for each observation. Default is one.

    """

    def __init__(
        self,
        obs: ArrayLike,
        obs_se: ArrayLike = 1.0,
        group_sizes: Sequence[int] | None = None,
        weight: ArrayLike = 1.0,
    ) -> None:
        self.obs = obs
        self.obs_se = obs_se
        self.weight = weight
        self.group_sizes = group_sizes

    @property
    def num_obs(self) -> int:
        """Number of observations."""
        return self.obs.size

    @property
    def num_groups(self) -> int:
        """Number of groups."""
        return self.group_sizes.size

    obs = property(operator.attrgetter("_obs"))

    @obs.setter
    def obs(self, vec: ArrayLike) -> None:
        vec = np.asarray(vec)
        if any(np.isnan(vec)):
            raise ValueError("`obs` must not containing nan(s).")
        self._obs = vec

    obs_se = property(operator.attrgetter("_obs_se"))

    @obs_se.setter
    def obs_se(self, vec: ArrayLike) -> None:
        vec = default_vec_factory(vec, self.num_obs, vec_name="obs_se")
        if any(vec <= 0.0):
            raise ValueError("`obs_se` must be all positive.")
        self._obs_se = vec

    weight = property(operator.attrgetter("_weight"))

    @weight.setter
    def weight(self, vec: ArrayLike) -> None:
        vec = default_vec_factory(vec, self.num_obs, vec_name="weight")
        if any(vec < 0) or any(vec > 1):
            raise ValueError("`weight` must be all between 0 and 1.")
        self._weight = vec

    group_sizes = property(operator.attrgetter("_group_sizes"))

    @group_sizes.setter
    def group_sizes(self, vec: Sequence[int] | None) -> None:
        if vec is None:
            group_sizes = np.ones(self.num_obs, dtype=int)
        else:
            group_sizes = np.asarray(vec, dtype=int)
            if group_sizes.ndim != 1:
                raise TypeError("Use `None` or a vector to set `group_sizes`.")
        if (group_sizes <= 0.0).any():
            raise ValueError("`group_sizes` must be all positive.")
        if group_sizes.sum() != self.num_obs:
            raise ValueError("Sum of `group_size` must equal to `num_obs`.")
        self._group_sizes = group_sizes

    def __repr__(self) -> str:
        return f"Data(num_obs={self.num_obs}, num_groups={self.num_groups})"
