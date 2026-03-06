"""
LimeTr: Robust Linear Mixed Effects Models

A Python package for solving mixed effects models with linear random effects
and robust regression capabilities through trimming.
"""

from .core import LimeTr
from .data import Data
from .variable import FeVariable, ReVariable, Variable
from .visual import funnel
from .stats import (
    Prior,
    GaussianPrior,
    UniformPrior,
    LinearPrior,
    LinearGaussianPrior,
    LinearUniformPrior
)
from .linalg import SmoothMapping, LinearMapping

__all__ = [
    "LimeTr",
    "Data", 
    "FeVariable",
    "ReVariable",
    "Variable",
    "funnel",
    "Prior",
    "GaussianPrior",
    "UniformPrior", 
    "LinearPrior",
    "LinearGaussianPrior",
    "LinearUniformPrior",
    "SmoothMapping",
    "LinearMapping"
]