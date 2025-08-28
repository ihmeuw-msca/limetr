# LimeTr

[![PyPI](https://img.shields.io/pypi/v/limetr?color=purple)](https://pypi.org/project/limetr/)
![Python](https://img.shields.io/badge/python-3.10,_3.11,_3.12,_3.13-purple.svg)
[![License](https://img.shields.io/pypi/l/limetr?color=purple)](https://github.com/ihmeuw-msca/limetr/blob/main/LICENSE)
[![Version](https://img.shields.io/pypi/v/limetr?color=purple)](https://pypi.org/project/limetr)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ihmeuw-msca/limetr/python-build.yml?branch=main)](https://github.com/ihmeuw-msca/limetr/actions)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/limetr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=MAGENTA&left_text=Downloads)](https://pepy.tech/projects/limetr)


**LimeTr** (pronounced "lime tree") is a Python package for solving mixed effects models with linear random effects. The package provides robust regression capabilities through a technique called "trimming" to handle outliers and improve model stability.

## Features

- **Mixed Effects Models**: Fit linear mixed effects models with fixed and random effects
- **Robust Regression**: Built-in trimming functionality for outlier detection and robust estimation
- **Efficient Implementation**: Optimized algorithms for large-scale data analysis
- **Statistical Inference**: Comprehensive tools for model diagnostics and inference

## Installation

LimeTr requires Python 3.10 or higher. Install via pip:

```bash
pip install limetr>=0.2.0
```

### Development Installation

For developers, clone the repository and install in development mode:

```bash
git clone https://github.com/ihmeuw-msca/limetr.git
cd limetr
pip install -e ".[test,docs]"
```

## Quick Start

```python
import numpy as np
from limetr import LimeTr, Data
from limetr.variable import FeVariable, ReVariable

# Create sample data
n_obs = 100
n_fe = 2
n_re = 3

# Generate random data
y = np.random.normal(0, 1, n_obs)
fe_mat = np.random.normal(0, 1, (n_obs, n_fe))
re_mat = np.random.normal(0, 1, (n_obs, n_re))

# Create data and variable objects
data = Data(y)
fevar = FeVariable(fe_mat)
revar = ReVariable(re_mat, n_groups=10)

# Initialize and fit the model
model = LimeTr(data, fevar, revar, inlier_pct=0.9)
result = model.fit_model()

# Get parameter estimates
beta_est = result['beta']
gamma_est = result['gamma']
```

## Documentation

For detailed documentation, visit [the official documentation](https://limetr.readthedocs.io/).

## Requirements

- Python >= 3.10
- NumPy >= 1.25.1
- SciPy >= 1.11.1
- Matplotlib >= 3.7.2
- spmat >= 0.1.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use LimeTr in your research, please cite:

```bibtex
@software{limetr2024,
  title={LimeTr: Robust Linear Mixed Effects Models},
  author={IHME Math Sciences},
  year={2024},
  url={https://github.com/ihmeuw-msca/limetr}
}
```
