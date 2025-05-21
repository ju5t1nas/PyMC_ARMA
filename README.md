# PyMC_ARMA - Generalized ARMA(p,q) Model with PyMC and PyTensor
This repository provides a simple implementation of a generalized Auto Regressive Moving Average ARMA(p,q) time series model using PyMC and PyTensor's `scan` function.
The provided function enables fitting ARMA models of arbitrary order in a Bayesian setting.


## Why this Repository?

Implementing generalized ARMA models with PyMC is tricky due to the recursive nature of the moving-average (MA) component (and lack of easily accessibly documentation for PyMC, etc.). 

This repository offers a clean, reusable function that handles ARMA(p,q) models robustly and intuitively. Of course, you might need to mess with the priors (or what not) based on your specific needs. Be my guest! ;-)

## Dependencies

Install the following packages to use the provided function:

```bash
pip install pymc arviz numpy matplotlib
```
Hopefully I did not forget anything.

## Quick-Start

The simples way of using the function is probably this.

```Python
from arma_p_q_model import arma_p_q_model
import numpy as np

# Example time series data
data = np.random.randn(200)

# Fit ARMA(1,2) model
model, trace = arma_p_q_model(data, p=1, q=2, plot_trace=True)
```

## Good luck! <3