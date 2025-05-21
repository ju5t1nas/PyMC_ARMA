import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pymc.tensor as pt
from pytensor.scan import scan

def arma_p_q_model(series, 
                   p=1, 
                   q=2, 
                   name=None,
                   plot_trace=False):
    """
    Fit an ARMA(p,q) model to the given time series data using PyMC3.

    Parameters:
    ----------
    series : array-like
        The time series data to fit the ARMA(p,q) model to.
    p : int
        The order of the autoregressive (AR) part of the model.
    q : int
        The order of the moving average (MA) part of the model.
    name : str, optional
        The name of the model. If None, a default name will be generated.
    plot_trace : bool, optional
        Whether to plot the trace of the posterior samples.

    Returns:
    -------
    model : pymc3.Model
        The fitted ARMA model.
    trace : arviz.InferenceData
        The posterior samples from the fitted model.
    """
    
    x = np.asarray(series).astype(float)
    x = x[~np.isnan(x)]
    T = len(x)

    if T < 1 + 2 + 30:
        return None  # Need at least 30 samples beyond p+q
    
    if name is None:
        name = f"arma_{p}_{q}_model"

    y_obs = x[max(p, q):]
    x_shared = pt.as_tensor_variable(x)

    with pm.Model() as model:
        # Parameter Priors
        #-----------------------------------------------
        # AR(p) coefficients: phi_1, phi_2, ..., phi_p
        phi = pm.TruncatedNormal("phi", mu=0.0, sigma=1.0, lower=-1, upper=1.99, shape=(p,))
        # MA(q) coefficients: theta_1, theta_2, ..., theta_q
        theta = pm.TruncatedNormal("theta", mu=0.0, sigma=1.0, lower=-1, upper=1.99, shape=(q,))
        # Error term: sigma
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        # Initial errors: eps_{t-q},..., eps_{t-1}
        eps_init = pm.Normal("eps_init", mu=0.0, sigma=0.1, shape=(q,))
        #eps_init = pt.zeros(q)  # Initialize to zero


        # Step function for the scan
        #------------------------------------------------
        def step(t_idx, eps_hist, phi, theta):            
            # We redefine the lagged observations in x to depent on p
            x_lag = x_shared[t_idx - p:t_idx][::-1] # note: we reverse the order so that the 1st element is the most recent lagged value

            ar_term = pt.sum(phi * x_lag)
            ma_term= pt.sum(theta * eps_hist[::-1]) # note: we reverse the order so that the 1st element is the most recent lagged value.

            mu_t = ar_term + ma_term
            eps_t = x_shared[t_idx] - mu_t

            # Shift errors left and append eps_t
            new_eps_hist = pt.set_subtensor(eps_hist[:-1], eps_hist[1:])
            new_eps_hist = pt.set_subtensor(new_eps_hist[-1], eps_t)

            return new_eps_hist, mu_t

        scan_indices = pt.arange(max(p, q), T)

        (eps_hist_seq, mu_seq), _ = scan(
            fn=step,
            sequences=scan_indices,
            outputs_info=[eps_init, None],
            non_sequences=[phi, theta]
        )

        # Observation equation
        #------------------------------------------------
        pm.Normal("obs", mu=mu_seq, sigma=sigma, observed=y_obs)


        # Sampling from the posterior
        #------------------------------------------------
        trace = pm.sample(
            700, tune=400, target_accept=0.9,
            return_inferencedata=True, discard_tuned_samples=True,
            chains=4, cores=4
        )

    if plot_trace:
        az.plot_trace(trace, figsize=(12, 8))
        plt.suptitle(f"Trace Plots for ARMA({p},{q}) Model Parameters", fontsize=14)
        plt.tight_layout()
        plt.legend(loc="upper right")
        plt.show()

    return model, trace