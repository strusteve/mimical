import numpy as np
import time


def polynomial(unit_cube, param_prior_dist, poly_order, wavs):
    """Sample polynomial values and fit the coefficients."""

    # Define equally spaced wavelength nodes.
    nodes = np.linspace(wavs[0], wavs[-1], poly_order + 1)

    # Sample polynomial values at each node.
    values = (unit_cube[:poly_order+1] *
              (param_prior_dist[1] - param_prior_dist[0]) +
              param_prior_dist[0])

    # Fit polynomial and return coefficients.
    theta_curr = np.polynomial.polynomial.polyfit(nodes, values, poly_order)

    return theta_curr


def powerlaw(unit_cube, param_prior_dist, wavs, powerbounds):
    """Sample power-law endpoint values and calculate coefficients."""

    # Sample exponent.
    a_2 = (unit_cube[2] *
           (powerbounds[1] - powerbounds[0]) +
           powerbounds[0])

    # Sample endpoint values.
    y_min = (unit_cube[0] *
             (param_prior_dist[1] - param_prior_dist[0]) +
             param_prior_dist[0])

    y_max = (unit_cube[1] *
             (param_prior_dist[1] - param_prior_dist[0]) +
             param_prior_dist[0])

    # Calculate coefficients.
    a_1 = (y_max - y_min) / (wavs[-1]**a_2 - wavs[0]**a_2)
    a_0 = y_min - a_1 * wavs[0]**a_2

    return np.array([a_0, a_1, a_2])
