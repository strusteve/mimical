import numpy as np


def individual(unit_cube, param_prior_dist):
    return ((unit_cube * (param_prior_dist[1]-param_prior_dist[0])) +
            param_prior_dist[0])


def polynomial(unit_cube, param_prior_dist, poly_order, wavs):
    """ Automatically sample the polynomial coefficient priors under the
    condition that the polyniomial starts and ends within the user specified
    bounds. """
    # If user specifies 'Polynomial', each coefficient is a free parameter.
    # e.g., For order 0, only one free parameter is included a.k.a constant.
    # e.g., For order 1, two free parameters a.k.a straight-line relationship.
    # The lowest wavelength is chosen as the origin.

    theta_curr = np.zeros(poly_order+1)

    # Set the prior for the y-intercept first.
    theta_curr[0] = ((unit_cube[0] * (param_prior_dist[1]-param_prior_dist[0]))
                     + param_prior_dist[0])

    # Define a random order for which to loop through higher order coefficients
    random_order = np.append(0, np.random.choice(np.arange(poly_order),
                                                 size=poly_order,
                                                 replace=False)+1)

    # Calculate the conditional priors for higher order polynomial coefficients
    # based on the sum of the lower order components.
    for i in range(1, len(random_order)):

        # Load the previously sampled coefficients
        prev_coeffs = theta_curr[random_order[:i]]

        # Calculate the previous order wavelength multipliers
        prev_polywavs = np.pow(wavs[-1]-wavs[0], random_order[:i])

        # Calculate sum of previous polynomial components
        prev_comps = prev_coeffs * prev_polywavs
        prev_comps_summed = np.sum(prev_comps)

        # Define the current coefficient bounds based on the sum of the lower
        # order components and current wavelength multiplier.
        min = (param_prior_dist[0] - prev_comps_summed) /\
            (np.pow(wavs[-1]-wavs[0], random_order[i]))
        max = (param_prior_dist[1] - prev_comps_summed) /\
            (np.pow(wavs[-1]-wavs[0], random_order[i]))

        # Set the new coefficient prior sample
        theta_curr[random_order[i]] = (unit_cube[random_order[i]] *
                                       (max-min)) + min

    return theta_curr


def powerlaw(unit_cube, param_prior_dist, wavs, powerbounds, epsilon):
    """ Automatically sample the power-law coefficients under the condition
    that the power-law starts and ends within the user specified bounds. """

    theta_curr = np.zeros(3)

    # Sample the power across the user specified prior
    theta_curr[2] = (unit_cube[2] *
                     (powerbounds[1]-(powerbounds[0])) +
                     powerbounds[0])

    # If power is positive, simply sample the remaining coefficients which set
    # the parameter values at the lowest/highest wavelength.
    if theta_curr[2] >= 0:
        theta_curr[1] = (unit_cube[1] *
                         (param_prior_dist[1]-param_prior_dist[0]) +
                         param_prior_dist[0])
        theta_curr[0] = (unit_cube[0] *
                         (param_prior_dist[1]-param_prior_dist[0]) +
                         param_prior_dist[0])

    # If power is negative, set the 'b' coefficient, then conditionally sample
    # the 'a' coefficient under the condition that the lowest wavelength param
    # is within bounds.
    else:
        theta_curr[1] = (unit_cube[1] *
                         (param_prior_dist[1]-param_prior_dist[0]) +
                         param_prior_dist[0])

        mincept = (param_prior_dist[0] -
                   (theta_curr[1] * ((epsilon/((wavs[-1]-wavs[0])+epsilon)) **
                                     theta_curr[2]))) /\
                  (1 - ((epsilon/((wavs[-1]-wavs[0])+epsilon)) **
                        theta_curr[2]))
        maxcept = (param_prior_dist[1] -
                   (theta_curr[1] * ((epsilon/((wavs[-1]-wavs[0])+epsilon)) **
                                     theta_curr[2]))) /\
                  (1 - ((epsilon/((wavs[-1]-wavs[0])+epsilon)) **
                        theta_curr[2]))
        theta_curr[0] = (unit_cube[0] * (maxcept-mincept) + mincept)

    return theta_curr
