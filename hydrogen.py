import numpy as np
from scipy import special

def radial(r, n, l, Z=2):
    """
    Compute the radial part of the hydrogenic wavefunction.

    Parameters
    ----------
    r : float
        The radial distance from the nucleus.
    n : int
        The principal quantum number.
    l : int
        The azimuthal quantum number.
    Z : int
        The atomic number.

    Returns
    -------
    float
        The value of the radial wavefunction at r.
    """
    ## computing normalization constant
    norm = np.sqrt(((Z*2/n)**3)*(special.factorial(n-l-1)/(2*n*special.factorial(n+l))))
    ## computing exponent
    exp = np.exp(-Z*r/n)
    
    ## computing polynomial
    polynomial = (2*Z*r/n)**l
    
    ##computing Laguerre
    lag = special.assoc_laguerre(2*Z*r/n, n-l-1, 2*l+1)
    
    return norm*exp*polynomial*lag

def hydrogenic_wavefunction(r, theta, phi, n, l, m, Z=2):
    """
    Compute the hydrogenic wavefunction.

    Parameters
    ----------
    r : float
        The radial distance from the nucleus.
    theta : float
        The polar angle.
    phi : float
        The azimuthal angle.
    n : int
        The principal quantum number.
    l : int
        The angular momentum quantum number.
    m : int
        The magnetic quantum number.
    Z : int
        The atomic number.

    Returns
    -------
    float
        The value of the hydrogenic wavefunction at given r, theta, and phi.
    """
    return radial(r, n, l, Z)*special.sph_harm(m, l, phi, theta)