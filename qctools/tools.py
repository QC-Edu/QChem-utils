import numpy as np

def spherical_from_cartesian(x, y, z):
    """
    Convert cartesian coordinates to spherical coordinates.
    
    Parameters
    ----------
    x : float
        x-coordinate
    y : float
        y-coordinate
    z : float
        z-coordinate
        
    Returns
    -------
    r : float
        radial coordinate
    theta : float
        polar angle
    phi : float
        azimuthal angle
    """

    r = np.sqrt(x**2 + y**2 + z**2 )
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)

    return np.array([r, theta, phi])