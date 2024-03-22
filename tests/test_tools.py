from qctools.tools import *
import numpy as np

def test_spherical_from_cartesian():
    """
    testing spherical_from_cartesian function
    """
    np.random.seed(42)
    x_lst = np.random.uniform(-1, 1, 10)
    y_lst = np.random.uniform(-1, 1, 10)
    z_lst = np.random.uniform(-1, 1, 10)

    res = []
    for x, y, z in zip(x_lst, y_lst, z_lst):
        res.append(spherical_from_cartesian(x, y, z))

    res = np.array(res) 
    
    # convert back to cartesian
    x_res = res[:, 0] * np.sin(res[:, 1]) * np.cos(res[:, 2])
    y_res = res[:, 0] * np.sin(res[:, 1]) * np.sin(res[:, 2])
    z_res = res[:, 0] * np.cos(res[:, 1])

    np.testing.assert_allclose(x_res, x_lst, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(y_res, y_lst, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(z_res, z_lst, atol=1e-5, rtol=1e-5)
