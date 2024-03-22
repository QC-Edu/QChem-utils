from qctools.hydrogen import *
import numpy as np

def test_radial():
    """
    testing radial function
    """
    np.random.seed(42)
    r_lst = np.random.uniform(0, 20, 10)
    n_lst = np.random.randint(1, 4, 10)
    l_lst = np.random.randint(0, 2, 10)
    z_lst = np.random.randint(1, 4, 10)

    res = []
    for r, n, l, z in zip(r_lst, n_lst, l_lst, z_lst):
        res.append(radial(r, n, l, z))

    true_answer = np.array([-0.0004960033300780658, 
                            0.0, 
                            -1.196249898221174e-05, 
                            -9.880164133461835e-07, 
                            -0.1871895195464458, 
                            -0.12547826943024146, 
                            0.3185578727068723, 
                            0.0, 
                            5.634522373742176e-07, 
                            0.0024315433516219594])
    
    np.testing.assert_allclose(res, true_answer, atol=1e-5, rtol=1e-3)


def test_hydrogenic_wavefunction():
    """
    testing hydrogenic_wavefunction function
    """
    np.random.seed(2)
    theta_lst = np.random.uniform(0, 1, 10)
    phi_lst = np.random.uniform(-1, 1, 10)
    l_lst = np.random.randint(0, 10, 10)
    m_lst = np.random.randint(0, 5, 10)
    n_lst = np.random.randint(1, 10, 10)
    r_lst = np.random.uniform(0, 2, 10)

    res = []
    for theta, phi, l, n, r in zip(theta_lst, phi_lst, l_lst, n_lst, r_lst):
        res.append(hydrogenic_wavefunction(r, theta, phi, n, l, m=0))

    true_answer = np.array([0.+0.j, 0.+0.j, 0.+0.j, -0.+0.j, 0.00661248+0.j, 0.02022248+0.j, 0.+0.j, 0.0049428+0.j, 0.00040025+0.j, 0.01440961+0.j])
    
    np.testing.assert_allclose(res, true_answer, atol=1e-5, rtol=1e-3)

def test_calc_energy():
    """
    testing calc_energy function
    """
    E_h = calc_energy(1, Z=1, units='hartree')
    E_h_true = -0.5
    np.testing.assert_allclose(E_h, E_h_true, rtol=1e-5, atol=1e-5)

    E_ev = calc_energy(1, Z=1, units='eV')
    E_ev_true = -13.605693009
    np.testing.assert_allclose(E_ev, E_ev_true, rtol=1e-5, atol=1e-5)

    E_h = calc_energy(2, Z=1, units='SI')
    E_h_true = -2.18e-18
    np.testing.assert_allclose(E_h, E_h_true, rtol=1e-5, atol=1e-5)