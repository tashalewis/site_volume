import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import fsolve
from scipy.signal import argrelextrema

def get_coex_1(vm, P):
    # create a grid of per particle volumes and the corresponding pressures
    vm_test = np.linspace(min(vm), max(vm), len(P))
    vm_grid, P_grid = np.meshgrid(vm_test, P)

    # Find intersection points
    P_back = P_grid[:, 2:] - P[:-2]
    P_fwd = P_grid[:, :-2] - P[2:]
    P_sign = P_fwd * P_back

    # Label intersection points
    intersection = P_sign < 0
    count = np.zeros(vm_grid.shape)[:, 1:-1]
    count[intersection] += 1

    coex_vm = []
    coex_P = []
    # Test all possible coexistence pressures
    for i in range(vm_grid.shape[0]):
        vm_grid_i = vm_grid[i, 1:-1]
        coex_vm_i = vm_grid_i[intersection[i, :]]
        # Potential coexistence pressures should intersect the pressure curve 3 times
        if len(coex_vm_i) > 2:
            # Record the potential coexistence per particle volumes
            vml = min(coex_vm_i)
            vmg = max(coex_vm_i)
            relevant_P = P[(vm > vml) & (vm < vmg)]
            # Make sure the per particle volumes are not the same point
            if len(relevant_P[relevant_P > P[i]]) > 0 and len(relevant_P[relevant_P < P[i]]) > 0:
                # Store the potential coexistence pressure and per particle volumes
                coex_vm.append((min(coex_vm_i), max(coex_vm_i)))
                coex_P.append(P[i])

    res = []
    # Perform the maxwell construction on the potential coexistence pressures
    for vm_pair, P_coex in zip(coex_vm, coex_P):
        vm_int = vm[(vm >= vm_pair[0]) & (vm <= vm_pair[1])]
        maxwell_int = np.abs(np.sum(P[(vm >= vm_pair[0]) & (vm <= vm_pair[1])] - P_coex) * np.diff(vm_int)[0])
        res.append((vm_pair[0], vm_pair[1], P_coex, maxwell_int))

    # Return the coexistence point with the smallest value of the maxwell integral (P - P_coex) dv
    res = sorted(res, key=lambda x: x[3])
    if len(res) > 0:
        return res[0]
    else:
        return None

def get_coex_2(rho, mu):
    # create a grid of densities and the corresponding chemical potentials
    rho_test = np.linspace(min(rho), max(rho), len(mu))
    rho_grid, mu_grid = np.meshgrid(rho_test, mu)

    # Find intersection points
    mu_back = mu_grid[:, 2:] - mu[:-2]
    mu_fwd = mu_grid[:, :-2] - mu[2:]
    mu_sign = mu_fwd * mu_back

    # Label intersection points
    intersection = mu_sign < 0
    count = np.zeros(rho_grid.shape)[:, 1:-1]
    count[intersection] += 1

    coex_rho = []
    coex_mu = []
    # Test all possible coexistence chemical potentials
    for i in range(rho_grid.shape[0]):
        rho_grid_i = rho_grid[i, 1:-1]
        coex_rho_i = rho_grid_i[intersection[i, :]]
        # Potential coexistence chemical potentials should intersect the chemical potential curve 3 times
        if len(coex_rho_i) > 2:
            # Record the potential coexistence densities
            rhol = max(coex_rho_i)
            rhog = min(coex_rho_i)
            relevant_mu = mu[(rho > rhog) & (rho < rhol)]
            # Make sure the per particle volumes are not the same point
            if len(relevant_mu[relevant_mu > mu[i]]) > 0 and len(relevant_mu[relevant_mu < mu[i]]) > 0:
                # Store the potential coexistence chemical potential and densities
                coex_rho.append((min(coex_rho_i), max(coex_rho_i)))
                coex_mu.append(mu[i])

    res = []
    # Perform the maxwell construction on the potential coexistence chemical potentiala
    for rho_pair, mu_coex in zip(coex_rho, coex_mu):
        rho_int = rho[(rho >= rho_pair[0]) & (rho <= rho_pair[1])]
        maxwell_int = np.abs(np.sum(mu[(rho >= rho_pair[0]) & (rho <= rho_pair[1])] - mu_coex) * np.diff(rho_int)[0])
        res.append((rho_pair[0], rho_pair[1], mu_coex, maxwell_int))

    # Return the coexistence point with the smallest value of the maxwell integral (mu - mu_coex) drho
    res = sorted(res, key=lambda x: x[3])
    if len(res) > 0:
        return res[0]
    else:
        return None

def get_coex_3(rho, P, mu):
    # Define the function for fsolve
    def func(x):
        # Quadratic penalty for unphysical densities
        diff0 = 0
        diff1 = 0
        if x[0] < min(rho):
            diff0 = np.abs(x[0] - min(rho))**2
        if x[0] > max(rho):
            diff0 = np.abs(x[0] - max(rho))**2
        if x[1] < min(rho):
            diff1 = np.abs(x[1] - min(rho))**2
        if x[1] > max(rho):
            diff1 = np.abs(x[1] - max(rho))**2
        if sum([diff0, diff1]) > 1e-5:
            return np.array([diff0**2, diff1**2])
        else:
            # Find pressure and chemical potential differences
            i = np.argmin(np.abs(x[0] - rho))
            j = np.argmin(np.abs(x[1] - rho))
            return np.abs(np.array([P[i] - P[j], mu[i] - mu[j]]))
    roots = fsolve(func, [1, 2])
    return min(roots), max(roots)


def get_coex_4(rho, f):
    # Record the interval between densities
    d_rho = np.diff(rho)[0]
    # Create a convex hull in scipy
    points = np.hstack((rho.reshape(-1, 1), f.reshape(-1, 1)))
    hull = ConvexHull(points)

    # Find the vertices and density difference between vertices
    verts = points[hull.vertices[:]]
    max_i = np.argmax(verts[:, 1])
    vert_inds = hull.vertices[(hull.vertices != 0) & (hull.vertices != (len(points) - 1))]
    d_rho_hull = np.diff(points[vert_inds][:, 0])

    # Liquid-Gas coexistence occurs along a line between two vertices on the convex hull
    # The difference in density between the vertices corresponding to liquid and gas along
    # this line should be larger than any other two consecutive points on the convex hull
    relevant = d_rho_hull > d_rho * 1.001
    rho_hull = points[:, 0][vert_inds]
    rho_g = None
    rho_l = None

    # Extract the liquid and gas densities
    if len(rho_hull[1:][relevant]) > 0:
        rho_g = rho_hull[:-1][relevant][0]
        rho_l = rho_hull[1:][relevant][0]
    return rho_g, rho_l

def get_spinodal(rho_or_vm, mu_or_P):
    # Use scipy to find local maxima and minima
    maxima = rho_or_vm[argrelextrema(mu_or_P, np.greater)]
    minima = rho_or_vm[argrelextrema(mu_or_P, np.less)]
    val1 = None
    val2 = None
    # Get spinodal densities / per particle volumes
    if len(minima) > 0:
        val1 = minima[0]
    if len(maxima) > 0:
        val2 = maxima[0]
    return val1, val2

"""
# Method 1
vm = np.linspace(0.35, 110, 5000)
rho = (1 / vm)[::-1]

b = 1.0 / 3.0
a = 3.0

Ts = np.linspace(0.3, 4, 100)
data = []
meta_data = []
for T in Ts:
    P = T / (vm - b) - a / (vm**2)
    res = get_coex_1(vm, P)
    if res is not None:
        vml, vmg, P_coex, maxwell_int = res
        rhog = 1 / vmg
        rhol = 1 / vml
        data.append((rhog, T))
        data.append((rhol, T))
    spinodal = get_spinodal(vm, P)
    if spinodal is not None:
        if spinodal[0] is not None and spinodal[1] is not None:
            vml, vmg = spinodal
            meta_data.append((meta_rhog, T))
            meta_data.append((meta_rhol, T))


data = np.array(sorted(data, key=lambda x: x[0]))
meta_data = np.array(sorted(meta_data, key=lambda x: x[0]))
critical_point = np.argmax(data[:, 1])
meta_data[:, 0] /= data[critical_point, 0]
meta_data[:, 1] /= data[critical_point, 1]
data[:, 0] /= data[critical_point, 0]
data[:, 1] /= data[critical_point, 1]

plt.figure(figsize=(16, 12), dpi=200)
plt.plot(data[::3, 0], data[::3, 1], lw=3)
plt.plot(meta_data[::3, 0], meta_data[::3, 1], lw=3)
plt.xlabel(r'$\rho / \rho_c$', fontsize=24)
plt.ylabel(r'$T / T_c$', fontsize=24, rotation=0, labelpad=10)
plt.legend(['Binodal', 'Spinodal'], fontsize=20)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
plt.savefig('PS2_1.png')


"""

"""
# Method 2
rho = np.linspace(0.001, 4, 500)

b = 1.0 / 3.0
a = 3.0

Ts = np.linspace(0.3, 4, 100)
data = []
meta_data = []
for T in Ts:
    P = T * rho / (1 - rho * b) - a * rho**2
    f = T * rho * np.log(rho / (1 - rho * b)) - a * rho**2
    mu = (f + P) / rho

    res = get_coex_2(rho, mu)
    if res is not None:
        rhog, rhol, mu_coex, maxwell_int = res
        data.append((rhog, T))
        data.append((rhol, T))
    spinodal = get_spinodal(rho, mu)
    if spinodal is not None:
        if spinodal[0] is not None and spinodal[1] is not None:
            meta_rhog, meta_rhol = spinodal
            meta_data.append((meta_rhog, T))
            meta_data.append((meta_rhol, T))


data = np.array(sorted(data, key=lambda x: x[0]))
meta_data = np.array(sorted(meta_data, key=lambda x: x[0]))
critical_point = np.argmax(data[:, 1])
meta_data[:, 0] /= data[critical_point, 0]
meta_data[:, 1] /= data[critical_point, 1]
data[:, 0] /= data[critical_point, 0]
data[:, 1] /= data[critical_point, 1]

plt.figure(figsize=(16, 12), dpi=200)
plt.plot(data[:, 0], data[:, 1], lw=3)
plt.plot(meta_data[:, 0], meta_data[:, 1], lw=3)
plt.xlabel(r'$\rho / \rho_c$', fontsize=24)
plt.ylabel(r'$T / T_c$', fontsize=24, rotation=0, labelpad=10)
plt.legend(['Binodal', 'Spinodal'], fontsize=20)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
plt.savefig('PS2_2.png')
"""

"""
# Method 3
rho = np.linspace(0.0001, 2.99, 1000)

b = 1.0 / 3.0
a = 3.0

Ts = np.linspace(1, 5, 150)
data = []
meta_data - []
for T in Ts:
    P = T * rho / (1 - rho * b) - a * rho**2
    f = T * rho * np.log(rho / (1 - rho * b)) - a * rho**2
    mu = (f + P) / rho
    res = get_coex_3(rho[1:], P[1:], mu)
    spinodal = get_spinodal(rho[1:], mu)
    if res is not None:
        if res[0] is not None and res[1] is not None:
            rhog, rhol = res
            data.append((rhog, T))
            data.append((rhol, T))
    if spinodal is not None:
        if spinodal[0] is not None and spinodal[1] is not None:
            meta_rhog, meta_rhol = spinodal
            meta_data.append((meta_rhog, T))
            meta_data.append((meta_rhol, T))


data = np.array(sorted(data, key=lambda x: x[0]))
meta_data = np.array(sorted(meta_data, key=lambda x: x[0]))
critical_point = np.argmax(data[:, 1])
meta_data[:, 0] /= data[critical_point, 0]
meta_data[:, 1] /= data[critical_point, 1]
data[:, 0] /= data[critical_point, 0]
data[:, 1] /= data[critical_point, 1]

plt.figure(figsize=(16, 12), dpi=200)
plt.plot(data[:, 0], data[:, 1], lw=3)
plt.plot(meta_data[:, 0], meta_data[:, 1], lw=3)
plt.xlabel(r'$\rho / \rho_c$', fontsize=24)
plt.ylabel(r'$T / T_c$', fontsize=24, rotation=0, labelpad=10)
plt.legend(['Binodal', 'Spinodal'], fontsize=20)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
plt.savefig('PS2_3.png')
"""


# Method 4
rho = np.linspace(0.0001, 2.99, 1000)

b = 1.0 / 3.0
a = 3.0

Ts = np.linspace(1, 5, 150)
data = []
meta_data = []
for T in Ts:
    P = T * rho / (1 - rho * b) - a * rho**2
    f = T * rho * np.log(rho / (1 - rho * b)) - a * rho**2
    mu = (f + P) / rho

    spinodal = get_spinodal(rho, mu)
    res = get_coex_4(rho, f)
    if res is not None:
        if res[0] is not None and res[1] is not None:
            rhog, rhol = res
            data.append((rhog, T))
            data.append((rhol, T))
    if spinodal is not None:
        if spinodal[0] is not None and spinodal[1] is not None:
            meta_rhog, meta_rhol = spinodal
            meta_data.append((meta_rhog, T))
            meta_data.append((meta_rhol, T))


data = np.array(sorted(data, key=lambda x: x[0]))
meta_data = np.array(sorted(meta_data, key=lambda x: x[0]))
critical_point = np.argmax(data[:, 1])
meta_data[:, 0] /= data[critical_point, 0]
meta_data[:, 1] /= data[critical_point, 1]
data[:, 0] /= data[critical_point, 0]
data[:, 1] /= data[critical_point, 1]

plt.figure(figsize=(16, 12), dpi=200)
plt.plot(data[:, 0], data[:, 1], lw=3)
plt.plot(meta_data[:, 0], meta_data[:, 1], lw=3)
plt.xlabel(r'$\rho / \rho_c$', fontsize=24)
plt.ylabel(r'$T / T_c$', fontsize=24, rotation=0, labelpad=10)
plt.legend(['Binodal', 'Spinodal'], fontsize=20)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
plt.savefig('PS2_4.png')



