"""Independent steady 1D channel BVP, not a CFD time integrator."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_bvp
from scipy.special import roots_legendre


MACH = .3
REYNOLDS = 1000.
FORCE = .003
GAMMA = 1.4
PRANDTL = .72
SUTHERLAND = 110.3 / 273.15


def transport(temperature, variable=True):
    mu = (temperature**1.5 * (1 + SUTHERLAND) / (temperature + SUTHERLAND)
          if variable else np.ones_like(temperature)) / REYNOLDS
    return mu, mu / (PRANDTL * (GAMMA - 1) * MACH**2)


@dataclass
class ChannelReference:
    solution: object
    pressure: float
    variable: bool

    def primitive(self, y):
        u, _, temperature, _ = self.solution.sol(y)
        rho = GAMMA * MACH**2 * self.pressure / temperature
        return rho, u, temperature

    def checks(self):
        y, weights = roots_legendre(256)
        u, tau, temperature, heat_gradient = self.solution.sol(y)
        rho, _, _ = self.primitive(y)
        ends = self.solution.sol(np.array([-1., 1.]))
        power = FORCE * float(weights @ u)
        heat = float(ends[3, 0] - ends[3, 1])
        return dict(mean_density=float(weights @ rho / 2), pressure=self.pressure,
                    minimum_temperature=float(temperature.min()),
                    force_shear_residual=float(ends[1, 1] - ends[1, 0] + 2*FORCE),
                    power=power, outgoing_heat=heat, power_heat_residual=power-heat,
                    maximum_bvp_residual=float(self.solution.rms_residuals.max()),
                    maximum_shear_error=float(np.max(np.abs(tau + FORCE*y))))


@lru_cache(maxsize=2)
def steady_reference(variable=True):
    y = np.linspace(-1., 1., 129)
    theta = PRANDTL * (GAMMA-1) * MACH**2 * FORCE**2 * REYNOLDS**2 / 12
    u = FORCE * REYNOLDS * (1-y*y) / 2
    temperature = 1 + theta * (1-y**4)
    _, k = transport(temperature, variable=False)
    guess = np.stack((u, -FORCE*y, temperature, -4*k*theta*y**3))

    def equations(y, state):
        _, tau, temperature, heat_gradient = state
        mu, conductivity = transport(temperature, variable)
        return np.stack((tau/mu, -FORCE*np.ones_like(y),
                         heat_gradient/conductivity, -tau*tau/mu))

    def boundary(left, right):
        return np.array([left[0], right[0], left[2]-1, right[2]-1])

    solution = solve_bvp(equations, boundary, y, guess, tol=1e-10, max_nodes=8193)
    if not solution.success:
        raise RuntimeError(solution.message)
    yq, weights = roots_legendre(256)
    inverse_temperature_mean = float(weights @ (1/solution.sol(yq)[2]) / 2)
    pressure = 1 / (GAMMA*MACH**2*inverse_temperature_mean)
    return ChannelReference(solution, pressure, variable)


def initial_fields(grid, manufactured=False, wall_type=41, perturbation=0.):
    if wall_type != 41:
        raise ValueError('Steady forced reference is for isothermal no-slip walls only')
    nx, ny, nz = grid
    xv = np.linspace(0, 2*np.pi, nx+1)
    yv = np.linspace(-1, 1, ny+1)
    zv = np.linspace(0, np.pi, nz+1)
    x, y, z = np.meshgrid(xv, yv, zv, indexing='ij')
    rho, u, temperature = steady_reference().primitive(y)
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    if perturbation:
        # Curl of (0,0,psi): divergence free and vanishing through wall order three.
        s = np.pi*(y+1)/2
        shape = np.sin(s)**6
        derivative = 3*np.pi*np.sin(s)**5*np.cos(s)
        u = u + perturbation*derivative*np.sin(x)*np.cos(2*z)/(3*np.pi)
        v = -perturbation*shape*np.cos(x)*np.cos(2*z)/(3*np.pi)
        for array in (u, v, w):
            array[:, 0, :] = 0
            array[:, -1, :] = 0
    return (x, y, z), dict(ro=rho, u1=u, u2=v, u3=w, t=temperature)
