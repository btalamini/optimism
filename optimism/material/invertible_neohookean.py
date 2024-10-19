import jax
import jax.numpy as np

from optimism import ScalarRootFind
from optimism import TensorMath
from optimism.material.MaterialModel import MaterialModel


def create_material_model_functions(properties):
    lam, mu = _make_properties(properties['elastic modulus'],
                               properties['poisson ratio'])
    J_min = properties['J extrapolation point']
            
    def strain_energy(dudX, internal_variables, dt=0.0):
        del internal_variables, dt
        return energy_density(dudX, lam, mu, J_min)

    def compute_state_new(dudX, internal_variables, dt=0.0):
        del dudX, dt
        return internal_variables

    density = properties.get('density')

    return MaterialModel(strain_energy,
                         _make_initial_state,
                         compute_state_new,
                         density)

def _make_properties(E, nu):
    lam = E*nu/(1 + nu)/(1 - 2*nu)
    mu = 0.5*E/(1 + nu)
    return lam, mu

def energy_density(dudX, lam, mu, J_min):
    J = TensorMath.det(dudX + np.identity(3))
    return jax.lax.cond(J >= J_min, standard_energy, continued_energy, dudX, lam, mu, J_min)

def standard_energy(dudX, lam, mu, J_min):
    I1m3 = 2*np.trace(dudX) + np.tensordot(dudX, dudX)
    Jm1 = TensorMath.detpIm1(dudX)
    logJ = np.log1p(Jm1)
    return 0.5*mu*I1m3 - mu*logJ + 0.5*lam*logJ**2

def continued_energy(dudX, lam, mu, J_min):
    F = dudX + np.identity(3)
    C = F.T@F
    stretches_squared, _ = TensorMath.eigen_sym33_unit(C)
    stretches = np.sqrt(stretches_squared)
    stretches = stretches.at[0].set(np.where(np.linalg.det(F) < 0, -stretches[0], stretches[0]))
    ee = stretches - 1
    I1 = ee[0] + ee[1] + ee[2]
    I2 = ee[0]*ee[1] + ee[1]*ee[2] + ee[2]*ee[0]
    I3 = ee[0]*ee[1]*ee[2]
    solver_settings = ScalarRootFind.get_settings(x_tol=1e-8)
    s, _ = ScalarRootFind.find_root(lambda x: I3*x**3 + I2*x**2 + I1*x + (1 - J_min), 0.5, np.array([0.0, 1.0]), solver_settings)
    q = 1 + s*ee # series expansion point
    h = np.linalg.norm(stretches - q)
    v = h*ee/np.linalg.norm(ee) # h*u in the paper
    W = lambda x: _energy_from_principal_stretches(x, lam, mu)
    psi0, psi1 = jax.jvp(W, (q,), (v,))
    hess = jax.hessian(W)(q)
    psi2 = 0.5*np.dot(v, hess.dot(v))
    return psi0 + psi1 + psi2

def _energy_from_principal_stretches(stretches, lam, mu):
    J = stretches[0]*stretches[1]*stretches[2]
    return 0.5*mu*(np.sum(stretches**2) - 3) - mu*np.log(J) + 0.5*lam*np.log(J)**2

def _make_initial_state():
    return np.array([])
