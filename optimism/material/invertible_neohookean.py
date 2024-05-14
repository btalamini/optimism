import jax
import jax.numpy as np

from optimism import TensorMath
from optimism.material.MaterialModel import MaterialModel

CONTINUATION_POINT = 0.9

def create_material_model_functions(properties):
    lam, mu = _make_properties(properties['elastic modulus'],
                               properties['poisson ratio'])
            
    def strain_energy(dudX, internal_variables, dt):
        del internal_variables, dt
        return energy_density(dudX, lam, mu)

    def compute_state_new(dudX, internal_variables, dt):
        del dudX, dt
        return internal_variables

    density = properties.get('density')

    return MaterialModel(strain_energy,
                         _make_initial_state,
                         compute_state_new,
                         density)

def _make_properties(E, nu):
    lam = E/(1 + nu)/(1 - 2*nu)
    mu = 0.5*E/(1 + nu)
    return lam, mu

def energy_density(dudX, lam, mu):
    J = TensorMath.det(dudX + np.identity(3))
    return np.where(J >= CONTINUATION_POINT, standard_energy(dudX, lam, mu), continued_energy(dudX, lam, mu))

def standard_energy(dudX, lam, mu):
    I1m3 = 2*np.trace(dudX) + np.tensordot(dudX, dudX)
    Jm1 = TensorMath.detpIm1(dudX)
    logJ = np.log1p(Jm1)
    return 0.5*mu*I1m3 - mu*logJ + 0.5*lam*logJ**2

def continued_energy(dudX, lam, mu):
    F = dudX + np.identity(3)
    J = np.linalg.det(F)
    C = F.T@F
    stretches_squared, right_evecs = TensorMath.eigen_sym33_unit(C)
    stretches = np.sqrt(stretches_squared)
    # stretches = stretches.at[0].set(np.where(J < 0, -stretches[0], stretches[0]))
    r = np.ones(3)
    ee = stretches - 1
    I1 = ee[0] + ee[1] + ee[2]
    I2 = ee[0]*ee[1] + ee[1]*ee[2] + ee[2]*ee[0]
    I3 = ee[0]*ee[1]*ee[2]
    s = cubic_three_real_roots(np.array([1 - CONTINUATION_POINT, I1, I2, I3]))
    jax.debug.print("s={s}", s=s)
    q = r + s*ee
    h = np.linalg.norm(stretches - q)
    u = -ee/np.linalg.norm(ee)
    psi0 = _energy_from_principal_stretches(q, lam, mu)
    psi1 = 0.0 #jax.jvp(_energy_from_principal_stretches, (q, lam, mu), (h*u, 0.0, 0.0))[0]
    tmp = jax.jvp(jax.grad(_energy_from_principal_stretches, 0), (q, lam, mu), (h*u, 0.0, 0.0))[0]
    psi2 = 0.0 #np.dot(tmp, 0.5*h*u)
    return psi0 + psi1 + psi2

def _energy_from_principal_stretches(stretches, lam, mu):
    J = stretches[0]*stretches[1]*stretches[2]
    return 0.5*mu*(np.sum(stretches**2) - 3) - mu*np.log(J) + 0.5*lam*np.log(J)**2

def _make_initial_state():
    return np.array([])


def cubic_one_real_root(b):
    a1 = b[2]/b[3]
    a2 = b[1]/b[3]
    a3 = b[0]/b[3]
    Q = (a1**2 - 3*a2)/9
    R = (2*a1**3 - 9*a1*a2 + 27*a3)/54
    A = -np.sign(R)*(np.sqrt(R**2 - Q**3) + np.abs(R))**(1/3)
    B = np.where(A != 0, Q/A, 0.0)
    return A + B - a1/3

def cubic_three_real_roots(b):
    jax.debug.print("b={b}", b=b)
    a1 = b[2]/b[3]
    a2 = b[1]/b[3]
    a3 = b[0]/b[3]
    Q = (a1**2 - 3*a2)/9
    R = (2*a1**3 - 9*a1*a2 + 27*a3)/54
    theta = np.acos(R/np.sqrt(Q**3))
    x = -2*np.sqrt(Q)*np.cos(np.array([theta, theta + 2*np.pi, theta - 2*np.pi])/3) - a1/3
    jax.debug.print("Q={Q}, R={R}, theta={theta}, x={x}", Q=Q, R=R, theta=theta, x=x)
    return np.sort(x)[-1]