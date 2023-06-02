import jax.numpy as np

from optimism.material.MaterialModel import MaterialModel

# props
PROPS_E     = 0
PROPS_NU    = 1
PROPS_ALPHA = 2
PROPS_REF_TEMP = 3
PROPS_MU    = 4
PROPS_KAPPA = 5

def create_material_model_functions(properties):
    props = _parse_material_properties(properties)

    _strain = green_lagrange_strain
    
    def strain_energy(dispGrad, internalVars, dt, temperature):
        del internalVars
        del dt
        strain = _strain(dispGrad)
        return _energy_density(strain, temperature, props)

    density = properties.get('density')

    return MaterialModel(compute_energy_density = strain_energy,
                         compute_initial_state = make_initial_state,
                         compute_state_new = compute_state_new,
                         density = density)


def _parse_material_properties(properties):
    E = properties['elastic modulus']
    nu = properties['poisson ratio']
    alpha = properties['thermal expansion coefficient']
    theta0 = properties['reference temperature']
    mu = 0.5*E/(1.0 + nu)
    kappa = E / 3.0 / (1.0 - 2.0*nu)
    return np.array([E, nu, alpha, theta0, mu, kappa])


def _energy_density(strain, temperature, props):
    traceStrain = np.trace(strain)
    dil = 1.0/3.0 * traceStrain
    strainDev = strain - dil*np.identity(3)
    kappa = props[PROPS_KAPPA]
    mu = props[PROPS_MU]
    alpha = props[PROPS_ALPHA]
    theta0 = props[PROPS_REF_TEMP]
    return 0.5*kappa*traceStrain**2 + mu*np.tensordot(strainDev,strainDev) - 3*kappa*alpha*(temperature - theta0)*traceStrain


def make_initial_state():
    return np.array([])


def compute_state_new(dispGrad, internalVars, dt):
    del dispGrad
    del dt
    return internalVars


def green_lagrange_strain(dispGrad):
    return 0.5*(dispGrad + dispGrad.T + dispGrad.T@dispGrad)