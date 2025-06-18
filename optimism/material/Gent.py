import jax.numpy as np

from optimism.material.MaterialModel import MaterialModel


PROPS_KAPPA = 0
PROPS_MU    = 1
PROPS_JM    = 2


def create_material_properties(properties):
    props = _make_properties(properties['bulk modulus'],
                             properties['shear modulus'],
                             properties['Jm parameter'])
    return np.array(props)


def create_material_functions(properties):
    energy_density = _gent_3D_energy_density

    def strain_energy(dispGrad, internalVars, props, dt):
        del internalVars
        del dt
        return energy_density(dispGrad, props)
    
    def compute_state_new(dispGrad, internalVars, props, dt):
        del dispGrad
        del props
        del dt
        return internalVars

    density = properties.get('density')

    return MaterialModel(compute_energy_density = strain_energy,
                         compute_initial_state = make_initial_state,
                         compute_state_new = compute_state_new,
                         density = density)

def _make_properties(K, mu, Jm):
    return np.array([K, mu, Jm])

def _gent_3D_energy_density(dispGrad, props):
    F = dispGrad + np.eye(3)
    J = np.linalg.det(F)
    I1_bar = np.power(J, -2. / 3.) * np.tensordot(F, F)
    Wvol = 0.5*props[PROPS_KAPPA]*(0.5*J**2 - 0.5 - np.log(J))
    Wdev = -0.5 * (props[PROPS_MU] * props[PROPS_JM]) * np.log(1. - (I1_bar - 3.) / props[PROPS_JM])
    return Wdev + Wvol

def make_initial_state():
    return np.array([])

