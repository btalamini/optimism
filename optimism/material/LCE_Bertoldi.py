import jax.numpy as np
from optimism.material.MaterialModel import MaterialModel

# props
P_E      = 0
P_NU     = 1
P_BETA   = 2
P_ST     = 3
P_S0     = 4
P_GAMMA  = 5

def create_material_model_functions(properties):
    props = _make_properties(properties['elastic modulus'],
                             properties['poisson ratio'],
                             properties['beta parameter'],
                             properties['current order parameter'],
                             properties['reference order parameter'],
                             properties['LC angle alignment'])

    energy_density = _lce_bertoldi_energy

    def strain_energy(dispGrad, internalVars, dt, currentOrder):
        del dt
        return energy_density(dispGrad, internalVars, props, currentOrder)

    def compute_state_new(dispGrad, internalVars, currentOrder):
        del dt
        return _compute_state_new(dispGrad, internalVars, props, dt, currentOrder)

    density = properties.get('density')

    return MaterialModel(strain_energy,
                         make_initial_state,
                         compute_state_new,
                         density)

def _make_properties(E, nu, beta, currentOrder, refOrder, gammaAngle):
    # mu = 0.5*E/(1.0 + nu)
    # lamda = E*nu/(1 + nu)/(1 - 2*nu)
    return np.array([E, nu, beta, currentOrder, refOrder, gammaAngle])

def _lce_bertoldi_energy(dispGrad, internalVariables, props, currentOrder):
    
    # material-dependent scalars
    mu = 0.5*props[P_E]/(1.0 + props[P_NU])
    lamda = props[P_E]*props[P_NU]/(1 + props[P_NU])/(1 - 2*props[P_NU])
    # eta = 0.5*props[P_BETA] * (props[P_ST]-props[P_S0])
    eta = 0.5*props[P_BETA] * (currentOrder-props[P_S0])

    # LC direction
    angle = P_GAMMA/180.0*np.pi
    n = np.array([np.cos(angle), np.sin(angle), 0.0])

    # tensor definitions
    I = np.eye(3)
    F = dispGrad + I
    C = np.dot(F.T, F)
    E = 0.5 * (C-I)

    ## Components of strain energy
    strEn1 = 0.5*lamda*np.trace(E)**2
    strEn2 = mu*np.tensordot(E, E)
    strEn3 = eta*np.tensordot(np.outer(n, n) - I, E)

    return strEn1 + strEn2 + strEn3

def make_initial_state():
    return np.array([])

# def _compute_state_new(dispGrad, internalVars, props, currentOrder):
#     return internalVars

def _compute_state_new(dispGrad, internalVars, props, currentOrder):
    del dispGrad
    del props
    del currentOrder
    return internalVars
