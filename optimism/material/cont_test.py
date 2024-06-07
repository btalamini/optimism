import jax.numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import invertible_neohookean

E = 1.0
nu = 0.25
J_min = 0.9
props = {'elastic modulus': E, 'poisson ratio': nu, 'J extrapolation point': J_min}

material = invertible_neohookean.create_material_model_functions(props)

stretch_history = np.flip(np.linspace(-0.1, 1.0, 20))
Q = material.compute_initial_state()
energy_history = []

for stretch in stretch_history:
    H = np.diag(np.array([stretch - 1.0, 0.0, 0.0]))
    W = jax.jit(material.compute_energy_density)(H, Q, 0.0)
    energy_history.append(W)

print(stretch_history)
print(energy_history)
import matplotlib.pyplot as plt
plt.plot(stretch_history, energy_history, '-o')
plt.show()
