import jax
import jax.numpy as np
import unittest

from optimism.material import invertible_neohookean
from optimism.test import TestFixture

@jax.jit
def random_displacement_gradient_with_set_J(key, J):
    H = jax.random.uniform(key, (3, 3))       
    F = H + np.identity(3)
    F *= np.sign(J)*(np.abs(J)/np.linalg.det(F))**(1/3)
    return F - np.identity(3)

def random_rotation(key):
    Q = jax.random.orthogonal(key, 3)
    # if matrix is a reflection, swap the first 2 columns to get a rotation
    R = np.where(np.linalg.det(Q) > 0, Q, Q[:, (1, 0, 2)])
    return R

class TestInvertibleNeoHookean(TestFixture.TestFixture):
    def setUp(self):
        self.E = 10.0
        self.nu = 0.25
        self.J_min = 0.9
        properties = {"elastic modulus": self.E,
                      "poisson ratio": self.nu,
                      "J extrapolation point": self.J_min}
        self.material = invertible_neohookean.create_material_model_functions(properties)
        self.compute_energy_density = jax.jit(self.material.compute_energy_density)
        self.internalVariables = self.material.compute_initial_state()
    
    def test_zero_point(self):
        dudX = np.zeros((3, 3))
        W = self.compute_energy_density(dudX, self.internalVariables, dt=0.0)
        self.assertEqual(W, 0.0)
    
    def test_frame_indifference(self):
        # generate a displacement gradient with J > J_min
        key = jax.random.PRNGKey(1)
        dispGrad = random_displacement_gradient_with_set_J(key, 1.01*self.J_min)
        
        subkeys = jax.random.split(key, 2)
        
        W = self.compute_energy_density(dispGrad, self.internalVariables, dt=0.0)
        R = random_rotation(subkeys[0])
        self.assertGreater(np.linalg.det(R), 0.0)
        dispGradTransformed = R@(dispGrad + np.identity(3)) - np.identity(3)
        WStar = self.compute_energy_density(dispGradTransformed, self.internalVariables, dt=0.0)
        self.assertAlmostEqual(W, WStar, 12)
        
    def test_frame_indifference_in_extrapolation_range(self):
        # generate a displacement gradient with J < J_min
        key = jax.random.PRNGKey(1)
        dispGrad = random_displacement_gradient_with_set_J(key, 0.99*self.J_min)
        
        subkeys = jax.random.split(key, 2)
        
        W = self.compute_energy_density(dispGrad, self.internalVariables, dt=0.0)
        R = random_rotation(subkeys[0])
        self.assertGreater(np.linalg.det(R), 0.0)
        dispGradTransformed = R@(dispGrad + np.identity(3)) - np.identity(3)
        WStar = self.compute_energy_density(dispGradTransformed, self.internalVariables, dt=0.0)
        self.assertAlmostEqual(W, WStar, 12)
    
    def test_extrapolation_is_continuous(self):
        key = jax.random.PRNGKey(0)
        dispGrad1 = random_displacement_gradient_with_set_J(key, (1 + 1e-8)*self.J_min)
        W1 = self.compute_energy_density(dispGrad1, self.internalVariables)
        dispGrad2 = random_displacement_gradient_with_set_J(key, (1 - 1e-8)*self.J_min)
        W2 = self.compute_energy_density(dispGrad2, self.internalVariables)
        self.assertLess((W1 - W2)/W1, 1e-7)
    
    def test_extrapolation_derivative_is_continuous(self):
        key = jax.random.PRNGKey(0)
        dispGrad1 = random_displacement_gradient_with_set_J(key, (1 + 1e-8)*self.J_min)
        compute_stress = jax.jit(jax.grad(self.material.compute_energy_density))
        P1 = compute_stress(dispGrad1, self.internalVariables)
        dispGrad2 = random_displacement_gradient_with_set_J(key, (1 - 1e-8)*self.J_min)
        P2 = compute_stress(dispGrad2, self.internalVariables)
        diff = np.linalg.norm(P1 - P2)/np.linalg.norm(P1)
        self.assertLess(diff, 1e-7)
    
    def test_extrapolation_2nd_derivative_is_continuous(self):
        key = jax.random.PRNGKey(3)
        dispGrad1 = random_displacement_gradient_with_set_J(key, (1 + 1e-8)*self.J_min)
        subkey = jax.random.split(key)
        w = jax.random.uniform(key, (3, 3))
        energy = lambda H: self.compute_energy_density(H, self.internalVariables)
        matvec = lambda v: jax.jvp(jax.grad(energy), (dispGrad1,), (v,))[1]
        #AH = jax.jit(matvec)(w)
        #print(f"AH={AH}")
        #compute_tangents = jax.jit(jax.hessian(self.material.compute_energy_density))
        #A1 = compute_tangents(dispGrad1, self.internalVariables).reshape(9, 9)
    #     dispGrad2 = random_displacement_gradient_with_set_J(key, (1 - 1e-8)*self.J_min)
    #     A2 = compute_tangents(dispGrad2, self.internalVariables).reshape(9, 9)
    #     idx = np.argmax( np.abs((A1 - A2)/A1) )
    #     diff = np.abs((A1.ravel()[idx] - A2.ravel()[idx])/A1.ravel()[idx])
    #     self.assertLess(diff, 1e-5)

    def test_negative_jacobian_no_nan(self):
        key = jax.random.PRNGKey(0)
        dispGrad = random_displacement_gradient_with_set_J(key, -0.2)
        energy = self.compute_energy_density(dispGrad, self.internalVariables)
        self.assertFalse(np.isnan(energy))

if __name__ == "__main__":
    unittest.main()