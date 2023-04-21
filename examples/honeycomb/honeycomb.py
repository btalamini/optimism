import jax
import jax.numpy as np
import numpy as onp

from optimism import EquationSolver
from optimism import FunctionSpace
from optimism.material import Neohookean
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism import SparseMatrixAssembler
from optimism import VTKWriter

mesh = ReadExodusMesh.read_exodus_mesh("honeycomb.g")
mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, useBubbleElement=False, copyNodeSets=False, createNodeSetsFromSideSets=True)

ebcs = [FunctionSpace.EssentialBC(nodeSet="left", component=0),
        FunctionSpace.EssentialBC(nodeSet="right", component=0),
        FunctionSpace.EssentialBC(nodeSet="bottom", component=1),
        FunctionSpace.EssentialBC(nodeSet="top", component=1)]

quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2*(mesh.parentElement.degree - 1))
fs = FunctionSpace.construct_function_space(mesh, quadRule, mode2D="cartesian")
ebcManager = FunctionSpace.DofManager(fs, dim=2, EssentialBCs=ebcs)

kappa = 10.0
nu = 0.3
E = 3*kappa*(1 - 2*nu)
props = {'elastic modulus': E,
            'poisson ratio': nu,
            'version': 'coupled'}
materialModel = Neohookean.create_material_model_functions(props)


solidMechanics = Mechanics.create_mechanics_functions(fs,
                                                      mode2D="plane strain",
                                                      materialModel=materialModel)

solverSettings = EquationSolver.get_settings()

outputForce = []
outputDisp = []

def get_ubcs(p):
    yLoc = p[0]
    V = np.zeros_like(mesh.coords)
    index = mesh.nodeSets["top"], 1
    V = V.at[index].set(yLoc)
    return ebcManager.get_bc_values(V)

def create_field(Uu, p):
    return ebcManager.create_field(Uu, get_ubcs(p))

def energy_function(Uu, p):
    U = create_field(Uu, p)
    internalVariables = p[1]
    return solidMechanics.compute_strain_energy(U, internalVariables)

def assemble_sparse_preconditioner(Uu, p):
    U = create_field(Uu, p)
    internalVariables = p[1]
    elementStiffnesses =  solidMechanics.compute_element_stiffnesses(U, internalVariables)
    return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementStiffnesses,
                                                                  fs.mesh.conns,
                                                                  ebcManager)

def compute_energy_from_bcs(Uu, Ubc, internalVariables):
    U = ebcManager.create_field(Uu, Ubc)
    return solidMechanics.compute_strain_energy(U, internalVariables)

compute_bc_reactions = jax.jit(jax.grad(compute_energy_from_bcs, 1))

def write_output(U, p, step):
    plotName = 'honeycomb-'+str(step).zfill(3)
    writer = VTKWriter.VTKWriter(mesh, baseFileName=plotName)
    
    writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

    bcs = onp.array(ebcManager.isBc, dtype=onp.int64)
    writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)

    Ubc = get_ubcs(p)
    internalVariables = p[1]
    rxnBc = compute_bc_reactions(ebcManager.get_unknown_values(U), Ubc, internalVariables)
    reactions = np.zeros(U.shape).at[ebcManager.isBc].set(rxnBc)
    writer.add_nodal_field(name='reactions', nodalData=reactions, fieldType=VTKWriter.VTKFieldType.VECTORS)

    energyDensities, stresses = solidMechanics.compute_output_energy_densities_and_stresses(U, internalVariables)
    cellEnergyDensities = FunctionSpace.project_quadrature_field_to_element_field(fs, energyDensities)
    cellStresses = FunctionSpace.project_quadrature_field_to_element_field(fs, stresses)
    writer.add_cell_field(name='strain_energy_density',
                            cellData=cellEnergyDensities,
                            fieldType=VTKWriter.VTKFieldType.SCALARS)
    writer.add_cell_field(name='piola_stress',
                            cellData=cellStresses,
                            fieldType=VTKWriter.VTKFieldType.TENSORS)
    
    writer.write()

    outputForce.append(float(-np.sum(reactions[mesh.nodeSets["top"], 1])))
    outputDisp.append(float(-p[0]))

    with open('force_displacement.npz','wb') as f:
        np.savez(f, force=np.array(outputForce),
                 displacement=np.array(outputDisp))


def run():
    Uu = ebcManager.get_unknown_values(np.zeros_like(mesh.coords))
    disp = 0.0
    ivs = solidMechanics.compute_initial_state()
    p = Objective.Params(disp, ivs)

    precondStrategy = Objective.PrecondStrategy(assemble_sparse_preconditioner)
    objective = Objective.Objective(energy_function, Uu, p, precondStrategy)

    write_output(create_field(Uu, p), p, step=0)
        
    steps = 20
    maxDisp = 2*1.52e-3
    for i in range(1, steps):
        print('--------------------------------------')
        print('LOAD STEP ', i)

        disp -= maxDisp/steps
        p = Objective.param_index_update(p, 0, disp)
        Uu = EquationSolver.nonlinear_equation_solve(objective,
                                                     Uu,
                                                     p,
                                                     solverSettings)
        U = create_field(Uu, p)
        write_output(U, p, i)

if __name__ == "__main__":
    run()
    
    from matplotlib import pyplot as plt
    plotData = np.load('force_displacement.npz')
    F = plotData['force']
    U = plotData['displacement']
    plt.plot(U, F, '-o')
    ax = plt.gca()
    ax.set(xlabel='Displacement', ylabel='Force')
    plt.show()