from functools import partial
import jax
import jax.numpy as np
import numpy as onp
import sys

from optimism import AlSolver
from optimism.contact import Contact
from optimism import ConstrainedObjective
from optimism import EquationSolver
from optimism import FunctionSpace
from optimism.material import LCE_Bertoldi
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import ReadExodusMesh
from optimism import SparseMatrixAssembler
from optimism import VTKWriter

useFullMesh = True

mesh = ReadExodusMesh.read_exodus_mesh("honeycomb_tri3.g")
if useFullMesh:
    mesh = ReadExodusMesh.read_exodus_mesh("honeycomb_full_dom_tris.g")

#mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, useBubbleElement=False, copyNodeSets=False, createNodeSetsFromSideSets=True)

ebcs = [FunctionSpace.EssentialBC(nodeSet="left", component=0),
        FunctionSpace.EssentialBC(nodeSet="right", component=0),
        FunctionSpace.EssentialBC(nodeSet="bottom", component=1),
        FunctionSpace.EssentialBC(nodeSet="top", component=1)]

# ebcs = [FunctionSpace.EssentialBC(nodeSet="bottom", component=0),
#         FunctionSpace.EssentialBC(nodeSet="bottom", component=1),
#         FunctionSpace.EssentialBC(nodeSet="top", component=1)]

# ebcs = [FunctionSpace.EssentialBC(nodeSet="left", component=0),
#         FunctionSpace.EssentialBC(nodeSet="bottom", component=1),
#         FunctionSpace.EssentialBC(nodeSet="top", component=1)]

quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2*(mesh.parentElement.degree - 1))
fs = FunctionSpace.construct_function_space(mesh, quadRule, mode2D="cartesian")
ebcManager = FunctionSpace.DofManager(fs, dim=2, EssentialBCs=ebcs)

E = 0.25e6
nu = 0.48
beta = 5.2e4
currentOrder = 0.3
refOrder = 0.4
gammaAngle = 0.0 # degrees

props = {'elastic modulus': E,
            'poisson ratio': nu,
            'beta parameter': beta,
            'current order parameter': currentOrder,
            'reference order parameter': refOrder,
            'LC angle alignment': gammaAngle}
materialModel = LCE_Bertoldi.create_material_model_functions(props)

solidMechanics = Mechanics.create_mechanics_functions(fs,
                                                      mode2D="plane strain",
                                                      materialModel=materialModel)

contactQuadRule = QuadratureRule.create_quadrature_rule_1D(4)

solverSettings = EquationSolver.get_settings(tol=1e-7)

alSettings = AlSolver.get_settings(max_gmres_iters=300,
                                   num_initial_low_order_iterations=5,
                                   use_second_order_update=True,
                                   penalty_scaling = 1.05,
                                   target_constraint_decrease_factor=0.5,
                                   tol=2e-7)

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

def compute_potential(Uu, p):
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

h = 2.5e-4 # approximate element size
closest_distance = partial(Contact.compute_closest_distance_to_each_side_smooth, smoothingTol=1e-1*h)

def compute_constraints(Uu, p):
    U = create_field(Uu, p)
    interactionList1 = p[3][0]
    contactDists1 = closest_distance(mesh, U, contactQuadRule, interactionList1, mesh.sideSets["cell_0_contact_B"])
    interactionList2 = p[3][1]
    contactDists2 = closest_distance(mesh, U, contactQuadRule, interactionList2, mesh.sideSets["cell_1_contact_B"])
    interactionList3 = p[3][2]
    contactDists3 = closest_distance(mesh, U, contactQuadRule, interactionList3, mesh.sideSets["cell_2_contact_B"])

    if useFullMesh:
        interactionList4 = p[3][3]
        contactDists4 = closest_distance(mesh, U, contactQuadRule, interactionList4, mesh.sideSets["cell_3_contact_B"])
        interactionList5 = p[3][4]
        contactDists5 = closest_distance(mesh, U, contactQuadRule, interactionList5, mesh.sideSets["cell_4_contact_B"])
        interactionList6 = p[3][5]
        contactDists6 = closest_distance(mesh, U, contactQuadRule, interactionList6, mesh.sideSets["cell_5_contact_B"])

        interactionList7 = p[3][6]
        contactDists7 = closest_distance(mesh, U, contactQuadRule, interactionList7, mesh.sideSets["cell_6_contact_B"])
        interactionList8 = p[3][7]
        contactDists8 = closest_distance(mesh, U, contactQuadRule, interactionList8, mesh.sideSets["cell_7_contact_B"])
        interactionList9 = p[3][8]
        contactDists9 = closest_distance(mesh, U, contactQuadRule, interactionList9, mesh.sideSets["cell_8_contact_B"])

        interactionList10 = p[3][9]
        contactDists10 = closest_distance(mesh, U, contactQuadRule, interactionList10, mesh.sideSets["cell_9_contact_B"])
        interactionList11 = p[3][10]
        contactDists11 = closest_distance(mesh, U, contactQuadRule, interactionList11, mesh.sideSets["cell_10_contact_B"])
        interactionList12 = p[3][11]
        contactDists12 = closest_distance(mesh, U, contactQuadRule, interactionList12, mesh.sideSets["cell_11_contact_B"])

        return np.hstack((contactDists1.ravel(), contactDists2.ravel(), contactDists3.ravel(), contactDists4.ravel(), contactDists5.ravel(), contactDists6.ravel(), contactDists7.ravel(), contactDists8.ravel(), contactDists9.ravel(), contactDists10.ravel(), contactDists11.ravel(), contactDists12.ravel()))
    
    return np.hstack((contactDists1.ravel(), contactDists2.ravel(), contactDists3.ravel()))


def compute_energy_from_bcs(Uu, Ubc, internalVariables):
    U = ebcManager.create_field(Uu, Ubc)
    return solidMechanics.compute_strain_energy(U, internalVariables)

compute_bc_reactions = jax.jit(jax.grad(compute_energy_from_bcs, 1))

def write_output(U, p, step):
    plotName = 'lce-full-honeycomb-order-diffBCs-'+str(step).zfill(3)
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


def update_contact_params(Uu, p):
    U = create_field(Uu, p)
    maxContactNeighbors = 4
    
    # First group of (three) contact interfaces
    sideA = mesh.sideSets["cell_0_contact_A"]
    sideB = mesh.sideSets["cell_0_contact_B"]
    interactionList1 = Contact.get_potential_interaction_list(sideA, sideB,
                                                              mesh, U, maxContactNeighbors)
    
    sideA = mesh.sideSets["cell_1_contact_A"]
    sideB = mesh.sideSets["cell_1_contact_B"]
    interactionList2 = Contact.get_potential_interaction_list(sideA, sideB,
                                                              mesh, U, maxContactNeighbors)
    
    sideA = mesh.sideSets["cell_2_contact_A"]
    sideB = mesh.sideSets["cell_2_contact_B"]
    interactionList3 = Contact.get_potential_interaction_list(sideA, sideB,
                                                              mesh, U, maxContactNeighbors)
    if useFullMesh:
        # Second group of (three) contact interfaces
        sideA = mesh.sideSets["cell_3_contact_A"]
        sideB = mesh.sideSets["cell_3_contact_B"]
        interactionList4 = Contact.get_potential_interaction_list(sideA, sideB,
                                                                mesh, U, maxContactNeighbors)
        
        sideA = mesh.sideSets["cell_4_contact_A"]
        sideB = mesh.sideSets["cell_4_contact_B"]
        interactionList5 = Contact.get_potential_interaction_list(sideA, sideB,
                                                                mesh, U, maxContactNeighbors)
        
        sideA = mesh.sideSets["cell_5_contact_A"]
        sideB = mesh.sideSets["cell_5_contact_B"]
        interactionList6 = Contact.get_potential_interaction_list(sideA, sideB,
                                                                mesh, U, maxContactNeighbors)
        # Third group of (three) contact interfaces
        sideA = mesh.sideSets["cell_6_contact_A"]
        sideB = mesh.sideSets["cell_6_contact_B"]
        interactionList7 = Contact.get_potential_interaction_list(sideA, sideB,
                                                                mesh, U, maxContactNeighbors)
        
        sideA = mesh.sideSets["cell_7_contact_A"]
        sideB = mesh.sideSets["cell_7_contact_B"]
        interactionList8 = Contact.get_potential_interaction_list(sideA, sideB,
                                                                mesh, U, maxContactNeighbors)
        
        sideA = mesh.sideSets["cell_8_contact_A"]
        sideB = mesh.sideSets["cell_8_contact_B"]
        interactionList9 = Contact.get_potential_interaction_list(sideA, sideB,
                                                                mesh, U, maxContactNeighbors)
        
        # Fourth group of (three) contact interfaces
        sideA = mesh.sideSets["cell_9_contact_A"]
        sideB = mesh.sideSets["cell_9_contact_B"]
        interactionList10 = Contact.get_potential_interaction_list(sideA, sideB,
                                                                mesh, U, maxContactNeighbors)
        
        sideA = mesh.sideSets["cell_10_contact_A"]
        sideB = mesh.sideSets["cell_10_contact_B"]
        interactionList11 = Contact.get_potential_interaction_list(sideA, sideB,
                                                                mesh, U, maxContactNeighbors)
        
        sideA = mesh.sideSets["cell_11_contact_A"]
        sideB = mesh.sideSets["cell_11_contact_B"]
        interactionList12 = Contact.get_potential_interaction_list(sideA, sideB,
                                                                mesh, U, maxContactNeighbors)
        p = Objective.param_index_update(p, 3, (interactionList1, interactionList2, interactionList3, interactionList4, interactionList5, interactionList6, interactionList7, interactionList8, interactionList9, interactionList10, interactionList11, interactionList12))
    
    else:
        p = Objective.param_index_update(p, 3, (interactionList1, interactionList2, interactionList3))

    return p


def run():
    Uu = ebcManager.get_unknown_values(np.zeros_like(mesh.coords))
    disp = 0.0
    ivs = solidMechanics.compute_initial_state()
    p = Objective.Params(disp, ivs)
    
    searchFrequency = 1
    p = update_contact_params(Uu, p)

    c = compute_constraints(Uu, p)
    initialMultiplier = 4.0
    kappa0 = initialMultiplier * np.ones_like(c)
    lam0 = 1e-4*np.abs(kappa0*c)
    print(f"c.shape={c.shape}")
    print(f"kappa0.shape={kappa0.shape}")
    print(f"lam0.shape={lam0.shape}")
    
    precondStrategy = Objective.PrecondStrategy(assemble_sparse_preconditioner)
    #objective = Objective.Objective(compute_potential, Uu, p, precondStrategy)
    objective = ConstrainedObjective.ConstrainedObjective(compute_potential, compute_constraints, Uu, p, lam0, kappa0)

    write_output(create_field(Uu, p), p, step=0)
    
    steps = 40
    maxDisp = 2.5*1.5e-3
    if useFullMesh:
        maxDisp = 5.5*1.5e-3

    for i in range(1, steps):
        print('')
        print('')
        print('=============')
        print('LOAD STEP ', i)
        print('=============')
        print('')
        print('')

        disp -= maxDisp/steps
        p = Objective.param_index_update(p, 0, disp)
        contactDists = compute_constraints(Uu, p)
        print(f"Contact dists = {np.amin(contactDists)}, {np.amax(contactDists)}")
        # Uu = EquationSolver.nonlinear_equation_solve(objective,
        #                                              Uu,
        #                                              p,
        #                                              solverSettings)
        Uu = AlSolver.augmented_lagrange_solve(objective, Uu, p, alSettings, solverSettings)
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