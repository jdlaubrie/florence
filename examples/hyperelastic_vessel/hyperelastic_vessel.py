# System libraries and interaction with user
import sys, os
# Mathematics libraries
import numpy as np
from numpy import einsum
# Build a path for python to Florence
from Florence import *

#============================================================
#============= ANISOTROPIC FIBRE DIRECTIONS  ================
#============================================================
def FibreDirections(mesh):
    """
        Routine dedicated to compute the fibre direction of components in integration point for 
        the Material in Florence and for the auxiliar routines in this script. First three directions 
        are taken into the code for Rotation matrix, so always it should be present in this order,
        Normal, Tangential, Axial.
    """
    ndim = mesh.InferSpatialDimension()
    nfibre = 2
    # Geometric definitions per element
    divider = mesh.elements.shape[1]
    directrix = [0.,1.,0.]
    fibre_direction = np.zeros((mesh.nelem,nfibre,ndim),dtype=np.float64)
    # Loop throught the element in the mesh
    for elem in range(mesh.nelem):
        # Geometric definitions per element
        center = np.sum(mesh.points[mesh.elements[elem,:],:],axis=0)/divider
        tangential = np.cross(directrix,center)
        tangential = tangential/np.linalg.norm(tangential)
        normal = np.cross(tangential,directrix)
        # Define the anisotropic orientations
        fibre_direction[elem,0,:]=np.multiply(directrix,np.cos(np.pi/4.)) + np.multiply(tangential,np.sin(np.pi/4.))
        fibre_direction[elem,1,:]=np.multiply(directrix,np.cos(np.pi/4.)) - np.multiply(tangential,np.sin(np.pi/4.))

    return fibre_direction

#============================================================
#===============  HOMOGENIZED CMT  ==========================
#============================================================
def hyperelastic_vessel(p=1):

    ProblemPath = os.getcwd()
    mesh_file = ProblemPath + '/Half_Cylinder.msh'

    #===============  MESH PROCESING  ==========================
    # Build mesh with Florence tools from GMSH mesh
    mesh = Mesh()
    mesh.Read(filename=mesh_file, reader_type="gmsh", element_type="hex",read_surface_info=True)
    ndim = mesh.InferSpatialDimension()
    mesh.GetHighOrderMesh(p=p)

    #Boolean arrays for boundary condition in Dirichlet
    BottomSurface = np.zeros(mesh.nnode,dtype=bool)
    TopSurface = np.zeros(mesh.nnode,dtype=bool)
    Symmetry_Z = np.zeros(mesh.nnode,dtype=bool)
    #Boolean array for boundary condition in Internal Pressure mesh.faces[id,nodes]
    InnerSurface = np.zeros(mesh.faces.shape[0],dtype=bool)

    for idface in range(mesh.faces.shape[0]):
        if mesh.face_to_surface[idface] == 13 or mesh.face_to_surface[idface] == 35:
            for i in range(mesh.faces.shape[1]):
                BottomSurface[mesh.faces[idface,i]] = True
        elif mesh.face_to_surface[idface] == 21 or mesh.face_to_surface[idface] == 43:
            for i in range(mesh.faces.shape[1]):
                TopSurface[mesh.faces[idface,i]] = True
        elif mesh.face_to_surface[idface] == 4 or mesh.face_to_surface[idface] == 48:
            for i in range(mesh.faces.shape[1]):
                Symmetry_Z[mesh.faces[idface,i]] = True
        elif mesh.face_to_surface[idface] == 25 or mesh.face_to_surface[idface] == 47:
            InnerSurface[idface] = True

    DirichletBoundary = {}
    DirichletBoundary['Bottom'] = BottomSurface
    DirichletBoundary['Top'] = TopSurface
    DirichletBoundary['SymmetryZ'] = Symmetry_Z

    PressureBoundary = {}
    PressureBoundary['InnerLogic'] = InnerSurface

    #===============  MATERIAL DEFINITION  ====================

    fibre_direction = FibreDirections(mesh)

    # Define hyperelastic material for the vessel
    material = AnisotropicFungQuadratic(ndim,
            mu=72.*525.,
            kappa=72.*525.*33.,
            k1=568.*525.,
            k2=11.2,
            anisotropic_orientations=fibre_direction)

    # kappa/mu=20  => nu=0.475 (Poisson's ratio)
    # kappa/mu=33  => nu=0.485 (Poisson's ratio)
    # kappa/mu=100 => nu=0.495 (Poisson's ratio)

    #==================  FORMULATION  =========================
    formulation = DisplacementFormulation(mesh)

    #===============  BOUNDARY CONDITIONS  ====================
    # Dirichlet Boundary Conditions
    def Dirichlet_Function(mesh, DirichletBoundary):
        boundary_data = np.zeros((mesh.nnode, 3))+np.NAN
        # boundary conditions base on BoundarySurface boolean array
        boundary_data[DirichletBoundary['Bottom'],1] = 0.
        boundary_data[DirichletBoundary['Top'],:] = 0.
        boundary_data[DirichletBoundary['SymmetryZ'],2] = 0.

        return boundary_data

    # Pressure Boundary Conditions
    def Pressure_Function(mesh, PressureBoundary):
        boundary_flags = np.zeros(mesh.faces.shape[0],dtype=np.uint8)
        boundary_data = np.zeros((mesh.faces.shape[0]))
        # Force magnitud
        mag = 13.3322e3

        for idf in range(mesh.faces.shape[0]):
            if PressureBoundary['InnerLogic'][idf]:
                boundary_data[idf] = mag

        boundary_flags[PressureBoundary['InnerLogic']] = True

        return boundary_flags, boundary_data

    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(Dirichlet_Function, mesh, DirichletBoundary)
    boundary_condition.SetPressureCriteria(Pressure_Function, mesh, PressureBoundary)

    #===============  SOLVER DEFINITION  ======================
    fem_solver = FEMSolver(analysis_nature="nonlinear",
                       analysis_type="static",
                       break_at_stagnation=False,
                       maximum_iteration_for_newton_raphson=50,
                       optimise=False,
                       parallelise=False,
                       print_incremental_log=True,
                       has_moving_boundary=True)


    #================= SOLUTION  =======================
    # Call the solver
    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, material=material,
        boundary_condition=boundary_condition)
    # Print solution to VTK
    solution.WriteVTK('hyperelastic_vessel',quantity=0)
    # Check displacements at (inner and outer nodes, X-coordinate, last step)
    print(solution.sol[[0,1],0,-1])
    

if __name__ == "__main__":
    hyperelastic_vessel(p=1)
