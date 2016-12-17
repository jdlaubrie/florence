from __future__ import print_function
import numpy as np, scipy as sp, sys, os, gc
from warnings import warn
from time import time
# import scipy as sp
# from DirichletBoundaryDataFromCAD import IGAKitWrapper, PostMeshWrapper
# import numpy as np, sys, gc

from Florence.QuadratureRules import GaussLobattoQuadrature
from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints

from CurvilinearMeshing.IGAKitPlugin.IdentifyNURBSBoundaries import GetDirichletData
# from Florence import PostMeshCurvePy as PostMeshCurve 
# from Florence import PostMeshSurfacePy as PostMeshSurface 


class BoundaryCondition(object):
    """Base class for applying all types of boundary conditions"""

    def __init__(self):
        # TYPE OF BOUNDARY straight OF nurbs
        self.boundary_type = 'straight'
        self.dirichlet_data_applied_at = 'node' # or 'faces'
        self.neumann_data_applied_at = 'node' # or 'faces'
        self.requires_cad = False
        self.cad_file = None
        # PROJECTION TYPE FOR CAD EITHER orthogonal OR arc_length
        self.projection_type = 'orthogonal'
        # WHAT TYPE OF ARC LENGTH BASED PROJECTION, EITHER 'equal' OR 'fekete'
        self.nodal_spacing_for_cad = 'equal'
        self.project_on_curves = False
        self.scale_mesh_on_projection = False
        self.scale_value_on_projection = 1.0
        self.condition_for_projection = 1.0e20
        self.has_planar_surfaces = False
        self.solve_for_planar_faces = True
        self.projection_flags = None
        # FIX DEGREES OF FREEDOM EVERY WHERE CAD PROJECTION IS NOT APPLIED 
        self.fix_dof_elsewhere = None
        # FOR 3D ARC-LENGTH PROJECTION
        self.orthogonal_fallback_tolerance = 1.0
        # WHICH ALGORITHM TO USE FOR SURFACE IDENTIFICATION, EITHER 'minimisation' or 'pure_projection'
        self.surface_identification_algorithm = 'minimisation'
        # MODIFY LINEAR MESH ON PROJECTION
        self.modify_linear_mesh_on_projection = 1

        # FOR IGAKit WRAPPER
        self.nurbs_info = None
        self.nurbs_condition = None

        self.analysis_type = 'static'
        self.analysis_nature = 'linear'

        self.is_dirichlet_computed = False
        self.columns_out = None
        self.columns_in = None
        self.applied_dirichlet = None

        self.dirichlet_flags = None
        self.neumann_flags = None


        # NODAL FORCES GENERATED BASED ON DIRICHLET OR NEUMANN ARE NOT 
        # IMPLEMENTED AS PART OF BOUNDARY CONDITION YET. THIS ESSENTIALLY
        # MEANS SAVING MULTIPLE RHS VALUES
        # self.dirichlet_forces = None
        # self.neumann_forces = None

        # # THE FOLLOWING MEMBERS ARE NOT UPDATED, TO REDUCE MEMORY FOOTPRINT 
        # self.external_nodal_forces = None
        # self.internal_traction_forces = None
        # self.residual = None


    def SetAnalysisParameters(self,analysis_type='static',analysis_nature='linear',
        columns_in=None,columns_out=None,applied_dirichlet=None):
        """Set analysis parameters such as analysis type, analysis nature and even
            Dirichlet boundary conditions if known a priori
        """
        self.analysis_type = analysis_type
        self.analysis_nature = analysis_nature
        self.columns_out = columns_out
        self.columns_in = columns_in
        self.applied_dirichlet = applied_dirichlet



    def SetCADProjectionParameters(self, cad_file=None, requires_cad=True, projection_type='orthogonal', 
        nodal_spacing='equal', project_on_curves=False, has_planar_surfaces=False, solve_for_planar_faces=True, 
        scale=1.0,condition=1.0e20, projection_flags=None, fix_dof_elsewhere=True,
        orthogonal_fallback_tolerance=1.0, surface_identification_algorithm='minimisation',
        modify_linear_mesh_on_projection=True):
        """Set parameters for CAD projection in order to obtain dirichlet boundary
            conditinos
        """

        self.boundary_type = 'nurbs'
        self.requires_cad = requires_cad
        self.cad_file = cad_file
        self.projection_type = projection_type
        self.scale_mesh_on_projection = True
        self.scale_value_on_projection = 1.0*scale
        self.condition_for_projection = 1.0*condition
        self.project_on_curves = project_on_curves
        self.has_planar_surfaces = has_planar_surfaces
        self.solve_for_planar_faces = solve_for_planar_faces
        self.projection_flags = projection_flags
        self.fix_dof_elsewhere = fix_dof_elsewhere
        self.orthogonal_fallback_tolerance = orthogonal_fallback_tolerance
        self.surface_identification_algorithm = surface_identification_algorithm
        self.modify_linear_mesh_on_projection = modify_linear_mesh_on_projection
        self.nodal_spacing_for_cad = nodal_spacing

        self.project_on_curves = int(self.project_on_curves)
        self.modify_linear_mesh_on_projection = int(self.modify_linear_mesh_on_projection)


    def SetProjectionCriteria(self, proj_func, mesh, takes_self=False, **kwargs):
        """Factory function for setting projection criteria specific 
            to a problem

            input:
                func                [function] function that computes projection criteria
                mesh                [Mesh] an instance of mesh class
                **kwargs            optional keyword arguments 
        """

        if takes_self:
            self.projection_flags  = proj_func(mesh,self)
        else:
            self.projection_flags = proj_func(mesh)

        if isinstance(self.projection_flags,np.ndarray):
            if self.projection_flags.ndim==1:
                self.projection_flags.reshape(-1,1)
                ndim = mesh.InferSpatialDimension()
                if self.projection_flags.shape[0] != mesh.edges.shape[0] and ndim == 2:
                    raise ValueError("Projection flags are incorrect. "
                        "Ensure that your projection function returns an ndarray of shape (mesh.edges.shape[0],1)")
                elif self.projection_flags.shape[0] != mesh.faces.shape[0] and ndim == 3:
                    raise ValueError("Projection flags are incorrect. "
                        "Ensure that your projection function returns an ndarray of shape (mesh.faces.shape[0],1)")
        else:
            raise ValueError("Projection flags for CAD not set. "
                "Ensure that your projection function returns an ndarray")


    def GetProjectionCriteria(self,mesh):
        """Convenience method for computing projection flags, as many problems
        require this type of projection
        """

        if mesh.element_type == "tet":
            projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
            num = mesh.faces.shape[1]
            for iface in range(mesh.faces.shape[0]):
                x = np.sum(mesh.points[mesh.faces[iface,:],0])/num
                y = np.sum(mesh.points[mesh.faces[iface,:],1])/num
                z = np.sum(mesh.points[mesh.faces[iface,:],2])/num
                x *= self.scale_value_on_projection
                y *= self.scale_value_on_projection
                z *= self.scale_value_on_projection
                if np.sqrt(x*x+y*y+z*z)< self.condition_for_projection:
                    projection_faces[iface]=1

            self.projection_flags = projection_faces

        elif mesh.element_type == "tri":
            projection_edges = np.zeros((mesh.edges.shape[0],1),dtype=np.uint64)
            num = mesh.edges.shape[1]
            for iedge in range(mesh.edges.shape[0]):
                x = np.sum(mesh.points[mesh.edges[iedge,:],0])/num
                y = np.sum(mesh.points[mesh.edges[iedge,:],1])/num
                x *= self.scale_value_on_projection
                y *= self.scale_value_on_projection
                if np.sqrt(x*x+y*y)< self.condition_for_projection:
                    projection_edges[iedge,0]=1
            
            self.projection_flags = projection_edges


    def GetGeometryMeshScale(self,gpoints,mesh):
        """Compares CAD geometry and mesh, to check if the mesh coordinates
            require scaling

            raises an error if scaling is 
        """

        gminx, gmaxx = np.min(gpoints[:,0]), np.max(gpoints[:,0])
        gminy, gmaxy = np.min(gpoints[:,1]), np.max(gpoints[:,1])
        gmax = np.max(gpoints)
        mmax = np.max(mesh.points)

        # NOTE THAT THE BOUNDS OF NURBS BOUNDARY ISN'T 
        # NECESSARILY EXACTLY THE SAME AS THE MESH, EVEN IF 
        # SCALED APPROPRIATELY 
        gbounds = np.array([[gminx,gminy],[gmaxx,gmaxy]])

        units_scalar = [1000.,25.4]
        for scale in units_scalar:
            if np.isclose(gmax/mmax,scale):
                if self.scale_value_on_projection != scale:
                    self.scale_value_on_projection = scale
                    raise ValueError('Geometry to mesh scale seems incorrect. Change it to %9.3f' % scale)
                    # A SIMPLE WARNING IS NOT POSSIBLE AT THE MOMENT
                    # warn('Geometry to mesh scale seems incorrect. Change it to %9.3f' % scale)
                    # break

        return gbounds
 


    def SetDirichletCriteria(self, func, *args, **kwargs):
        self.dirichlet_flags = func(*args, **kwargs)
        return self.dirichlet_flags


    def SetNeumannCriteria(self, func, *args, **kwargs):
        self.neumann_flags = func(*args, **kwargs)
        return self.neumann_flags


    def GetDirichletBoundaryConditions(self, formulation, mesh, material=None, solver=None, fem_solver=None):

        nvar = formulation.nvar
        ndim = formulation.ndim

        # ColumnsOut = []; AppliedDirichlet = []
        self.columns_in, self.applied_dirichlet = [], []


        #----------------------------------------------------------------------------------------------------#
        #-------------------------------------- NURBS BASED SOLUTION ----------------------------------------#
        #----------------------------------------------------------------------------------------------------#
        if self.boundary_type == 'nurbs':

            tCAD = time()

            IsHighOrder = mesh.IsHighOrder
            # IsDirichletComputed = getattr(MainData.BoundaryData,"IsDirichletComputed",None)
                
            IsHighOrder = False

            if IsHighOrder is False:

                if not self.is_dirichlet_computed:

                    # GET DIRICHLET BOUNDARY CONDITIONS BASED ON THE EXACT GEOMETRY FROM CAD
                    if self.requires_cad:
                        # CALL POSTMESH WRAPPER
                        nodesDBC, Dirichlet = self.PostMeshWrapper(formulation, mesh, material, solver, fem_solver)
                    else:
                        # CALL IGAKIT WRAPPER
                        nodesDBC, Dirichlet = self.IGAKitWrapper(mesh)

                else:
                    nodesDBC, Dirichlet = self.nodesDBC, self.Dirichlet

                # if mesh.points.shape[1]==3:
                #     np.savetxt("/home/roman/DirichletF05_P2_06Dec.dat", Dirichlet,fmt="%9.9f")
                #     np.savetxt("/home/roman/nodesDBCF05_P2_06Dec.dat", nodesDBC,fmt="%i")

                # Dirichlet = np.loadtxt("/home/roman/DirichletF05_P2_04Dec.dat", dtype=np.float64)
                # nodesDBC = np.loadtxt("/home/roman/nodesDBCF05_P2_04Dec.dat", dtype=np.int64)
                # nodesDBC = nodesDBC[:,None]

                # GET DIRICHLET DoFs
                self.columns_out = (np.repeat(nodesDBC,nvar,axis=1)*nvar +\
                 np.tile(np.arange(nvar)[None,:],nodesDBC.shape[0]).reshape(nodesDBC.shape[0],formulation.ndim)).ravel()
                self.applied_dirichlet = Dirichlet.ravel()



                # FIX THE DOF IN THE REST OF THE BOUNDARY
                if self.fix_dof_elsewhere:
                    if ndim==2:
                        Rest_DOFs = np.setdiff1d(np.unique(mesh.edges),nodesDBC)
                    elif ndim==3:
                        Rest_DOFs = np.setdiff1d(np.unique(mesh.faces),nodesDBC)
                    for inode in range(Rest_DOFs.shape[0]):
                        for i in range(nvar):
                            self.columns_out = np.append(self.columns_out,nvar*Rest_DOFs[inode]+i)
                            self.applied_dirichlet = np.append(self.applied_dirichlet,0.0)

                print('Finished identifying Dirichlet boundary conditions from CAD geometry.', 
                    ' Time taken', time()-tCAD, 'seconds')

            else:
                
                end = -3
                self.applied_dirichlet = np.loadtxt(mesh.filename.split(".")[0][:end]+"_Dirichlet_"+"P"+str(MainData.C+1)+".dat",dtype=np.float64)
                self.columns_out = np.loadtxt(mesh.filename.split(".")[0][:end]+"_ColumnsOut_"+"P"+str(MainData.C+1)+".dat")

                print('Finished identifying Dirichlet boundary conditions from CAD geometry.', 
                    ' Time taken', time()-tCAD, 'seconds')

        #----------------------------------------------------------------------------------------------------#
        #------------------------------------- NON-NURBS BASED SOLUTION -------------------------------------#
        #----------------------------------------------------------------------------------------------------#

        elif self.boundary_type == 'straight' or self.boundary_type == 'mixed':
            # IF DIRICHLET BOUNDARY CONDITIONS ARE APPLIED DIRECTLY AT NODES
            if self.dirichlet_data_applied_at == 'node':
  
                # self.columns_out = []
                # for inode in range(0,self.dirichlet_flags.shape[0]):
                #     for i in range(nvar):
                #         if not np.isnan(self.dirichlet_flags[inode, i]):
                #             self.columns_out = np.append(self.columns_out,nvar*inode+i)
                #             self.applied_dirichlet = np.append(self.applied_dirichlet,self.dirichlet_flags[inode,i])

                flat_dirich = self.dirichlet_flags.flatten()
                self.columns_out = np.arange(self.dirichlet_flags.size)[~np.isnan(flat_dirich)]
                self.applied_dirichlet = flat_dirich[~np.isnan(flat_dirich)]

        # GENERAL PROCEDURE - GET REDUCED MATRICES FOR FINAL SOLUTION
        self.columns_out = self.columns_out.astype(np.int64)
        self.columns_in = np.delete(np.arange(0,nvar*mesh.points.shape[0]),self.columns_out)




    def IGAKitWrapper(self,mesh):
        """Calls IGAKit wrapper to get exact Dirichlet boundary conditions"""

        # GET THE NURBS CURVE FROM PROBLEMDATA
        # nurbs = self.NURBSParameterisation()
        # IDENTIFIY DIRICHLET BOUNDARY CONDITIONS BASED ON THE EXACT GEOMETRY
        C = mesh.InferPolynomialDegree() - 1
        nodesDBC, Dirichlet = GetDirichletData(mesh,self.nurbs_info,self,C) 

        return nodesDBC[:,None], Dirichlet



    def PostMeshWrapper(self, formulation, mesh, material, solver, fem_solver):
        """Calls PostMesh wrapper to get exact Dirichlet boundary conditions"""

        from CurvilinearMeshing import (PostMeshCurvePy as PostMeshCurve,
            PostMeshSurfacePy as PostMeshSurface)

        C = mesh.InferPolynomialDegree() - 1

        from Florence.FunctionSpace import Tri

        # GET BOUNDARY FEKETE POINTS
        if formulation.ndim == 2:
            
            # CHOOSE TYPE OF BOUNDARY SPACING 
            boundary_fekete = np.array([[]])
            if self.nodal_spacing_for_cad == 'fekete':
                boundary_fekete = GaussLobattoQuadrature(C+2)[0]
            else:
                boundary_fekete = EquallySpacedPoints(formulation.ndim,C)
            # IT IS IMPORTANT TO ENSURE THAT THE DATA IS C-CONITGUOUS
            boundary_fekete = boundary_fekete.copy(order="c")

            curvilinear_mesh = PostMeshCurve(mesh.element_type,dimension=formulation.ndim)
            curvilinear_mesh.SetMeshElements(mesh.elements)
            curvilinear_mesh.SetMeshPoints(mesh.points)
            curvilinear_mesh.SetMeshEdges(mesh.edges)
            curvilinear_mesh.SetMeshFaces(np.zeros((1,4),dtype=np.uint64))
            curvilinear_mesh.SetScale(self.scale_value_on_projection)
            curvilinear_mesh.SetCondition(self.condition_for_projection)
            curvilinear_mesh.SetProjectionPrecision(1.0e-04)
            curvilinear_mesh.SetProjectionCriteria(self.projection_flags)
            curvilinear_mesh.ScaleMesh()
            # curvilinear_mesh.InferInterpolationPolynomialDegree() 
            curvilinear_mesh.SetNodalSpacing(boundary_fekete)
            curvilinear_mesh.GetBoundaryPointsOrder()
            # READ THE GEOMETRY FROM THE IGES FILE
            curvilinear_mesh.ReadIGES(self.cad_file)
            # EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
            geometry_points = curvilinear_mesh.GetGeomVertices()
            self.GetGeometryMeshScale(geometry_points,mesh)
            # print(np.max(geometry_points[:,0]), mesh.Bounds)
            # exit()
            curvilinear_mesh.GetGeomEdges()
            curvilinear_mesh.GetGeomFaces()
            curvilinear_mesh.GetGeomPointsOnCorrespondingEdges()
            # FIRST IDENTIFY WHICH CURVES CONTAIN WHICH EDGES
            curvilinear_mesh.IdentifyCurvesContainingEdges()
            # PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
            curvilinear_mesh.ProjectMeshOnCurve()
            # FIX IMAGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
            curvilinear_mesh.RepairDualProjectedParameters()
            # PERFORM POINT INVERSION FOR THE INTERIOR POINTS
            if self.projection_type == 'orthogonal':
                curvilinear_mesh.MeshPointInversionCurve()
            elif self.projection_type == 'arc_length':
                curvilinear_mesh.MeshPointInversionCurveArcLength()
            else:
                print("projection type not understood. Arc length based projection is going to be used")
                curvilinear_mesh.MeshPointInversionCurveArcLength()
            # OBTAIN MODIFIED MESH POINTS - THIS IS NECESSARY TO ENSURE LINEAR MESH IS ALSO CORRECT
            curvilinear_mesh.ReturnModifiedMeshPoints(mesh.points)
            # GET DIRICHLET MainData
            nodesDBC, Dirichlet = curvilinear_mesh.GetDirichletData() 

            # GET ACTUAL CURVE POINTS - THIS FUNCTION IS EXPENSIVE
            # self.ActualCurve = curvilinear_mesh.DiscretiseCurves(100)

        elif formulation.ndim == 3:

            boundary_fekete = FeketePointsTri(C)

            curvilinear_mesh = PostMeshSurface(mesh.element_type,dimension=formulation.ndim)
            curvilinear_mesh.SetMeshElements(mesh.elements)
            curvilinear_mesh.SetMeshPoints(mesh.points)
            if mesh.edges.ndim == 2 and mesh.edges.shape[1]==0:
                mesh.edges = np.zeros((1,4),dtype=np.uint64)
            else:
                curvilinear_mesh.SetMeshEdges(mesh.edges)
            curvilinear_mesh.SetMeshFaces(mesh.faces)
            curvilinear_mesh.SetScale(self.scale_value_on_projection)
            curvilinear_mesh.SetCondition(self.condition_for_projection)
            curvilinear_mesh.SetProjectionPrecision(1.0e-04)
            curvilinear_mesh.SetProjectionCriteria(self.projection_flags)
            curvilinear_mesh.ScaleMesh()
            curvilinear_mesh.SetNodalSpacing(boundary_fekete)
            # curvilinear_mesh.GetBoundaryPointsOrder()
            # READ THE GEOMETRY FROM THE IGES FILE
            curvilinear_mesh.ReadIGES(self.cad_file)
            # EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
            geometry_points = curvilinear_mesh.GetGeomVertices()
            self.GetGeometryMeshScale(geometry_points,mesh)
            # print([np.min(geometry_points[:,2]),np.max(geometry_points[:,2])], mesh.Bounds)
            # exit()
            curvilinear_mesh.GetGeomEdges()
            curvilinear_mesh.GetGeomFaces()
            print("CAD geometry has", curvilinear_mesh.NbPoints, "points,", \
            curvilinear_mesh.NbCurves, "curves and", curvilinear_mesh.NbSurfaces, \
            "surfaces")
            curvilinear_mesh.GetGeomPointsOnCorrespondingFaces()

            # FIRST IDENTIFY WHICH SURFACES CONTAIN WHICH FACES
            if getattr(mesh,"face_to_surface",None) is not None:
                if mesh.faces.shape[0] == mesh.face_to_surface.size:
                    if mesh.face_to_surface.size != mesh.face_to_surface.shape[0]:
                        mesh.face_to_surface = np.ascontiguousarray(mesh.face_to_surface.flatten(),dtype=np.int64)
                    curvilinear_mesh.SupplySurfacesContainingFaces(mesh.face_to_surface,already_mapped=1)
                else:
                    raise AssertionError("face-to-surface mapping does not seem correct. " 
                        "Point projection is going to stop")
            else:
                # curvilinear_mesh.IdentifySurfacesContainingFacesByPureProjection()
                curvilinear_mesh.IdentifySurfacesContainingFaces() 
            
            # IDENTIFY WHICH EDGES ARE SHARED BETWEEN SURFACES
            curvilinear_mesh.IdentifySurfacesIntersections()
            
            # PERFORM POINT INVERSION FOR THE INTERIOR POINTS
            Neval = np.zeros((3,boundary_fekete.shape[0]),dtype=np.float64)
            hpBases = Tri.hpNodal.hpBases
            for i in range(3,boundary_fekete.shape[0]):
                Neval[:,i]  = hpBases(0,boundary_fekete[i,0],boundary_fekete[i,1],1)[0]
            # OrthTol = 0.5
            # project_on_curves = 0

            if self.projection_type == 'orthogonal':
                curvilinear_mesh.MeshPointInversionSurface(self.project_on_curves, self.modify_linear_mesh_on_projection)
            elif projection_type == 'arc_length':
                # PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE SURFACE
                curvilinear_mesh.ProjectMeshOnSurface()
                # curvilinear_mesh.RepairDualProjectedParameters()
                curvilinear_mesh.MeshPointInversionSurfaceArcLength(self.project_on_curves,
                    self.orthogonal_fallback_tolerance,Neval)
            else:
                warn("projection type not understood. Orthogonal projection is going to be used")
                curvilinear_mesh.MeshPointInversionSurface(self.project_on_curves)

            # OBTAIN MODIFIED MESH POINTS - THIS IS NECESSARY TO ENSURE LINEAR MESH IS ALSO CORRECT
            if self.modify_linear_mesh_on_projection:
                curvilinear_mesh.ReturnModifiedMeshPoints(mesh.points)
            # GET DIRICHLET DATA
            nodesDBC, Dirichlet = curvilinear_mesh.GetDirichletData()
            # print(Dirichlet)
            # GET DIRICHLET FACES (IF REQUIRED)
            dirichlet_faces = curvilinear_mesh.GetDirichletFaces()
            # print(dirichlet_faces[:100,:])

            # FOR GEOMETRIES CONTAINING PLANAR SURFACES
            planar_mesh_faces = curvilinear_mesh.GetMeshFacesOnPlanarSurfaces()
            # self.planar_mesh_faces = planar_mesh_faces


            if self.solve_for_planar_faces:
                if planar_mesh_faces.shape[0] != 0:
                    # SOLVE A 2D PROBLEM FOR PLANAR SURFACES
                    switcher = fem_solver.parallel
                    if fem_solver.parallel is True:
                        fem_solver.parallel = False

                    self.GetDirichletDataForPlanarFaces(formulation, material,
                        mesh, solver, fem_solver, planar_mesh_faces, nodesDBC, Dirichlet, plot=False)
                    fem_solver.parallel == switcher

        return nodesDBC, Dirichlet


    @staticmethod
    def GetDirichletDataForPlanarFaces(formulation, material, 
        mesh, solver, fem_solver, planar_mesh_faces, nodesDBC, Dirichlet, plot=False):
        """Solve a 2D problem for planar faces. Modifies Dirichlet"""

        from copy import deepcopy
        from Florence.Tensor import itemfreq, makezero
        from Florence import Mesh
        # from Florence.FiniteElements.Solvers.Solver import MainSolver
        from Florence import FunctionSpace, QuadratureRule
        from Florence.PostProcessing import PostProcess

        surface_flags = itemfreq(planar_mesh_faces[:,1])
        number_of_planar_surfaces = surface_flags.shape[0]

        C = mesh.InferPolynomialDegree() - 1

        E1 = [1.,0.,0.]
        E2 = [0.,1.,0.]
        E3 = [0.,0.,1.]

        # MAKE A SINGLE INSTANCE OF MATERIAL AND UPDATE IF NECESSARY
        import Florence.MaterialLibrary
        pmaterial_func = getattr(Florence.MaterialLibrary,material.mtype,None)
        pmaterial = pmaterial_func(2,E=material.E,nu=material.nu,E_A=material.E_A,G_A=material.G_A)
        
        print("The problem requires 2D analyses. Solving", number_of_planar_surfaces, "2D problems")
        for niter in range(number_of_planar_surfaces):
            
            pmesh = Mesh()
            pmesh.element_type = "tri"
            pmesh.elements = mesh.faces[planar_mesh_faces[planar_mesh_faces[:,1]==surface_flags[niter,0],0],:]
            pmesh.nelem = np.int64(surface_flags[niter,1])
            pmesh.GetBoundaryEdgesTri()
            unique_edges = np.unique(pmesh.edges)
            Dirichlet2D = np.zeros((unique_edges.shape[0],3))
            nodesDBC2D = np.zeros(unique_edges.shape[0]).astype(np.int64)
            
            unique_elements, inv  = np.unique(pmesh.elements, return_inverse=True)
            aranger = np.arange(unique_elements.shape[0],dtype=np.uint64)
            pmesh.elements = aranger[inv].reshape(pmesh.elements.shape)

            # elements = np.zeros_like(pmesh.elements)
            # unique_elements = np.unique(pmesh.elements)
            # counter = 0
            # for i in unique_elements:
            #     elements[pmesh.elements==i] = counter
            #     counter += 1
            # pmesh.elements = elements

            counter = 0
            for i in unique_edges:
                nodesDBC2D[counter] = np.where(nodesDBC==i)[0][0]
                Dirichlet2D[counter,:] = Dirichlet[nodesDBC2D[counter],:]
                counter += 1
            nodesDBC2D = nodesDBC2D.astype(np.int64)

            temp_dict = []
            for i in nodesDBC[nodesDBC2D].flatten():
                temp_dict.append(np.where(unique_elements==i)[0][0])
            nodesDBC2D = np.array(temp_dict,copy=False)

            pmesh.points = mesh.points[unique_elements,:]

            one_element_coord = pmesh.points[pmesh.elements[0,:3],:]

            # FOR COORDINATE TRANSFORMATION
            AB = one_element_coord[0,:] - one_element_coord[1,:]
            AC = one_element_coord[0,:] - one_element_coord[2,:]

            normal = np.cross(AB,AC)
            unit_normal = normal/np.linalg.norm(normal)

            e1 = AB/np.linalg.norm(AB)
            e2 = np.cross(normal,AB)/np.linalg.norm(np.cross(normal,AB))
            e3 = unit_normal

            # TRANSFORMATION MATRIX
            Q = np.array([
                [np.einsum('i,i',e1,E1), np.einsum('i,i',e1,E2), np.einsum('i,i',e1,E3)],
                [np.einsum('i,i',e2,E1), np.einsum('i,i',e2,E2), np.einsum('i,i',e2,E3)],
                [np.einsum('i,i',e3,E1), np.einsum('i,i',e3,E2), np.einsum('i,i',e3,E3)]
                ])

            pmesh.points = np.dot(pmesh.points,Q.T)
            # assert np.allclose(pmesh.points[:,2],pmesh.points[0,2])
            # z_plane = pmesh.points[0,2]

            pmesh.points = pmesh.points[:,:2]

            Dirichlet2D = np.dot(Dirichlet2D,Q.T)
            Dirichlet2D = Dirichlet2D[:,:2]

            pmesh.edges = None
            pmesh.GetBoundaryEdgesTri()

            # GET BONDAR#y CONDITION FOR 2D PROBLEM
            pboundary_condition = BoundaryCondition()
            pboundary_condition.SetCADProjectionParameters()
            pboundary_condition.is_dirichlet_computed = True
            pboundary_condition.nodesDBC = nodesDBC2D[:,None]
            pboundary_condition.Dirichlet = Dirichlet2D


            psolver = deepcopy(solver)
            # pformulation = deepcopy(formulation)
            pformulation_func = getattr(Florence.VariationalPrinciple, type(formulation).__name__,None)
            pformulation = pformulation_func(pmesh)
            # pformulation.ndim = 2
            # pformulation.nvar = 2
            # pfem_solver = deepcopy(fem_solver)


            # COMPUTE BASES FOR TRIANGULAR ELEMENTS
            QuadratureOpt = 3   # OPTION FOR QUADRATURE TECHNIQUE FOR TRIS AND TETS
            norder = 2*C
            if norder == 0:
                # TAKE CARE OF C=0 CASE
                norder = 1



            print('Solving planar problem number', niter, 'Number of DoF is', pmesh.points.shape[0]*pformulation.nvar)

            # # GET QUADRATURE
            # quadrature = QuadratureRule(optimal=QuadratureOpt, norder=norder, mesh_type=pmesh.element_type)
            # pfunction_space = FunctionSpace(pmesh, quadrature, p=C+1)

            # norder_post = (C+1)+(C+1)
            # post_quadrature = QuadratureRule(optimal=QuadratureOpt, norder=norder_post, mesh_type=pmesh.element_type)

            # ppost_function_space = FunctionSpace(pmesh, post_quadrature, p=C+1)
            # pfunction_spaces = (pfunction_space,ppost_function_space)           
            
            if pmesh.points.shape[0] != Dirichlet2D.shape[0]:
                # CALL THE FEM SOLVER FOR SOLVING THE 2D PROBLEM
                # solution = pfem_solver.Solve(function_spaces=pfunction_spaces, 
                # formulation=pformulation, mesh=pmesh, material=pmaterial, 
                # boundary_condition=pboundary_condition, solver=psolver)
                solution = fem_solver.Solve(formulation=pformulation, 
                    mesh=pmesh, material=pmaterial, 
                    boundary_condition=pboundary_condition)
                TotalDisp = solution.sol
            else:
                # IF THERE IS NO DEGREE OF FREEDOM TO SOLVE FOR (ONE ELEMENT CASE)
                TotalDisp = Dirichlet2D[:,:,None]

            Disp = np.zeros((TotalDisp.shape[0],3))
            Disp[:,:2] = TotalDisp[:,:,-1]

            temp_dict = []
            for i in unique_elements:
                temp_dict.append(np.where(nodesDBC==i)[0][0])

            Dirichlet[temp_dict,:] = np.dot(Disp,Q)

            if plot:
                post_process = PostProcess(2,2)
                post_process.CurvilinearPlotTri(pmesh, TotalDisp, 
                    QuantityToPlot=solution.ScaledJacobian, interpolation_degree=40)
                import matplotlib.pyplot as plt
                plt.show()

            # del pmesh, pboundary_condition, pfem_solver, pfunction_spaces
            del pmesh, pboundary_condition#, pfem_solver

        gc.collect()






    def GetReducedMatrices(self,stiffness,F,mass=None):

        # GET REDUCED FORCE VECTOR
        F_b = F[self.columns_in,0]

        # GET REDUCED STIFFNESS MATRIX
        stiffness_b = stiffness[self.columns_in,:][:,self.columns_in]

        # GET REDUCED MASS MATRIX
        mass_b = np.array([])
        if self.analysis_type != 'static':
            mass_b = mass[self.columns_in,:][:,self.columns_in]

        return stiffness_b, F_b, mass_b


    def ApplyDirichletGetReducedMatrices(self, stiffness, F, AppliedDirichlet, LoadFactor=1., mass=None):
        """AppliedDirichlet is a non-member because it can be external incremental Dirichlet,
            which is currently not implemented as member of BoundaryCondition. F also does not 
            correspond to Dirichlet forces, as it can be residual in incrementally linearised
            framework.
        """

        # tt=time()
        # F1 = np.copy(F)
        # # APPLY DIRICHLET BOUNDARY CONDITIONS
        # for i in range(self.columns_out.shape[0]):
            # F = F - LoadFactor*AppliedDirichlet[i]*stiffness.getcol(self.columns_out[i])

        # print(np.linalg.norm((stiffness[:,self.columns_out]*AppliedDirichlet*LoadFactor)))
        # MUCH FASTER APPROACH
        # print(np.linalg.norm(F))
        # print(AppliedDirichlet.shape, self.columns_in.shape, self.columns_out.shape)
        # print(AppliedDirichlet, LoadFactor)
        # F = F - (stiffness[:,self.columns_out]*AppliedDirichlet*LoadFactor)[:,None]
        # F = F - (stiffness[self.columns_in,:][:,self.columns_out]*AppliedDirichlet*LoadFactor)[:,None]
        # print(np.linalg.norm((stiffness[self.columns_in,:][:,self.columns_out].todense())))
        # print(stiffness[self.columns_in,:][:,self.columns_out].shape)
        # F[self.columns_in] = F[self.columns_in] - (stiffness[self.columns_in,:][:,self.columns_out]*AppliedDirichlet*LoadFactor)[:,None]
        # print(F[self.columns_in])
        # from scipy.io import savemat
        # savemat("/home/romanK")
        # print(stiffness.todense()[302,92])
        # print(np.linalg.norm(stiffness.todense()))
        # print(np.linalg.norm(F[self.columns_in]), np.linalg.norm(stiffness[self.columns_in,:][:,self.columns_out].todense()), np.linalg.norm(AppliedDirichlet*LoadFactor))
        # print(np.linalg.norm(F))
        # nnz_cols = ~np.isclose(AppliedDirichlet,0.0)
        # F = F - (stiffness[:,self.columns_out[nnz_cols]]*AppliedDirichlet[nnz_cols]*LoadFactor)[:,None]
        # F = F - self.__dirichlet_helper__(stiffness,AppliedDirichlet,self.columns_out)
        # F = F - self.__dirichlet_helper__(stiffness,AppliedDirichlet[nnz_cols],self.columns_out[nnz_cols])
        # print((F-F1)[:100])
        # print(time()-tt)
        # exit()
        # print(np.linalg.norm((stiffness[:,self.columns_out]*AppliedDirichlet*LoadFactor)[:,None]), np.linalg.norm(F))


        nnz_cols = ~np.isclose(AppliedDirichlet,0.0)
        F[self.columns_in] = F[self.columns_in] - (stiffness[self.columns_in,:][:,self.columns_out[nnz_cols]]*AppliedDirichlet[nnz_cols]*LoadFactor)[:,None]

        # GET REDUCED FORCE VECTOR
        F_b = F[self.columns_in,0]
        # print(np.linalg.norm(stiffness.todense().flatten()))

        # print int(sp.__version__.split('.')[1] )
        # FOR UMFPACK SOLVER TAKE SPECIAL CARE
        if int(sp.__version__.split('.')[1]) < 15:
            F_b_umf = np.zeros(F_b.shape[0])
            # F_b_umf[:] = F_b[:,0] # DOESN'T WORK
            for i in range(F_b_umf.shape[0]):
                F_b_umf[i] = F_b[i,0]
            F_b = np.copy(F_b_umf)

        # GET REDUCED STIFFNESS
        stiffness_b = stiffness[self.columns_in,:][:,self.columns_in]

        # GET REDUCED MASS MATRIX
        if self.analysis_type != 'static':
            mass_b = mass[self.columns_in,:][:,self.columns_in]
            return stiffness_b, F_b, F, mass_b

        return stiffness_b, F_b, F


    def UpdateFixDoFs(self, AppliedDirichletInc, fsize, nvar):
        """Updates the geometry (DoFs) with incremental Dirichlet boundary conditions 
            for fixed/constrained degrees of freedom only. Needs to be applied per time steps"""

        # GET TOTAL SOLUTION
        TotalSol = np.zeros((fsize,1))
        TotalSol[self.columns_out,0] = AppliedDirichletInc
                
        # RE-ORDER SOLUTION COMPONENTS
        dU = TotalSol.reshape(TotalSol.shape[0]/nvar,nvar)

        return dU

    def UpdateFreeDoFs(self, sol, fsize, nvar):
        """Updates the geometry with iterative solutions of Newton-Raphson 
            for free degrees of freedom only. Needs to be applied per time NR iteration"""

        # GET TOTAL SOLUTION
        TotalSol = np.zeros((fsize,1))
        TotalSol[self.columns_in,0] = sol
        
        # RE-ORDER SOLUTION COMPONENTS
        dU = TotalSol.reshape(TotalSol.shape[0]/nvar,nvar)

        return dU



    def SetNURBSParameterisation(self,nurbs_func,*args):
        self.nurbs_info = nurbs_func(*args)
        

    def SetNURBSCondition(self,nurbs_func,*args):
        self.nurbs_condition = nurbs_func(*args)


    def ComputeNeumannForces(self, mesh, material, dynamic_step=0):
        """A Dirichlet type methodology for applying Neumann boundary conditions"""

        if self.neumann_flags is None:
            return np.zeros((mesh.points.shape[0]*material.nvar,1),dtype=np.float64)

        nvar = material.nvar
        F = np.zeros((mesh.points.shape[0]*nvar,1))

        if self.neumann_data_applied_at == 'node':

            flat_neu = self.neumann_flags.flatten()
            to_apply = np.arange(self.neumann_flags.size)[~np.isnan(flat_neu)]
            applied_neumann = flat_neu[~np.isnan(flat_neu)]
            F[to_apply,0] = applied_neumann

        return F




    def __dirichlet_helper__(self,stiffness, AppliedDirichlet, columns_out):
        from scipy.sparse import csc_matrix
        M = csc_matrix((AppliedDirichlet,
            (columns_out,np.zeros_like(columns_out))),
            shape=(stiffness.shape[1],1))
        return (stiffness*M).A