cimport cython
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t

ctypedef int64_t Integer
ctypedef uint64_t UInteger
ctypedef double Real
# type int is int32 in numpy, in this case

cdef extern from "FollowerForces.h" nogil:
    void FollowerForcesAssembler(const UInteger *faces,
                               const Real *Eulerx,
                               const Real *Bases,
                               const Real *Jm,
                               const Real *AllGauss,
                               const int *pressure_flags,
                               const Real *applied_pressure,
                               const Real pressure_increment,
                               int recompute_sparsity_pattern,
                               int squeeze_sparsity_pattern,
                               const int *data_global_indices,
                               const int *data_local_indices,
                               const UInteger *sorted_elements,
                               const Integer *sorter,
                               int *I_stiff,
                               int *J_stiff,
                               Real *V_stiff,
                               Real *F,
                               Integer nface,
                               Integer nodeperface,
                               Integer ngauss,
                               Integer local_size,
                               Integer nvar)

def FollowerForces(boundary_condition, mesh, material, function_space, fem_solver, Real[:,::1] Eulerx):
    """
    This routine assembles the stiffness matrix and forces vector from pressure over the material
    so the number of variables is same than the number of dimensions of deformation.
    nvar = ndim
    """

    # GET VARIABLES FOR DISPATCHING TO C
    cdef Integer ndim    = material.ndim
    cdef Integer nvar    = material.ndim

    cdef np.ndarray[int,ndim=1, mode='c'] pressure_flags    = boundary_condition.pressure_flags.astype(np.int32)
    cdef np.ndarray[Real,ndim=1, mode='c'] applied_pressure = boundary_condition.applied_pressure
    cdef Real pressure_increment                            = boundary_condition.pressure_increment

    cdef np.ndarray[Real,ndim=2, mode='c'] Bases         = function_space.Bases
    cdef np.ndarray[Real,ndim=3, mode='c'] Jm            = function_space.Jm
    cdef np.ndarray[Real,ndim=1, mode='c'] AllGauss      = function_space.AllGauss.flatten()
    cdef Integer ngauss                                  = function_space.AllGauss.shape[0]
    cdef Integer local_size                              = function_space.Bases.shape[0]*nvar

    cdef int squeeze_sparsity_pattern       = fem_solver.squeeze_sparsity_pattern
    cdef int recompute_sparsity_pattern     = fem_solver.recompute_sparsity_pattern

    cdef np.ndarray[UInteger,ndim=2, mode='c'] faces = np.zeros((1,1),np.uint64)
    cdef Integer nface                               = 0
    cdef Integer nodeperface                         = 0

    cdef np.ndarray[int,ndim=1,mode='c'] I_stiff    = np.zeros(1,np.int32)
    cdef np.ndarray[int,ndim=1,mode='c'] J_stiff    = np.zeros(1,np.int32)
    cdef np.ndarray[Real,ndim=1,mode='c'] V_stiff   = np.zeros(1,np.float64)

    cdef np.ndarray[int,ndim=1,mode='c'] data_global_indices   = np.zeros(1,np.int32)
    cdef np.ndarray[int,ndim=1,mode='c'] data_local_indices    = np.zeros(1,np.int32)
    cdef np.ndarray[Integer,ndim=2, mode='c'] sorter           = np.zeros((1,1),np.int64)
    cdef np.ndarray[UInteger,ndim=2, mode='c'] sorted_elements = np.zeros((1,1),np.uint64)

    cdef np.ndarray[Real,ndim=1, mode='c'] F = np.zeros(mesh.points.shape[0]*nvar,np.float64)

    if ndim == 2:
        faces = mesh.edges
        nface = mesh.edges.shape[0]
        nodeperface = mesh.edges.shape[1]
    else:
        faces = mesh.faces
        nface = mesh.faces.shape[0]
        nodeperface = mesh.faces.shape[1]

    if fem_solver.recompute_sparsity_pattern:
        # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF STIFFNESS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
        I_stiff = np.zeros(int((nvar*nodeperface)**2*nface),np.int32)
        J_stiff = np.zeros(int((nvar*nodeperface)**2*nface),np.int32)
        V_stiff = np.zeros(int((nvar*nodeperface)**2*nface),np.float64)
    else:
        I_stiff = fem_solver.indices
        J_stiff = fem_solver.indptr
        V_stiff = np.zeros(fem_solver.indices.shape[0],dtype=np.float64)
        data_global_indices = fem_solver.data_global_indices
        data_local_indices = fem_solver.data_local_indices
        if fem_solver.squeeze_sparsity_pattern:
            sorter = mesh.element_sorter
            sorted_elements = mesh.sorted_elements

    FollowerForcesAssembler(  &faces[0,0],
                            &Eulerx[0,0],
                            &Bases[0,0],
                            &Jm[0,0,0],
                            &AllGauss[0],
                            &pressure_flags[0],
                            &applied_pressure[0],
                            pressure_increment,
                            recompute_sparsity_pattern,
                            squeeze_sparsity_pattern,
                            &data_global_indices[0],
                            &data_local_indices[0],
                            &sorted_elements[0,0],
                            &sorter[0,0],
                            &I_stiff[0],
                            &J_stiff[0],
                            &V_stiff[0],
                            &F[0],
                            nface,
                            nodeperface,
                            ngauss,
                            local_size,
                            nvar)

    if fem_solver.recompute_sparsity_pattern:
        return I_stiff, J_stiff, V_stiff, F
    else:
        return V_stiff, F


