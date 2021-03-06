#ifndef FOLLOWER_FORCES_H
#define FOLLOWER_FORCES_H

#include <algorithm>

#ifdef __SSE4_2__
#include <emmintrin.h>
#include <mm_malloc.h>
#endif

#include <Fastor/Fastor.h>
#include "assembly_helper.h"
#include "SparseAssemblyNative.h"

#ifndef LL_TYPES
#define LL_TYPES
using Real = double;
using Integer = std::int64_t;
using UInteger = std::uint64_t;
#endif

/*---------------------------------------------------------------------------------------------*/
#ifndef CUSTOM_ALLOCATION_
#define CUSTOM_ALLOCATION_
template<typename T>
FASTOR_INLINE T *allocate(Integer size) {
#if defined(__AVX__)
    T *out = (T*)_mm_malloc(sizeof(T)*size,32);
#elif defined(__SSE__)
    T *out = (T*)_mm_malloc(sizeof(T)*size,16);
#else
    T *out = (T*)malloc(sizeof(T)*size);
#endif
    return out;
}

template<typename T>
FASTOR_INLINE void deallocate(T *a) {
#if defined(__SSE__)
    _mm_free(a);
#else
    free(a);
#endif
}
#endif //CUSTOM_ALLOCATION_
/*---------------------------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------------------------*/
void GetFacesForces(Real *stiff_face,
	            Real *force,
		    const Real pressure,
		    const Real *Bases,
		    const Real *Jm,
		    const Real *AllGauss,
		    const Real *EulerElemCoords,
		    Integer nodeperface,
		    Integer ngauss,
		    Integer nvar)
{
   Integer ndof = nodeperface*nvar;
   //Integer local_capacity = ndof*ndof;
   
   Real *alternating = (Real*)malloc(3*3*3*sizeof(Real)); //alternating[i*3*3 + j*3 + k]
   std::fill(alternating,alternating+3*3*3,0.0);          //              j k     k
   alternating[0*3*3 + 1*3 + 2] =  1.0;
   alternating[0*3*3 + 2*3 + 1] = -1.0;
   alternating[1*3*3 + 0*3 + 2] = -1.0;
   alternating[1*3*3 + 2*3 + 0] =  1.0;
   alternating[2*3*3 + 0*3 + 1] =  1.0;
   alternating[2*3*3 + 1*3 + 0] = -1.0;

   Real *N           = (Real*)malloc(ndof*nvar*ngauss*sizeof(Real));
   Real *gNx         = (Real*)malloc(ndof*nvar*ngauss*sizeof(Real));
   Real *gNy         = (Real*)malloc(ndof*nvar*ngauss*sizeof(Real));
   Real *tangentialx = (Real*)malloc(ngauss*nvar*sizeof(Real));
   Real *tangentialy = (Real*)malloc(ngauss*nvar*sizeof(Real));
   Real *normal      = (Real*)malloc(ngauss*nvar*sizeof(Real));
   Real *crossx      = (Real*)malloc(ndof*ndof*nvar*ngauss*sizeof(Real));
   Real *crossy      = (Real*)malloc(ndof*ndof*nvar*ngauss*sizeof(Real));

   std::fill(N,N+ndof*nvar*ngauss,0.0);
   std::fill(gNx,gNx+ndof*nvar*ngauss,0.0);
   std::fill(gNy,gNy+ndof*nvar*ngauss,0.0);
   // Loop to fill function spaces for dimensons
   // 0::3=>0,3,6,9 -- 1::3=>1,4,7,10 -- 2::3=>2,5,8,11
   for (Integer i=0; i<nvar; ++i) {
      for (Integer j=0; j<nodeperface; ++j) {
         Integer idof = j*nvar + i;
         for (Integer k=0; k<ngauss; ++k) {
            N[idof*nvar*ngauss+i*ngauss+k] = Bases[j*ngauss+k];
            gNx[idof*nvar*ngauss+i*ngauss+k] = Jm[0*nodeperface*ngauss+j*ngauss+k];
            gNy[idof*nvar*ngauss+i*ngauss+k] = Jm[1*nodeperface*ngauss+j*ngauss+k];
	     }
      }
   }
   std::fill(tangentialx,tangentialx+ngauss*nvar,0.0);
   std::fill(tangentialy,tangentialy+ngauss*nvar,0.0);
   // mapping tangential vectors [\partial\vec{x}/ \partial\zeta (ngauss x ndim)]
   //tangentialx = einsum("ij,ik->jk",gBasesx,EulerElemCoords)
   //tangentialy = einsum("ij,ik->jk",gBasesy,EulerElemCoords)
   for (Integer j=0; j<ngauss; ++j) {
      for (Integer k=0; k<nvar; ++k) {
         for (Integer i=0; i<nodeperface; ++i) {
            tangentialx[j*nvar+k] += Jm[0*nodeperface*ngauss+i*ngauss+j]*EulerElemCoords[i*nvar+k];
            tangentialy[j*nvar+k] += Jm[1*nodeperface*ngauss+i*ngauss+j]*EulerElemCoords[i*nvar+k];
	 }
      }
   }
   std::fill(normal,normal+ngauss*nvar,0.0);
   // mapping normal (ngauss x ndim)
   //normal = einsum("ijk,lj,lk->li",alternating,tangentialx,tangentialy)
   for (Integer l=0; l<ngauss; ++l) {
      for (Integer i=0; i<nvar; ++i) {
         for (Integer j=0; j<nvar; ++j) {
            for (Integer k=0; k<nvar; ++k) {
               normal[l*nvar+i] += alternating[i*nvar*nvar+j*nvar+k]*tangentialx[l*nvar+j]*tangentialy[l*nvar+k];
	    }
	 }
      }
   }
   // Gauss quadrature of follower load (traction)
   //force = einsum("ijk,kj,k->ik",N,normal,AllGauss[:,0]).sum(axis=1)
   for (Integer i=0; i<ndof; ++i) {
      for (Integer k=0; k<ngauss; ++k) {
         for (Integer j=0; j<nvar; ++j) {
            force[i] += pressure*N[i*nvar*ngauss+j*ngauss+k]*normal[k*nvar+j]*AllGauss[k];
	 }
      }
   }
   std::fill(crossx,crossx+ndof*ndof*nvar*ngauss,0.0);
   std::fill(crossy,crossy+ndof*ndof*nvar*ngauss,0.0);
   // Gauss quadrature of follower load (stiffness)
   //crossy=einsum("ijk,ljm,nkm->lnim",alternating,gNy,N)-np.einsum("ijk,ljm,nkm->lnim",alternating,N,gNy)
   //crossx=einsum("ijk,ljm,nkm->lnim",alternating,gNx,N)-np.einsum("ijk,ljm,nkm->lnim",alternating,N,gNx)
   for (Integer l=0; l<ndof; ++l) {
      for (Integer n=0; n<ndof; ++n) {
         for (Integer i=0; i<nvar; ++i) {
            for (Integer m=0; m<ngauss; ++m) {
               for (Integer j=0; j<nvar; ++j) {
                  for (Integer k=0; k<nvar; ++k) {
                     crossy[l*ndof*nvar*ngauss+n*nvar*ngauss+i*ngauss+m] += 
		      alternating[i*nvar*nvar+j*nvar+k]*gNy[l*nvar*ngauss+j*ngauss+m]*
		      N[n*nvar*ngauss+k*ngauss+m] - alternating[i*nvar*nvar+j*nvar+k]*
		      N[l*nvar*ngauss+j*ngauss+m]*gNy[n*nvar*ngauss+k*ngauss+m];
                     crossx[l*ndof*nvar*ngauss+n*nvar*ngauss+i*ngauss+m] += 
                      alternating[i*nvar*nvar+j*nvar+k]*gNx[l*nvar*ngauss+j*ngauss+m]*
		      N[n*nvar*ngauss+k*ngauss+m] - alternating[i*nvar*nvar+j*nvar+k]*
		      N[l*nvar*ngauss+j*ngauss+m]*gNx[n*nvar*ngauss+k*ngauss+m];
		  }
	       }
	    }
	 }
      }
   }
   //quadrature1 = einsum("ij,klji,i->kli",tangentialx,crossy,AllGauss[:,0]).sum(axis=2)
   //quadrature2 = einsum("ij,klji,i->kli",tangentialy,crossx,AllGauss[:,0]).sum(axis=2)
   //0.5*pressure*(quadrature1-quadrature2)
   for (Integer k=0; k<ndof; ++k) {
      for (Integer l=0; l<ndof; ++l) {
         for (Integer i=0; i<ngauss; ++i) {
            for (Integer j=0; j<nvar; ++j) {
               stiff_face[k*ndof+l] += 0.5*pressure*(tangentialx[i*nvar+j]*
		crossy[k*ndof*nvar*ngauss+l*nvar*ngauss+j*ngauss+i] - 
		tangentialy[i*nvar+j]*crossx[k*ndof*nvar*ngauss+l*nvar*ngauss+j*ngauss+i])*AllGauss[i];
            }
	 }
      }
   }
   free(alternating);
   free(N);
   free(gNx);
   free(gNy);
   free(tangentialx);
   free(tangentialy);
   free(normal);
   free(crossx);
   free(crossy);
}

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
{
   Integer ndof = nodeperface*nvar;
   Integer local_capacity = ndof*ndof;

   Real *EulerElemCoords = (Real*)malloc(sizeof(Real)*nodeperface*nvar);

   Real *force = (Real*)malloc(ndof*sizeof(Real));
   Real *stiff_face = (Real*)malloc(local_capacity*sizeof(Real));

   // LOOP OVER FACES
   for (Integer face=0; face<nface; ++face) {
      if (pressure_flags[face]) {
	 // APPLIED PRESSURE BY INCREMENT AND FACE
         Real pressure = pressure_increment*applied_pressure[face];
	 // GET FIELD AT ELEMENT LEVEL (JUST CURRENT)
         for (Integer i=0; i<nodeperface; ++i) {
            Integer inode = faces[face*nodeperface+i];
            for (Integer j=0; j<nvar; ++j) {
               EulerElemCoords[i*nvar+j] = Eulerx[inode*nvar+j];
	    }
	 }
	 // COMPUTE STIFFNESS AND FORCE
	 std::fill(force,force+ndof,0.0);
	 std::fill(stiff_face,stiff_face+local_capacity,0.0);
	 GetFacesForces( stiff_face,
			 force,
			 pressure,
			 Bases,
			 Jm,
			 AllGauss,
			 EulerElemCoords,
			 nodeperface,
			 ngauss,
			 nvar);

         // ASSEMBLE CONSTITUTIVE STIFFNESS
         fill_global_data(
                nullptr,
                nullptr,
                stiff_face,
                I_stiff,
                J_stiff,
                V_stiff,
                face,
                nvar,
                nodeperface,
                faces,
                local_capacity,
                local_capacity,
                recompute_sparsity_pattern,
                squeeze_sparsity_pattern,
                data_local_indices,
                data_global_indices,
                sorted_elements,
		sorter);

	 // ASSEMBLE FORCES
	 // F[faces[face,:]*nvar+ivar,0]+=force[ivar::nvar,0]
	 for (Integer i=0; i<nodeperface; ++i) {
	    Integer inode = faces[face*nodeperface+i]*nvar;
	    for (Integer j=0; j<nvar; ++j) {
	       F[inode+j] += force[i*nvar+j];
	    }
	 }
      }
   }
   free(force);
   free(stiff_face);
   free(EulerElemCoords);
}
/*---------------------------------------------------------------------------------------------*/
#endif  //FOLLOWER_FORCES_H
