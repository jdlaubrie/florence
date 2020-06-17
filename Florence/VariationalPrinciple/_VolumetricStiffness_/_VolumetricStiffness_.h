#ifndef VOLUMETRICSTIFFNESS_H
#define VOLUMETRICSTIFFNESS_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#ifndef LL_TYPES
#define LL_TYPES
using Real = double;
using Integer = std::int64_t;
using UInteger = std::uint64_t;
#endif

void _VolumetricStiffnessFiller_(Real *volumetric_stiffness, Real *mean_volume,
    const Real *SpatialGradient, const Real *detJ, const Real *dV,
    Integer ndim, Integer nvar, Integer nodeperelem, Integer ngauss)
{
    Real volumex = 0.;
    Real VolumeX = 0.;
    for (Integer g=0; g<ngauss; ++g) {
        volumex += detJ[g];
        VolumeX += dV[g];
    }
    mean_volume[0] = (volumex-VolumeX)/VolumeX;

    Real *AverageSpatialGradient    = (Real*)malloc(sizeof(Real)*ndim*nodeperelem);
    std::fill(AverageSpatialGradient,AverageSpatialGradient+nodeperelem*ndim,0.);
    
    for (Integer a=0; a<nodeperelem; ++a) {
        for (Integer i=0; i<ndim; ++i) {
            for (Integer g=0; g<ngauss; ++g) {
                AverageSpatialGradient[a*ndim+i] += SpatialGradient[g*nodeperelem*ndim+a*ndim+i]*detJ[g];
            }
        }
    }
    for (Integer a=0; a<nodeperelem; ++a) {
        for (Integer b=0; b<nodeperelem; ++b) {
            for (Integer i=0; i<ndim; ++i) {
                for (Integer j=0; j<ndim; ++j) {
                    volumetric_stiffness[(a*nvar+i)*nodeperelem*nvar+(b*nvar+j)] += 
                             AverageSpatialGradient[a*ndim+i]*AverageSpatialGradient[b*ndim+j]/VolumeX;
                }
            }
        }
    }
    free(AverageSpatialGradient);
}

#endif //VOLUMETRICSTIFFNESS_H
