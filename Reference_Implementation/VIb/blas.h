#ifndef _BLAS_H_
#define _BLAS_H_


#include "gf31.h"

#include "blas_config.h"


#ifdef _BLAS_SSE_
error here.
#else

#include "blas_u32.h"


#define gf31v_add _gf31v_add
#define gf31v_sub _gf31v_sub

#define gf31v_mul_scalar _gf31v_mul_scalar
#define gf31v_madd _gf31v_madd
#define gf31v_msub _gf31v_msub
#define gf31v_madd_2col _gf31v_madd_2col

#define gf31mat_prod _gf31mat_prod
#define gf31mat_gauss_elim _gf31mat_gauss_elim

#endif


#include "blas_comm.h"



#endif

