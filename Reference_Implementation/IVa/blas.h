#ifndef _BLAS_H_
#define _BLAS_H_

#include <stdint.h>

#include <stdio.h>




#include "blas_config.h"


#include "blas_u32.h"

#define gf16v_mul_scalar  _gf16v_mul_scalar_u32
#define gf16v_madd        _gf16v_madd_u32

#define gf256v_add        _gf256v_add_u32
#define gf256v_mul_scalar  _gf256v_mul_scalar_u32
#define gf256v_madd        _gf256v_madd_u32


/// gaussian elim
#define gf256v_predicated_add        _gf256v_predicated_add_u32
/// gf16_rowmat_prod
#define gf16v_dot        _gf16v_dot_u32



#include "blas_comm.h"




#endif

