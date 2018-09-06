
#ifndef _MPKC_H_
#define _MPKC_H_


#include "blas.h"


#ifndef TERMS_QUAD_POLY
#define TERMS_QUAD_POLY(N) (((N)*(N+1)/2)+N+1)
#endif


#define gf16mpkc_mq_eval_n_m    _gf16mpkc_mq_eval_n_m




#ifdef  __cplusplus
extern  "C" {
#endif


void _gf16mpkc_mq_eval_n_m( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m );

void _gf16mpkc_interpolate_n_m( uint8_t * poly , void (*quad_poly)(void *,const void *,const void *) , const void * key , unsigned n , unsigned m );

#ifdef  __cplusplus
}
#endif


#endif
