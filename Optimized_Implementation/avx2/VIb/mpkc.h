
#ifndef _MPKC_H_
#define _MPKC_H_

#include "stdint.h"


#ifndef TERMS_QUAD_POLY
#define TERMS_QUAD_POLY(N) (((N)*(N+1)/2)+N+1)
#endif

#include "blas_config.h"

#ifdef _BLAS_AVX2_
#include "mpkc_avx2.h"
#define gf31mpkc_mq_eval_n_m  gf31mpkc_mq_eval_n_m_avx2
#else
error here.
#endif


#ifdef  __cplusplus
extern  "C" {
#endif


void to_maddusb_format_mq( uint8_t * z , const uint8_t * x , unsigned _n, unsigned m );

void maddusb_to_normal_mq( uint8_t * z , const uint8_t * x , unsigned _n, unsigned m );

void gf31mpkc_interpolate_n_m( uint8_t * poly , void (*quad_poly)(void *,const void *,const void *) , const void * key , unsigned n, unsigned m);


void _gf31mpkc_mq_eval_n_m( uint8_t * z , const uint8_t * pk_mat , const uint8_t * w , unsigned n, unsigned m);



#ifdef  __cplusplus
}
#endif


#endif
