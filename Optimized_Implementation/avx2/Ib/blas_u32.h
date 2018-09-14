#ifndef _BLAS_U32_H_
#define _BLAS_U32_H_

#include <stdint.h>
#include <stdio.h>


#include "gf31.h"


#ifdef  __cplusplus
extern  "C" {
#endif



static inline
void _gf31v_add( uint8_t * accu_b, const uint8_t * a , unsigned _num_byte ) {
	for(unsigned i=0;i<_num_byte;i++) accu_b[i] = gf31_add( accu_b[i] , a[i] );
}

static inline
void _gf31v_sub( uint8_t * accu_b, const uint8_t * a , unsigned _num_byte ) {
	for(unsigned i=0;i<_num_byte;i++) accu_b[i] = gf31_sub( accu_b[i] , a[i] );
}


////////////////////


static inline
void _gf31v_mul_scalar( uint8_t *a, uint8_t b, unsigned _num_byte ) {
	for(unsigned i=0;i<_num_byte;i++) a[i] = gf31_mul( a[i] , b );
}

static inline
void _gf31v_madd( uint8_t * accu_c, const uint8_t * a , uint8_t b, unsigned _num_byte ) {
	for(unsigned i=0;i<_num_byte;i++) accu_c[i] = gf31_add( accu_c[i] , gf31_mul( a[i] , b ) );
}


static inline
void _gf31v_msub( uint8_t * accu_c, const uint8_t * a , uint8_t b, unsigned _num_byte ) {
	for(unsigned i=0;i<_num_byte;i++) accu_c[i] = gf31_sub( accu_c[i] , gf31_mul( a[i] , b ) );
}



static inline
void _gf31v_madd_2col( uint8_t * accu_c, const uint8_t * a , uint8_t b1, uint8_t b2 , unsigned n ) {
	for(unsigned i=0;i<n;i++) {
		unsigned char c = gf31_add( gf31_mul( a[i*2] , b1 ) , gf31_mul( a[i*2+1] , b2 ) );
		accu_c[i] = gf31_add( accu_c[i] , c );
	}
}




#ifdef  __cplusplus
}
#endif



#endif

