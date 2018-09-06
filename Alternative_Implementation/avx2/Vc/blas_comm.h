#ifndef _BLAS_COMM_H_
#define _BLAS_COMM_H_

#include <stdint.h>

#include <stdio.h>

#include "blas.h"



#ifdef  __cplusplus
extern  "C" {
#endif



///////////////////  prng //////////////////////////////////


void gf16v_rand( uint8_t * a , unsigned _num_ele );

void gf256v_rand( uint8_t * a , unsigned _num_byte );


/////////////////  input/output   /////////////////////////////////////


void gf256v_fdump(FILE * fp, const uint8_t *v, unsigned _num_byte);

void gf16mat_fdump(FILE * fp, const uint8_t *v, unsigned n_vec_byte , unsigned n_vec );

void gf256mat_fdump(FILE * fp, const uint8_t *v, unsigned n_vec_byte , unsigned n_vec );


/////////////////////////////////////


static inline
unsigned char gf16v_get_ele( const uint8_t * a , unsigned i ) {
	unsigned char r = a[i>>1];
	r = ( i&1 )? (r>>4):(r&0xf);
	return r;
}


static inline
unsigned char gf16v_set_ele( uint8_t * a , unsigned i , uint8_t v ) {
	unsigned char m = (i&1)? 0xf : 0xf0;
	a[i>>1] &= m;
	m = ( i&1 )? v<<4 : v&0xf;
	a[i>>1] |= m;
	return v;
}



static inline
void gf16v_split( uint8_t * z , const uint8_t * x , unsigned n )
{
	for(unsigned i=0;i<n;i+=2 ) {
		z[i] = x[i>>1] & 0xf;
		z[i+1] = x[i>>1] >>4;
	}
}


/////////////////////////////////////


void gf256v_set_zero( uint8_t * b, unsigned _num_byte );

unsigned gf256v_is_zero( const uint8_t * a, unsigned _num_byte );




///////////////// multiplications  ////////////////////////////////


/// polynomial multplication
/// School boook
void gf256v_polymul( uint8_t * c, const uint8_t * a , const uint8_t * b , unsigned _num );

/// matrix-vector


void gf16rowmat_prod( uint8_t * c , const uint8_t * matA , unsigned n_A_height , unsigned n_A_vec_byte , const uint8_t * b );

void _gf16mat_prod( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b );

void _gf256mat_prod( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b );

/// matrix-matrix

void gf16mat_mul( uint8_t * c , const uint8_t * a , const uint8_t * b , unsigned len_vec );

void gf256mat_mul( uint8_t * c , const uint8_t * a , const uint8_t * b , unsigned len_vec );



/////////////////   algorithms:  gaussian elim  //////////////////


unsigned _gf16mat_gauss_elim( uint8_t * mat , unsigned h , unsigned w );

unsigned _gf16mat_solve_linear_eq( uint8_t * sol , const uint8_t * inp_mat , const uint8_t * c_terms , unsigned n );


unsigned _gf256mat_gauss_elim( uint8_t * mat , unsigned h , unsigned w );

unsigned _gf256mat_solve_linear_eq( uint8_t * sol , const uint8_t * inp_mat , const uint8_t * c_terms , unsigned n );




////////////////  rand for matrices   //////////////////////////

/// buffer has to be as large as the input matrix
unsigned gf16mat_inv( uint8_t * inv_a , const uint8_t * a , unsigned H , uint8_t * buffer );

/// buffer has to be as large as the input matrix
unsigned gf16mat_rand_inv( uint8_t * a , uint8_t * b , unsigned H , uint8_t * buffer );

/// buffer has to be as large as the input matrix
unsigned gf256mat_inv( uint8_t * inv_a , const uint8_t * a , unsigned H , uint8_t * buffer );

/// buffer has to be as large as the input matrix
unsigned gf256mat_rand_inv( uint8_t * a , uint8_t * b , unsigned H , uint8_t * buffer );



#ifdef  __cplusplus
}
#endif



#endif

