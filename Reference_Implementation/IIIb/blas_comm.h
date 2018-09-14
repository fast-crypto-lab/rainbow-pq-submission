#ifndef _BLAS_COMM_H_
#define _BLAS_COMM_H_

#include <stdint.h>
#include <stdio.h>

#include "blas_config.h"


#ifdef  __cplusplus
extern  "C" {
#endif


////////////// convert vector format for "two column matrix" ///////

#ifdef _TWO_COL_MAT_

void to_maddusb_format( uint8_t * z , const uint8_t * x , unsigned n, unsigned m );

void maddusb_to_normal( uint8_t * z , const uint8_t * x , unsigned n, unsigned m );

#endif


////////////////   PRNG   //////////////////////////////

void gf31v_rand( uint8_t * a , unsigned _num_byte );


//////////////// input/output ////////////////////////////

void gf256v_fdump(FILE * fp, const uint8_t *v, unsigned _num_byte);

void gf256mat_fdump(FILE * fp, const uint8_t *v, unsigned n_vec_byte , unsigned n_vec );


void gf31v_set_zero( uint8_t * b, unsigned _num_byte );

unsigned gf31v_is_zero( const uint8_t * a, unsigned _num_byte );


///////////////// matrix multiplication  ////////////////////////

//// matrix-vector
void _gf31mat_prod( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b );

//// matrix-matrix
void gf31mat_mul( uint8_t * c , const uint8_t * a , const uint8_t * b , unsigned len_vec );


///////////////// gaussian elimination  ///////////////////////////


unsigned _gf31mat_gauss_elim( uint8_t * mat , unsigned h , unsigned w );

void gf31mat_submat( uint8_t * mat2 , unsigned w2 , unsigned st , const uint8_t * mat , unsigned w , unsigned h );

unsigned gf31mat_inv( uint8_t * inv_a , const uint8_t * a , unsigned H , uint8_t * buffer );

unsigned gf31mat_rand_inv( uint8_t * a , uint8_t * b , unsigned H , uint8_t * buffer );



#ifdef  __cplusplus
}
#endif



#endif  /// #define _BLAS_COMM_H_


