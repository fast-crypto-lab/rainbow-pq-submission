
#include <stdint.h>

#include <stdio.h>

#include "gf31.h"

#include "blas.h"

#include "prng_utils.h"

#include "blas_config.h"
#include <assert.h>  /// have to be included at last.




////////////// convert vector format for "two column matrix" ///////

#ifdef _TWO_COL_MAT_
void to_maddusb_format( uint8_t * z , const uint8_t * x , unsigned n, unsigned m )
{
	assert( m <= 256 );
	uint8_t temp[512];
	while( n > 1 ) {
		for(unsigned i=0;i<m;i++) temp[i*2] = x[i];
		for(unsigned i=0;i<m;i++) temp[i*2+1] = x[m+i];
		for(unsigned i=0;i<m*2;i++) z[i] = temp[i];

		n -= 2;
		x += 2*m;
		z += 2*m;
	}
	if( 1 == n ) {
		for(unsigned i=0;i<m;i++) z[i] = x[i];
	}
}

void maddusb_to_normal( uint8_t * z , const uint8_t * x , unsigned n, unsigned m )
{
	assert( m <= 256 );
	uint8_t temp[512];
	while( n > 1 ) {
		for(unsigned i=0;i<m;i++) temp[i] = x[i*2];
		for(unsigned i=0;i<m;i++) temp[m+i] = x[i*2+1];
		for(unsigned i=0;i<m*2;i++) z[i] = temp[i];

		n -= 2;
		x += 2*m;
		z += 2*m;
	}
	if( 1 == n ) {
		for(unsigned i=0;i<m;i++) z[i] = x[i];
	}
}
#endif





////////////////   PRNG   //////////////////////////////



void gf31v_rand( uint8_t * a , unsigned _num_byte ) {
	for(unsigned i=0;i<_num_byte;i++) {
		prng_bytes( a+i , 1 );
		/// Reject sampling
		while( 0 == (a[i]>>3) ) prng_bytes( a+i , 1 );
		a[i]%=31;
	}
}




//////////////// input/output ////////////////////////////

void gf256v_fdump(FILE * fp, const uint8_t *v, unsigned _num_byte) {
	fprintf(fp,"[%2d][",_num_byte);
	for(unsigned i=0;i<_num_byte;i++) { fprintf(fp,"0x%02x,",v[i]); if(7==(i%8)) fprintf(fp," ");}
	fprintf(fp,"]");
}


void gf256mat_fdump(FILE * fp, const uint8_t *v, unsigned n_vec_byte , unsigned n_vec ) {
	for(unsigned i=0;i<n_vec;i++) {
		fprintf(fp,"[%d]",i);
		gf256v_fdump(fp,v,n_vec_byte);
		fprintf(fp,"\n");
		v += n_vec_byte;
	}
}



void gf31v_set_zero( uint8_t * b, unsigned _num_byte ) {
	for(unsigned i=0;i<_num_byte;i++) b[i]=0;
}


unsigned gf31v_is_zero( const uint8_t * a, unsigned _num_byte ) {
	unsigned char r = 0;
	for(unsigned i=0;i<_num_byte;i++) r |= a[i];
	return (0==r);
}




///////////////// matrix multiplication  ////////////////////////


//// matrix-vector
void _gf31mat_prod( uint8_t * c , const uint8_t * matA , unsigned n_A_vec_byte , unsigned n_A_width , const uint8_t * b ) {
#ifdef _TWO_COL_MAT_
	gf31v_set_zero(c,n_A_vec_byte);
	unsigned odd = n_A_width&1;
	n_A_width ^= odd;
	for(unsigned i=0;i<n_A_width;i+=2) {
		gf31v_madd_2col( c , matA , b[i] , b[i+1] , n_A_vec_byte );
		matA += 2*n_A_vec_byte;
	}
	if( 1 == odd ) gf31v_madd( c , matA , b[n_A_width] , n_A_vec_byte );
#else
	gf31v_set_zero(c,n_A_vec_byte);
	for(unsigned i=0;i<n_A_width;i++) {
		gf31v_madd( c , matA , b[i] , n_A_vec_byte );
		matA += n_A_vec_byte;
	}
#endif
}


//// matrix-matrix
void gf31mat_mul( uint8_t * c , const uint8_t * a , const uint8_t * b , unsigned len_vec ) {
	unsigned n_vec_byte = len_vec;
	for(unsigned k=0;k<len_vec;k++){
		gf31v_set_zero( c , n_vec_byte );
		const uint8_t * bk = b + n_vec_byte * k;
		for(unsigned i=0;i<len_vec;i++) {
			gf31v_madd( c , a + n_vec_byte * i , bk[i] , n_vec_byte  );
		}
		c += n_vec_byte;
	}
}





///////////////// gaussian elimination  ///////////////////////////



unsigned _gf31mat_gauss_elim( uint8_t * mat , unsigned h , unsigned w )
{
	unsigned char r8 = 1;
	for(unsigned i=0;i<h;i++) {
		uint8_t * ai = mat + w*i;
		for(unsigned j=i+1;j<h;j++) {
			uint8_t * aj = mat + w*j;
			gf31v_madd( ai , aj , gf31_is_nonzero(ai[i])^gf31_is_nonzero(aj[i]) , w );
		}
		r8 &= gf31_is_nonzero(ai[i]);
		uint8_t pivot = ai[i];
		pivot = gf31_inv( pivot );
		gf31v_mul_scalar( ai , pivot , w );
		for(unsigned j=0;j<h;j++) {
			if(i==j) continue;
			uint8_t * aj = mat + w*j;
			gf31v_msub( aj , ai , aj[i] , w );
		}
	}
	return r8;
}

void gf31mat_submat( uint8_t * mat2 , unsigned w2 , unsigned st , const uint8_t * mat , unsigned w , unsigned h )
{
	for(unsigned i=0;i<h;i++) {
		for(unsigned j=0;j<w2;j++) mat2[i*w2+j] = mat[i*w+st+j];
	}
}


unsigned gf31mat_inv( uint8_t * inv_a , const uint8_t * a , unsigned H , uint8_t * buffer )
{
	for(unsigned i=0;i<H;i++){
		uint8_t * ai = buffer + i*2*H;
		for(unsigned j=0;j<H;j++) ai[j] = a[i*H+j];
		gf31v_set_zero( ai + H , H );
		ai[H+i] = 1;
	}
	unsigned r = gf31mat_gauss_elim( buffer , H , 2*H );
	gf31mat_submat( inv_a , H , H , buffer , 2*H , H );
	return r;
}

unsigned gf31mat_rand_inv( uint8_t * a , uint8_t * b , unsigned H , uint8_t * buffer )
{
	//gf31v_rand( a , H*H );
	for(unsigned i=0;i<H;i++) gf31v_rand( a+i*H , H );
	unsigned r = gf31mat_inv( b , a , H , buffer );
	return r;
}




